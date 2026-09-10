"""A slimmer encoder for the serving path: TorchScript + tokenizers, no transformers.

WHY. The service was measured at 456 MB resident with torch, transformers, MiniLM and the
index loaded - which fits a 512 MB instance, but with only ~56 MB of headroom, and that is
not enough to survive concurrent requests.

The measurement showed where it goes:

    baseline python          12 MB
    + numpy                  32 MB
    + torch                 197 MB
    + transformers          412 MB   <- 215 MB for the LIBRARY, before any model
    + MiniLM                417 MB
    + index                 417 MB
    + one query             456 MB

`transformers` costs more than the model does. It is needed to BUILD the artifact - to
load the checkpoint and trace it - but not to RUN it: a traced TorchScript module needs
only torch, and the tokenizer is available standalone from `tokenizers`, the same Rust
library transformers wraps.

This loads the traced encoder if it is present and falls back to the full transformers
path otherwise, so local development is unchanged and only the image is slimmer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


class TracedEmbedder:
    """MiniLM as a TorchScript module plus a standalone tokenizer."""

    def __init__(self, artifacts: Path, batch_size: int = 128):
        import torch
        from tokenizers import Tokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.module = torch.jit.load(str(artifacts / "encoder_traced.pt"),
                                     map_location="cpu")
        self.module.eval()
        self.tokenizer = Tokenizer.from_file(str(artifacts / "tokenizer.json"))
        meta = json.loads((artifacts / "encoder_meta.json").read_text())
        self.name = meta["model"]
        self.dim = meta["dim"]
        self.max_length = meta.get("max_length", 64)
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(length=None)

    def encode(self, texts: Sequence[str], max_length: Optional[int] = None) -> np.ndarray:
        # Batched, not one pass over everything: the corpus is 10,003 messages, and their
        # hidden states in one tensor are 10,003 x 64 x 384 floats - just under a
        # gigabyte, on an instance that has 512 MB in total. Batching caps the peak at
        # the batch, and padding to the longest text in each batch rather than to a fixed
        # 64 makes the short ones cheaper still.
        torch = self.torch
        out: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            if not chunk:
                continue
            encoded = self.tokenizer.encode_batch(chunk)
            ids = torch.tensor([e.ids for e in encoded], dtype=torch.long)
            mask = torch.tensor([e.attention_mask for e in encoded], dtype=torch.long)
            with torch.no_grad():
                hidden = self.module(ids, mask)
                # The same masked mean pooling the full path uses - padding must not
                # contribute, or short texts batched with long ones get diluted.
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(out)


def get_serving_embedder(artifacts: Path, model_name: str):
    """Traced encoder when the artifact exists, full transformers otherwise."""
    if (artifacts / "encoder_traced.pt").exists() and \
       (artifacts / "tokenizer.json").exists():
        return TracedEmbedder(artifacts), "traced"
    from src.retriever import Embedder
    return Embedder(model_name), "transformers"
