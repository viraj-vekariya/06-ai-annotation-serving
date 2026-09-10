"""Dense bi-encoder shortlist: the first stage, and the one that sets the ceiling.

WHAT IT DOES. Embed every labelled training message once. At query time, embed the incoming
message and find its nearest neighbours; each neighbour votes for its own label. The top-N
labels become the candidate set that the language model then scores.

WHY A SHORTLIST AT ALL. Scoring an LLM against all 77 labels costs 77 forward passes per
message. Shortlisting to 8 costs 8, a 9.6x reduction - and the recall@8 measured below says
how much accuracy that trades away. That number is the CEILING on the whole pipeline: if the
right label is not in the shortlist, no amount of downstream cleverness can recover it.

Measuring and reporting that ceiling is the point. A two-stage system whose first stage
quietly loses 12% of the answers will look like a second-stage problem forever.

kNN voting rather than a centroid per class: intents like `card_arrival` have genuinely
multi-modal phrasings ("where is my card", "still waiting", "ordered two weeks ago") and a
single centroid averages them into something that matches none of them well.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_NAME = os.environ.get("ANNOTATION_EMBED_MODEL",
                            "sentence-transformers/all-MiniLM-L6-v2")


class Embedder:
    """MiniLM with masked mean pooling.

    Mean pooling, not [CLS]: this model family's sentence-level behaviour comes from a
    mean-pooling training objective, and [CLS] is not trained to carry it. Using [CLS]
    here is a common mistake that silently degrades every downstream number.
    """

    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 128):
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.torch = torch
        self.name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.dim = self.model.config.hidden_size

    def encode(self, texts: Sequence[str], max_length: int = 64) -> np.ndarray:
        """max_length=64 because the messages average 11.9 words; 512 would pad every
        batch to 8x its necessary width and cost 8x the compute for nothing."""
        torch = self.torch
        out: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                                         max_length=max_length, return_tensors="pt")
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)


@dataclass
class Shortlist:
    text: str
    labels: List[str]
    scores: List[float]
    neighbours: List[Tuple[str, str, float]]     # (neighbour text, label, similarity)

    def as_dict(self) -> Dict[str, object]:
        return {"labels": self.labels, "scores": [round(s, 4) for s in self.scores],
                "neighbours": [{"text": t, "label": l, "similarity": round(s, 4)}
                               for t, l, s in self.neighbours[:5]]}


class Retriever:
    def __init__(self, embedder: Embedder, texts: Sequence[str], labels: Sequence[str],
                 k_neighbours: int = 24):
        self.embedder = embedder
        self.texts = list(texts)
        self.labels = list(labels)
        self.k_neighbours = k_neighbours
        self.label_set = sorted(set(labels))
        self.matrix = embedder.encode(self.texts)

    def _neighbours(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Both sides are unit-normalised, so a dot product IS cosine similarity. One
        # matrix multiply for the whole batch beats a Python loop by two orders of
        # magnitude, and at 10,003 x 384 an exact scan takes single-digit milliseconds -
        # an approximate index would be slower and non-deterministic.
        similarity = query_vectors @ self.matrix.T
        k = min(k, similarity.shape[1])
        idx = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        rows = np.arange(len(query_vectors))[:, None]
        order = np.argsort(-similarity[rows, idx], axis=1)
        idx = idx[rows, order]
        return idx, similarity[rows, idx]

    def shortlist(self, queries: Sequence[str], top_n: int = 8) -> List[Shortlist]:
        vectors = self.embedder.encode(queries)
        idx, sims = self._neighbours(vectors, self.k_neighbours)

        results: List[Shortlist] = []
        for qi, query in enumerate(queries):
            votes: Dict[str, float] = defaultdict(float)
            neighbours: List[Tuple[str, str, float]] = []
            for j in range(idx.shape[1]):
                neighbour = int(idx[qi, j])
                similarity = float(sims[qi, j])
                label = self.labels[neighbour]
                # Similarity-weighted votes, not raw counts. A neighbour at cosine 0.91
                # is much stronger evidence than one at 0.42, and counting them equally
                # lets a cluster of weak matches outvote one strong one.
                votes[label] += similarity
                neighbours.append((self.texts[neighbour], label, similarity))

            ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            total = sum(v for _, v in ranked) or 1.0
            results.append(Shortlist(
                text=query,
                labels=[l for l, _ in ranked],
                scores=[v / total for _, v in ranked],
                neighbours=neighbours))
        return results


def recall_at_n(shortlists: Sequence[Shortlist], gold: Sequence[str]) -> Dict[str, float]:
    """The ceiling on everything downstream.

    Reported at several N so the shortlist size is a visible trade rather than a constant
    somebody chose once. If recall@8 is 0.88, the best any downstream stage can achieve
    is 0.88 - and a pipeline reporting 0.63 accuracy has a second-stage problem, not a
    retrieval problem. Knowing which is the entire value of this number.
    """
    out: Dict[str, float] = {}
    for n in (1, 3, 5, 8):
        hits = sum(1 for s, g in zip(shortlists, gold) if g in s.labels[:n])
        out[f"recall_at_{n}"] = round(hits / len(gold), 4) if gold else 0.0
    out["n"] = len(gold)
    return out


def top1_accuracy(shortlists: Sequence[Shortlist], gold: Sequence[str]) -> float:
    """The retriever ALONE, with no language model at all.

    This is the baseline the LLM has to beat. If the LLM cannot improve on a kNN vote over
    embeddings, it is not earning its latency or its cost - and that is a finding worth
    having rather than an embarrassment to hide.
    """
    hits = sum(1 for s, g in zip(shortlists, gold) if s.labels and s.labels[0] == g)
    return round(hits / len(gold), 4) if gold else 0.0
