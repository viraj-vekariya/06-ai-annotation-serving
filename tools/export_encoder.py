"""Trace MiniLM to TorchScript so the serving image does not need `transformers`.

Run at image build time. Verifies the traced encoder reproduces the full model's output
before writing anything - an encoder that silently drifted would change every prediction
while every test still passed.

Run:  python3 tools/export_encoder.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARTIFACTS = ROOT / "artifacts"
TOLERANCE = 1e-4


def main() -> int:
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = json.loads((ARTIFACTS / "serving.json").read_text())
    name = cfg["embed_model"]

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()

    sample = ["my card has not arrived", "how do I top up my account?",
              "I need to reset my pin", "asdfgh qwerty"]
    enc = tokenizer(sample, padding=True, truncation=True, max_length=64,
                    return_tensors="pt")

    class Encoder(torch.nn.Module):
        """Wraps the model so the traced signature is (ids, mask) -> hidden states.

        Tracing the raw model would capture its keyword-argument signature, which a
        TorchScript consumer cannot call without transformers' own tooling.
        """

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state

    wrapper = Encoder(model).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (enc["input_ids"], enc["attention_mask"]),
                                 strict=False)

    out = ARTIFACTS / "encoder_traced.pt"
    traced.save(str(out))

    reloaded = torch.jit.load(str(out), map_location="cpu")
    reloaded.eval()
    with torch.no_grad():
        original = wrapper(enc["input_ids"], enc["attention_mask"])
        replayed = reloaded(enc["input_ids"], enc["attention_mask"])
    diff = float((original - replayed).abs().max())
    if diff > TOLERANCE:
        out.unlink(missing_ok=True)
        print(f"EXPORT FAILED: traced encoder differs by {diff:.2e}", file=sys.stderr)
        return 1

    tokenizer.backend_tokenizer.save(str(ARTIFACTS / "tokenizer.json"))
    (ARTIFACTS / "encoder_meta.json").write_text(json.dumps({
        "model": name, "dim": model.config.hidden_size, "max_length": 64,
        "max_abs_difference": float(f"{diff:.3e}"),
    }, indent=2))

    print(f"  traced {name}")
    print(f"  verified on {len(sample)} texts, max difference {diff:.2e}")
    print(f"  wrote artifacts/encoder_traced.pt "
          f"({out.stat().st_size/1e6:.0f} MB) + tokenizer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
