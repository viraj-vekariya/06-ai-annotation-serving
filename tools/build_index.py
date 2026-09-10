"""Rebuild the serving index inside the image, and prove it is the measured one.

WHY THIS EXISTS. `artifacts/index.npz` is 14 MB of derived floats and is deliberately not
committed - binaries in git history make every diff unreadable. But the image needs it, so
the build has to regenerate it from BANKING77.

Regenerating opens a real risk: an index built from a different encoder, a different
tokenizer version or a shuffled corpus would still load, still answer, and be quietly
wrong. So this does not just rebuild - it replays the headline measurement. The retriever
scored 0.8920 top-1 on the first 1,500 test messages. If the rebuilt index does not
reproduce that exactly, the build fails rather than shipping a service whose numbers no
longer match the CV bullet.

Run:  python3 tools/build_index.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    from data.fetch import fetch
    from serve.light_embedder import get_serving_embedder
    from src.retriever import Retriever, top1_accuracy

    cfg = json.loads((ARTIFACTS / "serving.json").read_text())
    measured = json.loads((ROOT / "outputs" / "results.json").read_text())
    expected = measured["retrieval"]["top1_accuracy"]
    # Replay over exactly the rows the recorded number was computed on, read from the
    # results file rather than hard-coded - CI runs the pipeline at a reduced test size,
    # and a fixed 1,500 here would compare two different measurements and fail honestly
    # but uselessly.
    verify_n = int(measured["retrieval"]["ceiling"]["n"])

    data = fetch()
    train_texts = [t for t, _ in data["train"]]
    train_labels = [l for _, l in data["train"]]
    test = data["test"][:verify_n]

    embedder, backend = get_serving_embedder(ARTIFACTS, cfg["embed_model"])
    print(f"  encoder backend: {backend}")

    retriever = Retriever(embedder, train_texts, train_labels)
    print(f"  embedded {len(train_texts):,} corpus messages "
          f"-> {retriever.matrix.shape}")

    shortlists = retriever.shortlist([t for t, _ in test], top_n=1)
    actual = top1_accuracy(shortlists, [l for _, l in test])
    print(f"  replayed top-1 on {len(test):,} test messages: "
          f"{actual:.4f} (measured {expected:.4f})")
    if abs(actual - expected) > 1e-9:
        print("BUILD FAILED: the rebuilt index does not reproduce the measured "
              "accuracy, so the service would not be the system that was evaluated.",
              file=sys.stderr)
        return 1

    np.savez_compressed(ARTIFACTS / "index.npz", matrix=retriever.matrix,
                        labels=np.array(train_labels),
                        texts=np.array(train_texts, dtype=object))
    size = (ARTIFACTS / "index.npz").stat().st_size / 1e6
    print(f"  wrote artifacts/index.npz ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
