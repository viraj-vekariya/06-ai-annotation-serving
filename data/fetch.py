"""Download BANKING77: 13,083 real customer-support messages across 77 intents.

Source: PolyAI's task-specific-datasets repository. Public, no credentials.

WHY THIS DATASET. The intents are deliberately fine-grained and near-synonymous -
`card_arrival` vs `card_delivery_estimate`, `card_payment_fee_charged` vs
`extra_charge_on_statement`, `top_up_failed` vs `top_up_reverted`. That is what makes it a
real annotation problem: a human annotator has to think, and a model has to distinguish
categories that a keyword matcher never could. A dataset with 5 obvious classes would make
every method look good and tell you nothing about which to use.

77 classes over 13,083 examples also means roughly 170 examples per class, which is exactly
the regime where "should we pay humans or use an LLM?" is a live question rather than an
obvious one.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ANNOTATION_DATA", ROOT / "data"))
BASE = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        "master/banking_data")
UA = {"User-Agent": "Mozilla/5.0 (compatible; placement-project/1.0)"}


def _get(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                timeout=90) as response:
        dest.write_bytes(response.read())
    return dest


def fetch() -> Dict[str, object]:
    train = _get(f"{BASE}/train.csv", DATA / "raw" / "train.csv")
    test = _get(f"{BASE}/test.csv", DATA / "raw" / "test.csv")
    categories = _get(f"{BASE}/categories.json", DATA / "raw" / "categories.json")

    labels = json.loads(categories.read_text())

    def read(path: Path) -> List[Tuple[str, str]]:
        # csv.reader, not a manual split: several messages contain commas inside quoted
        # fields ("I still have not received my new card, I ordered over a week ago.").
        # A naive split on ',' silently truncates them and shifts the label column.
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            assert header == ["text", "category"], f"unexpected header {header}"
            return [(row[0], row[1]) for row in reader if len(row) == 2]

    train_rows = read(train)
    test_rows = read(test)

    unknown = {c for _, c in train_rows + test_rows} - set(labels)
    if unknown:
        raise ValueError(f"labels not in categories.json: {sorted(unknown)[:5]}")

    counts = Counter(c for _, c in train_rows)
    return {
        "labels": labels,
        "n_labels": len(labels),
        "train": train_rows,
        "test": test_rows,
        "stats": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "total_rows": len(train_rows) + len(test_rows),
            "labels": len(labels),
            "examples_per_label_mean": round(len(train_rows) / len(labels), 1),
            "examples_per_label_min": min(counts.values()),
            "examples_per_label_max": max(counts.values()),
            "rarest_label": counts.most_common()[-1][0],
            "commonest_label": counts.most_common(1)[0][0],
            "mean_words": round(
                sum(len(t.split()) for t, _ in train_rows) / len(train_rows), 1),
        },
    }


if __name__ == "__main__":
    payload = fetch()
    print(json.dumps(payload["stats"], indent=2))
    print(f"\n  first 3 labels: {payload['labels'][:3]}")
    print(f"  example: {payload['train'][0][0]!r} -> {payload['train'][0][1]}")
