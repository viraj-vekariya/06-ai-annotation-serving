"""Combine the retriever's score with the language model's score.

THE PREMISE. The two stages know different things. The retriever knows which training
messages this one resembles; the language model knows what the label words mean. Neither
subsumes the other, so a weighted combination can beat both - or it can beat neither, and
the sweep below is what decides which.

THE STANDARDISATION IS THE LOAD-BEARING PART. The retriever produces normalised vote
shares; the LM produces a softmax over log-likelihoods. Those live on completely different
scales and have completely different spreads, and adding them raw means whichever happens
to have the larger variance dominates regardless of which is more informative. Both are
z-scored WITHIN each example before mixing, so alpha means what it appears to mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def standardise(values: Sequence[float]) -> np.ndarray:
    """Z-score within one example's candidate set.

    Within-example, not across the dataset: the comparison that matters is between
    candidates for the SAME message. A global standardisation would let an easy message
    with high scores everywhere shift a hard one's ranking.
    """
    arr = np.asarray(values, dtype=float)
    sd = arr.std()
    if sd < 1e-12:
        # All candidates identical - no information to extract, and dividing by ~0 would
        # manufacture enormous spurious differences from floating-point noise.
        return np.zeros_like(arr)
    return (arr - arr.mean()) / sd


@dataclass
class FusedPrediction:
    text: str
    labels: List[str]
    fused_scores: List[float]
    probabilities: List[float]
    alpha: float

    def top(self) -> Tuple[str, float]:
        i = int(np.argmax(self.probabilities))
        return self.labels[i], self.probabilities[i]

    def as_dict(self) -> Dict[str, object]:
        order = np.argsort(-np.asarray(self.probabilities))
        return {"alpha": self.alpha,
                "predictions": [{"label": self.labels[i],
                                 "probability": round(self.probabilities[i], 4)}
                                for i in order[:5]]}


def fuse(labels: Sequence[str], retriever_scores: Sequence[float],
         lm_scores: Sequence[float], alpha: float, text: str = "") -> FusedPrediction:
    """alpha=1 is retriever-only, alpha=0 is LM-only."""
    r = standardise(retriever_scores)
    l = standardise(lm_scores)
    fused = alpha * r + (1 - alpha) * l

    shifted = np.exp(fused - fused.max())
    probabilities = shifted / shifted.sum()
    return FusedPrediction(text, list(labels), [float(v) for v in fused],
                           [float(p) for p in probabilities], alpha)


def sweep_alpha(labels_list: Sequence[Sequence[str]],
                retriever_list: Sequence[Sequence[float]],
                lm_list: Sequence[Sequence[float]],
                gold: Sequence[str],
                alphas: Sequence[float] = tuple(np.round(np.arange(0, 1.01, 0.1), 2))
                ) -> Dict[str, object]:
    """Accuracy across the whole alpha range.

    The endpoints are the individual systems, so the sweep answers "does fusion beat
    either component?" in one table rather than as a claim. Reporting the whole curve
    rather than the argmax also shows whether the optimum is a broad plateau (robust) or
    a narrow spike (probably overfitting to this test set).
    """
    rows = []
    for alpha in alphas:
        hits = 0
        for labels, r, l, g in zip(labels_list, retriever_list, lm_list, gold):
            if fuse(labels, r, l, float(alpha)).top()[0] == g:
                hits += 1
        rows.append({"alpha": float(alpha),
                     "accuracy": round(hits / len(gold), 4) if gold else 0.0})

    best = max(rows, key=lambda row: row["accuracy"])
    retriever_only = next(r for r in rows if r["alpha"] == 1.0)
    lm_only = next(r for r in rows if r["alpha"] == 0.0)

    return {
        "sweep": rows,
        "best": best,
        "retriever_only": retriever_only,
        "lm_only": lm_only,
        "fusion_beats_retriever": best["accuracy"] > retriever_only["accuracy"],
        "fusion_beats_lm": best["accuracy"] > lm_only["accuracy"],
        "gain_over_retriever": round(best["accuracy"] - retriever_only["accuracy"], 4),
        "gain_over_lm": round(best["accuracy"] - lm_only["accuracy"], 4),
    }
