"""The abstention gate: when to answer, and when to send it to a human.

THE DECISION. An annotation system does not have to answer everything. It has to answer
what it can get right, and route the rest to a person. The gate is a confidence threshold,
and the only question is where to put it.

THE THRESHOLD IS DERIVED, NOT CHOSEN. Picking 0.8 because it looks reasonable is how these
systems end up either flooding the review queue or shipping errors. Instead: state the
PRECISION the business requires on auto-accepted items, then find the lowest threshold that
achieves it - which maximises coverage subject to that precision.

That inverts the usual framing. The business does not have an opinion about confidence
scores; it has an opinion about how often an automatic answer may be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GateResult:
    threshold: float
    coverage: float                 # share auto-accepted
    precision: float                # accuracy ON the auto-accepted share
    recall_of_correct: float        # share of all correct answers that are auto-accepted
    n_auto: int
    n_review: int
    errors_shipped: int
    target_precision: float
    achieved: bool

    def as_dict(self) -> Dict[str, object]:
        return {"threshold": round(self.threshold, 4),
                "coverage": round(self.coverage, 4),
                "precision": round(self.precision, 4),
                "recall_of_correct": round(self.recall_of_correct, 4),
                "auto_accepted": self.n_auto, "sent_to_review": self.n_review,
                "errors_shipped": self.errors_shipped,
                "target_precision": self.target_precision,
                "target_achieved": self.achieved}


def coverage_at_precision(confidence: np.ndarray, correct: np.ndarray,
                          target_precision: float = 0.95,
                          grid: int = 400) -> GateResult:
    """The lowest threshold meeting the precision target - i.e. maximum coverage.

    Lowest, not highest: any threshold above it also meets the target but auto-accepts
    fewer items, so it costs more human review for the same guarantee. Sweeping upward and
    taking the first success is the whole search.
    """
    thresholds = np.linspace(0.0, 1.0, grid)
    best: Optional[GateResult] = None

    for threshold in thresholds:
        mask = confidence >= threshold
        n_auto = int(mask.sum())
        if n_auto == 0:
            continue
        precision = float(correct[mask].mean())
        if precision >= target_precision:
            best = GateResult(
                threshold=float(threshold),
                coverage=n_auto / len(confidence),
                precision=precision,
                recall_of_correct=float(correct[mask].sum() / max(1, correct.sum())),
                n_auto=n_auto, n_review=len(confidence) - n_auto,
                errors_shipped=int((~correct[mask].astype(bool)).sum()),
                target_precision=target_precision, achieved=True)
            break

    if best is None:
        # Even accepting only the single most confident item misses the target. Report
        # the degenerate honest answer rather than silently returning a threshold that
        # does not do what it claims.
        mask = confidence >= confidence.max()
        best = GateResult(
            threshold=float(confidence.max()), coverage=float(mask.mean()),
            precision=float(correct[mask].mean()),
            recall_of_correct=float(correct[mask].sum() / max(1, correct.sum())),
            n_auto=int(mask.sum()), n_review=len(confidence) - int(mask.sum()),
            errors_shipped=int((~correct[mask].astype(bool)).sum()),
            target_precision=target_precision, achieved=False)
    return best


def sweep_targets(confidence: np.ndarray, correct: np.ndarray,
                  targets: Sequence[float] = (0.90, 0.95, 0.97, 0.99)
                  ) -> List[Dict[str, object]]:
    """The precision/coverage trade, as a table.

    One threshold is a decision; the table is what lets someone else make a different
    decision. Raising the precision requirement from 95% to 99% has a coverage cost, and
    that cost is what the conversation should be about.
    """
    return [coverage_at_precision(confidence, correct, t).as_dict() for t in targets]


def operating_curve(confidence: np.ndarray, correct: np.ndarray,
                    points: int = 25) -> List[Dict[str, float]]:
    """Coverage and precision across the whole threshold range, for plotting."""
    rows = []
    for threshold in np.linspace(0, 1, points):
        mask = confidence >= threshold
        if not mask.any():
            continue
        rows.append({"threshold": round(float(threshold), 4),
                     "coverage": round(float(mask.mean()), 4),
                     "precision": round(float(correct[mask].mean()), 4),
                     "n_auto": int(mask.sum())})
    return rows
