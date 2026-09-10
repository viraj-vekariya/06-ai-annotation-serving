"""Temperature scaling, and the reliability evidence behind it.

WHY CALIBRATION IS NOT OPTIONAL HERE. The whole deployment routes on confidence: anything
below a threshold is sent to a human. That routing is built on a lie unless the model's
0.90 actually means "right about 90% of the time".

Temperature scaling is one scalar T, fitted by minimising NLL on a held-out split. It
cannot change any prediction - dividing every logit by the same positive number preserves
the argmax exactly - so accuracy is untouched and only the confidences move. That property
is what makes it safe to apply after the fact.

T > 1 softens (the model was over-confident); T < 1 sharpens (under-confident). Which way
it goes is itself a finding worth reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CalibrationReport:
    temperature: float
    ece_before: float
    ece_after: float
    mean_confidence_before: float
    mean_confidence_after: float
    accuracy: float
    n: int
    reliability: List[Dict[str, float]]
    direction: str

    def as_dict(self) -> Dict[str, object]:
        return {"temperature": round(self.temperature, 4),
                "ece_before": round(self.ece_before, 4),
                "ece_after": round(self.ece_after, 4),
                "ece_reduction_pct": round(
                    100 * (self.ece_before - self.ece_after) / self.ece_before, 2)
                if self.ece_before else 0.0,
                "mean_confidence_before": round(self.mean_confidence_before, 4),
                "mean_confidence_after": round(self.mean_confidence_after, 4),
                "accuracy": round(self.accuracy, 4), "n": self.n,
                "direction": self.direction, "reliability": self.reliability}


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = scores / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=1, keepdims=True)   # overflow-safe, exact
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities: np.ndarray, correct: np.ndarray,
                               bins: int = 15) -> float:
    """Average gap between confidence and accuracy, weighted by bin population."""
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def reliability_table(probabilities: np.ndarray, correct: np.ndarray,
                      bins: int = 10) -> List[Dict[str, float]]:
    """The evidence behind the ECE number.

    ECE is a single summary and equal-width bins under-weight the high-confidence region
    where most predictions live. The table is the real evidence; the scalar is the
    headline.
    """
    confidence = probabilities.max(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                     "confidence": round(float(confidence[mask].mean()), 4),
                     "accuracy": round(float(correct[mask].mean()), 4),
                     "gap": round(float(correct[mask].mean() - confidence[mask].mean()), 4)})
    return rows


def fit_temperature(scores: np.ndarray, correct_index: np.ndarray,
                    lo: float = 0.05, hi: float = 20.0, iterations: int = 60) -> float:
    """Minimise NLL over T by golden-section search.

    NLL as a function of T is smooth and unimodal on this interval, so a derivative-free
    line search converges in a handful of evaluations. It is used instead of gradient
    descent because there is nothing to tune - no learning rate, no stopping criterion -
    and one scalar does not justify an optimiser.
    """
    def nll(t: float) -> float:
        probabilities = softmax(scores, t)
        chosen = probabilities[np.arange(len(correct_index)), correct_index]
        return float(-np.log(np.clip(chosen, 1e-12, 1.0)).mean())

    phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
    return (a + b) / 2


def calibrate(scores: Sequence[Sequence[float]], labels: Sequence[Sequence[str]],
              gold: Sequence[str]) -> CalibrationReport:
    """Fit T and report before/after. Scores are per-example candidate scores."""
    width = max(len(s) for s in scores)
    # Ragged candidate lists are padded with -inf, which softmaxes to exactly zero. A
    # zero pad would instead give every short row extra probability mass on a candidate
    # that does not exist.
    matrix = np.full((len(scores), width), -np.inf)
    for i, row in enumerate(scores):
        matrix[i, :len(row)] = row

    correct_index = np.array([
        labels[i].index(gold[i]) if gold[i] in labels[i] else 0
        for i in range(len(gold))])
    in_shortlist = np.array([gold[i] in labels[i] for i in range(len(gold))])

    # T is fitted only on examples whose gold label is present. For the rest there is no
    # correct class to assign probability to, and including them would push T toward
    # flattening every distribution to hedge against the unreachable.
    temperature = fit_temperature(matrix[in_shortlist], correct_index[in_shortlist])

    before = softmax(matrix, 1.0)
    after = softmax(matrix, temperature)
    predicted = before.argmax(axis=1)
    correct = np.array([labels[i][predicted[i]] == gold[i] for i in range(len(gold))],
                       dtype=float)

    return CalibrationReport(
        temperature=temperature,
        ece_before=expected_calibration_error(before, correct),
        ece_after=expected_calibration_error(after, correct),
        mean_confidence_before=float(before.max(axis=1).mean()),
        mean_confidence_after=float(after.max(axis=1).mean()),
        accuracy=float(correct.mean()), n=len(gold),
        reliability=reliability_table(after, correct),
        direction=("over-confident (T>1 softens it)" if temperature > 1
                   else "under-confident (T<1 sharpens it)"))
