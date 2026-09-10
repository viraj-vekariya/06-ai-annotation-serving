"""Train the student on the TEACHER'S labels, and measure what it costs.

THE EXPERIMENT THAT MATTERS. There are two things a student could be trained on:

  * the GOLD labels - ordinary supervised learning, and not distillation at all;
  * the TEACHER'S labels - what you would actually have in production, where the whole
    point is that gold labels do not exist for new data.

Training on teacher labels is the honest version, and it has a ceiling: the student cannot
learn what the teacher got wrong. Whether it lands near that ceiling, or below it, or
somehow above it, is the measurement.

A student that BEATS its teacher is possible and worth watching for: it happens when the
teacher's errors are inconsistent noise and the student's inductive bias smooths them out.
It is also a claim that demands evidence, so both are reported side by side.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .student import LinearStudent, NeuralStudent, StudentReport


def encode_labels(labels: Sequence[str], classes: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[l] for l in labels], dtype=np.int64)


def distil(train_X: np.ndarray, teacher_labels: Sequence[str],
           test_X: np.ndarray, test_gold: Sequence[str],
           teacher_test_labels: Sequence[str], classes: Sequence[str],
           epochs: int = 40) -> Dict[str, object]:
    """Train both students on the teacher's labels and compare all three."""
    y = encode_labels(teacher_labels, classes)
    teacher_accuracy = float(np.mean([p == g for p, g in
                                      zip(teacher_test_labels, test_gold)]))

    started = time.time()
    neural = NeuralStudent(train_X.shape[1], len(classes)).fit(
        train_X, y, classes, epochs=epochs)
    neural_predictions = neural.predict(test_X)
    neural_seconds = time.time() - started

    started = time.time()
    linear = LinearStudent().fit(train_X, y, classes)
    linear_predictions = linear.predict(test_X)
    linear_seconds = time.time() - started

    def report(name: str, predictions: Sequence[str], params: int,
               seconds: float) -> StudentReport:
        accuracy = float(np.mean([p == g for p, g in zip(predictions, test_gold)]))
        agreement = float(np.mean([p == t for p, t in
                                   zip(predictions, teacher_test_labels)]))
        return StudentReport(
            architecture=name, parameters=params, train_examples=len(train_X),
            teacher_accuracy=teacher_accuracy, student_accuracy=accuracy,
            agreement_with_teacher=agreement,
            beats_teacher=accuracy > teacher_accuracy,
            gap=accuracy - teacher_accuracy, epochs=epochs, seconds=seconds)

    neural_report = report("MLP over frozen embeddings", neural_predictions,
                           neural.parameter_count(), neural_seconds)
    linear_report = report("multinomial logistic regression", linear_predictions,
                           linear.parameter_count(), linear_seconds)

    best = max((neural_report, linear_report), key=lambda r: r.student_accuracy)
    return {
        "teacher_accuracy": round(teacher_accuracy, 4),
        "neural_student": neural_report.as_dict(),
        "linear_student": linear_report.as_dict(),
        "mlp_beats_linear": neural_report.student_accuracy > linear_report.student_accuracy,
        "best_student": best.architecture,
        "verdict": _verdict(neural_report, linear_report),
        "models": {"neural": neural, "linear": linear},
    }


def _verdict(neural: StudentReport, linear: StudentReport) -> str:
    best = max(neural, linear, key=lambda r: r.student_accuracy)
    delta = best.student_accuracy - best.teacher_accuracy
    if best.beats_teacher:
        return (f"The {best.architecture} ({best.parameters:,} params) reaches "
                f"{best.student_accuracy:.4f}, BEATING its teacher's "
                f"{best.teacher_accuracy:.4f} by {delta:+.4f}. The teacher's errors were "
                f"inconsistent enough for the student's inductive bias to smooth them out.")
    retained = best.student_accuracy / best.teacher_accuracy if best.teacher_accuracy else 0
    return (f"The {best.architecture} ({best.parameters:,} params) reaches "
            f"{best.student_accuracy:.4f} against the teacher's "
            f"{best.teacher_accuracy:.4f} - {retained:.1%} of the teacher's accuracy, "
            f"{delta:+.4f}. Whether that trade is worth taking is a cost question, "
            f"not an accuracy question; see src/cost.py.")


def learning_curve(train_X: np.ndarray, teacher_labels: Sequence[str],
                   test_X: np.ndarray, test_gold: Sequence[str],
                   classes: Sequence[str],
                   fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
                   seed: int = 20260910) -> List[Dict[str, object]]:
    """Student accuracy against training-set size.

    This is what distinguishes "the student is too small" from "the student needs more
    labels". A curve still climbing at 100% means more teacher labels would help - which
    is cheap, since the teacher generates them. A flat curve means the ceiling is the
    architecture, and more data is wasted money. The two diagnoses have opposite actions.
    """
    rng = np.random.default_rng(seed)
    y = encode_labels(teacher_labels, classes)
    rows = []
    for fraction in fractions:
        n = max(len(classes), int(len(train_X) * fraction))
        idx = rng.choice(len(train_X), size=n, replace=False)
        student = NeuralStudent(train_X.shape[1], len(classes)).fit(
            train_X[idx], y[idx], classes, epochs=30)
        predictions = student.predict(test_X)
        rows.append({
            "fraction": fraction, "train_examples": n,
            "accuracy": round(float(np.mean([p == g for p, g in
                                             zip(predictions, test_gold)])), 4),
        })
    if len(rows) >= 2:
        last_gain = rows[-1]["accuracy"] - rows[-2]["accuracy"]
        for row in rows:
            row["still_improving"] = last_gain > 0.005
    return rows
