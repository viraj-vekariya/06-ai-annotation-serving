"""A small student model, trained on the pipeline's own labels.

THE POINT OF DISTILLATION HERE. The retriever needs a 22.7M-parameter transformer and a
10,003-vector index in memory to answer one message. If a much smaller model trained on the
retriever's OUTPUT can match it, the serving path loses the transformer entirely - and that
is a real operational win: smaller image, faster cold start, no torch on the request path.

The question is whether it can. That is measured, not assumed.

ARCHITECTURE: a single hidden layer over frozen sentence embeddings. Deliberately small.
The interesting comparison is not "can a big model match a big model" - it is how much can
be given up before quality moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class StudentReport:
    architecture: str
    parameters: int
    train_examples: int
    teacher_accuracy: float
    student_accuracy: float
    agreement_with_teacher: float
    beats_teacher: bool
    gap: float
    epochs: int
    seconds: float

    def as_dict(self) -> Dict[str, object]:
        return {"architecture": self.architecture, "parameters": self.parameters,
                "train_examples": self.train_examples,
                "teacher_accuracy": round(self.teacher_accuracy, 4),
                "student_accuracy": round(self.student_accuracy, 4),
                "agreement_with_teacher": round(self.agreement_with_teacher, 4),
                "beats_teacher": self.beats_teacher, "gap": round(self.gap, 4),
                "epochs": self.epochs, "seconds": round(self.seconds, 1)}


class NeuralStudent:
    """MLP over frozen embeddings: Linear -> ReLU -> Dropout -> Linear.

    Frozen embeddings rather than fine-tuning the encoder: fine-tuning would need the
    encoder at serving time, which is the exact dependency this is trying to remove.
    """

    def __init__(self, input_dim: int, n_classes: int, hidden: int = 256,
                 dropout: float = 0.2, seed: int = 20260910):
        import torch
        import torch.nn as nn

        self.torch = torch
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.n_classes = n_classes
        self.classes: List[str] = []

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def fit(self, X: np.ndarray, y: np.ndarray, classes: Sequence[str],
            epochs: int = 40, batch_size: int = 128, lr: float = 1e-3,
            weight_decay: float = 1e-4, seed: int = 20260910) -> "NeuralStudent":
        import torch
        import torch.nn as nn

        self.classes = list(classes)
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        optimiser = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        # Cosine annealing rather than a fixed rate: 40 epochs on 10k rows overfits at a
        # constant learning rate, and annealing is one line against a scheduler to tune.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        n = len(X_t)
        self.model.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(self.model(X_t[idx]), y_t[idx])
                loss.backward()
                optimiser.step()
            scheduler.step()
        self.model.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: np.ndarray) -> List[str]:
        idx = self.predict_proba(X).argmax(axis=1)
        return [self.classes[i] for i in idx]

    def save(self, path) -> None:
        import torch

        torch.save({"state_dict": self.model.state_dict(),
                    "classes": self.classes,
                    "input_dim": self.model[0].in_features,
                    "hidden": self.model[0].out_features}, path)

    @classmethod
    def load(cls, path) -> "NeuralStudent":
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
        student = cls(blob["input_dim"], len(blob["classes"]), hidden=blob["hidden"])
        student.model.load_state_dict(blob["state_dict"])
        student.model.eval()
        student.classes = blob["classes"]
        return student


class LinearStudent:
    """Multinomial logistic regression on the same features.

    The control. If the MLP does not beat this, the extra capacity is not buying anything
    and the simplest model should ship - a comparison that is easy to skip and easy to be
    wrong about.
    """

    def __init__(self, seed: int = 20260910):
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(max_iter=1500, C=4.0, random_state=seed,
                                        n_jobs=2)
        self.classes: List[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, classes: Sequence[str]) -> "LinearStudent":
        self.classes = list(classes)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> List[str]:
        return [self.classes[i] for i in self.model.predict(X)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def parameter_count(self) -> int:
        return int(self.model.coef_.size + self.model.intercept_.size)
