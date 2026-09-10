"""Score shortlisted candidate labels with a language model.

THE METHOD. For each candidate label, ask the model how likely it is that this message has
that intent, and read the answer off the model's own probabilities rather than its
generated text. Concretely: build a prompt per (message, candidate) and take the
length-normalised log-likelihood the model assigns to the candidate's verbalisation.

WHY SCORING RATHER THAN GENERATING. Asking a model to *emit* one of 77 labels means parsing
free text, handling near-misses ("card arrival" vs "card_arrival"), and having no
confidence number at the end. Constrained scoring gives a real distribution over exactly
the candidates, which is what calibration and abstention downstream both need.

WHY LENGTH-NORMALISED. Log-likelihood is a sum over tokens, so a longer verbalisation is
mechanically less likely. Without dividing by token count the model would systematically
prefer `atm_support` over `card_payment_wrong_exchange_rate` for reasons that have nothing
to do with the message. This is the single most common bug in constrained-decoding
classifiers.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_NAME = os.environ.get("ANNOTATION_LM", "google/flan-t5-base")

PROMPT = ("A bank customer wrote: \"{text}\"\n"
          "Question: is this message about {label}?\n"
          "Answer:")


def verbalise(label: str) -> str:
    """`card_payment_fee_charged` -> `card payment fee charged`.

    The raw label is a snake_case identifier that never appears in natural text, so the
    model has no learned representation of it. Verbalising costs nothing and lets the
    model use what it actually knows about the words.
    """
    return label.replace("_", " ")


@dataclass
class ScoredCandidates:
    text: str
    labels: List[str]
    log_likelihoods: List[float]
    probabilities: List[float]

    def top(self) -> Tuple[str, float]:
        i = int(np.argmax(self.probabilities))
        return self.labels[i], self.probabilities[i]

    def as_dict(self) -> Dict[str, object]:
        return {"labels": self.labels,
                "log_likelihoods": [round(v, 4) for v in self.log_likelihoods],
                "probabilities": [round(v, 4) for v in self.probabilities]}


class LMScorer:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32,
                 device: Optional[str] = None):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.torch = torch
        self.name = model_name
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        # MPS is available on this machine and is ~3x faster than CPU for this model.
        # Explicit rather than automatic so a CI run on CPU behaves identically.
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)
        self.calls = 0

    def _score_batch(self, prompts: Sequence[str],
                     targets: Sequence[str]) -> List[float]:
        """Length-normalised log P(target | prompt) for each pair."""
        torch = self.torch
        encoded = self.tokenizer(list(prompts), padding=True, truncation=True,
                                 max_length=128, return_tensors="pt").to(self.device)
        labels = self.tokenizer(list(targets), padding=True, truncation=True,
                                max_length=16, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**encoded, labels=labels["input_ids"]).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(
                2, labels["input_ids"].unsqueeze(-1)).squeeze(-1)
            # Padding positions must not contribute. Without the mask, a short target
            # padded to the batch width accumulates the log-probability of PAD tokens,
            # which is arbitrary and differs by batch composition - so the same input
            # would score differently depending on what it was batched with.
            mask = labels["attention_mask"].float()
            summed = (token_lp * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            normalised = summed / lengths

        self.calls += len(prompts)
        return [float(v) for v in normalised.cpu().numpy()]

    def score(self, text: str, candidates: Sequence[str]) -> ScoredCandidates:
        prompts = [PROMPT.format(text=text, label=verbalise(c)) for c in candidates]
        # "yes" for every pair: the score is how strongly the model endorses the
        # proposition, and comparing endorsement across candidates is the ranking.
        targets = ["yes"] * len(candidates)

        log_likelihoods: List[float] = []
        for start in range(0, len(prompts), self.batch_size):
            log_likelihoods += self._score_batch(
                prompts[start:start + self.batch_size],
                targets[start:start + self.batch_size])

        # Softmax over candidates. Subtracting the max first is not cosmetic: raw
        # exponentials of log-likelihoods overflow, and the shift is mathematically
        # exact.
        arr = np.array(log_likelihoods)
        shifted = np.exp(arr - arr.max())
        probabilities = shifted / shifted.sum()

        return ScoredCandidates(text, list(candidates), log_likelihoods,
                                [float(p) for p in probabilities])

    def score_many(self, texts: Sequence[str],
                   candidate_lists: Sequence[Sequence[str]],
                   progress_every: int = 200) -> List[ScoredCandidates]:
        out: List[ScoredCandidates] = []
        started = time.time()
        for i, (text, candidates) in enumerate(zip(texts, candidate_lists)):
            out.append(self.score(text, candidates))
            if progress_every and (i + 1) % progress_every == 0:
                rate = (i + 1) / (time.time() - started)
                print(f"    scored {i + 1}/{len(texts)}  ({rate:.1f}/s)", flush=True)
        return out


def accuracy(scored: Sequence[ScoredCandidates], gold: Sequence[str]) -> float:
    hits = sum(1 for s, g in zip(scored, gold) if s.top()[0] == g)
    return round(hits / len(gold), 4) if gold else 0.0
