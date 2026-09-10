"""The four-arm ablation: which component actually earns its place?

THE ARMS, each isolating one variable:

  A  retriever only        kNN vote over embeddings. No language model at all.
  B  LM only               the LM scores the shortlist; the retriever's own ranking is
                           discarded. Isolates what the LM contributes on its own.
  C  fusion                both, weighted. Does combining beat either?
  D  LM with few-shot      the same LM, given labelled examples in the prompt. Isolates
                           whether in-context examples help.

WHY ARM D IS SCREENED ON A SUBSET. Its prompts carry several example messages, so they are
roughly 3x longer and 3x slower. Running it on the full test set would triple the ablation's
runtime to answer one question. It is evaluated on a fixed subset, and the WINNER RULE is
two-step: eliminate D on its own rows if it loses there, then choose among A/B/C on the
full set. Comparing D's subset score directly against the others' full-set scores would be
comparing different quantities.

Everything is cached to disk. The LM scoring pass is the expensive part of the whole
project, and re-running it to change a downstream threshold would make iteration
impossible.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "artifacts"

FEW_SHOT_PROMPT = (
    "Classify the bank customer's message into one of the candidate intents.\n\n"
    "{examples}\n"
    "Message: \"{text}\"\n"
    "Question: is this message about {label}?\n"
    "Answer:")


@dataclass
class ArmResult:
    arm: str
    description: str
    accuracy: float
    n: int
    lm_calls: int
    seconds: float
    full_test_set: bool

    def as_dict(self) -> Dict[str, object]:
        return {"arm": self.arm, "description": self.description,
                "accuracy": self.accuracy, "n": self.n, "lm_calls": self.lm_calls,
                "seconds": round(self.seconds, 1),
                "evaluated_on_full_test_set": self.full_test_set}


def build_few_shot_block(retriever, text: str, n_examples: int = 3) -> str:
    """Nearest labelled neighbours as in-context examples.

    Nearest rather than random: the point of few-shot is to show the model what the
    decision looks like *near this input*. Random examples from 77 classes are almost
    always irrelevant and mostly consume context.
    """
    shortlist = retriever.shortlist([text], top_n=1)[0]
    lines = [f"Message: \"{t}\" -> {lab.replace('_', ' ')}"
             for t, lab, _ in shortlist.neighbours[:n_examples]]
    return "\n".join(lines)


def run_ablation(retriever, scorer, test_texts: Sequence[str], gold: Sequence[str],
                 top_n: int = 8, few_shot_subset: int = 400,
                 cache_name: str = "ablation_cache.json") -> Dict[str, object]:
    from .fusion import fuse, sweep_alpha
    from .retriever import recall_at_n, top1_accuracy
    from .scorer import accuracy as lm_accuracy, verbalise

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE / cache_name

    # -- shortlist (shared by every arm) ----------------------------------------------
    started = time.time()
    shortlists = retriever.shortlist(list(test_texts), top_n=top_n)
    shortlist_seconds = time.time() - started
    ceiling = recall_at_n(shortlists, gold)

    # -- the expensive LM pass, cached -------------------------------------------------
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    signature = f"{scorer.name}|{len(test_texts)}|{top_n}"

    if cached.get("signature") == signature:
        lm_scores = cached["lm_scores"]
        lm_seconds = cached["lm_seconds"]
        lm_calls = cached["lm_calls"]
        print(f"    reusing cached LM scores ({len(lm_scores)} examples)")
    else:
        print(f"    scoring {len(test_texts)} x {top_n} candidates with {scorer.name}")
        started = time.time()
        before = scorer.calls
        scored = scorer.score_many(list(test_texts), [s.labels for s in shortlists])
        lm_seconds = time.time() - started
        lm_calls = scorer.calls - before
        lm_scores = [s.log_likelihoods for s in scored]
        cache_path.write_text(json.dumps({
            "signature": signature, "lm_scores": lm_scores,
            "lm_seconds": lm_seconds, "lm_calls": lm_calls}))

    labels_list = [s.labels for s in shortlists]
    retriever_scores = [s.scores for s in shortlists]

    # -- arm A: retriever only ---------------------------------------------------------
    arm_a = ArmResult(
        "A", "retriever only (kNN vote over embeddings, no LM)",
        top1_accuracy(shortlists, gold), len(gold), 0, shortlist_seconds, True)

    # -- arm B: LM only ----------------------------------------------------------------
    lm_hits = sum(1 for labels, scores, g in zip(labels_list, lm_scores, gold)
                  if labels[int(np.argmax(scores))] == g)
    arm_b = ArmResult(
        "B", "LM only (LM re-ranks the shortlist, retriever ranking discarded)",
        round(lm_hits / len(gold), 4), len(gold), lm_calls, lm_seconds, True)

    # -- arm C: fusion -----------------------------------------------------------------
    started = time.time()
    sweep = sweep_alpha(labels_list, retriever_scores, lm_scores, gold)
    arm_c = ArmResult(
        "C", f"fusion (alpha={sweep['best']['alpha']}, both signals z-scored per example)",
        sweep["best"]["accuracy"], len(gold), lm_calls,
        lm_seconds + (time.time() - started), True)

    # -- arm D: few-shot LM, on a subset ------------------------------------------------
    subset = min(few_shot_subset, len(test_texts))
    print(f"    arm D: few-shot on a {subset}-row subset (prompts are ~3x longer)")
    started = time.time()
    before = scorer.calls
    few_hits = 0
    for i in range(subset):
        text = test_texts[i]
        examples = build_few_shot_block(retriever, text)
        candidates = labels_list[i]
        prompts = [FEW_SHOT_PROMPT.format(examples=examples, text=text,
                                          label=verbalise(c)) for c in candidates]
        scores: List[float] = []
        for start in range(0, len(prompts), scorer.batch_size):
            chunk = prompts[start:start + scorer.batch_size]
            scores += scorer._score_batch(chunk, ["yes"] * len(chunk))
        if candidates[int(np.argmax(scores))] == gold[i]:
            few_hits += 1
    arm_d = ArmResult(
        "D", f"LM with 3 nearest-neighbour examples in the prompt (subset of {subset})",
        round(few_hits / subset, 4), subset, scorer.calls - before,
        time.time() - started, False)

    # -- the two-step winner rule ------------------------------------------------------
    # D is compared against the others RESTRICTED TO ITS OWN ROWS, because comparing a
    # subset score against full-set scores compares different quantities.
    subset_gold = list(gold[:subset])
    a_on_subset = top1_accuracy(shortlists[:subset], subset_gold)
    d_survives = arm_d.accuracy > a_on_subset

    full_set_arms = [arm_a, arm_b, arm_c]
    winner = max(full_set_arms, key=lambda a: a.accuracy)

    return {
        "shortlist_ceiling": ceiling,
        "arms": [a.as_dict() for a in (arm_a, arm_b, arm_c, arm_d)],
        "alpha_sweep": sweep,
        "few_shot_comparison": {
            "arm_d_accuracy_on_subset": arm_d.accuracy,
            "arm_a_accuracy_on_same_subset": a_on_subset,
            "few_shot_helps": d_survives,
            "subset_size": subset,
        },
        "winner": {"arm": winner.arm, "description": winner.description,
                   "accuracy": winner.accuracy},
        "verdict": _verdict(arm_a, arm_b, arm_c, ceiling),
    }


def _verdict(arm_a: ArmResult, arm_b: ArmResult, arm_c: ArmResult,
             ceiling: Dict[str, float]) -> str:
    best = max(arm_a.accuracy, arm_b.accuracy, arm_c.accuracy)
    if arm_a.accuracy >= best:
        return (f"The retriever alone wins at {arm_a.accuracy:.4f}. The language model "
                f"does not earn its {arm_b.lm_calls:,} forward passes: on its own it "
                f"scores {arm_b.accuracy:.4f}, and fusing it in reaches "
                f"{arm_c.accuracy:.4f}. Ship the retriever.")
    if arm_c.accuracy > max(arm_a.accuracy, arm_b.accuracy):
        return (f"Fusion wins at {arm_c.accuracy:.4f}, above the retriever's "
                f"{arm_a.accuracy:.4f} and the LM's {arm_b.accuracy:.4f} - the two "
                f"signals are complementary rather than redundant.")
    return (f"The LM alone wins at {arm_b.accuracy:.4f} against the retriever's "
            f"{arm_a.accuracy:.4f}.")
