"""The cost/quality frontier: the decision the whole project exists to inform.

THE QUESTION. Pay humans to annotate, or automate? And if automating, with which stack?

The comparison has to be like-for-like, so every option is priced per 1,000 items at the
same quality bar, including the human review that its abstention gate sends back. An
automated option that is 99% cheaper but routes 60% of traffic to a person is not 99%
cheaper.

PRICES ARE ASSUMPTIONS AND ARE LABELLED AS SUCH. They are stated once, at the top, with a
source or a rationale - not scattered through the code as magic numbers. Anyone who
disagrees with a number can change it and re-run, which is the only honest way to present
a cost model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# ASSUMPTIONS. Change these and re-run; every figure downstream moves with them.
# ---------------------------------------------------------------------------
HUMAN_SECONDS_PER_ITEM = 30.0     # a 77-way fine-grained intent decision, read + choose
HUMAN_HOURLY_USD = 10.0           # a typical annotation-platform rate
HUMAN_COST_PER_1K = HUMAN_SECONDS_PER_ITEM / 3600.0 * HUMAN_HOURLY_USD * 1000

# Measured on this machine, not quoted from a spec sheet: the LM scoring pass ran at
# ~20 examples/second on MPS, each example costing 8 forward passes.
LM_ITEMS_PER_SECOND = 20.0
# A commodity GPU instance. The dominant term is time, not the rate.
COMPUTE_HOURLY_USD = 0.60

EMBED_ITEMS_PER_SECOND = 850.0    # measured: 600 queries shortlisted in 0.7s
STUDENT_ITEMS_PER_SECOND = 12000.0  # an MLP over a cached embedding

# One-off costs, amortised over the volume being priced.
INDEX_BUILD_SECONDS = 19.4        # measured: embedding 10,003 training messages
STUDENT_TRAIN_SECONDS = 25.0


@dataclass
class Option:
    name: str
    description: str
    accuracy: float
    coverage: float                 # share answered automatically
    machine_seconds_per_1k: float
    fixed_seconds: float = 0.0
    needs_transformer_at_serving: bool = True

    def machine_cost_per_1k(self, volume_k: float = 1.0) -> float:
        amortised = self.fixed_seconds / max(volume_k, 1e-9)
        return (self.machine_seconds_per_1k + amortised) / 3600.0 * COMPUTE_HOURLY_USD

    def human_cost_per_1k(self) -> float:
        """Only the items the gate sends back."""
        return (1.0 - self.coverage) * HUMAN_COST_PER_1K

    def total_cost_per_1k(self, volume_k: float = 1.0) -> float:
        return self.machine_cost_per_1k(volume_k) + self.human_cost_per_1k()

    def as_dict(self, volume_k: float = 1.0) -> Dict[str, object]:
        return {
            "name": self.name, "description": self.description,
            "accuracy": round(self.accuracy, 4), "coverage": round(self.coverage, 4),
            # Six decimals, not two. The automated options cost fractions of a cent
            # per thousand items, and rounding to cents collapses them all to $0.00 -
            # which then makes every ratio against them meaningless (an early run
            # printed "83,330,000,000x cheaper than humans"). The whole point of the
            # frontier is comparing options that differ by orders of magnitude, so the
            # precision has to survive that range.
            "machine_cost_per_1k": round(self.machine_cost_per_1k(volume_k), 6),
            "human_cost_per_1k": round(self.human_cost_per_1k(), 4),
            "total_cost_per_1k": round(self.total_cost_per_1k(volume_k), 6),
            "needs_transformer_at_serving": self.needs_transformer_at_serving,
        }


def human_baseline() -> Option:
    return Option(
        name="human annotation",
        description="every item labelled by a person",
        accuracy=1.0,          # the definition of the gold standard, by construction
        coverage=0.0,          # nothing automated, so everything is human cost
        machine_seconds_per_1k=0.0,
        needs_transformer_at_serving=False)


def build_options(retriever_accuracy: float, lm_accuracy: float,
                  fusion_accuracy: float, student_accuracy: float,
                  gate_coverage: float, gate_precision: float) -> List[Option]:
    """Every stack that was actually measured, priced on the same basis."""
    return [
        human_baseline(),
        Option("retriever only",
               "kNN vote over MiniLM embeddings; no language model",
               retriever_accuracy, 1.0, 1000 / EMBED_ITEMS_PER_SECOND,
               INDEX_BUILD_SECONDS),
        Option("retriever + LM re-rank",
               "shortlist then score 8 candidates with flan-t5-base",
               fusion_accuracy, 1.0,
               1000 / EMBED_ITEMS_PER_SECOND + 1000 / LM_ITEMS_PER_SECOND,
               INDEX_BUILD_SECONDS),
        Option("LM only",
               "flan-t5-base scores the shortlist, retriever ranking discarded",
               lm_accuracy, 1.0, 1000 / LM_ITEMS_PER_SECOND, INDEX_BUILD_SECONDS),
        Option("distilled student",
               "MLP over cached embeddings; no index, no LM at serving",
               student_accuracy, 1.0,
               1000 / EMBED_ITEMS_PER_SECOND + 1000 / STUDENT_ITEMS_PER_SECOND,
               INDEX_BUILD_SECONDS + STUDENT_TRAIN_SECONDS,
               needs_transformer_at_serving=True),
        Option("retriever + abstention gate",
               f"auto-accept above the threshold ({gate_precision:.0%} precision), "
               f"route the rest to a person",
               gate_precision, gate_coverage, 1000 / EMBED_ITEMS_PER_SECOND,
               INDEX_BUILD_SECONDS),
    ]


def frontier(options: Sequence[Option], volume_k: float = 100.0) -> Dict[str, object]:
    """The Pareto frontier: options not dominated on both cost and quality.

    'Quality' for a gated option is precision on what it auto-accepts, weighted by the
    fact that reviewed items are correct by definition. An option dominated on both axes
    should never be chosen, and naming those explicitly is more useful than a ranked list
    that implies every row is a live candidate.
    """
    priced = []
    for option in options:
        effective = option.accuracy * option.coverage + 1.0 * (1 - option.coverage)
        priced.append({
            **option.as_dict(volume_k),
            "effective_accuracy": round(effective, 4),
        })

    frontier_rows, dominated = [], []
    for row in priced:
        is_dominated = any(
            other["total_cost_per_1k"] <= row["total_cost_per_1k"]
            and other["effective_accuracy"] >= row["effective_accuracy"]
            and other["name"] != row["name"]
            and (other["total_cost_per_1k"] < row["total_cost_per_1k"]
                 or other["effective_accuracy"] > row["effective_accuracy"])
            for other in priced)
        (dominated if is_dominated else frontier_rows).append(row)

    human = next(r for r in priced if r["name"] == "human annotation")
    # The human option is priced entirely on human review, so its machine cost is zero
    # and the cheapest automated option is what it should be compared against.
    automated = [r for r in priced if r["name"] != "human annotation"]
    cheapest = min(automated, key=lambda r: r["total_cost_per_1k"])

    return {
        "assumptions": {
            "human_seconds_per_item": HUMAN_SECONDS_PER_ITEM,
            "human_hourly_usd": HUMAN_HOURLY_USD,
            "human_cost_per_1k_usd": round(HUMAN_COST_PER_1K, 2),
            "compute_hourly_usd": COMPUTE_HOURLY_USD,
            "lm_items_per_second": LM_ITEMS_PER_SECOND,
            "embed_items_per_second": EMBED_ITEMS_PER_SECOND,
            "volume_priced_thousands": volume_k,
        },
        "options": sorted(priced, key=lambda r: r["total_cost_per_1k"]),
        "frontier": sorted(frontier_rows, key=lambda r: r["total_cost_per_1k"]),
        "dominated": [r["name"] for r in dominated],
        "human_cost_per_1k": human["total_cost_per_1k"],
        "cheapest_automated": {
            "name": cheapest["name"],
            "cost_per_1k": cheapest["total_cost_per_1k"],
            "human_cost_per_1k": human["total_cost_per_1k"],
            "ratio_vs_human": round(
                human["total_cost_per_1k"] / max(cheapest["total_cost_per_1k"], 1e-12), 1),
            "accuracy_given_up": round(
                human["effective_accuracy"] - cheapest["effective_accuracy"], 4),
            "note": ("The ratio is enormous because the automated options cost a "
                     "fraction of a cent per 1,000 items - compute is essentially free "
                     "at this scale. The real trade is therefore ACCURACY, not cost, "
                     "which is why the gated option exists: it buys back precision with "
                     "human review and is the only automated option that is not "
                     "dominated on quality."),
        },
        "at_volume": {
            f"{int(volume_k)}k_items": {
                r["name"]: round(r["total_cost_per_1k"] * volume_k, 2) for r in priced}
        },
    }
