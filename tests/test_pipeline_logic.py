"""Fusion, calibration, abstention and cost: the maths, with no models loaded.

These run in milliseconds and need no downloads. The model-dependent behaviour is
measured by src/report.py and asserted in tests/test_serving.py; this file pins the logic
that the measurements are computed WITH.
"""

import math

import numpy as np
import pytest

from src.abstain import coverage_at_precision, operating_curve, sweep_targets
from src.calibrate import (expected_calibration_error, fit_temperature,
                           reliability_table, softmax)
from src.cost import HUMAN_COST_PER_1K, Option, build_options, frontier, human_baseline
from src.fusion import fuse, standardise, sweep_alpha


# -- fusion ------------------------------------------------------------------

def test_standardisation_puts_both_signals_on_one_scale():
    """The load-bearing step. The retriever produces vote shares and the LM produces a
    softmax over log-likelihoods; adding them raw lets whichever has the larger variance
    dominate regardless of which is more informative."""
    z = standardise([1.0, 2.0, 3.0, 4.0])
    assert abs(z.mean()) < 1e-9
    assert abs(z.std() - 1.0) < 1e-9


def test_standardising_identical_values_does_not_divide_by_zero():
    """All candidates equal means no information. Dividing by ~0 would manufacture
    enormous differences out of floating-point noise."""
    assert np.allclose(standardise([0.5, 0.5, 0.5]), 0.0)


def test_alpha_one_is_retriever_only_and_alpha_zero_is_lm_only():
    labels = ["a", "b", "c"]
    retriever = [0.7, 0.2, 0.1]
    lm = [0.1, 0.2, 0.7]
    assert fuse(labels, retriever, lm, alpha=1.0).top()[0] == "a"
    assert fuse(labels, retriever, lm, alpha=0.0).top()[0] == "c"


def test_fused_probabilities_are_a_distribution():
    p = fuse(["a", "b", "c"], [0.6, 0.3, 0.1], [0.2, 0.5, 0.3], 0.5).probabilities
    assert abs(sum(p) - 1.0) < 1e-9
    assert all(0 <= v <= 1 for v in p)


def test_the_alpha_sweep_reports_both_endpoints():
    """The endpoints ARE the individual systems, which is what lets the sweep answer
    'does fusion beat either component?' rather than assert it."""
    labels = [["a", "b"]] * 20
    retriever = [[0.9, 0.1]] * 20
    lm = [[0.1, 0.9]] * 20
    gold = ["a"] * 20
    result = sweep_alpha(labels, retriever, lm, gold)
    assert result["retriever_only"]["accuracy"] == 1.0
    assert result["lm_only"]["accuracy"] == 0.0
    assert len(result["sweep"]) == 11


# -- calibration -------------------------------------------------------------

def test_temperature_scaling_never_changes_a_prediction():
    """Dividing every logit by the same positive number preserves the argmax exactly.
    That property is what makes it safe to apply after the fact - accuracy cannot move,
    only confidence."""
    scores = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
    for t in (0.2, 1.0, 5.0):
        assert np.array_equal(softmax(scores, t).argmax(axis=1), scores.argmax(axis=1))


def test_softmax_is_overflow_safe():
    """Raw exponentials of large log-likelihoods overflow to inf; subtracting the max
    first is mathematically exact and finite."""
    p = softmax(np.array([[900.0, 800.0, 700.0]]), 1.0)
    assert np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-9


def test_a_higher_temperature_softens_confidence():
    scores = np.array([[5.0, 1.0, 0.5]])
    assert softmax(scores, 5.0).max() < softmax(scores, 1.0).max()


def test_a_lower_temperature_sharpens_confidence():
    scores = np.array([[5.0, 1.0, 0.5]])
    assert softmax(scores, 0.2).max() > softmax(scores, 1.0).max()


def test_perfect_calibration_has_zero_ece():
    probabilities = np.array([[0.9, 0.1]] * 100)
    correct = np.array([1.0] * 90 + [0.0] * 10)
    assert expected_calibration_error(probabilities, correct) < 0.02


def test_gross_overconfidence_has_high_ece():
    probabilities = np.array([[0.99, 0.01]] * 100)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    assert expected_calibration_error(probabilities, correct) > 0.4


def test_fitting_temperature_on_overconfident_scores_returns_t_above_one():
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 6, size=(600, 5))          # very peaked -> over-confident
    correct_index = rng.integers(0, 5, size=600)      # but the labels are random
    assert fit_temperature(scores, correct_index) > 1.0


def test_reliability_bins_only_contain_populated_rows():
    probabilities = np.array([[0.95, 0.05]] * 40)
    correct = np.array([1.0] * 38 + [0.0] * 2)
    rows = reliability_table(probabilities, correct)
    assert rows and all(r["n"] > 0 for r in rows)


# -- abstention --------------------------------------------------------------

def test_the_gate_meets_its_precision_target():
    rng = np.random.default_rng(11)
    confidence = rng.uniform(0, 1, 3000)
    # Correctness rises with confidence, which is what a calibrated model looks like.
    correct = (rng.uniform(0, 1, 3000) < confidence).astype(float)
    gate = coverage_at_precision(confidence, correct, target_precision=0.9)
    assert gate.achieved and gate.precision >= 0.9


def test_the_gate_picks_the_LOWEST_qualifying_threshold():
    """Any higher threshold also meets the target but auto-accepts fewer items, costing
    more human review for the same guarantee. Maximum coverage is the objective."""
    rng = np.random.default_rng(12)
    confidence = rng.uniform(0, 1, 3000)
    correct = (rng.uniform(0, 1, 3000) < confidence).astype(float)
    gate = coverage_at_precision(confidence, correct, target_precision=0.85)
    lower = confidence >= (gate.threshold - 0.05)
    assert correct[lower].mean() < 0.85 or gate.threshold < 0.02


def test_a_stricter_target_costs_coverage():
    """The whole trade, and the reason the sweep is reported rather than one number."""
    rng = np.random.default_rng(13)
    confidence = rng.uniform(0, 1, 4000)
    correct = (rng.uniform(0, 1, 4000) < confidence).astype(float)
    loose = coverage_at_precision(confidence, correct, 0.80)
    strict = coverage_at_precision(confidence, correct, 0.97)
    assert strict.coverage < loose.coverage


def test_an_unachievable_target_is_reported_honestly():
    """Rather than returning a threshold that silently does not do what it claims."""
    confidence = np.linspace(0.1, 0.9, 200)
    correct = np.zeros(200)                       # never right
    gate = coverage_at_precision(confidence, correct, target_precision=0.95)
    assert not gate.achieved


def test_sweeping_targets_returns_one_row_each():
    rng = np.random.default_rng(14)
    confidence = rng.uniform(0, 1, 1500)
    correct = (rng.uniform(0, 1, 1500) < confidence).astype(float)
    assert len(sweep_targets(confidence, correct)) == 4


def test_the_operating_curve_is_monotone_in_coverage():
    rng = np.random.default_rng(15)
    confidence = rng.uniform(0, 1, 1200)
    correct = (rng.uniform(0, 1, 1200) < confidence).astype(float)
    coverages = [r["coverage"] for r in operating_curve(confidence, correct)]
    assert coverages == sorted(coverages, reverse=True)


# -- cost --------------------------------------------------------------------

def test_the_human_baseline_is_priced_entirely_on_human_time():
    human = human_baseline()
    assert human.machine_cost_per_1k() == 0.0
    assert human.human_cost_per_1k() == pytest.approx(HUMAN_COST_PER_1K)


def test_a_gated_option_pays_for_the_items_it_sends_back():
    """An automated option that routes 60% of traffic to a person is not 99% cheaper,
    and pricing must reflect that."""
    gated = Option("gated", "", accuracy=0.95, coverage=0.6,
                   machine_seconds_per_1k=1.0)
    assert gated.human_cost_per_1k() == pytest.approx(0.4 * HUMAN_COST_PER_1K)


def test_fixed_costs_amortise_over_volume():
    option = Option("x", "", 0.9, 1.0, machine_seconds_per_1k=1.0, fixed_seconds=3600.0)
    assert option.total_cost_per_1k(1.0) > option.total_cost_per_1k(1000.0)


def test_dominated_options_are_identified():
    """An option worse on BOTH cost and quality should never be chosen, and naming it is
    more useful than a ranked list implying every row is a live candidate."""
    options = build_options(retriever_accuracy=0.89, lm_accuracy=0.60,
                            fusion_accuracy=0.89, student_accuracy=0.88,
                            gate_coverage=0.87, gate_precision=0.95)
    result = frontier(options)
    assert "LM only" in result["dominated"], result["dominated"]
    assert any(r["name"] == "retriever only" for r in result["frontier"])


def test_the_cost_of_every_automated_option_is_far_below_human():
    options = build_options(0.89, 0.60, 0.89, 0.88, 0.87, 0.95)
    result = frontier(options)
    human = result["human_cost_per_1k"]
    for row in result["options"]:
        if row["name"] not in ("human annotation", "retriever + abstention gate"):
            assert row["total_cost_per_1k"] < human / 100


def test_costs_keep_sub_cent_precision():
    """REGRESSION: rounding to 2 decimals collapsed every automated option to $0.00 and
    made the ratio against humans print as 83,330,000,000x."""
    options = build_options(0.89, 0.60, 0.89, 0.88, 0.87, 0.95)
    result = frontier(options)
    retriever = next(r for r in result["options"] if r["name"] == "retriever only")
    assert retriever["total_cost_per_1k"] > 0, "sub-cent costs were rounded away"
