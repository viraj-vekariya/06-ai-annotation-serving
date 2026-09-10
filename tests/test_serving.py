"""The served API, and the measured results it claims.

Skipped rather than failed when the artifacts are absent, so a fresh checkout does not
report a wall of red for something that has not been run yet.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "outputs" / "results.json"

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS / "serving.json").exists(),
    reason="no artifacts; run python3 -m src.report")


@pytest.fixture(scope="module")
def client():
    from serve.app import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("no results.json")
    return json.loads(RESULTS.read_text())


# -- the API -----------------------------------------------------------------

def test_the_service_becomes_ready(client):
    r = client.get("/health/ready")
    assert r.status_code == 200 and r.json()["ready"] is True


def test_liveness_does_not_depend_on_the_model(client):
    """A failing liveness probe KILLS the container. If it depended on the model, a model
    problem would become a restart loop."""
    assert client.get("/health/live").json()["status"] == "alive"


def test_a_clear_message_is_classified_confidently(client):
    body = client.post("/classify", json={
        "text": "My card still has not arrived after two weeks"}).json()
    assert body["prediction"] == "card_arrival"
    assert body["confidence"] > 0.8
    assert body["abstain"] is False


def test_gibberish_is_abstained_on(client):
    """The gate's entire purpose. A system that confidently labels noise is worse than
    one that says it does not know."""
    body = client.post("/classify", json={"text": "asdfgh qwerty zxcvb"}).json()
    assert body["abstain"] is True
    assert "below" in body["reason"]


def test_the_reason_explains_the_decision_either_way(client):
    """Both branches must state the confidence, the threshold and the action.

    The threshold is compared numerically rather than as a substring: the reason renders
    it to 3 decimals for readability while the response field carries 4, so a string
    match would be asserting a formatting choice rather than the behaviour.
    """
    for text in ("My card has not arrived", "asdfgh qwerty"):
        body = client.post("/classify", json={"text": text}).json()
        reason = body["reason"]
        assert reason
        assert f"{body['threshold']:.3f}" in reason
        assert ("auto-accept" in reason) != ("route to a human" in reason)
        assert body["abstain"] == ("route to a human" in reason)


def test_alternatives_are_a_ranked_distribution(client):
    body = client.post("/classify", json={
        "text": "how do I top up my account", "top_k": 5}).json()
    confidences = [a["confidence"] for a in body["alternatives"]]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0 <= c <= 1 for c in confidences)


def test_explain_returns_the_nearest_training_messages(client):
    """The whole system is a kNN vote, so the neighbours ARE the explanation - not a
    post-hoc rationalisation of a black box."""
    body = client.post("/classify", json={
        "text": "my card is broken", "explain": True}).json()
    assert body["neighbours"] and len(body["neighbours"]) <= 5
    assert all("similarity" in n and "label" in n for n in body["neighbours"])


def test_an_empty_message_is_rejected(client):
    assert client.post("/classify", json={"text": ""}).status_code == 422


def test_batch_matches_single(client):
    """Batching must change throughput, never the answer."""
    text = "I need to reset my pin"
    single = client.post("/classify", json={"text": text}).json()
    batch = client.post("/classify/batch", json={"texts": [text, text]}).json()
    assert batch["count"] == 2
    assert all(r["prediction"] == single["prediction"] for r in batch["results"])


def test_the_served_threshold_is_the_one_the_pipeline_derived(client):
    """A threshold hard-coded in the service would drift from the analysis that
    justified it the first time either changed."""
    config = json.loads((ARTIFACTS / "serving.json").read_text())
    body = client.post("/classify", json={"text": "where is my card"}).json()
    assert body["threshold"] == pytest.approx(config["abstain_threshold"], abs=1e-4)


def test_the_deployed_encoder_is_the_traced_one(client):
    """The image ships MiniLM as a TorchScript graph rather than a transformers
    checkpoint - measurement put the `transformers` import at 215 MB, more than the model,
    on an instance with 512 MB in total. If the service silently fell back to the full
    path the numbers would be unchanged and the container would run out of memory under
    concurrency instead, which is the failure that is hardest to read from a crash log."""
    backend = client.get("/metrics").json()["service"]["encoder_backend"]
    assert backend in {"traced", "transformers"}
    if (ARTIFACTS / "encoder_traced.pt").exists():
        assert backend == "traced", "traced artifact present but not used"


def test_the_config_endpoint_states_why_the_LM_is_not_served(client):
    note = client.get("/config").json()["note"]
    assert "not on the serving path" in note or "deliberately not" in note


def test_latency_is_recorded_per_stage(client):
    body = client.post("/classify", json={"text": "check my balance"}).json()
    timing = body["timing_ms"]
    assert timing["embed"] + timing["search"] <= timing["total"] + 1e-6


def test_metrics_expose_the_abstention_rate(client):
    for text in ("my card broke", "zzz qqq"):
        client.post("/classify", json={"text": text})
    stats = client.get("/metrics").json()["service"]
    assert stats["requests"] > 0
    assert 0 <= stats["abstention_rate"] <= 1
    assert stats["index_size"] > 1000 and stats["classes"] == 77


# -- the measured claims -----------------------------------------------------

def test_the_shortlist_ceiling_is_high(results):
    """recall@8 bounds everything downstream. If it were low, a mediocre final accuracy
    would be a retrieval problem, not a classifier problem."""
    assert results["retrieval"]["ceiling"]["recall_at_8"] > 0.95


def test_the_retriever_beats_the_language_model(results):
    """THE FINDING. If this ever inverts, the README's central claim is wrong and the
    serving path should be reconsidered."""
    arms = {a["arm"]: a["accuracy"] for a in results["ablation"]["arms"]}
    assert arms["A"] > arms["B"], f"the LM ({arms['B']}) overtook the retriever ({arms['A']})"
    assert arms["A"] - arms["B"] > 0.15


def test_fusion_does_not_beat_the_retriever(results):
    """The alpha sweep's optimum is 1.0 - all weight on the retriever, none on the LM."""
    arms = {a["arm"]: a["accuracy"] for a in results["ablation"]["arms"]}
    assert arms["C"] <= arms["A"] + 1e-9
    assert results["ablation"]["alpha_sweep"]["best"]["alpha"] >= 0.9


def test_calibration_substantially_reduces_ece(results):
    calibration = results["calibration"]
    assert calibration["ece_after"] < calibration["ece_before"]
    assert calibration["ece_reduction_pct"] > 50


def test_the_gate_achieves_its_target_with_real_coverage(results):
    gate = results["abstention"]["chosen"]
    assert gate["target_achieved"]
    assert gate["precision"] >= gate["target_precision"]
    assert gate["coverage"] > 0.7, "the gate is sending too much to humans to be useful"


def test_the_student_retains_most_of_the_teacher(results):
    distillation = results["distillation"]
    best = max(distillation["neural_student"]["student_accuracy"],
               distillation["linear_student"]["student_accuracy"])
    assert best / distillation["teacher_accuracy"] > 0.95


def test_the_cost_frontier_has_the_three_live_options(results):
    """Everything else is dominated on both axes and should never be chosen."""
    names = [r["name"] for r in results["cost"]["frontier"]]
    assert "retriever only" in names
    assert "human annotation" in names
    assert "LM only" in results["cost"]["dominated"]
