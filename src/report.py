"""Run the whole pipeline end to end and write every measured number.

Run:  python3 -m src.report [--test-size 1500]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.fetch import fetch                                   # noqa: E402
from src.abstain import coverage_at_precision, operating_curve, sweep_targets  # noqa: E402
from src.ablation import run_ablation                          # noqa: E402
from src.calibrate import calibrate                            # noqa: E402
from src.cost import build_options, frontier                   # noqa: E402
from src.distill import distil, learning_curve                 # noqa: E402
from src.retriever import Embedder, Retriever, recall_at_n, top1_accuracy  # noqa: E402
from src.scorer import LMScorer                                # noqa: E402

OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=int, default=1500)
    ap.add_argument("--few-shot-subset", type=int, default=300)
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()

    started = time.time()
    data = fetch()
    classes = sorted(data["labels"])
    train_texts = [t for t, _ in data["train"]]
    train_labels = [l for _, l in data["train"]]
    test = data["test"][:args.test_size]
    test_texts = [t for t, _ in test]
    test_gold = [l for _, l in test]

    print(f"BANKING77: {len(train_texts):,} train, {len(test_texts):,} test, "
          f"{len(classes)} intents")

    print("\n1. retrieval")
    embedder = Embedder()
    retriever = Retriever(embedder, train_texts, train_labels)
    shortlists = retriever.shortlist(test_texts, top_n=8)
    ceiling = recall_at_n(shortlists, test_gold)
    retriever_accuracy = top1_accuracy(shortlists, test_gold)
    print(f"   recall@1 {ceiling['recall_at_1']}  recall@8 {ceiling['recall_at_8']} "
          f"(the ceiling on everything downstream)")
    print(f"   retriever top-1 accuracy: {retriever_accuracy}")

    print("\n2. four-arm ablation")
    scorer = LMScorer()
    ablation = run_ablation(retriever, scorer, test_texts, test_gold,
                            few_shot_subset=args.few_shot_subset)
    for arm in ablation["arms"]:
        tag = "" if arm["evaluated_on_full_test_set"] else "  (subset)"
        print(f"   arm {arm['arm']}  {arm['accuracy']:.4f}  "
              f"{arm['lm_calls']:>5} LM calls  {arm['seconds']:>5.0f}s{tag}")
    print(f"   -> {ablation['verdict']}")

    print("\n3. calibration")
    retriever_scores = [s.scores for s in shortlists]
    labels_list = [s.labels for s in shortlists]
    calibration = calibrate(retriever_scores, labels_list, test_gold)
    print(f"   T = {calibration.temperature:.4f}  ({calibration.direction})")
    print(f"   ECE {calibration.ece_before:.4f} -> {calibration.ece_after:.4f}")

    print("\n4. abstention gate")
    from src.calibrate import softmax
    width = max(len(s) for s in retriever_scores)
    matrix = np.full((len(retriever_scores), width), -np.inf)
    for i, row in enumerate(retriever_scores):
        matrix[i, :len(row)] = row
    calibrated = softmax(matrix, calibration.temperature)
    confidence = calibrated.max(axis=1)
    predicted = calibrated.argmax(axis=1)
    correct = np.array([labels_list[i][predicted[i]] == test_gold[i]
                        for i in range(len(test_gold))], dtype=float)

    gate = coverage_at_precision(confidence, correct, target_precision=0.95)
    print(f"   at 95% precision: threshold {gate.threshold:.4f}, "
          f"coverage {gate.coverage:.4f}, {gate.n_review:,} of {len(test_gold):,} reviewed")
    targets = sweep_targets(confidence, correct)

    print("\n5. distillation")
    train_X = retriever.matrix
    test_X = embedder.encode(test_texts)
    # The teacher labels the TRAINING texts. Using gold labels here would be ordinary
    # supervised learning, not distillation - and would not reflect production, where
    # gold labels for new data do not exist.
    teacher_train = [s.labels[0] for s in retriever.shortlist(train_texts, top_n=1)]
    teacher_test = [s.labels[0] for s in shortlists]
    distillation = distil(train_X, teacher_train, test_X, test_gold, teacher_test,
                          classes)
    print(f"   teacher {distillation['teacher_accuracy']:.4f}")
    print(f"   neural student {distillation['neural_student']['student_accuracy']:.4f} "
          f"({distillation['neural_student']['parameters']:,} params)")
    print(f"   linear student {distillation['linear_student']['student_accuracy']:.4f} "
          f"({distillation['linear_student']['parameters']:,} params)")
    print(f"   -> {distillation['verdict']}")

    curve = learning_curve(train_X, teacher_train, test_X, test_gold, classes)
    print(f"   learning curve still improving at 100%: {curve[-1].get('still_improving')}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    distillation["models"]["neural"].save(ARTIFACTS / "student.pt")
    np.savez_compressed(ARTIFACTS / "index.npz", matrix=retriever.matrix,
                        labels=np.array(train_labels), texts=np.array(train_texts,
                                                                     dtype=object))
    (ARTIFACTS / "serving.json").write_text(json.dumps({
        "classes": classes,
        "temperature": calibration.temperature,
        "abstain_threshold": gate.threshold,
        "target_precision": gate.target_precision,
        "expected_coverage": gate.coverage,
        "expected_precision": gate.precision,
        "embed_model": embedder.name,
        "student_parameters": distillation["neural_student"]["parameters"],
    }, indent=2))

    print("\n6. cost frontier")
    best_student = max(distillation["neural_student"]["student_accuracy"],
                       distillation["linear_student"]["student_accuracy"])
    options = build_options(
        retriever_accuracy=retriever_accuracy,
        lm_accuracy=[a for a in ablation["arms"] if a["arm"] == "B"][0]["accuracy"],
        fusion_accuracy=[a for a in ablation["arms"] if a["arm"] == "C"][0]["accuracy"],
        student_accuracy=best_student,
        gate_coverage=gate.coverage, gate_precision=gate.precision)
    costs = frontier(options, volume_k=100.0)
    for row in costs["options"]:
        cost = row["total_cost_per_1k"]
        shown = f"${cost:>10.4f}" if cost < 1 else f"${cost:>10.2f}"
        print(f"   {row['name']:<30} {shown}/1k  acc {row['accuracy']:.4f}  "
              f"coverage {row['coverage']:.2f}  effective {row['effective_accuracy']:.4f}")
    c = costs["cheapest_automated"]
    print(f"   cheapest automated: {c['name']} at ${c['cost_per_1k']:.4f}/1k "
          f"vs ${c['human_cost_per_1k']:.2f}/1k for humans ({c['ratio_vs_human']:,.0f}x)")
    print(f"   dominated: {costs['dominated']}")

    tests = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no",
                            "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True)
    test_line = (tests.stdout.strip().splitlines() or ["not run"])[-1]

    report = {
        "project": "AI Annotation & Served Inference",
        "role": "CV2 / Data Science + Business Analytics - applied ML and MLOps",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data": data["stats"],
        "retrieval": {"ceiling": ceiling, "top1_accuracy": retriever_accuracy,
                      "model": embedder.name},
        "ablation": {k: v for k, v in ablation.items()},
        "calibration": calibration.as_dict(),
        "abstention": {"chosen": gate.as_dict(), "targets": targets,
                       "operating_curve": operating_curve(confidence, correct)},
        "distillation": {k: v for k, v in distillation.items() if k != "models"},
        "learning_curve": curve,
        "cost": costs,
        "tests": test_line,
        "seconds": round(time.time() - started, 1),
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n   tests: {test_line}")
    print(f"   wrote outputs/results.json ({report['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
