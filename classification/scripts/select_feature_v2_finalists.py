"""Seal V2 finalists and the V1/V2 acceptance decision using validation only."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.context_v2 import FEATURE_GENERATION, build_experiment_context_v2


SINGLE_MODELS = {
    "logistic_regression", "random_forest", "extra_trees", "xgboost",
    "lightgbm", "catboost", "bagging", "stacking", "mlp", "tabnet",
    "tabular_transformer", "modern_hopfield",
}
PRACTICAL_MODELS = {"logistic_regression", "lightgbm"}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best(rows: list[dict[str, str]], models: set[str]) -> dict[str, str]:
    candidates = [
        row for row in rows
        if row["model"] in models
        and row["status"] == "completed"
        and row["degenerate"].lower() != "true"
    ]
    if not candidates:
        raise RuntimeError(f"no valid candidates for {sorted(models)}")
    return max(candidates, key=lambda row: (float(row["pr_auc"]), float(row["mcc"])))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = PROJECT_ROOT
    report_dir = root / "reports" / "generations" / "feature_v2"
    artifact_dir = root / "artifacts" / "generations" / "feature_v2"
    ledger_path = report_dir / "EXPERIMENT_RESULTS.csv"
    rules = json.loads((report_dir / "ACCEPTANCE_RULE.json").read_text(encoding="utf-8"))
    v2 = [
        row for row in _read(ledger_path)
        if row["feature_set"].startswith(FEATURE_GENERATION)
        and row["sample_size"] == "60000"
    ]
    completed = [row for row in v2 if row["status"] == "completed"]
    if len(completed) != 42 or len(v2) != 42:
        raise RuntimeError(f"expected exactly 42 completed V2 rows, found {len(completed)}/{len(v2)}")
    model_counts: dict[str, int] = {}
    for row in completed:
        model_counts[row["model"]] = model_counts.get(row["model"], 0) + 1
    if len(model_counts) != 14 or set(model_counts.values()) != {3}:
        raise RuntimeError(f"V2 model/variant matrix is incomplete: {model_counts}")

    context = build_experiment_context_v2(
        root=root, max_fit_rows=60_000, seed=20260809, reuse_preprocessor=True
    )
    prediction_dir = artifact_dir / "predictions"
    for row in completed:
        archive = np.load(prediction_dir / f"{row['experiment_id']}_validation.npz")
        if not np.array_equal(archive["source_indices"], context.validation_indices):
            raise RuntimeError(f"validation-index mismatch: {row['experiment_id']}")
        if set(archive["source_indices"]).intersection(context.splits.test):
            raise RuntimeError(f"test row present in validation archive: {row['experiment_id']}")

    v2_predictive = _best(completed, {"hybrid_ml_dl"})
    v2_single = _best(completed, SINGLE_MODELS)
    v2_practical = _best(completed, PRACTICAL_MODELS)

    baseline_rows = [
        row for row in _read(root / "reports" / "EXPERIMENT_RESULTS.csv")
        if row["status"] == "completed"
        and row["sample_size"] == "60000"
        and "_v" in row["experiment_id"]
    ]
    v1_predictive = _best(baseline_rows, {"hybrid_ml_dl"})
    deltas = {
        metric: float(v2_predictive[metric]) - float(v1_predictive[metric])
        for metric in ("pr_auc", "mcc", "balanced_accuracy")
    }
    limits = rules["accept_v2_if"]
    gate_checks = {
        "pr_auc": deltas["pr_auc"] >= float(limits["pr_auc_delta_minimum"]),
        "mcc": deltas["mcc"] >= float(limits["mcc_delta_minimum"]),
        "balanced_accuracy": deltas["balanced_accuracy"] >= float(limits["balanced_accuracy_delta_minimum"]),
        "non_degenerate": v2_predictive["degenerate"].lower() != "true",
        "two_prediction_classes": int(float(v2_predictive["unique_prediction_classes"])) >= 2,
    }
    accepted = all(gate_checks.values())

    comparison_rows: list[dict[str, object]] = []
    baseline_by_key = {(row["model"], row["dataset_variant"]): row for row in baseline_rows}
    for row in completed:
        baseline = baseline_by_key.get((row["model"], row["dataset_variant"]))
        comparison_rows.append({
            "model": row["model"],
            "dataset_variant": row["dataset_variant"],
            "v1_experiment_id": baseline["experiment_id"] if baseline else "",
            "v2_experiment_id": row["experiment_id"],
            "v1_pr_auc": float(baseline["pr_auc"]) if baseline else "",
            "v2_pr_auc": float(row["pr_auc"]),
            "delta_pr_auc": float(row["pr_auc"]) - float(baseline["pr_auc"]) if baseline else "",
            "v1_mcc": float(baseline["mcc"]) if baseline else "",
            "v2_mcc": float(row["mcc"]),
            "delta_mcc": float(row["mcc"]) - float(baseline["mcc"]) if baseline else "",
            "v1_balanced_accuracy": float(baseline["balanced_accuracy"]) if baseline else "",
            "v2_balanced_accuracy": float(row["balanced_accuracy"]),
            "delta_balanced_accuracy": float(row["balanced_accuracy"]) - float(baseline["balanced_accuracy"]) if baseline else "",
        })
    comparison_path = report_dir / "VALIDATION_V1_V2_COMPARISON.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    selection = {
        "generation_id": "feature_v2_onehot_numeric_equal_60k",
        "selection_data": "validation_only",
        "selection_rule": "PR-AUC primary, MCC tie-break; practical restricted to logistic/lightgbm",
        "best_predictive": v2_predictive["experiment_id"],
        "best_single_model": v2_single["experiment_id"],
        "best_practical": v2_practical["experiment_id"],
        "v1_best_predictive": v1_predictive["experiment_id"],
        "validation_delta_v2_minus_v1": deltas,
        "acceptance_checks": gate_checks,
        "v2_accepted_by_pretest_rule": accepted,
        "recommended_generation_before_test": "V2" if accepted else "V1",
        "source_sha256": context.prepared.source_sha256,
        "split_index_sha256": context.splits.hashes(),
        "v2_ledger_sha256": _sha256(ledger_path),
        "acceptance_rule_sha256": _sha256(report_dir / "ACCEPTANCE_RULE.json"),
        "validation_comparison_sha256": _sha256(comparison_path),
        "test_accessed_during_selection": False,
    }
    selection_path = artifact_dir / "PRETEST_SELECTION.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        if existing != selection:
            raise RuntimeError("sealed V2 pretest selection differs from current selection")
    else:
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        with selection_path.open("x", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(selection, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
