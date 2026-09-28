"""Validation-only soft blending of qualified ML and DL predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import optimize_threshold
from src.training.context import build_experiment_context
from src.training.tracking import (
    append_record,
    build_record,
    cumulative_training_seconds,
    experiment_contract_fingerprint,
    experiment_exists,
    write_experiment_provenance,
)


ML_MODELS = {"logistic_regression", "random_forest", "extra_trees", "xgboost", "lightgbm", "catboost", "bagging", "stacking"}
DL_MODELS = {"mlp", "tabnet", "tabular_transformer", "modern_hopfield"}


def _qualified_rows(ledger: Path, variant: str) -> list[dict[str, str]]:
    with ledger.open("r", encoding="utf-8", newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["status"] == "completed"
            and row["dataset_variant"] == variant
            and row["sample_size"] == "60000"
            and "_v" in row["experiment_id"]
        ]


def _best(rows: list[dict[str, str]], allowed: set[str]) -> dict[str, str]:
    candidates = [row for row in rows if row["model"] in allowed and row["degenerate"].lower() != "true"]
    if not candidates:
        raise RuntimeError(f"no qualified candidate among {sorted(allowed)}")
    return max(candidates, key=lambda row: (float(row["pr_auc"]), float(row["mcc"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    root = args.root.resolve()
    ledger = root / "reports" / "EXPERIMENT_RESULTS.csv"
    context = build_experiment_context(root=root, max_fit_rows=60000, seed=args.seed)
    code_files = [
        "src/evaluation/metrics.py",
        "src/training/context.py",
        "src/training/tracking.py",
        "scripts/train_hybrid.py",
    ]
    for variant in ("original_weighted", "oversampled", "undersampled"):
        rows = _qualified_rows(ledger, variant)
        ml = _best(rows, ML_MODELS)
        dl = _best(rows, DL_MODELS)
        base_rows = min(int(ml["sample_size"]), int(dl["sample_size"]))
        hybrid_contract = {
            **context.contract_metadata,
            "ml_base_experiment_id": ml["experiment_id"],
            "dl_base_experiment_id": dl["experiment_id"],
        }
        contract_fingerprint = experiment_contract_fingerprint(root, hybrid_contract, code_files)
        experiment_id = (
            f"hybrid_ml_dl_{variant}_s{args.seed}_n{base_rows}_v{contract_fingerprint[:10]}"
        )
        if experiment_exists(ledger, experiment_id):
            continue
        ml_data = np.load(root / "artifacts" / "predictions" / f"{ml['experiment_id']}_validation.npz")
        dl_data = np.load(root / "artifacts" / "predictions" / f"{dl['experiment_id']}_validation.npz")
        for key in ("source_indices", "y_true"):
            if not np.array_equal(ml_data[key], dl_data[key]):
                raise RuntimeError(f"hybrid base prediction mismatch for {key}")
        y_true = ml_data["y_true"]
        ml_probability = ml_data["denial_probability"]
        dl_probability = dl_data["denial_probability"]
        started = time.perf_counter()
        best: tuple[float, float, object, np.ndarray] | None = None
        for ml_weight in np.linspace(0.0, 1.0, 41):
            probability = ml_weight * ml_probability + (1.0 - ml_weight) * dl_probability
            threshold = optimize_threshold(y_true, probability)
            score = float(threshold.metrics["pr_auc"] + 0.1 * threshold.metrics["mcc"])
            if best is None or score > best[0]:
                best = (score, float(ml_weight), threshold, probability)
        assert best is not None
        runtime = time.perf_counter() - started
        _, ml_weight, threshold_result, probability = best
        artifact = {
            "experiment_id": experiment_id,
            "strategy": "validation_selected_soft_blend",
            "ml_experiment_id": ml["experiment_id"],
            "dl_experiment_id": dl["experiment_id"],
            "ml_weight": ml_weight,
            "dl_weight": 1.0 - ml_weight,
            "threshold": threshold_result.threshold,
            "test_predictions_used_for_fitting": False,
        }
        artifact_path = root / "artifacts" / "models" / f"{experiment_id}.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        np.savez_compressed(
            root / "artifacts" / "predictions" / f"{experiment_id}_validation.npz",
            source_indices=ml_data["source_indices"],
            y_true=y_true,
            denial_probability=probability,
            threshold=np.array([threshold_result.threshold]),
        )
        write_experiment_provenance(
            root,
            experiment_id,
            contract_fingerprint,
            hybrid_contract,
            code_files,
        )
        record = build_record(
            root=root,
            experiment_id=experiment_id,
            dataset_variant=variant,
            sample_size=base_rows,
            feature_set="primary_financial_v1_base_probabilities",
            model="hybrid_ml_dl",
            hyperparameters=artifact,
            seed=args.seed,
            class_weighting="inherited_from_bases",
            sampler="inherited_from_bases",
            train_runtime_seconds=runtime,
            threshold=threshold_result.threshold,
            status="completed",
            metrics=threshold_result.metrics,
        )
        append_record(ledger, record)
        print(json.dumps({"experiment_id": experiment_id, "ml": ml["experiment_id"], "dl": dl["experiment_id"], "ml_weight": ml_weight, "pr_auc": threshold_result.metrics["pr_auc"], "mcc": threshold_result.metrics["mcc"]}))
    print(json.dumps({"cumulative_training_seconds": cumulative_training_seconds(ledger)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
