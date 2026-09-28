"""Validation-only ML+DL blends for the isolated V2 generation."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from scripts.train_feature_v2_ml import _total_measured_fit_seconds
from src.evaluation.metrics import optimize_threshold
from src.training.context_v2 import FEATURE_GENERATION, build_experiment_context_v2
from src.training.tracking import (
    append_record,
    build_record,
    cumulative_training_seconds,
    experiment_contract_fingerprint,
    experiment_exists,
    write_experiment_provenance,
)


ML_MODELS = {
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "bagging",
    "stacking",
}
DL_MODELS = {"mlp", "tabnet", "tabular_transformer", "modern_hopfield"}


def _rows(ledger: Path, variant: str) -> list[dict[str, str]]:
    with ledger.open("r", encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["status"] == "completed"
            and row["dataset_variant"] == variant
            and row["sample_size"] == "60000"
            and row["feature_set"] == FEATURE_GENERATION
            and row["model"] != "hybrid_ml_dl"
        ]


def _best(rows: list[dict[str, str]], allowed: set[str]) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row["model"] in allowed and row["degenerate"].lower() != "true"
    ]
    if not candidates:
        raise RuntimeError(f"no qualified V2 candidate among {sorted(allowed)}")
    return max(candidates, key=lambda row: (float(row["pr_auc"]), float(row["mcc"])))


def main() -> int:
    root = PROJECT_ROOT
    report_dir = root / "reports" / "generations" / "feature_v2"
    artifact_root = root / "artifacts" / "generations" / "feature_v2"
    model_dir = artifact_root / "models"
    prediction_dir = artifact_root / "predictions"
    ledger = report_dir / "EXPERIMENT_RESULTS.csv"
    context = build_experiment_context_v2(
        root=root, max_fit_rows=60_000, seed=20260809, reuse_preprocessor=True
    )
    code_files = [
        "src/features/advanced.py",
        "src/evaluation/metrics.py",
        "src/training/context_v2.py",
        "src/training/tracking.py",
        "scripts/train_feature_v2_hybrid.py",
    ]
    completed = 0
    for variant in ("original_weighted", "oversampled", "undersampled"):
        qualified = _rows(ledger, variant)
        ml = _best(qualified, ML_MODELS)
        dl = _best(qualified, DL_MODELS)
        contract = {
            **context.contract_metadata,
            "variant": variant,
            "ml_base_experiment_id": ml["experiment_id"],
            "dl_base_experiment_id": dl["experiment_id"],
        }
        fingerprint = experiment_contract_fingerprint(root, contract, code_files)
        experiment_id = (
            f"v2_hybrid_ml_dl_{variant}_s20260809_n60000_v{fingerprint[:10]}"
        )
        artifact_path = model_dir / f"{experiment_id}.json"
        prediction_path = prediction_dir / f"{experiment_id}_validation.npz"
        provenance_path = model_dir / f"{experiment_id}.provenance.json"
        if experiment_exists(ledger, experiment_id):
            if not all(path.is_file() for path in (artifact_path, prediction_path, provenance_path)):
                raise RuntimeError("recorded V2 hybrid is missing an output")
            continue
        if _total_measured_fit_seconds(root, ledger) >= 43_200.0:
            raise RuntimeError("hard cumulative training budget reached")
        ml_data = np.load(prediction_dir / f"{ml['experiment_id']}_validation.npz")
        dl_data = np.load(prediction_dir / f"{dl['experiment_id']}_validation.npz")
        for key in ("source_indices", "y_true"):
            if not np.array_equal(ml_data[key], dl_data[key]):
                raise RuntimeError(f"V2 hybrid base mismatch: {key}")
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
        _, ml_weight, threshold, probability = best
        artifact = {
            "experiment_id": experiment_id,
            "feature_generation": FEATURE_GENERATION,
            "strategy": "validation_selected_soft_blend",
            "ml_experiment_id": ml["experiment_id"],
            "dl_experiment_id": dl["experiment_id"],
            "ml_weight": ml_weight,
            "dl_weight": 1.0 - ml_weight,
            "threshold": threshold.threshold,
            "test_predictions_used_for_fitting": False,
        }
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            prediction_path,
            source_indices=ml_data["source_indices"],
            y_true=y_true,
            denial_probability=probability,
            threshold=np.array([threshold.threshold]),
        )
        write_experiment_provenance(
            root,
            experiment_id,
            fingerprint,
            contract,
            code_files,
            artifact_dir=model_dir,
        )
        append_record(
            ledger,
            build_record(
                root=root,
                experiment_id=experiment_id,
                dataset_variant=variant,
                sample_size=60_000,
                feature_set=f"{FEATURE_GENERATION}_base_probabilities",
                model="hybrid_ml_dl",
                hyperparameters=artifact,
                seed=20260809,
                class_weighting="inherited_from_bases",
                sampler="inherited_from_bases",
                train_runtime_seconds=runtime,
                threshold=threshold.threshold,
                status="completed",
                metrics=threshold.metrics,
            ),
        )
        completed += 1
        print(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "ml": ml["experiment_id"],
                    "dl": dl["experiment_id"],
                    "ml_weight": ml_weight,
                    "pr_auc": threshold.metrics["pr_auc"],
                    "mcc": threshold.metrics["mcc"],
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "completed": completed,
                "v2_ledger_fit_seconds": cumulative_training_seconds(ledger),
                "total_measured_fit_seconds": _total_measured_fit_seconds(root, ledger),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
