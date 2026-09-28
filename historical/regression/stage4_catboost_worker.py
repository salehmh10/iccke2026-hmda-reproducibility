"""Run exactly one Stage 4C CatBoost fit and save an atomic checkpoint."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import joblib
import pandas as pd

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4


def run_one(root: Path, config_path: Path) -> dict:
    config = c4.read_json(config_path)
    experiment_id = str(config["experiment_id"])
    kind = str(config["kind"])
    checkpoint = root / config["checkpoint_path"]
    prediction_path = root / config["prediction_path"]
    model_path = root / config["model_path"]
    if kind == "screening":
        train_ids, validation_ids = c4.screening_ids(root)
        sample_path = c4.paths(root)["splits"] / "stage4c_screening_subset.csv"
        use_early_stopping = True
    elif kind == "controlled":
        train_ids, validation_ids = c4.discovery_ids(root)
        sample_path = c4.paths(root)["splits"] / "stage4_discovery_sample.csv"
        use_early_stopping = False
    else:
        raise ValueError(f"Unknown fit kind: {kind}")
    started_at = s4.utc_now()
    try:
        fitted = c4.fit_pipeline(
            root=root,
            pack_name=config["feature_pack"],
            sensitive_mode=config["sensitive_mode"],
            target_mode=config["target_mode"],
            parameters=config["parameters"],
            train_ids=train_ids,
            validation_ids=validation_ids,
            use_early_stopping=use_early_stopping,
        )
        predictions = c4.prediction_frame(fitted, config["target_mode"], config["sensitive_mode"], experiment_id)
        c4.ensure_directories(root)
        s4.atomic_write_csv(predictions, prediction_path)
        bundle = {
            "stage": c4.STAGE_ID,
            "stage_name": c4.STAGE_NAME,
            "experiment_id": experiment_id,
            "feature_pack": config["feature_pack"],
            "sensitive_mode": config["sensitive_mode"],
            "target_mode": config["target_mode"],
            "parameters": config["parameters"],
            "random_seed": c4.SEED,
            "execution_mode": c4.EXECUTION_MODE,
            "thread_count": c4.THREAD_COUNT,
            "pipeline": fitted["pipeline"],
            "raw_columns": fitted["raw_columns"],
            "selected_features": fitted["selected_features"],
            "categorical_features": fitted["categorical_features"],
            "feature_pack_digest": c4.feature_pack_digest(root, config["feature_pack"]),
            "sample_digest": c4.file_digest(sample_path),
            "source_hash": c4.source_digest(root, config["sensitive_mode"]),
            "catboost_version": "1.2.10",
        }
        s4.atomic_write_joblib(bundle, model_path)
        metadata = {
            "stage": c4.STAGE_ID,
            "stage_name": c4.STAGE_NAME,
            "candidate_id": config.get("candidate_id"),
            "experiment_id": experiment_id,
            "kind": kind,
            "feature_pack": config["feature_pack"],
            "sensitive_mode": config["sensitive_mode"],
            "target_mode": config["target_mode"],
            "parameters": config["parameters"],
            "random_seed": c4.SEED,
            "execution_mode": c4.EXECUTION_MODE,
            "thread_count": c4.THREAD_COUNT,
            "train_rows": len(train_ids),
            "validation_rows": len(validation_ids),
            "test_rows": 0,
            "best_iteration_zero_based": fitted["best_iteration_zero_based"],
            "fixed_iteration_count": fitted["tree_count"],
            "fit_seconds": fitted["fit_seconds"],
            "prediction_seconds": fitted["prediction_seconds"],
            "metrics": fitted["metrics"],
            "training_metrics": fitted["training_metrics"],
            "feature_count": len(fitted["selected_features"]),
            "categorical_feature_count": len(fitted["categorical_features"]),
            "prediction_path": str(prediction_path.relative_to(root)),
            "model_path": str(model_path.relative_to(root)),
            "feature_pack_digest": c4.feature_pack_digest(root, config["feature_pack"]),
            "sample_digest": c4.file_digest(sample_path),
            "source_hash_digest": c4.source_digest(root, config["sensitive_mode"]),
            "configuration_digest": s4.configuration_digest(config, length=64),
            "predictions_finite": True,
            "started_at_utc": started_at,
            "completed_at_utc": s4.utc_now(),
            "warning_text": "",
            "error_text": "",
            "status": "PASS",
        }
        s4.atomic_write_json(checkpoint, metadata)
        return metadata
    except Exception as exc:
        failure = {
            "stage": c4.STAGE_ID,
            "candidate_id": config.get("candidate_id"),
            "experiment_id": experiment_id,
            "kind": kind,
            "started_at_utc": started_at,
            "completed_at_utc": s4.utc_now(),
            "configuration_digest": s4.configuration_digest(config, length=64),
            "warning_text": "",
            "error_text": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "status": "FAIL",
        }
        s4.atomic_write_json(checkpoint, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    result = run_one(root, Path(args.config).resolve())
    print(json.dumps({"experiment_id": result["experiment_id"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
