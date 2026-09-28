"""Run exactly one bounded LightGBM fit and save an atomic checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import stage4_boosting_utils as s4
import stage4_lightgbm_utils as l4


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def _atomic_joblib(value, path: Path) -> None:
    s4.atomic_write_joblib(value, path, compress=3)
    loaded = joblib.load(path)
    if not isinstance(loaded, l4.LightGBMModelBundle):
        raise TypeError("Saved LightGBM bundle did not reload as the expected type.")


def run_one(root: Path, config_path: Path) -> dict:
    config = l4.read_json(config_path)
    required = {
        "stage_id", "experiment_id", "job_type", "sample_name", "feature_pack",
        "selected_proposals", "target_mode", "sensitive_mode", "parameters",
        "early_stopping_rounds", "model_path", "native_model_path",
        "prediction_path", "checkpoint_path",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Worker config is missing fields: {missing}")
    checkpoint_path = root / config["checkpoint_path"]
    existing = l4.validate_checkpoint(root, checkpoint_path)
    if existing["valid"]:
        print(json.dumps({"status": "REUSED", "checkpoint_path": str(checkpoint_path)}))
        return existing["checkpoint"]
    spec = l4.feature_pack_spec(root, config["feature_pack"], config["selected_proposals"])
    if config["job_type"] == "validation":
        train_ids, validation_ids = l4.sample_ids(root, config["sample_name"])
        result = l4.fit_validation(
            root, spec, config["sensitive_mode"], config["target_mode"],
            config["parameters"], train_ids, validation_ids,
            config["early_stopping_rounds"],
        )
        predictions = l4.prediction_frame(result, config)
    elif config["job_type"] == "full_train":
        train_ids = l4.all_train_ids(root)
        result = l4.fit_full_train(
            root, spec, config["sensitive_mode"], config["target_mode"],
            config["parameters"], train_ids,
        )
        predictions = pd.DataFrame({
            "row_id": result["reference_ids"],
            "y_pred": result["reference_prediction"],
            "stage_id": config["stage_id"],
            "experiment_id": config["experiment_id"],
            "sensitive_mode": config["sensitive_mode"],
            "target_mode": config["target_mode"],
        })
    else:
        raise ValueError(f"Unknown job type: {config['job_type']}")
    metadata = l4.bundle_metadata(root, config, spec, result)
    bundle = l4.LightGBMModelBundle(result["pipeline"], config["target_mode"], metadata)
    model_path = root / config["model_path"]
    native_path = root / config["native_model_path"]
    prediction_path = root / config["prediction_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_joblib(bundle, model_path)
    l4.atomic_native_save(result["pipeline"].named_steps["model"], native_path)
    s4.atomic_write_csv(predictions, prediction_path)
    checkpoint = {
        "stage_id": config["stage_id"],
        "stage_name": l4.STAGE_NAMES[config["stage_id"]],
        "experiment_id": config["experiment_id"],
        "job_type": config["job_type"],
        "feature_pack": spec["pack_id"],
        "feature_engineer_version": spec["version"],
        "selected_proposals": list(config["selected_proposals"]),
        "target_mode": config["target_mode"],
        "sensitive_mode": config["sensitive_mode"],
        "parameters": config["parameters"],
        "fixed_or_best_iteration": int(result["best_iteration"]),
        "random_seed": l4.SEED,
        "thread_count": l4.THREAD_COUNT,
        "sample_name": config["sample_name"],
        "sample_digest": l4.sample_digest(root, config["sample_name"]),
        "feature_pack_digest": l4.feature_pack_digest(spec),
        "source_hash_digest": s4.sha256_file(l4.source_path(root, config["sensitive_mode"])),
        "training_row_count": int(len(result["train_ids"])),
        "validation_row_count": int(len(result.get("validation_ids", ()))),
        "test_row_count": 0,
        "fit_time_seconds": float(result["fit_seconds"]),
        "prediction_time_seconds": float(result.get("prediction_seconds", 0.0)),
        "metrics": result.get("metrics", {}),
        "model_path": _relative(root, model_path),
        "native_model_path": _relative(root, native_path),
        "prediction_path": _relative(root, prediction_path),
        "model_sha256": s4.sha256_file(model_path),
        "native_model_sha256": s4.sha256_file(native_path),
        "prediction_sha256": s4.sha256_file(prediction_path),
        "warnings": [],
        "errors": [],
        "status": "PASS",
        "created_at_utc": s4.utc_now(),
    }
    s4.atomic_write_json(checkpoint_path, checkpoint)
    validation = l4.validate_checkpoint(root, checkpoint_path)
    if not validation["valid"]:
        raise IOError(f"Saved checkpoint failed validation: {validation}")
    print(json.dumps(checkpoint, indent=2))
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config_path = Path(args.config)
    config_path = config_path if config_path.is_absolute() else root / config_path
    try:
        run_one(root, config_path)
        return 0
    except Exception as exc:
        error = {"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()}
        try:
            config = l4.read_json(config_path)
            checkpoint_path = root / config.get("checkpoint_path", "artifacts/checkpoints/stage4/lightgbm/worker_failure.json")
            s4.atomic_write_json(checkpoint_path.with_name(checkpoint_path.stem + "__failure.json"), error)
        except Exception:
            pass
        print(json.dumps(error, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
