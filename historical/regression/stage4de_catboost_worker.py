"""Run exactly one Stage 4D–E CatBoost fit and save its checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4
import stage4de_catboost_utils as de


def _sample_digest(root: Path, config: dict) -> str:
    if config["kind"] == "full_train":
        return s4.sha256_file(root / "artifacts/splits/train_row_ids.csv")
    filename = {
        "feature_confirmation": "stage4_feature_confirmation_sample.csv",
        "final_selection": "stage4_final_selection_sample.csv",
    }[config["sample_name"]]
    return s4.sha256_file(root / "artifacts/splits/stage4" / filename)


def _native_save(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    model.save_model(str(temporary), format="cbm")
    os.replace(temporary, path)


def run_one(root: Path, config_path: Path) -> dict:
    config = de.read_json(config_path)
    checkpoint = root / config["checkpoint_path"]
    model_path = root / config["model_path"]
    prediction_path = root / config["prediction_path"] if config.get("prediction_path") else None
    native_path = root / config["native_model_path"] if config.get("native_model_path") else None
    started_at = s4.utc_now()
    try:
        if config["kind"] == "validation":
            train_ids, validation_ids = de.sample_ids(root, config["sample_name"])
            fitted = de.fit_validation(
                root, config["feature_pack"], config["sensitive_mode"], config["target_mode"],
                config["parameters"], train_ids, validation_ids, config.get("early_stopping_rounds"),
            )
            predictions = de.prediction_frame(fitted, config["experiment_id"], config["sensitive_mode"], config["target_mode"])
            s4.atomic_write_csv(predictions, prediction_path)
            metrics = fitted["metrics"]
            validation_rows = len(validation_ids)
            prediction_seconds = fitted["prediction_seconds"]
            best_iteration = fitted["best_iteration_zero_based"]
        elif config["kind"] == "full_train":
            train_ids = pd.read_csv(root / "artifacts/splits/train_row_ids.csv", dtype={"row_id": "int64"})["row_id"].to_numpy(dtype=np.int64)
            validation_ids = np.asarray([], dtype=np.int64)
            fitted = de.fit_full_train(root, config["feature_pack"], config["sensitive_mode"], config["target_mode"], config["parameters"], train_ids)
            reference_ids = np.sort(train_ids)[:500]
            raw = fitted["raw_columns"]
            X_reference = s4.read_training_rows(de.source_path(root, config["sensitive_mode"]), reference_ids, raw).loc[reference_ids, raw].copy()
            reference_prediction = s4.inverse_target(fitted["pipeline"].predict(X_reference), config["target_mode"])
            predictions = pd.DataFrame({"row_id": reference_ids, "y_pred": reference_prediction})
            s4.atomic_write_csv(predictions, prediction_path)
            metrics = {}
            validation_rows = 0
            prediction_seconds = 0.0
            best_iteration = int(config["parameters"]["iterations"]) - 1
        else:
            raise ValueError(f"Unknown worker kind: {config['kind']}")

        metadata = {
            "stage": de.STAGE_ID,
            "stage_name": de.STAGE_NAME,
            "version": de.VERSION,
            "experiment_id": config["experiment_id"],
            "fit_id": config["fit_id"],
            "kind": config["kind"],
            "evaluation_stage": config["evaluation_stage"],
            "sample_name": config.get("sample_name", "all_saved_train"),
            "feature_pack_id": config["feature_pack"]["pack_id"],
            "feature_pack": config["feature_pack"],
            "target_mode": config["target_mode"],
            "sensitive_mode": config["sensitive_mode"],
            "parameters": config["parameters"],
            "fixed_iteration_count": int(fitted["tree_count"]),
            "random_seed": de.SEED,
            "execution_mode": de.EXECUTION_MODE,
            "thread_count": de.THREAD_COUNT,
            "training_row_count": len(train_ids),
            "validation_row_count": validation_rows,
            "test_row_count": 0,
            "fit_seconds": fitted["fit_seconds"],
            "prediction_seconds": prediction_seconds,
            "metrics": metrics,
            "feature_count": len(fitted["selected_features"]),
            "categorical_feature_count": len(fitted["categorical_features"]),
            "raw_columns": fitted["raw_columns"],
            "selected_features": fitted["selected_features"],
            "categorical_features": fitted["categorical_features"],
            "sample_digest": _sample_digest(root, config),
            "source_hash_digest": c4.source_digest(root, config["sensitive_mode"]),
            "configuration_digest": s4.configuration_digest(config, length=64),
            "prediction_path": str(prediction_path.relative_to(root)),
            "model_path": str(model_path.relative_to(root)),
            "native_model_path": str(native_path.relative_to(root)) if native_path else None,
            "started_at_utc": started_at,
            "completed_at_utc": s4.utc_now(),
            "warning_text": "",
            "error_text": "",
            "status": "PASS",
        }
        metadata.update(de.bundle_provenance(root, config["sensitive_mode"], config["target_mode"], fitted["raw_columns"]))
        bundle = de.CatBoostModelBundle(fitted["pipeline"], config["target_mode"], metadata)
        s4.atomic_write_joblib(bundle, model_path)
        if native_path:
            _native_save(fitted["pipeline"].named_steps["model"], native_path)
        s4.atomic_write_json(checkpoint, metadata)
        return metadata
    except Exception as exc:
        failure = {
            "stage": de.STAGE_ID,
            "experiment_id": config.get("experiment_id"),
            "fit_id": config.get("fit_id"),
            "kind": config.get("kind"),
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
    print(json.dumps({"fit_id": result["fit_id"], "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
