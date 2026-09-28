"""Resumable Stage 3 Fold checkpoints and cache validation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage3_tree_utils import (
    BASE_CATEGORICAL_FEATURES,
    SENSITIVE_CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    canonical_json,
    configuration_digest,
    deterministic_experiment_id,
    evaluate_regression_predictions,
    feature_lists,
    read_training_rows,
    sha256_file,
    write_json,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "artifacts/results/stage3"
FOLD_RESULTS = RESULTS / "folds"
PREDICTIONS = ROOT / "artifacts/predictions/tree"
FOLD_PREDICTIONS = PREDICTIONS / "folds"
MODELS = ROOT / "artifacts/models/tree"
REPORTS = ROOT / "artifacts/reports"
MANIFESTS = ROOT / "artifacts/manifests"
SPLITS = ROOT / "artifacts/splits"
WITH_SOURCE = ROOT / "data/regression_with_sensitive_features.csv"
WITHOUT_SOURCE = ROOT / "data/regression_without_sensitive_features.csv"
CONFIG_PATH = RESULTS / "selected_tree_configurations.json"
TRAIN_IDS_PATH = SPLITS / "train_row_ids.csv"
TEST_IDS_PATH = SPLITS / "test_row_ids.csv"
CV_PATH = SPLITS / "cv_fold_assignments.csv"
DEV_PATH = SPLITS / "prompt2_development_sample.csv"
FEATURE_PACK_PATH = ROOT / "artifacts/features/tree/stage3_tree_feature_packs.json"
PROGRESS_PATH = REPORTS / "stage3_progress.json"
RECOVERY_VERSION = "stage3_recovery_v1_20260714"


for directory in (RESULTS, FOLD_RESULTS, PREDICTIONS, FOLD_PREDICTIONS, MODELS, REPORTS, MANIFESTS):
    directory.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_save_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def full_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def selected_configurations() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def source_path(sensitive_mode: str) -> Path:
    if sensitive_mode == "without_sensitive":
        return WITHOUT_SOURCE
    if sensitive_mode == "with_sensitive":
        return WITH_SOURCE
    raise ValueError(f"Unknown sensitive mode: {sensitive_mode}")


def split_hashes() -> dict[str, str]:
    return {
        "train_row_ids": sha256_file(TRAIN_IDS_PATH),
        "test_row_ids": sha256_file(TEST_IDS_PATH),
        "cv_fold_assignments": sha256_file(CV_PATH),
    }


def split_digest() -> str:
    return full_digest(split_hashes())


def feature_pack_digest(configuration: dict[str, Any], sensitive_mode: str) -> str:
    return full_digest({
        "feature_pack": configuration["feature_pack"],
        "lists": feature_lists(configuration["feature_pack"], sensitive_mode),
        "feature_pack_artifact_sha256": sha256_file(FEATURE_PACK_PATH),
    })


def experiment_digest(
    configuration: dict[str, Any], sensitive_mode: str, fold: int | None, purpose: str
) -> str:
    return full_digest({
        "recovery_version": RECOVERY_VERSION,
        "purpose": purpose,
        "model_name": configuration["model_name"],
        "configuration": configuration,
        "sensitive_mode": sensitive_mode,
        "fold": fold,
        "feature_pack_digest": feature_pack_digest(configuration, sensitive_mode),
    })


def fold_paths(model_name: str, sensitive_mode: str, fold: int) -> tuple[Path, Path]:
    stem = f"{model_name}__{sensitive_mode}__fold-{int(fold)}"
    return FOLD_RESULTS / f"{stem}.json", FOLD_PREDICTIONS / f"{stem}.csv"


def load_ids() -> tuple[np.ndarray, set[int], pd.DataFrame]:
    train_ids = pd.read_csv(TRAIN_IDS_PATH, dtype={"row_id": "int64"})["row_id"].to_numpy(dtype=np.int64)
    test_ids = set(pd.read_csv(TEST_IDS_PATH, dtype={"row_id": "int64"})["row_id"].astype(int))
    folds = pd.read_csv(CV_PATH, dtype={"row_id": "int64", "fold": "int64"})
    if len(folds) != len(train_ids) or not folds["row_id"].is_unique or set(folds["row_id"]) != set(train_ids):
        raise AssertionError("Saved CV assignments do not cover the saved Train rows exactly.")
    if set(folds["fold"]) != {0, 1, 2} or set(folds["row_id"]).intersection(test_ids):
        raise AssertionError("Saved CV assignments are invalid or overlap the locked Test Set.")
    return train_ids, test_ids, folds


def load_rows(
    configuration: dict[str, Any], sensitive_mode: str, row_ids: np.ndarray | list[int]
) -> tuple[pd.DataFrame, pd.Series]:
    ids = np.sort(np.asarray(row_ids, dtype=np.int64))
    raw_features = feature_lists(configuration["feature_pack"], sensitive_mode)["raw"]
    usecols = list(dict.fromkeys(raw_features + [TARGET_COLUMN]))
    categorical = list(dict.fromkeys(BASE_CATEGORICAL_FEATURES + SENSITIVE_CATEGORICAL_FEATURES))
    frame = read_training_rows(source_path(sensitive_mode), ids, categorical, usecols=usecols)
    target = pd.to_numeric(frame.pop(TARGET_COLUMN), errors="raise").astype(float)
    return frame[raw_features].copy(deep=False), target


def load_all_targets() -> pd.Series:
    train_ids, _, _ = load_ids()
    frame = read_training_rows(WITHOUT_SOURCE, train_ids, (), usecols=[TARGET_COLUMN])
    return pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(float)


def tail_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    absolute = np.abs(pred - true)
    result: dict[str, Any] = {"p99_absolute_error": float(np.quantile(absolute, 0.99))}
    for threshold in (1000, 5000):
        mask = true >= threshold
        result[f"mae_target_ge_{threshold}"] = float(absolute[mask].mean()) if mask.any() else None
        result[f"rows_target_ge_{threshold}"] = int(mask.sum())
    return result


def checkpoint_metadata(
    configuration: dict[str, Any], sensitive_mode: str, fold: int, purpose: str
) -> dict[str, Any]:
    return {
        "recovery_version": RECOVERY_VERSION,
        "purpose": purpose,
        "model_name": configuration["model_name"],
        "sensitive_mode": sensitive_mode,
        "fold": int(fold),
        "configuration": configuration,
        "configuration_digest": experiment_digest(configuration, sensitive_mode, fold, purpose),
        "feature_pack": configuration["feature_pack"],
        "feature_pack_digest": feature_pack_digest(configuration, sensitive_mode),
        "source_sha256": sha256_file(source_path(sensitive_mode)),
        "split_hashes": split_hashes(),
        "split_digest": split_digest(),
    }


def validate_fold_checkpoint(
    configuration: dict[str, Any], sensitive_mode: str, fold: int, y_all: pd.Series | None = None
) -> tuple[bool, str, dict[str, Any] | None, pd.DataFrame | None]:
    result_path, prediction_path = fold_paths(configuration["model_name"], sensitive_mode, fold)
    if not result_path.exists() or not prediction_path.exists():
        return False, "missing Fold result or prediction", None, None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        expected = checkpoint_metadata(configuration, sensitive_mode, fold, "cv_fold")
        for key in ("configuration_digest", "feature_pack_digest", "source_sha256", "split_digest"):
            if result.get(key) != expected[key]:
                return False, f"{key} mismatch", result, None
        if result.get("status") != "success" or int(result.get("fold", -1)) != int(fold):
            return False, "status or Fold mismatch", result, None
        frame = pd.read_csv(prediction_path, dtype={"row_id": "int64", "fold": "int64"})
        required = {"row_id", "fold", "y_true", "y_pred"}
        if not required.issubset(frame.columns) or not frame["row_id"].is_unique:
            return False, "prediction schema or unique-ID failure", result, frame
        _, test_ids, assignments = load_ids()
        expected_ids = set(assignments.loc[assignments["fold"].eq(fold), "row_id"].astype(int))
        actual_ids = set(frame["row_id"].astype(int))
        if actual_ids != expected_ids or actual_ids.intersection(test_ids):
            return False, "validation IDs do not match the saved Fold", result, frame
        if set(frame["fold"].astype(int)) != {int(fold)}:
            return False, "prediction Fold label mismatch", result, frame
        if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
            return False, "non-finite target or prediction", result, frame
        if y_all is None:
            y_all = load_all_targets()
        expected_y = y_all.loc[frame["row_id"].to_numpy(dtype=np.int64)].to_numpy(dtype=float)
        if not np.allclose(expected_y, frame["y_true"].to_numpy(dtype=float), rtol=0, atol=1e-12):
            return False, "target alignment mismatch", result, frame
        return True, "valid", result, frame
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None, None


def save_fold_checkpoint(
    configuration: dict[str, Any], sensitive_mode: str, fold: int,
    prediction_frame: pd.DataFrame, result: dict[str, Any], purpose: str = "cv_fold"
) -> tuple[Path, Path]:
    result_path, prediction_path = fold_paths(configuration["model_name"], sensitive_mode, fold)
    metadata = checkpoint_metadata(configuration, sensitive_mode, fold, purpose)
    payload = {**metadata, **result, "prediction_path": str(prediction_path.relative_to(ROOT))}
    atomic_save_csv(prediction_frame, prediction_path)
    write_json(result_path, payload)
    return result_path, prediction_path


def aggregate_completed_cv(configurations: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_all = load_all_targets()
    train_ids, test_ids, assignments = load_ids()
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for configuration in configurations:
        model_name = configuration["model_name"]
        for sensitive_mode in ("without_sensitive", "with_sensitive"):
            pieces: list[pd.DataFrame] = []
            results: list[dict[str, Any]] = []
            for fold in (0, 1, 2):
                valid, reason, result, frame = validate_fold_checkpoint(
                    configuration, sensitive_mode, fold, y_all=y_all
                )
                if not valid or result is None or frame is None:
                    raise AssertionError(f"Invalid {model_name} {sensitive_mode} Fold {fold}: {reason}")
                pieces.append(frame)
                results.append(result)
                fold_rows.append({key: value for key, value in result.items() if key not in {"configuration", "split_hashes"}})
            oof = pd.concat(pieces, ignore_index=True).sort_values("row_id").reset_index(drop=True)
            if len(oof) != len(train_ids) or not oof["row_id"].is_unique or set(oof["row_id"]) != set(train_ids):
                raise AssertionError(f"Incomplete OOF coverage for {model_name} {sensitive_mode}.")
            if set(oof["row_id"]).intersection(test_ids) or not np.isfinite(oof["y_pred"]).all():
                raise AssertionError("OOF contains Test rows or non-finite predictions.")
            metrics = evaluate_regression_predictions(oof["y_true"], oof["y_pred"])
            tails = tail_metrics(oof["y_true"].to_numpy(), oof["y_pred"].to_numpy())
            experiment_id = deterministic_experiment_id(
                model_name, sensitive_mode, configuration["target_mode"], "oof_summary", None,
                configuration, configuration["feature_pack"],
            )
            oof["absolute_error"] = np.abs(oof["y_pred"] - oof["y_true"])
            oof["signed_error"] = oof["y_pred"] - oof["y_true"]
            oof["model_name"] = model_name
            oof["sensitive_mode"] = sensitive_mode
            oof["target_mode"] = configuration["target_mode"]
            oof["feature_pack"] = configuration["feature_pack"]
            oof["experiment_id"] = experiment_id
            oof_path = PREDICTIONS / f"{model_name}__{sensitive_mode}__oof.csv"
            atomic_save_csv(oof, oof_path)
            summary_rows.append({
                "experiment_id": experiment_id,
                "model_name": model_name,
                "sensitive_mode": sensitive_mode,
                "target_mode": configuration["target_mode"],
                "feature_pack": configuration["feature_pack"],
                "configuration_json": canonical_json(configuration),
                **{key: value for key, value in metrics.items() if key != "metric_warnings"},
                **tails,
                "fold_mae_mean": float(np.mean([row["mae"] for row in results])),
                "fold_mae_std": float(np.std([row["mae"] for row in results], ddof=0)),
                "total_fit_time_seconds": float(np.nansum([
                    np.nan if row.get("fit_time_seconds") is None else row.get("fit_time_seconds") for row in results
                ])) if any(row.get("fit_time_seconds") is not None for row in results) else np.nan,
                "total_prediction_time_seconds": float(np.nansum([
                    np.nan if row.get("prediction_time_seconds") is None else row.get("prediction_time_seconds") for row in results
                ])) if any(row.get("prediction_time_seconds") is not None for row in results) else np.nan,
                "oof_rows": len(oof),
                "oof_path": str(oof_path.relative_to(ROOT)),
                "status": "success",
            })
    folds_frame = pd.DataFrame(fold_rows)
    summaries = pd.DataFrame(summary_rows)
    atomic_save_csv(folds_frame, RESULTS / "cv_fold_results.csv")
    atomic_save_csv(summaries, RESULTS / "cv_oof_summary.csv")
    write_json(MANIFESTS / "stage3_cv_manifest.json", {
        "recovery_version": RECOVERY_VERSION,
        "per_experiment_digests": {
            f"{c['model_name']}__{mode}": experiment_digest(c, mode, None, "oof_summary")
            for c in configurations for mode in ("without_sensitive", "with_sensitive")
        },
        "fold_fits": len(folds_frame),
        "oof_experiments": len(summaries),
        "saved_folds_reused": [0, 1, 2],
        "test_predictions": 0,
    })
    return folds_frame, summaries


def update_progress(**updates: Any) -> dict[str, Any]:
    state = json.loads(PROGRESS_PATH.read_text(encoding="utf-8")) if PROGRESS_PATH.exists() else {
        "recovery_version": RECOVERY_VERSION,
        "recovery_started_at_utc": utc_now(),
        "targeted_execution_attempts": 0,
        "full_notebook_execution_attempts": 0,
        "repair_iterations": 0,
    }
    state.update(updates)
    state["updated_at_utc"] = utc_now()
    write_json(PROGRESS_PATH, state)
    return state
