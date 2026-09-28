"""Prompt 5A one-time IID evaluation and artifact-only reporting.

The two original IID files are opened only inside the guarded access commands.
No function in this module fits a model or a preprocessor.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Frozen Prompt 4C bundles were serialized with top-level module names while
# their implementations live in this project directory.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from nbclient import NotebookClient

try:
    from .prompt4_metrics import compute_regression_metrics, provisional_acceptance, quantile_membership
    from .prompt4c_bundles import (
        FinalGlobalBundle,
        FinalPrimaryBundle,
        duplicate_safe_target_deciles,
        mape_details,
        wape_percent,
    )
except ImportError:
    from prompt4_metrics import compute_regression_metrics, provisional_acceptance, quantile_membership
    from prompt4c_bundles import (
        FinalGlobalBundle,
        FinalPrimaryBundle,
        duplicate_safe_target_deciles,
        mape_details,
        wape_percent,
    )

# Make joblib's historical top-level module reference resolve to the same
# class objects imported above.
sys.modules.setdefault("prompt4c_bundles", sys.modules[FinalPrimaryBundle.__module__])


AUTHORIZATION_ID = "regression_v2_prompt5a_one_time_iid_evaluation"
STATUS_PASS = "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS"
FREEZE_SHA = "8f2b8e4fda80056770b916b7859ad0f6f89f2948236320e6815e082fadca35c1"
PRIMARY_SHA = "5349ab15fd1c8182ef539f435cc4f065e71ee7da047de09a4dc271c9097e0c08"
GLOBAL_SHA = "6f61a0be1fc90d2331b08dada63a453f6c5410d5783f46f1e0cbcbf425f75597"
PRIMARY_MODEL_ID = "final_primary_stage3_500k"
GLOBAL_MODEL_ID = "final_global_500k"
PERMITTED_MODELS = (PRIMARY_MODEL_ID, GLOBAL_MODEL_ID)
BLOCKA_ID = "nf_global2_oldraw_direct_cap25"
TARGET = "loan_amount_000s"
EXPECTED_ROWS = 75_000
SEED = 42
BOOTSTRAP_RESAMPLES = 500

REPORTS = Path("outputs/reports")
TMP = Path("outputs/tmp/prompt5a")
PREDICTIONS = Path("outputs/predictions/prompt5a/iid")
POST_IID = Path("outputs/data/post_iid")
FIGURES = Path("outputs/figures/prompt5a")
NOTEBOOK = Path("notebooks/05A_ONE_TIME_IID_EVALUATION_AND_ERROR_ANALYSIS.ipynb")
FREEZE = REPORTS / "FINAL_PRE_IID_FREEZE.json"
PRIMARY_BUNDLE = Path("outputs/models/final_pre_iid/primary_stage3/bundle.joblib")
GLOBAL_BUNDLE = Path("outputs/models/final_pre_iid/global_comparator/bundle.joblib")
ORIGINAL_FEATURES = Path("outputs/data/iid_holdout_features.parquet")
ORIGINAL_TARGETS = Path("outputs/data/iid_holdout_targets.parquet")
MODEL_SNAPSHOT = POST_IID / "iid_model_features_snapshot.parquet"
FAIRNESS_SNAPSHOT = POST_IID / "iid_fairness_audit_snapshot.parquet"
TARGET_SNAPSHOT = POST_IID / "iid_target_snapshot.parquet"
EVALUATION_FRAME = POST_IID / "iid_evaluation_frame.parquet"
PRIMARY_PREDICTION = PREDICTIONS / f"{PRIMARY_MODEL_ID}.parquet"
GLOBAL_PREDICTION = PREDICTIONS / f"{GLOBAL_MODEL_ID}.parquet"
LEDGER = REPORTS / "prompt5a_iid_access_ledger.json"
LOCK = REPORTS / "prompt5a_prediction_lock.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_path(value: str | Path | None = None) -> Path:
    return Path(value or Path(__file__).resolve().parents[1]).resolve()


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def row_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def read_json(root: Path, relative: Path) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def atomic_json(root: Path, relative: Path, payload: dict[str, Any]) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, destination)
    return destination


def atomic_csv(root: Path, relative: Path, frame: pd.DataFrame) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary)
    os.replace(temporary, destination)
    return destination


def atomic_parquet(root: Path, relative: Path, frame: pd.DataFrame) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp{destination.suffix}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    check = pd.read_parquet(temporary, columns=[str(frame.columns[0])])
    if len(check) != len(frame):
        raise RuntimeError(f"Parquet reload failed: {relative.as_posix()}")
    os.replace(temporary, destination)
    return destination


def append_event(ledger: dict[str, Any], event: str, **evidence: Any) -> None:
    ledger.setdefault("events", []).append({"event": event, "timestamp_utc": utc_now(), **evidence})
    ledger["updated_at_utc"] = utc_now()


def load_ledger(root: Path) -> dict[str, Any]:
    return read_json(root, LEDGER)


def save_ledger(root: Path, ledger: dict[str, Any]) -> None:
    atomic_json(root, LEDGER, ledger)


def schema_details(path: Path) -> dict[str, Any]:
    table = pq.read_schema(path)
    return {
        "columns": [field.name for field in table],
        "types": {field.name: str(field.type) for field in table},
        "nullable": {field.name: bool(field.nullable) for field in table},
    }


def assert_exact_model_set(model_ids: Iterable[str]) -> None:
    values = tuple(model_ids)
    if values != PERMITTED_MODELS or len(values) != 2:
        raise PermissionError(f"Prompt 5A permits exactly {PERMITTED_MODELS}, not {values}.")


def prohibit_training_operation(operation: str) -> None:
    if operation.lower() in {"fit", "fit_transform", "partial_fit", "calibrate", "tune", "search"}:
        raise PermissionError("BLOCKED_POST_IID_FIT_ATTEMPT")


def assert_target_access_allowed(ledger: dict[str, Any], lock_valid: bool) -> None:
    if ledger["feature_successful_content_reads"] != 1 or not lock_valid:
        raise PermissionError("IID target access requires a valid prediction lock.")
    if ledger["target_successful_content_reads"] != 0:
        raise PermissionError("The original IID target read allowance is exhausted.")


def assert_prediction_allowed(ledger: dict[str, Any]) -> None:
    if ledger["target_successful_content_reads"] != 0:
        raise PermissionError("No prediction is authorized after IID target access.")
    if ledger["prediction_models"] != 0:
        raise PermissionError("The frozen IID predictions already exist.")


def assign_development_bands(y_true: Any, cutpoints: dict[str, float]) -> np.ndarray:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    ordered = np.array([float(cutpoints[f"q{i:02d}"]) for i in range(10, 100, 10)], dtype=np.float64)
    if y.size == 0 or not np.isfinite(y).all() or not np.all(np.diff(ordered) >= 0.0):
        raise ValueError("Frozen-band inputs are invalid.")
    return np.searchsorted(ordered, y, side="left").astype(np.int16) + 1


def metric_row(y_true: Any, prediction: Any) -> dict[str, Any]:
    result = compute_regression_metrics(y_true, prediction)
    result.update(mape_details(y_true, prediction))
    result["wape_percent"] = wape_percent(y_true, prediction)
    return result


def group_metric_row(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = prediction - y_true
    absolute = np.abs(error)
    mape = mape_details(y_true, prediction)
    return {
        "n": int(y_true.size),
        "target_min": float(np.min(y_true)),
        "target_max": float(np.max(y_true)),
        "mean_target": float(np.mean(y_true)),
        "median_target": float(np.median(y_true)),
        "mean_prediction": float(np.mean(prediction)),
        "median_prediction": float(np.median(prediction)),
        "mae": float(np.mean(absolute)),
        "mape_percent": mape["mape_percent"],
        "mape_invalid_nonpositive_rows": mape["mape_invalid_nonpositive_rows"],
        "mape_valid_rows": mape["mape_valid_rows"],
        "mape_valid_coverage": mape["mape_valid_coverage"],
        "wape_percent": wape_percent(y_true, prediction),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_signed_error": float(np.mean(error)),
        "underprediction_rate": float(np.mean(error < 0.0)),
    }


def six_condition_rows(primary: dict[str, Any], global_metrics: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = provisional_acceptance(primary, global_metrics)
    specs = [
        ("C1", "Primary overall MAE improves over Global", "mae", "<", float(global_metrics["mae"])),
        ("C2", "Primary Top-decile MAE improves by at least 3%", "top_decile_mae", "<=", float(global_metrics["top_decile_mae"]) * 0.97),
        ("C3", "Primary Bottom-90 MAE worsens by no more than 0.25%", "bottom_90_mae", "<=", float(global_metrics["bottom_90_mae"]) * 1.0025),
        ("C4", "Primary RMSE worsens by no more than 0.25%", "rmse", "<=", float(global_metrics["rmse"]) * 1.0025),
        ("C5", "Primary Top-decile signed error is closer to zero", "top_decile_signed_error", "abs(primary) < abs(global)", abs(float(global_metrics["top_decile_signed_error"]))),
        ("C6", "Primary Top-decile underprediction rate decreases", "top_decile_underprediction_rate", "<", float(global_metrics["top_decile_underprediction_rate"])),
    ]
    keys = [
        "complete_mae_lower",
        "top_decile_mae_improves_3pct",
        "bottom_90_mae_worsens_at_most_0_25pct",
        "rmse_worsens_at_most_0_25pct",
        "top_decile_signed_error_closer_to_zero",
        "top_decile_underprediction_rate_decreases",
    ]
    rows = []
    for spec, key in zip(specs, keys):
        cid, description, field, operator, threshold = spec
        rows.append({
            "condition": cid,
            "description": description,
            "metric": field,
            "operator": operator,
            "threshold": threshold,
            "primary_value": float(primary[field]),
            "global_value": float(global_metrics[field]),
            "status": "PASS" if result[key] else "FAIL",
        })
    summary = {
        "status": "PASS" if result["conditions_passed"] == 6 else "PARTIAL" if result["conditions_passed"] else "FAIL",
        "conditions_passed": int(result["conditions_passed"]),
        "conditions_total": 6,
        "conditions": rows,
        "model_selection_effect": "none; the frozen Primary remains unchanged",
    }
    return rows, summary


def _bootstrap_value(metric: str, y: np.ndarray, p: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        y, p = y[mask], p[mask]
    error = p - y
    if metric.endswith("MAE"):
        return float(np.mean(np.abs(error)))
    if metric == "RMSE":
        return float(np.sqrt(np.mean(error**2)))
    if metric == "MAPE":
        valid = y > 0.0
        return float(100.0 * np.mean(np.abs(error[valid]) / y[valid]))
    if metric == "WAPE":
        return float(100.0 * np.sum(np.abs(error)) / np.sum(y))
    raise ValueError(metric)


def paired_bootstrap(y: Any, primary: Any, global_prediction: Any, *, n_resamples: int = BOOTSTRAP_RESAMPLES, seed: int = SEED) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(primary, dtype=np.float64).reshape(-1)
    g = np.asarray(global_prediction, dtype=np.float64).reshape(-1)
    if not (target.shape == p.shape == g.shape) or not np.isfinite(np.column_stack([target, p, g])).all():
        raise ValueError("Bootstrap inputs are invalid.")
    top10 = quantile_membership(target, 0.90)
    top05 = quantile_membership(target, 0.95)
    metric_masks = {
        "MAE": None,
        "RMSE": None,
        "MAPE": None,
        "WAPE": None,
        "Bottom-90 MAE": ~top10,
        "Top-decile MAE": top10,
        "Top-5% MAE": top05,
    }
    observed = {name: _bootstrap_value(name, target, p, mask) - _bootstrap_value(name, target, g, mask) for name, mask in metric_masks.items()}
    samples = {name: np.empty(n_resamples, dtype=np.float64) for name in metric_masks}
    rng = np.random.default_rng(seed)
    for index in range(n_resamples):
        chosen = rng.integers(0, target.size, size=target.size)
        yy, pp, gg = target[chosen], p[chosen], g[chosen]
        for name, base_mask in metric_masks.items():
            mask = None if base_mask is None else base_mask[chosen]
            samples[name][index] = _bootstrap_value(name, yy, pp, mask) - _bootstrap_value(name, yy, gg, mask)
    rows = []
    for name, values in samples.items():
        rows.append({
            "metric": name,
            "observed_difference_primary_minus_global": observed[name],
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_median": float(np.median(values)),
            "percentile_2_5": float(np.quantile(values, 0.025)),
            "percentile_97_5": float(np.quantile(values, 0.975)),
            "fraction_primary_lower": float(np.mean(values < 0.0)),
            "n_resamples": int(n_resamples),
            "seed": int(seed),
        })
    return pd.DataFrame(rows), samples


def validate_prediction_lock(root: Path) -> dict[str, Any]:
    if not (root / LOCK).is_file():
        raise RuntimeError("Prediction lock is missing.")
    lock = read_json(root, LOCK)
    validate_lock_schema(lock)
    expected = {PRIMARY_MODEL_ID: PRIMARY_PREDICTION, GLOBAL_MODEL_ID: GLOBAL_PREDICTION}
    for model_id, relative in expected.items():
        path = root / relative
        item = lock["predictions"][model_id]
        frame = pd.read_parquet(path)
        if sha256(path) != item["sha256"] or len(frame) != EXPECTED_ROWS or not frame["row_hash"].is_unique:
            raise RuntimeError(f"Locked prediction is invalid: {model_id}")
        if row_digest(frame["row_hash"]) != lock["iid_feature_snapshot_row_digest"]:
            raise RuntimeError(f"Locked row digest differs: {model_id}")
        if not np.isfinite(frame["prediction"].to_numpy(np.float64)).all() or TARGET in frame.columns:
            raise RuntimeError(f"Locked prediction content is invalid: {model_id}")
    if lock["target_access_successful_read_count"] != 0:
        raise RuntimeError("Prediction lock was not created before target access.")
    return lock


def validate_lock_schema(lock: dict[str, Any]) -> None:
    required = {
        "status", "created_at_utc", "authorization_id", "freeze_sha256",
        "primary_bundle_sha256", "global_bundle_sha256",
        "iid_feature_snapshot_row_digest", "predictions",
        "feature_access_successful_read_count", "target_access_successful_read_count",
    }
    if required - set(lock):
        raise ValueError(f"Prediction lock fields are missing: {sorted(required - set(lock))}")
    if set(lock["predictions"]) != set(PERMITTED_MODELS):
        raise ValueError("Prediction lock model set is invalid.")
    if lock["feature_access_successful_read_count"] != 1 or lock["target_access_successful_read_count"] != 0:
        raise ValueError("Prediction lock access counts are invalid.")


def preflight(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    required = {FREEZE: FREEZE_SHA, PRIMARY_BUNDLE: PRIMARY_SHA, GLOBAL_BUNDLE: GLOBAL_SHA}
    failures = {relative.as_posix(): {"expected": expected, "actual": sha256(root / relative) if (root / relative).is_file() else "MISSING"} for relative, expected in required.items() if not (root / relative).is_file() or sha256(root / relative) != expected}
    freeze = read_json(root, FREEZE)
    ready = read_json(root, REPORTS / "PROMPT4C_READY.json")
    reviewer = read_json(root, REPORTS / "prompt4c_reviewer.json")
    verification = read_json(root, REPORTS / "prompt4c_verification.json")
    ledger4c = read_json(root, REPORTS / "prompt4c_final_refit_ledger.json")
    roles = read_json(root, REPORTS / "feature_roles.json")
    config = read_json(root, Path("config.json"))
    features = freeze["project_identity"]["features"]
    checks = {
        "freeze_hash": not failures.get(FREEZE.as_posix()),
        "primary_bundle_hash": not failures.get(PRIMARY_BUNDLE.as_posix()),
        "global_bundle_hash": not failures.get(GLOBAL_BUNDLE.as_posix()),
        "prompt4c_ready": ready.get("status") == "PASS_FINAL_PRE_IID_FREEZE",
        "prompt4c_reviewer": reviewer.get("status") == "PASS",
        "prompt4c_verification": verification.get("status") == "PASS" and verification.get("check_count") == 30,
        "final_refit_roles": ledger4c.get("completed_role_count") == 11 and len(ledger4c.get("completed_roles", [])) == 11,
        "zero_technical_retries": ledger4c.get("technical_retry_count") == 0,
        "zero_scientific_searches": ledger4c.get("scientific_candidate_searches") == 0,
        "oof_rows": freeze["final_refit"].get("oof_rows") == 500_000,
        "zero_self_fit_rows": freeze["final_refit"].get("zero_self_fit_rows") == 0,
        "model_set_exact": tuple(item["model_id"] for item in freeze["iid_protocol"]["permitted_models"]) == PERMITTED_MODELS,
        "blocka_excluded": freeze["historical_blocka"].get("iid_eligible") is False,
        "feature_contract_exact": features == roles["contracts"]["main_without_sensitive_without_lender"] and len(features) == 35,
        "state_counts_zero": all(config["prompt5a"].get(key) == 0 for key in ("iid_feature_access_count", "iid_target_access_count", "iid_prediction_rows_total")),
        "prompt5_not_previously_executed": freeze["safety"].get("prompt5_executed") is False,
    }
    if failures or not all(checks.values()):
        raise RuntimeError("BLOCKED_PRE_IID_FREEZE_MISMATCH")
    assert_exact_model_set(PERMITTED_MODELS)
    if (root / LEDGER).exists():
        ledger = load_ledger(root)
        if ledger["authorization_id"] != AUTHORIZATION_ID:
            raise RuntimeError("Existing access ledger has the wrong authorization.")
    else:
        ledger = {
            "status": "INITIALIZED_PRE_IID",
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "authorization_id": AUTHORIZATION_ID,
            "feature_successful_content_reads": 0,
            "target_successful_content_reads": 0,
            "prediction_models": 0,
            "prediction_rows": 0,
            "target_joined": False,
            "fit_count": 0,
            "refit_count": 0,
            "tuning_operation_count": 0,
            "post_iid_model_changes": 0,
            "original_feature_status": "NOT_YET_OPENED",
            "original_target_status": "NOT_YET_OPENED",
            "events": [],
        }
        append_event(ledger, "access_ledger_created")
        save_ledger(root, ledger)
    report = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "checks": checks,
        "hashes": {relative.as_posix(): expected for relative, expected in required.items()},
        "permitted_models": list(PERMITTED_MODELS),
        "historical_blocka_iid_eligible": False,
        "iid_content_reads_during_preflight": 0,
    }
    atomic_json(root, REPORTS / "prompt5a_handoff_validation.json", report)
    runtime = {"status": "IN_PROGRESS", "created_at_utc": utc_now(), "preflight_seconds": time.perf_counter() - started, "feature_prediction_seconds": 0.0, "target_evaluation_seconds": 0.0, "figures_seconds": 0.0, "notebook_seconds": 0.0, "review_seconds": 0.0, "verification_seconds": 0.0}
    atomic_json(root, REPORTS / "prompt5a_runtime.json", runtime)
    return report


def record_focused_tests(root: Path, passed: int, failed: int = 0) -> dict[str, Any]:
    if load_ledger(root)["feature_successful_content_reads"] != 0:
        raise RuntimeError("Focused tests must be recorded before IID feature access.")
    payload = {"status": "PASS" if passed >= 18 and failed == 0 else "FAIL", "created_at_utc": utc_now(), "passed": int(passed), "failed": int(failed), "iid_content_reads": 0}
    atomic_json(root, REPORTS / "prompt5a_focused_tests.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError("Focused Prompt 5A test gate failed.")
    return payload


def _validate_feature_frame(frame: pd.DataFrame, features: list[str], sensitive: list[str], target_aliases: list[str]) -> None:
    if len(frame) != EXPECTED_ROWS or "row_hash" not in frame or not frame["row_hash"].is_unique:
        raise RuntimeError("IID feature membership is invalid.")
    if any(name in frame.columns for name in target_aliases) or TARGET in frame.columns:
        raise RuntimeError("IID feature file contains a target or target alias.")
    required = ["row_hash", *features, *sensitive]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"IID feature contract is incomplete: {missing}")
    model = frame.loc[:, features]
    if model.isna().any().any():
        raise RuntimeError("IID model features contain null values.")
    numeric = model.select_dtypes(include=[np.number])
    if numeric.shape[1] and not np.isfinite(numeric.to_numpy(np.float64)).all():
        raise RuntimeError("IID numeric inference inputs are not finite.")


def feature_predict(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    focused = read_json(root, REPORTS / "prompt5a_focused_tests.json")
    if focused.get("status") != "PASS":
        raise RuntimeError("Focused test gate is not PASS.")
    preflight(root)
    ledger = load_ledger(root)
    if ledger["target_successful_content_reads"] != 0:
        raise PermissionError("No prediction is authorized after target access.")
    if (root / LOCK).is_file():
        return validate_prediction_lock(root)
    freeze = read_json(root, FREEZE)
    roles = read_json(root, REPORTS / "feature_roles.json")
    features = list(freeze["project_identity"]["features"])
    sensitive = list(roles["sensitive_fields"])
    if ledger["feature_successful_content_reads"] == 0:
        if (root / MODEL_SNAPSHOT).exists() or (root / FAIRNESS_SNAPSHOT).exists():
            raise RuntimeError("Inconsistent pre-feature snapshot state.")
        ledger["feature_read_attempt_started_at_utc"] = utc_now()
        append_event(ledger, "feature_read_attempt_started")
        save_ledger(root, ledger)
        iid_features = pd.read_parquet(root / ORIGINAL_FEATURES)
        ledger = load_ledger(root)
        ledger["feature_successful_content_reads"] = 1
        ledger["original_feature_status"] = "OPENED_ONCE_IN_MEMORY"
        append_event(ledger, "feature_read_success", rows=len(iid_features))
        save_ledger(root, ledger)
        _validate_feature_frame(iid_features, features, sensitive, roles["target_and_alias_exclusions"])
        model_snapshot = iid_features.loc[:, ["row_hash", *features]].copy()
        fairness_snapshot = iid_features.loc[:, ["row_hash", *sensitive]].copy()
        atomic_parquet(root, MODEL_SNAPSHOT, model_snapshot)
        atomic_parquet(root, FAIRNESS_SNAPSHOT, fairness_snapshot)
        if len(pd.read_parquet(root / MODEL_SNAPSHOT)) != EXPECTED_ROWS or len(pd.read_parquet(root / FAIRNESS_SNAPSHOT)) != EXPECTED_ROWS:
            raise RuntimeError("IID feature snapshot reload failed.")
        ledger = load_ledger(root)
        ledger["model_feature_snapshot_sha256"] = sha256(root / MODEL_SNAPSHOT)
        ledger["fairness_audit_snapshot_sha256"] = sha256(root / FAIRNESS_SNAPSHOT)
        ledger["iid_feature_snapshot_row_digest"] = row_digest(model_snapshot["row_hash"])
        ledger["status"] = "FEATURE_SNAPSHOTS_READY"
        append_event(ledger, "feature_snapshots_persisted_and_reloaded")
        save_ledger(root, ledger)
        del iid_features, fairness_snapshot
    else:
        if ledger["feature_successful_content_reads"] != 1 or not (root / MODEL_SNAPSHOT).is_file() or not (root / FAIRNESS_SNAPSHOT).is_file():
            raise RuntimeError("BLOCKED_IID_SNAPSHOT_LOSS_AFTER_ACCESS")
        model_snapshot = pd.read_parquet(root / MODEL_SNAPSHOT)
    ledger = load_ledger(root)
    assert_prediction_allowed(ledger)
    assert_exact_model_set(PERMITTED_MODELS)
    model_snapshot = pd.read_parquet(root / MODEL_SNAPSHOT)
    if sha256(root / MODEL_SNAPSHOT) != ledger["model_feature_snapshot_sha256"]:
        raise RuntimeError("Saved IID model-feature snapshot changed.")
    model_frame = model_snapshot.loc[:, features]
    primary_bundle = joblib.load(root / PRIMARY_BUNDLE)
    global_bundle = joblib.load(root / GLOBAL_BUNDLE)
    if not isinstance(primary_bundle, FinalPrimaryBundle) or not isinstance(global_bundle, FinalGlobalBundle):
        raise TypeError("Frozen final bundle type is invalid.")
    details = primary_bundle.predict_details(model_frame)
    global_prediction = np.asarray(global_bundle.predict(model_frame), dtype=np.float64)
    primary_frame = pd.DataFrame({
        "row_hash": model_snapshot["row_hash"].astype(str),
        "prediction": details["prediction"],
        "global_base_prediction": details["global_prediction"],
        "meta_gate_probability": details["meta_gate_probability"],
        "residual_specialist_proposal": details["residual_prediction"],
        "routing_strength": details["routing_strength"],
        "routing_condition_activated": details["meta_gate_probability"] > float(primary_bundle.gate_threshold),
        "applied_residual_correction": details["routing_strength"] * details["residual_prediction"],
    })
    global_frame = pd.DataFrame({"row_hash": model_snapshot["row_hash"].astype(str), "prediction": global_prediction})
    for frame in (primary_frame, global_frame):
        if len(frame) != EXPECTED_ROWS or not frame["row_hash"].is_unique or not np.isfinite(frame["prediction"]).all() or TARGET in frame:
            raise RuntimeError("Frozen IID prediction is invalid.")
    atomic_parquet(root, PRIMARY_PREDICTION, primary_frame)
    atomic_parquet(root, GLOBAL_PREDICTION, global_frame)
    ledger = load_ledger(root)
    ledger["prediction_models"] = 2
    ledger["prediction_rows"] = 150_000
    ledger["prediction_rows_per_model"] = 75_000
    append_event(ledger, "prediction_generation_complete", models=list(PERMITTED_MODELS), total_rows=150_000)
    save_ledger(root, ledger)
    reloaded = {PRIMARY_MODEL_ID: pd.read_parquet(root / PRIMARY_PREDICTION), GLOBAL_MODEL_ID: pd.read_parquet(root / GLOBAL_PREDICTION)}
    digest = row_digest(model_snapshot["row_hash"])
    prediction_items = {}
    for model_id, relative in ((PRIMARY_MODEL_ID, PRIMARY_PREDICTION), (GLOBAL_MODEL_ID, GLOBAL_PREDICTION)):
        frame = reloaded[model_id]
        if len(frame) != EXPECTED_ROWS or row_digest(frame["row_hash"]) != digest or not np.isfinite(frame["prediction"]).all():
            raise RuntimeError("Prediction reload verification failed.")
        prediction_items[model_id] = {"path": relative.as_posix(), "sha256": sha256(root / relative), "rows": EXPECTED_ROWS, "finite": True, "row_digest": digest, "columns": list(frame.columns)}
    lock = {
        "status": "PASS_PREDICTIONS_LOCKED_BEFORE_TARGET_ACCESS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "freeze_sha256": FREEZE_SHA,
        "primary_bundle_sha256": PRIMARY_SHA,
        "global_bundle_sha256": GLOBAL_SHA,
        "iid_feature_snapshot_path": MODEL_SNAPSHOT.as_posix(),
        "iid_feature_snapshot_sha256": sha256(root / MODEL_SNAPSHOT),
        "iid_feature_snapshot_row_digest": digest,
        "predictions": prediction_items,
        "feature_access_successful_read_count": 1,
        "target_access_successful_read_count": 0,
        "internal_diagnostics_available": True,
        "model_prediction_calls_completed": 2,
        "model_prediction_calls_after_target_access": 0,
    }
    atomic_json(root, LOCK, lock)
    validate_prediction_lock(root)
    ledger = load_ledger(root)
    ledger["status"] = "PREDICTION_LOCKED"
    ledger["prediction_lock_sha256"] = sha256(root / LOCK)
    append_event(ledger, "prediction_lock_created_and_reloaded", prediction_lock_sha256=ledger["prediction_lock_sha256"])
    save_ledger(root, ledger)
    manifest = {"status": "PASS", "created_at_utc": utc_now(), "model_count": 2, "total_prediction_rows": 150_000, "predictions": prediction_items, "prediction_lock_sha256": sha256(root / LOCK), "target_access_count_at_creation": 0}
    atomic_json(root, REPORTS / "prompt5a_prediction_manifest.json", manifest)
    runtime = read_json(root, REPORTS / "prompt5a_runtime.json")
    runtime["feature_prediction_seconds"] = time.perf_counter() - started
    atomic_json(root, REPORTS / "prompt5a_runtime.json", runtime)
    return lock


def _aligned_saved_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_ids = pd.read_parquet(root / MODEL_SNAPSHOT, columns=["row_hash"])
    target = pd.read_parquet(root / TARGET_SNAPSHOT)
    primary = pd.read_parquet(root / PRIMARY_PREDICTION)
    global_frame = pd.read_parquet(root / GLOBAL_PREDICTION)
    sets = [set(frame["row_hash"].astype(str)) for frame in (model_ids, target, primary, global_frame)]
    if any(value != sets[0] for value in sets[1:]) or any(not frame["row_hash"].is_unique for frame in (model_ids, target, primary, global_frame)):
        raise RuntimeError("BLOCKED_IID_ALIGNMENT")
    order = model_ids["row_hash"].astype(str)
    target_map = target.set_index(target["row_hash"].astype(str))[TARGET]
    primary_map = primary.set_index(primary["row_hash"].astype(str))
    global_map = global_frame.set_index(global_frame["row_hash"].astype(str))["prediction"]
    aligned = pd.DataFrame({
        "row_hash": order,
        "y_true": target_map.loc[order].to_numpy(np.float64),
        "primary_prediction": primary_map.loc[order, "prediction"].to_numpy(np.float64),
        "global_prediction": global_map.loc[order].to_numpy(np.float64),
    })
    diagnostics = primary_map.loc[order].reset_index(drop=True)
    return aligned, diagnostics


def _rowwise_summary(frame: pd.DataFrame, scope: str, mask: np.ndarray) -> dict[str, Any]:
    values = frame.loc[mask, "delta_abs_error"].to_numpy(np.float64)
    primary_win = values < 0.0
    global_win = values > 0.0
    ties = values == 0.0
    return {
        "scope": scope,
        "n": int(values.size),
        "fraction_primary_better": float(np.mean(primary_win)),
        "fraction_global_better": float(np.mean(global_win)),
        "exact_ties": int(np.count_nonzero(ties)),
        "tie_fraction": float(np.mean(ties)),
        "mean_primary_improvement_on_primary_win_rows": float(np.mean(-values[primary_win])) if primary_win.any() else np.nan,
        "median_primary_improvement_on_primary_win_rows": float(np.median(-values[primary_win])) if primary_win.any() else np.nan,
        "mean_primary_damage_on_global_win_rows": float(np.mean(values[global_win])) if global_win.any() else np.nan,
        "median_primary_damage_on_global_win_rows": float(np.median(values[global_win])) if global_win.any() else np.nan,
    }


def _error_distribution(frame: pd.DataFrame, model: str, scope: str, mask: np.ndarray) -> dict[str, Any]:
    absolute = frame.loc[mask, f"{model}_abs_error"].to_numpy(np.float64)
    signed = frame.loc[mask, f"{model}_signed_error"].to_numpy(np.float64)
    result = {"model_id": PRIMARY_MODEL_ID if model == "primary" else GLOBAL_MODEL_ID, "scope": scope, "n": int(absolute.size)}
    for level in (0.50, 0.75, 0.90, 0.95, 0.99):
        result[f"absolute_error_p{int(level*100):02d}"] = float(np.quantile(absolute, level))
    result["absolute_error_maximum"] = float(np.max(absolute))
    for level in (0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99):
        result[f"signed_error_p{int(level*100):02d}"] = float(np.quantile(signed, level))
    return result


def _underprediction_row(frame: pd.DataFrame, model: str, scope_type: str, scope: str, mask: np.ndarray) -> dict[str, Any]:
    signed = frame.loc[mask, f"{model}_signed_error"].to_numpy(np.float64)
    under = signed < 0.0
    over = signed > 0.0
    return {
        "model_id": PRIMARY_MODEL_ID if model == "primary" else GLOBAL_MODEL_ID,
        "scope_type": scope_type,
        "scope": scope,
        "n": int(signed.size),
        "underprediction_rate": float(np.mean(under)),
        "mean_underprediction_magnitude": float(np.mean(-signed[under])) if under.any() else np.nan,
        "median_underprediction_magnitude": float(np.median(-signed[under])) if under.any() else np.nan,
        "mean_overprediction_magnitude": float(np.mean(signed[over])) if over.any() else np.nan,
        "median_overprediction_magnitude": float(np.median(signed[over])) if over.any() else np.nan,
    }


def _build_reports(root: Path, frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["y_true"].to_numpy(np.float64)
    predictions = {PRIMARY_MODEL_ID: frame["primary_prediction"].to_numpy(np.float64), GLOBAL_MODEL_ID: frame["global_prediction"].to_numpy(np.float64)}
    overall_rows = []
    metrics_by_model = {}
    for model_id, prediction in predictions.items():
        row = {"model_id": model_id, "role": "Primary" if model_id == PRIMARY_MODEL_ID else "Global comparator", **metric_row(y, prediction)}
        overall_rows.append(row)
        metrics_by_model[model_id] = row
    overall = pd.DataFrame(overall_rows)
    atomic_csv(root, REPORTS / "prompt5a_iid_overall_metrics.csv", overall)

    decile_rows = []
    for model_id, prediction in predictions.items():
        for index in range(1, 11):
            mask = frame["iid_local_decile"].eq(f"D{index}").to_numpy()
            decile_rows.append({"model_id": model_id, "decile": f"D{index}", "decile_index": index, **group_metric_row(y[mask], prediction[mask])})
    deciles = pd.DataFrame(decile_rows)
    atomic_csv(root, REPORTS / "prompt5a_iid_local_decile_metrics.csv", deciles)

    cutpoints = read_json(root, FREEZE)["iid_protocol"]["development_frozen_target_cutpoints"]
    band_rows = []
    for model_id, prediction in predictions.items():
        for index in range(1, 11):
            mask = frame["development_frozen_band_index"].eq(index).to_numpy()
            band_rows.append({"model_id": model_id, "band": frame.loc[mask, "development_frozen_band"].iloc[0] if mask.any() else f"B{index}", "band_index": index, **group_metric_row(y[mask], prediction[mask])})
    bands = pd.DataFrame(band_rows)
    atomic_csv(root, REPORTS / "prompt5a_development_frozen_band_metrics.csv", bands)

    condition_rows, conditions = six_condition_rows(metrics_by_model[PRIMARY_MODEL_ID], metrics_by_model[GLOBAL_MODEL_ID])
    conditions.update({"created_at_utc": utc_now(), "primary_model_id": PRIMARY_MODEL_ID, "global_model_id": GLOBAL_MODEL_ID})
    atomic_json(root, REPORTS / "prompt5a_iid_six_condition_check.json", conditions)

    bootstrap, bootstrap_samples = paired_bootstrap(y, predictions[PRIMARY_MODEL_ID], predictions[GLOBAL_MODEL_ID])
    atomic_csv(root, REPORTS / "prompt5a_iid_bootstrap.csv", bootstrap)
    bootstrap_plot = pd.DataFrame({name: values for name, values in bootstrap_samples.items()})
    atomic_csv(root, REPORTS / "prompt5a_plot_bootstrap_samples.csv", bootstrap_plot)

    rowwise = [_rowwise_summary(frame, "All IID", np.ones(len(frame), dtype=bool))]
    for index in range(1, 11):
        rowwise.append(_rowwise_summary(frame, f"D{index}", frame["iid_local_decile"].eq(f"D{index}").to_numpy()))
    rowwise_frame = pd.DataFrame(rowwise)
    atomic_csv(root, REPORTS / "prompt5a_primary_vs_global_rowwise.csv", rowwise_frame)

    top10 = quantile_membership(y, 0.90)
    top05 = quantile_membership(y, 0.95)
    scopes = {"All IID": np.ones(len(frame), dtype=bool), "Bottom-90": ~top10, "Top-decile": top10, "Top-5%": top05}
    distributions = [_error_distribution(frame, model, scope, mask) for model in ("primary", "global") for scope, mask in scopes.items()]
    atomic_csv(root, REPORTS / "prompt5a_error_distribution.csv", pd.DataFrame(distributions))

    large_rows = []
    for label, level in (("Primary top-5% absolute error", 0.95), ("Primary top-1% absolute error", 0.99)):
        threshold = float(np.quantile(frame["primary_abs_error"], level))
        mask = frame["primary_abs_error"].to_numpy() >= threshold
        large_rows.append({
            "subset": label, "threshold": threshold, "n": int(mask.sum()),
            "mean_target": float(np.mean(y[mask])), "median_target": float(np.median(y[mask])),
            "mean_primary_prediction": float(np.mean(predictions[PRIMARY_MODEL_ID][mask])), "mean_global_prediction": float(np.mean(predictions[GLOBAL_MODEL_ID][mask])),
            "primary_mae": float(np.mean(frame.loc[mask, "primary_abs_error"])), "global_mae": float(np.mean(frame.loc[mask, "global_abs_error"])),
            "primary_mean_signed_error": float(np.mean(frame.loc[mask, "primary_signed_error"])), "global_mean_signed_error": float(np.mean(frame.loc[mask, "global_signed_error"])),
            "primary_underprediction_rate": float(np.mean(frame.loc[mask, "primary_signed_error"] < 0)), "global_underprediction_rate": float(np.mean(frame.loc[mask, "global_signed_error"] < 0)),
            "fraction_from_iid_d10": float(np.mean(frame.loc[mask, "iid_local_decile"].eq("D10"))), "fraction_from_iid_top_5_percent_target": float(np.mean(top05[mask])),
        })
    large = pd.DataFrame(large_rows)
    atomic_csv(root, REPORTS / "prompt5a_large_error_analysis.csv", large)

    under_rows = []
    for model in ("primary", "global"):
        under_rows.append(_underprediction_row(frame, model, "overall", "All IID", np.ones(len(frame), dtype=bool)))
        for index in range(1, 11):
            under_rows.append(_underprediction_row(frame, model, "iid_local_decile", f"D{index}", frame["iid_local_decile"].eq(f"D{index}").to_numpy()))
        for index in range(1, 11):
            mask = frame["development_frozen_band_index"].eq(index).to_numpy()
            under_rows.append(_underprediction_row(frame, model, "development_frozen_band", f"B{index}", mask))
    under = pd.DataFrame(under_rows)
    atomic_csv(root, REPORTS / "prompt5a_underprediction_analysis.csv", under)

    routing_rows = []
    routed = frame["routing_condition_activated"].to_numpy(bool)
    routing_scopes = {"All IID": np.ones(len(frame), dtype=bool), **{f"D{i}": frame["iid_local_decile"].eq(f"D{i}").to_numpy() for i in range(1, 11)}, "Top-decile": top10, "Top-5%": top05}
    for scope, mask in routing_scopes.items():
        selected = mask
        routed_scope = routed & selected
        correction = frame.loc[selected, "applied_residual_correction"].to_numpy(np.float64)
        routing_rows.append({
            "scope": scope, "n": int(selected.sum()), "routed_rows": int(routed_scope.sum()), "routed_fraction": float(np.mean(routed[selected])),
            "mean_applied_correction": float(np.mean(correction)), "median_applied_correction": float(np.median(correction)),
            "correction_abs_p50": float(np.quantile(np.abs(correction), 0.50)), "correction_abs_p90": float(np.quantile(np.abs(correction), 0.90)), "correction_abs_p95": float(np.quantile(np.abs(correction), 0.95)), "correction_abs_p99": float(np.quantile(np.abs(correction), 0.99)),
            "primary_mae_routed": float(np.mean(frame.loc[routed_scope, "primary_abs_error"])) if routed_scope.any() else np.nan,
            "global_mae_routed": float(np.mean(frame.loc[routed_scope, "global_abs_error"])) if routed_scope.any() else np.nan,
            "primary_advantage_routed": float(np.mean(-frame.loc[routed_scope, "delta_abs_error"])) if routed_scope.any() else np.nan,
            "primary_damage_on_routed_global_win_rows": float(np.mean(frame.loc[routed_scope & frame["delta_abs_error"].gt(0).to_numpy(), "delta_abs_error"])) if np.any(routed_scope & frame["delta_abs_error"].gt(0).to_numpy()) else np.nan,
        })
    nonrouted = ~routed
    routing_rows[0]["primary_mae_non_routed"] = float(np.mean(frame.loc[nonrouted, "primary_abs_error"]))
    routing_rows[0]["global_mae_non_routed"] = float(np.mean(frame.loc[nonrouted, "global_abs_error"]))
    routing = pd.DataFrame(routing_rows)
    atomic_csv(root, REPORTS / "prompt5a_routing_diagnostics.csv", routing)

    historical = pd.read_csv(root / REPORTS / "prompt4c_historical_metric_extension.csv")
    mapping = {PRIMARY_MODEL_ID: "stage3_residual_t75_a75", GLOBAL_MODEL_ID: "ens_boost_cat060"}
    transport_rows = []
    for model_id, historical_id in mapping.items():
        dev = historical.loc[historical["model_id"].eq(historical_id)].iloc[0]
        iid = metrics_by_model[model_id]
        transport_rows.append({
            "model_id": model_id, "development_evidence_label": "adaptive Development Validation", "iid_evidence_label": "final untouched IID holdout",
            "development_mae": dev["mae"], "iid_mae": iid["mae"], "mae_absolute_change_iid_minus_development": iid["mae"] - dev["mae"], "mae_relative_change_percent": 100.0 * (iid["mae"] - dev["mae"]) / dev["mae"],
            "development_mape_percent": dev["mape_percent"], "iid_mape_percent": iid["mape_percent"],
            "development_wape_percent": dev["wape_percent"], "iid_wape_percent": iid["wape_percent"],
            "development_bottom_90_mae": dev["bottom_90_mae"], "iid_bottom_90_mae": iid["bottom_90_mae"],
            "development_top_decile_mae": dev["top_decile_mae"], "iid_top_decile_mae": iid["top_decile_mae"],
            "development_top_decile_underprediction_rate": dev["top_decile_underprediction_rate"], "iid_top_decile_underprediction_rate": iid["top_decile_underprediction_rate"],
            "interpretation_rule": "descriptive comparison only; Development was adaptive and IID is the final untouched holdout",
        })
    transport = pd.DataFrame(transport_rows)
    atomic_csv(root, REPORTS / "prompt5a_development_iid_transport.csv", transport)

    atomic_csv(root, REPORTS / "prompt5a_plot_overall_metrics.csv", overall[["model_id", "mae", "rmse", "mape_percent", "wape_percent"]])
    atomic_csv(root, REPORTS / "prompt5a_plot_deciles.csv", deciles)
    atomic_csv(root, REPORTS / "prompt5a_plot_body_tail.csv", overall[["model_id", "bottom_90_mae", "top_decile_mae", "top_five_percent_mae"]])
    atomic_csv(root, REPORTS / "prompt5a_plot_error_distribution.csv", pd.DataFrame(distributions))
    atomic_csv(root, REPORTS / "prompt5a_plot_routing.csv", routing)
    return {"overall": overall, "deciles": deciles, "bands": bands, "bootstrap": bootstrap, "rowwise": rowwise_frame, "large": large, "under": under, "routing": routing, "transport": transport, "conditions": conditions}


def target_evaluate(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    lock = validate_prediction_lock(root)
    ledger = load_ledger(root)
    if ledger["target_successful_content_reads"] == 0:
        assert_target_access_allowed(ledger, lock.get("status") == "PASS_PREDICTIONS_LOCKED_BEFORE_TARGET_ACCESS")
        if (root / TARGET_SNAPSHOT).exists():
            raise RuntimeError("Inconsistent pre-target snapshot state.")
        ledger["target_read_attempt_started_at_utc"] = utc_now()
        append_event(ledger, "target_read_attempt_started", prediction_lock_sha256=sha256(root / LOCK))
        save_ledger(root, ledger)
        targets = pd.read_parquet(root / ORIGINAL_TARGETS)
        ledger = load_ledger(root)
        ledger["target_successful_content_reads"] = 1
        ledger["original_target_status"] = "OPENED_ONCE_IN_MEMORY"
        append_event(ledger, "target_read_success", rows=len(targets))
        save_ledger(root, ledger)
        if len(targets) != EXPECTED_ROWS or list(targets.columns) != ["row_hash", TARGET] or not targets["row_hash"].is_unique:
            raise RuntimeError("IID target contract is invalid.")
        y = targets[TARGET].to_numpy(np.float64)
        if not np.isfinite(y).all():
            raise RuntimeError("IID targets are not finite.")
        target_set = set(targets["row_hash"].astype(str))
        prediction_set = set(pd.read_parquet(root / PRIMARY_PREDICTION, columns=["row_hash"])["row_hash"].astype(str))
        if target_set != prediction_set:
            raise RuntimeError("BLOCKED_IID_ALIGNMENT")
        snapshot = targets.loc[:, ["row_hash", TARGET]].copy()
        atomic_parquet(root, TARGET_SNAPSHOT, snapshot)
        saved = pd.read_parquet(root / TARGET_SNAPSHOT)
        if len(saved) != EXPECTED_ROWS or not saved["row_hash"].is_unique:
            raise RuntimeError("IID target snapshot reload failed.")
        ledger = load_ledger(root)
        ledger["target_snapshot_sha256"] = sha256(root / TARGET_SNAPSHOT)
        ledger["target_positive_count"] = int(np.count_nonzero(y > 0.0))
        ledger["target_nonpositive_count"] = int(np.count_nonzero(y <= 0.0))
        ledger["status"] = "TARGET_SNAPSHOT_READY"
        append_event(ledger, "target_snapshot_persisted_and_reloaded", sha256=ledger["target_snapshot_sha256"])
        save_ledger(root, ledger)
        del targets, snapshot
    elif ledger["target_successful_content_reads"] == 1:
        if not (root / TARGET_SNAPSHOT).is_file():
            raise RuntimeError("BLOCKED_IID_SNAPSHOT_LOSS_AFTER_ACCESS")
        if sha256(root / TARGET_SNAPSHOT) != ledger.get("target_snapshot_sha256"):
            raise RuntimeError("Saved IID target snapshot changed.")
    else:
        raise RuntimeError("Original IID target read count exceeds one.")
    if (root / EVALUATION_FRAME).is_file():
        frame = pd.read_parquet(root / EVALUATION_FRAME)
    else:
        aligned, diagnostics = _aligned_saved_frame(root)
        y = aligned["y_true"].to_numpy(np.float64)
        aligned["primary_signed_error"] = aligned["primary_prediction"] - y
        aligned["global_signed_error"] = aligned["global_prediction"] - y
        aligned["primary_abs_error"] = np.abs(aligned["primary_signed_error"])
        aligned["global_abs_error"] = np.abs(aligned["global_signed_error"])
        aligned["primary_ape"] = np.where(y > 0.0, aligned["primary_abs_error"] / y, np.nan)
        aligned["global_ape"] = np.where(y > 0.0, aligned["global_abs_error"] / y, np.nan)
        aligned["delta_abs_error"] = aligned["primary_abs_error"] - aligned["global_abs_error"]
        aligned["primary_better_flag"] = aligned["delta_abs_error"] < 0.0
        local = duplicate_safe_target_deciles(y) + 1
        aligned["iid_local_decile"] = [f"D{value}" for value in local]
        cutpoints = read_json(root, FREEZE)["iid_protocol"]["development_frozen_target_cutpoints"]
        band = assign_development_bands(y, cutpoints)
        labels = {1: "B1_le_q10", 2: "B2_q10_q20", 3: "B3_q20_q30", 4: "B4_q30_q40", 5: "B5_q40_q50", 6: "B6_q50_q60", 7: "B7_q60_q70", 8: "B8_q70_q80", 9: "B9_q80_q90", 10: "B10_gt_q90"}
        aligned["development_frozen_band_index"] = band
        aligned["development_frozen_band"] = [labels[int(value)] for value in band]
        for name in ("global_base_prediction", "meta_gate_probability", "residual_specialist_proposal", "routing_strength", "routing_condition_activated", "applied_residual_correction"):
            aligned[name] = diagnostics[name].to_numpy()
        frame = aligned
        atomic_parquet(root, EVALUATION_FRAME, frame)
        reloaded = pd.read_parquet(root / EVALUATION_FRAME)
        if len(reloaded) != EXPECTED_ROWS or row_digest(reloaded["row_hash"]) != lock["iid_feature_snapshot_row_digest"]:
            raise RuntimeError("IID evaluation-frame reload failed.")
    reports = _build_reports(root, frame)
    ledger = load_ledger(root)
    ledger["target_joined"] = True
    ledger["aligned_evaluation_rows"] = EXPECTED_ROWS
    ledger["status"] = "EVALUATION_COMPLETE_ORIGINAL_IID_CLOSED"
    ledger["original_feature_status"] = "ORIGINAL_IID_CONTENT_CLOSED_AFTER_PROMPT5A"
    ledger["original_target_status"] = "ORIGINAL_IID_CONTENT_CLOSED_AFTER_PROMPT5A"
    ledger["model_prediction_calls_after_target_access"] = 0
    if not any(item.get("event") == "evaluation_complete" for item in ledger.get("events", [])):
        append_event(ledger, "evaluation_complete", evaluation_frame_sha256=sha256(root / EVALUATION_FRAME))
    save_ledger(root, ledger)
    manifest_items = {}
    for name, relative in {
        "iid_model_features_snapshot": MODEL_SNAPSHOT,
        "iid_fairness_audit_snapshot": FAIRNESS_SNAPSHOT,
        "iid_target_snapshot": TARGET_SNAPSHOT,
        "iid_evaluation_frame": EVALUATION_FRAME,
        "primary_prediction": PRIMARY_PREDICTION,
        "global_prediction": GLOBAL_PREDICTION,
    }.items():
        manifest_items[name] = {"path": relative.as_posix(), "sha256": sha256(root / relative), "rows": pq.ParquetFile(root / relative).metadata.num_rows, "schema": schema_details(root / relative)}
    manifest = {"status": "PASS", "created_at_utc": utc_now(), "authorization_id": AUTHORIZATION_ID, "original_iid_files_closed": True, "future_iid_sources_restricted_to_saved_snapshots": True, "artifacts": manifest_items}
    atomic_json(root, REPORTS / "prompt5a_post_iid_snapshot_manifest.json", manifest)
    build_figures(root, reports)
    runtime = read_json(root, REPORTS / "prompt5a_runtime.json")
    runtime["target_evaluation_seconds"] = time.perf_counter() - started
    atomic_json(root, REPORTS / "prompt5a_runtime.json", runtime)
    return {"status": "PASS", "primary": reports["overall"].iloc[0].to_dict(), "global": reports["overall"].iloc[1].to_dict(), "conditions": reports["conditions"]}


def _save_figure(root: Path, name: str, figure: plt.Figure) -> None:
    path = root / FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=150, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def build_figures(root: Path, reports: dict[str, Any]) -> None:
    started = time.perf_counter()
    colors = {PRIMARY_MODEL_ID: "#2b6cb0", GLOBAL_MODEL_ID: "#dd6b20"}
    overall = reports["overall"].set_index("model_id")
    deciles = reports["deciles"]
    short = ["mae", "rmse", "mape_percent", "wape_percent"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for axis, metric in zip(axes, short):
        values = [overall.loc[mid, metric] for mid in PERMITTED_MODELS]
        axis.bar(["Primary", "Global"], values, color=[colors[mid] for mid in PERMITTED_MODELS])
        axis.set_title(metric.replace("_", " ").upper()); axis.grid(axis="y", alpha=.25)
    fig.suptitle("Final IID overall metrics (lower is better)")
    _save_figure(root, "01_overall_metric_comparison.png", fig)
    for number, metric, title in ((2, "mae", "MAE"), (3, "mape_percent", "MAPE (%)"), (4, "wape_percent", "WAPE (%)"), (5, "mean_signed_error", "Mean signed error"), (6, "underprediction_rate", "Underprediction rate")):
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for model_id in PERMITTED_MODELS:
            part = deciles[deciles["model_id"].eq(model_id)].sort_values("decile_index")
            axis.plot(part["decile_index"], part[metric], marker="o", label="Primary" if model_id == PRIMARY_MODEL_ID else "Global", color=colors[model_id])
        if metric == "mean_signed_error": axis.axhline(0, color="black", lw=.8)
        axis.set(xlabel="IID-local target decile", ylabel=title, xticks=range(1, 11), title=f"{title} by IID-local target decile")
        axis.grid(alpha=.25); axis.legend()
        _save_figure(root, f"{number:02d}_{metric}_by_decile.png", fig)
    pivot = deciles.pivot(index="decile_index", columns="model_id", values="mae")
    difference = pivot[PRIMARY_MODEL_ID] - pivot[GLOBAL_MODEL_ID]
    fig, axis = plt.subplots(figsize=(8, 4.5)); axis.bar(difference.index, difference, color=np.where(difference < 0, "#2f855a", "#c53030")); axis.axhline(0, color="black", lw=.8); axis.set(xlabel="IID-local target decile", ylabel="Primary minus Global MAE", title="Primary-minus-Global MAE by target decile", xticks=range(1,11)); axis.grid(axis="y", alpha=.25)
    _save_figure(root, "07_primary_minus_global_mae_by_decile.png", fig)
    frame = pd.read_parquet(root / EVALUATION_FRAME, columns=["primary_abs_error", "global_abs_error"])
    cap = float(np.quantile(np.concatenate([frame["primary_abs_error"], frame["global_abs_error"]]), .99))
    fig, axis = plt.subplots(figsize=(8, 4.5)); axis.hist(frame["primary_abs_error"].clip(upper=cap), bins=60, density=True, alpha=.55, label="Primary", color=colors[PRIMARY_MODEL_ID]); axis.hist(frame["global_abs_error"].clip(upper=cap), bins=60, density=True, alpha=.45, label="Global", color=colors[GLOBAL_MODEL_ID]); axis.set(xlabel="Absolute error (clipped at combined P99)", ylabel="Density", title="IID absolute-error distribution"); axis.grid(alpha=.2); axis.legend()
    _save_figure(root, "08_absolute_error_distribution.png", fig)
    body = reports["overall"].melt(id_vars="model_id", value_vars=["bottom_90_mae", "top_decile_mae", "top_five_percent_mae"], var_name="scope", value_name="metric_value")
    fig, axis = plt.subplots(figsize=(9, 4.5)); x=np.arange(3); width=.36
    for offset, model_id in enumerate(PERMITTED_MODELS):
        values = body[body.model_id.eq(model_id)]["metric_value"].to_numpy(); axis.bar(x + (offset-.5)*width, values, width, label="Primary" if offset==0 else "Global", color=colors[model_id])
    axis.set(xticks=x, xticklabels=["Bottom-90", "Top-decile", "Top-5%"], ylabel="MAE", title="IID Body and Tail MAE"); axis.grid(axis="y", alpha=.25); axis.legend()
    _save_figure(root, "09_body_tail_mae.png", fig)
    boot = pd.read_csv(root / REPORTS / "prompt5a_plot_bootstrap_samples.csv")
    fig, axis = plt.subplots(figsize=(8, 4.5)); axis.hist(boot["MAE"], bins=40, color="#4a5568", alpha=.8); axis.axvline(0, color="#c53030", lw=1.4); axis.axvline(float(boot["MAE"].mean()), color="#2b6cb0", lw=1.4); axis.set(xlabel="Primary minus Global MAE", ylabel="Bootstrap resamples", title="Paired IID bootstrap: MAE difference"); axis.grid(alpha=.2)
    _save_figure(root, "10_bootstrap_mae_difference.png", fig)
    routing = reports["routing"]
    routed_deciles = routing[routing["scope"].str.match(r"D\d+")].copy(); routed_deciles["decile"] = routed_deciles["scope"].str[1:].astype(int)
    fig, axes = plt.subplots(1,2,figsize=(11,4.3)); axes[0].plot(routed_deciles["decile"], routed_deciles["routed_fraction"], marker="o"); axes[0].set(xlabel="IID-local target decile", ylabel="Routed fraction", title="Frozen routing rate", xticks=range(1,11)); axes[1].bar(routed_deciles["decile"], routed_deciles["primary_advantage_routed"], color=np.where(routed_deciles["primary_advantage_routed"]>=0,"#2f855a","#c53030")); axes[1].axhline(0,color="black",lw=.8); axes[1].set(xlabel="IID-local target decile", ylabel="Global MAE minus Primary MAE", title="Realized gain on routed rows", xticks=range(1,11)); [a.grid(alpha=.25) for a in axes]
    _save_figure(root, "11_routing_fraction_and_gain.png", fig)
    runtime = read_json(root, REPORTS / "prompt5a_runtime.json")
    runtime["figures_seconds"] = time.perf_counter() - started
    atomic_json(root, REPORTS / "prompt5a_runtime.json", runtime)


def build_and_execute_notebook(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells = [nbformat.v4.new_markdown_cell("# Prompt 5A - One-Time IID Evaluation and Final Error Analysis\n\nPrompt 4C froze the final Primary and Global comparator before IID. Predictions were locked before target access. Only these two models were evaluated, and Block A was excluded before IID. IID was never used for tuning."),
             nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path('..')\nR=ROOT/'outputs/reports'\nF=ROOT/'outputs/figures/prompt5a'"),
             nbformat.v4.new_markdown_cell("## Access order and frozen scope\n\nThe access ledger proves one feature read, prediction locking, and then one target read. No model was fitted or changed."),
             nbformat.v4.new_code_cell("ledger=json.loads((R/'prompt5a_iid_access_ledger.json').read_text())\nlock=json.loads((R/'prompt5a_prediction_lock.json').read_text())\ndisplay(pd.DataFrame([{'feature reads':ledger['feature_successful_content_reads'],'target reads':ledger['target_successful_content_reads'],'models':ledger['prediction_models'],'prediction rows':ledger['prediction_rows'],'post-target predictions':ledger['model_prediction_calls_after_target_access']}]))"),
             nbformat.v4.new_markdown_cell("## Overall IID performance\n\nMAE remains the Primary metric. MAPE and WAPE are secondary percentage summaries."),
             nbformat.v4.new_code_cell("overall=pd.read_csv(R/'prompt5a_iid_overall_metrics.csv')\ndisplay(overall)\ndisplay(Image(filename=str(F/'01_overall_metric_comparison.png')))"),
             nbformat.v4.new_markdown_cell("## IID-local target deciles\n\nThese deciles use true IID targets only for descriptive error analysis."),
             nbformat.v4.new_code_cell("deciles=pd.read_csv(R/'prompt5a_iid_local_decile_metrics.csv')\ndisplay(deciles)\nfor name in ['02_mae_by_decile.png','03_mape_percent_by_decile.png','04_wape_percent_by_decile.png','05_mean_signed_error_by_decile.png','06_underprediction_rate_by_decile.png','07_primary_minus_global_mae_by_decile.png']:\n    display(Image(filename=str(F/name)))"),
             nbformat.v4.new_markdown_cell("## Development-frozen target bands\n\nThese absolute bands use the q10-q90 cutpoints frozen before IID."),
             nbformat.v4.new_code_cell("bands=pd.read_csv(R/'prompt5a_development_frozen_band_metrics.csv')\ndisplay(bands)"),
             nbformat.v4.new_markdown_cell("## Body, Tail, and error distributions\n\nSigned error is prediction minus truth. Negative values mean underprediction."),
             nbformat.v4.new_code_cell("display(Image(filename=str(F/'08_absolute_error_distribution.png')))\ndisplay(Image(filename=str(F/'09_body_tail_mae.png')))\ndisplay(pd.read_csv(R/'prompt5a_error_distribution.csv'))\ndisplay(pd.read_csv(R/'prompt5a_underprediction_analysis.csv'))"),
             nbformat.v4.new_markdown_cell("## Frozen six-condition check\n\nThis check describes IID generalization. It cannot change the frozen Primary."),
             nbformat.v4.new_code_cell("six=json.loads((R/'prompt5a_iid_six_condition_check.json').read_text())\ndisplay(pd.DataFrame(six['conditions']))\ndisplay(pd.DataFrame([{'status':six['status'],'passed':six['conditions_passed'],'total':six['conditions_total']}]))"),
             nbformat.v4.new_markdown_cell("## Paired IID bootstrap uncertainty\n\nThe same resampled rows were used for both models in all 500 resamples with seed 42."),
             nbformat.v4.new_code_cell("display(pd.read_csv(R/'prompt5a_iid_bootstrap.csv'))\ndisplay(Image(filename=str(F/'10_bootstrap_mae_difference.png')))"),
             nbformat.v4.new_markdown_cell("## Row-level win/loss and large-error concentration\n\nOnly aggregate results are displayed here. The sealed evaluation frame stores the row-level analysis data."),
             nbformat.v4.new_code_cell("display(pd.read_csv(R/'prompt5a_primary_vs_global_rowwise.csv'))\ndisplay(pd.read_csv(R/'prompt5a_large_error_analysis.csv'))"),
             nbformat.v4.new_markdown_cell("## Frozen Stage 3 routing diagnostics\n\nThese values come from the existing deterministic bundle. No threshold was changed."),
             nbformat.v4.new_code_cell("display(pd.read_csv(R/'prompt5a_routing_diagnostics.csv'))\ndisplay(Image(filename=str(F/'11_routing_fraction_and_gain.png')))"),
             nbformat.v4.new_markdown_cell("## Development-to-IID transport\n\nDevelopment was adaptive. IID is the final untouched holdout, so the differences are descriptive."),
             nbformat.v4.new_code_cell("display(pd.read_csv(R/'prompt5a_development_iid_transport.csv'))"),
             nbformat.v4.new_markdown_cell("## Closure and later-stage boundary\n\nThe final model was not changed after IID. The original IID files are closed after this one-time evaluation. Parts 2 and 3 must use the immutable post-IID snapshots and saved Prompt 5A evidence only."),
             nbformat.v4.new_code_cell("manifest=json.loads((R/'prompt5a_post_iid_snapshot_manifest.json').read_text())\ndisplay(pd.DataFrame([{'artifact':k,**{'rows':v['rows'],'sha256':v['sha256']}} for k,v in manifest['artifacts'].items()]))")]
    nb.cells = cells
    path = root / NOTEBOOK
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    executed = client.execute()
    nbformat.write(executed, path)
    code = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [out for cell in code for out in cell.get("outputs", []) if out.get("output_type") == "error"]
    images = sum(1 for cell in code for out in cell.get("outputs", []) if out.get("output_type") == "display_data" and "image/png" in out.get("data", {}))
    tables = sum(1 for cell in code for out in cell.get("outputs", []) if "text/html" in out.get("data", {}))
    sources = "\n".join(cell.source for cell in code)
    prohibited = [token for token in ("iid_holdout_features", "iid_holdout_targets", ".fit(", ".predict(", "joblib.load") if token in sources]
    report = {"status": "PASS" if not errors and images >= 11 and tables >= 10 and not prohibited else "FAIL", "created_at_utc": utc_now(), "notebook_path": NOTEBOOK.as_posix(), "attempt": 1, "code_cells": len(code), "error_count": len(errors), "inline_images": images, "inline_tables": tables, "fit_calls": 0, "preprocessor_fit_calls": 0, "prediction_calls": 0, "original_iid_accesses": 0, "prohibited_code_tokens": prohibited}
    atomic_json(root, REPORTS / "prompt5a_notebook_execution.json", report)
    runtime = read_json(root, REPORTS / "prompt5a_runtime.json"); runtime["notebook_seconds"] = time.perf_counter() - started; atomic_json(root, REPORTS / "prompt5a_runtime.json", runtime)
    if report["status"] != "PASS":
        raise RuntimeError("Artifact-only Prompt 5A notebook failed.")
    return report


def create_candidate(root: Path) -> dict[str, Any]:
    notebook = read_json(root, REPORTS / "prompt5a_notebook_execution.json")
    ledger = load_ledger(root); lock = validate_prediction_lock(root)
    if notebook["status"] != "PASS" or ledger["status"] != "EVALUATION_COMPLETE_ORIGINAL_IID_CLOSED":
        raise RuntimeError("Prompt 5A artifacts are not ready for a candidate handoff.")
    overall = pd.read_csv(root / REPORTS / "prompt5a_iid_overall_metrics.csv").set_index("model_id")
    six = read_json(root, REPORTS / "prompt5a_iid_six_condition_check.json")
    bootstrap = pd.read_csv(root / REPORTS / "prompt5a_iid_bootstrap.csv")
    manifest_path = root / REPORTS / "prompt5a_post_iid_snapshot_manifest.json"
    payload = {
        "status": "CANDIDATE_COMPLETE_AWAITING_REVIEW_AND_VERIFICATION", "created_at_utc": utc_now(), "authorization_id": AUTHORIZATION_ID,
        "freeze_identity": {"sha256": FREEZE_SHA, "primary_bundle_sha256": PRIMARY_SHA, "global_bundle_sha256": GLOBAL_SHA},
        "access_sequence": [item["event"] for item in ledger["events"]],
        "access_counts": {"feature_successful_content_reads": 1, "target_successful_content_reads": 1, "evaluated_models": 2, "prediction_rows_per_model": 75_000, "prediction_rows_total": 150_000},
        "prediction_lock_sha256": sha256(root / LOCK), "prediction_hashes": {key: value["sha256"] for key,value in lock["predictions"].items()},
        "target_snapshot_sha256": sha256(root / TARGET_SNAPSHOT), "evaluation_frame_sha256": sha256(root / EVALUATION_FRAME),
        "primary_overall_metrics": overall.loc[PRIMARY_MODEL_ID].to_dict(), "global_overall_metrics": overall.loc[GLOBAL_MODEL_ID].to_dict(),
        "six_condition_count": six["conditions_passed"], "six_condition_total": six["conditions_total"],
        "bootstrap_summary": bootstrap.to_dict(orient="records"),
        "report_hashes": {name: sha256(root / REPORTS / name) for name in ("prompt5a_iid_local_decile_metrics.csv", "prompt5a_development_frozen_band_metrics.csv", "prompt5a_iid_bootstrap.csv")},
        "post_iid_snapshot_manifest_sha256": sha256(manifest_path),
        "notebook_sha256": sha256(root / NOTEBOOK),
        "no_fit": True, "no_refit": True, "no_tuning": True, "no_model_change": True, "original_iid_files_closed": True,
    }
    atomic_json(root, REPORTS / "prompt5a_final_iid_evaluation_candidate.json", payload)
    return payload


def static_no_training_check(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prohibited = {"fit", "fit_transform", "partial_fit"}
    return not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in prohibited for node in ast.walk(tree))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "record-tests", "feature-predict", "target-evaluate", "notebook", "candidate"))
    parser.add_argument("--root", default=None)
    parser.add_argument("--passed", type=int, default=0)
    parser.add_argument("--failed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); root = root_path(args.root)
    if args.command == "preflight": result = preflight(root)
    elif args.command == "record-tests": result = record_focused_tests(root, args.passed, args.failed)
    elif args.command == "feature-predict": result = feature_predict(root)
    elif args.command == "target-evaluate": result = target_evaluate(root)
    elif args.command == "notebook": result = build_and_execute_notebook(root)
    else: result = create_candidate(root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
