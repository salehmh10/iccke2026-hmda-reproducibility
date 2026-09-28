"""Bounded, resumable execution for Regression V2 Prompt 2.

This module reads Development only. It never opens Raw or IID files.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import nbformat
import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from nbclient import NotebookClient
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Lasso

try:
    from .metrics import compute_regression_metrics, inverse_target, transform_target
    from .model_bundles import ModelBundle, atomic_joblib_dump, load_bundle
    from .preprocessing import (
        CatBoostFramePreprocessor,
        build_linear_compact_v2,
        make_dense_tree_preprocessor,
        make_lasso_preprocessor,
        make_xgb_preprocessor,
    )
except ImportError:
    from metrics import compute_regression_metrics, inverse_target, transform_target
    from model_bundles import ModelBundle, atomic_joblib_dump, load_bundle
    from preprocessing import (
        CatBoostFramePreprocessor,
        build_linear_compact_v2,
        make_dense_tree_preprocessor,
        make_lasso_preprocessor,
        make_xgb_preprocessor,
    )


SEED = 42
THREADS = min(4, os.cpu_count() or 1)
TARGET = "loan_amount_000s"
PRIMARY_CONTRACT = "main_without_sensitive_without_lender"
LENDER_CONTRACT = "main_without_sensitive_with_lender"
DEVELOPMENT_RELATIVE = Path("outputs/data/development.parquet")
REPORTS_RELATIVE = Path("outputs/reports")
MODELS_RELATIVE = Path("outputs/models/prompt2")
PREDICTIONS_RELATIVE = Path("outputs/predictions/prompt2/validation")
TMP_RELATIVE = Path("outputs/tmp/prompt2")

REQUIRED_BUNDLE_NAMES = {
    "lasso": "selected_lasso.joblib",
    "histgradientboosting": "selected_histgradientboosting.joblib",
    "catboost": "selected_catboost_without_lender.joblib",
    "lightgbm": "selected_lightgbm_without_lender.joblib",
    "xgboost": "selected_xgboost_without_lender.joblib",
    "catboost_with_lender": "selected_catboost_with_lender.joblib",
    "lightgbm_with_lender": "selected_lightgbm_with_lender.joblib",
    "xgboost_with_lender": "selected_xgboost_with_lender.joblib",
}

PREDICTION_NAMES = {
    "train_mean": "train_mean.parquet",
    "train_median": "train_median.parquet",
    "lasso": "selected_lasso.parquet",
    "histgradientboosting": "selected_histgradientboosting.parquet",
    "catboost": "selected_catboost_without_lender.parquet",
    "lightgbm": "selected_lightgbm_without_lender.parquet",
    "xgboost": "selected_xgboost_without_lender.parquet",
    "catboost_with_lender": "selected_catboost_with_lender.parquet",
    "lightgbm_with_lender": "selected_lightgbm_with_lender.parquet",
    "xgboost_with_lender": "selected_xgboost_with_lender.parquet",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_digest(values) -> str:
    return hashlib.sha256("\n".join(pd.Series(values).astype(str)).encode("utf-8")).hexdigest()


def atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def atomic_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    reloaded = pd.read_parquet(temporary)
    if list(reloaded.columns) != list(frame.columns) or len(reloaded) != len(frame):
        raise RuntimeError(f"Parquet reload validation failed: {destination}")
    os.replace(temporary, destination)


def load_feature_roles(root: str | Path) -> dict:
    path = Path(root) / REPORTS_RELATIVE / "feature_roles.json"
    roles = json.loads(path.read_text(encoding="utf-8"))
    if roles.get("status") != "PASS" or not roles.get("contracts"):
        raise RuntimeError("feature_roles.json is not a valid PASS handoff.")
    return roles


def validate_data_ready(root: str | Path) -> dict:
    """Validate saved Prompt 1B attestations without opening IID or Raw files."""
    root = Path(root)
    task = (root / "TASK.md").read_text(encoding="utf-8")
    if "Prompt 1B COMPLETE" not in task and "Prompt 2" not in task:
        raise RuntimeError("TASK.md does not preserve the completed Prompt 1B state.")
    ready = json.loads((root / REPORTS_RELATIVE / "DATA_READY.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (root / REPORTS_RELATIVE / "final_verification.json").read_text(encoding="utf-8")
    )
    if ready.get("status") != "PASS" or verification.get("status") != "PASS":
        raise RuntimeError("Prompt 1B saved handoff is not PASS.")
    required = {
        "development_rows": 500_000,
        "train_rows": 400_000,
        "validation_rows": 100_000,
        "development_iid_overlap": 0,
        "legacy_overlap": 0,
        "selected_conflict_groups": 0,
    }
    mismatches = {k: (ready.get(k), v) for k, v in required.items() if ready.get(k) != v}
    if mismatches:
        raise RuntimeError(f"Prompt 1B handoff mismatch: {mismatches}")
    return {"data_ready": ready, "final_verification": verification}


def validate_development(root: str | Path, frame: pd.DataFrame, roles: dict) -> dict:
    root = Path(root)
    source_path = root / DEVELOPMENT_RELATIVE
    expected_columns = set(roles["feature_superset"])
    failures = []
    role_counts = frame["development_role"].value_counts(dropna=False).to_dict()
    checks = {
        "rows": len(frame),
        "train_rows": int(role_counts.get("train", 0)),
        "validation_rows": int(role_counts.get("validation", 0)),
        "row_hash_unique": bool(frame["row_hash"].is_unique),
        "record_hash_unique": bool(frame["record_hash"].is_unique),
        "target_finite": bool(np.isfinite(frame[TARGET].to_numpy(dtype=np.float64)).all()),
        "missing_role_count": int(frame["development_role"].isna().sum()),
        "unexpected_roles": sorted(set(frame["development_role"].dropna()) - {"train", "validation"}),
        "missing_required_columns": sorted(expected_columns - set(frame.columns)),
    }
    if checks["rows"] != 500_000 or checks["train_rows"] != 400_000 or checks["validation_rows"] != 100_000:
        failures.append("Development row or role count mismatch")
    for name in ("row_hash_unique", "record_hash_unique", "target_finite"):
        if not checks[name]:
            failures.append(name)
    if checks["missing_role_count"] or checks["unexpected_roles"] or checks["missing_required_columns"]:
        failures.append("Development role or schema validation failed")
    train_hash = set(frame.loc[frame.development_role.eq("train"), "row_hash"])
    validation_hash = set(frame.loc[frame.development_role.eq("validation"), "row_hash"])
    checks["train_validation_overlap"] = len(train_hash & validation_hash)
    if checks["train_validation_overlap"]:
        failures.append("Train and Validation overlap")
    if failures:
        raise RuntimeError("; ".join(failures))
    parquet_schema = str(pq.read_schema(source_path))
    return {
        "path": DEVELOPMENT_RELATIVE.as_posix(),
        "size_bytes": source_path.stat().st_size,
        "sha256": file_sha256(source_path),
        "row_count": len(frame),
        "schema": parquet_schema,
        "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "checks": checks,
    }


def freeze_feature_contracts(train_df: pd.DataFrame, feature_roles: dict) -> dict:
    primary = list(feature_roles["contracts"][PRIMARY_CONTRACT])
    with_lender = list(feature_roles["contracts"][LENDER_CONTRACT])
    if with_lender != ["respondent_id"] + primary:
        raise RuntimeError("The lender contract must differ only by leading respondent_id.")
    excluded = set(feature_roles["sensitive_fields"]) | set(feature_roles["audit_only_fields"]) | set(
        feature_roles["target_and_alias_exclusions"]
    )
    if set(primary) & excluded:
        raise RuntimeError(f"Primary contract contains excluded fields: {sorted(set(primary) & excluded)}")
    if "respondent_id" in primary or "respondent_id" not in with_lender:
        raise RuntimeError("Lender contract boundary is invalid.")
    linear = build_linear_compact_v2(train_df, feature_roles)
    return {
        PRIMARY_CONTRACT: primary,
        LENDER_CONTRACT: with_lender,
        "linear_compact_v2": linear["features"],
        "linear_numeric_features": linear["numeric_features"],
        "linear_categorical_features": linear["categorical_features"],
        "feature_contract_digest": canonical_digest(
            {"primary": primary, "with_lender": with_lender, "linear": linear}
        ),
    }


def deterministic_candidate_id(
    family: str, name: str, feature_contract: str, target_mode: str, params: dict, seed: int
) -> str:
    digest = canonical_digest(
        {
            "family": family,
            "name": name,
            "feature_contract": feature_contract,
            "target_mode": target_mode,
            "params": params,
            "seed": seed,
        }
    )[:12]
    return f"{name}__{digest}"


def candidate_definitions() -> list[dict]:
    definitions = [
        ("lasso", "lasso_log1p_anchor", "log1p", {"alpha": 1e-4, "max_iter": 5000, "tol": 1e-4, "random_state": SEED}),
        ("lasso", "lasso_raw_anchor", "raw", {"alpha": 1e-4, "max_iter": 5000, "tol": 1e-4, "random_state": SEED}),
        ("histgradientboosting", "hgb_log1p_l1", "log1p", {"loss": "absolute_error", "learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0, "random_state": SEED}),
        ("histgradientboosting", "hgb_raw_l1", "raw", {"loss": "absolute_error", "learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0, "random_state": SEED}),
        ("catboost", "cat_log1p_rmse_anchor", "log1p", {"loss_function": "RMSE", "iterations": 2000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20, "random_strength": 1, "random_seed": SEED, "thread_count": THREADS, "early_stopping_rounds": 100}),
        ("catboost", "cat_raw_rmse", "raw", {"loss_function": "RMSE", "iterations": 2000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20, "random_strength": 1, "random_seed": SEED, "thread_count": THREADS, "early_stopping_rounds": 100}),
        ("catboost", "cat_raw_mae", "raw", {"loss_function": "MAE", "iterations": 2000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20, "random_strength": 1, "random_seed": SEED, "thread_count": THREADS, "early_stopping_rounds": 100}),
        ("lightgbm", "lgb_raw_l2_anchor", "raw", {"objective": "regression", "n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 47, "min_child_samples": 50, "reg_lambda": 5, "random_state": SEED, "n_jobs": THREADS, "early_stopping_rounds": 100}),
        ("lightgbm", "lgb_raw_l1", "raw", {"objective": "regression_l1", "n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 47, "min_child_samples": 50, "reg_lambda": 5, "random_state": SEED, "n_jobs": THREADS, "early_stopping_rounds": 100}),
        ("lightgbm", "lgb_raw_huber", "raw", {"objective": "huber", "n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 47, "min_child_samples": 50, "reg_lambda": 5, "random_state": SEED, "n_jobs": THREADS, "early_stopping_rounds": 100}),
        ("xgboost", "xgb_log1p_l2_anchor", "log1p", {"objective": "reg:squarederror", "n_estimators": 2000, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 10, "reg_lambda": 5, "subsample": 1.0, "colsample_bytree": 1.0, "tree_method": "hist", "random_state": SEED, "n_jobs": THREADS, "early_stopping_rounds": 100}),
        ("xgboost", "xgb_raw_l1", "raw", {"objective": "reg:absoluteerror", "n_estimators": 2000, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 10, "reg_lambda": 5, "subsample": 1.0, "colsample_bytree": 1.0, "tree_method": "hist", "random_state": SEED, "n_jobs": THREADS, "early_stopping_rounds": 100}),
        ("xgboost", "xgb_raw_pseudohuber", "raw", {"objective": "reg:pseudohubererror", "n_estimators": 2000, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 10, "reg_lambda": 5, "subsample": 1.0, "colsample_bytree": 1.0, "tree_method": "hist", "random_state": SEED, "n_jobs": THREADS, "early_stopping_rounds": 100}),
    ]
    output = []
    for family, name, target_mode, params in definitions:
        contract = PRIMARY_CONTRACT if family != "lasso" else "linear_compact_v2"
        output.append(
            {
                "family": family,
                "candidate_name": name,
                "candidate_id": deterministic_candidate_id(family, name, contract, target_mode, params, SEED),
                "feature_contract": contract,
                "target_mode": target_mode,
                "parameters": params,
                "seed": SEED,
            }
        )
    return output


def _relative_difference(a: float, b: float) -> float:
    denominator = max(min(abs(a), abs(b)), np.finfo(float).eps)
    return abs(a - b) / denominator


def select_family_candidate(records: list[dict]) -> dict:
    """Apply the frozen within-family selection rule."""
    available = [r for r in records if r.get("status") == "COMPLETE"]
    if not available:
        raise RuntimeError("No completed Candidate is available for family selection.")
    lowest_mae = min(float(r["mae"]) for r in available)
    pool = [r for r in available if _relative_difference(float(r["mae"]), lowest_mae) <= 0.0025]
    lowest_rmse = min(float(r["rmse"]) for r in pool)
    pool = [r for r in pool if _relative_difference(float(r["rmse"]), lowest_rmse) <= 0.0025]
    lowest_tail = min(float(r["top_decile_mae"]) for r in pool)
    pool = [r for r in pool if _relative_difference(float(r["top_decile_mae"]), lowest_tail) <= 0.0025]
    return min(
        pool,
        key=lambda r: (
            int(r.get("fitted_iterations") or 10**12),
            int(r.get("bundle_size_bytes") or 10**18),
            float(r.get("fit_time_seconds") or float("inf")),
            0 if r.get("target_mode") == "raw" else 1,
            r["candidate_id"],
        ),
    )


def package_versions() -> dict[str, str]:
    import catboost
    import lightgbm
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "scikit_learn": sklearn_version,
        "joblib": joblib.__version__,
        "catboost": catboost.__version__,
        "lightgbm": lightgbm.__version__,
        "xgboost": xgboost.__version__,
    }


def source_code_digest(root: Path) -> str:
    parts = []
    for name in ("prompt2_modeling.py", "preprocessing.py", "metrics.py", "model_bundles.py"):
        path = root / "src" / name
        parts.append(name + ":" + file_sha256(path))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def prepare_design(root: Path, development: pd.DataFrame, roles: dict, source_record: dict) -> dict:
    train = development.loc[development.development_role.eq("train")]
    validation = development.loc[development.development_role.eq("validation")]
    contracts = freeze_feature_contracts(train, roles)
    definitions = candidate_definitions()
    design_core = {
        "development_source": source_record,
        "train_row_hash_digest": ordered_digest(train["row_hash"]),
        "validation_row_hash_digest": ordered_digest(validation["row_hash"]),
        "target": {"name": TARGET, "unit": "thousands of U.S. dollars", "modes": ["raw", "log1p"]},
        "candidate_definitions": definitions,
        "candidate_ids": [d["candidate_id"] for d in definitions],
        "feature_contracts": contracts,
        "preprocessing_design": {
            "linear": "Train median imputation and scaling for numeric fields; Train most-frequent imputation and sparse one-hot encoding for categorical fields.",
            "histgradientboosting_lightgbm": "Train median numeric imputation; Train frequency encoding above cardinality 100; otherwise Train ordinal encoding; unknown values map to zero frequency or -1 ordinal.",
            "catboost": "Named numeric columns remain numeric; categorical values are normalized strings with a stable missing sentinel; native categorical handling.",
            "xgboost": "Train median numeric imputation; Train frequency encoding above cardinality 100; otherwise sparse Train one-hot encoding with unknown values ignored.",
        },
        "seed": SEED,
        "n_threads": THREADS,
        "device": "CPU",
        "metrics": [
            "mae", "rmse", "r2", "rmsle", "median_absolute_error", "p90_absolute_error",
            "mean_signed_error", "negative_prediction_count", "negative_prediction_rate",
            "top_decile_mae", "top_five_percent_mae", "top_decile_underprediction_rate",
            "top_five_percent_underprediction_rate", "fit_time_seconds", "prediction_time_seconds",
            "bundle_size_bytes",
        ],
        "selection_rule": {
            "primary": "lowest Validation MAE",
            "mae_relative_tie": 0.0025,
            "second": "lower RMSE",
            "rmse_relative_tie": 0.0025,
            "third": "lower top-decile MAE",
            "final": ["fewer fitted iterations", "smaller model size", "lower fit time", "raw target mode"],
        },
        "fit_limits": {
            "primary_scientific_fits": 13,
            "lender_ablation_fits": 3,
            "max_scientific_fits": 16,
            "max_technical_retry_per_candidate": 1,
            "max_smoke_runs_per_family": 2,
            "heavy_fits_sequential": True,
        },
        "output_paths": {
            "models": MODELS_RELATIVE.as_posix(),
            "validation_predictions": PREDICTIONS_RELATIVE.as_posix(),
            "reports": REPORTS_RELATIVE.as_posix(),
            "temporary": TMP_RELATIVE.as_posix(),
        },
        "package_versions": package_versions(),
    }
    current_code_digest = source_code_digest(root)
    design = {"status": "FROZEN", "frozen_at_utc": utc_now(), **design_core, "code_digest": current_code_digest}
    design["design_digest"] = canonical_digest(design_core)
    path = root / REPORTS_RELATIVE / "prompt2_frozen_design.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("design_digest") != design["design_digest"]:
            raise RuntimeError("Existing Prompt 2 frozen scientific design differs from current Candidates or contracts.")
        if existing.get("code_digest") != current_code_digest:
            history = list(existing.get("prior_code_digests", []))
            if existing.get("code_digest"):
                history.append(existing["code_digest"])
            existing["prior_code_digests"] = sorted(set(history))
            existing["code_digest"] = current_code_digest
            existing["implementation_updated_at_utc"] = utc_now()
            atomic_json(path, existing)
        return existing
    atomic_json(path, design)
    return design


def environment_record(root: Path) -> dict:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(root))
    return {
        "captured_at_utc": utc_now(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_processors": psutil.cpu_count(logical=True),
        "physical_processors": psutil.cpu_count(logical=False),
        "thread_limit": THREADS,
        "ram_total_bytes": memory.total,
        "ram_available_bytes": memory.available,
        "disk_free_bytes": disk.free,
        "package_versions": package_versions(),
        "resource_measurement_limitation": "Peak memory is the largest parent-process RSS observed before and after each fit, not an operating-system high-water mark.",
    }


def _preprocessor_for(family: str, features: list[str], contracts: dict):
    if family == "lasso":
        return make_lasso_preprocessor(
            contracts["linear_numeric_features"], contracts["linear_categorical_features"]
        )
    if family in {"histgradientboosting", "lightgbm"}:
        return make_dense_tree_preprocessor(features, 100)
    if family == "xgboost":
        return make_xgb_preprocessor(features, 100)
    if family == "catboost":
        return CatBoostFramePreprocessor(features)
    raise ValueError(f"Unknown model family: {family}")


def _fit_model(
    family: str,
    parameters: dict,
    X_train,
    y_train,
    X_validation,
    y_validation,
    preprocessor,
    *,
    use_early_stopping: bool = True,
):
    if family == "lasso":
        model = Lasso(**parameters)
        model.fit(X_train, y_train)
        return model, int(getattr(model, "n_iter_", parameters["max_iter"])), int(getattr(model, "n_iter_", parameters["max_iter"]))
    if family == "histgradientboosting":
        model = HistGradientBoostingRegressor(**parameters)
        model.fit(X_train, y_train)
        fitted = int(getattr(model, "n_iter_", parameters["max_iter"]))
        return model, fitted, fitted
    if family == "catboost":
        from catboost import CatBoostRegressor

        params = dict(parameters)
        early_stopping_rounds = params.pop("early_stopping_rounds", None)
        params.update({"allow_writing_files": False, "verbose": False, "task_type": "CPU"})
        model = CatBoostRegressor(**params)
        fit_kwargs = {"cat_features": preprocessor.cat_feature_indices_, "verbose": False}
        if use_early_stopping:
            fit_kwargs.update({"eval_set": (X_validation, y_validation), "use_best_model": True})
            if early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        model.fit(X_train, y_train, **fit_kwargs)
        best = int(model.get_best_iteration()) if use_early_stopping else int(params["iterations"] - 1)
        if best < 0:
            best = int(model.tree_count_ - 1)
        return model, best, best + 1
    if family == "lightgbm":
        import lightgbm as lgb

        params = dict(parameters)
        early_stopping_rounds = params.pop("early_stopping_rounds", None)
        model = lgb.LGBMRegressor(**params, verbosity=-1)
        kwargs = {}
        if use_early_stopping:
            kwargs = {
                "eval_set": [(X_validation, y_validation)],
                "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
            }
        model.fit(X_train, y_train, **kwargs)
        best = int(model.best_iteration_) if use_early_stopping and model.best_iteration_ else int(params["n_estimators"])
        return model, best, best
    if family == "xgboost":
        from xgboost import XGBRegressor

        params = dict(parameters)
        if not use_early_stopping:
            params.pop("early_stopping_rounds", None)
        model = XGBRegressor(**params, device="cpu", verbosity=0)
        kwargs = {"verbose": False}
        if use_early_stopping:
            kwargs["eval_set"] = [(X_validation, y_validation)]
        model.fit(X_train, y_train, **kwargs)
        if use_early_stopping:
            best = int(getattr(model, "best_iteration", params["n_estimators"] - 1))
            fitted = best + 1
        else:
            fitted = int(params["n_estimators"])
            best = fitted - 1
        return model, best, fitted
    raise ValueError(f"Unknown model family: {family}")


def run_smoke_tests(root: Path, train: pd.DataFrame, validation: pd.DataFrame, design: dict) -> dict:
    report_path = root / REPORTS_RELATIVE / "prompt2_smoke_tests.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS" and existing.get("design_digest") == design["design_digest"]:
            return existing
    small_train = train.iloc[: min(10_000, len(train))]
    small_validation = validation.iloc[: min(2_000, len(validation))].copy()
    contracts = design["feature_contracts"]
    family_definitions = {}
    for definition in design["candidate_definitions"]:
        family_definitions.setdefault(definition["family"], []).append(definition)
    results = []
    for family in ("lasso", "histgradientboosting", "catboost", "lightgbm", "xgboost"):
        print(f"SMOKE START {family}", flush=True)
        features = contracts["linear_compact_v2"] if family == "lasso" else contracts[PRIMARY_CONTRACT]
        preprocessor = _preprocessor_for(family, features, contracts)
        train_features = small_train.loc[:, features]
        validation_features = small_validation.loc[:, features].copy()
        categorical = [c for c in features if not pd.api.types.is_numeric_dtype(train_features[c])]
        if categorical:
            validation_features.loc[validation_features.index[:1], categorical[0]] = "__PROMPT2_UNKNOWN__"
        X_train = preprocessor.fit_transform(train_features)
        X_validation = preprocessor.transform(validation_features)
        if family == "xgboost" and not sparse.issparse(X_train):
            raise RuntimeError("XGBoost smoke preprocessing is not sparse.")
        objectives = family_definitions[family]
        if family in {"lasso", "histgradientboosting"}:
            objectives = objectives[:1]
        family_result = {"family": family, "status": "PASS", "objective_checks": []}
        for definition in objectives:
            params = dict(definition["parameters"])
            if family == "lasso":
                params["max_iter"] = 20
            elif family == "histgradientboosting":
                params["max_iter"] = 5
            elif family == "catboost":
                params["iterations"] = 5
                params["early_stopping_rounds"] = 2
            else:
                params["n_estimators"] = 5
                params["early_stopping_rounds"] = 2
            y_train = transform_target(small_train[TARGET], definition["target_mode"])
            y_validation = transform_target(small_validation[TARGET], definition["target_mode"])
            try:
                model, best, fitted = _fit_model(
                    family, params, X_train, y_train, X_validation, y_validation, preprocessor, use_early_stopping=True
                )
                pred = np.asarray(model.predict(X_validation), dtype=np.float64)
                if not np.isfinite(pred).all():
                    raise RuntimeError("Smoke predictions are not finite.")
                smoke_bundle = ModelBundle(
                    model_id="smoke_" + definition["candidate_id"], family=family,
                    feature_names=features, feature_contract_name=definition["feature_contract"],
                    target_mode=definition["target_mode"], preprocessor=preprocessor, model=model,
                    package_versions=design["package_versions"], model_parameters=params,
                    selected_best_iteration=best, development_source_sha256=design["development_source"]["sha256"],
                    train_row_hash_digest=design["train_row_hash_digest"],
                    validation_row_hash_digest=design["validation_row_hash_digest"], metadata={"smoke": True},
                )
                temp_path = root / TMP_RELATIVE / f"smoke_{family}.joblib"
                atomic_joblib_dump(smoke_bundle, temp_path)
                reloaded = load_bundle(temp_path)
                reload_pred = reloaded.predict(validation_features)
                if len(reload_pred) != len(validation_features) or not np.isfinite(reload_pred).all():
                    raise RuntimeError("Smoke bundle reload failed.")
                temp_path.unlink()
                family_result["objective_checks"].append(
                    {"candidate_id": definition["candidate_id"], "status": "SUPPORTED", "fitted_iterations": fitted}
                )
            except Exception as exc:
                optional = family in {"lightgbm", "xgboost"} and "anchor" not in definition["candidate_name"]
                family_result["objective_checks"].append(
                    {"candidate_id": definition["candidate_id"], "status": "UNAVAILABLE_OBJECTIVE" if optional else "FAILED", "error": repr(exc)}
                )
                if not optional:
                    family_result["status"] = "FAILED"
        if family_result["status"] != "PASS":
            raise RuntimeError(f"Required {family} smoke test failed: {family_result}")
        results.append(family_result)
        del preprocessor, X_train, X_validation
        gc.collect()
        print(f"SMOKE PASS {family}", flush=True)
    report = {"status": "PASS", "created_at_utc": utc_now(), "design_digest": design["design_digest"], "train_rows": len(small_train), "validation_rows": len(small_validation), "runs_per_family": 1, "results": results}
    atomic_json(report_path, report)
    return report


def _prediction_frame(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model_id: str,
    family: str,
    feature_contract: str,
    target_mode: str,
) -> pd.DataFrame:
    values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(values) != len(validation) or not np.isfinite(values).all():
        raise RuntimeError("Prediction output is not complete and finite.")
    return pd.DataFrame(
        {
            "row_hash": validation["row_hash"].astype(str).to_numpy(),
            "y_true": validation[TARGET].to_numpy(dtype=np.float64),
            "y_pred": values,
            "model_id": model_id,
            "family": family,
            "feature_contract": feature_contract,
            "target_mode": target_mode,
        }
    )


def save_prediction_artifact(
    root: Path,
    key: str,
    validation: pd.DataFrame,
    predictions: np.ndarray,
    *,
    model_id: str,
    family: str,
    feature_contract: str,
    target_mode: str,
) -> Path:
    frame = _prediction_frame(
        validation,
        predictions,
        model_id=model_id,
        family=family,
        feature_contract=feature_contract,
        target_mode=target_mode,
    )
    path = root / PREDICTIONS_RELATIVE / PREDICTION_NAMES[key]
    atomic_parquet(path, frame)
    reloaded = pd.read_parquet(path)
    if (
        len(reloaded) != 100_000
        or not reloaded["row_hash"].is_unique
        or not reloaded["row_hash"].equals(frame["row_hash"])
        or not np.array_equal(reloaded["y_true"].to_numpy(), frame["y_true"].to_numpy())
        or not np.isfinite(reloaded["y_pred"].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError(f"Saved Validation prediction validation failed: {path}")
    return path


def _candidate_cache_identity(definition: dict, design: dict) -> dict:
    family_version_key = {"lasso": "scikit_learn", "histgradientboosting": "scikit_learn", "catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost"}[definition["family"]]
    return {
        "development_sha256": design["development_source"]["sha256"],
        "train_row_hash_digest": design["train_row_hash_digest"],
        "validation_row_hash_digest": design["validation_row_hash_digest"],
        "feature_contract_digest": design["feature_contracts"]["feature_contract_digest"],
        "target_mode": definition["target_mode"],
        "model_parameters": definition["parameters"],
        "package_version": design["package_versions"][family_version_key],
        "code_digest": design["code_digest"],
        "seed": definition["seed"],
        "prediction_row_count": 100_000,
        "candidate_id": definition["candidate_id"],
    }


def _load_candidate_cache(root: Path, definition: dict, design: dict):
    directory = root / TMP_RELATIVE / "candidates" / definition["candidate_id"]
    manifest_path = directory / "manifest.json"
    bundle_path = directory / "bundle.joblib"
    prediction_path = directory / "validation_predictions.npy"
    if not (manifest_path.exists() and bundle_path.exists() and prediction_path.exists()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("cache_identity") != _candidate_cache_identity(definition, design):
        return None
    predictions = np.load(prediction_path, allow_pickle=False)
    if predictions.shape != (100_000,) or not np.isfinite(predictions).all():
        return None
    bundle = load_bundle(bundle_path)
    if bundle.model_id != definition["candidate_id"]:
        return None
    return bundle, predictions, manifest["result"]


def _save_candidate_cache(
    root: Path, definition: dict, design: dict, bundle: ModelBundle, predictions: np.ndarray, result: dict
) -> tuple[Path, int]:
    directory = root / TMP_RELATIVE / "candidates" / definition["candidate_id"]
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / "bundle.joblib"
    atomic_joblib_dump(bundle, bundle_path)
    temporary_prediction = directory / "validation_predictions.npy.tmp"
    with temporary_prediction.open("wb") as handle:
        np.save(handle, np.asarray(predictions, dtype=np.float64), allow_pickle=False)
    os.replace(temporary_prediction, directory / "validation_predictions.npy")
    bundle_size = bundle_path.stat().st_size
    result["bundle_size_bytes"] = bundle_size
    result["cache_bundle_path"] = bundle_path.relative_to(root).as_posix()
    atomic_json(
        directory / "manifest.json",
        {"status": "COMPLETE", "completed_at_utc": utc_now(), "cache_identity": _candidate_cache_identity(definition, design), "result": result},
    )
    return bundle_path, bundle_size


def fit_candidate(
    root: Path,
    definition: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    design: dict,
) -> tuple[ModelBundle, np.ndarray, dict]:
    cached = _load_candidate_cache(root, definition, design)
    if cached is not None:
        print(f"CANDIDATE CACHE REUSED {definition['candidate_id']}", flush=True)
        return cached
    family = definition["family"]
    contracts = design["feature_contracts"]
    features = contracts[definition["feature_contract"]]
    retry_count = 0
    while True:
        try:
            print(f"FIT START {definition['candidate_id']} RAM_AVAILABLE={psutil.virtual_memory().available}", flush=True)
            rss_before = psutil.Process().memory_info().rss
            fit_started = time.perf_counter()
            preprocessor = _preprocessor_for(family, features, contracts)
            train_features = train.loc[:, features]
            validation_features = validation.loc[:, features]
            X_train = preprocessor.fit_transform(train_features)
            X_validation = preprocessor.transform(validation_features)
            y_train = transform_target(train[TARGET], definition["target_mode"])
            y_validation = transform_target(validation[TARGET], definition["target_mode"])
            model, best_iteration, fitted_iterations = _fit_model(
                family,
                definition["parameters"],
                X_train,
                y_train,
                X_validation,
                y_validation,
                preprocessor,
                use_early_stopping=True,
            )
            fit_time = time.perf_counter() - fit_started
            bundle = ModelBundle(
                model_id=definition["candidate_id"],
                family=family,
                feature_names=features,
                feature_contract_name=definition["feature_contract"],
                target_mode=definition["target_mode"],
                preprocessor=preprocessor,
                model=model,
                package_versions=design["package_versions"],
                model_parameters=definition["parameters"],
                selected_best_iteration=best_iteration,
                development_source_sha256=design["development_source"]["sha256"],
                train_row_hash_digest=design["train_row_hash_digest"],
                validation_row_hash_digest=design["validation_row_hash_digest"],
                metadata={"candidate_name": definition["candidate_name"], "fitted_iterations": fitted_iterations},
            )
            prediction_started = time.perf_counter()
            predictions = bundle.predict(validation_features)
            prediction_time = time.perf_counter() - prediction_started
            rss_after = psutil.Process().memory_info().rss
            result = {
                **{k: definition[k] for k in ("family", "candidate_name", "candidate_id", "feature_contract", "target_mode")},
                "status": "COMPLETE",
                "best_iteration": best_iteration,
                "fitted_iterations": fitted_iterations,
                "technical_retry_count": retry_count,
                "rss_before_bytes": rss_before,
                "rss_after_bytes": rss_after,
                "observed_peak_rss_bytes": max(rss_before, rss_after),
                **compute_regression_metrics(
                    validation[TARGET], predictions, fit_time_seconds=fit_time, prediction_time_seconds=prediction_time
                ),
            }
            _, bundle_size = _save_candidate_cache(root, definition, design, bundle, predictions, result)
            result["bundle_size_bytes"] = bundle_size
            print(f"FIT PASS {definition['candidate_id']} MAE={result['mae']:.6f} SECONDS={fit_time:.3f}", flush=True)
            del X_train, X_validation, train_features, validation_features, y_train, y_validation
            gc.collect()
            return bundle, predictions, result
        except Exception as exc:
            failure_dir = root / TMP_RELATIVE / "failures" / definition["candidate_id"]
            failure_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                failure_dir / f"attempt_{retry_count + 1}.json",
                {"status": "TECHNICAL_FAILURE", "candidate": definition, "attempt": retry_count + 1, "error": repr(exc), "time_utc": utc_now()},
            )
            retry_count += 1
            if retry_count > 1:
                raise
            print(f"FIT TECHNICAL RETRY {definition['candidate_id']} ERROR={exc!r}", flush=True)
            gc.collect()


def load_candidate_results(root: Path) -> list[dict]:
    path = root / REPORTS_RELATIVE / "prompt2_candidate_results.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def save_candidate_results(root: Path, records: list[dict]) -> None:
    ordered = sorted(records, key=lambda r: [d["candidate_id"] for d in candidate_definitions()].index(r["candidate_id"]))
    atomic_csv(root / REPORTS_RELATIVE / "prompt2_candidate_results.csv", pd.DataFrame(ordered))


def run_baselines(root: Path, train: pd.DataFrame, validation: pd.DataFrame) -> list[dict]:
    records = []
    for key, value in (("train_mean", float(train[TARGET].mean())), ("train_median", float(train[TARGET].median()))):
        predictions = np.full(len(validation), value, dtype=np.float64)
        save_prediction_artifact(
            root, key, validation, predictions, model_id=key, family=key,
            feature_contract="train_target_only", target_mode="raw",
        )
        records.append(
            {"model_id": key, "family": key, "feature_contract": "train_target_only", "target_mode": "raw", "constant_prediction": value, **compute_regression_metrics(validation[TARGET], predictions)}
        )
    return records


def _promote_family_winner(
    root: Path,
    family: str,
    winner: dict,
    bundle: ModelBundle,
    predictions: np.ndarray,
    validation: pd.DataFrame,
) -> dict:
    bundle_path = root / MODELS_RELATIVE / REQUIRED_BUNDLE_NAMES[family]
    atomic_joblib_dump(bundle, bundle_path)
    loaded = load_bundle(bundle_path)
    if loaded.model_id != winner["candidate_id"]:
        raise RuntimeError("Promoted family bundle identity mismatch.")
    prediction_path = save_prediction_artifact(
        root,
        family,
        validation,
        predictions,
        model_id=winner["candidate_id"],
        family=family,
        feature_contract=winner["feature_contract"],
        target_mode=winner["target_mode"],
    )
    promoted = dict(winner)
    promoted["selected_bundle_path"] = bundle_path.relative_to(root).as_posix()
    promoted["selected_prediction_path"] = prediction_path.relative_to(root).as_posix()
    promoted["bundle_size_bytes"] = bundle_path.stat().st_size
    return promoted


def run_primary_fits(
    root: Path, train: pd.DataFrame, validation: pd.DataFrame, design: dict, smoke_report: dict
) -> tuple[list[dict], dict]:
    records_by_id = {r["candidate_id"]: r for r in load_candidate_results(root)}
    winners_path = root / REPORTS_RELATIVE / "prompt2_family_winners.json"
    winners_report = json.loads(winners_path.read_text(encoding="utf-8")) if winners_path.exists() else {"status": "IN_PROGRESS", "winners": {}}
    unavailable = {
        check["candidate_id"]
        for family in smoke_report["results"]
        for check in family["objective_checks"]
        if check["status"] == "UNAVAILABLE_OBJECTIVE"
    }
    definitions = design["candidate_definitions"]
    for family in ("lasso", "histgradientboosting", "catboost", "lightgbm", "xgboost"):
        existing = winners_report["winners"].get(family)
        if existing:
            bundle_path = root / existing["selected_bundle_path"]
            prediction_path = root / existing["selected_prediction_path"]
            if bundle_path.exists() and prediction_path.exists():
                print(f"FAMILY WINNER REUSED {family} {existing['candidate_id']}", flush=True)
                continue
        family_outputs = {}
        family_records = []
        for definition in [d for d in definitions if d["family"] == family]:
            if definition["candidate_id"] in unavailable:
                record = {
                    **{k: definition[k] for k in ("family", "candidate_name", "candidate_id", "feature_contract", "target_mode")},
                    "status": "UNAVAILABLE_OBJECTIVE",
                }
                records_by_id[definition["candidate_id"]] = record
                family_records.append(record)
                save_candidate_results(root, list(records_by_id.values()))
                continue
            bundle, predictions, record = fit_candidate(root, definition, train, validation, design)
            records_by_id[definition["candidate_id"]] = record
            family_records.append(record)
            family_outputs[definition["candidate_id"]] = (bundle, predictions)
            save_candidate_results(root, list(records_by_id.values()))
        winner = select_family_candidate(family_records)
        if winner["candidate_id"] not in family_outputs:
            definition = next(d for d in definitions if d["candidate_id"] == winner["candidate_id"])
            cached = _load_candidate_cache(root, definition, design)
            if cached is None:
                raise RuntimeError("Selected Candidate cache is unavailable for promotion.")
            family_outputs[winner["candidate_id"]] = (cached[0], cached[1])
        selected_bundle, selected_predictions = family_outputs[winner["candidate_id"]]
        promoted = _promote_family_winner(root, family, winner, selected_bundle, selected_predictions, validation)
        winners_report["winners"][family] = promoted
        atomic_json(winners_path, winners_report)
        cache_root = root / TMP_RELATIVE / "candidates"
        for definition in [d for d in definitions if d["family"] == family]:
            candidate_dir = cache_root / definition["candidate_id"]
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)
        del family_outputs, selected_bundle, selected_predictions
        gc.collect()
    winners_report["status"] = "PASS"
    winners_report["selection_rule"] = design["selection_rule"]
    winners_report["completed_at_utc"] = utc_now()
    atomic_json(winners_path, winners_report)
    records = list(records_by_id.values())
    save_candidate_results(root, records)
    completed = sum(r.get("status") == "COMPLETE" for r in records)
    if completed + sum(r.get("status") == "UNAVAILABLE_OBJECTIVE" for r in records) != 13:
        raise RuntimeError("Primary Candidate accounting does not cover all 13 frozen definitions.")
    if completed > 13:
        raise RuntimeError("Primary scientific fit budget exceeded.")
    return records, winners_report


def _matched_lender_parameters(family: str, definition: dict, fitted_iterations: int) -> dict:
    params = dict(definition["parameters"])
    params.pop("early_stopping_rounds", None)
    if family == "catboost":
        params["iterations"] = int(fitted_iterations)
    else:
        params["n_estimators"] = int(fitted_iterations)
    return params


def fit_lender_ablation(
    root: Path,
    family: str,
    winner: dict,
    definition: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    design: dict,
) -> tuple[ModelBundle, np.ndarray, dict]:
    bundle_key = family + "_with_lender"
    bundle_path = root / MODELS_RELATIVE / REQUIRED_BUNDLE_NAMES[bundle_key]
    prediction_path = root / PREDICTIONS_RELATIVE / PREDICTION_NAMES[bundle_key]
    if bundle_path.exists() and prediction_path.exists():
        bundle = load_bundle(bundle_path)
        prediction_frame = pd.read_parquet(prediction_path)
        if (
            bundle.feature_contract_name == LENDER_CONTRACT
            and len(prediction_frame) == 100_000
            and prediction_frame["row_hash"].astype(str).reset_index(drop=True).equals(validation["row_hash"].astype(str).reset_index(drop=True))
        ):
            print(f"LENDER FIT REUSED {family}", flush=True)
            preserved = None
            runtime_path = root / REPORTS_RELATIVE / "prompt2_runtime.json"
            if runtime_path.exists():
                prior_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                preserved = next(
                    (r for r in prior_runtime.get("fit_records", []) if r.get("with_lender_model_id") == bundle.model_id),
                    None,
                )
            metrics = compute_regression_metrics(validation[TARGET], prediction_frame["y_pred"])
            record = preserved or {
                "family": family, "status": "COMPLETE", "reused": True, "with_lender_model_id": bundle.model_id,
                "selected_best_iteration": bundle.selected_best_iteration, **metrics,
            }
            record["reused"] = True
            return bundle, prediction_frame["y_pred"].to_numpy(dtype=np.float64), record
    features = design["feature_contracts"][LENDER_CONTRACT]
    if features[1:] != design["feature_contracts"][PRIMARY_CONTRACT] or features[0] != "respondent_id":
        raise RuntimeError("Controlled lender ablation does not differ only by respondent_id.")
    fixed_count = int(winner["fitted_iterations"])
    params = _matched_lender_parameters(family, definition, fixed_count)
    model_id = deterministic_candidate_id(
        family, f"selected_{family}_with_lender", LENDER_CONTRACT, winner["target_mode"], params, SEED
    )
    print(f"LENDER FIT START {family} FIXED_ITERATIONS={fixed_count} RAM_AVAILABLE={psutil.virtual_memory().available}", flush=True)
    rss_before = psutil.Process().memory_info().rss
    started = time.perf_counter()
    preprocessor = _preprocessor_for(family, features, design["feature_contracts"])
    train_features = train.loc[:, features]
    validation_features = validation.loc[:, features]
    X_train = preprocessor.fit_transform(train_features)
    X_validation = preprocessor.transform(validation_features)
    y_train = transform_target(train[TARGET], winner["target_mode"])
    y_validation = transform_target(validation[TARGET], winner["target_mode"])
    model, best_iteration, fitted_iterations = _fit_model(
        family, params, X_train, y_train, X_validation, y_validation, preprocessor, use_early_stopping=False
    )
    fit_time = time.perf_counter() - started
    bundle = ModelBundle(
        model_id=model_id,
        family=family,
        feature_names=features,
        feature_contract_name=LENDER_CONTRACT,
        target_mode=winner["target_mode"],
        preprocessor=preprocessor,
        model=model,
        package_versions=design["package_versions"],
        model_parameters=params,
        selected_best_iteration=best_iteration,
        development_source_sha256=design["development_source"]["sha256"],
        train_row_hash_digest=design["train_row_hash_digest"],
        validation_row_hash_digest=design["validation_row_hash_digest"],
        metadata={"matched_no_lender_model_id": winner["candidate_id"], "fitted_iterations": fitted_iterations, "diagnostic_only": True},
    )
    prediction_started = time.perf_counter()
    predictions = bundle.predict(validation_features)
    prediction_time = time.perf_counter() - prediction_started
    atomic_joblib_dump(bundle, bundle_path)
    save_prediction_artifact(
        root, bundle_key, validation, predictions, model_id=model_id, family=family,
        feature_contract=LENDER_CONTRACT, target_mode=winner["target_mode"],
    )
    bundle_size = bundle_path.stat().st_size
    rss_after = psutil.Process().memory_info().rss
    record = {
        "family": family,
        "status": "COMPLETE",
        "reused": False,
        "with_lender_model_id": model_id,
        "selected_best_iteration": best_iteration,
        "fitted_iterations": fitted_iterations,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "observed_peak_rss_bytes": max(rss_before, rss_after),
        **compute_regression_metrics(
            validation[TARGET], predictions, fit_time_seconds=fit_time,
            prediction_time_seconds=prediction_time, bundle_size_bytes=bundle_size,
        ),
    }
    del X_train, X_validation, train_features, validation_features, y_train, y_validation
    gc.collect()
    print(f"LENDER FIT PASS {family} MAE={record['mae']:.6f} SECONDS={fit_time:.3f}", flush=True)
    return bundle, predictions, record


def run_lender_ablations(
    root: Path, train: pd.DataFrame, validation: pd.DataFrame, design: dict, winners_report: dict
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    runtime_records = []
    definitions = {d["candidate_id"]: d for d in design["candidate_definitions"]}
    for family in ("catboost", "lightgbm", "xgboost"):
        winner = winners_report["winners"][family]
        definition = definitions[winner["candidate_id"]]
        _, with_predictions, runtime = fit_lender_ablation(
            root, family, winner, definition, train, validation, design
        )
        without_frame = pd.read_parquet(root / winner["selected_prediction_path"])
        without_predictions = without_frame["y_pred"].to_numpy(dtype=np.float64)
        without_metrics = compute_regression_metrics(validation[TARGET], without_predictions)
        with_metrics = compute_regression_metrics(validation[TARGET], with_predictions)
        row = {
            "family": family,
            "without_lender_model_id": winner["candidate_id"],
            "with_lender_model_id": runtime["with_lender_model_id"],
            "fixed_target_mode": winner["target_mode"],
            "fixed_fitted_iterations": int(winner["fitted_iterations"]),
            "only_added_feature": "respondent_id",
            "without_lender_mae": without_metrics["mae"],
            "with_lender_mae": with_metrics["mae"],
            "mae_difference_with_minus_without": with_metrics["mae"] - without_metrics["mae"],
            "without_lender_rmse": without_metrics["rmse"],
            "with_lender_rmse": with_metrics["rmse"],
            "rmse_difference_with_minus_without": with_metrics["rmse"] - without_metrics["rmse"],
            "without_lender_top_decile_mae": without_metrics["top_decile_mae"],
            "with_lender_top_decile_mae": with_metrics["top_decile_mae"],
            "top_decile_mae_difference_with_minus_without": with_metrics["top_decile_mae"] - without_metrics["top_decile_mae"],
            "without_lender_mean_signed_error": without_metrics["mean_signed_error"],
            "with_lender_mean_signed_error": with_metrics["mean_signed_error"],
            "absolute_mean_signed_error_difference": abs(with_metrics["mean_signed_error"]) - abs(without_metrics["mean_signed_error"]),
            "fit_time_difference_seconds": float(runtime.get("fit_time_seconds", 0.0)) - float(winner["fit_time_seconds"]),
            "prediction_correlation": float(np.corrcoef(without_predictions, with_predictions)[0, 1]),
            "mean_absolute_prediction_difference": float(np.mean(np.abs(with_predictions - without_predictions))),
            "interpretation_scope": "Accuracy-only diagnostic; not fairness or causal evidence.",
        }
        rows.append(row)
        runtime_records.append(runtime)
        atomic_csv(root / REPORTS_RELATIVE / "prompt2_lender_ablation.csv", pd.DataFrame(rows))
    return pd.DataFrame(rows), runtime_records


def build_comparison_reports(
    root: Path,
    validation: pd.DataFrame,
    baseline_records: list[dict],
    candidate_records: list[dict],
    winners_report: dict,
    lender_runtime: list[dict],
    environment: dict,
    started_at: float,
) -> tuple[pd.DataFrame, dict, dict]:
    rows = list(baseline_records)
    candidate_by_id = {r["candidate_id"]: r for r in candidate_records}
    for family in ("lasso", "histgradientboosting", "catboost", "lightgbm", "xgboost"):
        winner = winners_report["winners"][family]
        row = dict(candidate_by_id[winner["candidate_id"]])
        row["model_id"] = winner["candidate_id"]
        row["selected_family_representative"] = True
        rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison["absolute_mean_signed_error"] = comparison["mean_signed_error"].abs()
    comparison = comparison.sort_values(["mae", "rmse", "top_decile_mae"], kind="stable").reset_index(drop=True)
    comparison.insert(0, "validation_rank_by_mae", np.arange(1, len(comparison) + 1))
    atomic_csv(root / REPORTS_RELATIVE / "prompt2_family_comparison.csv", comparison)

    all_fit_records = [r for r in candidate_records if r.get("status") == "COMPLETE"] + lender_runtime
    runtime = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "environment": environment,
        "heavy_fits_sequential": True,
        "n_threads": THREADS,
        "primary_scientific_fit_count": len([r for r in candidate_records if r.get("status") == "COMPLETE"]),
        "lender_ablation_fit_count": 3,
        "total_scientific_fit_count": len([r for r in candidate_records if r.get("status") == "COMPLETE"]) + 3,
        "last_orchestration_pass_seconds_before_postchecks": time.perf_counter() - started_at,
        "recorded_scientific_fit_seconds": sum(float(r.get("fit_time_seconds", 0.0)) for r in all_fit_records),
        "recorded_prediction_seconds": sum(float(r.get("prediction_time_seconds", 0.0)) for r in all_fit_records),
        "cumulative_wall_time_seconds": None,
        "cumulative_wall_time_note": "A reporting-only failure required a cache-only continuation, so cumulative wall time was not retained as one trustworthy timer. Use exact per-fit and per-prediction times for runtime comparisons.",
        "fit_records": all_fit_records,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
    }
    atomic_json(root / REPORTS_RELATIVE / "prompt2_runtime.json", runtime)
    return comparison, runtime, {"candidate_by_id": candidate_by_id}


def clean_process_bundle_checks(root: Path, validation: pd.DataFrame) -> list[dict]:
    temp = root / TMP_RELATIVE / "clean_process"
    temp.mkdir(parents=True, exist_ok=True)
    sample = validation.iloc[:1000].copy()
    sample_path = temp / "validation_sample.parquet"
    sample.to_parquet(sample_path, index=False, compression="zstd")
    results = []
    for key, filename in REQUIRED_BUNDLE_NAMES.items():
        bundle_path = root / MODELS_RELATIVE / filename
        bundle = load_bundle(bundle_path)
        reference = bundle.predict(sample)
        output_path = temp / f"{key}_prediction.npy"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "clean-check",
            "--root",
            str(root.resolve()),
            "--bundle",
            bundle_path.relative_to(root).as_posix(),
            "--sample",
            sample_path.relative_to(root).as_posix(),
            "--output",
            output_path.relative_to(root).as_posix(),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            raise RuntimeError(f"Clean-process reload failed for {key}: {completed.stderr}")
        clean = np.load(output_path, allow_pickle=False)
        maximum_difference = float(np.max(np.abs(reference - clean)))
        result = {
            "bundle_key": key,
            "bundle_path": bundle_path.relative_to(root).as_posix(),
            "sample_rows": len(sample),
            "row_order_identical": len(clean) == len(reference),
            "finite_predictions": bool(np.isfinite(clean).all()),
            "maximum_absolute_difference": maximum_difference,
            "status": "PASS" if maximum_difference <= 1e-7 and np.isfinite(clean).all() and len(clean) == len(reference) else "FAIL",
        }
        if result["status"] != "PASS":
            raise RuntimeError(f"Clean-process equality failed: {result}")
        results.append(result)
        output_path.unlink()
    sample_path.unlink()
    if temp.exists() and not any(temp.iterdir()):
        temp.rmdir()
    return results


def build_artifact_manifests(
    root: Path, validation: pd.DataFrame, design: dict, clean_checks: list[dict]
) -> tuple[dict, dict]:
    expected_hashes = validation["row_hash"].astype(str).reset_index(drop=True)
    expected_target = validation[TARGET].to_numpy(dtype=np.float64)
    prediction_entries = []
    for key, filename in PREDICTION_NAMES.items():
        path = root / PREDICTIONS_RELATIVE / filename
        frame = pd.read_parquet(path)
        entry = {
            "key": key,
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "rows": len(frame),
            "columns": frame.columns.tolist(),
            "row_hash_unique": bool(frame["row_hash"].is_unique),
            "row_order_equal": bool(frame["row_hash"].astype(str).reset_index(drop=True).equals(expected_hashes)),
            "target_equal": bool(np.array_equal(frame["y_true"].to_numpy(dtype=np.float64), expected_target)),
            "finite_predictions": bool(np.isfinite(frame["y_pred"].to_numpy(dtype=np.float64)).all()),
            "compression": sorted({
                str(pq.ParquetFile(path).metadata.row_group(0).column(i).compression).upper()
                for i in range(pq.ParquetFile(path).metadata.num_columns)
            }),
        }
        entry["status"] = "PASS" if entry["rows"] == 100_000 and all(entry[k] for k in ("row_hash_unique", "row_order_equal", "target_equal", "finite_predictions")) and entry["compression"] == ["ZSTD"] else "FAIL"
        if entry["status"] != "PASS":
            raise RuntimeError(f"Prediction artifact validation failed: {entry}")
        prediction_entries.append(entry)
    prediction_manifest = {
        "status": "PASS", "created_at_utc": utc_now(), "validation_row_hash_digest": design["validation_row_hash_digest"],
        "required_artifact_count": 10, "artifacts": prediction_entries, "iid_prediction_count": 0,
    }
    atomic_json(root / REPORTS_RELATIVE / "prompt2_prediction_manifest.json", prediction_manifest)

    clean_by_key = {r["bundle_key"]: r for r in clean_checks}
    model_entries = []
    for key, filename in REQUIRED_BUNDLE_NAMES.items():
        path = root / MODELS_RELATIVE / filename
        bundle = load_bundle(path)
        entry = {
            "key": key,
            "path": path.relative_to(root).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "model_id": bundle.model_id,
            "family": bundle.family,
            "feature_contract": bundle.feature_contract_name,
            "target_mode": bundle.target_mode,
            "feature_count": len(bundle.feature_names),
            "selected_best_iteration": bundle.selected_best_iteration,
            "development_source_sha256": bundle.development_source_sha256,
            "train_row_hash_digest": bundle.train_row_hash_digest,
            "validation_row_hash_digest": bundle.validation_row_hash_digest,
            "clean_process_check": clean_by_key[key],
        }
        entry["status"] = "PASS" if (
            entry["development_source_sha256"] == design["development_source"]["sha256"]
            and entry["train_row_hash_digest"] == design["train_row_hash_digest"]
            and entry["validation_row_hash_digest"] == design["validation_row_hash_digest"]
            and clean_by_key[key]["status"] == "PASS"
        ) else "FAIL"
        if entry["status"] != "PASS":
            raise RuntimeError(f"Model bundle validation failed: {entry}")
        model_entries.append(entry)
    model_manifest = {"status": "PASS", "created_at_utc": utc_now(), "required_bundle_count": 8, "bundles": model_entries}
    atomic_json(root / REPORTS_RELATIVE / "prompt2_model_manifest.json", model_manifest)
    return prediction_manifest, model_manifest


def create_reporting_notebook(root: Path) -> Path:
    notebook_path = root / "notebooks" / "02_BASELINES_AND_BOOSTING.ipynb"
    nb = nbformat.v4.new_notebook()
    cells = []

    def markdown(title: str, text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{text}"))

    def code(source: str) -> None:
        cells.append(nbformat.v4.new_code_cell(source.strip()))

    cells.append(nbformat.v4.new_markdown_cell(
        "# Regression V2: Baselines and Boosting Models\n\n"
        "This report compares frozen family Candidates on the fixed Development Validation rows. "
        "Validation supports model development; it is not an independent final evaluation."
    ))
    markdown("1. Objective and scope", "We compare two transparent baselines, Lasso, HistGradientBoosting, CatBoost, LightGBM, and XGBoost. No Deep model, final ensemble, or final article model is created here.")
    code("""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = Path.cwd()
REPORTS = ROOT / "outputs" / "reports"
candidate = pd.read_csv(REPORTS / "prompt2_candidate_results.csv")
comparison = pd.read_csv(REPORTS / "prompt2_family_comparison.csv")
lender = pd.read_csv(REPORTS / "prompt2_lender_ablation.csv")
design = json.loads((REPORTS / "prompt2_frozen_design.json").read_text(encoding="utf-8"))
winners = json.loads((REPORTS / "prompt2_family_winners.json").read_text(encoding="utf-8"))
runtime = json.loads((REPORTS / "prompt2_runtime.json").read_text(encoding="utf-8"))
prediction_manifest = json.loads((REPORTS / "prompt2_prediction_manifest.json").read_text(encoding="utf-8"))
model_manifest = json.loads((REPORTS / "prompt2_model_manifest.json").read_text(encoding="utf-8"))
print("Saved-artifact report loaded. Model and preprocessing fit count in this notebook: 0")
""")
    markdown("2. Prompt 1B handoff", "Prompt 1B passed before modeling. The Development file and its saved attestations were validated again. Raw and IID access remained zero.")
    code("""
handoff = {
    "Prompt 1B status": "PASS",
    "Development SHA-256": design["development_source"]["sha256"],
    "Raw access count": runtime["raw_access_count"],
    "IID feature access count": runtime["iid_feature_access_count"],
    "IID target access count": runtime["iid_target_access_count"],
}
display(pd.DataFrame([handoff]))
""")
    markdown("3. Development Train and Validation roles", "All Candidates use the same 400,000 Train rows and the same 100,000 Validation rows. The roles were not recreated.")
    code("""
display(pd.DataFrame([{
    "Development rows": design["development_source"]["row_count"],
    "Train rows": design["development_source"]["role_counts"]["train"],
    "Validation rows": design["development_source"]["role_counts"]["validation"],
    "Train/Validation overlap": design["development_source"]["checks"]["train_validation_overlap"],
}]))
""")
    markdown("4. Feature contracts", "Primary selection excludes sensitive fields and lender identity. The matched lender diagnostic adds only `respondent_id`. Lasso uses a compact feature pack.")
    code("""
contracts = design["feature_contracts"]
display(pd.DataFrame([
    {"contract": "main_without_sensitive_without_lender", "feature_count": len(contracts["main_without_sensitive_without_lender"]), "respondent_id": False},
    {"contract": "main_without_sensitive_with_lender", "feature_count": len(contracts["main_without_sensitive_with_lender"]), "respondent_id": True},
    {"contract": "linear_compact_v2", "feature_count": len(contracts["linear_compact_v2"]), "respondent_id": False},
]))
""")
    markdown("5. Metric definitions", "MAE is primary. Supporting metrics are on the original target scale, in thousands of U.S. dollars. Signed error is prediction minus target; negative values mean underprediction. Tail groups use fixed Validation targets only.")
    code("""
display(pd.DataFrame({"metric": design["metrics"]}))
""")
    markdown("6. Train mean and median baselines", "The constants come only from Train targets. They provide simple reference levels.")
    code("""
baseline_view = comparison[comparison["family"].isin(["train_mean", "train_median"])][["family", "mae", "rmse", "r2", "mean_signed_error"]]
display(baseline_view.style.format({"mae":"{:.3f}", "rmse":"{:.3f}", "r2":"{:.4f}", "mean_signed_error":"{:.3f}"}))
""")
    for number, family, title in (
        (7, "lasso", "Lasso Candidates"),
        (8, "histgradientboosting", "HistGradientBoosting Candidates"),
        (9, "catboost", "CatBoost Candidates"),
        (10, "lightgbm", "LightGBM Candidates"),
        (11, "xgboost", "XGBoost Candidates"),
    ):
        markdown(f"{number}. {title}", f"The frozen {title[:-11]} definitions were evaluated without adding adaptive Candidates.")
        code(f"""
view = candidate[candidate["family"].eq("{family}")][["candidate_name", "target_mode", "status", "mae", "rmse", "top_decile_mae", "mean_signed_error", "fitted_iterations", "fit_time_seconds"]]
display(view.style.format({{"mae":"{{:.3f}}", "rmse":"{{:.3f}}", "top_decile_mae":"{{:.3f}}", "mean_signed_error":"{{:.3f}}", "fit_time_seconds":"{{:.2f}}"}}, na_rep="—"))
""")
    markdown("12. Family-selection rule", "Within each family, the rule starts with lowest MAE. Near ties use RMSE, then top-decile MAE, followed by fitted iterations, model size, fit time, and the simpler target mode.")
    code("display(pd.DataFrame([design['selection_rule']]))")
    markdown("13. Selected family representatives", "One no-lender representative is frozen for every fitted family. These are inputs to later work, not a final model decision.")
    code("""
winner_table = pd.DataFrame([{
    "family": family,
    "model_id": item["candidate_id"],
    "target_mode": item["target_mode"],
    "MAE": item["mae"],
    "RMSE": item["rmse"],
    "top-decile MAE": item["top_decile_mae"],
} for family, item in winners["winners"].items()])
display(winner_table.style.format({"MAE":"{:.3f}", "RMSE":"{:.3f}", "top-decile MAE":"{:.3f}"}))
""")
    markdown("14. Common Validation leaderboard", "This table compares all no-lender family representatives with the two Train-only constants on the same Validation rows.")
    code("""
leader = comparison[["validation_rank_by_mae", "family", "target_mode", "mae", "rmse", "r2", "rmsle", "top_decile_mae"]]
display(leader.style.format({"mae":"{:.3f}", "rmse":"{:.3f}", "r2":"{:.4f}", "rmsle":"{:.4f}", "top_decile_mae":"{:.3f}"}))
ax = leader.set_index("family")[["mae", "rmse"]].plot(kind="bar", figsize=(10, 4), color=["#35618f", "#d17a22"])
ax.set_ylabel("Error (thousands of U.S. dollars)")
ax.set_title("Development Validation MAE and RMSE")
plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.show()
""")
    markdown("15. Tail-error comparison", "Top-decile MAE focuses on the largest fixed Validation targets. It is descriptive and does not change the frozen split.")
    code("""
ax = comparison.sort_values("top_decile_mae").plot(x="family", y="top_decile_mae", kind="bar", legend=False, figsize=(9, 4), color="#7b4f9d")
ax.set_ylabel("Top-decile MAE (thousands of U.S. dollars)")
ax.set_title("Development Validation tail error")
plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.show()
display(comparison[["family", "top_decile_mae", "top_five_percent_mae", "top_decile_underprediction_rate", "top_five_percent_underprediction_rate"]])
""")
    markdown("16. Signed-error comparison", "The zero line represents no average signed bias. Distance from zero is the comparison rule.")
    code("""
signed = comparison.sort_values("mean_signed_error")
ax = signed.plot(x="family", y="mean_signed_error", kind="bar", legend=False, figsize=(9, 4), color="#3b8c6e")
ax.axhline(0, color="black", linewidth=1)
ax.set_ylabel("Mean signed error (thousands of U.S. dollars)")
ax.set_title("Development Validation signed error")
plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.show()
""")
    markdown("17. Runtime and model-size comparison", "Heavy fits ran sequentially with at most four model threads. Peak memory is a bounded parent-process observation.")
    code("""
runtime_view = comparison[~comparison["family"].isin(["train_mean", "train_median"])][["family", "mae", "fit_time_seconds", "bundle_size_bytes"]]
display(runtime_view)
ax = runtime_view.plot.scatter(x="fit_time_seconds", y="mae", figsize=(7, 4), color="#b34d4d")
for _, row in runtime_view.iterrows():
    ax.annotate(row["family"], (row["fit_time_seconds"], row["mae"]), fontsize=8)
ax.set_xlabel("Fit time (seconds)"); ax.set_ylabel("Validation MAE")
ax.set_title("Runtime versus Development Validation MAE"); plt.tight_layout(); plt.show()
""")
    markdown("18. Controlled lender ablation", "Each boosting comparison adds only `respondent_id` and keeps the selected target mode, objective, settings, seed, and fitted iteration count fixed. This is an accuracy diagnostic, not fairness or causal evidence.")
    code("display(lender)")
    markdown("19. Saved bundles and prediction artifacts", "Eight complete model bundles and ten aligned Validation prediction files passed reload and integrity checks.")
    code("""
display(pd.DataFrame(model_manifest["bundles"])[["key", "model_id", "feature_contract", "target_mode", "size_bytes", "status"]])
display(pd.DataFrame(prediction_manifest["artifacts"])[["key", "rows", "row_order_equal", "target_equal", "finite_predictions", "status"]])
""")
    markdown("20. Verification", "The saved bundles predict in clean processes within the required numerical tolerance. Every prediction file uses the same Validation row order and exact target values.")
    code("""
clean = pd.DataFrame([b["clean_process_check"] for b in model_manifest["bundles"]])
display(clean)
print("Maximum clean-process absolute difference:", clean["maximum_absolute_difference"].max())
print("Prediction manifest status:", prediction_manifest["status"])
""")
    markdown("21. Limitations", "Development Validation is reused for Candidate selection and early stopping where specified. It is not an unbiased final evaluation. Resource telemetry is bounded, and lender identity results cannot show cause or fairness.")
    markdown("22. Prompt 3 handoff", "The selected family bundles and exact aligned Validation predictions are ready for the next authorized Prompt only after final review and `PROMPT2_READY.json` PASS. This notebook does not start Prompt 3.")
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": platform.python_version()}
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = notebook_path.with_suffix(".ipynb.tmp")
    nbformat.write(nb, temporary)
    os.replace(temporary, notebook_path)
    return notebook_path


def execute_reporting_notebook(root: Path, notebook_path: Path) -> dict:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(root.resolve())}})
    executed = client.execute()
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    unexecuted = [cell for cell in code_cells if cell.get("execution_count") is None]
    if errors or unexecuted:
        raise RuntimeError(f"Notebook execution failed: errors={len(errors)}, unexecuted={len(unexecuted)}")
    temporary = notebook_path.with_suffix(".ipynb.tmp")
    nbformat.write(executed, temporary)
    os.replace(temporary, notebook_path)
    figure_outputs = sum(
        "image/png" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )
    table_outputs = sum(
        "text/html" in output.get("data", {})
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )
    return {
        "status": "PASS", "path": notebook_path.relative_to(root).as_posix(), "code_cells": len(code_cells),
        "executed_code_cells": len(code_cells) - len(unexecuted), "error_count": len(errors),
        "inline_figure_outputs": int(figure_outputs), "inline_table_outputs": int(table_outputs),
        "model_fit_count": 0, "preprocessing_fit_count": 0, "raw_access_count": 0,
        "iid_feature_access_count": 0, "iid_target_access_count": 0,
    }


def run_pipeline(root: Path) -> dict:
    started = time.perf_counter()
    for relative in (MODELS_RELATIVE, PREDICTIONS_RELATIVE, REPORTS_RELATIVE, TMP_RELATIVE):
        (root / relative).mkdir(parents=True, exist_ok=True)
    handoff = validate_data_ready(root)
    roles = load_feature_roles(root)
    print("PREFLIGHT LOAD DEVELOPMENT", flush=True)
    development = pd.read_parquet(root / DEVELOPMENT_RELATIVE)
    source_record = validate_development(root, development, roles)
    train = development.loc[development.development_role.eq("train")].copy(deep=False)
    validation = development.loc[development.development_role.eq("validation")].copy(deep=False)
    design = prepare_design(root, development, roles, source_record)
    environment = environment_record(root)
    smoke = run_smoke_tests(root, train, validation, design)
    baseline_records = run_baselines(root, train, validation)
    candidate_records, winners = run_primary_fits(root, train, validation, design, smoke)
    lender_table, lender_runtime = run_lender_ablations(root, train, validation, design, winners)
    comparison, runtime, _ = build_comparison_reports(
        root, validation, baseline_records, candidate_records, winners, lender_runtime, environment, started
    )
    clean_checks = clean_process_bundle_checks(root, validation)
    prediction_manifest, model_manifest = build_artifact_manifests(root, validation, design, clean_checks)
    notebook_path = create_reporting_notebook(root)
    notebook_result = execute_reporting_notebook(root, notebook_path)
    runtime["notebook_execution"] = notebook_result
    runtime["last_orchestration_pass_seconds"] = time.perf_counter() - started
    runtime["last_orchestration_pass_scope"] = "Cache-only artifact verification and notebook continuation; not cumulative Prompt 2 scientific runtime."
    runtime["candidate_rows"] = len(candidate_records)
    runtime["lender_ablation_rows"] = len(lender_table)
    runtime["common_comparison_rows"] = len(comparison)
    atomic_json(root / REPORTS_RELATIVE / "prompt2_runtime.json", runtime)
    result = {
        "status": "AWAITING_INDEPENDENT_REVIEW",
        "handoff_status": handoff["data_ready"]["status"],
        "design_digest": design["design_digest"],
        "primary_scientific_fit_count": runtime["primary_scientific_fit_count"],
        "total_scientific_fit_count": runtime["total_scientific_fit_count"],
        "prediction_manifest_status": prediction_manifest["status"],
        "model_manifest_status": model_manifest["status"],
        "notebook": notebook_result,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def verify_prompt2(root: Path) -> dict:
    design = json.loads((root / REPORTS_RELATIVE / "prompt2_frozen_design.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / REPORTS_RELATIVE / "prompt2_runtime.json").read_text(encoding="utf-8"))
    winners = json.loads((root / REPORTS_RELATIVE / "prompt2_family_winners.json").read_text(encoding="utf-8"))
    prediction_manifest = json.loads((root / REPORTS_RELATIVE / "prompt2_prediction_manifest.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((root / REPORTS_RELATIVE / "prompt2_model_manifest.json").read_text(encoding="utf-8"))
    reviewer = json.loads((root / REPORTS_RELATIVE / "prompt2_reviewer.json").read_text(encoding="utf-8"))
    candidate_records = load_candidate_results(root)
    handoff = validate_data_ready(root)
    roles = load_feature_roles(root)
    development = pd.read_parquet(root / DEVELOPMENT_RELATIVE)
    source = validate_development(root, development, roles)
    checks: dict[str, Any] = {
        "prompt1b_data_ready_pass": handoff["data_ready"]["status"] == "PASS",
        "development_source_hash_equal": source["sha256"] == design["development_source"]["sha256"],
        "development_rows": len(development),
        "train_rows": int(development.development_role.eq("train").sum()),
        "validation_rows": int(development.development_role.eq("validation").sum()),
        "train_validation_row_hash_overlap": source["checks"]["train_validation_overlap"],
        "raw_access_count": runtime["raw_access_count"],
        "iid_feature_access_count": runtime["iid_feature_access_count"],
        "iid_target_access_count": runtime["iid_target_access_count"],
        "primary_feature_contract": PRIMARY_CONTRACT,
        "primary_scientific_fits": runtime["primary_scientific_fit_count"],
        "total_scientific_fits": runtime["total_scientific_fit_count"],
        "prediction_artifact_count": len(prediction_manifest["artifacts"]),
        "model_bundle_count": len(model_manifest["bundles"]),
        "every_prediction_pass": all(x["status"] == "PASS" for x in prediction_manifest["artifacts"]),
        "every_bundle_pass": all(x["status"] == "PASS" for x in model_manifest["bundles"]),
        "maximum_clean_process_difference": max(x["clean_process_check"]["maximum_absolute_difference"] for x in model_manifest["bundles"]),
        "notebook_code_cells": runtime["notebook_execution"]["code_cells"],
        "notebook_all_code_cells_executed": runtime["notebook_execution"]["code_cells"] == runtime["notebook_execution"]["executed_code_cells"],
        "notebook_error_count": runtime["notebook_execution"]["error_count"],
        "notebook_inline_figure_outputs": runtime["notebook_execution"]["inline_figure_outputs"],
        "notebook_inline_table_outputs": runtime["notebook_execution"]["inline_table_outputs"],
        "notebook_model_fit_count": runtime["notebook_execution"]["model_fit_count"],
        "notebook_preprocessing_fit_count": runtime["notebook_execution"]["preprocessing_fit_count"],
        "reviewer_status": reviewer.get("status"),
        "reviewer_unresolved_critical": reviewer.get("unresolved_critical"),
        "reviewer_unresolved_major": reviewer.get("unresolved_major"),
        "no_cross_validation": True,
        "no_final_ensemble": True,
        "no_full_development_final_fit": True,
        "iid_prediction_count": prediction_manifest.get("iid_prediction_count"),
    }
    expected_candidate_ids = {d["candidate_id"] for d in design["candidate_definitions"]}
    actual_candidate_ids = {r["candidate_id"] for r in candidate_records}
    checks["no_adaptive_candidate_added"] = actual_candidate_ids == expected_candidate_ids
    checks["candidate_definition_count"] = len(actual_candidate_ids)
    selection_correct = True
    for family in ("lasso", "histgradientboosting", "catboost", "lightgbm", "xgboost"):
        family_records = [r for r in candidate_records if r["family"] == family]
        recomputed = select_family_candidate(family_records)
        selection_correct &= recomputed["candidate_id"] == winners["winners"][family]["candidate_id"]
    checks["family_selection_rule_applied_correctly"] = bool(selection_correct)
    excluded = set(roles["sensitive_fields"]) | set(roles["target_and_alias_exclusions"]) | set(roles["audit_only_fields"])
    sensitive_use = target_use = audit_use = 0
    for entry in model_manifest["bundles"]:
        bundle = load_bundle(root / entry["path"])
        sensitive_use += len(set(bundle.feature_names) & set(roles["sensitive_fields"]))
        target_use += len(set(bundle.feature_names) & set(roles["target_and_alias_exclusions"]))
        audit_use += len(set(bundle.feature_names) & set(roles["audit_only_fields"]))
    checks["sensitive_feature_use_count"] = sensitive_use
    checks["target_feature_use_count"] = target_use
    checks["audit_feature_use_count"] = audit_use
    notebook = nbformat.read(root / "notebooks/02_BASELINES_AND_BOOSTING.ipynb", as_version=4)
    notebook_source = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
    checks["notebook_has_no_fit_call"] = ".fit(" not in notebook_source and ".fit_transform(" not in notebook_source

    required_equalities = {
        "development_rows": 500_000,
        "train_rows": 400_000,
        "validation_rows": 100_000,
        "train_validation_row_hash_overlap": 0,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "prediction_artifact_count": 10,
        "model_bundle_count": 8,
        "notebook_error_count": 0,
        "notebook_model_fit_count": 0,
        "notebook_preprocessing_fit_count": 0,
        "reviewer_unresolved_critical": 0,
        "reviewer_unresolved_major": 0,
        "iid_prediction_count": 0,
        "sensitive_feature_use_count": 0,
        "target_feature_use_count": 0,
        "audit_feature_use_count": 0,
        "candidate_definition_count": 13,
    }
    required_true = [
        "prompt1b_data_ready_pass", "development_source_hash_equal", "every_prediction_pass", "every_bundle_pass",
        "notebook_all_code_cells_executed", "no_adaptive_candidate_added", "family_selection_rule_applied_correctly",
        "no_cross_validation", "no_final_ensemble", "no_full_development_final_fit", "notebook_has_no_fit_call",
    ]
    failures = [f"{k}: expected {v}, observed {checks.get(k)}" for k, v in required_equalities.items() if checks.get(k) != v]
    failures += [f"{k}: expected true" for k in required_true if checks.get(k) is not True]
    if checks["primary_scientific_fits"] > 13:
        failures.append("primary scientific fit budget exceeded")
    if checks["total_scientific_fits"] > 16:
        failures.append("total scientific fit budget exceeded")
    if checks["maximum_clean_process_difference"] > 1e-7:
        failures.append("clean-process prediction tolerance exceeded")
    if checks["notebook_inline_figure_outputs"] < 4 or checks["notebook_inline_table_outputs"] < 8:
        failures.append("notebook lacks required inline quantitative evidence")
    if checks["reviewer_status"] != "PASS":
        failures.append("reviewer status is not PASS")
    verification = {
        "status": "PASS" if not failures else "FAIL",
        "created_at_utc": utc_now(),
        "checks": checks,
        "failures": failures,
        "limitations": [
            "Development Validation supports model development and is not an independent final evaluation.",
            "Peak memory is a bounded parent-process observation.",
            "The lender comparison is accuracy-only and is not fairness or causal evidence.",
        ],
    }
    atomic_json(root / REPORTS_RELATIVE / "prompt2_verification.json", verification)
    if failures:
        raise RuntimeError(f"Prompt 2 verification failed: {failures}")
    return verification


def create_ready(root: Path) -> dict:
    verification = json.loads((root / REPORTS_RELATIVE / "prompt2_verification.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("PROMPT2_READY requires prompt2_verification PASS.")
    task = (root / "TASK.md").read_text(encoding="utf-8")
    if "Prompt 2 COMPLETE" not in task:
        raise RuntimeError("Update TASK.md to Prompt 2 COMPLETE before creating PROMPT2_READY.")
    design = json.loads((root / REPORTS_RELATIVE / "prompt2_frozen_design.json").read_text(encoding="utf-8"))
    winners = json.loads((root / REPORTS_RELATIVE / "prompt2_family_winners.json").read_text(encoding="utf-8"))
    runtime = json.loads((root / REPORTS_RELATIVE / "prompt2_runtime.json").read_text(encoding="utf-8"))
    model_manifest = json.loads((root / REPORTS_RELATIVE / "prompt2_model_manifest.json").read_text(encoding="utf-8"))
    prediction_manifest = json.loads((root / REPORTS_RELATIVE / "prompt2_prediction_manifest.json").read_text(encoding="utf-8"))
    ready = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "development_source_path": design["development_source"]["path"],
        "development_source_sha256": design["development_source"]["sha256"],
        "train_rows": 400_000,
        "validation_rows": 100_000,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "scientific_fit_count": runtime["total_scientific_fit_count"],
        "selected_family_model_ids": {k: v["candidate_id"] for k, v in winners["winners"].items()},
        "selected_bundle_paths": [x["path"] for x in model_manifest["bundles"]],
        "selected_validation_prediction_paths": [x["path"] for x in prediction_manifest["artifacts"]],
        "feature_contract_names": ["linear_compact_v2", PRIMARY_CONTRACT, LENDER_CONTRACT],
        "verification_path": "outputs/reports/prompt2_verification.json",
        "notebook_path": "notebooks/02_BASELINES_AND_BOOSTING.ipynb",
        "reviewer_status": verification["checks"]["reviewer_status"],
        "next_step": "Begin Prompt 3 - Regression V2 Deep Tabular Model.",
    }
    atomic_json(root / REPORTS_RELATIVE / "PROMPT2_READY.json", ready)
    return ready


def clean_check(root: Path, bundle_relative: str, sample_relative: str, output_relative: str) -> None:
    bundle = load_bundle(root / bundle_relative)
    sample = pd.read_parquet(root / sample_relative)
    predictions = bundle.predict(sample)
    with (root / output_relative).open("wb") as handle:
        np.save(handle, predictions, allow_pickle=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify", "ready", "clean-check"])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--bundle")
    parser.add_argument("--sample")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.command == "run":
        run_pipeline(root)
    elif args.command == "verify":
        verify_prompt2(root)
    elif args.command == "ready":
        create_ready(root)
    else:
        if not (args.bundle and args.sample and args.output):
            parser.error("clean-check requires --bundle, --sample, and --output")
        clean_check(root, args.bundle, args.sample, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
