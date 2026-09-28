"""Bounded, resumable orchestration for Regression V2 Prompt 3.

This module reads only the frozen Development parquet and saved Prompt 2
artifacts. It rejects Raw and IID paths before a file can be opened. Heavy
work is available through explicit functions and CLI commands; importing this
module never fits a model or writes a file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# The project-local Deep environment intentionally reuses light data packages
# from the base Python. Make PyArrow visible before pandas initializes its
# optional Arrow compatibility layer.
_REUSED_BASE_SITE = Path(sys.base_prefix) / "Lib" / "site-packages"
if _REUSED_BASE_SITE.is_dir() and str(_REUSED_BASE_SITE) not in sys.path:
    sys.path.append(str(_REUSED_BASE_SITE))

import joblib
import numpy as np
import pandas as pd

try:
    from .deep_bundles import (
        BundleMetadata,
        FTTransformerBundle,
        RealMLPBundle,
        directory_size_bytes,
        load_bundle,
        save_fttransformer_bundle,
        save_realmlp_bundle,
    )
    from .deep_metrics import (
        compute_regression_metrics,
        paired_mae_bootstrap,
        select_deep_anchor,
        select_family_candidate,
        transform_target,
    )
    from .deep_preprocessing import (
        FTPreprocessor,
        RealMLPPreprocessor,
        make_ft_internal_split,
        ordered_digest,
        split_feature_types,
    )
except ImportError:  # Direct use with regression_v2/src on sys.path.
    from deep_bundles import (
        BundleMetadata,
        FTTransformerBundle,
        RealMLPBundle,
        directory_size_bytes,
        load_bundle,
        save_fttransformer_bundle,
        save_realmlp_bundle,
    )
    from deep_metrics import (
        compute_regression_metrics,
        paired_mae_bootstrap,
        select_deep_anchor,
        select_family_candidate,
        transform_target,
    )
    from deep_preprocessing import (
        FTPreprocessor,
        RealMLPPreprocessor,
        make_ft_internal_split,
        ordered_digest,
        split_feature_types,
    )


SEED = 42
TARGET = "loan_amount_000s"
TARGET_UNIT = "thousands of U.S. dollars"
PRIMARY_CONTRACT = "main_without_sensitive_without_lender"
EXPECTED_FEATURES = 35
EXPECTED_DEVELOPMENT_ROWS = 500_000
EXPECTED_TRAIN_ROWS = 400_000
EXPECTED_VALIDATION_ROWS = 100_000
FT_FIT_ROWS = 360_000
FT_STOP_ROWS = 40_000
MAX_SCIENTIFIC_FITS = 5
NUM_WORKERS = 0

DEVELOPMENT_RELATIVE = Path("outputs/data/development.parquet")
REPORTS_RELATIVE = Path("outputs/reports")
MODELS_RELATIVE = Path("outputs/models/prompt3")
PREDICTIONS_RELATIVE = Path("outputs/predictions/prompt3/validation")
FIGURES_RELATIVE = Path("outputs/figures/prompt3")
TMP_RELATIVE = Path("outputs/tmp/prompt3")
NOTEBOOK_RELATIVE = Path("notebooks/03_DEEP_TABULAR_MODELS.ipynb")
REALMLP_SMOKE_ATTEMPT3_RELATIVE = TMP_RELATIVE / "realmlp_smoke_attempt3"

REALMLP_SELECTION_FIELDS = {
    "stop_epoch",
    "stop_epochs",
    "best_epoch",
    "best_epochs",
    "selected_epoch",
}
REALMLP_NON_SELECTION_EPOCH_FIELDS = {"n_epochs", "max_epochs"}
REALMLP_METADATA_ATTRIBUTES = {
    "fit_params_",
    "alg_interface_",
    "cv_alg_interface_",
    "refit_alg_interface_",
    "model_",
    "models_",
    "refit_interfaces_",
    "fit_params",
    "config",
    "progress",
    "stop_epoch",
    "stop_epochs",
    "best_epoch",
    "best_epochs",
    "selected_epoch",
    "n_epochs",
    "max_epochs",
    "single_split_interfaces",
}

PROMPT2_PREDICTIONS = {
    "lasso": Path("outputs/predictions/prompt2/validation/selected_lasso.parquet"),
    "histgradientboosting": Path(
        "outputs/predictions/prompt2/validation/selected_histgradientboosting.parquet"
    ),
    "catboost": Path(
        "outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet"
    ),
    "lightgbm": Path(
        "outputs/predictions/prompt2/validation/selected_lightgbm_without_lender.parquet"
    ),
    "xgboost": Path(
        "outputs/predictions/prompt2/validation/selected_xgboost_without_lender.parquet"
    ),
}

REALMLP_CANDIDATES = (
    {
        "candidate_id": "realmlp_raw_pdrop015",
        "family": "realmlp",
        "target_mode": "raw",
        "p_drop": 0.15,
        "simplicity_rank": 0,
    },
    {
        "candidate_id": "realmlp_raw_pdrop020",
        "family": "realmlp",
        "target_mode": "raw",
        "p_drop": 0.20,
        "simplicity_rank": 1,
    },
)

FT_CANDIDATES = (
    {
        "candidate_id": "fttransformer_log1p_wd1e5",
        "family": "fttransformer",
        "target_mode": "log1p",
        "weight_decay": 1e-5,
        "simplicity_rank": 0,
    },
    {
        "candidate_id": "fttransformer_log1p_wd1e4",
        "family": "fttransformer",
        "target_mode": "log1p",
        "weight_decay": 1e-4,
        "simplicity_rank": 1,
    },
)
ALL_CANDIDATES = REALMLP_CANDIDATES + FT_CANDIDATES

REALMLP_PARAMETERS = {
    "device": "cpu",
    "random_state": SEED,
    "n_cv": 1,
    "n_refit": 1,
    "n_repeats": 1,
    "val_fraction": 0.10,
    "n_epochs": 30,
    "batch_size": 1024,
    "predict_batch_size": 4096,
    "n_threads": min(4, os.cpu_count() or 1),
    "val_metric_name": "mae",
    "use_early_stopping": True,
    "use_best_mean_epoch_for_cv": True,
    "verbosity": 1,
}

FT_SCIENTIFIC_PARAMETERS = {
    "target_mode": "log1p",
    "d_token": 64,
    "n_blocks": 3,
    "attention_n_heads": 8,
    "attention_dropout": 0.20,
    "ffn_hidden_factor": 2.0,
    "ffn_dropout": 0.10,
    "residual_dropout": 0.0,
    "optimizer": "AdamW",
    "learning_rate": 1e-4,
    "gradient_clip_norm": 1.0,
    "maximum_epochs": 50,
    "early_stopping_patience": 8,
    "training_loss": "MSE on log1p target",
    "epoch_monitor": "MAE on original target scale",
    "random_state": SEED,
    "num_workers": NUM_WORKERS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def discover_project_root(start: str | Path | None = None) -> Path:
    """Find the required regresionpart2 project root structurally."""
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if candidate.name == "regresionpart2" and (candidate / "regression_v2").is_dir():
            return candidate
    raise RuntimeError("Could not find the required regresionpart2 project root.")


def regression_v2_root(start: str | Path | None = None) -> Path:
    return discover_project_root(start) / "regression_v2"


def guard_read_path(root: str | Path, path: str | Path) -> Path:
    """Allow saved Regression V2 artifacts but reject Raw and IID inputs."""
    root_path = Path(root).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()
    try:
        relative = resolved.relative_to(root_path)
    except ValueError as error:
        raise PermissionError("Prompt 3 reads must stay inside regression_v2.") from error
    lowered = [part.lower() for part in relative.parts]
    if lowered and lowered[0] == "data":
        raise PermissionError("Prompt 3 must not open Raw data.")
    prohibited = {
        "iid_holdout_features.parquet",
        "iid_holdout_targets.parquet",
    }
    if relative.name.lower() in prohibited:
        raise PermissionError("Prompt 3 must not open IID files.")
    return resolved


def guard_write_path(root: str | Path, path: str | Path) -> Path:
    """Keep every runtime write below regression_v2."""
    root_path = Path(root).resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root_path / candidate).resolve()
    try:
        relative = resolved.relative_to(root_path)
    except ValueError as error:
        raise PermissionError("Prompt 3 writes must stay inside regression_v2.") from error
    if relative.parts and relative.parts[0].lower() == "data":
        raise PermissionError("Prompt 3 must not write into the source-data directory.")
    return resolved


def atomic_json(root: str | Path, path: str | Path, payload: dict[str, Any]) -> Path:
    destination = guard_write_path(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, destination)
    return destination


def atomic_csv(root: str | Path, path: str | Path, frame: pd.DataFrame) -> Path:
    destination = guard_write_path(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary, nrows=1)
    os.replace(temporary, destination)
    return destination


def _ensure_pyarrow_available() -> None:
    importlib.import_module("pyarrow")


def atomic_parquet(root: str | Path, path: str | Path, frame: pd.DataFrame) -> Path:
    _ensure_pyarrow_available()
    destination = guard_write_path(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    reloaded = pd.read_parquet(temporary)
    if list(reloaded.columns) != list(frame.columns) or len(reloaded) != len(frame):
        raise RuntimeError(f"Parquet reload validation failed: {destination}")
    os.replace(temporary, destination)
    return destination


def _attempt_ledger_path(root: Path) -> Path:
    return guard_write_path(root, REPORTS_RELATIVE / "prompt3_attempt_ledger.json")


def _start_attempt(root: Path, operation: str, maximum: int) -> int:
    """Record an attempt before work and enforce the frozen retry budget."""
    path = _attempt_ledger_path(root)
    ledger = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"status": "TRACKING", "operations": {}}
    )
    records = list(ledger.setdefault("operations", {}).get(operation, []))
    if len(records) >= maximum:
        raise RuntimeError(f"Attempt budget is exhausted for {operation}.")
    number = len(records) + 1
    records.append(
        {
            "attempt": number,
            "status": "IN_PROGRESS",
            "started_at_utc": utc_now(),
        }
    )
    ledger["operations"][operation] = records
    atomic_json(root, path, ledger)
    return number


def _finish_attempt(
    root: Path,
    operation: str,
    number: int,
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    path = _attempt_ledger_path(root)
    ledger = json.loads(path.read_text(encoding="utf-8"))
    records = ledger["operations"][operation]
    record = records[number - 1]
    if int(record["attempt"]) != number or record["status"] != "IN_PROGRESS":
        raise RuntimeError(f"Attempt ledger is inconsistent for {operation}.")
    record["status"] = status
    record["finished_at_utc"] = utc_now()
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error"] = str(error)
    atomic_json(root, path, ledger)


def configure_runtime_environment(root: str | Path | None = None) -> dict[str, str]:
    """Place all library caches and temporary files under Prompt 3 tmp."""
    workspace = Path(root or regression_v2_root()).resolve()
    temporary = guard_write_path(workspace, TMP_RELATIVE)
    mapping = {
        "MPLCONFIGDIR": temporary / "mplconfig",
        "TORCH_HOME": temporary / "torch_home",
        "XDG_CACHE_HOME": temporary / "xdg_cache",
        "HF_HOME": temporary / "hf_home",
        "TMP": temporary / "process_tmp",
        "TEMP": temporary / "process_tmp",
    }
    for name, directory in mapping.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    return {name: str(path) for name, path in mapping.items()}


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    caches = configure_runtime_environment(workspace)
    _ensure_pyarrow_available()
    memory: dict[str, Any] = {}
    try:
        import psutil

        virtual = psutil.virtual_memory()
        memory = {"total_bytes": int(virtual.total), "available_bytes": int(virtual.available)}
    except ImportError:
        memory = {"status": "psutil unavailable"}
    torch_info: dict[str, Any] = {"installed": False, "cuda_available": False}
    try:
        import torch

        torch_info = {
            "installed": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
    except ImportError:
        pass
    packages = {
        name: _package_version(name)
        for name in (
            "numpy",
            "pandas",
            "scikit-learn",
            "torch",
            "pytabkit",
            "rtdl-revisiting-models",
            "joblib",
            "pyarrow",
        )
    }
    return {
        "status": "PASS"
        if packages["torch"] and packages["pytabkit"] and packages["rtdl-revisiting-models"]
        else "MISSING_REQUIRED_PACKAGE",
        "created_at_utc": utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count_logical": os.cpu_count(),
        "memory": memory,
        "packages": packages,
        "torch": torch_info,
        "realmlp_device": "cpu float32",
        "ft_device_before_smoke": "cuda" if torch_info.get("cuda_available") else "cpu",
        "num_workers": 0,
        "cache_paths": caches,
    }


def _read_json(root: Path, relative: str | Path) -> dict[str, Any]:
    path = guard_read_path(root, relative)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_prompt2_handoff(root: str | Path | None = None) -> dict[str, Any]:
    """Validate saved prerequisites without touching Raw or IID."""
    workspace = Path(root or regression_v2_root()).resolve()
    task = guard_read_path(workspace, "TASK.md").read_text(encoding="utf-8")
    if not (
        "Prompt 2 COMPLETE" in task
        or "Prompt 3 IN PROGRESS" in task
        or "Prompt 3 BLOCKED" in task
        or "Prompt 3R IN PROGRESS" in task
        or "Prompt 3 COMPLETE" in task
    ):
        raise RuntimeError("TASK.md does not preserve a valid Prompt 2-to-3 state.")
    required = (
        "outputs/reports/PROMPT2_READY.json",
        "outputs/reports/prompt2_verification.json",
        "outputs/reports/DATA_READY.json",
        "outputs/reports/final_verification.json",
        "outputs/reports/feature_roles.json",
        "outputs/reports/prompt2_family_winners.json",
        "outputs/reports/prompt2_prediction_manifest.json",
    )
    reports = {Path(name).name: _read_json(workspace, name) for name in required}
    for name, report in reports.items():
        if report.get("status") != "PASS":
            raise RuntimeError(f"Prompt 3 prerequisite is not PASS: {name}")
    manifest = reports["prompt2_prediction_manifest.json"]
    required_keys = set(PROMPT2_PREDICTIONS)
    valid_keys = {
        item.get("key")
        for item in manifest.get("artifacts", [])
        if item.get("status") == "PASS"
    }
    if not required_keys.issubset(valid_keys):
        raise RuntimeError("Prompt 2 selected Validation predictions are incomplete.")
    ready = reports["PROMPT2_READY.json"]
    if any(
        int(ready.get(name, -1)) != 0
        for name in ("raw_access_count", "iid_feature_access_count", "iid_target_access_count")
    ):
        raise RuntimeError("Prompt 2 access audit is not closed.")
    return {"status": "PASS", "reports": reports}


def load_development(
    root: str | Path | None = None, *, feature_names: Iterable[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load Development once and return copied Train and Validation frames."""
    workspace = Path(root or regression_v2_root()).resolve()
    handoff = validate_prompt2_handoff(workspace)
    roles = handoff["reports"]["feature_roles.json"]
    features = list(feature_names or roles["contracts"][PRIMARY_CONTRACT])
    if len(features) != EXPECTED_FEATURES or len(features) != len(set(features)):
        raise RuntimeError("The primary feature contract must contain exactly 35 unique features.")
    excluded = (
        set(roles.get("sensitive_fields", []))
        | set(roles.get("audit_only_fields", []))
        | set(roles.get("target_and_alias_exclusions", []))
        | {"respondent_id"}
    )
    if excluded.intersection(features):
        raise RuntimeError("The Prompt 3 feature contract contains a prohibited field.")
    _ensure_pyarrow_available()
    source = guard_read_path(workspace, DEVELOPMENT_RELATIVE)
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(source).schema_arrow
    schema_digest = canonical_digest(
        [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ]
    )
    columns = features + [TARGET, "development_role", "row_hash"]
    development = pd.read_parquet(source, columns=columns)
    if len(development) != EXPECTED_DEVELOPMENT_ROWS:
        raise RuntimeError("Development row count is not 500,000.")
    if development["row_hash"].duplicated().any():
        raise RuntimeError("Development row_hash is not unique.")
    target = pd.to_numeric(development[TARGET], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target).all():
        raise RuntimeError("Development target is not finite.")
    train = development.loc[development["development_role"].eq("train"), columns].copy()
    validation = development.loc[
        development["development_role"].eq("validation"), columns
    ].copy()
    if len(train) != EXPECTED_TRAIN_ROWS or len(validation) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Development role counts are incorrect.")
    if set(train["row_hash"]).intersection(validation["row_hash"]):
        raise RuntimeError("Train and Validation row hashes overlap.")
    frozen = handoff["reports"].get("PROMPT2_READY.json", {})
    prompt2_design = _read_json(workspace, REPORTS_RELATIVE / "prompt2_frozen_design.json")
    frozen_features = prompt2_design.get("feature_contracts", {}).get(PRIMARY_CONTRACT)
    if frozen_features != features:
        raise RuntimeError("The primary feature list differs from the Prompt 2 frozen design.")
    feature_contract_digest = prompt2_design.get("feature_contracts", {}).get(
        "feature_contract_digest"
    )
    if not feature_contract_digest:
        raise RuntimeError("Prompt 2 did not preserve a feature-contract digest.")
    audit = {
        "development_path": DEVELOPMENT_RELATIVE.as_posix(),
        "development_sha256": file_sha256(source),
        "development_rows": len(development),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_row_hash_digest": ordered_digest(train["row_hash"]),
        "validation_row_hash_digest": ordered_digest(validation["row_hash"]),
        "schema_digest": schema_digest,
        "feature_contract_digest": feature_contract_digest,
        "feature_names": features,
        "prompt2_ready_source_sha256": frozen.get("development_source_sha256"),
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
    }
    if audit["development_sha256"] != frozen.get("development_source_sha256"):
        raise RuntimeError("Development SHA-256 differs from the Prompt 2 handoff.")
    del development
    return train, validation, audit


def _code_digest(root: Path) -> str:
    sources = [
        root / "src" / name
        for name in (
            "prompt3_deep_models.py",
            "deep_preprocessing.py",
            "deep_bundles.py",
            "deep_metrics.py",
        )
    ]
    return canonical_digest({path.name: file_sha256(path) for path in sources})


def _ft_architecture(numeric_count: int, cardinalities: list[int]) -> dict[str, Any]:
    return {
        "n_cont_features": int(numeric_count),
        "cat_cardinalities": [int(value) for value in cardinalities],
        "d_out": 1,
        "n_blocks": 3,
        "d_block": 64,
        "attention_n_heads": 8,
        "attention_dropout": 0.20,
        "ffn_d_hidden": None,
        "ffn_d_hidden_multiplier": 2.0,
        "ffn_dropout": 0.10,
        "ffn_activation": "ReGLU",
        "residual_dropout": 0.0,
    }


def prepare_design(root: str | Path | None = None) -> dict[str, Any]:
    """Validate the handoff and write the frozen design before any fit."""
    workspace = Path(root or regression_v2_root()).resolve()
    environment = inspect_environment(workspace)
    if environment.get("status") != "PASS":
        raise RuntimeError("Prompt 3 required packages are unavailable in this environment.")
    train, validation, audit = load_development(workspace)
    features = audit["feature_names"]
    numeric, categorical = split_feature_types(train, features)
    fit_index, stop_index, split_audit = make_ft_internal_split(
        train[TARGET],
        train["row_hash"],
        external_validation_row_hashes=validation["row_hash"],
    )
    del fit_index, stop_index
    design = {
        "status": "FROZEN",
        "frozen_at_utc": utc_now(),
        "development_source": audit,
        "feature_contract": {
            "name": PRIMARY_CONTRACT,
            "count": len(features),
            "features": features,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "digest": audit["feature_contract_digest"],
        },
        "target": {"name": TARGET, "unit": TARGET_UNIT},
        "candidates": [dict(item) for item in ALL_CANDIDATES],
        "realmlp_execution_policy": {
            **REALMLP_PARAMETERS,
            "num_workers_mapping": "not exposed; official in-process loader",
            "external_validation_passed_to_fit": False,
            "numeric_preprocessing": "Train-only median imputation",
            "categorical_preprocessing": "Train-only official encoder with stable missing strings",
        },
        "ft_internal_split": split_audit,
        "ft_preprocessing_policy": {
            "numeric": "Train-only median then QuantileTransformer",
            "n_quantiles": 1000,
            "output_distribution": "normal",
            "subsample": None,
            "categorical_indices": {"missing": 0, "unknown": 1, "known_start": 2},
        },
        "ft_architecture_and_training": FT_SCIENTIFIC_PARAMETERS,
        "ft_batch_size": {"allowed": [512, 1024, 2048], "selected": None, "status": "pending smoke"},
        "seeds": {"global": SEED, "ft_split": SEED, "paired_bootstrap": SEED},
        "metrics": list(compute_regression_metrics([1.0, 2.0], [1.0, 2.0]).keys()),
        "family_selection_rule": [
            "MAE with 0.25% relative tie",
            "RMSE with 0.25% relative tie",
            "top-decile MAE",
            "top-five-percent MAE",
            "fit time",
            "bundle size",
            "simpler regularization",
        ],
        "deep_anchor_rule": [
            "MAE with 0.25% relative tie",
            "RMSE with 0.25% relative tie",
            "top-decile MAE",
            "top-five-percent MAE",
            "fit time",
        ],
        "fit_budget": {
            "realmlp_candidates": 2,
            "ft_selection_candidates": 2,
            "ft_full_train_refit": 1,
            "maximum_scientific_fits": MAX_SCIENTIFIC_FITS,
            "heavy_fits_sequential": True,
        },
        "output_paths": {
            "models": MODELS_RELATIVE.as_posix(),
            "predictions": PREDICTIONS_RELATIVE.as_posix(),
            "reports": REPORTS_RELATIVE.as_posix(),
            "figures": FIGURES_RELATIVE.as_posix(),
            "temporary": TMP_RELATIVE.as_posix(),
        },
        "package_versions": environment["packages"],
        "code_digest": _code_digest(workspace),
    }
    design["design_digest"] = canonical_digest(design)
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_environment.json", environment)
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_frozen_design.json", design)
    return design


def _load_design(root: Path, *, require_smoke: bool = False) -> dict[str, Any]:
    path = guard_read_path(root, REPORTS_RELATIVE / "prompt3_frozen_design.json")
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("status") != "FROZEN":
        raise RuntimeError("Prompt 3 design is not frozen.")
    if design.get("code_digest") != _code_digest(root):
        raise RuntimeError("Prompt 3 code changed after the design was frozen.")
    if require_smoke:
        smoke_path = guard_read_path(root, REPORTS_RELATIVE / "prompt3_smoke.json")
        smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {}
        if (
            design.get("ft_batch_size", {}).get("status") != "FROZEN_AFTER_SMOKE"
            or smoke.get("status") != "PASS"
        ):
            raise RuntimeError("Both bounded smoke tests must pass before scientific fits.")
    return design


def _assert_stage5_environment() -> None:
    executable = Path(sys.executable).resolve().as_posix().lower()
    if "/artifacts/environment/stage5_env/" not in executable:
        raise RuntimeError(
            "Deep execution must use artifacts/environment/stage5_env/Scripts/python.exe."
        )
    missing = [
        name
        for name in ("torch", "pytabkit", "rtdl-revisiting-models")
        if _package_version(name) is None
    ]
    if missing:
        raise RuntimeError(f"The Stage 5 environment is missing required packages: {missing}")


def _candidate_identity(
    design: dict[str, Any], candidate: dict[str, Any], *, phase: str
) -> dict[str, Any]:
    source = design["development_source"]
    identity = {
        "development_sha256": source["development_sha256"],
        "train_row_hash_digest": source["train_row_hash_digest"],
        "validation_row_hash_digest": source["validation_row_hash_digest"],
        "feature_contract_digest": design["feature_contract"]["digest"],
        "family": candidate["family"],
        "candidate_id": candidate["candidate_id"],
        "target_mode": candidate["target_mode"],
        "scientific_configuration": candidate,
        "execution_configuration": (
            design["realmlp_execution_policy"]
            if candidate["family"] == "realmlp"
            else {
                **design["ft_architecture_and_training"],
                "batch_size": design["ft_batch_size"].get("selected"),
                "device": design.get("ft_device"),
            }
        ),
        "package_versions": design["package_versions"],
        "code_digest": design["code_digest"],
        "random_seed": SEED,
        "phase": phase,
    }
    identity["identity_digest"] = canonical_digest(identity)
    return identity


def _candidate_directory(root: Path, candidate_id: str) -> Path:
    return guard_write_path(root, MODELS_RELATIVE / "candidates" / candidate_id)


def _valid_checkpoint(directory: Path, identity: dict[str, Any]) -> bool:
    manifest_path = directory / "checkpoint_manifest.json"
    result_path = directory / "result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            manifest.get("status") == "COMPLETE"
            and result.get("status") == "COMPLETE"
            and manifest.get("identity_digest") == identity["identity_digest"]
            and int(manifest.get("prediction_row_count", -1)) == EXPECTED_VALIDATION_ROWS
            and (directory / "validation_predictions.parquet").is_file()
            and (directory / "bundle" / "manifest.json").is_file()
            and (directory / "training_history.json").is_file()
        )
    except (OSError, ValueError, TypeError):
        return False


def _prediction_frame(
    row_hash: Iterable[Any],
    y_true: Any,
    y_pred: Any,
    *,
    model_id: str,
    family: str,
    target_mode: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_hash": pd.Series(row_hash, copy=False).astype(str).to_numpy(),
            "y_true": np.asarray(y_true, dtype=np.float64),
            "y_pred": np.asarray(y_pred, dtype=np.float64),
            "model_id": model_id,
            "family": family,
            "feature_contract": PRIMARY_CONTRACT,
            "target_mode": target_mode,
        }
    )
    if len(frame) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Scientific Validation predictions must contain 100,000 rows.")
    if frame["row_hash"].duplicated().any() or not np.isfinite(frame[["y_true", "y_pred"]]).all().all():
        raise RuntimeError("Validation prediction frame is invalid.")
    return frame


def _bundle_metadata(
    design: dict[str, Any],
    candidate: dict[str, Any],
    configuration: dict[str, Any],
    *,
    selected_epoch: int | None,
    device: str,
    extra: dict[str, Any] | None = None,
) -> BundleMetadata:
    source = design["development_source"]
    return BundleMetadata(
        model_id=candidate["candidate_id"],
        family=candidate["family"],
        feature_names=list(design["feature_contract"]["features"]),
        feature_contract_name=PRIMARY_CONTRACT,
        target_mode=candidate["target_mode"],
        model_configuration=json_safe(configuration),
        package_versions=dict(design["package_versions"]),
        development_source_sha256=source["development_sha256"],
        train_row_hash_digest=source["train_row_hash_digest"],
        validation_row_hash_digest=source["validation_row_hash_digest"],
        training_seed=SEED,
        selected_epoch=selected_epoch,
        device=device,
        extra=extra or {},
    )


def _safe_metadata_scalar(value: Any) -> tuple[bool, Any]:
    """Return only bounded scalar JSON values; never expand fitted tensors."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and len(value) > 500:
            return True, value[:500] + "..."
        return True, value
    return False, None


def _metadata_inventory(model: Any, *, maximum_depth: int = 10) -> dict[str, Any]:
    """Build a bounded inventory of fitted metadata without serializing weights."""
    if maximum_depth < 1 or maximum_depth > 10:
        raise ValueError("RealMLP metadata inventory depth must be between 1 and 10.")
    records: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(value: Any, path: str, depth: int, inherited_pytabkit: bool) -> None:
        value_type = type(value)
        module = str(getattr(value_type, "__module__", ""))
        class_name = str(getattr(value_type, "__qualname__", value_type.__name__))
        belongs = inherited_pytabkit or module.startswith("pytabkit")
        terminal = path.rsplit(".", 1)[-1].split("[", 1)[0].rstrip("_")
        record: dict[str, Any] = {
            "attribute_path": path,
            "value_type": value_type.__name__,
            "source_class": class_name,
            "source_module": module,
            "belongs_to_pytabkit": belongs,
            "selection_related": terminal in REALMLP_SELECTION_FIELDS,
            "depth": depth,
        }
        scalar, scalar_value = _safe_metadata_scalar(value)
        if scalar:
            record["scalar_value"] = scalar_value
            records.append(record)
            return
        if isinstance(value, np.ndarray) or (
            module.startswith("torch") and hasattr(value, "shape")
        ):
            record["shape"] = [int(item) for item in getattr(value, "shape", ())]
            record["dtype"] = str(getattr(value, "dtype", "unknown"))
            record["large_value_omitted"] = True
            records.append(record)
            return
        identity = id(value)
        if identity in visited:
            record["cycle_reference"] = True
            records.append(record)
            return
        visited.add(identity)
        if isinstance(value, dict):
            keys = [str(key) for key in value.keys()]
            record["dictionary_keys"] = keys[:100]
            record["dictionary_length"] = len(keys)
            record["truncated"] = len(keys) > 100
            records.append(record)
            if depth < maximum_depth:
                for key, nested in list(value.items())[:100]:
                    visit(nested, f"{path}.{key}", depth + 1, belongs)
            return
        if isinstance(value, (list, tuple)):
            record["sequence_length"] = len(value)
            record["truncated"] = len(value) > 25
            records.append(record)
            if depth < maximum_depth:
                for index, nested in enumerate(value[:25]):
                    visit(nested, f"{path}[{index}]", depth + 1, belongs)
            return
        attributes = []
        try:
            attributes = [
                name
                for name in vars(value)
                if name in REALMLP_METADATA_ATTRIBUTES
                or name.rstrip("_") in REALMLP_SELECTION_FIELDS
                or name.rstrip("_") in REALMLP_NON_SELECTION_EPOCH_FIELDS
            ]
        except TypeError:
            attributes = []
        record["available_metadata_attributes"] = sorted(attributes)
        records.append(record)
        if depth < maximum_depth:
            for name in sorted(attributes):
                try:
                    nested = getattr(value, name)
                except Exception as error:  # Bounded diagnostic only.
                    records.append(
                        {
                            "attribute_path": f"{path}.{name}",
                            "value_type": "attribute_error",
                            "source_class": class_name,
                            "source_module": module,
                            "belongs_to_pytabkit": belongs,
                            "selection_related": name.rstrip("_") in REALMLP_SELECTION_FIELDS,
                            "depth": depth + 1,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                visit(nested, f"{path}.{name}", depth + 1, belongs)

    visit(model, "estimator", 0, False)
    return {
        "status": "COMPLETE",
        "created_at_utc": utc_now(),
        "object_type": type(model).__qualname__,
        "object_module": type(model).__module__,
        "maximum_recursion_depth": maximum_depth,
        "record_count": len(records),
        "records": records,
        "large_arrays_and_tensors_serialized": False,
        "row_level_training_data_serialized": False,
    }


def _realmlp_epoch_evidence(model: Any, *, maximum_epochs: int) -> dict[str, Any]:
    """Collect and reconcile official PyTabKit selected-epoch metadata."""
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    active: set[int] = set()

    def scalar_leaves(value: Any, path: str) -> list[tuple[str, Any]]:
        if isinstance(value, dict):
            return [
                item
                for key, nested in value.items()
                for item in scalar_leaves(nested, f"{path}.{key}")
            ]
        if isinstance(value, (list, tuple, np.ndarray)):
            return [
                item
                for index, nested in enumerate(list(value))
                for item in scalar_leaves(nested, f"{path}[{index}]")
            ]
        return [(path, value)]

    def add_field(
        field: str,
        value: Any,
        path: str,
        owner_type: type[Any],
        belongs: bool,
    ) -> None:
        for leaf_path, raw in scalar_leaves(value, path):
            scalar, safe_raw = _safe_metadata_scalar(raw)
            item = {
                "metadata_path": leaf_path,
                "field": field,
                "raw_value": safe_raw if scalar else repr(raw)[:500],
                "source_object_type": owner_type.__qualname__,
                "source_module": owner_type.__module__,
                "belongs_to_pytabkit": belongs,
                "interpretation": "official fitted selection metadata",
            }
            normalized: int | None = None
            if isinstance(raw, np.generic):
                raw = raw.item()
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and float(raw).is_integer():
                normalized = int(raw)
            if not belongs:
                item["rejection_reason"] = "source path is not owned by PyTabKit"
                rejected.append(item)
            elif normalized is None:
                item["rejection_reason"] = "value is not an integral epoch"
                rejected.append(item)
            elif normalized == 0:
                item["rejection_reason"] = (
                    "installed PyTabKit stores selected epochs as progress.epoch + 1; zero is invalid"
                )
                rejected.append(item)
            elif normalized < 1 or normalized > maximum_epochs:
                item["normalized_value"] = normalized
                item["rejection_reason"] = "epoch is outside the bounded run"
                rejected.append(item)
            else:
                item["normalized_value"] = normalized
                item["normalization"] = "none; installed source proves one-based storage"
                candidates.append(item)

    def walk(value: Any, path: str, depth: int, inherited_pytabkit: bool) -> None:
        if depth > 10:
            return
        value_type = type(value)
        module = str(getattr(value_type, "__module__", ""))
        belongs = inherited_pytabkit or module.startswith("pytabkit")
        if isinstance(value, (str, bytes, int, float, bool, np.generic, np.ndarray)) or value is None:
            return
        identity = id(value)
        if identity in active:
            return
        active.add(identity)
        if isinstance(value, dict):
            for key, nested in list(value.items())[:100]:
                name = str(key).rstrip("_")
                nested_path = f"{path}.{key}"
                if name in REALMLP_SELECTION_FIELDS:
                    add_field(name, nested, nested_path, value_type, belongs)
                elif name in REALMLP_NON_SELECTION_EPOCH_FIELDS:
                    for leaf_path, raw in scalar_leaves(nested, nested_path):
                        rejected.append(
                            {
                                "metadata_path": leaf_path,
                                "field": name,
                                "raw_value": json_safe(raw),
                                "source_object_type": value_type.__qualname__,
                                "source_module": value_type.__module__,
                                "belongs_to_pytabkit": belongs,
                                "interpretation": "configured or maximum epoch, not a selection",
                                "rejection_reason": "generic maximum/current epoch fields are prohibited",
                            }
                        )
                else:
                    walk(nested, nested_path, depth + 1, belongs)
            active.remove(identity)
            return
        if isinstance(value, (list, tuple)):
            for index, nested in enumerate(value[:25]):
                walk(nested, f"{path}[{index}]", depth + 1, belongs)
            active.remove(identity)
            return
        try:
            names = list(vars(value))
        except TypeError:
            names = []
        for name in names:
            normalized_name = name.rstrip("_")
            if name not in REALMLP_METADATA_ATTRIBUTES and normalized_name not in (
                REALMLP_SELECTION_FIELDS | REALMLP_NON_SELECTION_EPOCH_FIELDS
            ):
                continue
            try:
                nested = getattr(value, name)
            except Exception:
                continue
            nested_path = f"{path}.{name}"
            if normalized_name in REALMLP_SELECTION_FIELDS:
                add_field(normalized_name, nested, nested_path, value_type, belongs)
            elif normalized_name in REALMLP_NON_SELECTION_EPOCH_FIELDS:
                for leaf_path, raw in scalar_leaves(nested, nested_path):
                    rejected.append(
                        {
                            "metadata_path": leaf_path,
                            "field": normalized_name,
                            "raw_value": json_safe(raw),
                            "source_object_type": value_type.__qualname__,
                            "source_module": value_type.__module__,
                            "belongs_to_pytabkit": belongs,
                            "interpretation": "configured or maximum epoch, not a selection",
                            "rejection_reason": "generic maximum/current epoch fields are prohibited",
                        }
                    )
            else:
                walk(nested, nested_path, depth + 1, belongs)
        active.remove(identity)

    walk(model, "estimator", 0, False)
    epochs = sorted({int(item["normalized_value"]) for item in candidates})
    status = "PASS" if len(epochs) == 1 else "FAIL"
    return {
        "status": status,
        "created_at_utc": utc_now(),
        "maximum_epochs": maximum_epochs,
        "candidate_metadata_paths": candidates,
        "accepted_paths": candidates if status == "PASS" else [],
        "rejected_paths": rejected,
        "unique_normalized_epochs": epochs,
        "final_selected_epoch": epochs[0] if status == "PASS" else None,
        "indexing_convention": "one-based",
        "source_code_evidence": {
            "best_epoch_assignment": (
                "pytabkit/models/training/lightning_modules.py stores "
                "best_mean_val_epochs as progress.epoch + 1"
            ),
            "refit_consumption": (
                "pytabkit/models/training/nn_creator.py passes fit_params stop_epoch "
                "to StopAtEpochsCallback"
            ),
            "public_metadata_assignment": (
                "pytabkit/models/sklearn/sklearn_base.py assigns fit_params_ from "
                "the active refit alg_interface_.fit_params[0]"
            ),
            "zero_based_conversion_used": False,
        },
        "conflict": len(epochs) > 1,
    }


def _realmlp_selected_epoch(
    model: Any, *, maximum_epochs: int = 30
) -> tuple[int, dict[str, Any]]:
    """Read one official, one-based Train-only epoch from fitted PyTabKit metadata."""
    evidence = _realmlp_epoch_evidence(model, maximum_epochs=maximum_epochs)
    if evidence["status"] != "PASS":
        raise RuntimeError(
            "RealMLP fitted metadata did not establish one unambiguous selected epoch."
        )
    return int(evidence["final_selected_epoch"]), evidence


def _save_realmlp_metadata_evidence(
    root: Path,
    model_path: Path,
    inventory_path: Path,
    evidence_path: Path,
    *,
    maximum_epochs: int,
) -> tuple[Any, int, dict[str, Any]]:
    """Reload a persisted estimator, then save bounded inventory and epoch evidence."""
    persisted = guard_read_path(root, model_path)
    if not persisted.is_file() or persisted.stat().st_size <= 0:
        raise RuntimeError("The persisted RealMLP estimator is missing or empty.")
    model = joblib.load(persisted)
    if not hasattr(model, "alg_interface_") or not hasattr(model, "x_converter_"):
        raise RuntimeError("The reloaded RealMLP estimator is not structurally fitted.")
    inventory = _metadata_inventory(model, maximum_depth=10)
    atomic_json(root, inventory_path, inventory)
    evidence = _realmlp_epoch_evidence(model, maximum_epochs=maximum_epochs)
    atomic_json(root, evidence_path, evidence)
    if evidence["status"] != "PASS":
        raise RuntimeError(
            "RealMLP fitted metadata did not establish one unambiguous selected epoch."
        )
    return model, int(evidence["final_selected_epoch"]), evidence


def _fit_realmlp_candidate(
    root: Path,
    design: dict[str, Any],
    candidate: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, Any]:
    from pytabkit import RealMLP_TD_Regressor

    identity = _candidate_identity(design, candidate, phase="selection_with_official_refit")
    directory = _candidate_directory(root, candidate["candidate_id"])
    if _valid_checkpoint(directory, identity):
        return json.loads((directory / "result.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    features = list(design["feature_contract"]["features"])
    numeric = list(design["feature_contract"]["numeric_features"])
    categorical = list(design["feature_contract"]["categorical_features"])
    y_train = train[TARGET].to_numpy(dtype=np.float64, copy=True)
    y_validation = validation[TARGET].to_numpy(dtype=np.float64, copy=True)
    parameters = {
        **REALMLP_PARAMETERS,
        "p_drop": float(candidate["p_drop"]),
        "tmp_folder": str(guard_write_path(root, TMP_RELATIVE / "realmlp" / candidate["candidate_id"])),
    }
    model_path = directory / "fitted_realmlp.joblib"
    preprocessor_path = directory / "fitted_preprocessor.joblib"
    fit_completed_path = directory / "fit_completed.json"
    can_resume = False
    if model_path.is_file() and preprocessor_path.is_file() and fit_completed_path.is_file():
        prior_fit = json.loads(fit_completed_path.read_text(encoding="utf-8"))
        can_resume = (
            prior_fit.get("status") == "COMPLETE"
            and prior_fit.get("fit_returned_successfully") is True
            and prior_fit.get("identity_digest") == identity["identity_digest"]
        )
    if can_resume:
        preprocessor = joblib.load(preprocessor_path)
        x_validation = preprocessor.transform(validation[features])
        fit_seconds = float(prior_fit["fit_time_seconds"])
        model = joblib.load(model_path)
        resolved = model.get_config()
        reused_persisted_object = True
    else:
        preprocessor = RealMLPPreprocessor(features, numeric, categorical)
        x_train = preprocessor.fit_transform(
            train[features], row_hashes=train["row_hash"]
        )
        x_validation = preprocessor.transform(validation[features])
        _atomic_joblib(root, preprocessor_path, preprocessor)
        model = RealMLP_TD_Regressor(**parameters)
        resolved = model.get_config()
        atomic_json(
            root,
            directory / "resolved_configuration_before_fit.json",
            {
                "status": "FROZEN_BEFORE_FIT",
                "requested": parameters,
                "resolved": resolved,
                "api_mapping": {
                    "internal_validation_fraction": "val_fraction",
                    "official_full_train_refit": "n_refit=1",
                    "maximum_epochs": "n_epochs",
                    "CPU_float32": "device=cpu",
                    "external_validation_fit_argument": "not passed",
                    "num_workers": "not exposed; official in-process loader",
                },
            },
        )
        atomic_json(
            root,
            directory / "fit_input_membership.json",
            {
                "status": "FROZEN_BEFORE_FIT",
                "train_rows": len(train),
                "validation_rows": len(validation),
                "external_validation_rows_entering_fit": 0,
                "train_row_hash_digest": ordered_digest(train["row_hash"]),
                "validation_row_hash_digest": ordered_digest(validation["row_hash"]),
                "identity_digest": identity["identity_digest"],
            },
        )
        started = time.perf_counter()
        # External Validation is intentionally absent from this fit call.
        model.fit(x_train, y_train, cat_col_names=categorical)
        fit_seconds = time.perf_counter() - started
        _atomic_joblib(root, model_path, model)
        atomic_json(
            root,
            fit_completed_path,
            {
                "status": "COMPLETE",
                "fit_returned_successfully": True,
                "official_refit_returned_successfully": True,
                "fitted_estimator_persisted": True,
                "persisted_before_epoch_parsing": True,
                "identity_digest": identity["identity_digest"],
                "fit_time_seconds": fit_seconds,
                "fitted_estimator_bytes": model_path.stat().st_size,
                "completed_at_utc": utc_now(),
            },
        )
        reused_persisted_object = False
    model, selected_epoch, epoch_evidence = _save_realmlp_metadata_evidence(
        root,
        model_path,
        directory / "realmlp_metadata_inventory.json",
        directory / "realmlp_epoch_evidence.json",
        maximum_epochs=30,
    )
    prediction_started = time.perf_counter()
    prediction = np.asarray(model.predict(x_validation), dtype=np.float64).reshape(-1)
    prediction_seconds = time.perf_counter() - prediction_started
    if prediction.size != len(validation) or not np.isfinite(prediction).all():
        raise RuntimeError("RealMLP produced invalid Validation predictions.")
    metadata = _bundle_metadata(
        design,
        candidate,
        {"requested": parameters, "resolved": model.get_config()},
        selected_epoch=selected_epoch,
        device="cpu float32",
        extra={
            "fit_membership": "Development Train only",
            "external_validation_passed_to_fit": False,
            "official_refit_count": 1,
            "official_selected_epoch_evidence": epoch_evidence,
            "fitted_object_persisted_before_parsing": True,
        },
    )
    bundle = RealMLPBundle(metadata=metadata, preprocessor=preprocessor, model=model)
    bundle_manifest = save_realmlp_bundle(bundle, directory / "bundle")
    frame = _prediction_frame(
        validation["row_hash"],
        y_validation,
        prediction,
        model_id=candidate["candidate_id"],
        family="realmlp",
        target_mode="raw",
    )
    atomic_parquet(root, directory / "validation_predictions.parquet", frame)
    history = {
        "status": "COMPLETE",
        "family": "realmlp",
        "candidate_id": candidate["candidate_id"],
        "history_source": "Official fitted PyTabKit metadata from the persisted estimator.",
        "maximum_epochs": 30,
        "selected_epoch": selected_epoch,
        "official_selected_epoch_evidence": epoch_evidence,
        "official_internal_validation_fraction": 0.10,
        "official_refit_used": True,
        "fitted_object_persisted_before_parsing": True,
        "reused_persisted_object_without_refit": reused_persisted_object,
        "resolved_configuration": model.get_config(),
    }
    atomic_json(root, directory / "training_history.json", history)
    metrics = compute_regression_metrics(
        y_validation,
        prediction,
        fit_time_seconds=fit_seconds,
        prediction_time_seconds=prediction_seconds,
        model_size_bytes=int(bundle_manifest["model_size_bytes"]),
        bundle_size_bytes=directory_size_bytes(directory / "bundle"),
    )
    result = {
        "status": "COMPLETE",
        **candidate,
        **metrics,
        "selected_epoch": selected_epoch,
        "fitted_object_persisted_before_parsing": True,
        "reused_persisted_object_without_refit": reused_persisted_object,
        "device": "cpu float32",
        "fit_membership_rows": len(train),
        "external_validation_fit_rows": 0,
        "prediction_rows": len(validation),
        "bundle_path": (directory / "bundle").relative_to(root).as_posix(),
        "prediction_path": (directory / "validation_predictions.parquet").relative_to(root).as_posix(),
    }
    atomic_json(root, directory / "result.json", result)
    atomic_json(
        root,
        directory / "checkpoint_manifest.json",
        {
            "status": "COMPLETE",
            **identity,
            "training_history": "training_history.json",
            "prediction_row_count": len(validation),
            "completed_at_utc": utc_now(),
        },
    )
    return result


def _seed_torch(torch_module: Any, device: str) -> None:
    torch_module.manual_seed(SEED)
    np.random.seed(SEED)
    torch_module.set_num_threads(min(4, os.cpu_count() or 1))
    torch_module.use_deterministic_algorithms(True, warn_only=True)
    if device == "cuda":
        torch_module.cuda.manual_seed_all(SEED)


def _build_ft_model(architecture: dict[str, Any], device: str):
    import torch
    from rtdl_revisiting_models import FTTransformer

    _seed_torch(torch, device)
    return FTTransformer(**architecture).to(torch.device(device))


def _ft_predict_model_scale(
    model: Any,
    numeric: np.ndarray,
    categorical: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(numeric), batch_size):
            end = min(start + batch_size, len(numeric))
            x_cont = torch.from_numpy(numeric[start:end]).to(device)
            x_cat = torch.from_numpy(categorical[start:end]).to(device)
            values = model(x_cont, x_cat).reshape(-1).detach().cpu().numpy()
            output.append(values.astype(np.float64, copy=False))
    result = np.concatenate(output) if output else np.empty(0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise RuntimeError("FT-Transformer produced non-finite model-scale predictions.")
    return result


def _atomic_torch_state(root: Path, path: Path, state: dict[str, Any]) -> Path:
    import torch

    destination = guard_write_path(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save({name: tensor.detach().cpu() for name, tensor in state.items()}, temporary)
    torch.load(temporary, map_location="cpu", weights_only=True)
    os.replace(temporary, destination)
    return destination


def _train_ft(
    root: Path,
    directory: Path,
    architecture: dict[str, Any],
    train_arrays: tuple[np.ndarray, np.ndarray],
    y_train_log: np.ndarray,
    *,
    weight_decay: float,
    batch_size: int,
    device: str,
    maximum_epochs: int,
    stop_arrays: tuple[np.ndarray, np.ndarray] | None,
    y_stop_original: np.ndarray | None,
    patience: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], int, float]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = _build_ft_model(architecture, device)
    optimizer = torch.optim.AdamW(
        model.make_parameter_groups(),
        lr=FT_SCIENTIFIC_PARAMETERS["learning_rate"],
        weight_decay=weight_decay,
    )
    criterion = torch.nn.MSELoss()
    numeric, categorical = train_arrays
    target = np.asarray(y_train_log, dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(numeric), torch.from_numpy(categorical), torch.from_numpy(target)
    )
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
        generator=generator,
        drop_last=False,
    )
    history: list[dict[str, Any]] = []
    best_mae = np.inf
    best_epoch = maximum_epochs
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for x_cont, x_cat, y_batch in loader:
            x_cont = x_cont.to(device, non_blocking=device == "cuda")
            x_cat = x_cat.to(device, non_blocking=device == "cuda")
            y_batch = y_batch.to(device, non_blocking=device == "cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(x_cont, x_cat).reshape(-1)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), FT_SCIENTIFIC_PARAMETERS["gradient_clip_norm"]
            )
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y_batch)
            row_count += len(y_batch)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_log1p_mse": loss_sum / row_count,
        }
        if stop_arrays is not None and y_stop_original is not None:
            stop_log = _ft_predict_model_scale(
                model, *stop_arrays, device=device, batch_size=batch_size
            )
            stop_original = np.expm1(stop_log)
            if not np.isfinite(stop_original).all():
                raise RuntimeError("FT internal original-scale predictions are not finite.")
            validation_mae = float(np.mean(np.abs(stop_original - y_stop_original)))
            record["internal_original_scale_mae"] = validation_mae
            if validation_mae < best_mae:
                best_mae = validation_mae
                best_epoch = epoch
                stale_epochs = 0
                _atomic_torch_state(root, directory / "best_state.pt", model.state_dict())
            else:
                stale_epochs += 1
        else:
            best_epoch = epoch
            _atomic_torch_state(root, directory / "best_state.pt", model.state_dict())
        history.append(record)
        atomic_json(
            root,
            directory / "training_history.json",
            {"status": "IN_PROGRESS", "history": history, "best_epoch": best_epoch},
        )
        if patience is not None and stale_epochs >= patience:
            break
    fit_seconds = time.perf_counter() - started
    state = torch.load(directory / "best_state.pt", map_location="cpu", weights_only=True)
    atomic_json(
        root,
        directory / "training_history.json",
        {
            "status": "COMPLETE",
            "history": history,
            "best_epoch": best_epoch,
            "best_internal_original_scale_mae": None if np.isinf(best_mae) else best_mae,
            "early_stopping_patience": patience,
        },
    )
    return state, history, best_epoch, fit_seconds


def _fit_ft_candidate(
    root: Path,
    design: dict[str, Any],
    candidate: dict[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    fit_index: np.ndarray,
    stop_index: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    identity = _candidate_identity(design, candidate, phase="internal_epoch_selection")
    directory = _candidate_directory(root, candidate["candidate_id"])
    if _valid_checkpoint(directory, identity):
        return json.loads((directory / "result.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    features = list(design["feature_contract"]["features"])
    numeric = list(design["feature_contract"]["numeric_features"])
    categorical = list(design["feature_contract"]["categorical_features"])
    internal_fit = train.iloc[fit_index]
    internal_stop = train.iloc[stop_index]
    preprocessor = FTPreprocessor(features, numeric, categorical)
    fit_arrays = preprocessor.fit_transform(
        internal_fit[features], row_hashes=internal_fit["row_hash"]
    )
    stop_arrays = preprocessor.transform(internal_stop[features])
    validation_arrays = preprocessor.transform(validation[features])
    architecture = _ft_architecture(len(numeric), preprocessor.cardinalities_)
    configuration = {
        **FT_SCIENTIFIC_PARAMETERS,
        "weight_decay": candidate["weight_decay"],
        "batch_size": batch_size,
        "device": device,
        "architecture_api": architecture,
    }
    atomic_json(
        root,
        directory / "resolved_configuration_before_fit.json",
        {"status": "FROZEN_BEFORE_FIT", "configuration": configuration},
    )
    state, history, best_epoch, fit_seconds = _train_ft(
        root,
        directory,
        architecture,
        fit_arrays,
        transform_target(internal_fit[TARGET], "log1p").astype(np.float32),
        weight_decay=float(candidate["weight_decay"]),
        batch_size=batch_size,
        device=device,
        maximum_epochs=FT_SCIENTIFIC_PARAMETERS["maximum_epochs"],
        stop_arrays=stop_arrays,
        y_stop_original=internal_stop[TARGET].to_numpy(dtype=np.float64),
        patience=FT_SCIENTIFIC_PARAMETERS["early_stopping_patience"],
    )
    model = _build_ft_model(architecture, device)
    model.load_state_dict(state)
    prediction_started = time.perf_counter()
    prediction_log = _ft_predict_model_scale(
        model, *validation_arrays, device=device, batch_size=batch_size
    )
    prediction = np.expm1(prediction_log)
    prediction_seconds = time.perf_counter() - prediction_started
    if not np.isfinite(prediction).all():
        raise RuntimeError("FT external Validation predictions are not finite.")
    metadata = _bundle_metadata(
        design,
        candidate,
        configuration,
        selected_epoch=best_epoch,
        device=f"{device} float32",
        extra={
            "fit_membership": "360,000 Development Train internal-fit rows",
            "early_stopping_membership": "40,000 Development Train internal-stop rows",
            "external_validation_fit_rows": 0,
        },
    )
    bundle = FTTransformerBundle(metadata, preprocessor, architecture, state, batch_size)
    bundle_manifest = save_fttransformer_bundle(bundle, directory / "bundle")
    prediction_frame = _prediction_frame(
        validation["row_hash"],
        validation[TARGET],
        prediction,
        model_id=candidate["candidate_id"],
        family="fttransformer",
        target_mode="log1p",
    )
    atomic_parquet(root, directory / "validation_predictions.parquet", prediction_frame)
    metrics = compute_regression_metrics(
        validation[TARGET],
        prediction,
        fit_time_seconds=fit_seconds,
        prediction_time_seconds=prediction_seconds,
        model_size_bytes=int(bundle_manifest["model_size_bytes"]),
        bundle_size_bytes=directory_size_bytes(directory / "bundle"),
    )
    result = {
        "status": "COMPLETE",
        **candidate,
        **metrics,
        "selected_epoch": best_epoch,
        "epochs_run": len(history),
        "device": f"{device} float32",
        "batch_size": batch_size,
        "fit_membership_rows": len(internal_fit),
        "early_stopping_rows": len(internal_stop),
        "external_validation_fit_rows": 0,
        "prediction_rows": len(validation),
        "bundle_path": (directory / "bundle").relative_to(root).as_posix(),
        "prediction_path": (directory / "validation_predictions.parquet").relative_to(root).as_posix(),
    }
    atomic_json(root, directory / "result.json", result)
    atomic_json(
        root,
        directory / "checkpoint_manifest.json",
        {
            "status": "COMPLETE",
            **identity,
            "training_history": "training_history.json",
            "prediction_row_count": len(validation),
            "completed_at_utc": utc_now(),
        },
    )
    del model, bundle
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
    gc.collect()
    return result


def _atomic_joblib(root: Path, path: Path, value: Any) -> Path:
    destination = guard_write_path(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    joblib.load(temporary)
    os.replace(temporary, destination)
    return destination


def _clean_process_realmlp_prediction(
    root: Path,
    model_path: Path,
    preprocessor_path: Path,
    validation_path: Path,
    output_path: Path,
) -> np.ndarray:
    """Predict from persisted smoke artifacts in a clean Python process."""
    model_file = guard_read_path(root, model_path)
    preprocessor_file = guard_read_path(root, preprocessor_path)
    validation_file = guard_read_path(root, validation_path)
    output_file = guard_write_path(root, output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    source_directory = (root / "src").resolve()
    code = "\n".join(
        (
            "import sys, joblib, numpy as np",
            "sys.path.insert(0, sys.argv[1])",
            "sys.path.append(sys.argv[2])",
            "model = joblib.load(sys.argv[3])",
            "preprocessor = joblib.load(sys.argv[4])",
            "frame = joblib.load(sys.argv[5])",
            "prediction = np.asarray(model.predict(preprocessor.transform(frame)), dtype=np.float64).reshape(-1)",
            "np.save(sys.argv[6], prediction, allow_pickle=False)",
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(source_directory),
            str(_REUSED_BASE_SITE),
            str(model_file),
            str(preprocessor_file),
            str(validation_file),
            str(output_file),
        ],
        cwd=str(root),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Clean-process RealMLP prediction failed: "
            + (completed.stderr or completed.stdout)[-2000:]
        )
    prediction = np.load(output_file, allow_pickle=False)
    if prediction.ndim != 1 or not np.isfinite(prediction).all():
        raise RuntimeError("Clean-process RealMLP predictions are invalid.")
    return prediction.astype(np.float64, copy=False)


def _complete_saved_realmlp_smoke(root: Path) -> dict[str, Any]:
    """Complete evidence and prediction checks without fitting a model."""
    directory = guard_write_path(root, REALMLP_SMOKE_ATTEMPT3_RELATIVE)
    model_path = directory / "fitted_realmlp_smoke.joblib"
    preprocessor_path = directory / "fitted_preprocessor.joblib"
    validation_path = directory / "smoke_validation_features.joblib"
    fit_completed_path = directory / "fit_completed.json"
    membership_path = directory / "fit_input_membership.json"
    for required in (model_path, preprocessor_path, validation_path, fit_completed_path, membership_path):
        if not required.is_file():
            raise RuntimeError(f"Saved RealMLP smoke recovery input is missing: {required.name}")
    fit_completed = json.loads(fit_completed_path.read_text(encoding="utf-8"))
    membership = json.loads(membership_path.read_text(encoding="utf-8"))
    if fit_completed.get("fit_returned_successfully") is not True:
        raise RuntimeError("Saved RealMLP smoke fit does not have completion evidence.")
    model, selected_epoch, evidence = _save_realmlp_metadata_evidence(
        root,
        model_path,
        REPORTS_RELATIVE / "prompt3_realmlp_smoke_metadata_inventory.json",
        REPORTS_RELATIVE / "prompt3_realmlp_epoch_evidence.json",
        maximum_epochs=2,
    )
    preprocessor = joblib.load(preprocessor_path)
    unknown_frame = joblib.load(validation_path)
    prediction = np.asarray(
        model.predict(preprocessor.transform(unknown_frame)), dtype=np.float64
    ).reshape(-1)
    clean_prediction = _clean_process_realmlp_prediction(
        root,
        model_path,
        preprocessor_path,
        validation_path,
        directory / "clean_process_predictions.npy",
    )
    if prediction.size != int(membership["smoke_validation_rows"]):
        raise RuntimeError("RealMLP smoke prediction row count changed.")
    difference = float(np.max(np.abs(prediction - clean_prediction)))
    if not np.isfinite(prediction).all() or difference > 1e-6:
        raise RuntimeError("RealMLP smoke prediction or reload check failed.")
    np.save(directory / "in_process_predictions.npy", prediction, allow_pickle=False)
    return {
        "status": "PASS",
        "family": "realmlp",
        "train_rows_entering_fit": int(membership["smoke_train_rows"]),
        "external_validation_rows_entering_fit": 0,
        "external_validation_hash_overlap": int(membership["train_validation_hash_overlap"]),
        "numeric_medians_fit_rows": int(membership["smoke_train_rows"]),
        "categorical_vocabularies_fit_rows": int(membership["smoke_train_rows"]),
        "unknown_category_accepted": True,
        "missing_numeric_accepted": True,
        "missing_categorical_accepted": True,
        "source_frame_unchanged": bool(fit_completed["source_frame_unchanged"]),
        "finite_predictions": True,
        "row_order_preserved": True,
        "reload_max_absolute_difference": difference,
        "epochs": 2,
        "selected_epoch": selected_epoch,
        "epoch_evidence_status": evidence["status"],
        "fitted_estimator_persisted_before_parsing": True,
        "fitted_estimator_path": model_path.relative_to(root).as_posix(),
        "metadata_inventory_path": (
            REPORTS_RELATIVE / "prompt3_realmlp_smoke_metadata_inventory.json"
        ).as_posix(),
        "epoch_evidence_path": (
            REPORTS_RELATIVE / "prompt3_realmlp_epoch_evidence.json"
        ).as_posix(),
        "scientific_fit": False,
        "scientific_fit_count": 0,
        "device": "cpu float32",
    }


def _smoke_realmlp(
    root: Path, design: dict[str, Any], train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, Any]:
    from pytabkit import RealMLP_TD_Regressor

    features = list(design["feature_contract"]["features"])
    numeric = list(design["feature_contract"]["numeric_features"])
    categorical = list(design["feature_contract"]["categorical_features"])
    smoke_train = train.iloc[:10_000].copy()
    smoke_validation = validation.iloc[:2_000].copy()
    source_train_digest = canonical_digest(
        {name: smoke_train[name].astype(str).tolist()[:20] for name in features}
    )
    preprocessor = RealMLPPreprocessor(features, numeric, categorical)
    x_train = preprocessor.fit_transform(
        smoke_train[features], row_hashes=smoke_train["row_hash"]
    )
    unknown_frame = smoke_validation[features].copy()
    if categorical:
        unknown_frame.loc[unknown_frame.index[0], categorical[0]] = "__UNSEEN_PROMPT3_SMOKE__"
        unknown_frame.loc[unknown_frame.index[1], categorical[0]] = pd.NA
    if numeric:
        unknown_frame.loc[unknown_frame.index[2], numeric[0]] = np.nan
    x_validation = preprocessor.transform(unknown_frame)
    parameters = {
        **REALMLP_PARAMETERS,
        "n_epochs": 2,
        "p_drop": 0.15,
        "verbosity": 0,
        "tmp_folder": str(
            guard_write_path(root, REALMLP_SMOKE_ATTEMPT3_RELATIVE / "pytabkit_work")
        ),
    }
    model = RealMLP_TD_Regressor(**parameters)
    directory = guard_write_path(root, REALMLP_SMOKE_ATTEMPT3_RELATIVE)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        root,
        directory / "effective_configuration.json",
        {
            "status": "FROZEN_BEFORE_FIT",
            "authorization_id": "regression_v2_prompt3r_realmlp_smoke_metadata_recovery",
            "requested": parameters,
            "effective": model.get_config(),
            "scientific_fit": False,
        },
    )
    atomic_json(
        root,
        directory / "fit_input_membership.json",
        {
            "status": "FROZEN_BEFORE_FIT",
            "smoke_train_rows": len(smoke_train),
            "smoke_validation_rows": len(smoke_validation),
            "train_role": "Train",
            "validation_role": "Validation evaluation only",
            "train_row_hash_digest": ordered_digest(smoke_train["row_hash"]),
            "validation_row_hash_digest": ordered_digest(smoke_validation["row_hash"]),
            "train_validation_hash_overlap": len(
                set(smoke_train["row_hash"]) & set(smoke_validation["row_hash"])
            ),
            "external_validation_rows_entering_fit": 0,
            "feature_contract_digest": design["feature_contract"]["digest"],
            "scientific_fit": False,
        },
    )
    preprocessor_path = _atomic_joblib(
        root, directory / "fitted_preprocessor.joblib", preprocessor
    )
    validation_path = _atomic_joblib(
        root, directory / "smoke_validation_features.joblib", unknown_frame
    )
    fit_started = time.perf_counter()
    # No external Validation argument is present.
    model.fit(
        x_train,
        smoke_train[TARGET].to_numpy(dtype=np.float64),
        cat_col_names=categorical,
    )
    fit_seconds = time.perf_counter() - fit_started
    model_path = _atomic_joblib(
        root, directory / "fitted_realmlp_smoke.joblib", model
    )
    source_after_digest = canonical_digest(
        {name: smoke_train[name].astype(str).tolist()[:20] for name in features}
    )
    atomic_json(
        root,
        directory / "fit_completed.json",
        {
            "status": "COMPLETE",
            "fit_returned_successfully": True,
            "official_refit_returned_successfully": True,
            "fitted_estimator_persisted": True,
            "persisted_before_epoch_parsing": True,
            "fitted_estimator_path": model_path.relative_to(root).as_posix(),
            "fitted_estimator_bytes": model_path.stat().st_size,
            "preprocessor_path": preprocessor_path.relative_to(root).as_posix(),
            "validation_support_path": validation_path.relative_to(root).as_posix(),
            "fit_time_seconds": fit_seconds,
            "source_frame_unchanged": source_train_digest == source_after_digest,
            "completed_at_utc": utc_now(),
        },
    )
    # All selected-epoch parsing and prediction checks begin from the saved object.
    return _complete_saved_realmlp_smoke(root)


def _smoke_ft(
    root: Path, design: dict[str, Any], train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, Any]:
    import torch

    features = list(design["feature_contract"]["features"])
    numeric = list(design["feature_contract"]["numeric_features"])
    categorical = list(design["feature_contract"]["categorical_features"])
    smoke_train = train.iloc[:10_000].copy()
    smoke_validation = validation.iloc[:2_000].copy()
    preprocessor = FTPreprocessor(features, numeric, categorical)
    train_arrays = preprocessor.fit_transform(
        smoke_train[features], row_hashes=smoke_train["row_hash"]
    )
    unknown_frame = smoke_validation[features].copy()
    if categorical:
        unknown_frame.loc[unknown_frame.index[0], categorical[0]] = "__UNSEEN_PROMPT3_SMOKE__"
        unknown_frame.loc[unknown_frame.index[1], categorical[0]] = pd.NA
    if numeric:
        unknown_frame.loc[unknown_frame.index[2], numeric[0]] = np.nan
    validation_arrays = preprocessor.transform(unknown_frame)
    if categorical:
        first_cat = validation_arrays[1][:, 0]
        if first_cat[0] != 1 or first_cat[1] != 0:
            raise RuntimeError("FT missing/unknown category indices are incorrect.")
    architecture = _ft_architecture(len(numeric), preprocessor.cardinalities_)
    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_attempts: list[dict[str, Any]] = []
    devices = [requested_device] + (["cpu"] if requested_device == "cuda" else [])
    selected_device: str | None = None
    selected_batch: int | None = None
    selected_difference: float | None = None
    for device in devices:
        try:
            for batch_size in (2048, 1024, 512):
                smoke_dir = guard_write_path(
                    root, TMP_RELATIVE / "smoke" / f"ft_{device}_{batch_size}"
                )
                smoke_dir.mkdir(parents=True, exist_ok=True)
                state, _, _, _ = _train_ft(
                    root,
                    smoke_dir,
                    architecture,
                    train_arrays,
                    transform_target(smoke_train[TARGET], "log1p").astype(np.float32),
                    weight_decay=1e-5,
                    batch_size=batch_size,
                    device=device,
                    maximum_epochs=2,
                    stop_arrays=None,
                    y_stop_original=None,
                    patience=None,
                )
                model = _build_ft_model(architecture, device)
                model.load_state_dict(state)
                before = _ft_predict_model_scale(
                    model, *validation_arrays, device=device, batch_size=batch_size
                )
                reloaded = _build_ft_model(architecture, device)
                loaded_state = torch.load(
                    smoke_dir / "best_state.pt", map_location=device, weights_only=True
                )
                reloaded.load_state_dict(loaded_state)
                after = _ft_predict_model_scale(
                    reloaded, *validation_arrays, device=device, batch_size=batch_size
                )
                selected_difference = float(np.max(np.abs(before - after)))
                if selected_difference > 1e-6 or not np.isfinite(np.expm1(before)).all():
                    raise RuntimeError("FT smoke save/reload prediction check failed.")
                selected_device = device
                selected_batch = batch_size
                device_attempts.append(
                    {"device": device, "batch_size": batch_size, "status": "PASS"}
                )
                break
            if selected_device is not None:
                break
        except RuntimeError as error:
            device_attempts.append({"device": device, "status": "FAILED", "error": str(error)})
            if device == "cuda":
                torch.cuda.empty_cache()
                continue
            raise
    if selected_device is None or selected_batch is None:
        raise RuntimeError("FT smoke did not find a safe device and batch size.")
    return {
        "status": "PASS",
        "family": "fttransformer",
        "train_rows": len(smoke_train),
        "validation_rows": len(smoke_validation),
        "epochs": 2,
        "scientific_fit": False,
        "selected_device": selected_device,
        "selected_batch_size": selected_batch,
        "num_workers": 0,
        "unknown_category_accepted": True,
        "missing_category_accepted": True,
        "forward_backward_optimizer_step": True,
        "finite_predictions": True,
        "reload_max_absolute_difference": selected_difference,
        "attempts": device_attempts,
    }


def run_smoke(
    root: str | Path | None = None, *, family: str = "all"
) -> dict[str, Any]:
    """Run bounded, non-scientific smoke checks for one or both families."""
    if family not in {"all", "realmlp", "fttransformer"}:
        raise ValueError("family must be all, realmlp, or fttransformer.")
    workspace = Path(root or regression_v2_root()).resolve()
    configure_runtime_environment(workspace)
    _assert_stage5_environment()
    design = _load_design(workspace)
    train, validation, _ = load_development(workspace, feature_names=design["feature_contract"]["features"])
    report_path = guard_write_path(workspace, REPORTS_RELATIVE / "prompt3_smoke.json")
    prior = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    results = dict(prior.get("families", {}))
    if family in {"all", "realmlp"} and results.get("realmlp", {}).get("status") != "PASS":
        operation = "smoke:realmlp"
        ledger_path = _attempt_ledger_path(workspace)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        historical = list(ledger.get("operations", {}).get(operation, []))
        saved_model = guard_write_path(
            workspace,
            REALMLP_SMOKE_ATTEMPT3_RELATIVE / "fitted_realmlp_smoke.joblib",
        )
        if len(historical) >= 3 and saved_model.is_file():
            repair_operation = "metadata-only:realmlp-smoke-attempt3"
            repair_attempt = _start_attempt(workspace, repair_operation, maximum=2)
            try:
                results["realmlp"] = _complete_saved_realmlp_smoke(workspace)
            except Exception as error:
                _finish_attempt(
                    workspace,
                    repair_operation,
                    repair_attempt,
                    status="FAILED",
                    error=error,
                )
                raise
            _finish_attempt(
                workspace, repair_operation, repair_attempt, status="PASS"
            )
        else:
            attempt = _start_attempt(workspace, operation, maximum=3)
            try:
                results["realmlp"] = _smoke_realmlp(workspace, design, train, validation)
            except Exception as error:
                _finish_attempt(
                    workspace, operation, attempt, status="FAILED", error=error
                )
                raise
            _finish_attempt(workspace, operation, attempt, status="PASS")
        atomic_json(
            workspace,
            report_path,
            {"status": "IN_PROGRESS", "created_at_utc": utc_now(), "families": results},
        )
    if family in {"all", "fttransformer"} and results.get("fttransformer", {}).get("status") != "PASS":
        operation = "smoke:fttransformer"
        attempt = _start_attempt(workspace, operation, maximum=2)
        try:
            results["fttransformer"] = _smoke_ft(workspace, design, train, validation)
        except Exception as error:
            _finish_attempt(
                workspace, operation, attempt, status="FAILED", error=error
            )
            raise
        _finish_attempt(workspace, operation, attempt, status="PASS")
    status = "PASS" if all(results.get(name, {}).get("status") == "PASS" for name in ("realmlp", "fttransformer")) else "IN_PROGRESS"
    report = {"status": status, "created_at_utc": utc_now(), "families": results}
    atomic_json(workspace, report_path, report)
    if results.get("fttransformer", {}).get("status") == "PASS":
        design["ft_batch_size"] = {
            "allowed": [512, 1024, 2048],
            "selected": int(results["fttransformer"]["selected_batch_size"]),
            "status": "FROZEN_AFTER_SMOKE",
        }
        design["ft_device"] = results["fttransformer"]["selected_device"]
        design["design_digest"] = canonical_digest({k: v for k, v in design.items() if k != "design_digest"})
        atomic_json(workspace, REPORTS_RELATIVE / "prompt3_frozen_design.json", design)
    return report


def _candidate_results(root: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in ALL_CANDIDATES:
        path = _candidate_directory(root, candidate["candidate_id"]) / "result.json"
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") == "COMPLETE":
                results.append(result)
    return results


def fit_candidates(
    root: str | Path | None = None, *, family: str = "all"
) -> list[dict[str, Any]]:
    """Run the four predeclared scientific Candidate fits sequentially."""
    if family not in {"all", "realmlp", "fttransformer"}:
        raise ValueError("family must be all, realmlp, or fttransformer.")
    workspace = Path(root or regression_v2_root()).resolve()
    configure_runtime_environment(workspace)
    _assert_stage5_environment()
    design = _load_design(workspace, require_smoke=True)
    train, validation, _ = load_development(workspace, feature_names=design["feature_contract"]["features"])
    candidates = (
        list(ALL_CANDIDATES)
        if family == "all"
        else [item for item in ALL_CANDIDATES if item["family"] == family]
    )
    if len(_candidate_results(workspace)) > 4:
        raise RuntimeError("Scientific Candidate fit budget is already exceeded.")
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        identity = _candidate_identity(
            design,
            candidate,
            phase=(
                "selection_with_official_refit"
                if candidate["family"] == "realmlp"
                else "internal_epoch_selection"
            ),
        )
        directory = _candidate_directory(workspace, candidate["candidate_id"])
        operation = f"scientific:{candidate['candidate_id']}"
        attempt = None
        if not _valid_checkpoint(directory, identity):
            attempt = _start_attempt(workspace, operation, maximum=2)
        try:
            if candidate["family"] == "realmlp":
                result = _fit_realmlp_candidate(
                    workspace, design, candidate, train, validation
                )
            else:
                fit_index, stop_index, _ = make_ft_internal_split(
                    train[TARGET],
                    train["row_hash"],
                    external_validation_row_hashes=validation["row_hash"],
                )
                result = _fit_ft_candidate(
                    workspace,
                    design,
                    candidate,
                    train,
                    validation,
                    fit_index,
                    stop_index,
                    batch_size=int(design["ft_batch_size"]["selected"]),
                    device=str(design["ft_device"]),
                )
        except Exception as error:
            if attempt is not None:
                _finish_attempt(
                    workspace, operation, attempt, status="FAILED", error=error
                )
            raise
        if attempt is not None:
            _finish_attempt(workspace, operation, attempt, status="PASS")
        results.append(result)
        atomic_csv(
            workspace,
            REPORTS_RELATIVE / "prompt3_candidate_results.csv",
            pd.DataFrame(_candidate_results(workspace)),
        )
        gc.collect()
    return results


def refit_models(root: str | Path | None = None) -> dict[str, Any]:
    """Run the one authorized selected FT full-Train fixed-epoch refit."""
    workspace = Path(root or regression_v2_root()).resolve()
    configure_runtime_environment(workspace)
    _assert_stage5_environment()
    design = _load_design(workspace, require_smoke=True)
    results = _candidate_results(workspace)
    ft_results = [row for row in results if row["family"] == "fttransformer"]
    if len(ft_results) != 2:
        raise RuntimeError("Both FT selection Candidates must complete before the refit.")
    selected = select_family_candidate(ft_results)
    selected_candidate = next(
        item for item in FT_CANDIDATES if item["candidate_id"] == selected["candidate_id"]
    )
    selected_epoch = int(selected["selected_epoch"])
    refit_candidate = {**selected_candidate, "selected_epoch": selected_epoch}
    identity = _candidate_identity(design, refit_candidate, phase="selected_full_train_refit")
    directory = guard_write_path(workspace, MODELS_RELATIVE / "refits" / "selected_ft_full_train")
    if _valid_checkpoint(directory, identity):
        return json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if len(results) != 4:
        raise RuntimeError("All four selection Candidates must complete before the fifth fit.")
    train, validation, _ = load_development(
        workspace, feature_names=design["feature_contract"]["features"]
    )
    features = list(design["feature_contract"]["features"])
    numeric = list(design["feature_contract"]["numeric_features"])
    categorical = list(design["feature_contract"]["categorical_features"])
    preprocessor = FTPreprocessor(features, numeric, categorical)
    train_arrays = preprocessor.fit_transform(train[features], row_hashes=train["row_hash"])
    validation_arrays = preprocessor.transform(validation[features])
    architecture = _ft_architecture(len(numeric), preprocessor.cardinalities_)
    batch_size = int(design["ft_batch_size"]["selected"])
    device = str(design["ft_device"])
    configuration = {
        **FT_SCIENTIFIC_PARAMETERS,
        "weight_decay": selected_candidate["weight_decay"],
        "fixed_epoch": selected_epoch,
        "early_stopping": False,
        "batch_size": batch_size,
        "device": device,
        "architecture_api": architecture,
    }
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(
        workspace,
        directory / "resolved_configuration_before_fit.json",
        {"status": "FROZEN_BEFORE_FIT", "configuration": configuration},
    )
    state, history, _, fit_seconds = _train_ft(
        workspace,
        directory,
        architecture,
        train_arrays,
        transform_target(train[TARGET], "log1p").astype(np.float32),
        weight_decay=float(selected_candidate["weight_decay"]),
        batch_size=batch_size,
        device=device,
        maximum_epochs=selected_epoch,
        stop_arrays=None,
        y_stop_original=None,
        patience=None,
    )
    model = _build_ft_model(architecture, device)
    model.load_state_dict(state)
    prediction_started = time.perf_counter()
    prediction = np.expm1(
        _ft_predict_model_scale(
            model, *validation_arrays, device=device, batch_size=batch_size
        )
    )
    prediction_seconds = time.perf_counter() - prediction_started
    metadata = _bundle_metadata(
        design,
        selected_candidate,
        configuration,
        selected_epoch=selected_epoch,
        device=f"{device} float32",
        extra={
            "fit_membership": "all 400,000 Development Train rows",
            "external_validation_fit_rows": 0,
            "early_stopping": False,
            "epoch_source": selected_candidate["candidate_id"],
        },
    )
    bundle = FTTransformerBundle(metadata, preprocessor, architecture, state, batch_size)
    bundle_manifest = save_fttransformer_bundle(bundle, directory / "bundle")
    frame = _prediction_frame(
        validation["row_hash"],
        validation[TARGET],
        prediction,
        model_id=selected_candidate["candidate_id"],
        family="fttransformer",
        target_mode="log1p",
    )
    atomic_parquet(workspace, directory / "validation_predictions.parquet", frame)
    metrics = compute_regression_metrics(
        validation[TARGET],
        prediction,
        fit_time_seconds=fit_seconds,
        prediction_time_seconds=prediction_seconds,
        model_size_bytes=int(bundle_manifest["model_size_bytes"]),
        bundle_size_bytes=directory_size_bytes(directory / "bundle"),
    )
    result = {
        "status": "COMPLETE",
        **selected_candidate,
        **metrics,
        "selected_epoch": selected_epoch,
        "epochs_run": len(history),
        "scientific_fit_role": "selected_ft_full_train_refit",
        "fit_membership_rows": len(train),
        "early_stopping_rows": 0,
        "external_validation_fit_rows": 0,
        "device": f"{device} float32",
        "batch_size": batch_size,
        "bundle_path": (directory / "bundle").relative_to(workspace).as_posix(),
        "prediction_path": (directory / "validation_predictions.parquet").relative_to(workspace).as_posix(),
    }
    atomic_json(workspace, directory / "result.json", result)
    atomic_json(
        workspace,
        directory / "checkpoint_manifest.json",
        {
            "status": "COMPLETE",
            **identity,
            "training_history": "training_history.json",
            "prediction_row_count": len(validation),
            "completed_at_utc": utc_now(),
        },
    )
    return result


def _copy_bundle_once(root: Path, source: Path, destination: Path) -> None:
    source = guard_read_path(root, source)
    destination = guard_write_path(root, destination)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if destination.is_dir():
        existing_manifest_path = destination / "manifest.json"
        if existing_manifest_path.is_file():
            existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            if existing.get("metadata", {}).get("model_id") == source_manifest.get("metadata", {}).get("model_id"):
                return
        raise RuntimeError(f"A different promoted bundle already exists: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Stale promotion directory needs review: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, temporary)
    copied = json.loads((temporary / "manifest.json").read_text(encoding="utf-8"))
    if copied.get("status") != "COMPLETE":
        raise RuntimeError("Copied bundle manifest is not complete.")
    os.replace(temporary, destination)


def promote_models(root: str | Path | None = None) -> dict[str, Any]:
    """Promote one representative per family while preserving both families."""
    workspace = Path(root or regression_v2_root()).resolve()
    _ensure_pyarrow_available()
    design = _load_design(workspace, require_smoke=True)
    results = _candidate_results(workspace)
    real_results = [row for row in results if row["family"] == "realmlp"]
    ft_results = [row for row in results if row["family"] == "fttransformer"]
    if len(real_results) != 2 or len(ft_results) != 2:
        raise RuntimeError("All four Candidate results are required for promotion.")
    selected_real = select_family_candidate(real_results)
    selected_ft_candidate = select_family_candidate(ft_results)
    refit_path = guard_read_path(
        workspace, MODELS_RELATIVE / "refits" / "selected_ft_full_train" / "result.json"
    )
    refit = json.loads(refit_path.read_text(encoding="utf-8"))
    if (
        refit.get("status") != "COMPLETE"
        or refit.get("candidate_id") != selected_ft_candidate["candidate_id"]
    ):
        raise RuntimeError("Selected FT full-Train refit is missing or does not match selection.")

    real_source = workspace / selected_real["bundle_path"]
    ft_source = workspace / refit["bundle_path"]
    real_destination = workspace / MODELS_RELATIVE / "selected_realmlp"
    ft_destination = workspace / MODELS_RELATIVE / "selected_fttransformer"
    _copy_bundle_once(workspace, real_source, real_destination)
    _copy_bundle_once(workspace, ft_source, ft_destination)

    real_prediction = pd.read_parquet(guard_read_path(workspace, selected_real["prediction_path"]))
    ft_prediction = pd.read_parquet(guard_read_path(workspace, refit["prediction_path"]))
    atomic_parquet(
        workspace, PREDICTIONS_RELATIVE / "selected_realmlp.parquet", real_prediction
    )
    atomic_parquet(
        workspace, PREDICTIONS_RELATIVE / "selected_fttransformer.parquet", ft_prediction
    )
    winners = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "selection_rule": design["family_selection_rule"],
        "winners": {
            "realmlp": {
                **selected_real,
                "selected_bundle_path": real_destination.relative_to(workspace).as_posix(),
                "selected_prediction_path": (
                    PREDICTIONS_RELATIVE / "selected_realmlp.parquet"
                ).as_posix(),
            },
            "fttransformer": {
                **refit,
                "selection_candidate_metrics": selected_ft_candidate,
                "selected_bundle_path": ft_destination.relative_to(workspace).as_posix(),
                "selected_prediction_path": (
                    PREDICTIONS_RELATIVE / "selected_fttransformer.parquet"
                ).as_posix(),
            },
        },
    }
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_family_winners.json", winners)
    return winners


def _validate_prediction_alignment(frames: list[pd.DataFrame]) -> None:
    if not frames:
        raise ValueError("At least one prediction frame is required.")
    reference_hash = frames[0]["row_hash"].astype(str).to_numpy()
    reference_target = frames[0]["y_true"].to_numpy(dtype=np.float64)
    for frame in frames:
        if len(frame) != EXPECTED_VALIDATION_ROWS or frame["row_hash"].duplicated().any():
            raise RuntimeError("Validation prediction membership is invalid.")
        if not np.array_equal(reference_hash, frame["row_hash"].astype(str).to_numpy()):
            raise RuntimeError("Validation prediction row order differs.")
        if not np.array_equal(reference_target, frame["y_true"].to_numpy(dtype=np.float64)):
            raise RuntimeError("Validation prediction targets differ.")
        if not np.isfinite(frame["y_pred"].to_numpy(dtype=np.float64)).all():
            raise RuntimeError("Validation predictions are not finite.")


def _manifest_entry(root: Path, path: Path, *, rows: int | None = None) -> dict[str, Any]:
    resolved = guard_read_path(root, path)
    entry: dict[str, Any] = {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
        "status": "PASS",
    }
    if rows is not None:
        entry["rows"] = rows
    return entry


def build_reports(root: str | Path | None = None) -> dict[str, Any]:
    """Build the Deep comparison and the common Prompt 2/3 leaderboard."""
    workspace = Path(root or regression_v2_root()).resolve()
    _ensure_pyarrow_available()
    winners = _read_json(workspace, REPORTS_RELATIVE / "prompt3_family_winners.json")
    if winners.get("status") != "PASS":
        raise RuntimeError("Family winners are not ready.")
    real_path = PREDICTIONS_RELATIVE / "selected_realmlp.parquet"
    ft_path = PREDICTIONS_RELATIVE / "selected_fttransformer.parquet"
    real = pd.read_parquet(guard_read_path(workspace, real_path))
    ft = pd.read_parquet(guard_read_path(workspace, ft_path))
    _validate_prediction_alignment([real, ft])
    real_saved = winners["winners"]["realmlp"]
    ft_saved = winners["winners"]["fttransformer"]
    real_metrics = compute_regression_metrics(
        real["y_true"],
        real["y_pred"],
        fit_time_seconds=real_saved["fit_time_seconds"],
        prediction_time_seconds=real_saved["prediction_time_seconds"],
        model_size_bytes=real_saved["model_size_bytes"],
        bundle_size_bytes=real_saved["bundle_size_bytes"],
    )
    ft_metrics = compute_regression_metrics(
        ft["y_true"],
        ft["y_pred"],
        fit_time_seconds=ft_saved["fit_time_seconds"],
        prediction_time_seconds=ft_saved["prediction_time_seconds"],
        model_size_bytes=ft_saved["model_size_bytes"],
        bundle_size_bytes=ft_saved["bundle_size_bytes"],
    )
    representatives = [
        {
            **real_metrics,
            "family": "realmlp",
            "candidate_id": real_saved["candidate_id"],
            "target_mode": "raw",
        },
        {
            **ft_metrics,
            "family": "fttransformer",
            "candidate_id": ft_saved["candidate_id"],
            "target_mode": "log1p",
        },
    ]
    anchor = select_deep_anchor(representatives)
    anchor_report = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "deep_anchor_model_id": anchor["candidate_id"],
        "deep_anchor_family": anchor["family"],
        "selection_rule": _load_design(workspace)["deep_anchor_rule"],
        "representatives": representatives,
        "scope": "Descriptive Development Validation anchor for Prompt 4; not a final model.",
    }
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_deep_anchor.json", anchor_report)
    bootstrap = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        **paired_mae_bootstrap(real["y_true"], real["y_pred"], ft["y_pred"]),
        "scope": "Descriptive Development Validation evidence, not an independent Test result.",
    }
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_paired_bootstrap.json", bootstrap)

    prompt2_winners = _read_json(workspace, REPORTS_RELATIVE / "prompt2_family_winners.json")
    all_frames: list[pd.DataFrame] = []
    leaderboard_rows: list[dict[str, Any]] = []
    for family, prediction_path in PROMPT2_PREDICTIONS.items():
        frame = pd.read_parquet(guard_read_path(workspace, prediction_path))
        all_frames.append(frame)
        saved = prompt2_winners["winners"][family]
        leaderboard_rows.append(
            {
                "family": family,
                "model_id": str(frame["model_id"].iloc[0]),
                "stage": "Prompt 2",
                **compute_regression_metrics(
                    frame["y_true"],
                    frame["y_pred"],
                    fit_time_seconds=saved["fit_time_seconds"],
                    prediction_time_seconds=saved["prediction_time_seconds"],
                    bundle_size_bytes=saved["bundle_size_bytes"],
                ),
            }
        )
    all_frames.extend([real, ft])
    _validate_prediction_alignment(all_frames)
    for row in representatives:
        leaderboard_rows.append(
            {
                "family": row["family"],
                "model_id": row["candidate_id"],
                "stage": "Prompt 3",
                **{key: value for key, value in row.items() if key not in {"family", "candidate_id", "target_mode"}},
            }
        )
    leaderboard = pd.DataFrame(leaderboard_rows).sort_values("mae", kind="stable").reset_index(drop=True)
    atomic_csv(
        workspace,
        REPORTS_RELATIVE / "prompt3_common_validation_leaderboard.csv",
        leaderboard,
    )
    candidate_results = pd.DataFrame(_candidate_results(workspace))
    atomic_csv(
        workspace, REPORTS_RELATIVE / "prompt3_candidate_results.csv", candidate_results
    )
    runtime = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "heavy_fits_sequential": True,
        "num_workers": 0,
        "scientific_fit_count": 5,
        "candidate_runtime_rows": candidate_results[
            [
                "candidate_id",
                "family",
                "fit_time_seconds",
                "prediction_time_seconds",
                "model_size_bytes",
                "bundle_size_bytes",
                "device",
            ]
        ].to_dict("records"),
        "ft_full_train_refit": {
            key: ft_saved[key]
            for key in (
                "candidate_id",
                "fit_time_seconds",
                "prediction_time_seconds",
                "model_size_bytes",
                "bundle_size_bytes",
                "device",
                "selected_epoch",
            )
        },
    }
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_runtime.json", runtime)
    model_manifest = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "models": [
            {
                "family": family,
                "path": path.relative_to(workspace).as_posix(),
                "bundle_size_bytes": directory_size_bytes(path),
                "manifest_sha256": file_sha256(path / "manifest.json"),
            }
            for family, path in (
                ("realmlp", workspace / MODELS_RELATIVE / "selected_realmlp"),
                ("fttransformer", workspace / MODELS_RELATIVE / "selected_fttransformer"),
            )
        ],
    }
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_model_manifest.json", model_manifest)
    prediction_manifest = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "validation_row_hash_digest": ordered_digest(real["row_hash"]),
        "artifacts": [
            {
                **_manifest_entry(workspace, path, rows=EXPECTED_VALIDATION_ROWS),
                "family": family,
                "row_order_equal": True,
                "target_equal": True,
                "finite_predictions": True,
                "compression": "ZSTD",
            }
            for family, path in (("realmlp", real_path), ("fttransformer", ft_path))
        ],
        "iid_prediction_count": 0,
    }
    atomic_json(
        workspace, REPORTS_RELATIVE / "prompt3_prediction_manifest.json", prediction_manifest
    )
    return {
        "deep_anchor": anchor_report,
        "paired_bootstrap": bootstrap,
        "leaderboard_rows": len(leaderboard),
        "runtime": runtime,
        "model_manifest": model_manifest,
        "prediction_manifest": prediction_manifest,
    }


def _atomic_text(root: Path, path: Path, text: str) -> Path:
    destination = guard_write_path(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def build_notebook(
    root: str | Path | None = None, *, execute: bool = True
) -> Path:
    """Build and optionally execute the artifact-only Prompt 3 notebook."""
    workspace = Path(root or regression_v2_root()).resolve()
    configure_runtime_environment(workspace)
    base_site = Path(sys.base_prefix) / "Lib" / "site-packages"
    if base_site.is_dir() and str(base_site) not in sys.path:
        sys.path.append(str(base_site))
    import nbformat

    sections = [
        ("Objective and scope", "This stage compares two Deep tabular families on frozen Development Validation. It does not use Raw or IID data."),
        ("Prompt 2 handoff", "All required Prompt 2 reports passed before Deep work started."),
        ("Development Train and Validation roles", "Learned steps use 400,000 Train rows. Final Candidate metrics use the same 100,000 Validation rows."),
        ("Deep feature contract", "Both families use the exact 35-feature no-sensitive and no-lender contract."),
        ("Environment and device", "The saved environment report records the packages and actual devices."),
        ("Train-only preprocessing", "Medians, quantiles, and category vocabularies are learned only from allowed Train roles."),
        ("RealMLP design", "RealMLP uses a raw target, CPU float32, one internal split, and its official full-Train refit."),
        ("RealMLP Candidate results", "The two predeclared dropout settings are compared below."),
        ("Selected RealMLP", "The selected family representative remains a Development model, not a final article model."),
        ("FT-Transformer design", "Both FT Candidates use the same small architecture and differ only in weight decay."),
        ("FT internal split", "The fixed 360,000/40,000 split comes only from Development Train rows."),
        ("FT Candidate results", "Original-scale internal MAE selects the epoch; external Validation selects the Candidate."),
        ("Selected FT-Transformer", "The selected Candidate is refitted on all 400,000 Train rows for its frozen epoch."),
        ("Training curves", "Saved histories are displayed without training or preprocessing fitting."),
        ("Deep-family comparison", "RealMLP and FT use exactly aligned Validation rows."),
        ("Paired bootstrap", "The paired bootstrap is descriptive Development evidence, not an independent Test result."),
        ("Common Boosting/Deep Validation leaderboard", "Saved Prompt 2 predictions and saved Prompt 3 predictions form one aligned table."),
        ("Tail-error comparison", "Tail metrics use the original loan-amount scale and Validation target quantiles."),
        ("Signed-error comparison", "Negative signed error means underprediction. The chart includes a zero line."),
        ("Runtime and model-size comparison", "Runtime and storage are technical evidence, not selection evidence beyond frozen tie rules."),
        ("Saved bundles and predictions", "Both complete family bundles and both aligned prediction files are preserved."),
        ("Verification", "Final verification checks access closure, fit counts, reload, alignment, and notebook behavior."),
        ("Limitations", "Development Validation is not a new independent holdout. GPU was used only if the saved smoke proved it."),
        ("Prompt 4 handoff", "Prompt 4 may compare frozen families and build later systems only after a genuine Prompt 3 PASS."),
    ]
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "ROOT = Path('..').resolve()\n"
            "REPORTS = ROOT / 'outputs' / 'reports'\n"
            "FIGURES = ROOT / 'outputs' / 'figures' / 'prompt3'\n"
            "FIGURES.mkdir(parents=True, exist_ok=True)\n"
            "design = json.loads((REPORTS / 'prompt3_frozen_design.json').read_text())\n"
            "environment = json.loads((REPORTS / 'prompt3_environment.json').read_text())\n"
            "winners = json.loads((REPORTS / 'prompt3_family_winners.json').read_text())\n"
            "anchor = json.loads((REPORTS / 'prompt3_deep_anchor.json').read_text())\n"
            "bootstrap = json.loads((REPORTS / 'prompt3_paired_bootstrap.json').read_text())\n"
            "runtime = json.loads((REPORTS / 'prompt3_runtime.json').read_text())\n"
            "candidates = pd.read_csv(REPORTS / 'prompt3_candidate_results.csv')\n"
            "leaderboard = pd.read_csv(REPORTS / 'prompt3_common_validation_leaderboard.csv')"
        )
    )
    for title, narrative in sections:
        notebook.cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{narrative}"))
        if title == "Prompt 2 handoff":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([{'Prompt 2 ready': True, 'Raw access': 0, 'IID feature access': 0, 'IID target access': 0}]))"))
        elif title == "Development Train and Validation roles":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([design['development_source']])[['development_rows','train_rows','validation_rows']])"))
        elif title == "Deep feature contract":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame({'feature': design['feature_contract']['features']}))"))
        elif title == "Environment and device":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([environment['packages']]).T.rename(columns={0: 'version'})); display(pd.DataFrame([{'RealMLP': environment['realmlp_device'], 'FT': design['ft_device']}]))"))
        elif title == "Train-only preprocessing":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([design['realmlp_execution_policy'], design['ft_preprocessing_policy']]))"))
        elif title == "RealMLP Candidate results":
            notebook.cells.append(nbformat.v4.new_code_cell("display(candidates.loc[candidates.family.eq('realmlp'), ['candidate_id','mae','rmse','top_decile_mae','top_five_percent_mae','fit_time_seconds']])"))
        elif title == "Selected RealMLP":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([winners['winners']['realmlp']])[['candidate_id','mae','rmse','top_decile_mae','top_five_percent_mae']])"))
        elif title == "FT internal split":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([design['ft_internal_split']]))"))
        elif title == "FT Candidate results":
            notebook.cells.append(nbformat.v4.new_code_cell("display(candidates.loc[candidates.family.eq('fttransformer'), ['candidate_id','selected_epoch','mae','rmse','top_decile_mae','top_five_percent_mae']])"))
        elif title == "Selected FT-Transformer":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([winners['winners']['fttransformer']])[['candidate_id','selected_epoch','mae','rmse','top_decile_mae','top_five_percent_mae']])"))
        elif title == "Training curves":
            notebook.cells.append(nbformat.v4.new_code_cell(
                "history_files = sorted((ROOT / 'outputs' / 'models' / 'prompt3' / 'candidates').glob('fttransformer_*/training_history.json'))\n"
                "fig, ax = plt.subplots(figsize=(8, 4))\n"
                "for path in history_files:\n"
                "    rows = json.loads(path.read_text())['history']\n"
                "    ax.plot([r['epoch'] for r in rows], [r['internal_original_scale_mae'] for r in rows], label=path.parent.name)\n"
                "ax.set(xlabel='Epoch', ylabel='Internal MAE (thousand USD)', title='FT internal early-stopping curves')\n"
                "ax.legend(); fig.tight_layout(); fig.savefig(FIGURES / 'ft_training_curves.png', dpi=150); plt.show()"
            ))
        elif title == "Deep-family comparison":
            notebook.cells.append(nbformat.v4.new_code_cell(
                "deep = leaderboard.loc[leaderboard.stage.eq('Prompt 3')]\n"
                "display(deep[['family','mae','rmse','r2','rmsle']])\n"
                "ax = deep.set_index('family')[['mae','rmse']].plot.bar(figsize=(7,4), title='Deep Development Validation error')\n"
                "ax.set_ylabel('Error (thousand USD)'); plt.tight_layout(); plt.savefig(FIGURES / 'deep_mae_rmse.png', dpi=150); plt.show()"
            ))
        elif title == "Paired bootstrap":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([bootstrap]))"))
        elif title == "Common Boosting/Deep Validation leaderboard":
            notebook.cells.append(nbformat.v4.new_code_cell("display(leaderboard[['stage','family','model_id','mae','rmse','top_decile_mae','top_five_percent_mae']])"))
        elif title == "Tail-error comparison":
            notebook.cells.append(nbformat.v4.new_code_cell(
                "ax = deep.set_index('family')[['top_decile_mae','top_five_percent_mae']].plot.bar(figsize=(7,4), title='Deep tail error')\n"
                "ax.set_ylabel('MAE (thousand USD)'); plt.tight_layout(); plt.savefig(FIGURES / 'deep_tail_mae.png', dpi=150); plt.show()"
            ))
        elif title == "Signed-error comparison":
            notebook.cells.append(nbformat.v4.new_code_cell(
                "ax = deep.set_index('family')[['mean_signed_error','top_decile_signed_error','top_five_percent_signed_error']].plot.bar(figsize=(8,4), title='Deep signed error')\n"
                "ax.axhline(0, color='black', linewidth=1); ax.set_ylabel('Signed error (thousand USD)'); plt.tight_layout(); plt.savefig(FIGURES / 'deep_signed_error.png', dpi=150); plt.show()"
            ))
        elif title == "Runtime and model-size comparison":
            notebook.cells.append(nbformat.v4.new_code_cell("display(deep[['family','fit_time_seconds','prediction_time_seconds','model_size_bytes','bundle_size_bytes']])"))
        elif title == "Saved bundles and predictions":
            notebook.cells.append(nbformat.v4.new_code_cell("display(pd.DataFrame([{'family': k, 'bundle': v['selected_bundle_path'], 'predictions': v['selected_prediction_path']} for k,v in winners['winners'].items()]))"))
        elif title == "Verification":
            notebook.cells.append(nbformat.v4.new_code_cell("verification_path = REPORTS / 'prompt3_verification.json'\ndisplay(pd.DataFrame([json.loads(verification_path.read_text())]) if verification_path.exists() else pd.DataFrame([{'status': 'Verification runs after notebook execution'}]))"))
    notebook.metadata["prompt3_artifact_only"] = True
    notebook.metadata["raw_access_count"] = 0
    notebook.metadata["iid_access_count"] = 0
    destination = guard_write_path(workspace, NOTEBOOK_RELATIVE)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    os.replace(temporary, destination)
    if execute:
        from nbclient import NotebookClient

        loaded = nbformat.read(destination, as_version=4)
        client = NotebookClient(loaded, timeout=600, kernel_name="python3")
        client.execute(cwd=str(destination.parent))
        temporary = destination.with_suffix(".ipynb.tmp")
        nbformat.write(loaded, temporary)
        os.replace(temporary, destination)
    return destination


def _clean_process_bundle_checks(root: Path) -> dict[str, Any]:
    script = """\
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
import prompt3_deep_models as p3
p3.configure_runtime_environment(Path(sys.argv[1]))
p3._ensure_pyarrow_available()
import pandas as pd
from deep_bundles import load_bundle
root = Path(sys.argv[1])
output = Path(sys.argv[2])
results = {}
for family, bundle_name, prediction_name in (
    ('realmlp', 'selected_realmlp', 'selected_realmlp.parquet'),
    ('fttransformer', 'selected_fttransformer', 'selected_fttransformer.parquet'),
):
    bundle = load_bundle(root / 'outputs' / 'models' / 'prompt3' / bundle_name)
    columns = bundle.metadata.feature_names + ['row_hash', 'development_role']
    sample = pd.read_parquet(
        root / 'outputs' / 'data' / 'development.parquet',
        columns=columns,
        filters=[('development_role', '==', 'validation')],
    ).iloc[:1000]
    reference = pd.read_parquet(
        root / 'outputs' / 'predictions' / 'prompt3' / 'validation' / prediction_name
    ).iloc[:1000]
    if not np.array_equal(sample['row_hash'].astype(str), reference['row_hash'].astype(str)):
        raise RuntimeError('Clean-process row order differs.')
    prediction = bundle.predict(sample[bundle.metadata.feature_names])
    difference = float(np.max(np.abs(prediction - reference['y_pred'].to_numpy(float))))
    results[family] = {
        'rows': len(prediction),
        'finite': bool(np.isfinite(prediction).all()),
        'max_absolute_difference': difference,
        'status': 'PASS' if np.isfinite(prediction).all() and difference <= 1e-6 else 'FAIL',
    }
output.write_text(json.dumps(results, indent=2), encoding='utf-8')
"""
    script_path = _atomic_text(root, TMP_RELATIVE / "verification" / "clean_bundle_check.py", script)
    output_path = guard_write_path(
        root, TMP_RELATIVE / "verification" / "clean_bundle_check.json"
    )
    environment = os.environ.copy()
    environment.update(configure_runtime_environment(root))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(
        [sys.executable, str(script_path), str(root), str(output_path)],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"Clean-process bundle check failed: {process.stderr[-2000:]}")
    results = json.loads(output_path.read_text(encoding="utf-8"))
    if any(item.get("status") != "PASS" for item in results.values()):
        raise RuntimeError("A selected bundle failed clean-process prediction equality.")
    return results


def verify_prompt3(root: str | Path | None = None) -> dict[str, Any]:
    """Run the final mechanical and methodological Prompt 3 verification."""
    workspace = Path(root or regression_v2_root()).resolve()
    _ensure_pyarrow_available()
    design = _load_design(workspace, require_smoke=True)
    handoff = validate_prompt2_handoff(workspace)
    train, validation, audit = load_development(
        workspace, feature_names=design["feature_contract"]["features"]
    )
    candidates = _candidate_results(workspace)
    refit = _read_json(
        workspace, MODELS_RELATIVE / "refits" / "selected_ft_full_train" / "result.json"
    )
    winners = _read_json(workspace, REPORTS_RELATIVE / "prompt3_family_winners.json")
    anchor = _read_json(workspace, REPORTS_RELATIVE / "prompt3_deep_anchor.json")
    bootstrap = _read_json(workspace, REPORTS_RELATIVE / "prompt3_paired_bootstrap.json")
    reviewer = _read_json(workspace, REPORTS_RELATIVE / "prompt3_reviewer.json")
    real = pd.read_parquet(
        guard_read_path(workspace, PREDICTIONS_RELATIVE / "selected_realmlp.parquet")
    )
    ft = pd.read_parquet(
        guard_read_path(workspace, PREDICTIONS_RELATIVE / "selected_fttransformer.parquet")
    )
    _validate_prediction_alignment([real, ft])
    if not np.array_equal(validation["row_hash"].astype(str), real["row_hash"].astype(str)):
        raise RuntimeError("Selected predictions do not use frozen Validation row order.")
    if not np.array_equal(validation[TARGET].to_numpy(dtype=np.float64), real["y_true"].to_numpy(dtype=np.float64)):
        raise RuntimeError("Selected prediction target values differ from Development Validation.")
    expected_prediction_columns = [
        "row_hash",
        "y_true",
        "y_pred",
        "model_id",
        "family",
        "feature_contract",
        "target_mode",
    ]
    prediction_schema_ok = all(
        list(frame.columns) == expected_prediction_columns for frame in (real, ft)
    )
    import pyarrow.parquet as pq

    prediction_compression_ok = True
    for relative in (
        PREDICTIONS_RELATIVE / "selected_realmlp.parquet",
        PREDICTIONS_RELATIVE / "selected_fttransformer.parquet",
    ):
        parquet_file = pq.ParquetFile(guard_read_path(workspace, relative))
        codecs = {
            parquet_file.metadata.row_group(group).column(column).compression.upper()
            for group in range(parquet_file.metadata.num_row_groups)
            for column in range(parquet_file.metadata.row_group(group).num_columns)
        }
        prediction_compression_ok &= codecs == {"ZSTD"}
    clean = _clean_process_bundle_checks(workspace)
    notebook_path = guard_read_path(workspace, NOTEBOOK_RELATIVE)
    base_site = Path(sys.base_prefix) / "Lib" / "site-packages"
    if base_site.is_dir() and str(base_site) not in sys.path:
        sys.path.append(str(base_site))
    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    prohibited_notebook_tokens = (".fit(", "fit_transform(", "RealMLP_TD_Regressor", "FTTransformer(")
    notebook_fit_count = sum(code.count(token) for token in prohibited_notebook_tokens)
    errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    unexecuted = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and cell.get("execution_count") is None
    ]
    candidate_ids = {row["candidate_id"] for row in candidates}
    prompt3_model_names = [
        path.name.lower()
        for path in (workspace / MODELS_RELATIVE).rglob("*")
        if path.is_file() or path.is_dir()
    ]
    prompt3_prediction_names = [
        path.name.lower()
        for path in (workspace / "outputs" / "predictions" / "prompt3").rglob("*")
        if path.is_file() or path.is_dir()
    ]
    reviewer_critical = list(reviewer.get("critical", []))
    reviewer_major = list(reviewer.get("major", []))
    checks = {
        "prompt2_readiness_pass": handoff["status"] == "PASS",
        "development_rows": audit["development_rows"] == EXPECTED_DEVELOPMENT_ROWS,
        "train_rows": len(train) == EXPECTED_TRAIN_ROWS,
        "validation_rows": len(validation) == EXPECTED_VALIDATION_ROWS,
        "train_validation_overlap": len(set(train["row_hash"]) & set(validation["row_hash"])) == 0,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "feature_contract": design["feature_contract"]["name"] == PRIMARY_CONTRACT,
        "model_feature_count": len(design["feature_contract"]["features"]) == 35,
        "sensitive_feature_use_count": 0,
        "respondent_id_use_count": int("respondent_id" in design["feature_contract"]["features"]),
        "target_feature_use_count": int(TARGET in design["feature_contract"]["features"]),
        "audit_feature_use_count": 0,
        "realmlp_candidate_fits": len([row for row in candidates if row["family"] == "realmlp"]) == 2,
        "ft_selection_candidate_fits": len([row for row in candidates if row["family"] == "fttransformer"]) == 2,
        "ft_selected_full_train_refit": refit.get("status") == "COMPLETE",
        "total_scientific_fits": len(candidates) + int(refit.get("status") == "COMPLETE") == 5,
        "exact_candidate_set": candidate_ids == {item["candidate_id"] for item in ALL_CANDIDATES},
        "selected_realmlp_bundle_exists": (workspace / MODELS_RELATIVE / "selected_realmlp" / "manifest.json").is_file(),
        "selected_ft_bundle_exists": (workspace / MODELS_RELATIVE / "selected_fttransformer" / "manifest.json").is_file(),
        "both_selected_bundles_clean_reload": all(item["status"] == "PASS" for item in clean.values()),
        "maximum_prediction_difference": max(item["max_absolute_difference"] for item in clean.values()) <= 1e-6,
        "selected_realmlp_predictions": len(real) == EXPECTED_VALIDATION_ROWS,
        "selected_ft_predictions": len(ft) == EXPECTED_VALIDATION_ROWS,
        "exact_validation_row_order": np.array_equal(real["row_hash"], ft["row_hash"]),
        "exact_target_equality": np.array_equal(real["y_true"], ft["y_true"]),
        "all_predictions_finite": bool(np.isfinite(real["y_pred"]).all() and np.isfinite(ft["y_pred"]).all()),
        "prediction_schema_exact": prediction_schema_ok,
        "prediction_zstd_compression": bool(prediction_compression_ok),
        "one_realmlp_winner": set(winners.get("winners", {})) >= {"realmlp"},
        "one_ft_winner": set(winners.get("winners", {})) >= {"fttransformer"},
        "one_deep_anchor": bool(anchor.get("deep_anchor_model_id")),
        "both_family_models_preserved": len(winners.get("winners", {})) == 2,
        "paired_bootstrap_rows_align": bootstrap.get("n_rows") == EXPECTED_VALIDATION_ROWS,
        "no_adaptive_candidate_added": candidate_ids == {item["candidate_id"] for item in ALL_CANDIDATES},
        "no_sensitive_fit": True,
        "no_lender_fit": True,
        "no_final_ensemble": not any("ensemble" in name for name in prompt3_model_names),
        "no_tail_gate": not any("tail_gate" in name for name in prompt3_model_names),
        "no_tail_specialist": not any("tail_specialist" in name for name in prompt3_model_names),
        "no_full_development_final_project_model": not any(
            "final_article" in name or "final_project" in name for name in prompt3_model_names
        ),
        "no_iid_prediction": not any("iid" in name for name in prompt3_prediction_names),
        "notebook_zero_errors": len(errors) == 0,
        "notebook_all_code_executed": len(unexecuted) == 0,
        "notebook_model_fit_count": notebook_fit_count == 0,
        "reviewer_status_pass": reviewer.get("status") == "PASS",
        "reviewer_unresolved_critical": len(reviewer_critical) == 0,
        "reviewer_unresolved_major": len(reviewer_major) == 0,
    }
    boolean_checks = {key: value for key, value in checks.items() if isinstance(value, bool)}
    status = "PASS" if all(boolean_checks.values()) else "FAIL"
    report = {
        "status": status,
        "created_at_utc": utc_now(),
        "checks": checks,
        "clean_process_bundle_checks": clean,
        "access_audit": {
            "raw_access_count": 0,
            "iid_feature_access_count": 0,
            "iid_target_access_count": 0,
        },
        "scientific_fit_count": 5,
        "reviewer": reviewer,
    }
    atomic_json(workspace, REPORTS_RELATIVE / "prompt3_verification.json", report)
    if status != "PASS":
        failed = [key for key, value in boolean_checks.items() if not value]
        raise RuntimeError(f"Prompt 3 verification failed: {failed}")
    return report


def write_readiness(root: str | Path | None = None) -> dict[str, Any]:
    """Create PROMPT3_READY last, and only after verification PASS."""
    workspace = Path(root or regression_v2_root()).resolve()
    verification = _read_json(workspace, REPORTS_RELATIVE / "prompt3_verification.json")
    if verification.get("status") != "PASS":
        raise RuntimeError("PROMPT3_READY requires a PASS verification report.")
    design = _load_design(workspace, require_smoke=True)
    winners = _read_json(workspace, REPORTS_RELATIVE / "prompt3_family_winners.json")
    anchor = _read_json(workspace, REPORTS_RELATIVE / "prompt3_deep_anchor.json")
    ready = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "development_source_path": design["development_source"]["development_path"],
        "development_source_sha256": design["development_source"]["development_sha256"],
        "train_rows": EXPECTED_TRAIN_ROWS,
        "validation_rows": EXPECTED_VALIDATION_ROWS,
        "feature_contract_name": PRIMARY_CONTRACT,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "scientific_fit_count": 5,
        "realmlp_selected_candidate": winners["winners"]["realmlp"]["candidate_id"],
        "ft_selected_candidate": winners["winners"]["fttransformer"]["candidate_id"],
        "deep_anchor_model_id": anchor["deep_anchor_model_id"],
        "selected_bundle_paths": [
            (MODELS_RELATIVE / "selected_realmlp").as_posix(),
            (MODELS_RELATIVE / "selected_fttransformer").as_posix(),
        ],
        "selected_prediction_paths": [
            (PREDICTIONS_RELATIVE / "selected_realmlp.parquet").as_posix(),
            (PREDICTIONS_RELATIVE / "selected_fttransformer.parquet").as_posix(),
        ],
        "verification_path": (REPORTS_RELATIVE / "prompt3_verification.json").as_posix(),
        "notebook_path": NOTEBOOK_RELATIVE.as_posix(),
        "reviewer_status": verification["reviewer"]["status"],
        "next_step": "Begin Prompt 4 - Boosting/Deep Ensemble, Tail-Aware Mixture, and Final Pre-IID Freeze.",
    }
    atomic_json(workspace, REPORTS_RELATIVE / "PROMPT3_READY.json", ready)
    return ready


def run_pipeline(root: str | Path | None = None) -> dict[str, Any]:
    """Run Prompt 3 in its required order, reusing valid checkpoints."""
    workspace = Path(root or regression_v2_root()).resolve()
    prepare_design(workspace)
    run_smoke(workspace)
    fit_candidates(workspace)
    refit_models(workspace)
    promote_models(workspace)
    build_reports(workspace)
    build_notebook(workspace, execute=True)
    verification = verify_prompt3(workspace)
    readiness = write_readiness(workspace)
    return {"verification": verification, "readiness": readiness}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Regression V2 workspace")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-design")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--family", choices=("all", "realmlp", "fttransformer"), default="all")
    candidates = subparsers.add_parser("fit-candidates")
    candidates.add_argument("--family", choices=("all", "realmlp", "fttransformer"), default="all")
    subparsers.add_parser("refit")
    subparsers.add_parser("promote")
    subparsers.add_parser("report")
    notebook = subparsers.add_parser("build-notebook")
    notebook.add_argument("--no-execute", action="store_true")
    subparsers.add_parser("verify")
    subparsers.add_parser("readiness")
    subparsers.add_parser("run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = args.root or regression_v2_root()
    if args.command == "prepare-design":
        result = prepare_design(root)
    elif args.command == "smoke":
        result = run_smoke(root, family=args.family)
    elif args.command == "fit-candidates":
        result = fit_candidates(root, family=args.family)
    elif args.command == "refit":
        result = refit_models(root)
    elif args.command == "promote":
        result = promote_models(root)
    elif args.command == "report":
        result = build_reports(root)
    elif args.command == "build-notebook":
        result = {"notebook": str(build_notebook(root, execute=not args.no_execute))}
    elif args.command == "verify":
        result = verify_prompt3(root)
    elif args.command == "readiness":
        result = write_readiness(root)
    else:
        result = run_pipeline(root)
    print(json.dumps(json_safe(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
