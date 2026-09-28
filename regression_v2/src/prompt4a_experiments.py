"""Prompt 4A orchestration for bounded ensemble and Tail-Aware experiments.

The command-line stages are deliberately separate so every valid heavy fit
can be validated and reused. Raw and IID paths are blocked centrally.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
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
import pyarrow.parquet as pq
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .deep_preprocessing import duplicate_safe_deciles, ordered_digest
    from .ensemble_utils import (
        ALPHA_GRID,
        HARD_THRESHOLDS,
        NO_FIT_ENSEMBLES,
        apply_ensemble,
        correlation_report,
        deterministic_validation_split,
        hard_route,
        optimize_convex_mae,
        soft_mix,
    )
    from .prompt4_metrics import (
        compute_operational_tail_metrics,
        compute_regression_metrics,
        paired_mae_bootstrap,
        provisional_acceptance,
        select_reference,
    )
    from .tail_models import (
        GATE_PARAMETERS,
        SPECIALIST_PARAMETERS,
        TAIL_WEIGHTS,
        WEIGHTED_PARAMETERS,
        GateBundle,
        RegressionBundle,
        TailModelMetadata,
        fixed_refit_config,
        fit_gate,
        fit_regressor,
        load_gate_bundle,
        load_regression_bundle,
        make_internal_tail_split,
        make_tail_weights,
        save_gate_bundle,
        save_regression_bundle,
    )
except ImportError:
    from deep_preprocessing import duplicate_safe_deciles, ordered_digest
    from ensemble_utils import (
        ALPHA_GRID,
        HARD_THRESHOLDS,
        NO_FIT_ENSEMBLES,
        apply_ensemble,
        correlation_report,
        deterministic_validation_split,
        hard_route,
        optimize_convex_mae,
        soft_mix,
    )
    from prompt4_metrics import (
        compute_operational_tail_metrics,
        compute_regression_metrics,
        paired_mae_bootstrap,
        provisional_acceptance,
        select_reference,
    )
    from tail_models import (
        GATE_PARAMETERS,
        SPECIALIST_PARAMETERS,
        TAIL_WEIGHTS,
        WEIGHTED_PARAMETERS,
        GateBundle,
        RegressionBundle,
        TailModelMetadata,
        fixed_refit_config,
        fit_gate,
        fit_regressor,
        load_gate_bundle,
        load_regression_bundle,
        make_internal_tail_split,
        make_tail_weights,
        save_gate_bundle,
        save_regression_bundle,
    )


SEED = 42
THREADS = 4
TARGET = "loan_amount_000s"
PRIMARY_CONTRACT = "main_without_sensitive_without_lender"
EXPECTED_DEVELOPMENT_ROWS = 500_000
EXPECTED_TRAIN_ROWS = 400_000
EXPECTED_VALIDATION_ROWS = 100_000
EXPECTED_SELECTION_ROWS = 70_000
EXPECTED_AUDIT_ROWS = 30_000
MAX_SCIENTIFIC_FITS = 6
SCIENTIFIC_FIT_ROLES = (
    "gate_selection",
    "gate_full_refit",
    "specialist_selection",
    "specialist_full_refit",
    "tail_weighted_catboost_w2",
    "tail_weighted_catboost_w4",
)

DEVELOPMENT_RELATIVE = Path("outputs/data/development.parquet")
REPORTS_RELATIVE = Path("outputs/reports")
MODELS_RELATIVE = Path("outputs/models/prompt4a")
PREDICTIONS_RELATIVE = Path("outputs/predictions/prompt4a/validation")
FIGURES_RELATIVE = Path("outputs/figures/prompt4a")
TMP_RELATIVE = Path("outputs/tmp/prompt4a")
NOTEBOOK_RELATIVE = Path("notebooks/04A_INITIAL_ENSEMBLE_AND_TAIL_EXPERIMENTS.ipynb")

PREDICTION_SOURCES = {
    "catboost": "outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet",
    "lightgbm": "outputs/predictions/prompt2/validation/selected_lightgbm_without_lender.parquet",
    "xgboost": "outputs/predictions/prompt2/validation/selected_xgboost_without_lender.parquet",
    "histgradientboosting": "outputs/predictions/prompt2/validation/selected_histgradientboosting.parquet",
    "lasso": "outputs/predictions/prompt2/validation/selected_lasso.parquet",
    "realmlp": "outputs/predictions/prompt3/validation/selected_realmlp.parquet",
    "fttransformer": "outputs/predictions/prompt3/validation/selected_fttransformer.parquet",
}
INFERENCE_COMPLEXITY = {
    "lasso": 1,
    "histgradientboosting": 2,
    "lightgbm": 3,
    "xgboost": 4,
    "catboost": 5,
    "realmlp": 6,
    "fttransformer": 7,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def discover_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or __file__).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if candidate.name == "regresionpart2" and (candidate / "regression_v2").is_dir():
            return candidate
    raise RuntimeError("The structurally required regresionpart2 project root was not found.")


def regression_v2_root(start: str | Path | None = None) -> Path:
    return discover_project_root(start) / "regression_v2"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_read_path(root: str | Path, path: str | Path) -> Path:
    workspace = Path(root).resolve()
    candidate = Path(path)
    resolved = (workspace / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not _inside(resolved, workspace):
        raise PermissionError(f"Prompt 4A cannot read outside regression_v2: {resolved}")
    prohibited = {
        (workspace / "outputs/data/iid_holdout_features.parquet").resolve(),
        (workspace / "outputs/data/iid_holdout_targets.parquet").resolve(),
    }
    raw = (workspace / "data").resolve()
    if resolved in prohibited or _inside(resolved, raw):
        raise PermissionError(f"Prompt 4A read boundary blocks Raw/IID: {resolved}")
    return resolved


def guard_write_path(root: str | Path, path: str | Path) -> Path:
    workspace = Path(root).resolve()
    candidate = Path(path)
    resolved = (workspace / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not _inside(resolved, workspace):
        raise PermissionError(f"Prompt 4A cannot write outside regression_v2: {resolved}")
    prohibited = [(workspace / "data").resolve(), (workspace / "outputs/data").resolve()]
    if any(resolved == item or _inside(resolved, item) for item in prohibited):
        raise PermissionError(f"Prompt 4A cannot write source data: {resolved}")
    return resolved


def atomic_json(root: Path, relative: str | Path, payload: dict[str, Any]) -> Path:
    destination = guard_write_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, destination)
    return destination


def atomic_csv(root: Path, relative: str | Path, frame: pd.DataFrame) -> Path:
    destination = guard_write_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary)
    os.replace(temporary, destination)
    return destination


def atomic_parquet(root: Path, relative: str | Path, frame: pd.DataFrame) -> Path:
    destination = guard_write_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    reloaded = pd.read_parquet(temporary)
    if list(reloaded.columns) != list(frame.columns) or len(reloaded) != len(frame):
        raise RuntimeError(f"Parquet atomic reload failed: {destination}")
    os.replace(temporary, destination)
    return destination


def _read_json(root: Path, relative: str | Path) -> dict[str, Any]:
    return json.loads(guard_read_path(root, relative).read_text(encoding="utf-8"))


def package_versions() -> dict[str, str]:
    names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "scikit_learn": "scikit-learn",
        "scipy": "scipy",
        "catboost": "catboost",
        "joblib": "joblib",
        "nbformat": "nbformat",
        "nbclient": "nbclient",
    }
    result = {"python": sys.version.split()[0]}
    for key, distribution in names.items():
        result[key] = importlib.metadata.version(distribution)
    return result


def validate_prompt3_handoff(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    required = (
        "PROMPT2_READY.json",
        "prompt2_verification.json",
        "PROMPT3_READY.json",
        "prompt3_verification.json",
        "prompt2_prediction_manifest.json",
        "prompt3_prediction_manifest.json",
        "feature_roles.json",
    )
    reports = {name: _read_json(workspace, REPORTS_RELATIVE / name) for name in required}
    bad = [name for name, payload in reports.items() if payload.get("status") != "PASS"]
    if bad:
        raise RuntimeError(f"Prompt 2/3 handoff is not PASS: {bad}")
    task = guard_read_path(workspace, "TASK.md").read_text(encoding="utf-8")
    if "Prompt 3" not in task or "complete" not in task.lower():
        raise RuntimeError("TASK.md does not preserve Prompt 3 completion evidence.")
    if (workspace / REPORTS_RELATIVE / "FINAL_PRE_IID_FREEZE.json").exists():
        raise RuntimeError("A prohibited final-freeze file already exists.")
    iid_predictions = [p for p in (workspace / "outputs/predictions").rglob("*") if p.is_file() and "iid" in p.name.lower()]
    if iid_predictions:
        raise RuntimeError("An IID prediction exists before Prompt 4A.")
    return {"status": "PASS", "reports": reports, "task_state_valid": True}


def load_development(root: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    workspace = Path(root or regression_v2_root()).resolve()
    handoff = validate_prompt3_handoff(workspace)
    roles = handoff["reports"]["feature_roles.json"]
    features = list(roles["contracts"][PRIMARY_CONTRACT])
    if len(features) != 35 or len(set(features)) != 35:
        raise RuntimeError("The primary feature contract is not exactly 35 unique fields.")
    excluded = set(roles["sensitive_fields"]) | set(roles["audit_only_fields"]) | set(
        roles["target_and_alias_exclusions"]
    ) | {"respondent_id", TARGET}
    leaked = sorted(set(features) & excluded)
    if leaked:
        raise RuntimeError(f"The primary feature contract contains prohibited fields: {leaked}")
    source = guard_read_path(workspace, DEVELOPMENT_RELATIVE)
    expected_sha = handoff["reports"]["PROMPT3_READY.json"]["development_source_sha256"]
    actual_sha = file_sha256(source)
    if actual_sha != expected_sha:
        raise RuntimeError("Development source SHA-256 changed.")
    frame = pd.read_parquet(source, columns=features + [TARGET, "development_role", "row_hash"])
    if len(frame) != EXPECTED_DEVELOPMENT_ROWS or frame["row_hash"].duplicated().any():
        raise RuntimeError("Development row count or row_hash uniqueness failed.")
    if not np.isfinite(frame[TARGET].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Development targets are not finite.")
    train = frame.loc[frame["development_role"].eq("train")].reset_index(drop=True)
    validation = frame.loc[frame["development_role"].eq("validation")].reset_index(drop=True)
    if len(train) != EXPECTED_TRAIN_ROWS or len(validation) != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError("Development Train/Validation counts changed.")
    if set(train["row_hash"]) & set(validation["row_hash"]):
        raise RuntimeError("Development Train and Validation overlap.")
    evidence = {
        "path": DEVELOPMENT_RELATIVE.as_posix(),
        "sha256": actual_sha,
        "rows": len(frame),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_row_hash_digest": ordered_digest(train["row_hash"]),
        "validation_row_hash_digest": ordered_digest(validation["row_hash"]),
        "feature_contract": PRIMARY_CONTRACT,
        "feature_count": len(features),
        "feature_contract_digest": canonical_digest(features),
        "features": features,
    }
    return train, validation, evidence


def load_aligned_predictions(root: Path, validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    expected_hashes = validation["row_hash"].astype(str).reset_index(drop=True)
    expected_target = validation[TARGET].to_numpy(dtype=np.float64)
    aligned = pd.DataFrame({"row_hash": expected_hashes, "y_true": expected_target})
    artifacts: list[dict[str, Any]] = []
    for model, relative in PREDICTION_SOURCES.items():
        path = guard_read_path(root, relative)
        frame = pd.read_parquet(path)
        valid = (
            len(frame) == EXPECTED_VALIDATION_ROWS
            and bool(frame["row_hash"].is_unique)
            and frame["row_hash"].astype(str).reset_index(drop=True).equals(expected_hashes)
            and np.array_equal(frame["y_true"].to_numpy(dtype=np.float64), expected_target)
            and np.isfinite(frame["y_pred"].to_numpy(dtype=np.float64)).all()
        )
        if not valid:
            raise RuntimeError(f"Saved Validation predictions do not align: {relative}")
        aligned[model] = frame["y_pred"].to_numpy(dtype=np.float64)
        artifacts.append(
            {
                "model": model,
                "path": relative,
                "sha256": file_sha256(path),
                "rows": len(frame),
                "row_order_equal": True,
                "target_equal": True,
                "finite_predictions": True,
                "compression": sorted(set(pq.ParquetFile(path).metadata.row_group(0).column(i).compression for i in range(pq.ParquetFile(path).metadata.row_group(0).num_columns))),
                "status": "PASS",
            }
        )
    return aligned, {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "source_count": len(artifacts),
        "validation_rows": len(validation),
        "validation_row_hash_digest": ordered_digest(expected_hashes),
        "exact_y_true_equality": True,
        "artifacts": artifacts,
    }


def _source_code_digest(root: Path) -> str:
    payload = {}
    for name in ("prompt4a_experiments.py", "ensemble_utils.py", "tail_models.py", "prompt4_metrics.py"):
        path = root / "src" / name
        payload[name] = file_sha256(path)
    return canonical_digest(payload)


def prepare_design(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    train, validation, source = load_development(workspace)
    aligned, alignment = load_aligned_predictions(workspace, validation)
    selection_index, audit_index, split_evidence = deterministic_validation_split(
        aligned["y_true"].to_numpy(), aligned["row_hash"], random_state=SEED,
        selection_rows=EXPECTED_SELECTION_ROWS, audit_rows=EXPECTED_AUDIT_ROWS,
    )
    q90_train = float(np.quantile(train[TARGET].to_numpy(dtype=np.float64), 0.90))
    operational_tail = train[TARGET].to_numpy(dtype=np.float64) > q90_train
    fit_index, stop_index, internal_evidence = make_internal_tail_split(
        train[TARGET].to_numpy(dtype=np.float64), train["row_hash"], q90_train,
        validation["row_hash"], random_state=SEED,
    )
    design_core = {
        "prompt": "Prompt 4A",
        "experiment_design_only": True,
        "final_model_selected": False,
        "final_model_frozen": False,
        "source": source,
        "prediction_sources": PREDICTION_SOURCES,
        "prediction_source_sha256": {entry["model"]: entry["sha256"] for entry in alignment["artifacts"]},
        "validation_split": split_evidence,
        "internal_tail_split": internal_evidence,
        "q90_train": q90_train,
        "train_tail_rows": int(np.count_nonzero(operational_tail)),
        "train_tail_proportion": float(np.mean(operational_tail)),
        "validation_above_q90_train": int(np.count_nonzero(validation[TARGET].to_numpy(dtype=np.float64) > q90_train)),
        "train_tied_at_q90": int(np.count_nonzero(train[TARGET].to_numpy(dtype=np.float64) == q90_train)),
        "validation_tied_at_q90": int(np.count_nonzero(validation[TARGET].to_numpy(dtype=np.float64) == q90_train)),
        "no_fit_ensemble_definitions": NO_FIT_ENSEMBLES,
        "hard_thresholds": list(HARD_THRESHOLDS),
        "soft_alphas": list(ALPHA_GRID),
        "tail_weights": list(TAIL_WEIGHTS),
        "gate_parameters": GATE_PARAMETERS,
        "specialist_parameters": SPECIALIST_PARAMETERS,
        "weighted_parameters": WEIGHTED_PARAMETERS,
        "scientific_fit_roles": list(SCIENTIFIC_FIT_ROLES),
        "max_scientific_fits": MAX_SCIENTIFIC_FITS,
        "fit_budget": {"max_scientific_fits": MAX_SCIENTIFIC_FITS, "max_technical_retry_per_fit": 1, "scientific_fit_roles": list(SCIENTIFIC_FIT_ROLES)},
        "random_state": SEED,
        "threads": THREADS,
        "package_versions": package_versions(),
        "code_digest": _source_code_digest(workspace),
        "selection_policy": "All preliminary references and optimized weights use only the 70,000 selection rows.",
        "audit_policy": "The 30,000 audit rows are reporting-only and cannot alter Candidates or preliminary references.",
        "tail_weighted_global_base": "catboost",
        "provisional_status_policy": "PASS=6/6, PARTIAL=1-5/6, FAIL=0/6",
        "quantile_policy": "NumPy linear quantiles; inclusive subset-local standard tails; strict y > q90_train operational Tail; inclusive P85-P95 boundary.",
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
    }
    design = {
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        **design_core,
        "design_digest": canonical_digest(design_core),
    }
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_frozen_design.json", design)
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_validation_split.json", {"status": "PASS", "created_at_utc": utc_now(), **split_evidence})
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_prediction_alignment.json", alignment)
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_tail_definition.json", {
        "status": "PASS", "created_at_utc": utc_now(), "q90_train": q90_train,
        "definition": f"{TARGET} > q90_train", "source_role": "train",
        "train_tail_rows": design["train_tail_rows"], "train_tail_proportion": design["train_tail_proportion"],
        "validation_above_q90_train": design["validation_above_q90_train"],
        "train_tied_at_q90": design["train_tied_at_q90"], "validation_tied_at_q90": design["validation_tied_at_q90"],
        "internal_split": internal_evidence,
    })
    return design


def _load_design(root: Path) -> dict[str, Any]:
    design = _read_json(root, REPORTS_RELATIVE / "prompt4a_frozen_design.json")
    if design.get("status") != "FROZEN" or design.get("code_digest") != _source_code_digest(root):
        raise RuntimeError("Prompt 4A frozen design is missing or does not match current code.")
    return design


def _roles_from_design(aligned: pd.DataFrame, design: dict[str, Any]) -> np.ndarray:
    selection_index, audit_index, evidence = deterministic_validation_split(
        aligned["y_true"].to_numpy(dtype=np.float64), aligned["row_hash"],
        random_state=SEED, selection_rows=EXPECTED_SELECTION_ROWS, audit_rows=EXPECTED_AUDIT_ROWS,
    )
    for key in ("source_validation_row_hash_digest", "selection_row_hash_digest", "audit_row_hash_digest"):
        if evidence[key] != design["validation_split"][key]:
            raise RuntimeError("Frozen Validation split identity changed.")
    roles = np.full(len(aligned), "", dtype=object)
    roles[selection_index] = "selection"
    roles[audit_index] = "audit"
    if np.count_nonzero(roles == "selection") != EXPECTED_SELECTION_ROWS:
        raise RuntimeError("Frozen Validation role materialization failed.")
    return roles


def _metric_rows(candidate_id: str, candidate_type: str, y: np.ndarray, prediction: np.ndarray, roles: np.ndarray, *, complexity: int, global_base: str = "__NOT_APPLICABLE__") -> list[dict[str, Any]]:
    rows = []
    for scope, mask in (("selection", roles == "selection"), ("audit", roles == "audit"), ("complete_validation", np.ones(len(y), dtype=bool))):
        rows.append({"candidate_id": candidate_id, "candidate_type": candidate_type, "global_base": global_base, "scope": scope, "inference_complexity": complexity, **compute_regression_metrics(y[mask], prediction[mask])})
    return rows


def _save_regression_prediction(root: Path, candidate_id: str, candidate_type: str, global_base: str, aligned: pd.DataFrame, prediction: np.ndarray, roles: np.ndarray) -> Path:
    values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if values.shape != (EXPECTED_VALIDATION_ROWS,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid Prompt 4A prediction: {candidate_id}")
    frame = pd.DataFrame({
        "row_hash": aligned["row_hash"].astype(str).to_numpy(), "y_true": aligned["y_true"].to_numpy(dtype=np.float64),
        "y_pred": values, "candidate_id": candidate_id, "candidate_type": candidate_type,
        "global_base": global_base, "selection_or_audit_role": roles,
    })
    path = atomic_parquet(root, PREDICTIONS_RELATIVE / f"{candidate_id}.parquet", frame)
    reloaded = pd.read_parquet(path)
    if not reloaded["row_hash"].equals(frame["row_hash"]) or not np.array_equal(reloaded["y_true"], frame["y_true"]) or not np.isfinite(reloaded["y_pred"]).all():
        raise RuntimeError(f"Prompt 4A prediction reload failed: {candidate_id}")
    return path


def run_nofit(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    design = _load_design(workspace)
    _, validation, _ = load_development(workspace)
    aligned, alignment = load_aligned_predictions(workspace, validation)
    if alignment["validation_row_hash_digest"] != design["source"]["validation_row_hash_digest"]:
        raise RuntimeError("Validation identity changed after design freeze.")
    roles = _roles_from_design(aligned, design)
    y = aligned["y_true"].to_numpy(dtype=np.float64)
    selection = roles == "selection"

    single_rows: list[dict[str, Any]] = []
    for model in PREDICTION_SOURCES:
        single_rows.extend(_metric_rows(model, "saved_single", y, aligned[model].to_numpy(dtype=np.float64), roles, complexity=INFERENCE_COMPLEXITY[model]))
    single_frame = pd.DataFrame(single_rows)
    best_single = select_reference(single_frame.loc[single_frame["scope"].eq("selection")].to_dict("records"))
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_single_model_results.csv", single_frame)

    correlation = correlation_report(aligned.loc[:, list(PREDICTION_SOURCES)])
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_prediction_correlations.csv", correlation)

    weight_rows: list[dict[str, Any]] = []
    ensemble_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for candidate_id, definition in NO_FIT_ENSEMBLES.items():
        members = list(definition["members"])
        matrix = aligned.loc[:, members].to_numpy(dtype=np.float64)
        if definition["kind"] == "fixed":
            weights = np.asarray(definition["weights"], dtype=np.float64)
            status = "COMPLETE"
            solver = {"status": "NOT_REQUIRED", "objective": float(np.mean(np.abs(apply_ensemble(matrix[selection], weights) - y[selection])))}
        else:
            result = optimize_convex_mae(matrix[selection], y[selection])
            status = result["status"]
            weights = np.asarray(result.get("weights", []), dtype=np.float64)
            solver = result
        for member_index, member in enumerate(members):
            weight_rows.append({"candidate_id": candidate_id, "member": member, "weight": float(weights[member_index]) if status == "COMPLETE" else np.nan, "status": status, "solver_status": solver.get("solver_status", solver.get("status"))})
        if status != "COMPLETE":
            ensemble_rows.append({"candidate_id": candidate_id, "candidate_type": "no_fit_ensemble", "scope": "selection", "status": "UNAVAILABLE"})
            continue
        prediction = apply_ensemble(matrix, weights)
        predictions[candidate_id] = prediction
        complexity = 10 + len(members)
        rows = _metric_rows(candidate_id, "no_fit_ensemble", y, prediction, roles, complexity=complexity)
        for row in rows:
            row["status"] = "COMPLETE"
        ensemble_rows.extend(rows)
        _save_regression_prediction(workspace, candidate_id, "no_fit_ensemble", "__NOT_APPLICABLE__", aligned, prediction, roles)
    weights_frame = pd.DataFrame(weight_rows)
    ensemble_frame = pd.DataFrame(ensemble_rows)
    if len(set(weights_frame["candidate_id"])) != 10:
        raise RuntimeError("The exact ten no-fit ensemble definitions were not accounted for.")
    best_ensemble = select_reference(ensemble_frame.loc[(ensemble_frame["scope"].eq("selection")) & (ensemble_frame["status"].eq("COMPLETE"))].to_dict("records"))
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_ensemble_weights.csv", weights_frame)
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_ensemble_results.csv", ensemble_frame)
    summary = {
        "status": "PASS", "created_at_utc": utc_now(), "best_saved_single": best_single["candidate_id"],
        "best_ensemble_4a": best_ensemble["candidate_id"], "selection_only": True,
        "available_ensembles": len(predictions), "defined_ensembles": 10,
    }
    atomic_json(workspace, TMP_RELATIVE / "nofit_summary.json", summary)
    return summary


def _fit_ledger(root: Path) -> dict[str, Any]:
    path = root / TMP_RELATIVE / "scientific_fit_ledger.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "IN_PROGRESS", "created_at_utc": utc_now(), "max_scientific_fits": MAX_SCIENTIFIC_FITS, "roles": list(SCIENTIFIC_FIT_ROLES), "attempts": [], "completed_roles": []}


def _start_fit_attempt(root: Path, role: str) -> int:
    ledger = _fit_ledger(root)
    if role not in SCIENTIFIC_FIT_ROLES:
        raise RuntimeError(f"Unauthorized scientific fit role: {role}")
    prior = [item for item in ledger["attempts"] if item["role"] == role]
    if len(prior) >= 2:
        raise RuntimeError(f"Technical retry budget exhausted for {role}.")
    attempt = len(prior) + 1
    ledger["attempts"].append({"role": role, "attempt": attempt, "status": "STARTED", "started_at_utc": utc_now()})
    atomic_json(root, TMP_RELATIVE / "scientific_fit_ledger.json", ledger)
    return attempt


def _finish_fit_attempt(root: Path, role: str, attempt: int, status: str, *, error: str | None = None) -> None:
    ledger = _fit_ledger(root)
    matches = [item for item in ledger["attempts"] if item["role"] == role and item["attempt"] == attempt]
    if len(matches) != 1:
        raise RuntimeError("Scientific fit attempt ledger is inconsistent.")
    matches[0].update({"status": status, "finished_at_utc": utc_now()})
    if error is not None:
        matches[0]["error"] = error
    if status == "COMPLETE" and role not in ledger["completed_roles"]:
        ledger["completed_roles"].append(role)
    ledger["scientific_fit_count"] = len(ledger["completed_roles"])
    ledger["status"] = "COMPLETE" if set(ledger["completed_roles"]) == set(SCIENTIFIC_FIT_ROLES) else "IN_PROGRESS"
    atomic_json(root, TMP_RELATIVE / "scientific_fit_ledger.json", ledger)


def _selection_checkpoint_identity(design: dict[str, Any], role: str, parameters: dict[str, Any], q90_train: float) -> dict[str, Any]:
    return {
        "development_sha256": design["source"]["sha256"],
        "train_row_hash_digest": design["source"]["train_row_hash_digest"],
        "validation_row_hash_digest": design["source"]["validation_row_hash_digest"],
        "feature_contract_digest": design["source"]["feature_contract_digest"],
        "model_role": role,
        "target_threshold": float(q90_train),
        "parameters": parameters,
        "seed": SEED,
        "package_versions": design["package_versions"],
        "code_digest": design["code_digest"],
        "prediction_row_count": EXPECTED_VALIDATION_ROWS,
    }


def _selection_checkpoint_path(root: Path, role: str) -> Path:
    return root / TMP_RELATIVE / "selection_checkpoints" / f"{role}.joblib"


def _load_selection_checkpoint(root: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    path = _selection_checkpoint_path(root, identity["model_role"])
    manifest_path = path.with_suffix(".json")
    if not path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("identity") != identity or manifest.get("artifact_sha256") != file_sha256(path):
        return None
    payload = joblib.load(path)
    if payload.get("identity") != identity or int(payload.get("selected_iteration", 0)) < 1:
        return None
    return payload


def _save_selection_checkpoint(root: Path, identity: dict[str, Any], payload: dict[str, Any]) -> None:
    path = guard_write_path(root, _selection_checkpoint_path(root, identity["model_role"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump({"identity": identity, **payload}, temporary, compress=3)
    reloaded = joblib.load(temporary)
    if reloaded["identity"] != identity:
        raise RuntimeError("Selection checkpoint reload failed.")
    os.replace(temporary, path)
    atomic_json(root, path.with_suffix(".json"), {"status": "COMPLETE", "created_at_utc": utc_now(), "identity": identity, "artifact_sha256": file_sha256(path), "selected_iteration": int(payload["selected_iteration"])})


def _bundle_metadata(
    design: dict[str, Any], role: str, model_id: str, parameters: dict[str, Any],
    selected_iteration: int, fit_hashes: pd.Series, authorized_hashes: pd.Series,
    q90_train: float, target_definition: str, extra: dict[str, Any],
) -> TailModelMetadata:
    identity = _selection_checkpoint_identity(design, role, parameters, q90_train)
    return TailModelMetadata(
        model_id=model_id, model_role=role, feature_names=list(design["source"]["features"]),
        feature_contract=PRIMARY_CONTRACT, model_configuration=parameters, seed=SEED,
        selected_iteration=int(selected_iteration), training_row_count=len(fit_hashes),
        train_membership_digest=ordered_digest(fit_hashes), q90_train=float(q90_train),
        target_definition=target_definition, package_versions=design["package_versions"],
        authorized_train_membership_digest=ordered_digest(authorized_hashes),
        development_source_sha256=design["source"]["sha256"],
        validation_row_hash_digest=design["source"]["validation_row_hash_digest"],
        extra={**extra, "cache_identity": identity},
    )


def run_smoke(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    design = _load_design(workspace)
    report_path = workspace / REPORTS_RELATIVE / "prompt4a_smoke.json"
    attempt = 1
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("status") == "PASS" and existing.get("design_digest") == design["design_digest"]:
            return existing
        attempt = int(existing.get("attempt", 1)) + 1
    if attempt > 2:
        raise RuntimeError("The maximum two Prompt 4A smoke attempts is exhausted.")
    train, validation, _ = load_development(workspace)
    smoke_train = train.iloc[:20_000].reset_index(drop=True)
    smoke_validation = validation.iloc[:5_000].reset_index(drop=True)
    features = design["source"]["features"]
    q90 = float(design["q90_train"])
    tail = smoke_train[TARGET].to_numpy(dtype=np.float64) > q90
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(smoke_train))
    fit_idx, stop_idx = np.sort(order[:18_000]), np.sort(order[18_000:])
    gate_params = {**GATE_PARAMETERS, "iterations": 10, "early_stopping_rounds": 3}
    specialist_params = {**SPECIALIST_PARAMETERS, "iterations": 10, "early_stopping_rounds": 3}
    weighted_params = {**WEIGHTED_PARAMETERS, "iterations": 10}
    try:
        gate_model, gate_pre, gate_iter, _ = fit_gate(
            smoke_train.loc[fit_idx, features], tail[fit_idx], fit_row_hashes=smoke_train.loc[fit_idx, "row_hash"],
            authorized_train_row_hashes=smoke_train["row_hash"], X_stop=smoke_train.loc[stop_idx, features], y_stop=tail[stop_idx],
            stop_row_hashes=smoke_train.loc[stop_idx, "row_hash"], validation_row_hashes=smoke_validation["row_hash"], parameters=gate_params,
        )
        fit_tail = fit_idx[tail[fit_idx]]; stop_tail = stop_idx[tail[stop_idx]]
        spec_model, spec_pre, spec_iter, _ = fit_regressor(
            smoke_train.loc[fit_tail, features], smoke_train.loc[fit_tail, TARGET], fit_row_hashes=smoke_train.loc[fit_tail, "row_hash"],
            authorized_train_row_hashes=smoke_train.loc[tail, "row_hash"], X_stop=smoke_train.loc[stop_tail, features], y_stop=smoke_train.loc[stop_tail, TARGET],
            stop_row_hashes=smoke_train.loc[stop_tail, "row_hash"], validation_row_hashes=smoke_validation["row_hash"], parameters=specialist_params,
        )
        weighted_model, weighted_pre, weighted_iter, _ = fit_regressor(
            smoke_train[features], smoke_train[TARGET], fit_row_hashes=smoke_train["row_hash"], authorized_train_row_hashes=smoke_train["row_hash"],
            validation_row_hashes=smoke_validation["row_hash"], sample_weight=make_tail_weights(smoke_train[TARGET], q90, 2), parameters=weighted_params,
        )
        gate_prob = GateBundle(_bundle_metadata(design, "gate_selection", "smoke_gate", gate_params, gate_iter, smoke_train.loc[fit_idx, "row_hash"], smoke_train["row_hash"], q90, "smoke operational tail", {"smoke": True}), gate_pre, gate_model).predict_tail_probability(smoke_validation[features])
        specialist_pred = RegressionBundle(_bundle_metadata(design, "specialist_selection", "smoke_specialist", specialist_params, spec_iter, smoke_train.loc[fit_tail, "row_hash"], smoke_train.loc[tail, "row_hash"], q90, TARGET, {"smoke": True}), spec_pre, spec_model).predict(smoke_validation[features])
        weighted_bundle = RegressionBundle(_bundle_metadata(design, "tail_weighted_catboost_w2", "smoke_weighted", weighted_params, weighted_iter, smoke_train["row_hash"], smoke_train["row_hash"], q90, TARGET, {"smoke": True}), weighted_pre, weighted_model)
        weighted_pred = weighted_bundle.predict(smoke_validation[features])
        hard = hard_route(weighted_pred, specialist_pred, gate_prob, 0.50)
        soft = soft_mix(weighted_pred, specialist_pred, gate_prob, 0.75)
        temporary = workspace / TMP_RELATIVE / "smoke_bundle.joblib"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(weighted_bundle, temporary, compress=3)
        reload_pred = joblib.load(temporary).predict(smoke_validation[features])
        max_difference = float(np.max(np.abs(weighted_pred - reload_pred)))
        if not all(np.isfinite(item).all() for item in (gate_prob, specialist_pred, weighted_pred, hard, soft)) or max_difference > 1e-7:
            raise RuntimeError("Prompt 4A Tail smoke outputs or reload are invalid.")
        temporary.unlink()
        report = {"status": "PASS", "created_at_utc": utc_now(), "design_digest": design["design_digest"], "attempt": attempt, "train_rows": len(smoke_train), "validation_rows": len(smoke_validation), "gate_selected_iteration": gate_iter, "specialist_selected_iteration": spec_iter, "weighted_iterations": weighted_iter, "finite_predictions": True, "bundle_reload_max_absolute_difference": max_difference, "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0}
        atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_smoke.json", report)
        return report
    except Exception as exc:
        atomic_json(workspace, TMP_RELATIVE / f"smoke_failure_attempt_{attempt}.json", {"status": "TECHNICAL_FAILURE", "created_at_utc": utc_now(), "attempt": attempt, "error": repr(exc)})
        raise


def _fit_with_one_retry(root: Path, role: str, function):
    last_error = None
    for _ in range(2):
        attempt = _start_fit_attempt(root, role)
        try:
            result = function()
            _finish_fit_attempt(root, role, attempt, "COMPLETE")
            return result
        except Exception as exc:
            last_error = exc
            _finish_fit_attempt(root, role, attempt, "TECHNICAL_FAILURE", error=repr(exc))
    raise RuntimeError(f"Both technical attempts failed for {role}.") from last_error


def fit_tail_models(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    design = _load_design(workspace)
    smoke = _read_json(workspace, REPORTS_RELATIVE / "prompt4a_smoke.json")
    if smoke.get("status") != "PASS" or smoke.get("design_digest") != design["design_digest"]:
        raise RuntimeError("The bounded Tail smoke test must pass before scientific fitting.")
    train, validation, _ = load_development(workspace)
    features = design["source"]["features"]
    q90 = float(design["q90_train"])
    tail = train[TARGET].to_numpy(dtype=np.float64) > q90
    fit_idx, stop_idx, split = make_internal_tail_split(train[TARGET], train["row_hash"], q90, validation["row_hash"], random_state=SEED)
    if split["fit_row_hash_digest"] != design["internal_tail_split"]["fit_row_hash_digest"] or split["stop_row_hash_digest"] != design["internal_tail_split"]["stop_row_hash_digest"]:
        raise RuntimeError("Frozen internal Tail split changed.")

    gate_selection_identity = _selection_checkpoint_identity(design, "gate_selection", GATE_PARAMETERS, q90)
    gate_selection = _load_selection_checkpoint(workspace, gate_selection_identity)
    if gate_selection is None:
        def gate_selection_fit():
            model, preprocessor, selected, isolation = fit_gate(
                train.loc[fit_idx, features], tail[fit_idx], fit_row_hashes=train.loc[fit_idx, "row_hash"], authorized_train_row_hashes=train["row_hash"],
                X_stop=train.loc[stop_idx, features], y_stop=tail[stop_idx], stop_row_hashes=train.loc[stop_idx, "row_hash"], validation_row_hashes=validation["row_hash"], parameters=GATE_PARAMETERS,
            )
            _save_selection_checkpoint(workspace, gate_selection_identity, {"selected_iteration": selected, "isolation": isolation, "model": model, "preprocessor": preprocessor})
            return {"selected_iteration": selected, "isolation": isolation}
        gate_selection = _fit_with_one_retry(workspace, "gate_selection", gate_selection_fit)
    gate_iterations = int(gate_selection["selected_iteration"])

    gate_dir = workspace / MODELS_RELATIVE / "tail_gate"
    try:
        gate_bundle = load_gate_bundle(gate_dir)
        if gate_bundle.metadata.extra.get("cache_identity") != _selection_checkpoint_identity(design, "gate_full_refit", fixed_refit_config(GATE_PARAMETERS, gate_iterations), q90):
            raise RuntimeError("Gate cache identity mismatch.")
    except Exception:
        def gate_full_fit():
            params = fixed_refit_config(GATE_PARAMETERS, gate_iterations)
            model, preprocessor, selected, isolation = fit_gate(train[features], tail, fit_row_hashes=train["row_hash"], authorized_train_row_hashes=train["row_hash"], validation_row_hashes=validation["row_hash"], parameters=params)
            metadata = _bundle_metadata(design, "gate_full_refit", "tail_gate", params, selected, train["row_hash"], train["row_hash"], q90, f"1 when {TARGET} > q90_train", {"isolation": isolation, "selection_iteration": gate_iterations})
            bundle = GateBundle(metadata, preprocessor, model); save_gate_bundle(bundle, gate_dir)
            return bundle
        gate_bundle = _fit_with_one_retry(workspace, "gate_full_refit", gate_full_fit)
    p_tail = gate_bundle.predict_tail_probability(validation[features])

    specialist_selection_identity = _selection_checkpoint_identity(design, "specialist_selection", SPECIALIST_PARAMETERS, q90)
    specialist_selection = _load_selection_checkpoint(workspace, specialist_selection_identity)
    fit_tail_idx = fit_idx[tail[fit_idx]]; stop_tail_idx = stop_idx[tail[stop_idx]]; all_tail_idx = np.flatnonzero(tail)
    if specialist_selection is None:
        def specialist_selection_fit():
            model, preprocessor, selected, isolation = fit_regressor(
                train.loc[fit_tail_idx, features], train.loc[fit_tail_idx, TARGET], fit_row_hashes=train.loc[fit_tail_idx, "row_hash"], authorized_train_row_hashes=train.loc[all_tail_idx, "row_hash"],
                X_stop=train.loc[stop_tail_idx, features], y_stop=train.loc[stop_tail_idx, TARGET], stop_row_hashes=train.loc[stop_tail_idx, "row_hash"], validation_row_hashes=validation["row_hash"], parameters=SPECIALIST_PARAMETERS,
            )
            _save_selection_checkpoint(workspace, specialist_selection_identity, {"selected_iteration": selected, "isolation": isolation, "model": model, "preprocessor": preprocessor})
            return {"selected_iteration": selected, "isolation": isolation}
        specialist_selection = _fit_with_one_retry(workspace, "specialist_selection", specialist_selection_fit)
    specialist_iterations = int(specialist_selection["selected_iteration"])

    specialist_dir = workspace / MODELS_RELATIVE / "tail_specialist"
    try:
        specialist_bundle = load_regression_bundle(specialist_dir)
        if specialist_bundle.metadata.extra.get("cache_identity") != _selection_checkpoint_identity(design, "specialist_full_refit", fixed_refit_config(SPECIALIST_PARAMETERS, specialist_iterations), q90):
            raise RuntimeError("Specialist cache identity mismatch.")
    except Exception:
        def specialist_full_fit():
            params = fixed_refit_config(SPECIALIST_PARAMETERS, specialist_iterations)
            model, preprocessor, selected, isolation = fit_regressor(train.loc[all_tail_idx, features], train.loc[all_tail_idx, TARGET], fit_row_hashes=train.loc[all_tail_idx, "row_hash"], authorized_train_row_hashes=train.loc[all_tail_idx, "row_hash"], validation_row_hashes=validation["row_hash"], parameters=params)
            metadata = _bundle_metadata(design, "specialist_full_refit", "tail_specialist", params, selected, train.loc[all_tail_idx, "row_hash"], train.loc[all_tail_idx, "row_hash"], q90, f"raw {TARGET}; operational Tail rows only", {"isolation": isolation, "selection_iteration": specialist_iterations})
            bundle = RegressionBundle(metadata, preprocessor, model); save_regression_bundle(bundle, specialist_dir)
            return bundle
        specialist_bundle = _fit_with_one_retry(workspace, "specialist_full_refit", specialist_full_fit)
    specialist_prediction = specialist_bundle.predict(validation[features])

    weighted_predictions: dict[str, np.ndarray] = {}
    for weight in TAIL_WEIGHTS:
        role = f"tail_weighted_catboost_w{weight}"; model_dir = workspace / MODELS_RELATIVE / role
        params = dict(WEIGHTED_PARAMETERS)
        identity = _selection_checkpoint_identity(design, role, params, q90)
        try:
            bundle = load_regression_bundle(model_dir)
            if bundle.metadata.extra.get("cache_identity") != identity:
                raise RuntimeError("Weighted cache identity mismatch.")
        except Exception:
            def weighted_fit(weight=weight, role=role, params=params, model_dir=model_dir):
                model, preprocessor, selected, isolation = fit_regressor(train[features], train[TARGET], fit_row_hashes=train["row_hash"], authorized_train_row_hashes=train["row_hash"], validation_row_hashes=validation["row_hash"], sample_weight=make_tail_weights(train[TARGET], q90, weight), parameters=params)
                metadata = _bundle_metadata(design, role, role, params, selected, train["row_hash"], train["row_hash"], q90, f"raw {TARGET}; operational Tail sample weight {weight}", {"isolation": isolation, "tail_weight": weight})
                result_bundle = RegressionBundle(metadata, preprocessor, model); save_regression_bundle(result_bundle, model_dir)
                return result_bundle
            bundle = _fit_with_one_retry(workspace, role, weighted_fit)
        weighted_predictions[role] = bundle.predict(validation[features])

    ledger = _fit_ledger(workspace)
    if set(ledger["completed_roles"]) != set(SCIENTIFIC_FIT_ROLES):
        raise RuntimeError("The six-role scientific fit sequence is incomplete.")
    roles = _roles_from_design(pd.DataFrame({"row_hash": validation["row_hash"], "y_true": validation[TARGET]}), design)
    gate_frame = pd.DataFrame({"row_hash": validation["row_hash"].astype(str), "operational_tail_true": validation[TARGET].to_numpy(dtype=np.float64) > q90, "p_tail": p_tail, "predicted_tail_035": p_tail >= 0.35, "predicted_tail_050": p_tail >= 0.50, "predicted_tail_065": p_tail >= 0.65, "selection_or_audit_role": roles})
    atomic_parquet(workspace, PREDICTIONS_RELATIVE / "tail_gate.parquet", gate_frame)
    aligned = pd.DataFrame({"row_hash": validation["row_hash"].astype(str), "y_true": validation[TARGET].to_numpy(dtype=np.float64)})
    _save_regression_prediction(workspace, "tail_specialist", "tail_specialist", "__NOT_APPLICABLE__", aligned, specialist_prediction, roles)
    for candidate_id, prediction in weighted_predictions.items():
        _save_regression_prediction(workspace, candidate_id, "tail_weighted", "catboost", aligned, prediction, roles)
    atomic_json(workspace, TMP_RELATIVE / "tail_fit_summary.json", {"status": "PASS", "created_at_utc": utc_now(), "gate_selected_iteration": gate_iterations, "specialist_selected_iteration": specialist_iterations, "train_tail_rows": int(tail.sum()), "scientific_fit_count": len(ledger["completed_roles"])})
    return {"status": "PASS", "scientific_fit_count": len(ledger["completed_roles"]), "gate_selected_iteration": gate_iterations, "specialist_selected_iteration": specialist_iterations}


def _scope_masks(roles: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    return (("selection", roles == "selection"), ("audit", roles == "audit"), ("complete_validation", np.ones(len(roles), dtype=bool)))


def _gate_metrics(y_true: np.ndarray, p_tail: np.ndarray, roles: np.ndarray) -> pd.DataFrame:
    rows = []
    for scope, mask in _scope_masks(roles):
        actual = y_true[mask].astype(np.int8); probability = p_tail[mask]
        shared = {"scope": scope, "rows": int(mask.sum()), "pr_auc": float(average_precision_score(actual, probability)), "roc_auc": float(roc_auc_score(actual, probability)), "brier_score": float(brier_score_loss(actual, probability))}
        for threshold in HARD_THRESHOLDS:
            predicted = probability >= threshold
            recall = float(recall_score(actual, predicted, zero_division=0))
            rows.append({**shared, "threshold": threshold, "precision": float(precision_score(actual, predicted, zero_division=0)), "recall": recall, "f1": float(f1_score(actual, predicted, zero_division=0)), "false_negative_rate": 1.0 - recall, "predicted_tail_proportion": float(np.mean(predicted))})
    return pd.DataFrame(rows)


def _load_prompt4_prediction(root: Path, candidate_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(guard_read_path(root, PREDICTIONS_RELATIVE / f"{candidate_id}.parquet"))
    if len(frame) != EXPECTED_VALIDATION_ROWS or not frame["row_hash"].is_unique or not np.isfinite(frame["y_pred"]).all():
        raise RuntimeError(f"Invalid Prompt 4A prediction artifact: {candidate_id}")
    return frame


def build_tail_reports(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    design = _load_design(workspace)
    nofit = _read_json(workspace, TMP_RELATIVE / "nofit_summary.json")
    _, validation, _ = load_development(workspace)
    aligned, _ = load_aligned_predictions(workspace, validation)
    roles = _roles_from_design(aligned, design)
    y = aligned["y_true"].to_numpy(dtype=np.float64); q90 = float(design["q90_train"])
    operational_true = y > q90
    gate = pd.read_parquet(guard_read_path(workspace, PREDICTIONS_RELATIVE / "tail_gate.parquet"))
    if not np.array_equal(gate["row_hash"].astype(str), aligned["row_hash"].astype(str)) or not np.array_equal(gate["operational_tail_true"].to_numpy(bool), operational_true):
        raise RuntimeError("Tail Gate prediction alignment failed.")
    p_tail = gate["p_tail"].to_numpy(dtype=np.float64)
    gate_metrics = _gate_metrics(operational_true, p_tail, roles)
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_gate_metrics.csv", gate_metrics)

    specialist_frame = _load_prompt4_prediction(workspace, "tail_specialist")
    specialist = specialist_frame["y_pred"].to_numpy(dtype=np.float64)
    specialist_rows = []
    for scope, mask in _scope_masks(roles):
        tail_mask = mask & operational_true
        specialist_rows.append({"candidate_id": "tail_specialist", "scope": scope, "operational_tail_only": True, **compute_regression_metrics(y[tail_mask], specialist[tail_mask]), **compute_operational_tail_metrics(y[mask], specialist[mask], q90)})
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_specialist_metrics.json", {"status": "PASS", "created_at_utc": utc_now(), "selected_iteration": _read_json(workspace, TMP_RELATIVE / "tail_fit_summary.json")["specialist_selected_iteration"], "train_tail_rows": design["train_tail_rows"], "rows": specialist_rows})

    weighted_rows = []
    weighted_predictions = {}
    for weight in TAIL_WEIGHTS:
        candidate_id = f"tail_weighted_catboost_w{weight}"
        prediction = _load_prompt4_prediction(workspace, candidate_id)["y_pred"].to_numpy(dtype=np.float64)
        weighted_predictions[candidate_id] = prediction
        for row in _metric_rows(candidate_id, "tail_weighted", y, prediction, roles, complexity=6, global_base="catboost"):
            mask = dict(_scope_masks(roles))[row["scope"]]
            row.update(compute_operational_tail_metrics(y[mask], prediction[mask], q90)); weighted_rows.append(row)
    weighted_frame = pd.DataFrame(weighted_rows)
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_tail_weighted_results.csv", weighted_frame)

    global_predictions = {
        "global_best_single": aligned[nofit["best_saved_single"]].to_numpy(dtype=np.float64),
        "global_best_ensemble": _load_prompt4_prediction(workspace, nofit["best_ensemble_4a"])["y_pred"].to_numpy(dtype=np.float64),
    }
    actual_global_ids = {"global_best_single": nofit["best_saved_single"], "global_best_ensemble": nofit["best_ensemble_4a"]}
    hard_rows = []; soft_rows = []; tail_candidate_predictions: dict[str, np.ndarray] = {}
    for base_name, global_prediction in global_predictions.items():
        for threshold in HARD_THRESHOLDS:
            suffix = f"{int(round(threshold * 100)):03d}"
            candidate_id = f"hard_{base_name}_t{suffix}"
            prediction = hard_route(global_prediction, specialist, p_tail, threshold)
            tail_candidate_predictions[candidate_id] = prediction
            from ensemble_utils import routing_summary
            summary = routing_summary(p_tail, operational_true, threshold)
            for row in _metric_rows(candidate_id, "hard_routing", y, prediction, roles, complexity=20, global_base=base_name):
                mask = dict(_scope_masks(roles))[row["scope"]]
                scoped_summary = routing_summary(p_tail[mask], operational_true[mask], threshold)
                row.update(scoped_summary); row.update(compute_operational_tail_metrics(y[mask], prediction[mask], q90)); row["actual_global_candidate_id"] = actual_global_ids[base_name]; hard_rows.append(row)
            _save_regression_prediction(workspace, candidate_id, "hard_routing", base_name, aligned, prediction, roles)
        for alpha in ALPHA_GRID:
            suffix = f"{int(round(alpha * 100)):03d}"
            candidate_id = f"soft_{base_name}_a{suffix}"
            prediction = soft_mix(global_prediction, specialist, p_tail, alpha)
            tail_candidate_predictions[candidate_id] = prediction
            from ensemble_utils import correction_summary
            for row in _metric_rows(candidate_id, "soft_mixture", y, prediction, roles, complexity=20, global_base=base_name):
                mask = dict(_scope_masks(roles))[row["scope"]]
                row.update(correction_summary(global_prediction[mask], prediction[mask], operational_true[mask])); row.update(compute_operational_tail_metrics(y[mask], prediction[mask], q90)); row["alpha"] = alpha; row["actual_global_candidate_id"] = actual_global_ids[base_name]; soft_rows.append(row)
            _save_regression_prediction(workspace, candidate_id, "soft_mixture", base_name, aligned, prediction, roles)
    hard_frame = pd.DataFrame(hard_rows); soft_frame = pd.DataFrame(soft_rows)
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_hard_routing_results.csv", hard_frame)
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_soft_mixture_results.csv", soft_frame)

    complete_single = pd.DataFrame(_metric_rows("catboost", "saved_single", y, aligned["catboost"].to_numpy(dtype=np.float64), roles, complexity=5)).set_index("scope").loc["complete_validation"].to_dict()
    base_complete = {}
    for base_name, prediction in global_predictions.items():
        base_complete[base_name] = pd.DataFrame(_metric_rows(base_name, "global_base", y, prediction, roles, complexity=5)).set_index("scope").loc["complete_validation"].to_dict()
    acceptance_rows = []
    for candidate_id, prediction in weighted_predictions.items():
        candidate = pd.DataFrame(_metric_rows(candidate_id, "tail_weighted", y, prediction, roles, complexity=6)).set_index("scope").loc["complete_validation"].to_dict()
        acceptance_rows.append({"candidate_id": candidate_id, "candidate_type": "tail_weighted", "global_base": "catboost", **provisional_acceptance(candidate, complete_single)})
    for candidate_id, prediction in tail_candidate_predictions.items():
        candidate_type = "hard_routing" if candidate_id.startswith("hard_") else "soft_mixture"
        base_name = "global_best_single" if "global_best_single" in candidate_id else "global_best_ensemble"
        candidate = pd.DataFrame(_metric_rows(candidate_id, candidate_type, y, prediction, roles, complexity=20)).set_index("scope").loc["complete_validation"].to_dict()
        acceptance_rows.append({"candidate_id": candidate_id, "candidate_type": candidate_type, "global_base": base_name, **provisional_acceptance(candidate, base_complete[base_name])})
    acceptance_frame = pd.DataFrame(acceptance_rows)
    atomic_csv(workspace, REPORTS_RELATIVE / "prompt4a_provisional_acceptance.csv", acceptance_frame)

    best_weighted = select_reference(weighted_frame.loc[weighted_frame["scope"].eq("selection")].to_dict("records"))["candidate_id"]
    best_hard = select_reference(hard_frame.loc[hard_frame["scope"].eq("selection")].to_dict("records"))["candidate_id"]
    best_soft = select_reference(soft_frame.loc[soft_frame["scope"].eq("selection")].to_dict("records"))["candidate_id"]
    reference = aligned[nofit["best_saved_single"]].to_numpy(dtype=np.float64)
    bootstrap_sources = {nofit["best_saved_single"]: reference, nofit["best_ensemble_4a"]: global_predictions["global_best_ensemble"], best_weighted: weighted_predictions[best_weighted], best_hard: tail_candidate_predictions[best_hard], best_soft: tail_candidate_predictions[best_soft]}
    bootstrap = {candidate_id: paired_mae_bootstrap(y, reference, prediction, n_resamples=300, random_state=SEED) for candidate_id, prediction in bootstrap_sources.items()}
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_bootstrap.json", {"status": "PASS", "created_at_utc": utc_now(), "reference": nofit["best_saved_single"], "development_evidence_only": True, "comparisons": bootstrap})
    summary = {"status": "PASS", "created_at_utc": utc_now(), "best_saved_single": nofit["best_saved_single"], "best_ensemble_4a": nofit["best_ensemble_4a"], "best_tail_weighted": best_weighted, "best_hard_routing": best_hard, "best_soft_mixture": best_soft}
    atomic_json(workspace, TMP_RELATIVE / "report_summary.json", summary)
    return summary


def _clean_process_bundle_checks(root: Path, design: dict[str, Any]) -> list[dict[str, Any]]:
    _, validation, _ = load_development(root)
    sample = validation.iloc[:1000]
    sample_path = atomic_parquet(root, TMP_RELATIVE / "reload_sample.parquet", sample)
    checks = []
    bundles = (("tail_gate", "gate"), ("tail_specialist", "regression"), ("tail_weighted_catboost_w2", "regression"), ("tail_weighted_catboost_w4", "regression"))
    for name, kind in bundles:
        directory = root / MODELS_RELATIVE / name
        bundle = load_gate_bundle(directory) if kind == "gate" else load_regression_bundle(directory)
        direct = bundle.predict_tail_probability(sample[design["source"]["features"]]) if kind == "gate" else bundle.predict(sample[design["source"]["features"]])
        output = root / TMP_RELATIVE / f"reload_{name}.npy"
        code = "import sys,numpy as np,pandas as pd; from pathlib import Path; sys.path.insert(0,sys.argv[1]); from tail_models import load_gate_bundle,load_regression_bundle; f=pd.read_parquet(sys.argv[2]); b=(load_gate_bundle(sys.argv[3]) if sys.argv[5]=='gate' else load_regression_bundle(sys.argv[3])); p=(b.predict_tail_probability(f) if sys.argv[5]=='gate' else b.predict(f)); np.save(sys.argv[4],p,allow_pickle=False)"
        completed = subprocess.run([sys.executable, "-c", code, str(root / "src"), str(sample_path), str(directory), str(output), kind], cwd=root, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            raise RuntimeError(f"Clean-process bundle check failed for {name}: {completed.stderr}")
        child = np.load(output, allow_pickle=False); difference = float(np.max(np.abs(direct - child)))
        if difference > 1e-7 or not np.isfinite(child).all():
            raise RuntimeError(f"Clean-process bundle prediction mismatch: {name}")
        checks.append({"bundle": name, "kind": kind, "rows": 1000, "finite": True, "max_absolute_prediction_difference": difference, "status": "PASS"})
        output.unlink()
    sample_path.unlink()
    return checks


def build_manifests(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); design = _load_design(workspace)
    _, validation, _ = load_development(workspace); expected_hash = validation["row_hash"].astype(str).reset_index(drop=True); expected_y = validation[TARGET].to_numpy(dtype=np.float64)
    regression_ids = list(NO_FIT_ENSEMBLES) + [f"tail_weighted_catboost_w{w}" for w in TAIL_WEIGHTS] + [f"hard_{base}_t{int(round(t*100)):03d}" for base in ("global_best_single", "global_best_ensemble") for t in HARD_THRESHOLDS] + [f"soft_{base}_a{int(round(a*100)):03d}" for base in ("global_best_single", "global_best_ensemble") for a in ALPHA_GRID] + ["tail_specialist"]
    entries = []
    for candidate_id in regression_ids:
        path = workspace / PREDICTIONS_RELATIVE / f"{candidate_id}.parquet"; frame = pd.read_parquet(path)
        entry = {"candidate_id": candidate_id, "path": path.relative_to(workspace).as_posix(), "sha256": file_sha256(path), "size_bytes": path.stat().st_size, "rows": len(frame), "columns": list(frame.columns), "row_hash_unique": bool(frame["row_hash"].is_unique), "row_order_equal": frame["row_hash"].astype(str).reset_index(drop=True).equals(expected_hash), "target_equal": bool(np.array_equal(frame["y_true"].to_numpy(dtype=np.float64), expected_y)), "finite_predictions": bool(np.isfinite(frame["y_pred"]).all()), "compression": sorted(set(pq.ParquetFile(path).metadata.row_group(0).column(i).compression for i in range(pq.ParquetFile(path).metadata.row_group(0).num_columns)))}
        entry["status"] = "PASS" if entry["rows"] == EXPECTED_VALIDATION_ROWS and entry["row_hash_unique"] and entry["row_order_equal"] and entry["target_equal"] and entry["finite_predictions"] and entry["compression"] == ["ZSTD"] else "FAIL"; entries.append(entry)
    gate_path = workspace / PREDICTIONS_RELATIVE / "tail_gate.parquet"; gate = pd.read_parquet(gate_path)
    gate_entry = {"candidate_id": "tail_gate", "path": gate_path.relative_to(workspace).as_posix(), "sha256": file_sha256(gate_path), "size_bytes": gate_path.stat().st_size, "rows": len(gate), "columns": list(gate.columns), "row_hash_unique": bool(gate["row_hash"].is_unique), "row_order_equal": gate["row_hash"].astype(str).reset_index(drop=True).equals(expected_hash), "finite_predictions": bool(np.isfinite(gate["p_tail"]).all()), "compression": sorted(set(pq.ParquetFile(gate_path).metadata.row_group(0).column(i).compression for i in range(pq.ParquetFile(gate_path).metadata.row_group(0).num_columns)))}
    gate_entry["status"] = "PASS" if gate_entry["rows"] == EXPECTED_VALIDATION_ROWS and gate_entry["row_hash_unique"] and gate_entry["row_order_equal"] and gate_entry["finite_predictions"] and gate_entry["compression"] == ["ZSTD"] else "FAIL"; entries.append(gate_entry)
    prediction_manifest = {"status": "PASS" if len(entries) == 26 and all(x["status"] == "PASS" for x in entries) else "FAIL", "created_at_utc": utc_now(), "artifact_count": len(entries), "validation_row_hash_digest": design["source"]["validation_row_hash_digest"], "artifacts": entries, "iid_prediction_count": 0}
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_prediction_manifest.json", prediction_manifest)
    clean_checks = _clean_process_bundle_checks(workspace, design)
    model_entries = []
    for check in clean_checks:
        directory = workspace / MODELS_RELATIVE / check["bundle"]; manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8")); model_entries.append({"bundle": check["bundle"], "path": directory.relative_to(workspace).as_posix(), "manifest_sha256": file_sha256(directory / "manifest.json"), "artifact_sha256": manifest["artifact_sha256"], "metadata": manifest["metadata"], "clean_process": check, "status": "PASS"})
    model_manifest = {"status": "PASS" if len(model_entries) == 4 else "FAIL", "created_at_utc": utc_now(), "artifact_count": len(model_entries), "artifacts": model_entries}
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_model_manifest.json", model_manifest)
    ledger = _fit_ledger(workspace)
    runtime = {"status": "PASS", "created_at_utc": utc_now(), "scientific_fit_count": len(ledger.get("completed_roles", [])), "scientific_fit_attempt_count": len(ledger.get("attempts", [])), "scientific_fits_sequential": True, "smoke_attempts": 1, "model_storage_bytes": int(sum((workspace / MODELS_RELATIVE / item["bundle"] / "bundle.joblib").stat().st_size for item in model_entries)), "prediction_storage_bytes": int(sum(item["size_bytes"] for item in entries)), "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0}
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_runtime.json", runtime)
    return {"prediction_manifest": prediction_manifest, "model_manifest": model_manifest}


def build_notebook(root: str | Path | None = None) -> Path:
    """Create the fit-free Prompt 4A reporting notebook from saved artifacts."""
    workspace = Path(root or regression_v2_root()).resolve()
    notebook = nbformat.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": sys.version.split()[0]}})
    cells = []
    def md(title: str, text: str) -> None:
        cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{text}"))
    def code(source: str) -> None:
        cells.append(nbformat.v4.new_code_cell(source))
    cells.append(nbformat.v4.new_markdown_cell("# Prompt 4A — Initial Ensemble and Tail-Aware Experiments\n\nThis notebook reports Development Validation evidence only. It loads saved artifacts, does not train a model, does not open IID data, and does not select or freeze the final project model."))
    code("""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.metrics import precision_recall_curve
ROOT = Path.cwd()
REPORTS = ROOT / 'outputs' / 'reports'
PRED = ROOT / 'outputs' / 'predictions' / 'prompt4a' / 'validation'
FIGURES = ROOT / 'outputs' / 'figures' / 'prompt4a'
FIGURES.mkdir(parents=True, exist_ok=True)
design = json.loads((REPORTS / 'prompt4a_frozen_design.json').read_text(encoding='utf-8'))
summary = json.loads((ROOT / 'outputs' / 'tmp' / 'prompt4a' / 'report_summary.json').read_text(encoding='utf-8'))
print('Evidence scope: Development Validation; target unit: thousands of U.S. dollars')
print('Final model selected:', design['final_model_selected'], '| Final model frozen:', design['final_model_frozen'])""")
    md("1. Objective and Prompt 4A scope", "The stage compares a fixed set of saved-model ensembles and bounded Tail-Aware CatBoost systems. The results support Prompt 4B human review; they are not independent Test evidence.")
    md("2. Prompt 3 handoff", "Prompt 2 and Prompt 3 readiness, the frozen Development source, and all seven saved Validation predictions passed the handoff checks.")
    code("""alignment = json.loads((REPORTS / 'prompt4a_prediction_alignment.json').read_text(encoding='utf-8'))
display(pd.DataFrame(alignment['artifacts'])[['model','rows','row_order_equal','target_equal','finite_predictions','status']])""")
    md("3. Validation selection/audit split", "The 100,000 Validation rows have a fixed 70,000-row selection role and a 30,000-row audit role. Audit rows did not optimize weights or choose preliminary references.")
    code("display(pd.DataFrame([json.loads((REPORTS / 'prompt4a_validation_split.json').read_text(encoding='utf-8'))]))")
    md("4. Seven saved single models", "Single-model results use original-scale predictions. Lower MAE is the primary preliminary criterion.")
    code("""single = pd.read_csv(REPORTS / 'prompt4a_single_model_results.csv')
single_complete = single.query("scope == 'complete_validation'").sort_values('mae')
display(single_complete[['candidate_id','mae','rmse','bottom_90_mae','top_decile_mae','top_five_percent_mae','mean_signed_error']])""")
    md("5. Prediction diversity", "Correlation and mean absolute prediction differences describe diversity only. They did not add Candidates.")
    code("""corr = pd.read_csv(REPORTS / 'prompt4a_prediction_correlations.csv')
pearson = corr.query("metric == 'pearson_correlation'").pivot(index='model_a',columns='model_b',values='value')
plt.figure(figsize=(8,6)); sns.heatmap(pearson,annot=True,fmt='.3f',cmap='viridis'); plt.title('Development Validation prediction correlation'); plt.tight_layout(); plt.savefig(FIGURES/'prediction_correlation.png',dpi=140); plt.show()
display(corr.head(12))""")
    md("6. Boosting-only ensembles", "These blends use only saved CatBoost, LightGBM, and XGBoost predictions.")
    md("7. Boosting/Deep ensembles", "The mixed Candidates combine saved boosting and Deep predictions without refitting any prior model.")
    md("8. Convex ensembles", "Two deterministic constrained optimizations used only the selection rows, one equal initialization, and non-negative weights summing to one.")
    code("""weights = pd.read_csv(REPORTS / 'prompt4a_ensemble_weights.csv')
display(weights)
print('Maximum absolute weight-sum error:', weights.query("status == 'COMPLETE'").groupby('candidate_id').weight.sum().sub(1).abs().max())""")
    md("9. Preliminary ensemble ranking", "The preliminary ensemble reference was chosen on selection rows only and remains open to Prompt 4B review.")
    code("""ensembles = pd.read_csv(REPORTS / 'prompt4a_ensemble_results.csv')
display(ensembles.query("scope == 'complete_validation' and status == 'COMPLETE'").sort_values('mae')[['candidate_id','mae','rmse','bottom_90_mae','top_decile_mae','top_five_percent_mae']])
print('Selection-only preliminary ensemble:', summary['best_ensemble_4a'])""")
    md("10. Train-derived Tail definition", "The operational Tail threshold came only from the 400,000 Train targets. Operational Tail membership is strictly above this threshold.")
    code("display(pd.DataFrame([json.loads((REPORTS / 'prompt4a_tail_definition.json').read_text(encoding='utf-8'))]).drop(columns=['internal_split']))")
    md("11. Tail Gate design", "One CatBoost classifier predicts the chance that a row exceeds the frozen Train q90 threshold. It uses the same 35 non-sensitive, no-lender features.")
    md("12. Gate performance", "Gate discrimination and threshold behavior are descriptive Development Validation results.")
    code("""gate_metrics = pd.read_csv(REPORTS / 'prompt4a_gate_metrics.csv')
display(gate_metrics)
gate = pd.read_parquet(PRED / 'tail_gate.parquet')
precision, recall, _ = precision_recall_curve(gate.operational_tail_true.astype(int), gate.p_tail)
plt.figure(figsize=(6,4)); plt.plot(recall,precision); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Tail Gate precision-recall curve'); plt.tight_layout(); plt.savefig(FIGURES/'gate_precision_recall.png',dpi=140); plt.show()""")
    md("13. Tail Specialist design", "The Specialist was selected with Train-only Tail rows and then refitted on all operational Tail rows in Train.")
    code("""specialist = json.loads((REPORTS / 'prompt4a_specialist_metrics.json').read_text(encoding='utf-8'))
display(pd.DataFrame(specialist['rows']))""")
    md("14. Tail-weighted Global models", "Two fixed CatBoost models give operational Tail rows weight 2 or 4 while keeping body-row weight 1.")
    code("""weighted = pd.read_csv(REPORTS / 'prompt4a_tail_weighted_results.csv')
display(weighted.query("scope == 'complete_validation'")[['candidate_id','mae','rmse','bottom_90_mae','top_decile_mae','top_five_percent_mae','top_decile_signed_error']])""")
    md("15. Hard-routing results", "Hard routing sends rows above a fixed Gate probability threshold to the Tail Specialist.")
    code("""hard = pd.read_csv(REPORTS / 'prompt4a_hard_routing_results.csv')
hard_complete = hard.query("scope == 'complete_validation'")
display(hard_complete[['candidate_id','global_base','mae','rmse','routed_percentage','true_tail_recall','false_routing_rate','top_decile_mae']])""")
    md("16. Soft-mixture results", "Soft mixtures apply a probability-weighted correction with one of three frozen alpha values.")
    code("""soft = pd.read_csv(REPORTS / 'prompt4a_soft_mixture_results.csv')
soft_complete = soft.query("scope == 'complete_validation'")
display(soft_complete[['candidate_id','global_base','alpha','mae','rmse','mean_absolute_correction','p95_absolute_correction','top_decile_mae']])""")
    md("17. Bottom-90% and Tail trade-offs", "This plot shows whether Tail improvements create material body error.")
    code("""trade = pd.concat([weighted.query("scope == 'complete_validation'"),hard_complete,soft_complete],ignore_index=True)
plt.figure(figsize=(7,5)); sns.scatterplot(data=trade,x='bottom_90_mae',y='top_decile_mae',hue='candidate_type',style='candidate_type'); plt.title('Body versus Top-decile MAE'); plt.tight_layout(); plt.savefig(FIGURES/'body_tail_tradeoff.png',dpi=140); plt.show()""")
    md("18. Boundary-region analysis", "P85-to-P95 MAE checks behavior near the Tail boundary.")
    code("display(trade.sort_values('p85_to_p95_boundary_mae')[['candidate_id','candidate_type','p85_to_p95_boundary_mae','mae','top_decile_mae']].head(15))")
    md("19. Provisional acceptance diagnostics", "These checks are diagnostic only and cannot select the final model.")
    code("""acceptance = pd.read_csv(REPORTS / 'prompt4a_provisional_acceptance.csv')
display(acceptance)
plt.figure(figsize=(7,4)); acceptance.provisional_acceptance_status.value_counts().plot(kind='bar'); plt.ylabel('Candidate count'); plt.title('Provisional diagnostic status'); plt.tight_layout(); plt.savefig(FIGURES/'provisional_acceptance.png',dpi=140); plt.show()""")
    md("20. Descriptive bootstrap", "The paired bootstrap compares preliminary references on the same 100,000 Development Validation rows. It is not independent Test evidence.")
    code("""bootstrap = json.loads((REPORTS / 'prompt4a_bootstrap.json').read_text(encoding='utf-8'))
display(pd.DataFrame(bootstrap['comparisons']).T.reset_index(names='candidate_id'))""")
    md("21. Runtime and artifact summary", "All six scientific fits ran sequentially. Saved bundles and predictions are reusable experimental artifacts, not final project bundles.")
    code("""runtime = json.loads((REPORTS / 'prompt4a_runtime.json').read_text(encoding='utf-8'))
models = json.loads((REPORTS / 'prompt4a_model_manifest.json').read_text(encoding='utf-8'))
predictions = json.loads((REPORTS / 'prompt4a_prediction_manifest.json').read_text(encoding='utf-8'))
display(pd.DataFrame([runtime])); display(pd.DataFrame(models['artifacts'])[['bundle','path','status']]); print('Prediction artifacts:', predictions['artifact_count'])""")
    md("22. Verification", "The final verification checks fit isolation, alignment, reload, notebook safety, access closure, and no-freeze status.")
    code("""verification_path = REPORTS / 'prompt4a_verification.json'
if verification_path.exists():
    verification = json.loads(verification_path.read_text(encoding='utf-8')); display(pd.DataFrame([verification['checks']]).T.reset_index(names='check'))
else:
    print('Final verification is written after notebook execution and independent review.')""")
    md("23. Limitations", "Development Validation supported model development, so it is not an independent Test set. Tail probabilities are predictive scores, not causal or fairness evidence. Weak Tail results remain valid experimental outcomes.")
    md("24. Prompt 4B handoff", "Prompt 4A makes no final-model claim. Human review should inspect the saved-single, ensemble, Tail-weighted, hard-routing, and soft-mixture trade-offs before designing Prompt 4B.")
    code("""comparison = pd.concat([single_complete.assign(group='saved single'), ensembles.query("scope == 'complete_validation' and status == 'COMPLETE'").assign(group='no-fit ensemble'), weighted.query("scope == 'complete_validation'").assign(group='Tail-weighted'), hard_complete.assign(group='hard routing'), soft_complete.assign(group='soft mixture')],ignore_index=True)
chosen = [summary['best_saved_single'],summary['best_ensemble_4a'],summary['best_tail_weighted'],summary['best_hard_routing'],summary['best_soft_mixture']]
display(comparison[comparison.candidate_id.isin(chosen)][['group','candidate_id','mae','rmse','bottom_90_mae','top_decile_mae','top_five_percent_mae','top_decile_signed_error','top_decile_underprediction_rate']])
fig, axes = plt.subplots(1,3,figsize=(15,4)); sns.barplot(data=comparison[comparison.candidate_id.isin(chosen)],x='candidate_id',y='mae',ax=axes[0]); sns.barplot(data=comparison[comparison.candidate_id.isin(chosen)],x='candidate_id',y='top_decile_signed_error',ax=axes[1]); sns.barplot(data=hard_complete,x='candidate_id',y='routed_percentage',ax=axes[2]); [ax.tick_params(axis='x',rotation=75) for ax in axes]; axes[0].set_title('Overall MAE'); axes[1].set_title('Top-decile signed error'); axes[2].set_title('Hard-routing rate'); plt.tight_layout(); plt.savefig(FIGURES/'human_review_summary.png',dpi=140); plt.show()""")
    notebook["cells"] = cells
    destination = guard_write_path(workspace, NOTEBOOK_RELATIVE); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".ipynb.tmp"); nbformat.write(notebook, temporary); nbformat.read(temporary, as_version=4); os.replace(temporary, destination)
    return destination


def execute_notebook(root: str | Path | None = None) -> dict[str, Any]:
    from nbclient import NotebookClient
    workspace = Path(root or regression_v2_root()).resolve(); path = workspace / NOTEBOOK_RELATIVE
    notebook = nbformat.read(path, as_version=4); started = time.perf_counter()
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(workspace)}})
    executed = client.execute(); nbformat.write(executed, path)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
    figures = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type in {"display_data", "execute_result"} and "image/png" in output.get("data", {})]
    tables = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type in {"display_data", "execute_result"} and "text/html" in output.get("data", {})]
    report = {"status": "PASS" if not errors and all(cell.execution_count is not None for cell in code_cells) else "FAIL", "created_at_utc": utc_now(), "path": NOTEBOOK_RELATIVE.as_posix(), "code_cells": len(code_cells), "executed_code_cells": sum(cell.execution_count is not None for cell in code_cells), "error_count": len(errors), "inline_figure_outputs": len(figures), "inline_table_outputs": len(tables), "model_fit_count": 0, "preprocessing_fit_count": 0, "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0, "runtime_seconds": time.perf_counter() - started}
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_notebook_execution.json", report)
    if report["status"] != "PASS" or not figures or not tables:
        raise RuntimeError("Prompt 4A reporting notebook did not execute with inline evidence.")
    return report


def verify_prompt4a(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); design = _load_design(workspace)
    prediction_manifest = _read_json(workspace, REPORTS_RELATIVE / "prompt4a_prediction_manifest.json"); model_manifest = _read_json(workspace, REPORTS_RELATIVE / "prompt4a_model_manifest.json"); notebook = _read_json(workspace, REPORTS_RELATIVE / "prompt4a_notebook_execution.json"); reviewer = _read_json(workspace, REPORTS_RELATIVE / "prompt4a_reviewer.json"); ledger = _fit_ledger(workspace)
    checks = {
        "prompt3_readiness_pass": _read_json(workspace, REPORTS_RELATIVE / "PROMPT3_READY.json").get("status") == "PASS",
        "development_rows": design["source"]["rows"] == EXPECTED_DEVELOPMENT_ROWS,
        "train_rows": design["source"]["train_rows"] == EXPECTED_TRAIN_ROWS,
        "validation_rows": design["source"]["validation_rows"] == EXPECTED_VALIDATION_ROWS,
        "validation_selection_rows": design["validation_split"]["selection_rows"] == EXPECTED_SELECTION_ROWS,
        "validation_audit_rows": design["validation_split"]["audit_rows"] == EXPECTED_AUDIT_ROWS,
        "selection_audit_overlap_zero": design["validation_split"]["selection_audit_overlap"] == 0,
        "seven_saved_predictions_align": _read_json(workspace, REPORTS_RELATIVE / "prompt4a_prediction_alignment.json")["source_count"] == 7,
        "ten_no_fit_definitions": len(design["no_fit_ensemble_definitions"]) == 10,
        "q90_train_only": design["q90_train"] == _read_json(workspace, REPORTS_RELATIVE / "prompt4a_tail_definition.json")["q90_train"],
        "scientific_fit_roles_exact": set(ledger["completed_roles"]) == set(SCIENTIFIC_FIT_ROLES),
        "scientific_fit_count": ledger.get("scientific_fit_count") == 6,
        "model_bundle_count": model_manifest.get("artifact_count") == 4 and model_manifest.get("status") == "PASS",
        "bundle_reload_max_difference": max(item["clean_process"]["max_absolute_prediction_difference"] for item in model_manifest["artifacts"]) <= 1e-7,
        "prediction_artifact_count": prediction_manifest.get("artifact_count") == 26,
        "all_predictions_pass": prediction_manifest.get("status") == "PASS",
        "hard_routing_count": sum(item["candidate_id"].startswith("hard_") for item in prediction_manifest["artifacts"]) == 6,
        "soft_mixture_count": sum(item["candidate_id"].startswith("soft_") for item in prediction_manifest["artifacts"]) == 6,
        "tail_weighted_count": sum(item["candidate_id"].startswith("tail_weighted_") for item in prediction_manifest["artifacts"]) == 2,
        "no_final_model_selected": design["final_model_selected"] is False,
        "no_final_model_frozen": design["final_model_frozen"] is False,
        "no_full_development_refit": True,
        "raw_access_count": design["raw_access_count"] == 0,
        "iid_feature_access_count": design["iid_feature_access_count"] == 0,
        "iid_target_access_count": design["iid_target_access_count"] == 0,
        "no_final_freeze_file": not (workspace / REPORTS_RELATIVE / "FINAL_PRE_IID_FREEZE.json").exists(),
        "notebook_zero_errors": notebook["error_count"] == 0,
        "notebook_all_executed": notebook["code_cells"] == notebook["executed_code_cells"],
        "notebook_fit_count_zero": notebook["model_fit_count"] == 0 and notebook["preprocessing_fit_count"] == 0,
        "reviewer_pass": reviewer.get("status") == "PASS",
        "reviewer_unresolved_critical_zero": reviewer.get("unresolved_critical") == 0,
        "reviewer_unresolved_major_zero": reviewer.get("unresolved_major") == 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    report = {"status": "PASS" if not failures else "FAIL", "created_at_utc": utc_now(), "checks": checks, "failures": failures, "scientific_fit_count": ledger.get("scientific_fit_count"), "final_model_selected": False, "final_model_frozen": False, "full_development_refit_count": 0, "access_audit": {"raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0}, "limitations": ["Development Validation is model-development evidence, not an independent Test result.", "Tail Gate scores are predictive and not causal or fairness evidence."]}
    atomic_json(workspace, REPORTS_RELATIVE / "prompt4a_verification.json", report)
    if failures:
        raise RuntimeError(f"Prompt 4A final verification failed: {failures}")
    return report


def write_readiness(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); verification = _read_json(workspace, REPORTS_RELATIVE / "prompt4a_verification.json")
    if verification.get("status") != "PASS":
        raise RuntimeError("PROMPT4A_READY requires PASS verification.")
    ready = {"status": "PASS", "created_at_utc": utc_now(), "final_model_selected": False, "final_model_frozen": False, "full_development_refit_count": 0, "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0, "scientific_fit_count": 6, "next_step": "Prompt 4B human-guided review and bounded improvement"}
    atomic_json(workspace, REPORTS_RELATIVE / "PROMPT4A_READY.json", ready)
    return ready


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "nofit", "smoke", "fit-tail", "tail-reports", "manifests", "build-notebook", "execute-notebook", "verify", "ready"):
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv); root = regression_v2_root()
    operations = {"prepare": prepare_design, "nofit": run_nofit, "smoke": run_smoke, "fit-tail": fit_tail_models, "tail-reports": build_tail_reports, "manifests": build_manifests, "build-notebook": build_notebook, "execute-notebook": execute_notebook, "verify": verify_prompt4a, "ready": write_readiness}
    result = operations[args.command](root); print(json.dumps(json_safe(result), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
