"""Prompt 4C final selection, fixed refit, packaging, and pre-IID freeze.

The command is recovery-safe.  It reads Development and frozen artifacts only.
It never reads either IID Parquet file and never reads the Raw data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# The Deep recovery environment reuses packages from the base Python install.
_BASE_SITE = Path(sys.base_prefix) / "Lib" / "site-packages"
if _BASE_SITE.is_dir() and str(_BASE_SITE) not in sys.path:
    sys.path.append(str(_BASE_SITE))

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

try:
    from .deep_preprocessing import duplicate_safe_deciles, ordered_digest
    from .model_bundles import ModelBundle
    from .prompt4_metrics import compute_operational_tail_metrics, compute_regression_metrics
    from .prompt4b_crossfit import (
        GLOBAL_FEATURE,
        MetaGateBundle,
        MetaPreprocessor,
        fixed_crossfit_parameters,
        membership_digest,
    )
    from .prompt4b_residual import ResidualSpecialistBundle, fit_residual_model, residual_target
    from .prompt4c_bundles import (
        FINAL_CORRECTION_ALPHA,
        FINAL_GATE_THRESHOLD,
        FinalGlobalBundle,
        FinalPrimaryBundle,
        development_target_cutpoints,
        duplicate_safe_target_deciles,
        load_final_bundle,
        mape_details,
        select_feature_frame,
        stage3_gate_strength,
        stage3_prediction,
        wape_percent,
    )
except ImportError:
    from deep_preprocessing import duplicate_safe_deciles, ordered_digest
    from model_bundles import ModelBundle
    from prompt4_metrics import compute_operational_tail_metrics, compute_regression_metrics
    from prompt4b_crossfit import GLOBAL_FEATURE, MetaGateBundle, MetaPreprocessor, fixed_crossfit_parameters, membership_digest
    from prompt4b_residual import ResidualSpecialistBundle, fit_residual_model, residual_target
    from prompt4c_bundles import (
        FINAL_CORRECTION_ALPHA,
        FINAL_GATE_THRESHOLD,
        FinalGlobalBundle,
        FinalPrimaryBundle,
        development_target_cutpoints,
        duplicate_safe_target_deciles,
        load_final_bundle,
        mape_details,
        select_feature_frame,
        stage3_gate_strength,
        stage3_prediction,
        wape_percent,
    )


AUTHORIZATION_ID = "regression_v2_prompt4c_final_selection_pre_iid_freeze"
DEVELOPMENT_SHA256 = "0ed232397be3ec4de1483c594954dce7b4704b375ca295397d899323dc4f0b6b"
PRIMARY_ID = "stage3_residual_t75_a75"
GLOBAL_ID = "ens_boost_cat060"
BLOCKA_ID = "nf_global2_oldraw_direct_cap25"
FEATURE_CONTRACT = "main_without_sensitive_without_lender"
TARGET = "loan_amount_000s"
TAIL_THRESHOLD = 438.0
META_ITERATIONS = 976
RESIDUAL_ITERATIONS = 791
SEED = 42
THREADS = 4
REPORTS = Path("outputs/reports")
TMP = Path("outputs/tmp/prompt4c")
PREDICTIONS = Path("outputs/predictions/prompt4c")
MODELS = Path("outputs/models/final_pre_iid")
FIGURES = Path("outputs/figures/prompt4c")
NOTEBOOK = Path("notebooks/04C_FINAL_SELECTION_AND_PRE_IID_FREEZE.ipynb")
FINAL_ROLES = (
    "prompt4c_oof_catboost_fold_a",
    "prompt4c_oof_lightgbm_fold_a",
    "prompt4c_oof_xgboost_fold_a",
    "prompt4c_oof_catboost_fold_b",
    "prompt4c_oof_lightgbm_fold_b",
    "prompt4c_oof_xgboost_fold_b",
    "prompt4c_full_catboost_500k",
    "prompt4c_full_lightgbm_500k",
    "prompt4c_full_xgboost_500k",
    "prompt4c_meta_gate_500k",
    "prompt4c_residual_specialist_500k",
)
META_PARAMETERS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": META_ITERATIONS,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10,
    "random_seed": SEED,
    "thread_count": THREADS,
    "verbose": False,
}
RESIDUAL_PARAMETERS = {
    "loss_function": "MAE",
    "iterations": RESIDUAL_ITERATIONS,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20,
    "random_strength": 1,
    "random_seed": SEED,
    "thread_count": THREADS,
    "verbose": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_path(root: str | Path | None = None) -> Path:
    return Path(root or Path(__file__).resolve().parents[1]).resolve()


def file_sha256(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()


def atomic_json(root: Path, relative: str | Path, payload: dict[str, Any]) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, destination)
    return destination


def atomic_csv(root: Path, relative: str | Path, frame: pd.DataFrame) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False)
    pd.read_csv(temporary)
    os.replace(temporary, destination)
    return destination


def atomic_parquet(root: Path, relative: str | Path, frame: pd.DataFrame) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    frame.to_parquet(temporary, index=False, compression="zstd")
    if len(pd.read_parquet(temporary, columns=[frame.columns[0]])) != len(frame):
        raise RuntimeError(f"Parquet reload failed: {relative}")
    os.replace(temporary, destination)
    return destination


def atomic_joblib(root: Path, relative: str | Path, value: Any) -> tuple[Path, str]:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    joblib.load(temporary)
    os.replace(temporary, destination)
    return destination, file_sha256(destination)


def read_json(root: Path, relative: str | Path) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "catboost", "lightgbm", "xgboost", "joblib")
    result = {"python": sys.version.split()[0]}
    for name in names:
        try:
            result[name.replace("-", "_")] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name.replace("-", "_")] = "NOT_INSTALLED"
    return result


def ensure_allowed_data_path(root: Path, path: Path) -> None:
    resolved = path.resolve()
    raw = (root / "data").resolve()
    iid_names = {"iid_holdout_features.parquet", "iid_holdout_targets.parquet"}
    if resolved == raw or raw in resolved.parents or resolved.name.lower() in iid_names:
        raise PermissionError(f"Prompt 4C prohibited data access: {resolved}")
    allowed = (root / "outputs/data/development.parquet").resolve()
    if (root / "outputs/data").resolve() in resolved.parents and resolved != allowed:
        raise PermissionError(f"Only Development may be opened in Prompt 4C: {resolved}")


def development_frame(root: Path, features: list[str], include_target: bool = True) -> pd.DataFrame:
    path = root / "outputs/data/development.parquet"
    ensure_allowed_data_path(root, path)
    columns = ["row_hash", "development_role"] + list(features)
    if include_target:
        columns.insert(2, TARGET)
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != 500_000 or not frame["row_hash"].is_unique:
        raise RuntimeError("Development membership is invalid.")
    return frame


def validation_frame(root: Path, features: list[str], include_target: bool = True) -> pd.DataFrame:
    development = development_frame(root, features, include_target=include_target)
    validation = development.loc[development["development_role"].astype(str).str.lower().eq("validation")].copy()
    validation.reset_index(drop=True, inplace=True)
    if len(validation) != 100_000:
        raise RuntimeError("Historical Validation membership is invalid.")
    return validation


def immutable_prompt4b4_snapshot(root: Path) -> dict[str, str]:
    files: list[Path] = []
    for relative in (Path("outputs/models/prompt4b4"), Path("outputs/predictions/prompt4b4"), Path("outputs/figures/prompt4b4")):
        directory = root / relative
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    files.extend((root / REPORTS).glob("prompt4b4*"))
    files.append(root / REPORTS / "PROMPT4B4_READY.json")
    notebook = root / "notebooks/04B4_IMBALANCE_AWARE_GLOBAL_TRAINING.ipynb"
    if notebook.exists():
        files.append(notebook)
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(set(files))
        if path.is_file()
    }


def validate_handoff(root: Path) -> dict[str, Any]:
    ready_names = (
        "PROMPT4B4_READY.json", "PROMPT4B3_READY.json", "PROMPT4B2_READY.json", "PROMPT4B_READY.json",
        "PROMPT4A_READY.json", "PROMPT3_READY.json", "PROMPT2_READY.json", "DATA_READY.json",
    )
    ready = {name: read_json(root, REPORTS / name) for name in ready_names}
    invalid = [name for name, payload in ready.items() if not str(payload.get("status", "")).startswith("PASS")]
    reviewer = read_json(root, REPORTS / "prompt4b4_reviewer.json")
    verification = read_json(root, REPORTS / "prompt4b4_verification.json")
    if invalid or reviewer.get("status") != "PASS" or verification.get("status") != "PASS":
        raise RuntimeError(f"Prompt 4B4/prior handoff is invalid: {invalid}")
    development = root / "outputs/data/development.parquet"
    ensure_allowed_data_path(root, development)
    digest = file_sha256(development)
    metadata = pq.ParquetFile(development).metadata
    feature_roles = read_json(root, REPORTS / "feature_roles.json")
    features = list(feature_roles["contracts"][FEATURE_CONTRACT])
    if digest != DEVELOPMENT_SHA256 or metadata.num_rows != 500_000 or len(features) != 35:
        raise RuntimeError("Development identity or feature contract changed.")
    if any(name in features for name in ("respondent_id", TARGET, "row_hash")):
        raise RuntimeError("The final feature contract contains a prohibited field.")
    prior = immutable_prompt4b4_snapshot(root)
    payload = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "development_sha256": digest,
        "development_rows": int(metadata.num_rows),
        "feature_contract": FEATURE_CONTRACT,
        "feature_count": len(features),
        "features": features,
        "readiness": {name: value.get("status") for name, value in ready.items()},
        "prompt4b4_reviewer": reviewer.get("status"),
        "prompt4b4_verification": verification.get("status"),
        "prompt4b4_immutable_snapshot": prior,
        "prompt4b4_immutable_file_count": len(prior),
        "development_research_closed": True,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "iid_prediction_count": 0,
        "full_development_final_refit_count": 0,
        "prompt5_executed": False,
    }
    atomic_json(root, REPORTS / "prompt4c_handoff_validation.json", payload)
    return payload


def recover_recipe(root: Path, features: list[str]) -> dict[str, Any]:
    stage2 = read_json(root, REPORTS / "prompt4b_stage2_report.json")
    stage3 = read_json(root, REPORTS / "prompt4b_stage3_report.json")
    tail = read_json(root, REPORTS / "prompt4a_tail_definition.json")
    meta = read_json(root, "outputs/models/prompt4b/meta_gate/full/manifest.json")
    residual = read_json(root, "outputs/models/prompt4b/residual_specialist/full/manifest.json")
    base_manifests = {
        family: read_json(root, f"outputs/models/prompt4b/crossfit/crossfit_{family}_fold_a/manifest.json")
        for family in ("catboost", "lightgbm", "xgboost")
    }
    exact = (
        stage2.get("meta_gate_selection_iteration") == META_ITERATIONS
        and stage3.get("residual_selection_iteration") == RESIDUAL_ITERATIONS
        and tail.get("q90_train") == TAIL_THRESHOLD
        and meta.get("feature_contract") == features + [GLOBAL_FEATURE]
        and residual.get("feature_contract") == features + [GLOBAL_FEATURE]
        and residual.get("target_definition") == "loan_amount_000s - leakage-safe global_oof_prediction"
        and residual.get("tail_only") is True
    )
    if not exact:
        raise RuntimeError("BLOCKED_STAGE3_RECIPE_AMBIGUOUS")
    return {
        "status": "RECOVERED_EXACTLY",
        "primary_recipe_id": PRIMARY_ID,
        "global_components": base_manifests,
        "global_weights": {"catboost": 0.60, "lightgbm": 0.20, "xgboost": 0.20},
        "feature_contract": features,
        "target_transforms": {family: value["target_mode"] for family, value in base_manifests.items()},
        "tail_threshold": TAIL_THRESHOLD,
        "tail_rule": "loan_amount_000s > 438.0 for training membership only",
        "meta_gate_configuration": meta["model_configuration"],
        "meta_gate_iteration": META_ITERATIONS,
        "meta_gate_features": features + [GLOBAL_FEATURE],
        "meta_gate_target": "1[loan_amount_000s > 438.0]",
        "residual_configuration": residual["model_configuration"],
        "residual_iteration": RESIDUAL_ITERATIONS,
        "residual_features": features + [GLOBAL_FEATURE],
        "residual_target": "loan_amount_000s - leakage-safe global_oof_prediction",
        "residual_training_population": "strict operational Tail: loan_amount_000s > 438.0",
        "routing_threshold": FINAL_GATE_THRESHOLD,
        "routing_comparison": "strict probability > 0.75",
        "correction_alpha": FINAL_CORRECTION_ALPHA,
        "routing_strength_formula": "0 if p<=0.75 else 0.75*(p-0.75)/(1-0.75)",
        "cap": None,
        "clip": None,
        "positivity_rule": None,
        "sign_rule": "preserve the fitted residual sign",
        "final_prediction_formula": "global_prediction + routing_strength * predicted_residual",
        "historical_sources": {
            "stage3_report": "outputs/reports/prompt4b_stage3_report.json",
            "stage2_report": "outputs/reports/prompt4b_stage2_report.json",
            "stage2_crossfit": "outputs/reports/prompt4b_stage2_crossfit.json",
            "tail_definition": "outputs/reports/prompt4a_tail_definition.json",
            "model_manifest": "outputs/reports/prompt4b_model_manifest.json",
            "prediction_manifest": "outputs/reports/prompt4b_prediction_manifest.json",
            "source_modules": [
                "src/prompt4b_experiments.py", "src/prompt4b_crossfit.py",
                "src/prompt4b_residual.py", "src/prompt4b_calibration.py",
            ],
        },
    }


def historical_reproduction(root: Path, features: list[str], recipe: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sources = {
        "catboost": pd.read_parquet(root / "outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet"),
        "lightgbm": pd.read_parquet(root / "outputs/predictions/prompt2/validation/selected_lightgbm_without_lender.parquet"),
        "xgboost": pd.read_parquet(root / "outputs/predictions/prompt2/validation/selected_xgboost_without_lender.parquet"),
    }
    saved_global = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet")
    saved_meta = pd.read_parquet(root / "outputs/predictions/prompt4b/validation/stage2_meta_gate_probability.parquet")
    saved_residual = pd.read_parquet(root / "outputs/predictions/prompt4b/validation/stage3_residual_prediction.parquet")
    saved_final = pd.read_parquet(root / "outputs/predictions/prompt4b/validation/stage3_residual_t75_a75.parquet")
    ordered = saved_global["row_hash"].astype(str)
    if not all(frame["row_hash"].astype(str).equals(ordered) for frame in [*sources.values(), saved_meta, saved_residual, saved_final]):
        raise RuntimeError("Historical prediction order changed.")
    global_prediction = (
        0.60 * sources["catboost"]["y_pred"].to_numpy(float)
        + 0.20 * sources["lightgbm"]["y_pred"].to_numpy(float)
        + 0.20 * sources["xgboost"]["y_pred"].to_numpy(float)
    )
    global_difference = float(np.max(np.abs(global_prediction - saved_global["y_pred"].to_numpy(float))))
    global_report = {
        "status": "PASS" if global_difference == 0.0 else "BLOCKED_GLOBAL_HISTORICAL_REPRODUCTION",
        "created_at_utc": utc_now(),
        "candidate_id": GLOBAL_ID,
        "rows": len(saved_global),
        "formula": "0.60*CatBoost + 0.20*LightGBM + 0.20*XGBoost",
        "maximum_absolute_prediction_difference": global_difference,
        "required_tolerance": 0.0,
        "row_hash_order_digest": ordered_digest(ordered),
        "component_prediction_paths": {
            name: f"outputs/predictions/prompt2/validation/selected_{name}_without_lender.parquet"
            for name in sources
        },
        "saved_prediction_path": "outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet",
    }
    atomic_json(root, REPORTS / "prompt4c_global_recipe_reproduction.json", global_report)
    if global_difference != 0.0:
        raise RuntimeError("BLOCKED_GLOBAL_HISTORICAL_REPRODUCTION")
    validation = validation_frame(root, features, include_target=True)
    if not validation["row_hash"].astype(str).equals(ordered):
        raise RuntimeError("Historical Validation features are not aligned.")
    meta_frame = validation[features].copy()
    meta_frame[GLOBAL_FEATURE] = global_prediction
    meta_bundle = joblib.load(root / "outputs/models/prompt4b/meta_gate/full/bundle.joblib")
    residual_bundle = joblib.load(root / "outputs/models/prompt4b/residual_specialist/full/bundle.joblib")
    probability = meta_bundle.predict_tail_probability(meta_frame)
    residual_prediction = residual_bundle.predict(meta_frame)
    reconstructed = stage3_prediction(global_prediction, residual_prediction, probability)
    differences = {
        "global": global_difference,
        "meta_gate_probability": float(np.max(np.abs(probability - saved_meta["p_meta_gate"].to_numpy(float)))),
        "residual_proposal": float(np.max(np.abs(residual_prediction - saved_residual["predicted_residual"].to_numpy(float)))),
        "final_prediction": float(np.max(np.abs(reconstructed - saved_final["y_pred"].to_numpy(float)))),
    }
    y = saved_final["y_true"].to_numpy(float)
    metrics = {**compute_regression_metrics(y, reconstructed), **compute_operational_tail_metrics(y, reconstructed, TAIL_THRESHOLD)}
    stage3_report = {
        "status": "PASS" if all(value == 0.0 for value in differences.values()) else "BLOCKED_STAGE3_HISTORICAL_REPRODUCTION",
        "created_at_utc": utc_now(),
        "candidate_id": PRIMARY_ID,
        "rows": len(saved_final),
        "row_hash_order_digest": ordered_digest(ordered),
        "component_maximum_absolute_differences": differences,
        "required_tolerance": 0.0,
        "recipe": recipe,
        "reproduced_metrics": metrics,
        "saved_prediction_path": "outputs/predictions/prompt4b/validation/stage3_residual_t75_a75.parquet",
    }
    atomic_json(root, REPORTS / "prompt4c_stage3_recipe_reproduction.json", stage3_report)
    if stage3_report["status"] != "PASS":
        raise RuntimeError("BLOCKED_STAGE3_HISTORICAL_REPRODUCTION")
    return global_report, stage3_report


def blocka_worker(root: Path, output: Path) -> None:
    from deep_bundles import load_bundle as load_deep_bundle
    from prompt4b2_metrics import direct_routing, gate_strength
    from tail_models import load_gate_bundle, load_regression_bundle

    design = read_json(root, REPORTS / "prompt4b2_frozen_design.json")
    features = list(design["features"])
    validation = validation_frame(root, features, include_target=False)
    base = {
        "catboost": joblib.load(root / "outputs/models/prompt2/selected_catboost_without_lender.joblib"),
        "lightgbm": joblib.load(root / "outputs/models/prompt2/selected_lightgbm_without_lender.joblib"),
        "xgboost": joblib.load(root / "outputs/models/prompt2/selected_xgboost_without_lender.joblib"),
        "realmlp": load_deep_bundle(root / "outputs/models/prompt3/selected_realmlp"),
        "fttransformer": load_deep_bundle(root / "outputs/models/prompt3/selected_fttransformer"),
    }
    paths = {
        "catboost": "outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet",
        "lightgbm": "outputs/predictions/prompt2/validation/selected_lightgbm_without_lender.parquet",
        "xgboost": "outputs/predictions/prompt2/validation/selected_xgboost_without_lender.parquet",
        "realmlp": "outputs/predictions/prompt3/validation/selected_realmlp.parquet",
        "fttransformer": "outputs/predictions/prompt3/validation/selected_fttransformer.parquet",
    }
    predictions: dict[str, np.ndarray] = {}
    component_differences: dict[str, float] = {}
    for name, bundle in base.items():
        predictions[name] = bundle.predict(validation[features])
        saved = pd.read_parquet(root / paths[name])["y_pred"].to_numpy(float)
        component_differences[name] = float(np.max(np.abs(predictions[name] - saved)))
    weights = {
        "catboost": 0.4925597328873232,
        "lightgbm": 0.13548668701796474,
        "xgboost": 0.2487078679591308,
        "realmlp": 4.246793116935918e-18,
        "fttransformer": 0.12324571213558137,
    }
    global2 = sum(weights[name] * predictions[name] for name in weights)
    saved_global2 = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/ens_convex_boosting_deep.parquet")["y_pred"].to_numpy(float)
    gate = load_gate_bundle(root / "outputs/models/prompt4a/tail_gate")
    specialist = load_regression_bundle(root / "outputs/models/prompt4a/tail_specialist")
    probability = gate.predict_tail_probability(validation[features])
    direct = specialist.predict(validation[features])
    saved_probability = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/tail_gate.parquet")["p_tail"].to_numpy(float)
    saved_direct = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/tail_specialist.parquet")["y_pred"].to_numpy(float)
    prediction, _ = direct_routing(global2, direct, gate_strength(probability, 0.85, 0.50, 1.0), 109.5)
    saved = pd.read_parquet(root / "outputs/predictions/prompt4b2/validation/nf_global2_oldraw_direct_cap25.parquet")["y_pred"].to_numpy(float)
    payload = {
        "status": "PASS" if float(np.max(np.abs(prediction - saved))) == 0.0 else "NONEXACT",
        "rows": len(prediction),
        "component_maximum_absolute_differences": component_differences,
        "global2_maximum_absolute_difference": float(np.max(np.abs(global2 - saved_global2))),
        "gate_maximum_absolute_difference": float(np.max(np.abs(probability - saved_probability))),
        "direct_specialist_maximum_absolute_difference": float(np.max(np.abs(direct - saved_direct))),
        "blocka_maximum_absolute_difference": float(np.max(np.abs(prediction - saved))),
        "finite": bool(np.isfinite(prediction).all()),
    }
    atomic_json(root, output.relative_to(root), payload)


def blocka_eligibility(root: Path) -> dict[str, Any]:
    from prompt4b2_metrics import direct_routing, gate_strength

    g2 = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/ens_convex_boosting_deep.parquet")
    probability = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/tail_gate.parquet")
    direct = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/tail_specialist.parquet")
    saved = pd.read_parquet(root / "outputs/predictions/prompt4b2/validation/nf_global2_oldraw_direct_cap25.parquet")
    if not all(frame["row_hash"].astype(str).equals(saved["row_hash"].astype(str)) for frame in (g2, probability, direct)):
        raise RuntimeError("Block A dependencies are not aligned.")
    reconstructed, correction = direct_routing(
        g2["y_pred"].to_numpy(float),
        direct["y_pred"].to_numpy(float),
        gate_strength(probability["p_tail"].to_numpy(float), 0.85, 0.50, 1.0),
        109.5,
    )
    formula_difference = float(np.max(np.abs(reconstructed - saved["y_pred"].to_numpy(float))))
    worker_output = root / TMP / "blocka_clean_inference_worker.json"
    deep_python = root.parent / "artifacts/environment/stage5_env/Scripts/python.exe"
    if not worker_output.exists():
        if not deep_python.exists():
            worker = {"status": "MISSING_DEEP_ENVIRONMENT"}
        else:
            command = [str(deep_python), str(Path(__file__).resolve()), "blocka-worker", "--root", str(root), "--output", str(worker_output)]
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=600)
            if completed.returncode != 0 or not worker_output.exists():
                worker = {"status": "WORKER_FAILURE", "returncode": completed.returncode, "stderr_tail": completed.stderr[-4000:]}
            else:
                worker = json.loads(worker_output.read_text(encoding="utf-8"))
    else:
        worker = json.loads(worker_output.read_text(encoding="utf-8"))
    exact_clean = worker.get("status") == "PASS" and worker.get("blocka_maximum_absolute_difference") == 0.0
    eligible = formula_difference == 0.0 and exact_clean
    report = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "candidate_id": BLOCKA_ID,
        "formula": "G2 + clip(s_old*(direct_specialist-G2), -109.5, +109.5)",
        "gate_strength": "0.50*max(p_old_raw-0.85,0)/(1-0.85)",
        "global2_weights": {
            "catboost": 0.4925597328873232,
            "lightgbm": 0.13548668701796474,
            "xgboost": 0.2487078679591308,
            "realmlp": 4.246793116935918e-18,
            "fttransformer": 0.12324571213558137,
        },
        "saved_component_formula_maximum_difference": formula_difference,
        "saved_component_formula_exact": formula_difference == 0.0,
        "correction_min": float(np.min(correction)),
        "correction_max": float(np.max(correction)),
        "clean_process_inference": worker,
        "historical_blocka_iid_eligible": eligible,
        "eligibility_reason": (
            "Exact zero-fit formula and clean-process inference both reproduce Validation."
            if eligible else
            "Excluded because the available frozen inference dependencies do not reproduce the immutable Validation prediction exactly; no refit or relaxed tolerance is authorized."
        ),
        "training_population": "historical 400,000 Train rows",
        "new_fit_count": 0,
        "bundle_created": False,
        "not_eligible_to_replace_primary_after_iid": True,
    }
    atomic_json(root, REPORTS / "prompt4c_blocka_package_eligibility.json", report)
    return report


def _complete_metric(root: Path, csv_path: str, candidate_id: str) -> dict[str, Any]:
    frame = pd.read_csv(root / csv_path)
    row = frame.loc[(frame["candidate_id"] == candidate_id) & (frame["scope"] == "complete_validation")]
    if len(row) != 1:
        raise RuntimeError(f"Historical metric row is ambiguous: {candidate_id}")
    return row.iloc[0].to_dict()


def final_refit_graph(features: list[str]) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    for fold in ("fold_a", "fold_b"):
        for family in ("catboost", "lightgbm", "xgboost"):
            roles.append({
                "role": f"prompt4c_oof_{family}_{fold}",
                "role_type": "OOF Global refit",
                "family": family,
                "training_population": "the deterministic 250,000-row fit fold",
                "prediction_population": "the opposite deterministic 250,000-row fold",
                "features": features,
                "target": TARGET,
                "source_recipe": f"Prompt 2 {family} representative used by ens_boost_cat060",
            })
    for family in ("catboost", "lightgbm", "xgboost"):
        roles.append({
            "role": f"prompt4c_full_{family}_500k",
            "role_type": "final full-Development Global",
            "family": family,
            "training_population": "all 500,000 Development rows",
            "features": features,
            "target": TARGET,
            "source_recipe": f"Prompt 2 {family} representative used by ens_boost_cat060",
        })
    roles.append({
        "role": "prompt4c_meta_gate_500k",
        "role_type": "final Meta-Gate",
        "family": "catboost_classifier",
        "training_population": "all 500,000 Development rows",
        "features": features + [GLOBAL_FEATURE],
        "target": "1[loan_amount_000s > 438.0]",
        "global_feature_source": "leakage-safe OOF Global prediction",
        "fixed_iterations": META_ITERATIONS,
    })
    roles.append({
        "role": "prompt4c_residual_specialist_500k",
        "role_type": "final Residual Specialist",
        "family": "catboost_regressor",
        "training_population": "all Development rows with loan_amount_000s > 438.0",
        "features": features + [GLOBAL_FEATURE],
        "target": "loan_amount_000s - leakage-safe OOF Global prediction",
        "fixed_iterations": RESIDUAL_ITERATIONS,
    })
    if tuple(role["role"] for role in roles) != FINAL_ROLES:
        raise RuntimeError("BLOCKED_FINAL_REFIT_INTEGRITY")
    return roles


def prompt5_protocol(blocka_eligible: bool) -> dict[str, Any]:
    permitted = [
        {"model_id": "final_primary_stage3_500k", "role": "Primary", "training_rows": 500_000},
        {"model_id": "final_global_500k", "role": "simple comparator", "training_rows": 500_000},
    ]
    if blocka_eligible:
        permitted.append({
            "model_id": "historical_blocka_400k",
            "role": "historical MAE research comparator",
            "training_rows": 400_000,
            "not_eligible_to_replace_primary": True,
        })
    return {
        "status": "FROZEN",
        "primary_iid_metric": "MAE",
        "permitted_models": permitted,
        "excluded_models": [
            "prompt4b3 champion", "prompt4b4 LDS", "prompt4b4 DenseWeight", "q60",
            "Stage 1", "Stage 2", "deep models", "additional ensembles",
        ],
        "overall_metrics": [
            "MAE", "RMSE", "R2", "RMSLE", "median absolute error", "P90 absolute error",
            "mean signed error", "MAPE%", "WAPE%", "Bottom-90 MAE", "Top-decile MAE",
            "Top-5% MAE", "P85-P95 boundary MAE", "Top-decile signed error",
            "Top-5% signed error", "Top-decile underprediction rate", "Top-5% underprediction rate",
        ],
        "mape_definition": "100*mean(abs(y-pred)/y) on strictly positive targets; no epsilon",
        "mape_nonpositive_rule": "report invalid count and valid coverage; calculate on positive rows only",
        "wape_definition": "100*sum(abs(y-pred))/sum(y)",
        "iid_local_deciles": "duplicate-safe qcut D1-D10 from true IID target after the one authorized target opening",
        "development_frozen_bands": "apply frozen complete-Development q10-q90 values to IID targets",
        "decile_metrics": [
            "n", "target min", "target max", "mean target", "median target", "mean prediction",
            "median prediction", "MAE", "MAPE%", "WAPE%", "RMSE", "signed error", "underprediction rate",
        ],
        "six_condition_rubric": {
            "C1": "overall MAE improves",
            "C2": "Top-decile MAE improves by at least 3%",
            "C3": "Bottom-90 MAE worsens by no more than 0.25%",
            "C4": "RMSE worsens by no more than 0.25%",
            "C5": "Top-decile signed error moves closer to zero",
            "C6": "Top-decile underprediction rate decreases",
        },
        "bootstrap": {
            "type": "paired row bootstrap",
            "resamples": 500,
            "seed": 42,
            "same_rows_within_each_pair": True,
            "metrics": ["MAE", "RMSE", "MAPE", "WAPE", "Bottom-90 MAE", "Top-decile MAE", "Top-5% MAE"],
            "primary_comparison": "final Primary minus final Global",
        },
        "iid_access_sequence": [
            "validate FINAL_PRE_IID_FREEZE.json",
            "validate final bundle hashes",
            "open IID features",
            "generate and save predictions for the permitted model set only",
            "open IID targets exactly once",
            "align by row_hash",
            "calculate frozen overall metrics",
            "calculate frozen decile and band analyses",
            "calculate frozen paired bootstrap uncertainty",
            "create final evaluation reports",
            "never refit",
        ],
        "after_first_iid_access_prohibitions": [
            "hyperparameter change", "model refit", "model replacement", "feature change",
            "threshold change", "alpha change", "cap change", "ensemble-weight change",
            "comparator addition", "metric-driven reselection",
        ],
        "primary_never_changes_after_iid": True,
    }


def create_selection_freeze(
    root: Path,
    handoff: dict[str, Any],
    recipe: dict[str, Any],
    blocka: dict[str, Any],
) -> dict[str, Any]:
    path = root / REPORTS / "prompt4c_final_selection_freeze.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "FROZEN"
            or existing.get("final_primary_recipe") != PRIMARY_ID
            or existing.get("development_research_closed") is not True
        ):
            raise RuntimeError("Existing Prompt 4C selection freeze is invalid.")
        return existing
    stage3 = _complete_metric(root, "outputs/reports/prompt4b_stage3_candidates.csv", PRIMARY_ID)
    blocka_metric = read_json(root, REPORTS / "prompt4b2_blockA_report.json")["champion_complete_validation"]
    global_metric = _complete_metric(root, "outputs/reports/prompt4a_ensemble_results.csv", GLOBAL_ID)
    protocol = prompt5_protocol(bool(blocka["historical_blocka_iid_eligible"]))
    graph = final_refit_graph(handoff["features"])
    payload = {
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "human_authorization_id": AUTHORIZATION_ID,
        "development_research_closed": True,
        "selection_frozen_before_prompt4c_percentage_metrics": True,
        "selection_philosophy": "Balanced conservative Tail-aware performance first; overall MAE second; reproducibility and methodological cleanliness third.",
        "primary_metric": "MAE",
        "final_primary_recipe": PRIMARY_ID,
        "historical_mae_challenger": BLOCKA_ID,
        "simple_comparator": GLOBAL_ID,
        "historical_metrics": {
            PRIMARY_ID: stage3,
            BLOCKA_ID: blocka_metric,
            GLOBAL_ID: global_metric,
        },
        "historical_paired_bootstrap_blocka_minus_stage3": {
            "mean_mae_difference": -0.0823,
            "percentile_2_5": -0.1741,
            "percentile_97_5": 0.0136,
            "interval_includes_zero": True,
        },
        "selection_rationale": [
            "Stage 3 MAE is effectively close to the lowest observed Development MAE.",
            "Stage 3 has materially better RMSE than Block A.",
            "Stage 3 preserves Bottom-90 performance better than Block A.",
            "The residual correction is methodologically cleaner than the direct-target alternative.",
            "The Block A MAE difference is inside the frozen adaptive Development noise interval.",
            "Stage 3 gives the stronger conservative Body and Tail compromise.",
        ],
        "mape_wape_do_not_affect_selection": True,
        "future_iid_cannot_change_primary": True,
        "feature_contract_name": FEATURE_CONTRACT,
        "feature_count": 35,
        "features": handoff["features"],
        "recipe": recipe,
        "reconstruction_source_hashes": {
            path: file_sha256(root / path) for path in recipe["historical_sources"]["source_modules"]
        },
        "final_refit_dependency_graph": graph,
        "final_refit_role_count": len(graph),
        "final_refit_scientific_candidate_count": 0,
        "oof_rule": {
            "folds": 2,
            "population_rows": 500_000,
            "fold_rows": 250_000,
            "algorithm": "train_test_split over ordered Development indices with train_size=test_size=250000, random_state=42, shuffle=True",
            "stratification": "duplicate_safe_deciles(y) combined with strict y>438.0 status",
            "sort_each_resulting_index_array": True,
            "one_oof_prediction_per_row": True,
            "zero_self_fit": True,
        },
        "full_development_fit_budget": {"fixed_roles": 11, "technical_retry_per_exact_role": 1, "scientific_search_fits": 0},
        "historical_blocka_iid_eligible": bool(blocka["historical_blocka_iid_eligible"]),
        "prompt5_protocol": protocol,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "iid_prediction_count": 0,
        "prompt5_executed": False,
    }
    atomic_json(root, REPORTS / "prompt4c_final_selection_freeze.json", payload)
    reloaded = read_json(root, REPORTS / "prompt4c_final_selection_freeze.json")
    if reloaded["final_primary_recipe"] != PRIMARY_ID or reloaded["final_refit_role_count"] != 11:
        raise RuntimeError("Final selection freeze reload failed.")
    return reloaded


def historical_prediction_sources(root: Path) -> dict[str, Path]:
    return {
        GLOBAL_ID: root / "outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet",
        PRIMARY_ID: root / "outputs/predictions/prompt4b/validation/stage3_residual_t75_a75.parquet",
        BLOCKA_ID: root / "outputs/predictions/prompt4b2/validation/nf_global2_oldraw_direct_cap25.parquet",
        "prompt4b3__beat_costaware": root / "outputs/predictions/prompt4b3/validation/prompt4b3__beat_costaware.parquet",
        "prompt4b4__sub_ldslgb": root / "outputs/predictions/prompt4b4/validation/prompt4b4__sub_ldslgb.parquet",
        "prompt4b4__sub_densecat": root / "outputs/predictions/prompt4b4/validation/prompt4b4__sub_densecat.parquet",
    }


def historical_metric_extension(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if not (root / REPORTS / "prompt4c_final_selection_freeze.json").exists():
        raise RuntimeError("Selection must be frozen before MAPE/WAPE reporting.")
    frames = {name: pd.read_parquet(path) for name, path in historical_prediction_sources(root).items()}
    ordered = frames[PRIMARY_ID]["row_hash"].astype(str)
    if not all(frame["row_hash"].astype(str).equals(ordered) for frame in frames.values()):
        raise RuntimeError("Historical reporting predictions are not aligned.")
    y = frames[PRIMARY_ID]["y_true"].to_numpy(float)
    if np.any(y <= 0.0):
        raise RuntimeError("Historical Development MAPE requires all targets to be positive.")
    rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    labels = duplicate_safe_target_deciles(y)
    for model_id, frame in frames.items():
        prediction = frame["y_pred"].to_numpy(float)
        rows.append({
            "model_id": model_id,
            "role": (
                "final Primary" if model_id == PRIMARY_ID else
                "simple comparator" if model_id == GLOBAL_ID else
                "historical MAE challenger" if model_id == BLOCKA_ID else
                "Development-only diagnostic"
            ),
            **compute_regression_metrics(y, prediction),
            **compute_operational_tail_metrics(y, prediction, TAIL_THRESHOLD),
            **mape_details(y, prediction),
            "wape_percent": wape_percent(y, prediction),
            "selection_effect": "none; reporting only",
        })
        error = prediction - y
        for label in range(10):
            mask = labels == label
            target = y[mask]
            pred = prediction[mask]
            decile_rows.append({
                "model_id": model_id,
                "decile": f"D{label + 1}",
                "decile_index": label + 1,
                "row_count": int(np.count_nonzero(mask)),
                "target_min": float(np.min(target)),
                "target_max": float(np.max(target)),
                "mean_target": float(np.mean(target)),
                "median_target": float(np.median(target)),
                "mean_prediction": float(np.mean(pred)),
                "median_prediction": float(np.median(pred)),
                "mae": float(np.mean(np.abs(pred - target))),
                "mape_percent": mape_details(target, pred)["mape_percent"],
                "wape_percent": wape_percent(target, pred),
                "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
                "mean_signed_error": float(np.mean(pred - target)),
                "underprediction_rate": float(np.mean(pred < target)),
            })
    metrics = pd.DataFrame(rows)
    deciles = pd.DataFrame(decile_rows)
    atomic_csv(root, REPORTS / "prompt4c_historical_metric_extension.csv", metrics)
    atomic_csv(root, REPORTS / "prompt4c_historical_decile_metrics.csv", deciles)
    development = development_frame(root, [], include_target=True)
    cutpoints = development_target_cutpoints(development[TARGET])
    return metrics, deciles, cutpoints


def build_figures(root: Path, metrics: pd.DataFrame, deciles: pd.DataFrame) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str((root / TMP / "matplotlib").resolve()))
    import matplotlib.pyplot as plt

    destination = root / FIGURES
    destination.mkdir(parents=True, exist_ok=True)
    short_ids = [GLOBAL_ID, PRIMARY_ID, BLOCKA_ID, "prompt4b3__beat_costaware", "prompt4b4__sub_ldslgb", "prompt4b4__sub_densecat"]
    short = metrics.set_index("model_id").loc[short_ids].reset_index()
    atomic_csv(root, REPORTS / "prompt4c_plot_shortlist.csv", short)
    labels = {
        GLOBAL_ID: "Global", PRIMARY_ID: "Stage 3", BLOCKA_ID: "Block A",
        "prompt4b3__beat_costaware": "4B3 Beat", "prompt4b4__sub_ldslgb": "4B4 LDS",
        "prompt4b4__sub_densecat": "4B4 DenseWeight",
    }
    paths: list[str] = []

    def save(name: str) -> None:
        path = destination / name
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path.relative_to(root).as_posix())

    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(short))
    plt.bar(x, short["mae"], color=["#4c78a8" if item != PRIMARY_ID else "#f58518" for item in short["model_id"]])
    plt.xticks(x, [labels[item] for item in short["model_id"]], rotation=20, ha="right")
    plt.ylabel("Historical Validation MAE (thousand USD)")
    plt.title("Frozen final shortlist: overall MAE")
    save("01_final_shortlist_mae.png")

    pareto_ids = [GLOBAL_ID, PRIMARY_ID, BLOCKA_ID, "prompt4b4__sub_ldslgb"]
    pareto = metrics.set_index("model_id").loc[pareto_ids].reset_index()
    atomic_csv(root, REPORTS / "prompt4c_plot_pareto.csv", pareto)
    plt.figure(figsize=(7, 5))
    plt.scatter(pareto["bottom_90_mae"], pareto["top_decile_mae"], s=70)
    for _, row in pareto.iterrows():
        plt.annotate(labels[row["model_id"]], (row["bottom_90_mae"], row["top_decile_mae"]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Bottom-90 MAE")
    plt.ylabel("Top-decile MAE")
    plt.title("Frozen Body and Tail trade-off")
    save("02_bottom90_topdecile_pareto.png")

    plot_ids = [GLOBAL_ID, PRIMARY_ID, BLOCKA_ID]
    plotting = deciles.loc[deciles["model_id"].isin(plot_ids)].copy()
    atomic_csv(root, REPORTS / "prompt4c_plot_deciles.csv", plotting)
    for index, (field, ylabel, filename) in enumerate((
        ("mae", "MAE (thousand USD)", "03_historical_mae_by_decile.png"),
        ("mape_percent", "MAPE (%)", "04_historical_mape_by_decile.png"),
        ("wape_percent", "WAPE (%)", "05_historical_wape_by_decile.png"),
    )):
        plt.figure(figsize=(8, 4.5))
        for model_id in plot_ids:
            subset = plotting.loc[plotting["model_id"] == model_id].sort_values("decile_index")
            plt.plot(subset["decile_index"], subset[field], marker="o", label=labels[model_id])
        plt.xticks(range(1, 11), [f"D{i}" for i in range(1, 11)])
        plt.xlabel("True-target decile")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f"Historical {field.replace('_percent', '').upper()} by true-target decile")
        save(filename)

    plt.figure(figsize=(10, 5.3))
    plt.axis("off")
    boxes = [
        (0.03, 0.66, 0.18, 0.18, "35 named\nfeatures"),
        (0.27, 0.66, 0.20, 0.18, "Cat / LGB / XGB\n60 / 20 / 20"),
        (0.55, 0.76, 0.18, 0.15, "Meta-Gate\np(Tail)"),
        (0.55, 0.49, 0.18, 0.15, "Residual\nSpecialist"),
        (0.80, 0.62, 0.17, 0.18, "Global + gated\nresidual"),
    ]
    for x0, y0, width, height, text in boxes:
        plt.gca().add_patch(plt.Rectangle((x0, y0), width, height, fill=False, linewidth=1.8))
        plt.text(x0 + width / 2, y0 + height / 2, text, ha="center", va="center")
    arrows = [((0.21, 0.75), (0.27, 0.75)), ((0.47, 0.75), (0.55, 0.83)), ((0.47, 0.72), (0.55, 0.56)), ((0.73, 0.83), (0.80, 0.73)), ((0.73, 0.56), (0.80, 0.68))]
    for start, end in arrows:
        plt.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.6})
    plt.text(0.5, 0.23, "Training meta-targets use two-fold OOF Global predictions. Inference uses the final 500k Global models.", ha="center")
    plt.title("Frozen Stage 3 final inference architecture")
    save("06_stage3_inference_architecture.png")
    return {"status": "PASS", "figure_count": len(paths), "figures": paths}


def deterministic_final_twofold(y_true: Any, row_hash: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    hashes = pd.Series(row_hash, copy=False).astype(str).reset_index(drop=True)
    if target.size != 500_000 or hashes.size != target.size or hashes.duplicated().any():
        raise ValueError("Final OOF construction requires exactly 500,000 unique Development rows.")
    decile = duplicate_safe_deciles(target)
    tail = (target > TAIL_THRESHOLD).astype(np.int8)
    stratify = np.asarray([f"{int(d)}_{int(t)}" for d, t in zip(decile, tail)], dtype=object)
    fold_a, fold_b = train_test_split(
        np.arange(target.size), train_size=250_000, test_size=250_000,
        random_state=SEED, shuffle=True, stratify=stratify,
    )
    fold_a = np.sort(fold_a)
    fold_b = np.sort(fold_b)
    if np.intersect1d(fold_a, fold_b).size or np.union1d(fold_a, fold_b).size != target.size:
        raise RuntimeError("Final OOF folds do not form an exact partition.")
    labels = np.full(target.size, "", dtype=object)
    labels[fold_a] = "fold_a"
    labels[fold_b] = "fold_b"
    fold_assignment_digest = hashlib.sha256(
        "\n".join(f"{row}|{fold}" for row, fold in zip(hashes, labels)).encode("utf-8")
    ).hexdigest()
    evidence = {
        "status": "PASS",
        "random_state": SEED,
        "stratification": "duplicate-safe target decile plus strict operational Tail status",
        "fold_a_rows": int(fold_a.size),
        "fold_b_rows": int(fold_b.size),
        "overlap_rows": 0,
        "fold_a_ordered_digest": ordered_digest(hashes.iloc[fold_a]),
        "fold_b_ordered_digest": ordered_digest(hashes.iloc[fold_b]),
        "fold_a_membership_digest": membership_digest(hashes.iloc[fold_a]),
        "fold_b_membership_digest": membership_digest(hashes.iloc[fold_b]),
        "fold_assignment_digest": fold_assignment_digest,
    }
    return fold_a, fold_b, evidence


def create_refit_plan(root: Path, features: list[str], recipe: dict[str, Any]) -> dict[str, Any]:
    roles = final_refit_graph(features)
    source_bundles = load_prompt2_bundles(root)
    fixed_parameters = {family: fixed_crossfit_parameters(bundle) for family, bundle in source_bundles.items()}
    for role in roles:
        family = role.get("family")
        if family in fixed_parameters:
            role["fixed_configuration"] = fixed_parameters[family]
            role["target_mode"] = source_bundles[family].target_mode
            role["configuration_hash"] = json_digest({"parameters": fixed_parameters[family], "target_mode": source_bundles[family].target_mode})
        elif family == "catboost_classifier":
            role["fixed_configuration"] = META_PARAMETERS
            role["configuration_hash"] = json_digest(META_PARAMETERS)
        else:
            role["fixed_configuration"] = RESIDUAL_PARAMETERS
            role["configuration_hash"] = json_digest(RESIDUAL_PARAMETERS)
    plan = {
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "primary_recipe": PRIMARY_ID,
        "recipe_digest": json_digest(recipe),
        "scientific_candidate_searches": 0,
        "role_count": len(roles),
        "roles": roles,
        "heavy_fits_sequential": True,
        "seed": SEED,
        "threads": THREADS,
        "maximum_identical_technical_retry_per_role": 1,
        "oof_rule": "two equal deterministic folds from full Development using duplicate-safe target decile plus y>438.0 stratification",
    }
    if plan["role_count"] != 11:
        raise RuntimeError("BLOCKED_FINAL_REFIT_INTEGRITY")
    atomic_json(root, REPORTS / "prompt4c_refit_plan.json", plan)
    return plan


def load_prompt2_bundles(root: Path) -> dict[str, ModelBundle]:
    paths = {
        "catboost": "selected_catboost_without_lender.joblib",
        "lightgbm": "selected_lightgbm_without_lender.joblib",
        "xgboost": "selected_xgboost_without_lender.joblib",
    }
    bundles = {family: joblib.load(root / "outputs/models/prompt2" / name) for family, name in paths.items()}
    if any(not isinstance(bundle, ModelBundle) for bundle in bundles.values()):
        raise TypeError("A frozen Prompt 2 component is not a ModelBundle.")
    return bundles


def ledger_path(root: Path) -> Path:
    return root / REPORTS / "prompt4c_final_refit_ledger.json"


def load_ledger(root: Path) -> dict[str, Any]:
    path = ledger_path(root)
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if tuple(ledger.get("authorized_roles", [])) != FINAL_ROLES:
            raise RuntimeError("Final refit ledger role list changed.")
        return ledger
    return {
        "status": "IN_PROGRESS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "authorized_roles": list(FINAL_ROLES),
        "authorized_role_count": 11,
        "scientific_candidate_searches": 0,
        "maximum_identical_technical_retry_per_role": 1,
        "attempts": [],
        "completed_roles": [],
        "nonfit_activity": [
            {"type": "zero-fit reconstruction", "roles_consumed": 0},
            {"type": "reporting-only process", "roles_consumed": 0},
        ],
    }


def save_ledger(root: Path, ledger: dict[str, Any]) -> None:
    atomic_json(root, REPORTS / "prompt4c_final_refit_ledger.json", ledger)


def valid_saved_role(destination: Path, expected: dict[str, Any]) -> tuple[Any, dict[str, Any]] | None:
    manifest_path = destination / "manifest.json"
    artifact = destination / "bundle.joblib"
    if not (manifest_path.exists() and artifact.exists()):
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or any(manifest.get(key) != value for key, value in expected.items()):
        return None
    if file_sha256(artifact) != manifest.get("model_sha256"):
        return None
    return joblib.load(artifact), manifest


def run_final_role(
    root: Path,
    role: str,
    destination: Path,
    expected: dict[str, Any],
    work: Callable[[], tuple[Any, dict[str, Any]]],
) -> tuple[Any, dict[str, Any], bool]:
    if role not in FINAL_ROLES:
        raise ValueError(f"Unauthorized final-refit role: {role}")
    saved = valid_saved_role(destination, expected)
    if saved is not None:
        bundle, manifest = saved
        ledger = load_ledger(root)
        if role not in ledger["completed_roles"]:
            ledger["completed_roles"].append(role)
            save_ledger(root, ledger)
        return bundle, manifest, True
    ledger = load_ledger(root)
    prior_attempts = [entry for entry in ledger["attempts"] if entry["role"] == role]
    next_attempt = len(prior_attempts) + 1
    if next_attempt > 2:
        raise RuntimeError("BLOCKED_FINAL_REFIT_INTEGRITY")
    while next_attempt <= 2:
        entry = {
            "role": role,
            "physical_attempt": next_attempt,
            "retry_count": next_attempt - 1,
            "started_at_utc": utc_now(),
            "status": "RUNNING",
            "configuration_hash": expected["configuration_hash"],
            "training_membership_digest": expected["training_membership_digest"],
        }
        ledger["attempts"].append(entry)
        save_ledger(root, ledger)
        started = time.perf_counter()
        try:
            bundle, manifest = work()
            entry.update({
                "status": "PASS",
                "completed_at_utc": utc_now(),
                "runtime_seconds": float(time.perf_counter() - started),
                "model_path": str((destination / "bundle.joblib").relative_to(root).as_posix()),
                "model_sha256": manifest["model_sha256"],
            })
            if role not in ledger["completed_roles"]:
                ledger["completed_roles"].append(role)
            save_ledger(root, ledger)
            return bundle, manifest, False
        except Exception as exc:
            entry.update({
                "status": "TECHNICAL_FAILURE",
                "completed_at_utc": utc_now(),
                "runtime_seconds": float(time.perf_counter() - started),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            save_ledger(root, ledger)
            saved_after_failure = valid_saved_role(destination, expected)
            if saved_after_failure is not None:
                bundle, manifest = saved_after_failure
                entry["status"] = "PASS_ARTIFACT_SAVED_BEFORE_REPORT_FAILURE"
                if role not in ledger["completed_roles"]:
                    ledger["completed_roles"].append(role)
                save_ledger(root, ledger)
                return bundle, manifest, True
            next_attempt += 1
            if next_attempt > 2:
                raise RuntimeError("BLOCKED_FINAL_REFIT_INTEGRITY") from exc
    raise AssertionError("Unreachable final-role state.")


def save_component_bundle(
    root: Path,
    destination: Path,
    bundle: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    artifact, digest = atomic_joblib(root, destination.relative_to(root) / "bundle.joblib", bundle)
    payload = {"status": "COMPLETE", "artifact": artifact.name, "model_sha256": digest, **manifest}
    atomic_json(root, destination.relative_to(root) / "manifest.json", payload)
    reloaded = valid_saved_role(destination, {key: manifest[key] for key in ("role", "configuration_hash", "training_membership_digest")})
    if reloaded is None:
        raise RuntimeError("Saved final component did not validate.")
    return payload


def fit_base_bundle(
    root: Path,
    role: str,
    family: str,
    source_bundle: ModelBundle,
    features: list[str],
    frame: pd.DataFrame,
    fit_index: np.ndarray,
    training_population: str,
    oof_fold: str | None,
    predict_fold: str | None,
) -> tuple[ModelBundle, dict[str, Any], bool]:
    from prompt4b_crossfit import _fit_one_family

    parameters = fixed_crossfit_parameters(source_bundle)
    configuration_hash = json_digest({"family": family, "parameters": parameters, "target_mode": source_bundle.target_mode})
    training_membership_digest = membership_digest(frame.iloc[fit_index]["row_hash"])
    expected = {
        "role": role,
        "configuration_hash": configuration_hash,
        "training_membership_digest": training_membership_digest,
    }
    destination = root / MODELS / "components" / role

    def work() -> tuple[ModelBundle, dict[str, Any]]:
        preprocessor, model = _fit_one_family(
            family,
            parameters,
            features,
            frame.iloc[fit_index][features],
            frame.iloc[fit_index][TARGET].to_numpy(float),
            source_bundle.target_mode,
        )
        bundle = ModelBundle(
            model_id=role,
            family=family,
            feature_names=features,
            feature_contract_name=FEATURE_CONTRACT,
            target_mode=source_bundle.target_mode,
            preprocessor=preprocessor,
            model=model,
            package_versions=package_versions(),
            model_parameters=parameters,
            selected_best_iteration=int(parameters.get("iterations", parameters.get("n_estimators"))),
            development_source_sha256=DEVELOPMENT_SHA256,
            train_row_hash_digest=ordered_digest(frame.iloc[fit_index]["row_hash"]),
            validation_row_hash_digest="",
            metadata={
                "fit_role": role,
                "fitted_iterations": int(parameters.get("iterations", parameters.get("n_estimators"))),
                "final_refit": True,
                "oof_fold": oof_fold,
                "predict_fold": predict_fold,
            },
        )
        manifest = save_component_bundle(root, destination, bundle, {
            **expected,
            "recipe_source": source_bundle.model_id,
            "model_family": family,
            "target_scale": source_bundle.target_mode,
            "objective": parameters.get("loss_function", parameters.get("objective")),
            "feature_contract": FEATURE_CONTRACT,
            "features": features,
            "training_rows": int(len(fit_index)),
            "training_population": training_population,
            "oof_fold": oof_fold,
            "predict_fold": predict_fold,
            "random_seed": SEED,
            "threads": THREADS,
            "fixed_iterations": int(parameters.get("iterations", parameters.get("n_estimators"))),
            "configuration": parameters,
            "ordered_training_digest": ordered_digest(frame.iloc[fit_index]["row_hash"]),
            "package_versions": package_versions(),
            "clean_reload_status": "PENDING_COMPOSITE_RELOAD",
        })
        return bundle, manifest

    return run_final_role(root, role, destination, expected, work)


def build_final_oof(
    root: Path,
    development: pd.DataFrame,
    features: list[str],
    source_bundles: dict[str, ModelBundle],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    fold_a, fold_b, split = deterministic_final_twofold(development[TARGET], development["row_hash"])
    components = {family: np.full(len(development), np.nan, dtype=np.float64) for family in source_bundles}
    manifests: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for fit_fold, fit_index, predict_index in (("fold_a", fold_a, fold_b), ("fold_b", fold_b, fold_a)):
        predict_fold = "fold_b" if fit_fold == "fold_a" else "fold_a"
        for family in ("catboost", "lightgbm", "xgboost"):
            role = f"prompt4c_oof_{family}_{fit_fold}"
            bundle, manifest, reused = fit_base_bundle(
                root, role, family, source_bundles[family], features, development,
                fit_index, f"deterministic {fit_fold} 250,000-row fit fold", fit_fold, predict_fold,
            )
            prediction = bundle.predict(development.iloc[predict_index][features])
            if prediction.shape != (250_000,) or not np.isfinite(prediction).all():
                raise RuntimeError(f"Invalid OOF prediction for {role}")
            components[family][predict_index] = prediction
            manifests[role] = manifest
            records.append({
                "role": role,
                "family": family,
                "fit_fold": fit_fold,
                "predict_fold": predict_fold,
                "fit_rows": 250_000,
                "predict_rows": 250_000,
                "reused": bool(reused),
                "model_sha256": manifest["model_sha256"],
            })
    if any(not np.isfinite(values).all() for values in components.values()):
        raise RuntimeError("OOF components are incomplete.")
    global_oof = 0.60 * components["catboost"] + 0.20 * components["lightgbm"] + 0.20 * components["xgboost"]
    folds = np.full(len(development), "", dtype=object)
    folds[fold_a] = "fold_a"
    folds[fold_b] = "fold_b"
    oof = pd.DataFrame({
        "row_hash": development["row_hash"].astype(str),
        "y_true": development[TARGET].to_numpy(float),
        "catboost_oof": components["catboost"],
        "lightgbm_oof": components["lightgbm"],
        "xgboost_oof": components["xgboost"],
        "global_oof_prediction": global_oof,
        "fold_id": folds,
        "exactly_one_oof_prediction": True,
        "self_fit": False,
    })
    if len(oof) != 500_000 or not oof["row_hash"].is_unique or int(oof["self_fit"].sum()) != 0:
        raise RuntimeError("Final OOF integrity failed.")
    path = atomic_parquet(root, PREDICTIONS / "oof_global_500k.parquet", oof)
    manifest = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "rows": len(oof),
        "row_hash_unique": bool(oof["row_hash"].is_unique),
        "row_order_digest": ordered_digest(oof["row_hash"]),
        "exactly_one_oof_per_row": bool(oof["exactly_one_oof_prediction"].all()),
        "zero_self_fit_rows": int(oof["self_fit"].sum()),
        "finite_predictions": bool(np.isfinite(oof[["catboost_oof", "lightgbm_oof", "xgboost_oof", "global_oof_prediction"]].to_numpy()).all()),
        "ensemble_formula": "0.60*CatBoost + 0.20*LightGBM + 0.20*XGBoost",
        "split": split,
        "fit_records": records,
        "target_usage": "training-construction evidence only; not generalization evidence",
    }
    atomic_json(root, REPORTS / "prompt4c_oof_manifest.json", manifest)
    return oof, manifest, manifests


def fit_meta_gate_bundle(
    root: Path,
    development: pd.DataFrame,
    features: list[str],
    global_oof: np.ndarray,
    oof_sha256: str,
) -> tuple[MetaGateBundle, dict[str, Any], bool]:
    from catboost import CatBoostClassifier

    role = "prompt4c_meta_gate_500k"
    configuration_hash = json_digest(META_PARAMETERS)
    training_membership_digest = membership_digest(development["row_hash"])
    expected = {"role": role, "configuration_hash": configuration_hash, "training_membership_digest": training_membership_digest}
    destination = root / MODELS / "components" / role

    def work() -> tuple[MetaGateBundle, dict[str, Any]]:
        frame = development[features].copy()
        frame[GLOBAL_FEATURE] = global_oof
        labels = (development[TARGET].to_numpy(float) > TAIL_THRESHOLD).astype(np.int8)
        preprocessor = MetaPreprocessor(features).fit(frame)
        model = CatBoostClassifier(**META_PARAMETERS, allow_writing_files=False, task_type="CPU")
        model.fit(preprocessor.transform(frame), labels, cat_features=preprocessor.cat_feature_indices_, verbose=False)
        if int(model.tree_count_) != META_ITERATIONS:
            raise RuntimeError("Final Meta-Gate iteration count changed.")
        metadata = {
            "model_role": role,
            "model_configuration": META_PARAMETERS,
            "selected_iteration": META_ITERATIONS,
            "feature_contract": features + [GLOBAL_FEATURE],
            "seed": SEED,
            "training_row_count": 500_000,
            "training_membership_digest": ordered_digest(development["row_hash"]),
            "oof_prediction_sha256": oof_sha256,
            "package_versions": package_versions(),
        }
        bundle = MetaGateBundle(preprocessor, model, metadata)
        manifest = save_component_bundle(root, destination, bundle, {
            **expected,
            "recipe_source": "outputs/models/prompt4b/meta_gate/full/manifest.json",
            "model_family": "catboost_classifier",
            "target_scale": "binary operational Tail label",
            "objective": "Logloss",
            "feature_contract": FEATURE_CONTRACT,
            "features": features + [GLOBAL_FEATURE],
            "training_rows": 500_000,
            "training_population": "all Development rows",
            "global_feature_source": "leakage-safe final two-fold OOF Global prediction",
            "random_seed": SEED,
            "threads": THREADS,
            "fixed_iterations": META_ITERATIONS,
            "configuration": META_PARAMETERS,
            "oof_prediction_sha256": oof_sha256,
            "clean_reload_status": "PENDING_COMPOSITE_RELOAD",
        })
        return bundle, manifest

    return run_final_role(root, role, destination, expected, work)


def fit_residual_bundle(
    root: Path,
    development: pd.DataFrame,
    features: list[str],
    global_oof: np.ndarray,
    oof_sha256: str,
) -> tuple[ResidualSpecialistBundle, dict[str, Any], bool]:
    role = "prompt4c_residual_specialist_500k"
    tail = development[TARGET].to_numpy(float) > TAIL_THRESHOLD
    tail_index = np.flatnonzero(tail)
    configuration_hash = json_digest(RESIDUAL_PARAMETERS)
    training_membership_digest = membership_digest(development.iloc[tail_index]["row_hash"])
    expected = {"role": role, "configuration_hash": configuration_hash, "training_membership_digest": training_membership_digest}
    destination = root / MODELS / "components" / role

    def work() -> tuple[ResidualSpecialistBundle, dict[str, Any]]:
        frame = development[features].copy()
        frame[GLOBAL_FEATURE] = global_oof
        target = residual_target(development[TARGET], global_oof)
        model, preprocessor, selected = fit_residual_model(
            frame.iloc[tail_index], target[tail_index], features, parameters=RESIDUAL_PARAMETERS,
        )
        if selected != RESIDUAL_ITERATIONS:
            raise RuntimeError("Final residual iteration count changed.")
        metadata = {
            "model_role": role,
            "model_configuration": RESIDUAL_PARAMETERS,
            "selected_iteration": RESIDUAL_ITERATIONS,
            "feature_contract": features + [GLOBAL_FEATURE],
            "target_definition": "loan_amount_000s - leakage-safe global_oof_prediction",
            "tail_only": True,
            "training_row_count": int(len(tail_index)),
            "training_membership_digest": ordered_digest(development.iloc[tail_index]["row_hash"]),
            "seed": SEED,
            "oof_prediction_sha256": oof_sha256,
            "package_versions": package_versions(),
        }
        bundle = ResidualSpecialistBundle(preprocessor, model, metadata)
        manifest = save_component_bundle(root, destination, bundle, {
            **expected,
            "recipe_source": "outputs/models/prompt4b/residual_specialist/full/manifest.json",
            "model_family": "catboost_regressor",
            "target_scale": "raw residual",
            "objective": "MAE",
            "feature_contract": FEATURE_CONTRACT,
            "features": features + [GLOBAL_FEATURE],
            "training_rows": int(len(tail_index)),
            "training_population": "strict operational Tail: loan_amount_000s > 438.0",
            "target_definition": "loan_amount_000s - leakage-safe global_oof_prediction",
            "global_feature_source": "leakage-safe final two-fold OOF Global prediction",
            "random_seed": SEED,
            "threads": THREADS,
            "fixed_iterations": RESIDUAL_ITERATIONS,
            "configuration": RESIDUAL_PARAMETERS,
            "oof_prediction_sha256": oof_sha256,
            "clean_reload_status": "PENDING_COMPOSITE_RELOAD",
        })
        return bundle, manifest

    return run_final_role(root, role, destination, expected, work)


def run_final_refits(root: Path, features: list[str]) -> dict[str, Any]:
    plan = read_json(root, REPORTS / "prompt4c_refit_plan.json")
    freeze = read_json(root, REPORTS / "prompt4c_final_selection_freeze.json")
    if plan.get("status") != "FROZEN" or plan.get("role_count") != 11 or freeze.get("final_primary_recipe") != PRIMARY_ID:
        raise RuntimeError("Final refit plan or selection freeze is invalid.")
    development = development_frame(root, features, include_target=True).copy()
    source_bundles = load_prompt2_bundles(root)
    oof_path = root / PREDICTIONS / "oof_global_500k.parquet"
    oof_manifest_path = root / REPORTS / "prompt4c_oof_manifest.json"
    if oof_path.exists() and oof_manifest_path.exists():
        oof = pd.read_parquet(oof_path)
        oof_manifest = read_json(root, REPORTS / "prompt4c_oof_manifest.json")
        valid_oof = (
            oof_manifest.get("status") == "PASS"
            and oof_manifest.get("sha256") == file_sha256(oof_path)
            and len(oof) == 500_000
            and oof["row_hash"].astype(str).equals(development["row_hash"].astype(str))
            and int(oof["self_fit"].sum()) == 0
        )
        if not valid_oof:
            raise RuntimeError("Existing final OOF artifact is invalid.")
        oof_component_manifests = {}
    else:
        oof, oof_manifest, oof_component_manifests = build_final_oof(root, development, features, source_bundles)
    full_bundles: dict[str, ModelBundle] = {}
    all_index = np.arange(len(development))
    full_manifests: dict[str, dict[str, Any]] = {}
    for family in ("catboost", "lightgbm", "xgboost"):
        role = f"prompt4c_full_{family}_500k"
        bundle, manifest, _ = fit_base_bundle(
            root, role, family, source_bundles[family], features, development,
            all_index, "all 500,000 Development rows", None, None,
        )
        full_bundles[family] = bundle
        full_manifests[role] = manifest
    meta_bundle, meta_manifest, _ = fit_meta_gate_bundle(
        root, development, features, oof["global_oof_prediction"].to_numpy(float), oof_manifest["sha256"]
    )
    residual_bundle, residual_manifest, _ = fit_residual_bundle(
        root, development, features, oof["global_oof_prediction"].to_numpy(float), oof_manifest["sha256"]
    )
    ledger = load_ledger(root)
    if set(ledger["completed_roles"]) != set(FINAL_ROLES):
        raise RuntimeError("Final refit roles are incomplete.")
    retries = sum(1 for entry in ledger["attempts"] if int(entry["physical_attempt"]) == 2)
    ledger["status"] = "COMPLETE"
    ledger["completed_role_count"] = len(ledger["completed_roles"])
    ledger["physical_attempt_count"] = len(ledger["attempts"])
    ledger["technical_retry_count"] = retries
    ledger["scientific_candidate_searches"] = 0
    ledger["completed_at_utc"] = utc_now()
    save_ledger(root, ledger)
    manifests = {}
    for role in FINAL_ROLES:
        manifest_path = root / MODELS / "components" / role / "manifest.json"
        manifests[role] = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {
        "development": development,
        "oof": oof,
        "oof_manifest": oof_manifest,
        "full_bundles": full_bundles,
        "meta_bundle": meta_bundle,
        "residual_bundle": residual_bundle,
        "manifests": manifests,
        "final_tail_rows": int(np.count_nonzero(development[TARGET].to_numpy(float) > TAIL_THRESHOLD)),
        "technical_retries": retries,
    }
    return result


def runtime_report(root: Path) -> dict[str, Any]:
    path = root / REPORTS / "prompt4c_runtime.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "IN_PROGRESS",
        "created_at_utc": utc_now(),
        "preflight": 0.0,
        "state_reconciliation": 0.0,
        "stage3_recipe_reconstruction": 0.0,
        "historical_prediction_reproduction": 0.0,
        "mape_wape_reporting_extension": 0.0,
        "oof_global_fits": 0.0,
        "oof_prediction_construction": 0.0,
        "final_catboost_fit": 0.0,
        "final_lightgbm_fit": 0.0,
        "final_xgboost_fit": 0.0,
        "final_meta_gate_fit": 0.0,
        "final_residual_specialist_fit": 0.0,
        "bundle_packaging": 0.0,
        "clean_reload": 0.0,
        "notebook": 0.0,
        "independent_review": 0.0,
        "final_verification": 0.0,
        "freeze_promotion": 0.0,
        "total_elapsed": 0.0,
    }


def update_runtime(root: Path, **durations: float) -> dict[str, Any]:
    report = runtime_report(root)
    for key, value in durations.items():
        report[key] = float(report.get(key, 0.0)) + float(value)
    phase_keys = [key for key in report if key not in {"status", "created_at_utc", "updated_at_utc", "total_elapsed"}]
    report["total_elapsed"] = float(sum(float(report.get(key, 0.0)) for key in phase_keys))
    report["updated_at_utc"] = utc_now()
    atomic_json(root, REPORTS / "prompt4c_runtime.json", report)
    return report


def _frame_digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def clean_reload_worker(root: Path, bundle_path: Path, sample_path: Path, output_path: Path) -> None:
    sample = pd.read_parquet(sample_path)
    row_hash = sample.pop("row_hash").astype(str)
    bundle = load_final_bundle(bundle_path)
    prediction = bundle.predict(sample)
    atomic_parquet(
        root,
        output_path.relative_to(root),
        pd.DataFrame({"row_hash": row_hash, "prediction": prediction}),
    )


def package_final_bundles(root: Path, refit: dict[str, Any], features: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    global_metadata = {
        "model_id": "final_global_500k",
        "role": "simple comparator",
        "training_rows": 500_000,
        "feature_contract": FEATURE_CONTRACT,
        "weights": {"catboost": 0.60, "lightgbm": 0.20, "xgboost": 0.20},
        "development_sha256": DEVELOPMENT_SHA256,
        "target_unit": "thousand USD",
    }
    global_bundle = FinalGlobalBundle(features, refit["full_bundles"], metadata=global_metadata)
    global_path, global_hash = atomic_joblib(root, MODELS / "global_comparator/bundle.joblib", global_bundle)
    global_manifest = {
        "status": "COMPLETE",
        "bundle_type": "FinalGlobalBundle",
        "artifact": "bundle.joblib",
        "bundle_sha256": global_hash,
        "metadata": global_metadata,
        "component_hashes": {
            family: refit["manifests"][f"prompt4c_full_{family}_500k"]["model_sha256"]
            for family in ("catboost", "lightgbm", "xgboost")
        },
        "predict_interface": "predict(features)",
        "target_required": False,
    }
    atomic_json(root, MODELS / "global_comparator/manifest.json", global_manifest)
    primary_metadata = {
        "model_id": "final_primary_stage3_500k",
        "historical_recipe_id": PRIMARY_ID,
        "role": "Primary",
        "training_rows": 500_000,
        "feature_contract": FEATURE_CONTRACT,
        "tail_threshold": TAIL_THRESHOLD,
        "gate_threshold": FINAL_GATE_THRESHOLD,
        "correction_alpha": FINAL_CORRECTION_ALPHA,
        "cap": None,
        "clip": None,
        "positivity_rule": None,
        "prediction_formula": "global + strength*residual",
        "development_sha256": DEVELOPMENT_SHA256,
        "target_unit": "thousand USD",
    }
    primary_bundle = FinalPrimaryBundle(
        features,
        global_bundle,
        refit["meta_bundle"],
        refit["residual_bundle"],
        metadata=primary_metadata,
    )
    primary_path, primary_hash = atomic_joblib(root, MODELS / "primary_stage3/bundle.joblib", primary_bundle)
    primary_manifest = {
        "status": "COMPLETE",
        "bundle_type": "FinalPrimaryBundle",
        "artifact": "bundle.joblib",
        "bundle_sha256": primary_hash,
        "metadata": primary_metadata,
        "component_hashes": {
            **global_manifest["component_hashes"],
            "meta_gate": refit["manifests"]["prompt4c_meta_gate_500k"]["model_sha256"],
            "residual_specialist": refit["manifests"]["prompt4c_residual_specialist_500k"]["model_sha256"],
        },
        "predict_interface": "predict(features)",
        "target_required": False,
        "membership_role_required": False,
        "oof_column_required": False,
        "realized_tail_label_required": False,
    }
    atomic_json(root, MODELS / "primary_stage3/manifest.json", primary_manifest)
    update_runtime(root, bundle_packaging=time.perf_counter() - started)

    reload_started = time.perf_counter()
    sample = refit["development"].iloc[:1000][["row_hash"] + features].copy()
    sample_path = atomic_parquet(root, TMP / "clean_reload_feature_sample.parquet", sample)
    input_frame = sample[features].copy()
    before_digest = _frame_digest(input_frame)
    expected = {
        "primary": primary_bundle.predict(input_frame),
        "global": global_bundle.predict(input_frame),
    }
    after_digest = _frame_digest(input_frame)
    if before_digest != after_digest:
        raise RuntimeError("Final bundle mutated its source frame.")
    results: list[dict[str, Any]] = []
    for name, bundle_path in (("primary", primary_path), ("global", global_path)):
        output = root / TMP / f"clean_reload_{name}.parquet"
        command = [
            sys.executable, str(Path(__file__).resolve()), "clean-reload-worker",
            "--root", str(root), "--bundle", str(bundle_path),
            "--sample", str(sample_path), "--output", str(output),
        ]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=600)
        if completed.returncode != 0:
            raise RuntimeError(f"Clean reload worker failed for {name}: {completed.stderr[-2000:]}")
        actual = pd.read_parquet(output)
        if not actual["row_hash"].astype(str).equals(sample["row_hash"].astype(str)):
            raise RuntimeError("Clean reload row order changed.")
        difference = float(np.max(np.abs(actual["prediction"].to_numpy(float) - expected[name])))
        results.append({
            "bundle": name,
            "path": bundle_path.relative_to(root).as_posix(),
            "bundle_sha256": file_sha256(bundle_path),
            "rows": len(actual),
            "maximum_absolute_difference": difference,
            "required_tolerance": 0.0,
            "finite": bool(np.isfinite(actual["prediction"].to_numpy(float)).all()),
            "source_frame_unchanged": before_digest == after_digest,
            "status": "PASS" if difference == 0.0 else "FAIL",
        })
    # Bundle input guards use the final Primary; the Global shares the same selector.
    missing_error = duplicate_error = False
    try:
        primary_bundle.predict(input_frame.drop(columns=[features[0]]))
    except ValueError:
        missing_error = True
    duplicate = pd.concat([input_frame, input_frame[[features[0]]]], axis=1)
    try:
        primary_bundle.predict(duplicate)
    except ValueError:
        duplicate_error = True
    reordered = primary_bundle.predict(input_frame.loc[:, list(reversed(features))])
    extras = input_frame.copy()
    extras["row_hash"] = sample["row_hash"].astype(str)
    extras["respondent_id"] = "PROHIBITED_EXTRA"
    extras["applicant_sex_name"] = "PROHIBITED_EXTRA"
    extra_prediction = primary_bundle.predict(extras)
    guards = {
        "exact_feature_count": len(features) == 35,
        "missing_required_feature_error": missing_error,
        "duplicate_feature_error": duplicate_error,
        "reordered_named_frame_maximum_difference": float(np.max(np.abs(reordered - expected["primary"]))),
        "prohibited_extra_columns_consumed": False,
        "prohibited_extra_columns_maximum_difference": float(np.max(np.abs(extra_prediction - expected["primary"]))),
        "target_required": False,
        "row_hash_consumed": False,
        "respondent_id_consumed": False,
        "sensitive_columns_consumed": False,
    }
    status = "PASS" if all(item["status"] == "PASS" for item in results) and missing_error and duplicate_error and guards["reordered_named_frame_maximum_difference"] == 0.0 and guards["prohibited_extra_columns_maximum_difference"] == 0.0 else "FAIL"
    report = {
        "status": status,
        "created_at_utc": utc_now(),
        "sample_rows": 1000,
        "sample_row_hash_digest": ordered_digest(sample["row_hash"]),
        "bundles": results,
        "input_guards": guards,
    }
    atomic_json(root, REPORTS / "prompt4c_bundle_reload.json", report)
    update_runtime(root, clean_reload=time.perf_counter() - reload_started)
    if status != "PASS":
        raise RuntimeError("BLOCKED_FINAL_BUNDLE_INTEGRITY")
    return {
        "primary_path": primary_path,
        "primary_hash": primary_hash,
        "global_path": global_path,
        "global_hash": global_hash,
        "primary_manifest": primary_manifest,
        "global_manifest": global_manifest,
        "reload": report,
    }


def create_final_manifests(root: Path, bundles: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = load_ledger(root)
    entries = []
    for role in FINAL_ROLES:
        manifest_path = root / MODELS / "components" / role / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        attempts = [entry for entry in ledger["attempts"] if entry["role"] == role]
        passed = [entry for entry in attempts if str(entry["status"]).startswith("PASS")]
        physical_attempt = max((int(entry["physical_attempt"]) for entry in passed), default=1)
        runtime_seconds = sum(float(entry.get("runtime_seconds", 0.0)) for entry in attempts)
        entries.append({
            "role": role,
            "recipe_source": manifest["recipe_source"],
            "model_family": manifest["model_family"],
            "target_scale": manifest["target_scale"],
            "objective": manifest["objective"],
            "feature_contract": manifest["feature_contract"],
            "training_rows": manifest["training_rows"],
            "oof_or_full_role": "OOF" if "_oof_" in role else "full",
            "fold_id": manifest.get("oof_fold"),
            "random_seed": manifest["random_seed"],
            "threads": manifest["threads"],
            "fixed_iterations": manifest["fixed_iterations"],
            "configuration_hash": manifest["configuration_hash"],
            "model_path": (manifest_path.parent / "bundle.joblib").relative_to(root).as_posix(),
            "model_sha256": manifest["model_sha256"],
            "runtime_seconds": runtime_seconds,
            "physical_attempt": physical_attempt,
            "retry_count": max(0, physical_attempt - 1),
            "clean_reload_status": "PASS_VIA_FINAL_COMPOSITE_RELOAD",
        })
    model_manifest = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "artifact_count": len(entries),
        "scientific_candidate_searches": 0,
        "artifacts": entries,
        "final_primary_bundle": {
            "path": bundles["primary_path"].relative_to(root).as_posix(),
            "sha256": bundles["primary_hash"],
        },
        "final_global_bundle": {
            "path": bundles["global_path"].relative_to(root).as_posix(),
            "sha256": bundles["global_hash"],
        },
    }
    atomic_json(root, REPORTS / "prompt4c_final_model_manifest.json", model_manifest)
    historical_sources = historical_prediction_sources(root)
    oof_path = root / PREDICTIONS / "oof_global_500k.parquet"
    prediction_manifest = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "iid_prediction_count": 0,
        "development_only": True,
        "artifacts": [
            {
                "role": "final OOF Global training construction",
                "path": oof_path.relative_to(root).as_posix(),
                "sha256": file_sha256(oof_path),
                "rows": 500_000,
            },
            *[
                {
                    "role": f"historical zero-fit reconstruction: {model_id}",
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "rows": 100_000,
                }
                for model_id, path in historical_sources.items()
            ],
            {
                "role": "clean reload feature sample",
                "path": (root / TMP / "clean_reload_feature_sample.parquet").relative_to(root).as_posix(),
                "sha256": file_sha256(root / TMP / "clean_reload_feature_sample.parquet"),
                "rows": 1000,
            },
        ],
    }
    atomic_json(root, REPORTS / "prompt4c_prediction_manifest.json", prediction_manifest)
    return model_manifest, prediction_manifest


def build_and_execute_notebook(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    import nbformat as nbf
    from nbclient import NotebookClient

    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells: list[Any] = []

    def md(title: str, text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{text}"))

    cells.append(nbf.v4.new_markdown_cell(
        "# Prompt 4C - Final Selection and Pre-IID Freeze\n\n"
        "This notebook uses saved artifacts only. It fits no model and opens no Raw or IID data."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport pandas as pd\n"
        "from IPython.display import Image, display\nROOT=Path('..').resolve()\n"
        "REPORTS=ROOT/'outputs/reports'\nFIGURES=ROOT/'outputs/figures/prompt4c'"
    ))
    md("1. Development research is closed", "Prompt 4C does not search for a better model. All Development research is closed. No Candidate reached a six-of-six breakthrough.")
    cells.append(nbf.v4.new_code_cell("freeze=json.loads((REPORTS/'prompt4c_final_selection_freeze.json').read_text()); display(pd.DataFrame([{'status':freeze['status'],'research_closed':freeze['development_research_closed'],'primary':freeze['final_primary_recipe'],'comparator':freeze['simple_comparator']}]))"))
    md("2. Final model choice", "Stage 3 is the final Primary. It gives a conservative Body and Tail balance. Block A had a slightly lower Development MAE, but its Body result was worse and its MAE advantage was not clearly separated from Stage 3.")
    cells.append(nbf.v4.new_code_cell("metrics=pd.read_csv(REPORTS/'prompt4c_historical_metric_extension.csv'); display(metrics[['model_id','role','mae','rmse','bottom_90_mae','top_decile_mae','mape_percent','wape_percent']].round(4))"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIGURES/'01_final_shortlist_mae.png')))"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIGURES/'02_bottom90_topdecile_pareto.png')))"))
    md("3. Percentage errors are secondary", "MAPE and WAPE help interpretation. They did not select the model and cannot change the Primary. The decile analysis separates absolute error from percentage error.")
    cells.append(nbf.v4.new_code_cell("deciles=pd.read_csv(REPORTS/'prompt4c_historical_decile_metrics.csv'); display(deciles.loc[deciles.model_id.isin(['ens_boost_cat060','stage3_residual_t75_a75','nf_global2_oldraw_direct_cap25'])].head(15).round(4))"))
    for name in ("03_historical_mae_by_decile.png", "04_historical_mape_by_decile.png", "05_historical_wape_by_decile.png"):
        cells.append(nbf.v4.new_code_cell(f"display(Image(filename=str(FIGURES/'{name}')))"))
    md("4. Exact historical reconstruction", "The saved Global and Stage 3 Validation predictions were reconstructed exactly before final fitting. The Meta-Gate and residual proposal also matched exactly.")
    cells.append(nbf.v4.new_code_cell("g=json.loads((REPORTS/'prompt4c_global_recipe_reproduction.json').read_text()); s=json.loads((REPORTS/'prompt4c_stage3_recipe_reproduction.json').read_text()); display(pd.DataFrame([{'check':'Global','status':g['status'],'max_difference':g['maximum_absolute_prediction_difference']},{'check':'Stage 3','status':s['status'],'max_difference':s['component_maximum_absolute_differences']['final_prediction']}]))"))
    md("5. Leakage-safe final refit", "The final meta-targets used two-fold OOF Global predictions. Each Development row was predicted only by models that did not fit that row. The final models then used all 500,000 Development rows.")
    cells.append(nbf.v4.new_code_cell("oof=json.loads((REPORTS/'prompt4c_oof_manifest.json').read_text()); ledger=json.loads((REPORTS/'prompt4c_final_refit_ledger.json').read_text()); display(pd.DataFrame([{'oof_rows':oof['rows'],'self_fit_rows':oof['zero_self_fit_rows'],'refit_roles':ledger['completed_role_count'],'technical_retries':ledger['technical_retry_count']}]))"))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIGURES/'06_stage3_inference_architecture.png')))"))
    md("6. Final bundle checks", "The Primary and Global bundles use features only. Both pass clean-process reload with exact prediction equality.")
    cells.append(nbf.v4.new_code_cell("reload=json.loads((REPORTS/'prompt4c_bundle_reload.json').read_text()); display(pd.DataFrame(reload['bundles']))"))
    md("7. IID remains closed", "Prompt 4C opened no IID feature or target file and created no IID prediction. Block A is included only when exact zero-fit packaging is possible.")
    cells.append(nbf.v4.new_code_cell("blocka=json.loads((REPORTS/'prompt4c_blocka_package_eligibility.json').read_text()); protocol=json.loads((REPORTS/'prompt4c_prompt5_protocol.json').read_text()); display(pd.DataFrame([{'blocka_iid_eligible':blocka['historical_blocka_iid_eligible'],'raw_access':protocol['raw_access_count'],'iid_feature_access':protocol['iid_feature_access_count'],'iid_target_access':protocol['iid_target_access_count'],'iid_predictions':protocol['iid_prediction_count']}]))"))
    md("8. Prompt 5 is frozen", "Prompt 5 may evaluate only the predeclared model set. It uses MAE as Primary, the frozen supporting metrics, two target-band views, and a paired bootstrap. It can never refit or replace the Primary.")
    cells.append(nbf.v4.new_code_cell("display(pd.DataFrame(protocol['protocol']['permitted_models'])); display(pd.DataFrame([protocol['protocol']['bootstrap']]))"))
    md("9. Pre-IID closure", "Final Stage 3 and Global were trained on all Development rows and pass clean reload. IID is still unopened. No IID result may change the Primary model.")
    notebook.cells = cells
    path = root / NOTEBOOK
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    executed = client.execute()
    nbf.write(executed, path)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    inline_images = sum(
        1 for cell in code_cells for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"} and "image/png" in output.get("data", {})
    )
    inline_tables = sum(
        1 for cell in code_cells for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"} and "text/html" in output.get("data", {})
    )
    source = "\n".join(cell.source for cell in code_cells).lower()
    report = {
        "status": "PASS" if not errors and inline_images >= 6 and inline_tables >= 6 else "FAIL",
        "created_at_utc": utc_now(),
        "attempt": 1,
        "notebook_path": NOTEBOOK.as_posix(),
        "code_cells": len(code_cells),
        "error_count": len(errors),
        "inline_image_outputs": inline_images,
        "inline_table_outputs": inline_tables,
        "fit_calls": source.count(".fit("),
        "raw_data_access": "outputs/data/" in source or "data/" in source,
        "iid_access": "iid_holdout" in source,
        "prediction_generation": ".predict(" in source,
        "artifact_only": True,
    }
    atomic_json(root, REPORTS / "prompt4c_notebook_execution.json", report)
    update_runtime(root, notebook=time.perf_counter() - started)
    if report["status"] != "PASS" or report["fit_calls"] != 0 or report["iid_access"]:
        raise RuntimeError("Artifact-only Prompt 4C notebook failed.")
    return report


def create_freeze_candidate(
    root: Path,
    features: list[str],
    bundles: dict[str, Any],
    cutpoints: dict[str, float],
) -> dict[str, Any]:
    selection = read_json(root, REPORTS / "prompt4c_final_selection_freeze.json")
    recipe = selection["recipe"]
    oof = read_json(root, REPORTS / "prompt4c_oof_manifest.json")
    ledger = load_ledger(root)
    model_manifest = read_json(root, REPORTS / "prompt4c_final_model_manifest.json")
    reload = read_json(root, REPORTS / "prompt4c_bundle_reload.json")
    blocka = read_json(root, REPORTS / "prompt4c_blocka_package_eligibility.json")
    protocol = read_json(root, REPORTS / "prompt4c_prompt5_protocol.json")["protocol"]
    stage3_reproduction = read_json(root, REPORTS / "prompt4c_stage3_recipe_reproduction.json")
    global_reproduction = read_json(root, REPORTS / "prompt4c_global_recipe_reproduction.json")
    essential = {
        "prompt4c_final_selection_freeze.json": file_sha256(root / REPORTS / "prompt4c_final_selection_freeze.json"),
        "prompt4c_oof_manifest.json": file_sha256(root / REPORTS / "prompt4c_oof_manifest.json"),
        "prompt4c_final_refit_ledger.json": file_sha256(root / REPORTS / "prompt4c_final_refit_ledger.json"),
        "prompt4c_final_model_manifest.json": file_sha256(root / REPORTS / "prompt4c_final_model_manifest.json"),
        "prompt4c_bundle_reload.json": file_sha256(root / REPORTS / "prompt4c_bundle_reload.json"),
        "prompt4c_prompt5_protocol.json": file_sha256(root / REPORTS / "prompt4c_prompt5_protocol.json"),
        NOTEBOOK.as_posix(): file_sha256(root / NOTEBOOK),
    }
    candidate = {
        "status": "READY_FOR_INDEPENDENT_REVIEW",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "project_identity": {
            "project_root": str(root),
            "development_path": "outputs/data/development.parquet",
            "development_sha256": DEVELOPMENT_SHA256,
            "development_rows": 500_000,
            "feature_contract": FEATURE_CONTRACT,
            "feature_count": len(features),
            "features": features,
            "target_name": TARGET,
            "target_unit": "thousand USD",
        },
        "final_selection": {
            "final_primary_id": "final_primary_stage3_500k",
            "historical_recipe_id": PRIMARY_ID,
            "status": "SELECTED_AND_FROZEN",
            "selection_philosophy": selection["selection_philosophy"],
            "historical_development_rationale": selection["selection_rationale"],
            "historical_mae_challenger": BLOCKA_ID,
            "simple_comparator": "final_global_500k",
            "mape_wape_affect_selection": False,
        },
        "primary_recipe": recipe,
        "final_refit": {
            "oof_fold_rule": selection["oof_rule"],
            "oof_fold_digest": oof["split"]["fold_assignment_digest"],
            "oof_prediction_path": oof["path"],
            "oof_prediction_sha256": oof["sha256"],
            "oof_rows": oof["rows"],
            "zero_self_fit_rows": oof["zero_self_fit_rows"],
            "completed_roles": ledger["completed_roles"],
            "completed_role_count": ledger["completed_role_count"],
            "physical_attempt_count": ledger["physical_attempt_count"],
            "technical_retry_count": ledger["technical_retry_count"],
            "scientific_candidate_searches": 0,
            "model_hashes": {item["role"]: item["model_sha256"] for item in model_manifest["artifacts"]},
            "final_tail_rows": read_json(root, REPORTS / "prompt4c_refit_summary.json")["final_tail_rows"],
            "meta_gate_iterations": META_ITERATIONS,
            "residual_specialist_iterations": RESIDUAL_ITERATIONS,
        },
        "primary_bundle": {
            "path": bundles["primary_path"].relative_to(root).as_posix(),
            "sha256": bundles["primary_hash"],
            "clean_reload_maximum_difference": next(item["maximum_absolute_difference"] for item in reload["bundles"] if item["bundle"] == "primary"),
        },
        "global_comparator": {
            "path": bundles["global_path"].relative_to(root).as_posix(),
            "sha256": bundles["global_hash"],
            "weights": {"catboost": 0.60, "lightgbm": 0.20, "xgboost": 0.20},
            "clean_reload_maximum_difference": next(item["maximum_absolute_difference"] for item in reload["bundles"] if item["bundle"] == "global"),
        },
        "historical_blocka": {
            "iid_eligible": blocka["historical_blocka_iid_eligible"],
            "bundle_hash": None,
            "training_population": "historical 400,000 Train rows",
            "not_eligible_to_replace_primary_after_iid": True,
            "exclusion_reason": None if blocka["historical_blocka_iid_eligible"] else blocka["eligibility_reason"],
        },
        "historical_reproduction": {
            "global_maximum_difference": global_reproduction["maximum_absolute_prediction_difference"],
            "stage3_maximum_difference": stage3_reproduction["component_maximum_absolute_differences"]["final_prediction"],
        },
        "iid_protocol": {
            **protocol,
            "development_frozen_target_cutpoints": cutpoints,
        },
        "artifact_hashes": essential,
        "safety": {
            "raw_access_count": 0,
            "iid_feature_access_count": 0,
            "iid_target_access_count": 0,
            "iid_prediction_count": 0,
            "prompt5_executed": False,
        },
    }
    atomic_json(root, REPORTS / "prompt4c_pre_iid_freeze_candidate.json", candidate)
    return candidate


def prepare_phase(root: Path) -> dict[str, Any]:
    total = time.perf_counter()
    started = time.perf_counter()
    handoff = validate_handoff(root)
    preflight_seconds = time.perf_counter() - started
    started = time.perf_counter()
    recipe = recover_recipe(root, handoff["features"])
    recipe_seconds = time.perf_counter() - started
    started = time.perf_counter()
    global_report, stage3_report = historical_reproduction(root, handoff["features"], recipe)
    blocka = blocka_eligibility(root)
    reproduction_seconds = time.perf_counter() - started
    freeze = create_selection_freeze(root, handoff, recipe, blocka)
    started = time.perf_counter()
    metrics, deciles, cutpoints = historical_metric_extension(root)
    figures = build_figures(root, metrics, deciles)
    reporting_seconds = time.perf_counter() - started
    plan = create_refit_plan(root, handoff["features"], recipe)
    protocol = {
        "status": "FROZEN",
        "created_at_utc": utc_now(),
        "protocol": prompt5_protocol(bool(blocka["historical_blocka_iid_eligible"])),
        "development_frozen_target_cutpoints": cutpoints,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "iid_prediction_count": 0,
        "prompt5_executed": False,
    }
    atomic_json(root, REPORTS / "prompt4c_prompt5_protocol.json", protocol)
    update_runtime(
        root,
        preflight=preflight_seconds,
        state_reconciliation=0.0,
        stage3_recipe_reconstruction=recipe_seconds,
        historical_prediction_reproduction=reproduction_seconds,
        mape_wape_reporting_extension=reporting_seconds,
    )
    return {
        "status": "PASS",
        "handoff": handoff["status"],
        "global_reproduction": global_report["status"],
        "stage3_reproduction": stage3_report["status"],
        "blocka_iid_eligible": blocka["historical_blocka_iid_eligible"],
        "selection_freeze": freeze["status"],
        "refit_roles": plan["role_count"],
        "figure_count": figures["figure_count"],
        "elapsed_seconds": time.perf_counter() - total,
    }


def fit_phase(root: Path) -> dict[str, Any]:
    handoff = read_json(root, REPORTS / "prompt4c_handoff_validation.json")
    started = time.perf_counter()
    refit = run_final_refits(root, handoff["features"])
    elapsed = time.perf_counter() - started
    ledger = load_ledger(root)
    role_runtime = {role: sum(float(entry.get("runtime_seconds", 0.0)) for entry in ledger["attempts"] if entry["role"] == role) for role in FINAL_ROLES}
    update_runtime(
        root,
        oof_global_fits=sum(role_runtime[role] for role in FINAL_ROLES[:6]),
        oof_prediction_construction=max(0.0, elapsed - sum(role_runtime.values())),
        final_catboost_fit=role_runtime["prompt4c_full_catboost_500k"],
        final_lightgbm_fit=role_runtime["prompt4c_full_lightgbm_500k"],
        final_xgboost_fit=role_runtime["prompt4c_full_xgboost_500k"],
        final_meta_gate_fit=role_runtime["prompt4c_meta_gate_500k"],
        final_residual_specialist_fit=role_runtime["prompt4c_residual_specialist_500k"],
    )
    summary = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "completed_roles": len(ledger["completed_roles"]),
        "physical_attempts": ledger["physical_attempt_count"],
        "technical_retries": ledger["technical_retry_count"],
        "scientific_candidate_searches": 0,
        "oof_rows": len(refit["oof"]),
        "zero_self_fit_rows": int(refit["oof"]["self_fit"].sum()),
        "final_tail_rows": refit["final_tail_rows"],
        "meta_gate_iterations": META_ITERATIONS,
        "residual_specialist_iterations": RESIDUAL_ITERATIONS,
        "elapsed_seconds": elapsed,
    }
    atomic_json(root, REPORTS / "prompt4c_refit_summary.json", summary)
    return summary


def load_refit_state(root: Path, features: list[str]) -> dict[str, Any]:
    development = development_frame(root, features, include_target=True).copy()
    oof = pd.read_parquet(root / PREDICTIONS / "oof_global_500k.parquet")
    manifests = {
        role: read_json(root, MODELS / "components" / role / "manifest.json") for role in FINAL_ROLES
    }
    full_bundles = {
        family: joblib.load(root / MODELS / "components" / f"prompt4c_full_{family}_500k" / "bundle.joblib")
        for family in ("catboost", "lightgbm", "xgboost")
    }
    return {
        "development": development,
        "oof": oof,
        "oof_manifest": read_json(root, REPORTS / "prompt4c_oof_manifest.json"),
        "full_bundles": full_bundles,
        "meta_bundle": joblib.load(root / MODELS / "components/prompt4c_meta_gate_500k/bundle.joblib"),
        "residual_bundle": joblib.load(root / MODELS / "components/prompt4c_residual_specialist_500k/bundle.joblib"),
        "manifests": manifests,
    }


def finalize_phase(root: Path) -> dict[str, Any]:
    handoff = read_json(root, REPORTS / "prompt4c_handoff_validation.json")
    refit = load_refit_state(root, handoff["features"])
    bundles = package_final_bundles(root, refit, handoff["features"])
    create_final_manifests(root, bundles)
    notebook = build_and_execute_notebook(root)
    protocol = read_json(root, REPORTS / "prompt4c_prompt5_protocol.json")
    candidate = create_freeze_candidate(root, handoff["features"], bundles, protocol["development_frozen_target_cutpoints"])
    return {
        "status": "PASS",
        "primary_bundle_sha256": bundles["primary_hash"],
        "global_bundle_sha256": bundles["global_hash"],
        "notebook": notebook["status"],
        "freeze_candidate": candidate["status"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit", "finalize", "all", "blocka-worker", "clean-reload-worker"))
    parser.add_argument("--root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--sample", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = root_path(args.root)
    if args.command == "blocka-worker":
        blocka_worker(root, Path(args.output).resolve())
        return
    if args.command == "clean-reload-worker":
        clean_reload_worker(root, Path(args.bundle).resolve(), Path(args.sample).resolve(), Path(args.output).resolve())
        return
    result: dict[str, Any] = {}
    if args.command in {"prepare", "all"}:
        result["prepare"] = prepare_phase(root)
    if args.command in {"fit", "all"}:
        result["fit"] = fit_phase(root)
    if args.command in {"finalize", "all"}:
        result["finalize"] = finalize_phase(root)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
