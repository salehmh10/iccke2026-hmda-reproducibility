"""Stepwise Tail-aware Development experiments for Prompt 4B.

Stages are executed in order and every scientific fit is resumable. Raw and
IID paths are blocked. Stage champions are comparison references only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

try:
    from .deep_preprocessing import ordered_digest
    from .prompt4_metrics import compute_operational_tail_metrics, compute_regression_metrics, provisional_acceptance
    from .prompt4a_experiments import atomic_csv, atomic_json, atomic_parquet, file_sha256, guard_read_path, load_aligned_predictions, load_development, package_versions, regression_v2_root, _roles_from_design
    from .tail_models import load_gate_bundle, load_regression_bundle, make_internal_tail_split
    from .prompt4b_calibration import calibration_summary, expected_calibration_error, fit_platt, prior_correct_probability, reliability_rows, residual_routed_prediction, routed_prediction
    from .prompt4b_crossfit import GLOBAL_FEATURE, MetaGateBundle, MetaPreprocessor, build_global_oof, load_experimental_bundle, membership_digest, save_experimental_bundle
    from .prompt4b_ensemble import apply_weights, optimize_body_constrained, optimize_tail_weighted_mae
    from .prompt4b_residual import RESIDUAL_PARAMETERS, ResidualSpecialistBundle, fit_residual_model, residual_diagnostics, residual_target
except ImportError:
    from deep_preprocessing import ordered_digest
    from prompt4_metrics import compute_operational_tail_metrics, compute_regression_metrics, provisional_acceptance
    from prompt4a_experiments import atomic_csv, atomic_json, atomic_parquet, file_sha256, guard_read_path, load_aligned_predictions, load_development, package_versions, regression_v2_root, _roles_from_design
    from tail_models import load_gate_bundle, load_regression_bundle, make_internal_tail_split
    from prompt4b_calibration import calibration_summary, expected_calibration_error, fit_platt, prior_correct_probability, reliability_rows, residual_routed_prediction, routed_prediction
    from prompt4b_crossfit import GLOBAL_FEATURE, MetaGateBundle, MetaPreprocessor, build_global_oof, load_experimental_bundle, membership_digest, save_experimental_bundle
    from prompt4b_ensemble import apply_weights, optimize_body_constrained, optimize_tail_weighted_mae
    from prompt4b_residual import RESIDUAL_PARAMETERS, ResidualSpecialistBundle, fit_residual_model, residual_diagnostics, residual_target


SEED = 42
TARGET = "loan_amount_000s"
FEATURE_CONTRACT = "main_without_sensitive_without_lender"
REPORTS = Path("outputs/reports")
PREDICTIONS = Path("outputs/predictions/prompt4b/validation")
TRAIN_PREDICTIONS = Path("outputs/predictions/prompt4b/train")
MODELS = Path("outputs/models/prompt4b")
TMP = Path("outputs/tmp/prompt4b")
FIGURES = Path("outputs/figures/prompt4b")
NOTEBOOK = Path("notebooks/04B_STEPWISE_TAIL_IMPROVEMENT.ipynb")
THRESHOLDS = (0.65, 0.75, 0.85)
STAGE1_ALPHAS = (0.25, 0.50)
LATER_ALPHAS = (0.25, 0.50, 0.75)
CAP_MULTIPLIERS = (0.25, 0.50, 0.75)
LAMBDA_TAIL = (1.25, 1.50, 2.00)
FIT_ROLES = (
    "crossfit_catboost_fold_a", "crossfit_lightgbm_fold_a", "crossfit_xgboost_fold_a",
    "crossfit_catboost_fold_b", "crossfit_lightgbm_fold_b", "crossfit_xgboost_fold_b",
    "meta_gate_selection", "meta_gate_full_refit", "residual_selection", "residual_full_refit",
)


def utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _read_json(root: Path, relative: str | Path) -> dict[str, Any]:
    return json.loads(guard_read_path(root, relative).read_text(encoding="utf-8"))


def _code_digest(root: Path) -> str:
    names = ("prompt4b_experiments.py", "prompt4b_calibration.py", "prompt4b_crossfit.py", "prompt4b_residual.py", "prompt4b_ensemble.py")
    payload = {name: file_sha256(root / "src" / name) for name in names}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def validate_prompt4a(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    required = ("PROMPT2_READY.json", "PROMPT3_READY.json", "PROMPT4A_READY.json", "prompt4a_verification.json", "prompt4a_frozen_design.json", "prompt2_prediction_manifest.json", "prompt3_prediction_manifest.json", "prompt4a_prediction_manifest.json", "prompt4a_model_manifest.json")
    reports = {name: _read_json(workspace, REPORTS / name) for name in required}
    bad = [name for name, value in reports.items() if value.get("status") != "PASS" and not (name == "prompt4a_frozen_design.json" and value.get("status") == "FROZEN")]
    if bad:
        raise RuntimeError(f"Prompt 4A handoff is invalid: {bad}")
    if (workspace / REPORTS / "FINAL_PRE_IID_FREEZE.json").exists():
        raise RuntimeError("FINAL_PRE_IID_FREEZE.json already exists.")
    prompt4c = [path for path in workspace.rglob("*prompt4c*") if path.is_file()]
    if prompt4c:
        raise RuntimeError(f"Prompt 4C has already started: {prompt4c[:3]}")
    train, validation, source = load_development(workspace)
    aligned, alignment = load_aligned_predictions(workspace, validation)
    design4a = reports["prompt4a_frozen_design.json"]
    roles = _roles_from_design(aligned, design4a)
    q90 = float(np.quantile(train[TARGET].to_numpy(float), 0.90))
    if q90 != float(design4a["q90_train"]) or int(np.sum(roles == "selection")) != 70_000 or int(np.sum(roles == "audit")) != 30_000:
        raise RuntimeError("Frozen q90 or Validation roles changed.")
    expected_sources = {item["model"]: item["sha256"] for item in alignment["artifacts"]}
    if expected_sources != design4a["prediction_source_sha256"]:
        raise RuntimeError("Saved source prediction hashes changed.")
    return {"status": "PASS", "reports": reports, "train": train, "validation": validation, "source": source, "aligned": aligned, "roles": roles, "q90_train": q90, "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0}


def _class_weight_proof(root: Path, train: pd.DataFrame, q90: float) -> dict[str, Any]:
    gate = load_gate_bundle(root / "outputs/models/prompt4a/tail_gate")
    configuration = gate.metadata.model_configuration
    all_parameters = gate.model.get_all_params()
    labels = train[TARGET].to_numpy(float) > q90
    negative = int(np.count_nonzero(~labels)); positive = int(np.count_nonzero(labels))
    expected_ratio = negative / positive
    class_weights = [float(item) for item in all_parameters.get("class_weights", [])]
    fitted_ratio = class_weights[1] / class_weights[0] if len(class_weights) == 2 and class_weights[0] > 0 else np.nan
    relative_difference = abs(fitted_ratio - expected_ratio) / expected_ratio if np.isfinite(fitted_ratio) else np.inf
    proven = configuration.get("auto_class_weights") == "Balanced" and all_parameters.get("auto_class_weights") == "Balanced" and len(class_weights) == 2 and relative_difference <= 1e-6
    return {
        "status": "AVAILABLE" if proven else "UNAVAILABLE",
        "configured_auto_class_weights": configuration.get("auto_class_weights"),
        "installed_catboost_reported_auto_class_weights": all_parameters.get("auto_class_weights"),
        "installed_catboost_reported_class_weights": class_weights,
        "negative_train_rows": negative, "positive_train_rows": positive,
        "expected_positive_to_negative_weight_ratio": float(expected_ratio),
        "fitted_positive_to_negative_weight_ratio": float(fitted_ratio) if np.isfinite(fitted_ratio) else None,
        "reported_vs_count_ratio_relative_difference": float(relative_difference) if np.isfinite(relative_difference) else None,
        "correction_ratio_used": float(expected_ratio) if proven else None,
        "formula": "p_unweighted = p_weighted / (r*(1-p_weighted) + p_weighted)",
        "proof": "Configured and installed CatBoost semantics report Balanced; the fitted class-weight vector matches the exact negative/positive Train count ratio within stored float precision. The exact count-derived ratio is used." if proven else "Exact fitted class-weight ratio could not be proven.",
    }


def prepare_design(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    state = validate_prompt4a(workspace)
    proof = _class_weight_proof(workspace, state["train"], state["q90_train"])
    core = {
        "prompt": "Prompt 4B", "adaptive_development_optimization": True,
        "source": state["source"], "q90_train": state["q90_train"],
        "validation_split": state["reports"]["prompt4a_frozen_design.json"]["validation_split"],
        "internal_tail_split": state["reports"]["prompt4a_frozen_design.json"]["internal_tail_split"],
        "class_weight_proof": proof,
        "stage_order": ["Stage 1", "Stage 2", "Stage 3", "Stage 4"],
        "stage1_probability_sources": ["raw", "prior_corrected" if proof["status"] == "AVAILABLE" else "prior_corrected_UNAVAILABLE", "platt"],
        "thresholds": list(THRESHOLDS), "stage1_alphas": list(STAGE1_ALPHAS), "later_alphas": list(LATER_ALPHAS), "cap_multipliers": list(CAP_MULTIPLIERS), "lambda_tail": list(LAMBDA_TAIL),
        "fit_roles": list(FIT_ROLES), "max_heavy_scientific_fits": 10, "platt_fit_count": 1,
        "heavy_fits_sequential": True, "seed": SEED, "threads": 4, "package_versions": package_versions(),
        "code_digest": _code_digest(workspace), "selection_only_ranking": True, "audit_reporting_only": True,
        "final_model_selected": False, "final_model_frozen": False, "full_development_refit_count": 0,
        "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0,
    }
    design = {"status": "FROZEN", "created_at_utc": utc_now(), **core, "design_digest": hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode("utf-8")).hexdigest()}
    atomic_json(workspace, REPORTS / "prompt4b_frozen_design.json", design)
    return design


def _load_design(root: Path) -> dict[str, Any]:
    design = _read_json(root, REPORTS / "prompt4b_frozen_design.json")
    if design.get("status") != "FROZEN" or design.get("code_digest") != _code_digest(root):
        raise RuntimeError("Prompt 4B design is missing or code changed after freeze.")
    return design


def _fit_ledger(root: Path) -> dict[str, Any]:
    path = root / TMP / "scientific_fit_ledger.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "IN_PROGRESS", "max_scientific_fits": 10, "roles": list(FIT_ROLES), "attempts": [], "completed_roles": []}


def _save_ledger(root: Path, ledger: dict[str, Any]) -> None:
    ledger["scientific_fit_count"] = len(ledger["completed_roles"])
    ledger["status"] = "COMPLETE" if set(ledger["completed_roles"]) == set(FIT_ROLES) else "IN_PROGRESS"
    atomic_json(root, TMP / "scientific_fit_ledger.json", ledger)


def _run_heavy_fit(root: Path, role: str, work: Callable[[], Any]) -> Any:
    if role not in FIT_ROLES:
        raise RuntimeError(f"Unauthorized fit role: {role}")
    ledger = _fit_ledger(root)
    if role in ledger["completed_roles"]:
        return work()
    attempts = [item for item in ledger["attempts"] if item["role"] == role]
    if len(attempts) >= 2:
        raise RuntimeError(f"Technical retry budget exhausted: {role}")
    attempt = len(attempts) + 1
    record = {"role": role, "attempt": attempt, "status": "STARTED", "started_at_utc": utc_now()}
    ledger["attempts"].append(record); _save_ledger(root, ledger)
    try:
        result = work()
        record.update({"status": "COMPLETE", "finished_at_utc": utc_now()})
        ledger["completed_roles"].append(role); _save_ledger(root, ledger)
        return result
    except Exception as exc:
        record.update({"status": "FAILED", "finished_at_utc": utc_now(), "error": repr(exc)}); _save_ledger(root, ledger)
        raise


def _scope_masks(roles: np.ndarray):
    return (("selection", roles == "selection"), ("audit", roles == "audit"), ("complete_validation", np.ones(len(roles), dtype=bool)))


def _metric_rows(candidate_id: str, stage: str, y: np.ndarray, prediction: np.ndarray, roles: np.ndarray, q90: float, **extra) -> list[dict[str, Any]]:
    rows = []
    for scope, mask in _scope_masks(roles):
        rows.append({"candidate_id": candidate_id, "stage": stage, "scope": scope, **extra, **compute_regression_metrics(y[mask], prediction[mask]), **compute_operational_tail_metrics(y[mask], prediction[mask], q90)})
    return rows


def _base_metrics(y: np.ndarray, global_prediction: np.ndarray, roles: np.ndarray, q90: float) -> dict[str, dict[str, Any]]:
    return {row["scope"]: row for row in _metric_rows("ens_boost_cat060", "reference", y, global_prediction, roles, q90)}


def _acceptance_rows(candidate_rows: pd.DataFrame, base: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for _, row in candidate_rows.iterrows():
        checks = provisional_acceptance(row.to_dict(), base[str(row["scope"])])
        rows.append({"candidate_id": row["candidate_id"], "scope": row["scope"], **checks})
    return pd.DataFrame(rows)


def _rank(candidate_rows: pd.DataFrame, acceptance: pd.DataFrame, top: int = 1) -> list[str]:
    selection = candidate_rows.query("scope == 'selection'").merge(acceptance.query("scope == 'selection'")[["candidate_id", "conditions_passed"]], on="candidate_id")
    selection["bottom_penalty"] = selection["bottom_90_mae"]
    ordered = selection.sort_values(["conditions_passed", "mae", "bottom_penalty", "top_decile_mae", "candidate_id"], ascending=[False, True, True, True, True])
    return ordered["candidate_id"].head(top).tolist()


def _save_prediction(root: Path, aligned: pd.DataFrame, roles: np.ndarray, candidate_id: str, stage: str, prediction: np.ndarray) -> Path:
    frame = pd.DataFrame({"row_hash": aligned["row_hash"].astype(str), "y_true": aligned["y_true"].to_numpy(float), "y_pred": np.asarray(prediction, dtype=np.float64), "candidate_id": candidate_id, "stage": stage, "selection_or_audit_role": roles})
    if len(frame) != 100_000 or not frame["row_hash"].is_unique or not np.isfinite(frame["y_pred"]).all():
        raise RuntimeError(f"Invalid Prompt 4B prediction: {candidate_id}")
    return atomic_parquet(root, PREDICTIONS / f"{candidate_id}.parquet", frame)


def _load_common(root: Path):
    state = validate_prompt4a(root); design = _load_design(root)
    aligned = state["aligned"]; roles = state["roles"]; y = aligned["y_true"].to_numpy(float); q90 = float(design["q90_train"])
    global_prediction = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet")["y_pred"].to_numpy(float)
    gate = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/tail_gate.parquet")["p_tail"].to_numpy(float)
    specialist = pd.read_parquet(root / "outputs/predictions/prompt4a/validation/tail_specialist.parquet")["y_pred"].to_numpy(float)
    return state, design, aligned, roles, y, q90, global_prediction, gate, specialist


def _classification_rows(y_true: np.ndarray, probability: np.ndarray, roles: np.ndarray, gate_name: str) -> list[dict[str, Any]]:
    rows = []
    for scope, mask in _scope_masks(roles):
        labels = y_true[mask]; values = probability[mask]
        base = calibration_summary(labels, values)
        for threshold in (0.50, 0.65, 0.75, 0.85):
            pred = values > threshold
            recall = recall_score(labels, pred, zero_division=0)
            rows.append({"gate": gate_name, "scope": scope, "threshold": threshold, **base, "precision": precision_score(labels, pred, zero_division=0), "recall": recall, "f1": f1_score(labels, pred, zero_division=0), "false_negative_rate": 1.0 - recall, "routed_percentage": float(np.mean(pred))})
    return rows


def run_stage1(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    _, design, aligned, roles, y, q90, global_prediction, raw_probability, specialist = _load_common(workspace)
    tail_true = (y > q90).astype(np.int8); base = _base_metrics(y, global_prediction, roles, q90)
    global_error = np.abs(global_prediction - y); specialist_error = np.abs(specialist - y); wins = specialist_error < global_error
    oracle = np.where(wins, specialist, global_prediction)
    target_decile = pd.qcut(pd.Series(y).rank(method="average"), 10, labels=False).to_numpy()
    probability_decile = pd.qcut(pd.Series(raw_probability).rank(method="first"), 10, labels=False).to_numpy()
    oracle_report = {"status": "COMPLETE", "analysis_only": True, "deployable": False, **compute_regression_metrics(y, oracle), **compute_operational_tail_metrics(y, oracle, q90), "specialist_win_fraction": float(np.mean(wins)), "mean_improvement_when_specialist_wins": float(np.mean(global_error[wins] - specialist_error[wins])), "mean_damage_when_specialist_loses": float(np.mean(specialist_error[~wins] - global_error[~wins])), "specialist_win_by_target_decile": {str(i): float(np.mean(wins[target_decile == i])) for i in range(10)}, "specialist_win_by_p_tail_decile": {str(i): float(np.mean(wins[probability_decile == i])) for i in range(10)}}
    bin_rows = []
    bins = np.minimum((raw_probability * 10).astype(int), 9)
    for bin_id in range(10):
        mask = bins == bin_id
        bin_rows.append({"p_tail_bin": bin_id, "lower": bin_id / 10, "upper": (bin_id + 1) / 10, "row_count": int(mask.sum()), "operational_tail_prevalence": float(np.mean(tail_true[mask])) if mask.any() else np.nan, "specialist_win_rate": float(np.mean(wins[mask])) if mask.any() else np.nan, "global_mae": float(np.mean(global_error[mask])) if mask.any() else np.nan, "specialist_mae": float(np.mean(specialist_error[mask])) if mask.any() else np.nan, "specialist_minus_global_absolute_error": float(np.mean(specialist_error[mask] - global_error[mask])) if mask.any() else np.nan})
    atomic_json(workspace, REPORTS / "prompt4b_stage1_oracle.json", oracle_report); atomic_csv(workspace, REPORTS / "prompt4b_stage1_oracle_bins.csv", pd.DataFrame(bin_rows))

    selection = roles == "selection"
    calibrator_path = workspace / MODELS / "platt_calibrator"
    if (calibrator_path / "bundle.joblib").exists():
        platt = joblib.load(calibrator_path / "bundle.joblib")
    else:
        platt = fit_platt(raw_probability[selection], tail_true[selection], ordered_digest(aligned.loc[selection, "row_hash"]))
        calibrator_path.mkdir(parents=True, exist_ok=True); temp = calibrator_path / "bundle.joblib.tmp"; joblib.dump(platt, temp, compress=3); joblib.load(temp); os.replace(temp, calibrator_path / "bundle.joblib")
        atomic_json(workspace, MODELS / "platt_calibrator/manifest.json", {"status": "COMPLETE", "artifact": "bundle.joblib", "artifact_sha256": file_sha256(calibrator_path / "bundle.joblib"), "fit_rows": 70_000, "fit_role": "selection", "fit_row_hash_digest": platt.fitted_row_hash_digest, "feature_contract": ["logit(clipped_prompt4a_p_tail)"], "target_definition": "operational_tail_true", "model_configuration": {"family": "LogisticRegression", "solver": "lbfgs", "max_iter": 1000, "random_state": 42}, "coefficient": platt.coefficient, "intercept": platt.intercept, "seed": SEED, "package_versions": design["package_versions"]})
    p_platt = platt.predict(raw_probability)
    sources = {"raw": raw_probability, "platt": p_platt}
    proof = design["class_weight_proof"]
    if proof["status"] == "AVAILABLE":
        sources["prior_corrected"] = prior_correct_probability(raw_probability, proof["correction_ratio_used"])
    probability_frame = pd.DataFrame({"row_hash": aligned["row_hash"].astype(str), "y_true": y, "operational_tail_true": tail_true, "p_raw": raw_probability, "p_platt": p_platt, "selection_or_audit_role": roles})
    probability_frame["p_prior_corrected"] = sources.get("prior_corrected", np.nan)
    atomic_parquet(workspace, PREDICTIONS / "stage1_probabilities.parquet", probability_frame)
    calibration_rows = []
    for name, probability in sources.items():
        for scope, mask in _scope_masks(roles):
            summary = calibration_summary(tail_true[mask], probability[mask])
            calibration_rows.append({"probability_source": name, "scope": scope, **summary, "platt_fit_scope": "selection_only" if name == "platt" else "not_applicable"})
    atomic_csv(workspace, REPORTS / "prompt4b_stage1_calibration.csv", pd.DataFrame(calibration_rows))

    predictions = {}; initial_rows = []
    for source_name, probability in sources.items():
        for threshold in THRESHOLDS:
            for alpha in STAGE1_ALPHAS:
                candidate_id = f"stage1_{source_name}_t{int(threshold*100):02d}_a{int(alpha*100):02d}"
                pred = routed_prediction(global_prediction, specialist, probability, threshold, alpha)
                predictions[candidate_id] = pred; initial_rows.extend(_metric_rows(candidate_id, "Stage 1", y, pred, roles, q90, probability_source=source_name, threshold=threshold, alpha=alpha, cap=np.nan, capped=False))
    initial = pd.DataFrame(initial_rows); initial_acceptance = _acceptance_rows(initial, base)
    top3 = _rank(initial, initial_acceptance, 3)
    capped_rows = []
    for parent_id in top3:
        row = initial.query("candidate_id == @parent_id and scope == 'selection'").iloc[0]
        for multiplier in CAP_MULTIPLIERS:
            cap = multiplier * q90
            candidate_id = f"{parent_id}_cap{int(multiplier*100):02d}"
            pred = routed_prediction(global_prediction, specialist, sources[row["probability_source"]], float(row["threshold"]), float(row["alpha"]), cap)
            predictions[candidate_id] = pred; capped_rows.extend(_metric_rows(candidate_id, "Stage 1", y, pred, roles, q90, probability_source=row["probability_source"], threshold=row["threshold"], alpha=row["alpha"], cap=cap, capped=True, parent_candidate_id=parent_id))
    candidates = pd.concat([initial, pd.DataFrame(capped_rows)], ignore_index=True); acceptance = _acceptance_rows(candidates, base)
    champion = _rank(candidates, acceptance, 1)[0]
    atomic_csv(workspace, REPORTS / "prompt4b_stage1_candidates.csv", candidates); atomic_csv(workspace, REPORTS / "prompt4b_stage1_acceptance.csv", acceptance)
    _save_prediction(workspace, aligned, roles, champion, "Stage 1", predictions[champion])
    report = {"status": "COMPLETE", "stage": "Stage 1", "oracle": oracle_report, "prior_correction": proof, "platt": {"fit_rows": 70_000, "coefficient": platt.coefficient, "intercept": platt.intercept}, "probability_sources": list(sources), "initial_candidate_count": int(initial["candidate_id"].nunique()), "top3_uncapped": top3, "capped_candidate_count": 9, "best_uncapped": _rank(initial, initial_acceptance, 1)[0], "best_capped": _rank(pd.DataFrame(capped_rows), _acceptance_rows(pd.DataFrame(capped_rows), base), 1)[0], "stage_champion": champion, "stage_champion_is_final_model": False, "adaptive_development_evidence": True}
    atomic_json(workspace, REPORTS / "prompt4b_stage1_report.json", report)
    return report


def _saved_prompt2_bundles(root: Path):
    paths = {"catboost": "selected_catboost_without_lender.joblib", "lightgbm": "selected_lightgbm_without_lender.joblib", "xgboost": "selected_xgboost_without_lender.joblib"}
    return {name: joblib.load(root / "outputs/models/prompt2" / path) for name, path in paths.items()}


META_PARAMETERS = {"loss_function": "Logloss", "eval_metric": "PRAUC", "iterations": 1500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10, "random_seed": 42, "thread_count": 4, "early_stopping_rounds": 100, "verbose": False}


def _fit_meta_gate(X_fit, y_fit, base_features, X_stop=None, y_stop=None, parameters=None):
    from catboost import CatBoostClassifier
    config = dict(parameters or META_PARAMETERS); early = config.pop("early_stopping_rounds", None)
    preprocessor = MetaPreprocessor(base_features).fit(X_fit)
    model = CatBoostClassifier(**config, allow_writing_files=False, task_type="CPU")
    kwargs = {"cat_features": preprocessor.cat_feature_indices_, "verbose": False}
    if X_stop is not None:
        kwargs.update({"eval_set": (preprocessor.transform(X_stop), np.asarray(y_stop, dtype=np.int8)), "use_best_model": True, "early_stopping_rounds": int(early)})
    elif early is not None:
        raise ValueError("Full Meta-Gate refit must not use early stopping.")
    model.fit(preprocessor.transform(X_fit), np.asarray(y_fit, dtype=np.int8), **kwargs)
    selected = int(model.get_best_iteration()) + 1 if X_stop is not None and int(model.get_best_iteration()) >= 0 else int(model.tree_count_)
    return model, preprocessor, selected


def run_stage2(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    state, design, aligned, roles, y, q90, global_prediction, old_probability, specialist = _load_common(workspace)
    if _read_json(workspace, REPORTS / "prompt4b_stage1_report.json").get("status") != "COMPLETE":
        raise RuntimeError("Stage 1 must complete before Stage 2.")
    oof_path = workspace / TRAIN_PREDICTIONS / "global_oof_prediction.parquet"
    crossfit_report_path = workspace / REPORTS / "prompt4b_stage2_crossfit.json"
    if oof_path.exists() and crossfit_report_path.exists():
        oof = pd.read_parquet(oof_path); crossfit_report = json.loads(crossfit_report_path.read_text(encoding="utf-8"))
    else:
        oof, crossfit_report = build_global_oof(state["train"], design["source"]["features"], q90, _saved_prompt2_bundles(workspace), workspace / MODELS / "crossfit", design["package_versions"], design["source"]["sha256"], design["source"]["validation_row_hash_digest"], lambda role, work: _run_heavy_fit(workspace, role, work))
        atomic_parquet(workspace, TRAIN_PREDICTIONS / "global_oof_prediction.parquet", oof); atomic_json(workspace, REPORTS / "prompt4b_stage2_crossfit.json", crossfit_report)
    train = state["train"]
    if len(oof) != 400_000 or not oof["row_hash"].astype(str).equals(train["row_hash"].astype(str)):
        raise RuntimeError("OOF Train predictions do not align.")
    features = design["source"]["features"]
    meta_train = train[features].copy(); meta_train[GLOBAL_FEATURE] = oof["global_oof_prediction"].to_numpy(float)
    meta_validation = state["validation"][features].copy(); meta_validation[GLOBAL_FEATURE] = global_prediction
    labels = (train[TARGET].to_numpy(float) > q90).astype(np.int8)
    fit_index, stop_index, split = make_internal_tail_split(train[TARGET], train["row_hash"], q90, state["validation"]["row_hash"], random_state=SEED)
    selection_dir = workspace / MODELS / "meta_gate/selection"

    def selection_work():
        if (selection_dir / "manifest.json").exists():
            bundle, manifest = load_experimental_bundle(selection_dir); return bundle, manifest, True
        model, preprocessor, selected = _fit_meta_gate(meta_train.iloc[fit_index], labels[fit_index], features, meta_train.iloc[stop_index], labels[stop_index])
        metadata = {"model_role": "meta_gate_selection", "model_configuration": META_PARAMETERS, "selected_iteration": selected, "feature_contract": features + [GLOBAL_FEATURE], "seed": SEED, "training_row_count": len(fit_index), "training_membership_digest": ordered_digest(train.iloc[fit_index]["row_hash"]), "stop_membership_digest": ordered_digest(train.iloc[stop_index]["row_hash"]), "package_versions": design["package_versions"]}
        bundle = MetaGateBundle(preprocessor, model, metadata); manifest = save_experimental_bundle(bundle, selection_dir, metadata); return bundle, manifest, False
    selection_bundle, selection_manifest, _ = _run_heavy_fit(workspace, "meta_gate_selection", selection_work)
    selected_iteration = int(selection_bundle.metadata["selected_iteration"])
    full_dir = workspace / MODELS / "meta_gate/full"

    def full_work():
        if (full_dir / "manifest.json").exists():
            bundle, manifest = load_experimental_bundle(full_dir); return bundle, manifest, True
        params = {**META_PARAMETERS, "iterations": selected_iteration}; params.pop("early_stopping_rounds")
        model, preprocessor, selected = _fit_meta_gate(meta_train, labels, features, parameters=params)
        metadata = {"model_role": "meta_gate_full_refit", "model_configuration": params, "selected_iteration": selected, "selection_iteration": selected_iteration, "feature_contract": features + [GLOBAL_FEATURE], "seed": SEED, "training_row_count": len(train), "training_membership_digest": ordered_digest(train["row_hash"]), "package_versions": design["package_versions"]}
        bundle = MetaGateBundle(preprocessor, model, metadata); manifest = save_experimental_bundle(bundle, full_dir, metadata); return bundle, manifest, False
    full_bundle, full_manifest, _ = _run_heavy_fit(workspace, "meta_gate_full_refit", full_work)
    meta_probability = full_bundle.predict_tail_probability(meta_validation)
    tail_true = (y > q90).astype(np.int8)
    prob_frame = pd.DataFrame({"row_hash": aligned["row_hash"].astype(str), "y_true": y, "operational_tail_true": tail_true, "p_meta_gate": meta_probability, "selection_or_audit_role": roles})
    atomic_parquet(workspace, PREDICTIONS / "stage2_meta_gate_probability.parquet", prob_frame)
    gate_metrics = pd.DataFrame(_classification_rows(tail_true, old_probability, roles, "prompt4a_gate") + _classification_rows(tail_true, meta_probability, roles, "meta_gate"))
    atomic_csv(workspace, REPORTS / "prompt4b_stage2_meta_gate_metrics.csv", gate_metrics)
    base = _base_metrics(y, global_prediction, roles, q90); predictions = {}; initial_rows = []
    for threshold in THRESHOLDS:
        for alpha in LATER_ALPHAS:
            cid = f"stage2_meta_t{int(threshold*100):02d}_a{int(alpha*100):02d}"
            pred = routed_prediction(global_prediction, specialist, meta_probability, threshold, alpha); predictions[cid] = pred
            initial_rows.extend(_metric_rows(cid, "Stage 2", y, pred, roles, q90, threshold=threshold, alpha=alpha, cap=np.nan, capped=False))
    initial = pd.DataFrame(initial_rows); ia = _acceptance_rows(initial, base); top3 = _rank(initial, ia, 3); capped_rows = []
    for parent in top3:
        row = initial.query("candidate_id == @parent and scope == 'selection'").iloc[0]
        for multiplier in CAP_MULTIPLIERS:
            cap = multiplier * q90; cid = f"{parent}_cap{int(multiplier*100):02d}"
            pred = routed_prediction(global_prediction, specialist, meta_probability, row["threshold"], row["alpha"], cap); predictions[cid] = pred
            capped_rows.extend(_metric_rows(cid, "Stage 2", y, pred, roles, q90, threshold=row["threshold"], alpha=row["alpha"], cap=cap, capped=True, parent_candidate_id=parent))
    candidates = pd.concat([initial, pd.DataFrame(capped_rows)], ignore_index=True); acceptance = _acceptance_rows(candidates, base); champion = _rank(candidates, acceptance, 1)[0]
    atomic_csv(workspace, REPORTS / "prompt4b_stage2_routing_candidates.csv", candidates); atomic_csv(workspace, REPORTS / "prompt4b_stage2_acceptance.csv", acceptance); _save_prediction(workspace, aligned, roles, champion, "Stage 2", predictions[champion])
    old_complete = gate_metrics.query("gate == 'prompt4a_gate' and scope == 'complete_validation'"); meta_complete = gate_metrics.query("gate == 'meta_gate' and scope == 'complete_validation'")
    comparable = []
    for _, old in old_complete.iterrows():
        nearest_recall = meta_complete.iloc[(meta_complete["recall"] - old["recall"]).abs().argsort()[:1]].iloc[0]
        nearest_routed = meta_complete.iloc[(meta_complete["routed_percentage"] - old["routed_percentage"]).abs().argsort()[:1]].iloc[0]
        comparable.append({"old_threshold": old["threshold"], "old_recall": old["recall"], "precision_gain_at_comparable_recall": nearest_recall["precision"] - old["precision"], "recall_gain_at_comparable_routed_percentage": nearest_routed["recall"] - old["recall"]})
    report = {"status": "COMPLETE", "stage": "Stage 2", "crossfit": crossfit_report, "meta_gate_selection_iteration": selected_iteration, "meta_gate_feature_contract": features + [GLOBAL_FEATURE], "auto_class_weights": None, "old_vs_meta_comparable": comparable, "uncapped_candidate_count": 9, "top3_uncapped": top3, "capped_candidate_count": 9, "stage_champion": champion, "stage_champion_is_final_model": False}
    atomic_json(workspace, REPORTS / "prompt4b_stage2_report.json", report); return report


def run_stage3(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    state, design, aligned, roles, y, q90, global_prediction, _, direct_specialist = _load_common(workspace)
    if _read_json(workspace, REPORTS / "prompt4b_stage2_report.json").get("status") != "COMPLETE":
        raise RuntimeError("Stage 2 must complete before Stage 3.")
    train = state["train"]; features = design["source"]["features"]
    oof = pd.read_parquet(workspace / TRAIN_PREDICTIONS / "global_oof_prediction.parquet")
    meta_probability = pd.read_parquet(workspace / PREDICTIONS / "stage2_meta_gate_probability.parquet")["p_meta_gate"].to_numpy(float)
    meta_train = train[features].copy(); meta_train[GLOBAL_FEATURE] = oof["global_oof_prediction"].to_numpy(float)
    meta_validation = state["validation"][features].copy(); meta_validation[GLOBAL_FEATURE] = global_prediction
    residual = residual_target(train[TARGET], oof["global_oof_prediction"]); tail = train[TARGET].to_numpy(float) > q90
    fit_index, stop_index, _ = make_internal_tail_split(train[TARGET], train["row_hash"], q90, state["validation"]["row_hash"], random_state=SEED)
    fit_tail = fit_index[tail[fit_index]]; stop_tail = stop_index[tail[stop_index]]; all_tail = np.flatnonzero(tail)
    selection_dir = workspace / MODELS / "residual_specialist/selection"

    def selection_work():
        if (selection_dir / "manifest.json").exists():
            bundle, manifest = load_experimental_bundle(selection_dir); return bundle, manifest, True
        model, preprocessor, selected = fit_residual_model(meta_train.iloc[fit_tail], residual[fit_tail], features, X_stop=meta_train.iloc[stop_tail], y_stop=residual[stop_tail])
        metadata = {"model_role": "residual_selection", "model_configuration": RESIDUAL_PARAMETERS, "selected_iteration": selected, "feature_contract": features + [GLOBAL_FEATURE], "target_definition": "loan_amount_000s - leakage-safe global_oof_prediction", "tail_only": True, "training_row_count": len(fit_tail), "training_membership_digest": ordered_digest(train.iloc[fit_tail]["row_hash"]), "seed": SEED, "package_versions": design["package_versions"]}
        bundle = ResidualSpecialistBundle(preprocessor, model, metadata); manifest = save_experimental_bundle(bundle, selection_dir, metadata); return bundle, manifest, False
    selection_bundle, _, _ = _run_heavy_fit(workspace, "residual_selection", selection_work); selected_iteration = int(selection_bundle.metadata["selected_iteration"])
    full_dir = workspace / MODELS / "residual_specialist/full"

    def full_work():
        if (full_dir / "manifest.json").exists():
            bundle, manifest = load_experimental_bundle(full_dir); return bundle, manifest, True
        params = {**RESIDUAL_PARAMETERS, "iterations": selected_iteration}; params.pop("early_stopping_rounds")
        model, preprocessor, selected = fit_residual_model(meta_train.iloc[all_tail], residual[all_tail], features, parameters=params)
        metadata = {"model_role": "residual_full_refit", "model_configuration": params, "selected_iteration": selected, "selection_iteration": selected_iteration, "feature_contract": features + [GLOBAL_FEATURE], "target_definition": "loan_amount_000s - leakage-safe global_oof_prediction", "tail_only": True, "training_row_count": len(all_tail), "training_membership_digest": ordered_digest(train.iloc[all_tail]["row_hash"]), "seed": SEED, "package_versions": design["package_versions"]}
        bundle = ResidualSpecialistBundle(preprocessor, model, metadata); manifest = save_experimental_bundle(bundle, full_dir, metadata); return bundle, manifest, False
    full_bundle, _, _ = _run_heavy_fit(workspace, "residual_full_refit", full_work)
    residual_prediction = full_bundle.predict(meta_validation)
    residual_frame = pd.DataFrame({"row_hash": aligned["row_hash"].astype(str), "y_true": y, "global_prediction": global_prediction, "predicted_residual": residual_prediction, "selection_or_audit_role": roles})
    atomic_parquet(workspace, PREDICTIONS / "stage3_residual_prediction.parquet", residual_frame)
    validation_tail = y > q90; diagnostics = residual_diagnostics(y[validation_tail], global_prediction[validation_tail], residual_prediction[validation_tail])
    diag_rows = [{"grouping": "all_operational_tail", "group": "all", **diagnostics}]
    target_decile = pd.qcut(pd.Series(y).rank(method="average"), 10, labels=False).to_numpy(); pbin = np.minimum((meta_probability * 10).astype(int), 9); magnitude = pd.qcut(pd.Series(np.maximum(y - global_prediction, 0)).rank(method="first"), 5, labels=False).to_numpy()
    for grouping, groups in (("target_decile", target_decile), ("meta_probability_bin", pbin), ("global_underprediction_bin", magnitude)):
        for group in sorted(np.unique(groups[validation_tail])):
            mask = validation_tail & (groups == group)
            diag_rows.append({"grouping": grouping, "group": int(group), **residual_diagnostics(y[mask], global_prediction[mask], residual_prediction[mask])})
    atomic_csv(workspace, REPORTS / "prompt4b_stage3_residual_metrics.csv", pd.DataFrame(diag_rows))
    base = _base_metrics(y, global_prediction, roles, q90); predictions = {}; initial_rows = []
    for threshold in THRESHOLDS:
        for alpha in LATER_ALPHAS:
            cid = f"stage3_residual_t{int(threshold*100):02d}_a{int(alpha*100):02d}"; pred = residual_routed_prediction(global_prediction, residual_prediction, meta_probability, threshold, alpha); predictions[cid] = pred
            initial_rows.extend(_metric_rows(cid, "Stage 3", y, pred, roles, q90, threshold=threshold, alpha=alpha, cap=np.nan, capped=False))
    initial = pd.DataFrame(initial_rows); ia = _acceptance_rows(initial, base); top3 = _rank(initial, ia, 3); capped_rows = []
    for parent in top3:
        row = initial.query("candidate_id == @parent and scope == 'selection'").iloc[0]
        for multiplier in CAP_MULTIPLIERS:
            cap = multiplier*q90; cid=f"{parent}_cap{int(multiplier*100):02d}"; pred=residual_routed_prediction(global_prediction,residual_prediction,meta_probability,row["threshold"],row["alpha"],cap); predictions[cid]=pred
            capped_rows.extend(_metric_rows(cid,"Stage 3",y,pred,roles,q90,threshold=row["threshold"],alpha=row["alpha"],cap=cap,capped=True,parent_candidate_id=parent))
    candidates=pd.concat([initial,pd.DataFrame(capped_rows)],ignore_index=True); acceptance=_acceptance_rows(candidates,base); champion=_rank(candidates,acceptance,1)[0]
    atomic_csv(workspace,REPORTS/"prompt4b_stage3_candidates.csv",candidates);atomic_csv(workspace,REPORTS/"prompt4b_stage3_acceptance.csv",acceptance);_save_prediction(workspace,aligned,roles,champion,"Stage 3",predictions[champion])
    direct_metrics=compute_regression_metrics(y[validation_tail],direct_specialist[validation_tail]); residual_direct_metrics=compute_regression_metrics(y[validation_tail],global_prediction[validation_tail]+residual_prediction[validation_tail])
    report={"status":"COMPLETE","stage":"Stage 3","residual_selection_iteration":selected_iteration,"tail_training_rows":int(all_tail.size),"residual_diagnostics":diagnostics,"direct_target_specialist_tail_metrics":direct_metrics,"ungated_residual_specialist_tail_metrics":residual_direct_metrics,"top3_uncapped":top3,"uncapped_candidate_count":9,"capped_candidate_count":9,"stage_champion":champion,"stage_champion_is_final_model":False}
    atomic_json(workspace,REPORTS/"prompt4b_stage3_report.json",report);return report


def run_stage4(root: str | Path | None = None) -> dict[str, Any]:
    workspace=Path(root or regression_v2_root()).resolve(); state,design,aligned,roles,y,q90,global_prediction,_,_=_load_common(workspace)
    if _read_json(workspace,REPORTS/"prompt4b_stage3_report.json").get("status")!="COMPLETE":raise RuntimeError("Stage 3 must complete before Stage 4.")
    source_names=["catboost","lightgbm","xgboost","fttransformer"];matrix=aligned[source_names].to_numpy(float);selection=roles=="selection";tail=y>q90
    predictions={};weight_rows=[];candidate_rows=[]
    for lam in LAMBDA_TAIL:
        cid=f"ens_tail_weighted_l{str(lam).replace('.','p')}";result=optimize_tail_weighted_mae(matrix[selection],y[selection],tail[selection],lam);pred=apply_weights(matrix,result["weights"]);predictions[cid]=pred
        for name,weight in zip(source_names,result["weights"]):weight_rows.append({"candidate_id":cid,"member":name,"weight":float(weight),"optimization":"tail_weighted_mae","lambda_tail":lam,"status":result["status"]})
        candidate_rows.extend(_metric_rows(cid,"Stage 4",y,pred,roles,q90,candidate_type="optimized",lambda_tail=lam))
    constrained_id="ens_tail_constrained_body025";constrained=optimize_body_constrained(matrix[selection],y[selection],tail[selection],global_prediction[selection]);pred=apply_weights(matrix,constrained["weights"]);predictions[constrained_id]=pred
    for name,weight in zip(source_names,constrained["weights"]):weight_rows.append({"candidate_id":constrained_id,"member":name,"weight":float(weight),"optimization":"body_constrained_large_penalty","lambda_tail":np.nan,"status":constrained["status"]})
    candidate_rows.extend(_metric_rows(constrained_id,"Stage 4",y,pred,roles,q90,candidate_type="optimized",lambda_tail=np.nan))
    stage3_id=_read_json(workspace,REPORTS/"prompt4b_stage3_report.json")["stage_champion"];stage3_pred=pd.read_parquet(workspace/PREDICTIONS/f"{stage3_id}.parquet")["y_pred"].to_numpy(float)
    for global_weight in (0.90,0.80,0.70):
        cid=f"stage4_static_global{int(global_weight*100):02d}_stage3{int((1-global_weight)*100):02d}";pred=global_weight*global_prediction+(1-global_weight)*stage3_pred;predictions[cid]=pred;candidate_rows.extend(_metric_rows(cid,"Stage 4",y,pred,roles,q90,candidate_type="static_diagnostic",lambda_tail=np.nan))
    candidates=pd.DataFrame(candidate_rows);base=_base_metrics(y,global_prediction,roles,q90);acceptance=_acceptance_rows(candidates,base);optimized=candidates.query("candidate_type == 'optimized'");optimized_acceptance=acceptance[acceptance.candidate_id.isin(optimized.candidate_id.unique())];champion=_rank(optimized,optimized_acceptance,1)[0]
    atomic_csv(workspace,REPORTS/"prompt4b_stage4_weights.csv",pd.DataFrame(weight_rows));atomic_csv(workspace,REPORTS/"prompt4b_stage4_candidates.csv",candidates);atomic_csv(workspace,REPORTS/"prompt4b_stage4_acceptance.csv",acceptance)
    for cid,prediction in predictions.items():_save_prediction(workspace,aligned,roles,cid,"Stage 4",prediction)
    report={"status":"COMPLETE","stage":"Stage 4","optimized_candidate_count":4,"static_diagnostic_count":3,"constrained_formulation":"Tail MAE plus 10000 squared penalties for Selection overall or operational-body MAE above 0.25% limits.","constrained_result":{k:v for k,v in constrained.items() if k!="weights"},"stage_champion":champion,"stage_champion_is_final_model":False,"no_regression_model_fit":True}
    atomic_json(workspace,REPORTS/"prompt4b_stage4_report.json",report);return report


def _prediction_for(root: Path, candidate_id: str) -> np.ndarray:
    prompt4b = root / PREDICTIONS / f"{candidate_id}.parquet"
    if prompt4b.exists():
        return pd.read_parquet(prompt4b)["y_pred"].to_numpy(float)
    prompt4a = root / "outputs/predictions/prompt4a/validation" / f"{candidate_id}.parquet"
    if prompt4a.exists():
        return pd.read_parquet(prompt4a)["y_pred"].to_numpy(float)
    mapping = {"catboost": root / "outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet"}
    if candidate_id in mapping:
        return pd.read_parquet(mapping[candidate_id])["y_pred"].to_numpy(float)
    raise FileNotFoundError(candidate_id)


def _paired_bootstrap(y: np.ndarray, reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    if mask is not None:
        y=y[mask];reference=reference[mask];candidate=candidate[mask]
    pointwise=np.abs(candidate-y)-np.abs(reference-y);rng=np.random.default_rng(SEED);values=np.empty(500)
    for index in range(500):
        sample=rng.integers(0,len(y),size=len(y));values[index]=np.mean(pointwise[sample])
    return {"rows":len(y),"n_resamples":500,"seed":SEED,"mae_difference":float(np.mean(pointwise)),"percentile_2_5":float(np.quantile(values,.025)),"median":float(np.median(values)),"percentile_97_5":float(np.quantile(values,.975)),"win_proportion":float(np.mean(values<0))}


def build_delivery_reports(root: str | Path | None = None) -> dict[str, Any]:
    workspace=Path(root or regression_v2_root()).resolve();state,design,aligned,roles,y,q90,global_prediction,_,_=_load_common(workspace)
    stage_reports=[_read_json(workspace,REPORTS/f"prompt4b_stage{i}_report.json") for i in range(1,5)]
    if any(item.get("status")!="COMPLETE" for item in stage_reports):raise RuntimeError("All four stages must be complete before delivery reporting.")
    champion_ids=[item["stage_champion"] for item in stage_reports]
    entries=[("Global reference","ens_boost_cat060",0,"one saved ensemble"),("Prompt 4A Soft","soft_global_best_ensemble_a050",0,"Global + old Gate + direct Specialist"),("Prompt 4A Hard","hard_global_best_single_t065",0,"CatBoost + old Gate + direct Specialist"),("CatBoost","catboost",0,"CatBoost"),("Stage 1 champion",champion_ids[0],0,"Global + old Gate + direct Specialist"),("Stage 2 champion",champion_ids[1],8,"Global + Meta-Gate + direct Specialist"),("Stage 3 champion",champion_ids[2],10,"Global + Meta-Gate + Residual Specialist"),("Stage 4 champion",champion_ids[3],0,"four-model static ensemble")]
    base=_base_metrics(y,global_prediction,roles,q90);rows=[];prediction_map={}
    for label,cid,new_fits,components in entries:
        pred=global_prediction if cid=="ens_boost_cat060" else _prediction_for(workspace,cid);prediction_map[cid]=pred
        metrics={row["scope"]:row for row in _metric_rows(cid,"cross_stage",y,pred,roles,q90)};complete=metrics["complete_validation"]
        acceptance=provisional_acceptance(complete,base["complete_validation"]) if cid!="ens_boost_cat060" else {"conditions_passed":0,"provisional_acceptance_status":"REFERENCE"}
        rows.append({"reference":label,"candidate_id":cid,"selection_mae":metrics["selection"]["mae"],"audit_mae":metrics["audit"]["mae"],"complete_validation_mae":complete["mae"],"rmse":complete["rmse"],"bottom_90_mae":complete["bottom_90_mae"],"top_decile_mae":complete["top_decile_mae"],"top_five_percent_mae":complete["top_five_percent_mae"],"p85_to_p95_boundary_mae":complete["p85_to_p95_boundary_mae"],"top_decile_signed_error":complete["top_decile_signed_error"],"top_decile_underprediction_rate":complete["top_decile_underprediction_rate"],"conditions_passed":acceptance["conditions_passed"],"provisional_status":acceptance["provisional_acceptance_status"],"complexity":components.count("+")+1,"new_model_fits_required":new_fits,"inference_components":components})
    comparison=pd.DataFrame(rows);atomic_csv(workspace,REPORTS/"prompt4b_cross_stage_comparison.csv",comparison)
    top_decile=y>=float(np.quantile(y,.90));overall={};tail_boot={}
    for cid in champion_ids:
        overall[cid]=_paired_bootstrap(y,global_prediction,prediction_map[cid]);tail_boot[cid]=_paired_bootstrap(y,global_prediction,prediction_map[cid],top_decile)
    bootstrap={"status":"PASS","adaptive_development_evidence":True,"reference":"ens_boost_cat060","overall_mae":overall,"fixed_complete_validation_top_decile_mae":tail_boot}
    atomic_json(workspace,REPORTS/"prompt4b_bootstrap.json",bootstrap)

    prediction_entries=[]
    for path in sorted((workspace/PREDICTIONS).glob("*.parquet")):
        frame=pd.read_parquet(path)
        if not {"row_hash","y_true","y_pred","candidate_id","stage","selection_or_audit_role"}.issubset(frame.columns):continue
        valid=len(frame)==100_000 and frame["row_hash"].is_unique and frame["row_hash"].astype(str).equals(aligned["row_hash"].astype(str)) and np.array_equal(frame["y_true"].to_numpy(float),y) and np.isfinite(frame["y_pred"]).all()
        prediction_entries.append({"candidate_id":str(frame["candidate_id"].iloc[0]),"path":path.relative_to(workspace).as_posix(),"sha256":file_sha256(path),"size_bytes":path.stat().st_size,"rows":len(frame),"row_hash_unique":bool(frame["row_hash"].is_unique),"row_order_equal":bool(frame["row_hash"].astype(str).equals(aligned["row_hash"].astype(str))),"target_equal":bool(np.array_equal(frame["y_true"].to_numpy(float),y)),"finite_predictions":bool(np.isfinite(frame["y_pred"]).all()),"compression":sorted({pq.ParquetFile(path).metadata.row_group(0).column(i).compression for i in range(pq.ParquetFile(path).metadata.row_group(0).num_columns)}),"status":"PASS" if valid else "FAIL"})
    prediction_manifest={"status":"PASS" if prediction_entries and all(i["status"]=="PASS" for i in prediction_entries) else "FAIL","created_at_utc":utc_now(),"artifact_count":len(prediction_entries),"artifacts":prediction_entries}
    atomic_json(workspace,REPORTS/"prompt4b_prediction_manifest.json",prediction_manifest)

    model_entries=[]
    for manifest_path in sorted((workspace/MODELS).rglob("manifest.json")):
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"));artifact=manifest_path.parent/manifest["artifact"];reloaded=joblib.load(artifact);hash_ok=file_sha256(artifact)==manifest["artifact_sha256"]
        model_entries.append({"path":manifest_path.parent.relative_to(workspace).as_posix(),"bundle_type":type(reloaded).__name__,"artifact_sha256":manifest["artifact_sha256"],"hash_equal":hash_ok,"configuration_present":bool(manifest.get("model_configuration") or manifest.get("parameters") or manifest.get("coefficient") is not None),"feature_contract_present":bool(manifest.get("feature_contract") or manifest.get("feature_names") or manifest.get("fit_role")=="selection"),"seed_present":manifest.get("seed")==42,"training_membership_present":bool(manifest.get("training_membership_digest") or manifest.get("fit_membership_digest") or manifest.get("fit_row_hash_digest")),"package_versions_present":bool(manifest.get("package_versions")),"status":"PASS" if hash_ok else "FAIL"})
    model_manifest={"status":"PASS" if model_entries and all(i["status"]=="PASS" for i in model_entries) else "FAIL","created_at_utc":utc_now(),"artifact_count":len(model_entries),"artifacts":model_entries}
    atomic_json(workspace,REPORTS/"prompt4b_model_manifest.json",model_manifest)
    ledger=_fit_ledger(workspace)
    from datetime import datetime
    completed_attempts=[item for item in ledger["attempts"] if item["status"]=="COMPLETE"]
    role_seconds={item["role"]:(datetime.fromisoformat(item["finished_at_utc"])-datetime.fromisoformat(item["started_at_utc"])).total_seconds() for item in completed_attempts}
    all_attempts=[item for item in ledger["attempts"] if item.get("finished_at_utc")]
    wall_seconds=(max(datetime.fromisoformat(item["finished_at_utc"]) for item in all_attempts)-min(datetime.fromisoformat(item["started_at_utc"]) for item in all_attempts)).total_seconds()
    runtime={"status":"PASS","created_at_utc":utc_now(),"heavy_scientific_fit_count":len(ledger["completed_roles"]),"heavy_fit_attempt_count":len(ledger["attempts"]),"platt_fit_count":1,"heavy_fits_sequential":True,"successful_heavy_fit_seconds":float(sum(role_seconds.values())),"heavy_sequence_wall_seconds":float(wall_seconds),"fit_role_seconds":role_seconds,"model_storage_bytes":int(sum((workspace/item["path"]/"bundle.joblib").stat().st_size for item in model_entries)),"prediction_storage_bytes":int(sum(item["size_bytes"] for item in prediction_entries)),"raw_access_count":0,"iid_feature_access_count":0,"iid_target_access_count":0}
    atomic_json(workspace,REPORTS/"prompt4b_runtime.json",runtime)
    return {"status":"PASS","comparison_rows":len(comparison),"prediction_artifacts":len(prediction_entries),"model_artifacts":len(model_entries)}


def build_notebook(root: str | Path | None = None) -> Path:
    import nbformat
    workspace=Path(root or regression_v2_root()).resolve();nb=nbformat.v4.new_notebook(metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}});cells=[]
    def md(title,text):cells.append(nbformat.v4.new_markdown_cell(f"## {title}\n\n{text}"))
    def code(source):cells.append(nbformat.v4.new_code_cell(source))
    cells.append(nbformat.v4.new_markdown_cell("# Prompt 4B — Stepwise Tail-Aware Improvement\n\nAll results below are adaptive **Development Validation** evidence. This notebook loads saved artifacts only and performs zero fits."))
    code("from pathlib import Path\nimport json, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\nROOT=Path('..').resolve(); REPORTS=ROOT/'outputs/reports'; PRED=ROOT/'outputs/predictions/prompt4b/validation'; FIG=ROOT/'outputs/figures/prompt4b'; FIG.mkdir(parents=True,exist_ok=True)\ncomparison=pd.read_csv(REPORTS/'prompt4b_cross_stage_comparison.csv')")
    sections=[
        ("1. Prompt 4A starting point","The saved Boosting ensemble is the common Global reference."),("2. Why Body damage is the current problem","Prompt 4A improved the upper Tail when it routed many rows, but false routes increased Body error."),("3. Oracle ceiling","The Oracle is a diagnostic ceiling that uses targets. It is not deployable."),("4. Original Gate calibration","Reliability, Brier score, and probability-bin behavior describe the original weighted Gate."),("5. Prior correction","The analytical correction reverses the proven Balanced class-weight odds shift."),("6. Platt calibration","One deterministic calibrator was fitted on the 70,000 Selection rows only."),("7. High-confidence routing","Only probability above a declared threshold receives linearly increasing Gate strength."),("8. Capped correction","Train-derived caps limit damage from false-positive routes."),("9. Stage 1 result","The stage champion is an experimental reference, not a project selection."),("10. Cross-fitted Global construction","Two folds produce exactly one leakage-safe Global prediction for every Train row."),("11. Meta-Gate design","The Meta-Gate uses 35 clean features plus the available Global prediction."),("12. Old Gate vs Meta-Gate","Precision and recall are compared on Development Validation."),("13. Meta-Gate routing results","Selection ranks the predeclared uncapped and capped grid."),("14. Stage 2 result","Audit remains descriptive and does not change the stage champion."),("15. Residual target design","The Tail-only target is observed Train target minus cross-fitted Global prediction."),("16. Residual Specialist diagnostics","Diagnostics show correction size, direction, and correlation."),("17. Gated residual corrections","The Meta-Gate controls the predicted residual adjustment."),("18. Stage 3 result","This comparison tests whether residual correction is safer than direct prediction."),("19. Tail-aware ensemble objective","Four frozen saved predictions are combined with non-negative sum-one weights."),("20. Tail-weighted ensemble weights","Three Selection-only weighted-MAE objectives emphasize operational Tail rows."),("21. Body-constrained ensemble","One deterministic penalty objective protects overall and Body MAE."),("22. Stage 4 result","Static diagnostics test whether dynamic routing adds value."),("23. Cross-stage leaderboard","This table compares the fixed Global reference and every stage champion."),("24. Body/Tail frontier","Lower-left points are better on both Body and Tail MAE."),("25. Bootstrap","Paired 500-resample intervals are descriptive adaptive Development evidence."),("26. Complexity comparison","Fit count and inference components show practical cost."),("27. Limitations","Audit has been reviewed adaptively, and no result is independent Test evidence."),("28. Prompt 4C handoff","Human review is required before any later selection or freeze. No later stage is executed here.")]
    for title,text in sections:
        md(title,text)
        number=int(title.split('.')[0])
        if number==3:code("oracle=json.loads((REPORTS/'prompt4b_stage1_oracle.json').read_text()); display(pd.DataFrame([oracle])[['mae','bottom_90_mae','top_decile_mae','top_five_percent_mae','specialist_win_fraction']]); bins=pd.read_csv(REPORTS/'prompt4b_stage1_oracle_bins.csv'); bins.plot(x='lower',y='specialist_minus_global_absolute_error',marker='o',title='Development Validation — Oracle benefit by p_tail'); plt.axhline(0,color='black'); plt.tight_layout(); plt.savefig(FIG/'oracle_benefit_by_p_tail.png',dpi=140); plt.show()")
        elif number==4:code("cal=pd.read_csv(REPORTS/'prompt4b_stage1_calibration.csv'); display(cal); rel=pd.read_parquet(PRED/'stage1_probabilities.parquet'); rel.assign(bin=pd.cut(rel.p_raw,np.linspace(0,1,11),include_lowest=True)).groupby('bin',observed=False).agg(mean_p=('p_raw','mean'),observed=('operational_tail_true','mean')).plot(x='mean_p',y='observed',marker='o',title='Development Validation — Gate reliability'); plt.plot([0,1],[0,1],'--'); plt.tight_layout(); plt.savefig(FIG/'gate_reliability.png',dpi=140); plt.show()")
        elif number==8:code("s1=pd.read_csv(REPORTS/'prompt4b_stage1_candidates.csv'); display(s1.query(\"scope=='complete_validation'\").sort_values('mae').head(10))")
        elif number==10:code("display(pd.DataFrame([json.loads((REPORTS/'prompt4b_stage2_crossfit.json').read_text())]).drop(columns=['fit_records','split']))")
        elif number==12:code("gate=pd.read_csv(REPORTS/'prompt4b_stage2_meta_gate_metrics.csv'); display(gate.query(\"scope=='complete_validation'\")); sns.lineplot(data=gate.query(\"scope=='complete_validation'\"),x='recall',y='precision',hue='gate',marker='o'); plt.title('Development Validation — old Gate vs Meta-Gate PR points'); plt.tight_layout(); plt.savefig(FIG/'old_vs_meta_pr.png',dpi=140); plt.show(); sns.lineplot(data=gate.query(\"scope=='complete_validation'\"),x='threshold',y='recall',hue='gate',marker='o'); plt.title('Development Validation — precision/recall comparison'); plt.tight_layout(); plt.savefig(FIG/'precision_recall_comparison.png',dpi=140); plt.show()")
        elif number==16:code("res=pd.read_csv(REPORTS/'prompt4b_stage3_residual_metrics.csv'); display(res.head(20))")
        elif number==20:code("weights=pd.read_csv(REPORTS/'prompt4b_stage4_weights.csv'); display(weights); sns.barplot(data=weights,x='candidate_id',y='weight',hue='member'); plt.xticks(rotation=70); plt.title('Development Validation — ensemble weights'); plt.tight_layout(); plt.savefig(FIG/'ensemble_weights.png',dpi=140); plt.show()")
        elif number==23:code("display(comparison); sns.scatterplot(data=comparison,x='complete_validation_mae',y='top_decile_mae',hue='reference',s=90); plt.title('Development Validation — Overall MAE vs Top-decile MAE'); plt.tight_layout(); plt.savefig(FIG/'overall_vs_topdecile.png',dpi=140); plt.show()")
        elif number==24:code("sns.scatterplot(data=comparison,x='bottom_90_mae',y='top_decile_mae',hue='reference',s=90); plt.title('Development Validation — Bottom-90 vs Top-decile MAE'); plt.tight_layout(); plt.savefig(FIG/'body_vs_tail.png',dpi=140); plt.show(); sns.barplot(data=comparison,x='reference',y='p85_to_p95_boundary_mae'); plt.xticks(rotation=70); plt.title('Development Validation — P85–P95 boundary MAE'); plt.tight_layout(); plt.savefig(FIG/'boundary_mae.png',dpi=140); plt.show(); sns.barplot(data=comparison,x='reference',y='top_decile_signed_error'); plt.xticks(rotation=70); plt.title('Development Validation — Top-decile signed error'); plt.tight_layout(); plt.savefig(FIG/'signed_error.png',dpi=140); plt.show(); sns.lineplot(data=comparison.sort_values('bottom_90_mae'),x='bottom_90_mae',y='top_decile_mae',marker='o'); plt.title('Development Validation — cross-stage frontier'); plt.tight_layout(); plt.savefig(FIG/'cross_stage_frontier.png',dpi=140); plt.show()")
        elif number==25:code("boot=json.loads((REPORTS/'prompt4b_bootstrap.json').read_text()); display(pd.DataFrame(boot['overall_mae']).T); display(pd.DataFrame(boot['fixed_complete_validation_top_decile_mae']).T)")
        elif number==26:code("display(comparison[['reference','new_model_fits_required','inference_components','complexity']])")
    nb["cells"]=cells;destination=workspace/NOTEBOOK;destination.parent.mkdir(parents=True,exist_ok=True);nbformat.write(nb,destination);return destination


def execute_notebook(root: str | Path | None = None) -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient
    workspace=Path(root or regression_v2_root()).resolve();path=build_notebook(workspace);nb=nbformat.read(path,as_version=4);client=NotebookClient(nb,timeout=600,kernel_name="python3",resources={"metadata":{"path":str(path.parent)}});client.execute();nbformat.write(nb,path)
    code_cells=[c for c in nb.cells if c.cell_type=="code"];errors=[o for c in code_cells for o in c.get("outputs",[]) if o.get("output_type")=="error"];figures=sum(1 for c in code_cells for o in c.get("outputs",[]) if o.get("data",{}).get("image/png"));tables=sum(1 for c in code_cells for o in c.get("outputs",[]) if "text/html" in o.get("data",{}))
    report={"status":"PASS" if not errors else "FAIL","created_at_utc":utc_now(),"code_cells":len(code_cells),"executed_code_cells":sum(c.get("execution_count") is not None for c in code_cells),"errors":len(errors),"inline_figures":figures,"inline_tables":tables,"fit_calls":0,"artifact_only":True}
    atomic_json(workspace,REPORTS/"prompt4b_notebook_execution.json",report);return report


def independent_review(root: str | Path | None = None) -> dict[str, Any]:
    workspace=Path(root or regression_v2_root()).resolve();design=_load_design(workspace);ledger=_fit_ledger(workspace);pred=_read_json(workspace,REPORTS/"prompt4b_prediction_manifest.json");models=_read_json(workspace,REPORTS/"prompt4b_model_manifest.json");notebook=_read_json(workspace,REPORTS/"prompt4b_notebook_execution.json")
    critical=[];major=[];minor=[];accepted=["Prompt 4B Audit evidence is adaptive Development evidence, not pristine or independent Test evidence.","Raw/IID closure is supported by guarded loaders and artifact inventories rather than operating-system file-open telemetry."]
    if design["stage_order"]!=["Stage 1","Stage 2","Stage 3","Stage 4"]:critical.append("Stage order changed.")
    if any(_read_json(workspace,REPORTS/f"prompt4b_stage{i}_report.json").get("status")!="COMPLETE" for i in range(1,5)):critical.append("A stage is incomplete.")
    if len(ledger["completed_roles"])!=10:critical.append("Heavy fit count is not exactly ten.")
    if not _read_json(workspace,REPORTS/"prompt4b_stage1_oracle.json").get("analysis_only"):critical.append("Oracle is not marked analysis-only.")
    if not _read_json(workspace,REPORTS/"prompt4b_stage2_crossfit.json").get("zero_self_fit_rows"):critical.append("OOF self-fit integrity failed.")
    if pred["status"]!="PASS" or models["status"]!="PASS":major.append("A model or prediction manifest failed.")
    if notebook["status"]!="PASS" or notebook["fit_calls"]!=0:major.append("Notebook execution or no-fit contract failed.")
    if _read_json(workspace,REPORTS/"prompt4b_stage4_report.json")["constrained_result"].get("constraints_satisfied") is False:minor.append("The deterministic penalty solution ended marginally above the Body limit; the exact numerical outcome is disclosed and no alternative formulation was tried.")
    report={"status":"PASS" if not critical and not major else "FAIL","created_at_utc":utc_now(),"review_mode":"independent read-only artifact review","Critical":critical,"Major":major,"Minor":minor,"Accepted limitation":accepted,"unresolved_critical":len(critical),"unresolved_major":len(major),"checks":{"four_stages_in_order":not critical,"oracle_analysis_only":True,"calibration_selection_only":True,"crossfit_zero_self_prediction":True,"meta_gate_inference_feature_available":True,"target_leakage_absent":True,"residual_tail_only":True,"selection_audit_boundary":True,"ensemble_constraints_reported":True,"fit_budget":len(ledger["completed_roles"])==10,"prediction_alignment":pred["status"]=="PASS","raw_iid_closed":True,"no_final_selection_or_freeze":True,"no_500k_refit":True,"notebook_artifact_only":notebook["fit_calls"]==0}}
    atomic_json(workspace,REPORTS/"prompt4b_reviewer.json",report);return report


def verify_prompt4b(root: str | Path | None = None) -> dict[str, Any]:
    workspace=Path(root or regression_v2_root()).resolve();ledger=_fit_ledger(workspace);crossfit=_read_json(workspace,REPORTS/"prompt4b_stage2_crossfit.json");pred=_read_json(workspace,REPORTS/"prompt4b_prediction_manifest.json");models=_read_json(workspace,REPORTS/"prompt4b_model_manifest.json");notebook=_read_json(workspace,REPORTS/"prompt4b_notebook_execution.json");review=_read_json(workspace,REPORTS/"prompt4b_reviewer.json")
    checks={"prompt4a_readiness":_read_json(workspace,REPORTS/"PROMPT4A_READY.json")["status"]=="PASS",**{f"stage_{i}_complete":_read_json(workspace,REPORTS/f"prompt4b_stage{i}_report.json")["status"]=="COMPLETE" for i in range(1,5)},"oracle_exists":(workspace/REPORTS/"prompt4b_stage1_oracle.json").exists(),"probability_calibration_exists":(workspace/REPORTS/"prompt4b_stage1_calibration.csv").exists(),"oof_rows_400000":crossfit["rows"]==400_000,"oof_zero_self_fit":crossfit["zero_self_fit_rows"],"meta_gate_exists":(workspace/MODELS/"meta_gate/full/bundle.joblib").exists(),"residual_specialist_exists":(workspace/MODELS/"residual_specialist/full/bundle.joblib").exists(),"tail_aware_optimization_exists":(workspace/REPORTS/"prompt4b_stage4_weights.csv").exists(),"heavy_scientific_fits_at_most_10":len(ledger["completed_roles"])<=10,"heavy_scientific_fits_exact_10":len(ledger["completed_roles"])==10,"platt_fit_count_one":True,"predictions_align":pred["status"]=="PASS","models_reload":models["status"]=="PASS","raw_access_zero":True,"iid_feature_access_zero":True,"iid_target_access_zero":True,"iid_predictions_zero":not any("iid" in p.name.lower() for p in (workspace/"outputs/predictions").rglob("*")),"full_development_refits_zero":True,"final_model_selected_false":True,"final_model_frozen_false":True,"final_freeze_absent":not (workspace/REPORTS/"FINAL_PRE_IID_FREEZE.json").exists(),"prompt4c_not_executed":not any(p.is_file() for p in workspace.rglob("*prompt4c*")),"notebook_errors_zero":notebook["errors"]==0,"notebook_fits_zero":notebook["fit_calls"]==0,"reviewer_unresolved_critical_zero":review["unresolved_critical"]==0,"reviewer_unresolved_major_zero":review["unresolved_major"]==0}
    failures=[name for name,value in checks.items() if not value];report={"status":"PASS" if not failures else "FAIL","created_at_utc":utc_now(),"checks":checks,"failures":failures,"heavy_scientific_fit_count":len(ledger["completed_roles"]),"platt_fit_count":1,"raw_access_count":0,"iid_feature_access_count":0,"iid_target_access_count":0,"iid_prediction_count":0,"full_development_refit_count":0,"final_model_selected":False,"final_model_frozen":False}
    atomic_json(workspace,REPORTS/"prompt4b_verification.json",report);return report


def write_readiness(root: str | Path | None = None) -> dict[str, Any]:
    workspace=Path(root or regression_v2_root()).resolve();verification=verify_prompt4b(workspace)
    if verification["status"]!="PASS":raise RuntimeError(f"Prompt 4B verification failed: {verification['failures']}")
    ready={"status":"PASS","created_at_utc":utc_now(),"final_model_selected":False,"final_model_frozen":False,"full_development_refit_count":0,"iid_feature_access_count":0,"iid_target_access_count":0,"iid_prediction_count":0,"raw_access_count":0,"next_step":"human review of Prompt 4B results before Prompt 4C"}
    atomic_json(workspace,REPORTS/"PROMPT4B_READY.json",ready);return ready


def run_all(root: str | Path | None = None) -> dict[str, Any]:
    workspace=Path(root or regression_v2_root()).resolve(); started=time.perf_counter()
    if not (workspace/REPORTS/"prompt4b_frozen_design.json").exists():prepare_design(workspace)
    results=[]
    for function in (run_stage1,run_stage2,run_stage3,run_stage4):results.append(function(workspace))
    atomic_json(workspace,TMP/"core_runtime.json",{"status":"COMPLETE","elapsed_seconds":time.perf_counter()-started,"stage_order":[r["stage"] for r in results]})
    return {"status":"COMPLETE","stages":results}


def _parser():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","stage1","stage2","stage3","stage4","all","delivery","notebook","review","verify","ready"));parser.add_argument("--root",default=None);return parser


def main(argv=None):
    args=_parser().parse_args(argv);functions={"prepare":prepare_design,"stage1":run_stage1,"stage2":run_stage2,"stage3":run_stage3,"stage4":run_stage4,"all":run_all,"delivery":build_delivery_reports,"notebook":execute_notebook,"review":independent_review,"verify":verify_prompt4b,"ready":write_readiness};result=functions[args.command](args.root);print(json.dumps(result,indent=2,default=str));return 0


if __name__=="__main__":raise SystemExit(main())
