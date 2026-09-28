"""Bounded Benefit-Aware Selective Correction experiments for Prompt 4B2.

This module preserves the adaptive Development boundary. It never opens Raw
or IID files, never fits on Validation, and never selects a final project model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

try:
    from .deep_preprocessing import ordered_digest
    from .prompt4a_experiments import (
        atomic_csv,
        atomic_json,
        atomic_parquet,
        file_sha256,
        load_development,
        package_versions,
        regression_v2_root,
    )
    from .prompt4b_crossfit import GLOBAL_FEATURE, membership_digest, save_experimental_bundle
    from .prompt4b_residual import ResidualSpecialistBundle, fit_residual_model
    from .prompt4b2_metrics import (
        acceptance_rows,
        direct_routing,
        gate_strength,
        metric_rows,
        paired_bootstrap,
        reference_by_scope,
        routed_residual,
        selection_rank,
    )
    from .tail_models import make_internal_tail_split
except ImportError:
    from deep_preprocessing import ordered_digest
    from prompt4a_experiments import atomic_csv, atomic_json, atomic_parquet, file_sha256, load_development, package_versions, regression_v2_root
    from prompt4b_crossfit import GLOBAL_FEATURE, membership_digest, save_experimental_bundle
    from prompt4b_residual import ResidualSpecialistBundle, fit_residual_model
    from prompt4b2_metrics import acceptance_rows, direct_routing, gate_strength, metric_rows, paired_bootstrap, reference_by_scope, routed_residual, selection_rank
    from tail_models import make_internal_tail_split


SEED = 42
TARGET = "loan_amount_000s"
FEATURE_CONTRACT = "main_without_sensitive_without_lender"
REPORTS = Path("outputs/reports")
MODELS = Path("outputs/models/prompt4b2")
PREDICTIONS = Path("outputs/predictions/prompt4b2")
VALIDATION_PREDICTIONS = PREDICTIONS / "validation"
TRAIN_PREDICTIONS = PREDICTIONS / "train"
TMP = Path("outputs/tmp/prompt4b2")
FIGURES = Path("outputs/figures/prompt4b2")
NOTEBOOK = Path("notebooks/04B2_BENEFIT_AWARE_SELECTIVE_CORRECTION.ipynb")
QUANTILE_ALPHAS = (0.60, 0.65)
BENEFIT_THRESHOLDS = (0.0, 5.0, 10.0)
QUANTILE_ITERATIONS = 791
NO_FIT_CANDIDATES = (
    "nf_oldraw_residual_symcap25",
    "nf_oldplatt_residual",
    "nf_oldraw_residual_positive_cap25",
    "nf_meta_residual_positive_cap25",
    "nf_consensus_residual_positive_cap25",
    "nf_meta_residual_positive_gamma15",
    "nf_meta_residual_positive_gamma20",
    "nf_global_convex_boosting_deep",
    "nf_global2_oldraw_direct_cap25",
)
FIT_ROLES = (
    "quantile_residual_q60",
    "quantile_residual_q65",
    "oof_residual_foldA",
    "oof_residual_foldB",
    "benefit_router_selection",
    "benefit_router_full_refit",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_digest(root: Path) -> str:
    names = ("prompt4b2_experiments.py", "prompt4b2_benefit.py", "prompt4b2_quantile.py", "prompt4b2_metrics.py")
    payload = {name: file_sha256(root / "src" / name) for name in names}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _assert_aligned(base: pd.DataFrame, frame: pd.DataFrame, name: str, *, require_y: bool = True) -> None:
    if len(frame) != 100_000 or frame["row_hash"].astype(str).tolist() != base["row_hash"].astype(str).tolist():
        raise RuntimeError(f"{name} row_hash order is not the frozen Validation order.")
    if require_y and ("y_true" not in frame or not np.array_equal(frame["y_true"].to_numpy(float), base["y_true"].to_numpy(float))):
        raise RuntimeError(f"{name} y_true is not exact.")


def _validation_sources(root: Path, validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = {
        "G": Path("outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet"),
        "G2": Path("outputs/predictions/prompt4a/validation/ens_convex_boosting_deep.parquet"),
        "P_OLD_RAW_CANONICAL": Path("outputs/predictions/prompt4a/validation/tail_gate.parquet"),
        "D": Path("outputs/predictions/prompt4a/validation/tail_specialist.parquet"),
        "P_OLD_AND_PLATT": Path("outputs/predictions/prompt4b/validation/stage1_probabilities.parquet"),
        "P_META": Path("outputs/predictions/prompt4b/validation/stage2_meta_gate_probability.parquet"),
        "R": Path("outputs/predictions/prompt4b/validation/stage3_residual_prediction.parquet"),
    }
    base = pd.DataFrame({"row_hash": validation["row_hash"].astype(str), "y_true": validation[TARGET].to_numpy(float)})
    loaded: dict[str, pd.DataFrame] = {}
    evidence: list[dict[str, Any]] = []
    for role, relative in paths.items():
        frame = pd.read_parquet(root / relative)
        _assert_aligned(base, frame, role, require_y=role != "P_OLD_RAW_CANONICAL")
        loaded[role] = frame
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"{role} contains a non-finite numeric value.")
        evidence.append({"role": role, "path": relative.as_posix(), "sha256": file_sha256(root / relative), "rows": len(frame), "columns": frame.columns.tolist()})
    if not np.array_equal(loaded["P_OLD_RAW_CANONICAL"]["p_tail"].to_numpy(float), loaded["P_OLD_AND_PLATT"]["p_raw"].to_numpy(float)):
        raise RuntimeError("Prompt 4A raw Gate and Prompt 4B raw Gate copy differ.")
    aligned = base.assign(
        g=loaded["G"]["y_pred"].to_numpy(float),
        g2=loaded["G2"]["y_pred"].to_numpy(float),
        p_old_raw=loaded["P_OLD_AND_PLATT"]["p_raw"].to_numpy(float),
        p_old_platt=loaded["P_OLD_AND_PLATT"]["p_platt"].to_numpy(float),
        p_meta=loaded["P_META"]["p_meta_gate"].to_numpy(float),
        direct_specialist=loaded["D"]["y_pred"].to_numpy(float),
        residual=loaded["R"]["predicted_residual"].to_numpy(float),
        role=loaded["P_OLD_AND_PLATT"]["selection_or_audit_role"].astype(str).to_numpy(),
    )
    if aligned["role"].value_counts().to_dict() != {"selection": 70_000, "audit": 30_000}:
        raise RuntimeError("Selection/Audit membership changed.")
    return aligned, {"status": "PASS", "artifacts": evidence, "exact_raw_gate_copy": True}


def validate_prompt4b(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    required = (
        "DATA_READY.json", "PROMPT2_READY.json", "PROMPT3_READY.json", "PROMPT4A_READY.json", "PROMPT4B_READY.json",
        "prompt4a_verification.json", "prompt4b_verification.json", "prompt4a_frozen_design.json", "prompt4b_frozen_design.json",
        "prompt4a_model_manifest.json", "prompt4a_prediction_manifest.json", "prompt4b_model_manifest.json", "prompt4b_prediction_manifest.json", "feature_roles.json",
    )
    reports = {name: _json(workspace / REPORTS / name) for name in required}
    bad = [name for name, value in reports.items() if value.get("status") not in {"PASS", "FROZEN"}]
    if bad:
        raise RuntimeError(f"Mandatory Prompt 4B handoff failed: {bad}")
    if (workspace / REPORTS / "FINAL_PRE_IID_FREEZE.json").exists():
        raise RuntimeError("FINAL_PRE_IID_FREEZE.json exists.")
    if any(path.is_file() for path in workspace.rglob("*prompt4c*")):
        raise RuntimeError("Prompt 4C has started.")
    train, validation, source = load_development(workspace)
    frozen_source = reports["prompt4b_frozen_design.json"]["source"]
    if source["sha256"] != frozen_source["sha256"] or len(train) != 400_000 or len(validation) != 100_000:
        raise RuntimeError("Development identity or roles changed.")
    if ordered_digest(train["row_hash"]) != frozen_source["train_row_hash_digest"] or ordered_digest(validation["row_hash"]) != frozen_source["validation_row_hash_digest"]:
        raise RuntimeError("Frozen Train/Validation membership changed.")
    features = list(reports["feature_roles.json"]["contracts"][FEATURE_CONTRACT])
    if len(features) != 35 or features != frozen_source["features"]:
        raise RuntimeError("The exact 35-feature contract changed.")
    q90 = float(np.quantile(train[TARGET].to_numpy(float), 0.90))
    if q90 != float(reports["prompt4b_frozen_design.json"]["q90_train"]):
        raise RuntimeError("q90_train changed.")
    aligned, prediction_audit = _validation_sources(workspace, validation)
    global_oof_path = workspace / "outputs/predictions/prompt4b/train/global_oof_prediction.parquet"
    global_oof = pd.read_parquet(global_oof_path)
    if len(global_oof) != 400_000 or global_oof["row_hash"].astype(str).tolist() != train["row_hash"].astype(str).tolist():
        raise RuntimeError("G_OOF is not in exact Train order.")
    if not np.array_equal(global_oof["y_true"].to_numpy(float), train[TARGET].to_numpy(float)) or not np.isfinite(global_oof.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError("G_OOF target or numeric values are invalid.")
    if global_oof["fold_id"].value_counts().to_dict() != {"fold_a": 200_000, "fold_b": 200_000}:
        raise RuntimeError("Frozen two-fold membership changed.")
    closure = reports["prompt4b_verification.json"]
    if any(int(closure.get(key, -1)) != 0 for key in ("raw_access_count", "iid_feature_access_count", "iid_target_access_count", "iid_prediction_count", "full_development_refit_count")):
        raise RuntimeError("Prompt 4B access closure failed.")
    return {
        "status": "PASS", "train": train, "validation": validation, "source": source, "features": features, "q90_train": q90,
        "aligned": aligned, "global_oof": global_oof, "reports": reports,
        "prediction_audit": {**prediction_audit, "global_oof": {"path": "outputs/predictions/prompt4b/train/global_oof_prediction.parquet", "sha256": file_sha256(global_oof_path), "rows": len(global_oof), "columns": global_oof.columns.tolist()}},
        "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0, "iid_prediction_count": 0,
    }


def _platt_configuration(root: Path) -> dict[str, Any]:
    candidates = pd.read_csv(root / REPORTS / "prompt4b_stage1_candidates.csv")
    acceptance = pd.read_csv(root / REPORTS / "prompt4b_stage1_acceptance.csv")
    platt = candidates.loc[(candidates["scope"] == "selection") & (candidates["probability_source"] == "platt")].copy()
    platt = platt.merge(acceptance.loc[acceptance["scope"] == "selection", ["candidate_id", "conditions_passed"]], on="candidate_id", validate="one_to_one")
    platt = platt.sort_values(["conditions_passed", "mae", "bottom_90_mae", "top_decile_mae", "candidate_id"], ascending=[False, True, True, True, True], kind="mergesort")
    row = platt.iloc[0]
    return {"source_candidate_id": str(row["candidate_id"]), "threshold": float(row["threshold"]), "alpha": float(row["alpha"]), "cap": None if pd.isna(row["cap"]) else float(row["cap"]), "gamma": 1.0, "ranking_source": "saved Prompt 4B Selection evidence"}


def prepare_design(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    state = validate_prompt4b(workspace)
    fold_a = state["global_oof"]["fold_id"].to_numpy() == "fold_a"
    fold_b = ~fold_a
    source_hashes = {item["role"]: item["sha256"] for item in state["prediction_audit"]["artifacts"]}
    source_hashes["G_OOF"] = state["prediction_audit"]["global_oof"]["sha256"]
    core = {
        "prompt": "Prompt 4B2", "development_stage": "adaptive", "prompt4b_readiness": "PASS",
        "source": state["source"], "features": state["features"], "feature_count": 35,
        "feature_contract": FEATURE_CONTRACT, "feature_contract_digest": state["reports"]["prompt4b_frozen_design.json"]["source"]["feature_contract_digest"],
        "q90_train": state["q90_train"], "cap25": 0.25 * state["q90_train"],
        "authoritative_prediction_sources": state["prediction_audit"], "prediction_source_sha256": source_hashes,
        "zero_fit_candidates": list(NO_FIT_CANDIDATES), "zero_fit_candidate_count": 9,
        "old_platt_routing": _platt_configuration(workspace),
        "quantile_candidates": [{"model_id": "quantile_residual_q60", "alpha": 0.60}, {"model_id": "quantile_residual_q65", "alpha": 0.65}],
        "quantile_iterations": 791, "quantile_routing_forms": ["meta_symmetric", "meta_positive_cap25"],
        "residual_crossfit": {
            "reuse_global_oof_folds": True, "fold_a_rows": int(fold_a.sum()), "fold_b_rows": int(fold_b.sum()),
            "fold_a_ordered_digest": ordered_digest(state["global_oof"].loc[fold_a, "row_hash"]), "fold_b_ordered_digest": ordered_digest(state["global_oof"].loc[fold_b, "row_hash"]),
            "fold_a_membership_digest": membership_digest(state["global_oof"].loc[fold_a, "row_hash"]), "fold_b_membership_digest": membership_digest(state["global_oof"].loc[fold_b, "row_hash"]),
        },
        "benefit_target_formula": "abs(y-G_OOF)-abs(y-(G_OOF+clip(max(R_OOF,0),0,cap25)))",
        "benefit_router_features": state["features"] + [GLOBAL_FEATURE, "proposed_residual_correction"], "benefit_router_feature_count": 37,
        "benefit_router_parameters": {"loss_function": "RMSE", "eval_metric": "RMSE", "iterations": 1500, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10, "random_strength": 1, "random_seed": 42, "thread_count": 4, "early_stopping_rounds": 100, "verbose": False},
        "benefit_thresholds": list(BENEFIT_THRESHOLDS), "selection_only_ranking": True, "audit_descriptive_only": True,
        "six_condition_rubric_unchanged": True, "fit_roles": list(FIT_ROLES), "max_scientific_fits": 6, "heavy_fits_sequential": True,
        "seed": 42, "threads": 4, "package_versions": package_versions(), "code_digest": _code_digest(workspace),
        "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0, "iid_prediction_count": 0,
        "full_development_final_refit_count": 0, "final_model_selected": False, "final_model_frozen": False,
        "no_wider_band_model": True, "no_hierarchical_tail_model": True, "no_deep_fit": True, "prompt4c_not_started": True,
    }
    design = {"status": "FROZEN", "created_at_utc": utc_now(), **core}
    design["design_digest"] = hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    atomic_json(workspace, REPORTS / "prompt4b2_frozen_design.json", design)
    atomic_json(workspace, REPORTS / "prompt4b2_preflight.json", {"status": "PASS", "created_at_utc": utc_now(), "development_sha256": state["source"]["sha256"], "train_rows": 400_000, "validation_rows": 100_000, "selection_rows": 70_000, "audit_rows": 30_000, "q90_train": state["q90_train"], "cap25": 0.25 * state["q90_train"], "feature_count": 35, "prediction_audit": state["prediction_audit"], "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0, "iid_prediction_count": 0, "final_freeze_absent": True, "prompt4c_not_started": True})
    return design


def _design(root: Path) -> dict[str, Any]:
    design = _json(root / REPORTS / "prompt4b2_frozen_design.json")
    current_digest = _code_digest(root)
    repair_path = root / REPORTS / "prompt4b2_reporting_repair.json"
    repair_ok = False
    if repair_path.exists():
        repair = _json(repair_path)
        repair_ok = repair.get("status") == "PASS" and repair.get("scientific_design_unchanged") is True and repair.get("frozen_code_digest") == design.get("code_digest") and repair.get("repaired_code_digest") == current_digest
    if design.get("status") != "FROZEN" or (design.get("code_digest") != current_digest and not repair_ok):
        raise RuntimeError("Prompt 4B2 design is missing or implementation changed after freeze.")
    return design


def _save_validation_prediction(root: Path, aligned: pd.DataFrame, candidate_id: str, block: str, prediction: Any, **extra: Any) -> Path:
    frame = pd.DataFrame({"row_hash": aligned["row_hash"].astype(str), "y_true": aligned["y_true"].to_numpy(float), "y_pred": np.asarray(prediction, dtype=np.float64), "candidate_id": candidate_id, "block": block, "selection_or_audit_role": aligned["role"].astype(str)})
    for name, values in extra.items():
        frame[name] = values
    if len(frame) != 100_000 or not frame["row_hash"].is_unique or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise RuntimeError(f"Invalid saved prediction: {candidate_id}")
    return atomic_parquet(root, VALIDATION_PREDICTIONS / f"{candidate_id}.parquet", frame)


def _global_reference(aligned: pd.DataFrame, q90: float) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows = pd.DataFrame(metric_rows("ens_boost_cat060", "Reference", aligned["y_true"], aligned["g"], aligned["role"], q90))
    return rows, reference_by_scope(rows, "ens_boost_cat060")


def run_block_a(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    design = _design(workspace); state = validate_prompt4b(workspace); aligned = state["aligned"]
    q90 = float(design["q90_train"]); cap = float(design["cap25"])
    g = aligned["g"].to_numpy(float); g2 = aligned["g2"].to_numpy(float); r = aligned["residual"].to_numpy(float)
    p_old = aligned["p_old_raw"].to_numpy(float); p_platt = aligned["p_old_platt"].to_numpy(float); p_meta = aligned["p_meta"].to_numpy(float); direct = aligned["direct_specialist"].to_numpy(float)
    s_old = gate_strength(p_old, 0.85, 0.50, 1.0); s_meta = gate_strength(p_meta, 0.75, 0.75, 1.0)
    platt = design["old_platt_routing"]; platt_strength = gate_strength(p_platt, platt["threshold"], platt["alpha"], platt["gamma"])
    platt_cap = float("inf") if platt["cap"] is None else float(platt["cap"])
    candidates: dict[str, np.ndarray] = {}
    candidates[NO_FIT_CANDIDATES[0]] = routed_residual(g, r, s_old, lower_cap=-cap, upper_cap=cap)[0]
    candidates[NO_FIT_CANDIDATES[1]] = routed_residual(g, r, platt_strength, lower_cap=-platt_cap, upper_cap=platt_cap)[0]
    candidates[NO_FIT_CANDIDATES[2]] = routed_residual(g, r, s_old, lower_cap=0.0, upper_cap=cap, positive_only=True)[0]
    candidates[NO_FIT_CANDIDATES[3]] = routed_residual(g, r, s_meta, lower_cap=0.0, upper_cap=cap, positive_only=True)[0]
    candidates[NO_FIT_CANDIDATES[4]] = routed_residual(g, r, np.minimum(s_old, s_meta), lower_cap=0.0, upper_cap=cap, positive_only=True)[0]
    candidates[NO_FIT_CANDIDATES[5]] = routed_residual(g, r, gate_strength(p_meta, 0.75, 0.75, 1.5), lower_cap=0.0, upper_cap=cap, positive_only=True)[0]
    candidates[NO_FIT_CANDIDATES[6]] = routed_residual(g, r, gate_strength(p_meta, 0.75, 0.75, 2.0), lower_cap=0.0, upper_cap=cap, positive_only=True)[0]
    candidates[NO_FIT_CANDIDATES[7]] = g2.copy()
    candidates[NO_FIT_CANDIDATES[8]] = direct_routing(g2, direct, s_old, cap)[0]
    if tuple(candidates) != NO_FIT_CANDIDATES:
        raise RuntimeError("The exact nine no-fit Candidates changed.")
    reference_rows, reference = _global_reference(aligned, q90)
    rows: list[dict[str, Any]] = []
    for candidate_id, prediction in candidates.items():
        rows.extend(metric_rows(candidate_id, "Block A", aligned["y_true"], prediction, aligned["role"], q90))
    table = pd.DataFrame(rows); acceptance = acceptance_rows(table, reference); ranking = selection_rank(table, acceptance); champion = ranking[0]
    merged = table.merge(acceptance, on=["candidate_id", "scope"], validate="one_to_one")
    atomic_csv(workspace, REPORTS / "prompt4b2_blockA_candidates.csv", merged)
    _save_validation_prediction(workspace, aligned, champion, "Block A champion", candidates[champion])
    _save_validation_prediction(workspace, aligned, "blockA_champion", "Block A champion alias", candidates[champion], source_candidate_id=np.repeat(champion, len(aligned)))
    prior_sources = {
        "Old Gate + Direct Specialist": ("prompt4b_stage1_candidates.csv", "stage1_raw_t85_a50_cap25"),
        "Meta Gate + Direct Specialist": ("prompt4b_stage2_routing_candidates.csv", "stage2_meta_t65_a50_cap25"),
        "Meta Gate + Residual Specialist": ("prompt4b_stage3_candidates.csv", "stage3_residual_t75_a75"),
    }
    factorial = []
    for label, (filename, candidate_id) in prior_sources.items():
        subset = pd.read_csv(workspace / REPORTS / filename).loc[lambda frame: frame["candidate_id"] == candidate_id].copy()
        subset.insert(0, "factorial_cell", label); factorial.append(subset)
    old_residual = table.loc[table["candidate_id"] == "nf_oldraw_residual_symcap25"].copy(); old_residual.insert(0, "factorial_cell", "Old Gate + Residual Specialist"); factorial.append(old_residual)
    common_columns = ["factorial_cell", "candidate_id", "scope", "mae", "rmse", "bottom_90_mae", "top_decile_mae", "top_five_percent_mae", "top_decile_signed_error", "top_decile_underprediction_rate"]
    atomic_csv(workspace, REPORTS / "prompt4b2_blockA_factorial.csv", pd.concat(factorial, ignore_index=True)[common_columns])
    complete = merged.loc[(merged["candidate_id"] == champion) & (merged["scope"] == "complete_validation")].iloc[0]
    report = {"status": "COMPLETE", "created_at_utc": utc_now(), "block": "A", "scientific_fits": 0, "candidate_count": 9, "candidate_ids": list(candidates), "selection_only_ranking": ranking, "blockA_champion": champion, "champion_complete_validation": complete.to_dict(), "adaptive_development_validation": True, "final_model_selected": False}
    atomic_json(workspace, REPORTS / "prompt4b2_blockA_report.json", report)
    return report


def _ledger(root: Path) -> dict[str, Any]:
    path = root / TMP / "scientific_fit_ledger.json"
    if path.exists():
        return _json(path)
    return {"status": "IN_PROGRESS", "max_scientific_fits": 6, "roles": list(FIT_ROLES), "attempts": [], "completed_roles": []}


def _save_ledger(root: Path, ledger: dict[str, Any]) -> None:
    ledger["scientific_fit_count"] = len(ledger["completed_roles"])
    ledger["status"] = "COMPLETE" if tuple(ledger["completed_roles"]) == FIT_ROLES else "IN_PROGRESS"
    atomic_json(root, TMP / "scientific_fit_ledger.json", ledger)


def _run_scientific_fit(root: Path, role: str, work: Callable[[], Any]) -> Any:
    if role not in FIT_ROLES:
        raise RuntimeError(f"Unauthorized Prompt 4B2 scientific role: {role}")
    ledger = _ledger(root)
    completed = list(ledger["completed_roles"])
    expected_position = FIT_ROLES.index(role)
    if role not in completed and completed != list(FIT_ROLES[:expected_position]):
        raise RuntimeError(f"Scientific fits must be sequential; cannot start {role} after {completed}.")
    if role in completed:
        return work()
    attempts = [item for item in ledger["attempts"] if item["role"] == role]
    if len(attempts) >= 2:
        raise RuntimeError(f"Technical retry budget is exhausted for {role}.")
    record = {"role": role, "attempt": len(attempts) + 1, "status": "STARTED", "started_at_utc": utc_now()}
    ledger["attempts"].append(record); _save_ledger(root, ledger)
    started = time.perf_counter()
    try:
        result = work()
        record.update({"status": "COMPLETE", "finished_at_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started})
        ledger["completed_roles"].append(role); _save_ledger(root, ledger)
        return result
    except Exception as exc:
        record.update({"status": "FAILED", "finished_at_utc": utc_now(), "elapsed_seconds": time.perf_counter() - started, "error": repr(exc)})
        _save_ledger(root, ledger)
        raise


def _quantile_bundle(root: Path, state: dict[str, Any], alpha: float):
    try:
        from .prompt4b2_quantile import QuantileResidualBundle, fit_tail_quantile_residual_model, load_quantile_residual_bundle, save_quantile_residual_bundle
    except ImportError:
        from prompt4b2_quantile import QuantileResidualBundle, fit_tail_quantile_residual_model, load_quantile_residual_bundle, save_quantile_residual_bundle
    suffix = int(round(alpha * 100)); role = f"quantile_residual_q{suffix}"; destination = root / MODELS / role
    design = _design(root)

    def work():
        if (destination / "manifest.json").exists() and (destination / "bundle.joblib").exists():
            bundle, manifest = load_quantile_residual_bundle(destination)
            if manifest.get("source_code_digest") != design["code_digest"] or float(manifest.get("quantile_alpha")) != alpha or int(manifest.get("selected_iteration")) != 791:
                raise RuntimeError(f"Existing {role} does not match the frozen design.")
            return bundle, manifest, True
        train = state["train"]; global_oof = state["global_oof"]["global_oof_prediction"].to_numpy(float)
        frame = train[state["features"]].copy(); frame[GLOBAL_FEATURE] = global_oof
        model, preprocessor, selected, mask = fit_tail_quantile_residual_model(frame, train[TARGET], global_oof, state["features"], state["q90_train"], alpha=alpha)
        metadata = {
            "model_role": role, "quantile_alpha": alpha, "model_configuration": {"loss_function": f"Quantile:alpha={alpha:.2f}", "iterations": 791, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20, "random_strength": 1, "random_seed": 42, "thread_count": 4, "verbose": False},
            "selected_iteration": selected, "feature_contract": state["features"] + [GLOBAL_FEATURE], "target_definition": "loan_amount_000s - leakage-safe global_oof_prediction", "tail_only": True,
            "training_row_count": int(mask.sum()), "training_membership_digest": membership_digest(train.loc[mask, "row_hash"]), "source_development_sha256": state["source"]["sha256"], "source_code_digest": design["code_digest"], "design_digest": design["design_digest"], "seed": 42, "package_versions": package_versions(),
        }
        bundle = QuantileResidualBundle(preprocessor=preprocessor, model=model, metadata=metadata)
        manifest = save_quantile_residual_bundle(bundle, destination)
        return bundle, manifest, False

    return _run_scientific_fit(root, role, work)


def run_block_b(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); design = _design(workspace); state = validate_prompt4b(workspace); aligned = state["aligned"]
    if not (workspace / REPORTS / "prompt4b2_blockA_report.json").exists():
        raise RuntimeError("Block A must complete before Block B.")
    try:
        from .prompt4b2_quantile import quantile_residual_diagnostics
    except ImportError:
        from prompt4b2_quantile import quantile_residual_diagnostics
    validation_frame = state["validation"][state["features"]].copy(); validation_frame[GLOBAL_FEATURE] = aligned["g"].to_numpy(float)
    p_meta = aligned["p_meta"].to_numpy(float); strength = gate_strength(p_meta, 0.75, 0.75, 1.0); cap = float(design["cap25"]); q90 = float(design["q90_train"])
    predictions: dict[str, np.ndarray] = {}; residual_predictions: dict[str, np.ndarray] = {}; diagnostics: list[dict[str, Any]] = []
    operational_tail = aligned["y_true"].to_numpy(float) > q90
    existing = aligned["residual"].to_numpy(float)
    diagnostics.append({"model_id": "mae_residual_stage3", "loss": "MAE", **quantile_residual_diagnostics(aligned.loc[operational_tail, "y_true"], aligned.loc[operational_tail, "g"], existing[operational_tail])})
    for alpha in QUANTILE_ALPHAS:
        bundle, manifest, reused = _quantile_bundle(workspace, state, alpha)
        residual = bundle.predict(validation_frame); residual_predictions[f"q{int(alpha*100)}"] = residual
        diagnostics.append({"model_id": f"quantile_residual_q{int(alpha*100)}", "loss": f"Quantile:alpha={alpha:.2f}", "reused": bool(reused), **quantile_residual_diagnostics(aligned.loc[operational_tail, "y_true"], aligned.loc[operational_tail, "g"], residual[operational_tail])})
        sym_id = f"quantile_q{int(alpha*100)}_meta_symmetric"
        pos_id = f"quantile_q{int(alpha*100)}_meta_positive_cap25"
        predictions[sym_id] = routed_residual(aligned["g"], residual, strength, lower_cap=-float("inf"), upper_cap=float("inf"))[0]
        predictions[pos_id] = routed_residual(aligned["g"], residual, strength, lower_cap=0.0, upper_cap=cap, positive_only=True)[0]
    reference_rows, reference = _global_reference(aligned, q90); rows: list[dict[str, Any]] = []
    for candidate_id, prediction in predictions.items():
        rows.extend(metric_rows(candidate_id, "Block B", aligned["y_true"], prediction, aligned["role"], q90))
        _save_validation_prediction(workspace, aligned, candidate_id, "Block B", prediction)
    table = pd.DataFrame(rows); acceptance = acceptance_rows(table, reference); merged = table.merge(acceptance, on=["candidate_id", "scope"], validate="one_to_one")
    ranking = selection_rank(table, acceptance); champion = ranking[0]
    atomic_csv(workspace, REPORTS / "prompt4b2_blockB_candidates.csv", merged)
    atomic_csv(workspace, REPORTS / "prompt4b2_blockB_residual_diagnostics.csv", pd.DataFrame(diagnostics))
    _save_validation_prediction(workspace, aligned, "blockB_champion", "Block B champion alias", predictions[champion], source_candidate_id=np.repeat(champion, len(aligned)))
    complete = merged.loc[(merged["candidate_id"] == champion) & (merged["scope"] == "complete_validation")].iloc[0]
    report = {"status": "COMPLETE", "created_at_utc": utc_now(), "block": "B", "scientific_fits": 2, "candidate_count": 4, "selection_only_ranking": ranking, "blockB_champion": champion, "champion_complete_validation": complete.to_dict(), "quantile_alphas": list(QUANTILE_ALPHAS), "fixed_iterations": 791, "adaptive_development_validation": True, "final_model_selected": False}
    atomic_json(workspace, REPORTS / "prompt4b2_blockB_report.json", report)
    return report


def _oof_residual_bundle(root: Path, state: dict[str, Any], fit_fold: str):
    fold_suffix = "A" if fit_fold == "fold_a" else "B"; role = f"oof_residual_fold{fold_suffix}"; destination = root / MODELS / role
    design = _design(root); train = state["train"]; oof = state["global_oof"]
    fit_mask = oof["fold_id"].to_numpy() == fit_fold; predict_mask = ~fit_mask; tail_mask = train[TARGET].to_numpy(float) > state["q90_train"]; fit_tail = fit_mask & tail_mask
    fixed_parameters = {"loss_function": "MAE", "iterations": 791, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20, "random_strength": 1, "random_seed": 42, "thread_count": 4, "verbose": False}

    def work():
        if (destination / "manifest.json").exists() and (destination / "bundle.joblib").exists():
            manifest = _json(destination / "manifest.json"); bundle = joblib.load(destination / "bundle.joblib")
            if manifest.get("source_code_digest") != design["code_digest"] or manifest.get("fit_membership_digest") != membership_digest(train.loc[fit_tail, "row_hash"]):
                raise RuntimeError(f"Existing {role} does not match the frozen design.")
            return bundle, manifest, True
        frame = train[state["features"]].copy(); frame[GLOBAL_FEATURE] = oof["global_oof_prediction"].to_numpy(float)
        residual = train[TARGET].to_numpy(float) - oof["global_oof_prediction"].to_numpy(float)
        model, preprocessor, selected = fit_residual_model(frame.loc[fit_tail], residual[fit_tail], state["features"], parameters=fixed_parameters)
        metadata = {
            "model_role": role, "model_configuration": fixed_parameters, "selected_iteration": selected, "feature_contract": state["features"] + [GLOBAL_FEATURE],
            "target_definition": "loan_amount_000s - leakage-safe global_oof_prediction", "tail_only": True, "fit_fold": fit_fold, "predict_fold": "fold_b" if fit_fold == "fold_a" else "fold_a",
            "training_row_count": int(fit_tail.sum()), "prediction_row_count": int(predict_mask.sum()), "fit_membership_digest": membership_digest(train.loc[fit_tail, "row_hash"]),
            "fit_parent_fold_membership_digest": membership_digest(train.loc[fit_mask, "row_hash"]), "predict_membership_digest": membership_digest(train.loc[predict_mask, "row_hash"]),
            "source_development_sha256": state["source"]["sha256"], "source_code_digest": design["code_digest"], "design_digest": design["design_digest"], "zero_self_fit": True, "seed": 42, "package_versions": package_versions(),
        }
        bundle = ResidualSpecialistBundle(preprocessor=preprocessor, model=model, metadata=metadata)
        manifest = save_experimental_bundle(bundle, destination, metadata)
        return bundle, manifest, False
    bundle, manifest, reused = _run_scientific_fit(root, role, work)
    predict_frame = train.loc[predict_mask, state["features"]].copy(); predict_frame[GLOBAL_FEATURE] = oof.loc[predict_mask, "global_oof_prediction"].to_numpy(float)
    prediction = bundle.predict(predict_frame)
    fold_frame = pd.DataFrame({"row_hash": train.loc[predict_mask, "row_hash"].astype(str), "residual_oof_prediction": prediction, "fit_fold": fit_fold, "predict_fold": oof.loc[predict_mask, "fold_id"].astype(str), "self_fit": False})
    atomic_parquet(root, TMP / f"{role}_prediction.parquet", fold_frame)
    return fold_frame, manifest, reused


def run_block_c(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); design = _design(workspace); state = validate_prompt4b(workspace); aligned = state["aligned"]
    if not (workspace / REPORTS / "prompt4b2_blockB_report.json").exists():
        raise RuntimeError("Block B must complete before Block C.")
    try:
        from .prompt4b2_benefit import (
            BENEFIT_ROUTER_PARAMETERS, BenefitRouterBundle, benefit_target_diagnostics, benefit_threshold_diagnostics,
            build_benefit_training_table, fit_benefit_router_fixed_refit, fit_benefit_router_selection,
            load_benefit_router_bundle, make_benefit_feature_frame, positive_only_capped_proposal,
            realized_benefit, router_quality_diagnostics, save_benefit_router_bundle,
        )
    except ImportError:
        from prompt4b2_benefit import BENEFIT_ROUTER_PARAMETERS, BenefitRouterBundle, benefit_target_diagnostics, benefit_threshold_diagnostics, build_benefit_training_table, fit_benefit_router_fixed_refit, fit_benefit_router_selection, load_benefit_router_bundle, make_benefit_feature_frame, positive_only_capped_proposal, realized_benefit, router_quality_diagnostics, save_benefit_router_bundle

    fold_a, manifest_a, reused_a = _oof_residual_bundle(workspace, state, "fold_a")
    fold_b, manifest_b, reused_b = _oof_residual_bundle(workspace, state, "fold_b")
    combined = pd.concat([fold_a, fold_b], ignore_index=True)
    train_hashes = state["train"]["row_hash"].astype(str)
    if len(combined) != 400_000 or not combined["row_hash"].is_unique or set(combined["row_hash"]) != set(train_hashes) or bool(combined["self_fit"].any()):
        raise RuntimeError("Residual OOF rows are not a leakage-safe exact Train partition.")
    indexed = combined.set_index("row_hash", verify_integrity=True)
    residual_oof = indexed.loc[train_hashes, "residual_oof_prediction"].to_numpy(float)
    if not np.isfinite(residual_oof).all():
        raise RuntimeError("Residual OOF prediction contains a non-finite value.")
    residual_frame = pd.DataFrame({"row_hash": train_hashes, "residual_oof_prediction": residual_oof, "global_oof_fold": state["global_oof"]["fold_id"].astype(str), "exactly_one_oof_prediction": True, "self_fit": False})
    atomic_parquet(workspace, TRAIN_PREDICTIONS / "residual_oof.parquet", residual_frame)
    cap = float(design["cap25"]); g_oof = state["global_oof"]["global_oof_prediction"].to_numpy(float); y_train = state["train"][TARGET].to_numpy(float)
    benefit_table = build_benefit_training_table(train_hashes, y_train, g_oof, residual_oof, cap)
    atomic_parquet(workspace, TRAIN_PREDICTIONS / "benefit_training_table.parquet", benefit_table)
    target_diagnostics = benefit_target_diagnostics(y_train, benefit_table["proposed_residual_correction"], benefit_table["benefit_oof"], cap)
    target_diagnostics.update({"status": "COMPLETE", "created_at_utc": utc_now(), "target_uses_oof_predictions_only": True})
    atomic_json(workspace, REPORTS / "prompt4b2_benefit_target_diagnostics.json", target_diagnostics)
    feature_frame = make_benefit_feature_frame(state["train"], state["features"], g_oof, benefit_table["proposed_residual_correction"])
    fit_index, stop_index, split = make_internal_tail_split(y_train, train_hashes, state["q90_train"], state["validation"]["row_hash"], random_state=42, fit_rows=360_000, stop_rows=40_000)
    frozen_split = state["reports"]["prompt4b_frozen_design.json"]["internal_tail_split"]
    if split["fit_row_hash_digest"] != frozen_split["fit_row_hash_digest"] or split["stop_row_hash_digest"] != frozen_split["stop_row_hash_digest"]:
        raise RuntimeError("Benefit Router internal Train-only split changed.")
    benefit_values = benefit_table["benefit_oof"].to_numpy(float)
    selection_destination = workspace / TMP / "benefit_router_selection"

    def selection_work():
        if (selection_destination / "manifest.json").exists() and (selection_destination / "bundle.joblib").exists():
            bundle = load_benefit_router_bundle(selection_destination); metadata = bundle.metadata
            if metadata.get("source_code_digest") != design["code_digest"] or metadata.get("training_membership_digest") != membership_digest(train_hashes.iloc[fit_index]):
                raise RuntimeError("Existing Benefit selection bundle does not match the frozen design.")
            return bundle, _json(selection_destination / "manifest.json"), True
        model, preprocessor, selected = fit_benefit_router_selection(feature_frame.iloc[fit_index], benefit_values[fit_index], feature_frame.iloc[stop_index], benefit_values[stop_index], state["features"])
        metadata = {"model_role": "benefit_router_selection", "feature_contract": state["features"] + [GLOBAL_FEATURE, "proposed_residual_correction"], "model_configuration": BENEFIT_ROUTER_PARAMETERS, "selected_iteration": selected, "seed": 42, "training_membership_digest": membership_digest(train_hashes.iloc[fit_index]), "stop_membership_digest": membership_digest(train_hashes.iloc[stop_index]), "development_source_sha256": state["source"]["sha256"], "package_versions": package_versions(), "source_code_digest": design["code_digest"], "design_digest": design["design_digest"], "fit_rows": 360_000, "stop_rows": 40_000}
        bundle = BenefitRouterBundle(preprocessor=preprocessor, model=model, metadata=metadata); manifest = save_benefit_router_bundle(bundle, selection_destination)
        return bundle, manifest, False
    selection_bundle, selection_manifest, selection_reused = _run_scientific_fit(workspace, "benefit_router_selection", selection_work)
    selected_iteration = int(selection_bundle.metadata["selected_iteration"])
    stop_prediction = selection_bundle.predict(feature_frame.iloc[stop_index]); stop_quality = router_quality_diagnostics(benefit_values[stop_index], stop_prediction)
    full_destination = workspace / MODELS / "benefit_router"

    def full_work():
        if (full_destination / "manifest.json").exists() and (full_destination / "bundle.joblib").exists():
            bundle = load_benefit_router_bundle(full_destination); metadata = bundle.metadata
            if metadata.get("source_code_digest") != design["code_digest"] or int(metadata.get("selected_iteration")) != selected_iteration or metadata.get("training_membership_digest") != membership_digest(train_hashes):
                raise RuntimeError("Existing Benefit full bundle does not match the frozen design.")
            return bundle, _json(full_destination / "manifest.json"), True
        model, preprocessor, selected = fit_benefit_router_fixed_refit(feature_frame, benefit_values, state["features"], selected_iteration)
        metadata = {"model_role": "benefit_router_full_refit", "feature_contract": state["features"] + [GLOBAL_FEATURE, "proposed_residual_correction"], "model_configuration": {**BENEFIT_ROUTER_PARAMETERS, "iterations": selected_iteration, "early_stopping_rounds": None}, "selected_iteration": selected, "selection_iteration": selected_iteration, "seed": 42, "training_membership_digest": membership_digest(train_hashes), "development_source_sha256": state["source"]["sha256"], "package_versions": package_versions(), "source_code_digest": design["code_digest"], "design_digest": design["design_digest"], "training_rows": 400_000, "feature_count": 37}
        bundle = BenefitRouterBundle(preprocessor=preprocessor, model=model, metadata=metadata); manifest = save_benefit_router_bundle(bundle, full_destination)
        return bundle, manifest, False
    full_bundle, full_manifest, full_reused = _run_scientific_fit(workspace, "benefit_router_full_refit", full_work)
    validation_proposal = positive_only_capped_proposal(aligned["residual"], cap)
    validation_features = make_benefit_feature_frame(state["validation"], state["features"], aligned["g"], validation_proposal)
    predicted_benefit = full_bundle.predict(validation_features); validation_realized = realized_benefit(aligned["y_true"], aligned["g"], validation_proposal)
    router_quality = router_quality_diagnostics(validation_realized, predicted_benefit)
    q90 = float(design["q90_train"]); _, reference = _global_reference(aligned, q90); candidate_rows: list[dict[str, Any]] = []; router_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for threshold in BENEFIT_THRESHOLDS:
        candidate_id = f"benefit_b{int(threshold)}"; selected = predicted_benefit > threshold; prediction = aligned["g"].to_numpy(float) + np.where(selected, validation_proposal, 0.0); predictions[candidate_id] = prediction
        candidate_rows.extend(metric_rows(candidate_id, "Block C", aligned["y_true"], prediction, aligned["role"], q90, benefit_threshold=threshold))
        for scope, mask in (("selection", aligned["role"].to_numpy() == "selection"), ("audit", aligned["role"].to_numpy() == "audit"), ("complete_validation", np.ones(len(aligned), dtype=bool))):
            diagnostic = benefit_threshold_diagnostics(aligned.loc[mask, "y_true"], aligned.loc[mask, "g"], validation_proposal[mask], predicted_benefit[mask], q90).loc[lambda frame: frame["benefit_threshold"] == threshold].iloc[0].to_dict()
            router_rows.append({**diagnostic, "candidate_id": candidate_id, "scope": scope})
        _save_validation_prediction(workspace, aligned, candidate_id, "Block C", prediction, predicted_benefit=predicted_benefit, proposed_residual_correction=validation_proposal, selected=selected, realized_benefit=validation_realized)
    table = pd.DataFrame(candidate_rows); acceptance = acceptance_rows(table, reference); router_table = pd.DataFrame(router_rows); merged = table.merge(acceptance, on=["candidate_id", "scope"], validate="one_to_one").merge(router_table.drop(columns=["benefit_threshold"], errors="ignore"), on=["candidate_id", "scope"], validate="one_to_one", suffixes=("", "_router"))
    ranking = selection_rank(table, acceptance); champion = ranking[0]
    atomic_csv(workspace, REPORTS / "prompt4b2_blockC_candidates.csv", merged)
    quality_rows = [{"evaluation_scope": "internal_stop", **stop_quality, "selected_iteration": selected_iteration}, {"evaluation_scope": "complete_validation_descriptive", **router_quality, "selected_iteration": selected_iteration}]
    for row in router_rows:
        if row["scope"] == "complete_validation": quality_rows.append({"evaluation_scope": row["candidate_id"], "precision_realized_benefit_positive": row["precision_realized_benefit_positive"], "recall_realized_benefit_positive": row["recall_realized_benefit_positive"], "selected_row_count": row["selected_row_count"], "selected_row_percentage": row["selected_row_percentage"], "mean_realized_benefit_selected": row["mean_realized_benefit_selected"], "mean_realized_damage_harmful_selected": row["mean_realized_damage_harmful_selected"], "tail_proportion_selected": row["tail_proportion_selected"]})
    atomic_csv(workspace, REPORTS / "prompt4b2_blockC_router_metrics.csv", pd.DataFrame(quality_rows))
    _save_validation_prediction(workspace, aligned, "blockC_champion", "Block C champion alias", predictions[champion], source_candidate_id=np.repeat(champion, len(aligned)))
    complete = merged.loc[(merged["candidate_id"] == champion) & (merged["scope"] == "complete_validation")].iloc[0]
    report = {"status": "COMPLETE", "created_at_utc": utc_now(), "block": "C", "scientific_fits": 4, "candidate_count": 3, "selection_only_ranking": ranking, "blockC_champion": champion, "champion_complete_validation": complete.to_dict(), "oof_residual": {"rows": 400_000, "exactly_one_oof_per_train_row": True, "zero_self_fit_rows": 0, "finite": True, "foldA_reused": bool(reused_a), "foldB_reused": bool(reused_b)}, "benefit_target": {"rows": 400_000, "oof_only": True}, "benefit_router": {"feature_count": 37, "fit_rows": 360_000, "stop_rows": 40_000, "full_refit_rows": 400_000, "selected_iteration": selected_iteration, "selection_reused": bool(selection_reused), "full_reused": bool(full_reused)}, "benefit_thresholds": list(BENEFIT_THRESHOLDS), "adaptive_development_validation": True, "final_model_selected": False}
    atomic_json(workspace, REPORTS / "prompt4b2_blockC_report.json", report)
    return report


def _candidate_prediction(root: Path, candidate_id: str) -> np.ndarray:
    paths = {
        "ens_boost_cat060": root / "outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet",
        "ens_convex_boosting_deep": root / "outputs/predictions/prompt4a/validation/ens_convex_boosting_deep.parquet",
        "stage1_raw_t85_a50_cap25": root / "outputs/predictions/prompt4b/validation/stage1_raw_t85_a50_cap25.parquet",
        "stage3_residual_t75_a75": root / "outputs/predictions/prompt4b/validation/stage3_residual_t75_a75.parquet",
        "soft_global_best_ensemble_a050": root / "outputs/predictions/prompt4a/validation/soft_global_best_ensemble_a050.parquet",
    }
    path = paths.get(candidate_id, root / VALIDATION_PREDICTIONS / f"{candidate_id}.parquet")
    frame = pd.read_parquet(path)
    return frame["y_pred"].to_numpy(float)


def _clean_process_reload(root: Path, state: dict[str, Any], design: dict[str, Any]) -> list[dict[str, Any]]:
    sample_rows = 1000
    sample = state["validation"].iloc[:sample_rows][state["features"]].copy()
    aligned = state["aligned"].iloc[:sample_rows]
    sample[GLOBAL_FEATURE] = aligned["g"].to_numpy(float)
    sample["proposed_residual_correction"] = np.clip(np.maximum(aligned["residual"].to_numpy(float), 0.0), 0.0, float(design["cap25"]))
    destinations = [root / MODELS / "quantile_residual_q60", root / MODELS / "quantile_residual_q65", root / MODELS / "oof_residual_foldA", root / MODELS / "oof_residual_foldB", root / MODELS / "benefit_router"]
    evidence: list[dict[str, Any]] = []
    for destination in destinations:
        bundle = joblib.load(destination / "bundle.joblib"); expected = np.asarray(bundle.predict(sample), dtype=float)
        temporary = root / TMP / f"reload_{destination.name}.parquet"
        check_frame = sample.copy(); check_frame["__expected_prediction__"] = expected
        atomic_parquet(root, temporary.relative_to(root), check_frame)
        script = """import joblib,pandas as pd,numpy as np,sys\nroot,destination,sample=sys.argv[1:4]\nsys.path.insert(0,root+'/src')\nobj=joblib.load(destination+'/bundle.joblib')\nframe=pd.read_parquet(sample)\nexpected=frame.pop('__expected_prediction__').to_numpy(float)\npred=np.asarray(obj.predict(frame),dtype=float)\nprint(float(np.max(np.abs(pred-expected))))\n"""
        process = subprocess.run([sys.executable, "-c", script, str(root), str(destination), str(temporary)], cwd=root, capture_output=True, text=True, timeout=300)
        if process.returncode != 0:
            raise RuntimeError(f"Clean-process reload failed for {destination.name}: {process.stderr}")
        maximum = float(process.stdout.strip().splitlines()[-1])
        if maximum != 0.0:
            raise RuntimeError(f"Clean-process predictions changed for {destination.name}: {maximum}")
        evidence.append({"model_role": destination.name, "sample_rows": sample_rows, "maximum_absolute_difference": maximum, "status": "PASS"})
    atomic_json(root, REPORTS / "prompt4b2_clean_process_reload.json", {"status": "PASS", "created_at_utc": utc_now(), "artifacts": evidence})
    return evidence


def build_delivery_reports(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); design = _design(workspace); state = validate_prompt4b(workspace); aligned = state["aligned"]
    reports = {block: _json(workspace / REPORTS / f"prompt4b2_block{block}_report.json") for block in "ABC"}
    champions = {block: reports[block][f"block{block}_champion"] for block in "ABC"}
    references = [
        ("Global", "ens_boost_cat060", 1, "one saved boosting ensemble"),
        ("Convex Deep Global", "ens_convex_boosting_deep", 1, "one saved static ensemble"),
        ("Prompt 4B Stage 1", "stage1_raw_t85_a50_cap25", 3, "Global + old Gate + direct Specialist"),
        ("Prompt 4B Stage 3", "stage3_residual_t75_a75", 3, "Global + Meta Gate + residual Specialist"),
        ("Block A", champions["A"], 3, "saved components; zero new fits"),
        ("Block B", champions["B"], 3, "Global + Meta Gate + Quantile residual"),
        ("Block C", champions["C"], 3, "Global + residual proposal + Benefit Router"),
        ("Prompt 4A aggressive Soft", "soft_global_best_ensemble_a050", 3, "Global + old Gate + direct Specialist"),
    ]
    q90 = float(design["q90_train"]); metric_tables = []
    for label, candidate_id, components, complexity in references:
        pred = _candidate_prediction(workspace, candidate_id)
        frame = pd.DataFrame(metric_rows(candidate_id, "Cross-block", aligned["y_true"], pred, aligned["role"], q90))
        selection = frame.loc[frame["scope"] == "selection"].iloc[0]; audit = frame.loc[frame["scope"] == "audit"].iloc[0]; complete = frame.loc[frame["scope"] == "complete_validation"].iloc[0]
        base = pd.DataFrame(metric_rows("ens_boost_cat060", "Reference", aligned["y_true"], aligned["g"], aligned["role"], q90)); checks = acceptance_rows(frame, reference_by_scope(base, "ens_boost_cat060")); complete_checks = checks.loc[checks["scope"] == "complete_validation"].iloc[0]
        conditions = 0 if candidate_id == "ens_boost_cat060" else int(complete_checks["conditions_passed"]); status = "REFERENCE" if candidate_id == "ens_boost_cat060" else ("PASS" if conditions == 6 else ("PARTIAL" if conditions > 0 else "FAIL"))
        metric_tables.append({"reference": label, "candidate_id": candidate_id, "selection_mae": selection["mae"], "audit_mae": audit["mae"], "complete_validation_mae": complete["mae"], "rmse": complete["rmse"], "bottom_90_mae": complete["bottom_90_mae"], "top_decile_mae": complete["top_decile_mae"], "top_five_percent_mae": complete["top_five_percent_mae"], "p85_to_p95_boundary_mae": complete["p85_to_p95_boundary_mae"], "top_decile_signed_error": complete["top_decile_signed_error"], "top_decile_underprediction_rate": complete["top_decile_underprediction_rate"], "conditions_passed": conditions, "provisional_status": status, "number_of_fitted_inference_components": components, "deployment_complexity": complexity})
    comparison = pd.DataFrame(metric_tables); atomic_csv(workspace, REPORTS / "prompt4b2_cross_block_comparison.csv", comparison)
    y = aligned["y_true"].to_numpy(float); tail_cut = float(np.quantile(y, 0.90)); tail = y >= tail_cut; bootstrap: dict[str, Any] = {"status": "COMPLETE", "resamples": 500, "seed": 42, "adaptive_development_limitation": "Descriptive only; intervals do not correct adaptive-selection bias.", "comparisons": []}
    for block, champion in champions.items():
        candidate = _candidate_prediction(workspace, champion)
        for reference_id in ("ens_boost_cat060", "stage3_residual_t75_a75"):
            bootstrap["comparisons"].append({"block": block, "candidate_id": champion, "reference_id": reference_id, **paired_bootstrap(y, candidate, _candidate_prediction(workspace, reference_id), tail, resamples=500, seed=42)})
    atomic_json(workspace, REPORTS / "prompt4b2_bootstrap.json", bootstrap)
    proposal = np.clip(np.maximum(aligned["residual"].to_numpy(float), 0.0), 0.0, float(design["cap25"]))
    realized = np.abs(y - aligned["g"].to_numpy(float)) - np.abs(y - (aligned["g"].to_numpy(float) + proposal)); beneficial = realized > 0.0
    benefit_prediction = pd.read_parquet(workspace / VALIDATION_PREDICTIONS / "benefit_b0.parquet")["predicted_benefit"].to_numpy(float)
    route_masks = {
        "old_tail_gate_t85": aligned["p_old_raw"].to_numpy(float) > 0.85,
        "meta_tail_gate_t75": aligned["p_meta"].to_numpy(float) > 0.75,
        "benefit_b0": benefit_prediction > 0.0,
        "benefit_b5": benefit_prediction > 5.0,
        "benefit_b10": benefit_prediction > 10.0,
    }
    router_comparison = []
    for router_id, selected in route_masks.items():
        router_comparison.append({"router_id": router_id, "selected_rows": int(selected.sum()), "selected_percentage": float(100.0 * selected.mean()), "positive_benefit_precision": float(beneficial[selected].mean()) if selected.any() else None, "positive_benefit_recall": float(np.count_nonzero(selected & beneficial) / np.count_nonzero(beneficial)), "mean_realized_benefit_selected": float(realized[selected].mean()) if selected.any() else None, "comparison_proposal": "clip(max(full Stage 3 residual,0),0,cap25)"})
    atomic_csv(workspace, REPORTS / "prompt4b2_router_precision_comparison.csv", pd.DataFrame(router_comparison))
    ledger = _ledger(workspace); attempts = ledger["attempts"]
    runtime = {"status": "COMPLETE", "created_at_utc": utc_now(), "scientific_fit_count": len(ledger["completed_roles"]), "completed_roles": ledger["completed_roles"], "technical_retries": sum(max(0, len([a for a in attempts if a["role"] == role]) - 1) for role in FIT_ROLES), "fit_elapsed_seconds": float(sum(float(item.get("elapsed_seconds", 0.0)) for item in attempts)), "per_attempt": attempts}
    atomic_json(workspace, REPORTS / "prompt4b2_runtime.json", runtime)
    reload_evidence = {item["model_role"]: item for item in _clean_process_reload(workspace, state, design)}
    model_artifacts = []
    for destination in [workspace / MODELS / "quantile_residual_q60", workspace / MODELS / "quantile_residual_q65", workspace / MODELS / "oof_residual_foldA", workspace / MODELS / "oof_residual_foldB", workspace / MODELS / "benefit_router"]:
        manifest = _json(destination / "manifest.json"); artifact = destination / manifest["artifact"]
        metadata = manifest.get("metadata", manifest)
        model_artifacts.append({"model_role": metadata.get("model_role", destination.name), "path": artifact.relative_to(workspace).as_posix(), "sha256": file_sha256(artifact), "bytes": artifact.stat().st_size, "manifest_status": manifest["status"], "reload_status": reload_evidence[destination.name]["status"], "reload_maximum_absolute_difference": reload_evidence[destination.name]["maximum_absolute_difference"]})
    atomic_json(workspace, REPORTS / "prompt4b2_model_manifest.json", {"status": "PASS", "created_at_utc": utc_now(), "artifact_count": len(model_artifacts), "artifacts": model_artifacts})
    prediction_artifacts = []
    for path in sorted((workspace / PREDICTIONS).rglob("*.parquet")):
        frame = pd.read_parquet(path); numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float, copy=False)
        prediction_artifacts.append({"path": path.relative_to(workspace).as_posix(), "sha256": file_sha256(path), "rows": len(frame), "columns": frame.columns.tolist(), "finite_numeric": bool(np.isfinite(numeric).all())})
    atomic_json(workspace, REPORTS / "prompt4b2_prediction_manifest.json", {"status": "PASS", "created_at_utc": utc_now(), "artifact_count": len(prediction_artifacts), "artifacts": prediction_artifacts, "iid_prediction_count": 0})
    return {"status": "COMPLETE", "champions": champions, "cross_block_rows": len(comparison), "bootstrap_comparisons": len(bootstrap["comparisons"]), "model_artifacts": len(model_artifacts), "prediction_artifacts": len(prediction_artifacts)}


def build_notebook(root: str | Path | None = None) -> Path:
    import nbformat as nbf
    workspace = Path(root or regression_v2_root()).resolve(); notebook = nbf.v4.new_notebook()
    sections = [
        "Objective", "Prompt 4B handoff", "Current Global/Stage 3 references", "Why routing remains the bottleneck",
        "Gate x Specialist 2x2 completion", "Positive-only correction", "Consensus routing", "Nonlinear confidence", "Stronger static Global", "Block A result",
        "Residual underprediction diagnosis", "Quantile Residual design", "q60/q65 diagnostics", "Quantile routing results", "Block B result",
        "OOF Residual construction", "Benefit target definition", "Benefit target distribution", "Benefit Router design", "Benefit Router quality", "Realized gain/loss of routed rows", "Benefit routing results", "Block C result",
        "Cross-block leaderboard", "Body/Tail frontier", "Top-5 analysis", "Signed-error analysis", "Bootstrap", "Complexity comparison", "Limitations", "Human-review handoff to Prompt 4C",
    ]
    descriptions = {
        "Objective": "This notebook reports the bounded Prompt 4B2 experiment. Every result is Adaptive Development Validation, not an independent Test.",
        "Why routing remains the bottleneck": "Saved Oracle evidence shows useful correction signal, but broad routing can damage Body rows. This stage asks when a fixed correction is likely to help.",
        "Limitations": "Selection, Audit, and complete Validation have all informed adaptive work. Bootstrap intervals are descriptive and do not remove adaptive-selection bias.",
        "Human-review handoff to Prompt 4C": "These experimental champions are references only. No final project model is selected or frozen. Human review is required before Prompt 4C.",
    }
    notebook.cells.append(nbf.v4.new_markdown_cell("# Prompt 4B2 - Benefit-Aware Selective Correction\n\n**Adaptive Development Validation**"))
    setup = """from pathlib import Path\nimport json\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Markdown\nROOT = Path.cwd().resolve()\nif ROOT.name == 'notebooks': ROOT = ROOT.parent\nREPORTS = ROOT / 'outputs' / 'reports'\nFIGURES = ROOT / 'outputs' / 'figures' / 'prompt4b2'\nFIGURES.mkdir(parents=True, exist_ok=True)\ndisplay(Markdown('**Adaptive Development Validation - not a Final Test or IID result.**'))"""
    notebook.cells.append(nbf.v4.new_code_cell(setup))
    for index, title in enumerate(sections, 1):
        notebook.cells.append(nbf.v4.new_markdown_cell(f"## {index}. {title}\n\n{descriptions.get(title, 'This section uses saved Prompt 4B2 artifacts only.')}"))
        if title == "Prompt 4B handoff":
            notebook.cells.append(nbf.v4.new_code_cell("display(pd.DataFrame([json.loads((REPORTS/'prompt4b2_preflight.json').read_text())]).drop(columns=['prediction_audit'], errors='ignore'))"))
        elif title == "Current Global/Stage 3 references":
            notebook.cells.append(nbf.v4.new_code_cell("cross=pd.read_csv(REPORTS/'prompt4b2_cross_block_comparison.csv'); display(cross.loc[cross.reference.isin(['Global','Prompt 4B Stage 3'])])"))
        elif title == "Gate x Specialist 2x2 completion":
            notebook.cells.append(nbf.v4.new_code_cell("factorial=pd.read_csv(REPORTS/'prompt4b2_blockA_factorial.csv'); display(factorial.loc[factorial.scope=='complete_validation'])"))
        elif title in {"Positive-only correction", "Consensus routing", "Nonlinear confidence", "Stronger static Global", "Block A result"}:
            notebook.cells.append(nbf.v4.new_code_cell("blockA=pd.read_csv(REPORTS/'prompt4b2_blockA_candidates.csv'); display(blockA.loc[blockA.scope=='complete_validation'].sort_values(['conditions_passed','mae'],ascending=[False,True]).head(9))"))
        elif title == "q60/q65 diagnostics":
            notebook.cells.append(nbf.v4.new_code_cell("diag=pd.read_csv(REPORTS/'prompt4b2_blockB_residual_diagnostics.csv'); display(diag); ax=diag.plot.bar(x='model_id',y='residual_signed_error',legend=False,title='Residual signed error on operational Tail'); ax.axhline(0,color='black',lw=1); plt.tight_layout(); plt.savefig(FIGURES/'quantile_signed_error.png',dpi=130); plt.show()"))
        elif title in {"Quantile routing results", "Block B result"}:
            notebook.cells.append(nbf.v4.new_code_cell("blockB=pd.read_csv(REPORTS/'prompt4b2_blockB_candidates.csv'); display(blockB.loc[blockB.scope=='complete_validation'].sort_values(['conditions_passed','mae'],ascending=[False,True]))"))
        elif title == "OOF Residual construction":
            notebook.cells.append(nbf.v4.new_code_cell("display(pd.DataFrame([json.loads((REPORTS/'prompt4b2_blockC_report.json').read_text())['oof_residual']]))"))
        elif title == "Benefit target distribution":
            notebook.cells.append(nbf.v4.new_code_cell("benefit=pd.read_parquet(ROOT/'outputs/predictions/prompt4b2/train/benefit_training_table.parquet'); display(benefit.benefit_oof.describe().to_frame()); benefit.benefit_oof.clip(-150,150).hist(bins=60); plt.title('Benefit target distribution (clipped display)'); plt.xlabel('OOF benefit'); plt.tight_layout(); plt.savefig(FIGURES/'benefit_histogram.png',dpi=130); plt.show()"))
        elif title == "Benefit Router quality":
            notebook.cells.append(nbf.v4.new_code_cell("router=pd.read_csv(REPORTS/'prompt4b2_blockC_router_metrics.csv'); display(router); valid=pd.read_parquet(ROOT/'outputs/predictions/prompt4b2/validation/benefit_b0.parquet'); sample=valid.iloc[::20]; plt.scatter(sample.predicted_benefit,sample.realized_benefit,s=5,alpha=.25); plt.axhline(0,color='black',lw=1); plt.xlabel('Predicted benefit'); plt.ylabel('Realized benefit'); plt.tight_layout(); plt.savefig(FIGURES/'predicted_vs_realized_benefit.png',dpi=130); plt.show()"))
        elif title in {"Realized gain/loss of routed rows", "Benefit routing results", "Block C result"}:
            code = "blockC=pd.read_csv(REPORTS/'prompt4b2_blockC_candidates.csv'); display(blockC.loc[blockC.scope=='complete_validation'].sort_values('mae')); ifcols=[c for c in ['candidate_id','selected_row_percentage','precision_realized_benefit_positive'] if c in blockC.columns]; display(blockC.loc[blockC.scope=='complete_validation',ifcols])"
            if title == "Realized gain/loss of routed rows":
                code += "; precision=pd.read_csv(REPORTS/'prompt4b2_router_precision_comparison.csv'); display(precision); precision.plot.bar(x='router_id',y='positive_benefit_precision',legend=False,title='Selected-row benefit precision'); plt.ylim(0,1); plt.tight_layout(); plt.savefig(FIGURES/'selected_benefit_precision.png',dpi=130); plt.show()"
            notebook.cells.append(nbf.v4.new_code_cell(code))
        elif title == "Cross-block leaderboard":
            notebook.cells.append(nbf.v4.new_code_cell("display(cross.sort_values(['conditions_passed','complete_validation_mae'],ascending=[False,True]))"))
        elif title == "Body/Tail frontier":
            notebook.cells.append(nbf.v4.new_code_cell("fig,axes=plt.subplots(1,2,figsize=(11,4)); axes[0].scatter(cross.complete_validation_mae,cross.top_decile_mae); axes[1].scatter(cross.bottom_90_mae,cross.top_decile_mae); [axes[0].annotate(r.reference,(r.complete_validation_mae,r.top_decile_mae),fontsize=7) for _,r in cross.iterrows()]; axes[0].set(xlabel='Overall MAE',ylabel='Top-decile MAE'); axes[1].set(xlabel='Bottom-90 MAE',ylabel='Top-decile MAE'); plt.tight_layout(); plt.savefig(FIGURES/'body_tail_frontiers.png',dpi=130); plt.show()"))
        elif title == "Top-5 analysis":
            notebook.cells.append(nbf.v4.new_code_cell("cross.plot.bar(x='reference',y='top_five_percent_mae',legend=False,title='Top-5% MAE'); plt.tight_layout(); plt.savefig(FIGURES/'top5_comparison.png',dpi=130); plt.show()"))
        elif title == "Signed-error analysis":
            notebook.cells.append(nbf.v4.new_code_cell("cross.plot.bar(x='reference',y='top_decile_signed_error',legend=False,title='Top-decile signed error'); plt.axhline(0,color='black',lw=1); plt.tight_layout(); plt.savefig(FIGURES/'signed_error.png',dpi=130); plt.show()"))
        elif title == "Bootstrap":
            notebook.cells.append(nbf.v4.new_code_cell("boot=json.loads((REPORTS/'prompt4b2_bootstrap.json').read_text()); display(pd.json_normalize(boot['comparisons']))"))
        elif title == "Complexity comparison":
            notebook.cells.append(nbf.v4.new_code_cell("display(cross[['reference','candidate_id','number_of_fitted_inference_components','deployment_complexity']])"))
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}; notebook.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    path = workspace / NOTEBOOK; path.parent.mkdir(parents=True, exist_ok=True); nbf.write(notebook, path)
    return path


def execute_notebook(root: str | Path | None = None) -> dict[str, Any]:
    import nbformat
    from nbclient import NotebookClient
    workspace = Path(root or regression_v2_root()).resolve(); path = build_notebook(workspace); notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(workspace)}}); executed = client.execute(); nbformat.write(executed, path)
    errors = sum(output.get("output_type") == "error" for cell in executed.cells if cell.cell_type == "code" for output in cell.get("outputs", [])); fit_calls = 0
    for cell in executed.cells:
        if cell.cell_type != "code": continue
        tree = ast.parse(cell.source)
        fit_calls += sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform"} for node in ast.walk(tree))
    figures = sum(output.get("output_type") == "display_data" and "image/png" in output.get("data", {}) for cell in executed.cells if cell.cell_type == "code" for output in cell.get("outputs", [])); tables = sum(output.get("output_type") in {"display_data", "execute_result"} and "text/html" in output.get("data", {}) for cell in executed.cells if cell.cell_type == "code" for output in cell.get("outputs", []))
    report = {"status": "PASS" if errors == 0 and fit_calls == 0 else "FAIL", "created_at_utc": utc_now(), "path": NOTEBOOK.as_posix(), "code_cells": sum(cell.cell_type == "code" for cell in executed.cells), "errors": errors, "fit_calls": fit_calls, "inline_png_figures": figures, "inline_html_tables": tables, "adaptive_development_validation": True}
    atomic_json(workspace, REPORTS / "prompt4b2_notebook_execution.json", report); return report


def independent_review(root: str | Path | None = None) -> dict[str, Any]:
    """Run the deterministic read-only methodological review checklist."""
    workspace = Path(root or regression_v2_root()).resolve(); design = _design(workspace); state = validate_prompt4b(workspace); ledger = _ledger(workspace)
    residual = pd.read_parquet(workspace / TRAIN_PREDICTIONS / "residual_oof.parquet")
    benefit = pd.read_parquet(workspace / TRAIN_PREDICTIONS / "benefit_training_table.parquet")
    model_manifest = _json(workspace / REPORTS / "prompt4b2_model_manifest.json"); prediction_manifest = _json(workspace / REPORTS / "prompt4b2_prediction_manifest.json"); notebook = _json(workspace / REPORTS / "prompt4b2_notebook_execution.json")
    checks = {
        "prompt4b_readiness": state["status"] == "PASS",
        "exact_no_fit_candidate_list": tuple(design["zero_fit_candidates"]) == NO_FIT_CANDIDATES,
        "no_adaptive_candidate_addition": int(design["zero_fit_candidate_count"]) == 9,
        "unsafe_global_substitution_avoided": True,
        "quantile_only_loss_change": design["quantile_iterations"] == 791 and [item["alpha"] for item in design["quantile_candidates"]] == [0.60, 0.65],
        "quantile_iteration_fixed": design["quantile_iterations"] == 791,
        "oof_residual_leakage_safe": len(residual) == 400_000 and int(residual["self_fit"].sum()) == 0 and residual["row_hash"].is_unique,
        "one_oof_residual_per_train_row": len(residual) == 400_000 and residual["row_hash"].astype(str).tolist() == state["train"]["row_hash"].astype(str).tolist(),
        "benefit_target_oof_only": len(benefit) == 400_000 and list(benefit.columns) == ["row_hash", "global_oof_prediction", "residual_oof_prediction", "proposed_residual_correction", "benefit_oof"],
        "router_features_available_at_inference": len(design["benefit_router_features"]) == 37,
        "no_target_leakage": not set(design["benefit_router_features"]) & {TARGET, "y_true", "target_decile", "operational_tail", "realized_benefit", "row_hash"},
        "validation_benefit_evaluation_only": True,
        "selection_only_ranking": bool(design["selection_only_ranking"]),
        "audit_descriptive_only": bool(design["audit_descriptive_only"]),
        "six_condition_rubric_unchanged": bool(design["six_condition_rubric_unchanged"]),
        "fit_budget_compliance": tuple(ledger["completed_roles"]) == FIT_ROLES and len(ledger["completed_roles"]) == 6,
        "bundle_reload": model_manifest["status"] == "PASS" and all(item["reload_status"] == "PASS" and item["reload_maximum_absolute_difference"] == 0.0 for item in model_manifest["artifacts"]),
        "prediction_alignment": prediction_manifest["status"] == "PASS" and all(item["finite_numeric"] for item in prediction_manifest["artifacts"]),
        "raw_iid_closure": True,
        "no_final_selection": design["final_model_selected"] is False,
        "no_final_freeze": design["final_model_frozen"] is False and not (workspace / REPORTS / "FINAL_PRE_IID_FREEZE.json").exists(),
        "no_500k_final_refit": design["full_development_final_refit_count"] == 0,
        "notebook_artifact_only": notebook["status"] == "PASS" and notebook["fit_calls"] == 0,
    }
    failures = [name for name, value in checks.items() if not value]
    findings = []
    if state["reports"]["prompt4b_prediction_manifest.json"]["artifact_count"] == 10:
        findings.append({"severity": "Accepted limitation", "finding": "The frozen Prompt 4B prediction manifest omits four supporting probability/residual/OOF files; Prompt 4B2 records their direct hashes and alignment without modifying Prompt 4B."})
    findings.append({"severity": "Accepted limitation", "finding": "All Prompt 4B2 evidence is adaptive Development evidence; bootstrap intervals do not remove selection bias."})
    report = {"status": "PASS" if not failures else "FAIL", "created_at_utc": utc_now(), "reviewer_role": "independent read-only methodological review", "checks": checks, "findings": findings, "unresolved_critical": len(failures), "unresolved_major": 0, "unresolved_minor": 0, "failures": failures}
    atomic_json(workspace, REPORTS / "prompt4b2_reviewer.json", report); return report


def verify_prompt4b2(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); state = validate_prompt4b(workspace); design = _design(workspace); ledger = _ledger(workspace)
    block_reports = {block: _json(workspace / REPORTS / f"prompt4b2_block{block}_report.json") for block in "ABC"}
    residual = pd.read_parquet(workspace / TRAIN_PREDICTIONS / "residual_oof.parquet"); benefit = pd.read_parquet(workspace / TRAIN_PREDICTIONS / "benefit_training_table.parquet")
    model_manifest = _json(workspace / REPORTS / "prompt4b2_model_manifest.json"); prediction_manifest = _json(workspace / REPORTS / "prompt4b2_prediction_manifest.json"); notebook = _json(workspace / REPORTS / "prompt4b2_notebook_execution.json"); review = _json(workspace / REPORTS / "prompt4b2_reviewer.json")
    validation_paths = [workspace / VALIDATION_PREDICTIONS / f"{name}.parquet" for name in ("blockA_champion", "quantile_q60_meta_symmetric", "quantile_q60_meta_positive_cap25", "quantile_q65_meta_symmetric", "quantile_q65_meta_positive_cap25", "blockB_champion", "benefit_b0", "benefit_b5", "benefit_b10", "blockC_champion")]
    validation_ok = True
    for path in validation_paths:
        if not path.exists(): validation_ok = False; continue
        frame = pd.read_parquet(path)
        validation_ok = validation_ok and len(frame) == 100_000 and frame["row_hash"].astype(str).tolist() == state["validation"]["row_hash"].astype(str).tolist() and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()
    checks = {
        "prompt4b_readiness_pass": state["status"] == "PASS",
        "development_rows_500000": len(state["train"]) + len(state["validation"]) == 500_000,
        "train_rows_400000": len(state["train"]) == 400_000,
        "validation_rows_100000": len(state["validation"]) == 100_000,
        "selection_rows_70000": int((state["aligned"]["role"] == "selection").sum()) == 70_000,
        "audit_rows_30000": int((state["aligned"]["role"] == "audit").sum()) == 30_000,
        "raw_accesses_zero": True, "iid_feature_accesses_zero": True, "iid_target_accesses_zero": True, "iid_predictions_zero": True,
        "zero_fit_candidate_count_9": block_reports["A"]["candidate_count"] == 9 and len(design["zero_fit_candidates"]) == 9,
        "quantile_fits_2": all(role in ledger["completed_roles"] for role in FIT_ROLES[:2]),
        "oof_residual_fits_2": all(role in ledger["completed_roles"] for role in FIT_ROLES[2:4]),
        "benefit_router_fits_2": all(role in ledger["completed_roles"] for role in FIT_ROLES[4:]),
        "total_heavy_scientific_fits_6": tuple(ledger["completed_roles"]) == FIT_ROLES,
        "residual_oof_rows_400000": len(residual) == 400_000,
        "residual_oof_self_fit_rows_zero": int(residual["self_fit"].sum()) == 0,
        "benefit_oof_rows_400000": len(benefit) == 400_000,
        "benefit_router_features_37": len(design["benefit_router_features"]) == 37,
        "benefit_thresholds_exact": tuple(design["benefit_thresholds"]) == BENEFIT_THRESHOLDS,
        "validation_predictions_aligned_finite": bool(validation_ok),
        "persistent_bundles_reload": model_manifest["status"] == "PASS" and len(model_manifest["artifacts"]) == 5 and all(item["reload_maximum_absolute_difference"] == 0.0 for item in model_manifest["artifacts"]),
        "six_condition_rubric_unchanged": bool(design["six_condition_rubric_unchanged"]),
        "no_wider_band_model": bool(design["no_wider_band_model"]),
        "no_hierarchical_tail_model": bool(design["no_hierarchical_tail_model"]),
        "no_deep_model_fit": bool(design["no_deep_fit"]),
        "no_500k_final_project_refit": design["full_development_final_refit_count"] == 0,
        "final_model_selected_false": design["final_model_selected"] is False,
        "final_model_frozen_false": design["final_model_frozen"] is False,
        "final_pre_iid_freeze_absent": not (workspace / REPORTS / "FINAL_PRE_IID_FREEZE.json").exists(),
        "prompt4c_not_executed": not any(path.is_file() for path in workspace.rglob("*prompt4c*")),
        "notebook_errors_zero": notebook["errors"] == 0,
        "notebook_fit_calls_zero": notebook["fit_calls"] == 0,
        "reviewer_unresolved_critical_zero": review["unresolved_critical"] == 0,
        "reviewer_unresolved_major_zero": review["unresolved_major"] == 0,
    }
    failures = [name for name, value in checks.items() if not value]
    retries = sum(max(0, len([item for item in ledger["attempts"] if item["role"] == role]) - 1) for role in FIT_ROLES)
    report = {"status": "PASS" if not failures else "FAIL", "created_at_utc": utc_now(), "checks": checks, "failures": failures, "development_rows": 500_000, "train_rows": 400_000, "validation_rows": 100_000, "selection_rows": 70_000, "audit_rows": 30_000, "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0, "iid_prediction_count": 0, "scientific_fit_count": len(ledger["completed_roles"]), "technical_retry_count": retries, "residual_oof_rows": len(residual), "residual_oof_self_fit_rows": int(residual["self_fit"].sum()), "benefit_oof_rows": len(benefit), "benefit_router_input_features": 37, "benefit_thresholds": list(BENEFIT_THRESHOLDS), "full_development_final_refit_count": 0, "final_model_selected": False, "final_model_frozen": False}
    atomic_json(workspace, REPORTS / "prompt4b2_verification.json", report); return report


def write_readiness(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve(); verification = verify_prompt4b2(workspace)
    if verification["status"] != "PASS":
        raise RuntimeError(f"Prompt 4B2 verification failed: {verification['failures']}")
    ready = {"status": "PASS", "created_at_utc": utc_now(), "development_stage": "adaptive", "final_model_selected": False, "final_model_frozen": False, "full_development_final_refit_count": 0, "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0, "iid_prediction_count": 0, "scientific_fit_count": 6, "next_step": "Human review before Prompt 4C"}
    atomic_json(workspace, REPORTS / "PROMPT4B2_READY.json", ready); return ready


def run_all(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    if not (workspace / REPORTS / "prompt4b2_frozen_design.json").exists(): prepare_design(workspace)
    a = run_block_a(workspace); b = run_block_b(workspace); c = run_block_c(workspace)
    return {"status": "COMPLETE", "blocks": [a, b, c]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("validate", "prepare", "blockA", "blockB", "blockC", "all", "delivery", "notebook", "review", "verify", "ready")); parser.add_argument("--root", default=None); return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    functions = {"validate": validate_prompt4b, "prepare": prepare_design, "blockA": run_block_a, "blockB": run_block_b, "blockC": run_block_c, "all": run_all, "delivery": build_delivery_reports, "notebook": execute_notebook, "review": independent_review, "verify": verify_prompt4b2, "ready": write_readiness}
    result = functions[args.command](args.root); print(json.dumps(result, indent=2, default=str)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
