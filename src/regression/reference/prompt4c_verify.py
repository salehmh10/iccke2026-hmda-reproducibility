"""Independent Prompt 4C verification, freeze promotion, and readiness seal.

This module never reads Raw data or either IID Parquet file.  It verifies only
Development and already-saved Prompt artifacts.  It does not import or call the
Prompt 4C fitting orchestrator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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


DEVELOPMENT_SHA256 = "0ed232397be3ec4de1483c594954dce7b4704b375ca295397d899323dc4f0b6b"
AUTHORIZATION_ID = "regression_v2_prompt4c_final_selection_pre_iid_freeze"
PRIMARY_ID = "stage3_residual_t75_a75"
GLOBAL_ID = "ens_boost_cat060"
BLOCKA_ID = "nf_global2_oldraw_direct_cap25"
FEATURE_CONTRACT = "main_without_sensitive_without_lender"
REPORTS = Path("outputs/reports")
DEVELOPMENT = Path("outputs/data/development.parquet")
NOTEBOOK = Path("notebooks/04C_FINAL_SELECTION_AND_PRE_IID_FREEZE.ipynb")
OOF = Path("outputs/predictions/prompt4c/oof_global_500k.parquet")
PRIMARY_BUNDLE = Path("outputs/models/final_pre_iid/primary_stage3/bundle.joblib")
GLOBAL_BUNDLE = Path("outputs/models/final_pre_iid/global_comparator/bundle.joblib")
SAMPLE = Path("outputs/tmp/prompt4c/clean_reload_feature_sample.parquet")
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(root: Path, relative: Path) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def atomic_json(root: Path, relative: Path, payload: dict[str, Any]) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def require(condition: bool, name: str, evidence: Any, checks: dict[str, Any]) -> None:
    checks[name] = {"status": "PASS" if condition else "FAIL", "evidence": evidence}
    if not condition:
        raise RuntimeError(f"Independent verification failed: {name}: {evidence}")


def update_runtime(root: Path, **values: float | str) -> None:
    path = root / REPORTS / "prompt4c_runtime.json"
    runtime = json.loads(path.read_text(encoding="utf-8"))
    for key, value in values.items():
        runtime[key] = value
    phase_keys = [
        key for key, value in runtime.items()
        if key not in {"status", "created_at_utc", "updated_at_utc", "total_elapsed"}
        and isinstance(value, (int, float))
    ]
    runtime["total_elapsed"] = float(sum(float(runtime[key]) for key in phase_keys))
    runtime["updated_at_utc"] = utc_now()
    atomic_json(root, REPORTS / "prompt4c_runtime.json", runtime)


def verify(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    checks: dict[str, Any] = {}
    handoff = read_json(root, REPORTS / "prompt4c_handoff_validation.json")
    selection = read_json(root, REPORTS / "prompt4c_final_selection_freeze.json")
    global_reproduction = read_json(root, REPORTS / "prompt4c_global_recipe_reproduction.json")
    stage3_reproduction = read_json(root, REPORTS / "prompt4c_stage3_recipe_reproduction.json")
    blocka = read_json(root, REPORTS / "prompt4c_blocka_package_eligibility.json")
    plan = read_json(root, REPORTS / "prompt4c_refit_plan.json")
    ledger = read_json(root, REPORTS / "prompt4c_final_refit_ledger.json")
    oof_manifest = read_json(root, REPORTS / "prompt4c_oof_manifest.json")
    model_manifest = read_json(root, REPORTS / "prompt4c_final_model_manifest.json")
    reload_report = read_json(root, REPORTS / "prompt4c_bundle_reload.json")
    protocol = read_json(root, REPORTS / "prompt4c_prompt5_protocol.json")
    notebook_report = read_json(root, REPORTS / "prompt4c_notebook_execution.json")
    prediction_manifest = read_json(root, REPORTS / "prompt4c_prediction_manifest.json")
    candidate = read_json(root, REPORTS / "prompt4c_pre_iid_freeze_candidate.json")
    reviewer = read_json(root, REPORTS / "prompt4c_reviewer.json")

    ready4b4 = read_json(root, REPORTS / "PROMPT4B4_READY.json")
    require(
        handoff["status"] == "PASS"
        and ready4b4.get("status") in {"PASS", "READY", "PASS_PROMPT4B4", "PASS_EXPERIMENT_COMPLETE_PARTIAL"},
        "prompt4b4_readiness_still_valid",
        {"handoff": handoff["status"], "ready": ready4b4.get("status")},
        checks,
    )
    immutable_failures = []
    for relative, expected in handoff["prompt4b4_immutable_snapshot"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            immutable_failures.append(relative)
    require(
        not immutable_failures and len(handoff["prompt4b4_immutable_snapshot"]) == handoff["prompt4b4_immutable_file_count"],
        "prior_immutable_hashes_unchanged",
        {"checked": len(handoff["prompt4b4_immutable_snapshot"]), "failures": immutable_failures},
        checks,
    )

    development_path = root / DEVELOPMENT
    metadata = pq.ParquetFile(development_path).metadata
    require(sha256(development_path) == DEVELOPMENT_SHA256, "development_sha_unchanged", DEVELOPMENT_SHA256, checks)
    require(metadata.num_rows == 500_000, "development_rows_exact", metadata.num_rows, checks)
    features = selection["features"]
    prohibited = ("respondent", "lender", "race", "ethnic", "sex", "target_decile", "loan_amount_000s")
    require(
        selection["feature_count"] == 35 and len(features) == 35 and len(set(features)) == 35,
        "feature_count_and_uniqueness",
        {"declared": selection["feature_count"], "actual": len(features)},
        checks,
    )
    require(
        selection["feature_contract_name"] == FEATURE_CONTRACT and not any(token in name.lower() for name in features for token in prohibited),
        "feature_contract_safe",
        FEATURE_CONTRACT,
        checks,
    )
    require(
        selection["status"] == "FROZEN"
        and selection["final_primary_recipe"] == PRIMARY_ID
        and selection["simple_comparator"] == GLOBAL_ID
        and selection["historical_mae_challenger"] == BLOCKA_ID
        and selection["selection_frozen_before_prompt4c_percentage_metrics"]
        and selection["mape_wape_do_not_affect_selection"],
        "final_selection_freeze",
        {"sha256": sha256(root / REPORTS / "prompt4c_final_selection_freeze.json")},
        checks,
    )
    require(
        candidate["artifact_hashes"]["prompt4c_final_selection_freeze.json"]
        == sha256(root / REPORTS / "prompt4c_final_selection_freeze.json"),
        "final_selection_freeze_hash",
        candidate["artifact_hashes"]["prompt4c_final_selection_freeze.json"],
        checks,
    )

    global_diff = global_reproduction["maximum_absolute_prediction_difference"]
    stage3_diffs = stage3_reproduction["component_maximum_absolute_differences"]
    require(global_reproduction["status"] == "PASS" and global_diff == 0.0, "historical_global_reproduction", global_diff, checks)
    require(
        stage3_reproduction["status"] == "PASS" and all(value == 0.0 for value in stage3_diffs.values()),
        "historical_stage3_reproduction",
        stage3_diffs,
        checks,
    )
    require(
        blocka["status"] == "PASS" and blocka["new_fit_count"] == 0 and blocka["historical_blocka_iid_eligible"] is False,
        "historical_blocka_zero_fit_exclusion",
        {"eligible": blocka["historical_blocka_iid_eligible"], "reason": blocka["eligibility_reason"]},
        checks,
    )

    roles = tuple(plan["roles"][i]["role"] for i in range(len(plan["roles"])))
    attempts = ledger["attempts"]
    require(
        plan["status"] == "FROZEN" and plan["role_count"] == 11 and roles == FINAL_ROLES,
        "exact_final_refit_graph",
        list(roles),
        checks,
    )
    require(
        ledger["status"] == "COMPLETE"
        and tuple(ledger["completed_roles"]) == FINAL_ROLES
        and ledger["completed_role_count"] == 11
        and ledger["technical_retry_count"] <= 11
        and all(entry["status"] == "PASS" and int(entry["physical_attempt"]) <= 2 for entry in attempts)
        and ledger["scientific_candidate_searches"] == 0,
        "refit_ledger_and_retry_limits",
        {"attempts": len(attempts), "retries": ledger["technical_retry_count"], "scientific_searches": ledger["scientific_candidate_searches"]},
        checks,
    )

    oof_path = root / OOF
    oof = pd.read_parquet(oof_path, columns=["row_hash", "global_oof_prediction", "self_fit"])
    development_ids = pd.read_parquet(development_path, columns=["row_hash"])["row_hash"].astype(str)
    require(
        oof_manifest["status"] == "PASS"
        and sha256(oof_path) == oof_manifest["sha256"]
        and len(oof) == 500_000
        and oof["row_hash"].astype(str).equals(development_ids)
        and oof["row_hash"].nunique() == 500_000,
        "oof_500k_exact_alignment",
        {"rows": len(oof), "unique": int(oof["row_hash"].nunique()), "sha256": oof_manifest["sha256"]},
        checks,
    )
    require(
        int(oof["self_fit"].sum()) == 0 and np.isfinite(oof["global_oof_prediction"].to_numpy(float)).all(),
        "oof_zero_self_fit_and_finite",
        {"self_fit_rows": int(oof["self_fit"].sum()), "finite": True},
        checks,
    )

    artifact_roles = tuple(item["role"] for item in model_manifest["artifacts"])
    artifact_failures = []
    for item in model_manifest["artifacts"]:
        path = root / item["model_path"]
        if not path.is_file() or sha256(path) != item["model_sha256"]:
            artifact_failures.append(item["role"])
    require(
        model_manifest["status"] == "PASS" and artifact_roles == FINAL_ROLES and not artifact_failures,
        "final_component_models_and_hashes",
        {"roles": len(artifact_roles), "failures": artifact_failures},
        checks,
    )
    primary_hash = sha256(root / PRIMARY_BUNDLE)
    global_hash = sha256(root / GLOBAL_BUNDLE)
    require(
        primary_hash == model_manifest["final_primary_bundle"]["sha256"]
        and global_hash == model_manifest["final_global_bundle"]["sha256"],
        "final_composite_bundle_hashes",
        {"primary": primary_hash, "global": global_hash},
        checks,
    )
    sys.path.insert(0, str(root / "src"))
    sample = pd.read_parquet(root / SAMPLE)
    sample = sample.drop(columns=["row_hash"])
    primary_model = joblib.load(root / PRIMARY_BUNDLE)
    global_model = joblib.load(root / GLOBAL_BUNDLE)
    primary_prediction = np.asarray(primary_model.predict(sample), dtype=float)
    global_prediction = np.asarray(global_model.predict(sample), dtype=float)
    require(
        reload_report["status"] == "PASS"
        and all(item["status"] == "PASS" and item["maximum_absolute_difference"] == 0.0 for item in reload_report["bundles"])
        and primary_prediction.shape == (len(sample),)
        and global_prediction.shape == (len(sample),)
        and np.isfinite(primary_prediction).all()
        and np.isfinite(global_prediction).all(),
        "final_bundle_reload_and_prediction",
        {"rows": len(sample), "clean_reload_max_difference": [item["maximum_absolute_difference"] for item in reload_report["bundles"]]},
        checks,
    )
    guards = reload_report["input_guards"]
    require(
        guards["exact_feature_count"]
        and guards["missing_required_feature_error"]
        and guards["duplicate_feature_error"]
        and guards["target_required"] is False
        and guards["row_hash_consumed"] is False
        and guards["respondent_id_consumed"] is False
        and guards["sensitive_columns_consumed"] is False,
        "final_bundle_input_guards",
        guards,
        checks,
    )

    from prompt4c_bundles import duplicate_safe_target_deciles, mape_details, wape_percent

    metric_target = np.array([100.0, 200.0, 0.0, -5.0])
    metric_prediction = np.array([90.0, 220.0, 4.0, -5.0])
    mape = mape_details(metric_target, metric_prediction)
    wape = wape_percent(np.array([100.0, 200.0]), np.array([90.0, 220.0]))
    require(
        mape["mape_percent"] == 10.0 and mape["mape_invalid_nonpositive_rows"] == 2 and mape["mape_valid_rows"] == 2,
        "mape_implementation_no_epsilon",
        mape,
        checks,
    )
    require(wape == 10.0, "wape_implementation", wape, checks)
    metric_table = pd.read_csv(root / REPORTS / "prompt4c_historical_metric_extension.csv")
    decile_table = pd.read_csv(root / REPORTS / "prompt4c_historical_decile_metrics.csv")
    synthetic_deciles = duplicate_safe_target_deciles(np.arange(1.0, 101.0))
    require(
        len(metric_table) == 6
        and len(decile_table) == 60
        and decile_table.groupby("model_id")["decile_index"].nunique().eq(10).all()
        and len(np.unique(synthetic_deciles)) == 10
        and np.isfinite(decile_table[["mae", "mape_percent", "wape_percent"]].to_numpy(float)).all(),
        "decile_and_percentage_reporting",
        {"models": int(metric_table["model_id"].nunique()), "decile_rows": len(decile_table)},
        checks,
    )

    frozen_protocol = protocol["protocol"]
    cutpoints = protocol["development_frozen_target_cutpoints"]
    require(
        protocol["status"] == "FROZEN"
        and frozen_protocol["primary_iid_metric"] == "MAE"
        and [item["model_id"] for item in frozen_protocol["permitted_models"]] == ["final_primary_stage3_500k", "final_global_500k"]
        and frozen_protocol["bootstrap"]["resamples"] == 500
        and frozen_protocol["bootstrap"]["seed"] == 42
        and len(frozen_protocol["six_condition_rubric"]) == 6,
        "prompt5_protocol_complete",
        {"models": frozen_protocol["permitted_models"], "bootstrap": frozen_protocol["bootstrap"]},
        checks,
    )
    require(
        list(cutpoints) == [f"q{index:02d}" for index in range(10, 100, 10)]
        and np.all(np.diff(np.array(list(cutpoints.values()), dtype=float)) >= 0),
        "development_target_band_cutpoints_frozen",
        cutpoints,
        checks,
    )

    notebook = nbformat.read(root / NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    error_outputs = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
    image_outputs = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "display_data" and "image/png" in output.get("data", {})]
    table_outputs = [output for cell in code_cells for output in cell.get("outputs", []) if "text/html" in output.get("data", {})]
    code_text = "\n".join(cell.source.lower() for cell in code_cells)
    require(
        notebook_report["status"] == "PASS"
        and not error_outputs
        and len(image_outputs) >= 6
        and len(table_outputs) >= 1
        and ".fit(" not in code_text
        and "iid_features.parquet" not in code_text
        and "iid_target.parquet" not in code_text,
        "artifact_only_notebook_inline",
        {"errors": len(error_outputs), "images": len(image_outputs), "tables": len(table_outputs)},
        checks,
    )
    figures = sorted((root / "outputs/figures/prompt4c").glob("*.png"))
    require(len(figures) == 6 and all(path.stat().st_size > 0 for path in figures), "six_quantitative_figures", [path.name for path in figures], checks)

    safety_sources = [handoff, selection, protocol, prediction_manifest, candidate]
    safety_ok = all(
        payload.get("raw_access_count", payload.get("safety", {}).get("raw_access_count", 0)) == 0
        and payload.get("iid_feature_access_count", payload.get("safety", {}).get("iid_feature_access_count", 0)) == 0
        and payload.get("iid_target_access_count", payload.get("safety", {}).get("iid_target_access_count", 0)) == 0
        and payload.get("iid_prediction_count", payload.get("safety", {}).get("iid_prediction_count", 0)) == 0
        and payload.get("prompt5_executed", payload.get("safety", {}).get("prompt5_executed", False)) is False
        for payload in safety_sources
    )
    require(safety_ok, "raw_iid_access_remains_zero", candidate["safety"], checks)
    component_directories = sorted(path.name for path in (root / "outputs/models/final_pre_iid/components").iterdir() if path.is_dir())
    prompt4c_prediction_files = sorted(path.name for path in (root / "outputs/predictions/prompt4c").iterdir() if path.is_file())
    require(
        component_directories == sorted(FINAL_ROLES)
        and prompt4c_prediction_files == ["oof_global_500k.parquet"]
        and model_manifest["scientific_candidate_searches"] == 0
        and notebook_report["fit_calls"] == 0,
        "no_post_selection_experiment",
        {"component_roles": len(component_directories), "prediction_files": prompt4c_prediction_files},
        checks,
    )
    require(reviewer["status"] == "PASS", "independent_reviewer_pass", reviewer.get("summary", reviewer.get("verdict")), checks)
    require(candidate["status"] == "READY_FOR_INDEPENDENT_REVIEW", "freeze_candidate_valid", sha256(root / REPORTS / "prompt4c_pre_iid_freeze_candidate.json"), checks)

    elapsed = time.perf_counter() - started
    reviewer_seconds = float(reviewer.get("elapsed_seconds", 0.0))
    update_runtime(root, independent_review=reviewer_seconds, final_verification=elapsed)
    result = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "independent_of_fit_orchestration": True,
        "check_count": len(checks),
        "checks": checks,
        "elapsed_seconds": elapsed,
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "iid_prediction_count": 0,
        "prompt5_executed": False,
    }
    atomic_json(root, REPORTS / "prompt4c_verification.json", result)
    return result


def validate_promoted_freeze(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    primary_hash = sha256(root / PRIMARY_BUNDLE)
    global_hash = sha256(root / GLOBAL_BUNDLE)
    protocol = payload["iid_protocol"]
    safety = payload["safety"]
    validations = {
        "json_readable": True,
        "status": payload["status"] == "PASS_FINAL_PRE_IID_FREEZE",
        "primary_id": payload["final_selection"]["final_primary_id"] == "final_primary_stage3_500k"
        and payload["final_selection"]["historical_recipe_id"] == PRIMARY_ID,
        "primary_hash": payload["primary_bundle"]["sha256"] == primary_hash,
        "global_id": payload["final_selection"]["simple_comparator"] == "final_global_500k",
        "global_hash": payload["global_comparator"]["sha256"] == global_hash,
        "blocka_eligibility_boolean": isinstance(payload["historical_blocka"]["iid_eligible"], bool),
        "mape_definition": "no epsilon" in protocol["mape_definition"],
        "wape_definition": protocol["wape_definition"] == "100*sum(abs(y-pred))/sum(y)",
        "decile_definitions": "duplicate-safe" in protocol["iid_local_deciles"]
        and len(protocol["development_frozen_target_cutpoints"]) == 9,
        "bootstrap": protocol["bootstrap"]["resamples"] == 500 and protocol["bootstrap"]["seed"] == 42,
        "iid_counts_zero": all(safety[key] == 0 for key in ("raw_access_count", "iid_feature_access_count", "iid_target_access_count", "iid_prediction_count")),
        "prompt5_not_executed": safety["prompt5_executed"] is False,
    }
    if not all(validations.values()):
        raise RuntimeError(f"Promoted freeze validation failed: {validations}")
    return validations


def promote(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    reviewer_path = root / REPORTS / "prompt4c_reviewer.json"
    verification_path = root / REPORTS / "prompt4c_verification.json"
    candidate_path = root / REPORTS / "prompt4c_pre_iid_freeze_candidate.json"
    reviewer = json.loads(reviewer_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if reviewer.get("status") != "PASS" or verification.get("status") != "PASS":
        raise RuntimeError("Reviewer and verification must both pass before promotion.")
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["status"] = "PASS_FINAL_PRE_IID_FREEZE"
    payload["promotion"] = {
        "promoted_at_utc": utc_now(),
        "candidate_sha256": sha256(candidate_path),
        "reviewer_sha256": sha256(reviewer_path),
        "verification_sha256": sha256(verification_path),
        "reviewer_status": reviewer["status"],
        "verification_status": verification["status"],
    }
    destination = atomic_json(root, REPORTS / "FINAL_PRE_IID_FREEZE.json", payload)
    reloaded = json.loads(destination.read_text(encoding="utf-8"))
    validations = validate_promoted_freeze(root, reloaded)
    elapsed = time.perf_counter() - started
    update_runtime(root, freeze_promotion=elapsed, status="PASS_FINAL_PRE_IID_FREEZE")
    return {"status": "PASS", "path": str(destination.relative_to(root)), "sha256": sha256(destination), "validations": validations, "elapsed_seconds": elapsed}


def validate_existing_freeze(root: Path) -> dict[str, Any]:
    """Reload and validate the already-promoted freeze without changing its bytes."""
    started = time.perf_counter()
    path = root / REPORTS / "FINAL_PRE_IID_FREEZE.json"
    before = sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validations = validate_promoted_freeze(root, payload)
    after = sha256(path)
    if before != after:
        raise RuntimeError("The promoted freeze changed during read-only validation.")
    elapsed = time.perf_counter() - started
    update_runtime(root, freeze_promotion=elapsed, status="PASS_FINAL_PRE_IID_FREEZE")
    return {
        "status": "PASS",
        "path": str(path.relative_to(root)),
        "sha256_before": before,
        "sha256_after": after,
        "freeze_unchanged": before == after,
        "validations": validations,
        "elapsed_seconds": elapsed,
    }


def ready(root: Path) -> dict[str, Any]:
    destination = root / REPORTS / "PROMPT4C_READY.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("status") != "PASS_FINAL_PRE_IID_FREEZE":
            raise RuntimeError("Existing Prompt 4C readiness file is invalid.")
        return existing
    freeze_path = root / REPORTS / "FINAL_PRE_IID_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validate_promoted_freeze(root, freeze)
    reviewer = read_json(root, REPORTS / "prompt4c_reviewer.json")
    verification = read_json(root, REPORTS / "prompt4c_verification.json")
    runtime = read_json(root, REPORTS / "prompt4c_runtime.json")
    if reviewer["status"] != "PASS" or verification["status"] != "PASS" or runtime["status"] != "PASS_FINAL_PRE_IID_FREEZE":
        raise RuntimeError("Final review, verification, or runtime status is not ready.")
    payload = {
        "status": "PASS_FINAL_PRE_IID_FREEZE",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "prompt4c_complete": True,
        "final_selection_freeze_sha256": sha256(root / REPORTS / "prompt4c_final_selection_freeze.json"),
        "final_primary_id": PRIMARY_ID,
        "final_primary_bundle_sha256": sha256(root / PRIMARY_BUNDLE),
        "final_global_comparator_id": GLOBAL_ID,
        "final_global_comparator_bundle_sha256": sha256(root / GLOBAL_BUNDLE),
        "historical_blocka_iid_eligible": freeze["historical_blocka"]["iid_eligible"],
        "reviewer_status": reviewer["status"],
        "verification_status": verification["status"],
        "final_pre_iid_freeze_path": "outputs/reports/FINAL_PRE_IID_FREEZE.json",
        "final_pre_iid_freeze_sha256": sha256(freeze_path),
        "raw_access_count": 0,
        "iid_feature_access_count": 0,
        "iid_target_access_count": 0,
        "iid_prediction_count": 0,
        "prompt5_executed": False,
        "next_step": "Prompt 5 — One-Time IID Evaluation",
        "report_creation_rule": "This is the last Prompt 4C report artifact.",
    }
    atomic_json(root, REPORTS / "PROMPT4C_READY.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "promote", "validate-freeze", "ready"))
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "verify":
        result = verify(root)
    elif args.command == "promote":
        result = promote(root)
    elif args.command == "validate-freeze":
        result = validate_existing_freeze(root)
    else:
        result = ready(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
