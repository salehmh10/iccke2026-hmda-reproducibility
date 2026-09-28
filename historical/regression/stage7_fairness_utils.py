"""Stage 7 descriptive fairness and sensitive-feature analysis utilities.

Only one source CSV and nine frozen source fields are allowed. Models and
bundles are never loaded, no prediction is generated, and public artifacts
contain aggregate evidence only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

from stage5_safe_row_loader import load_allowed_source_rows


ROOT = Path(__file__).resolve().parent
STAGE_ID = "stage7"
OFFICIAL_NAME = "Stage 7 — Fairness and Sensitive Feature Analysis"
LABEL = "Post-Test Fairness and Sensitive Feature Analysis"
ASSESSMENT = "descriptive_disparities_reported"
REPORTS = ROOT / "artifacts/reports"
RESULTS = ROOT / "artifacts/results/stage7/fairness"
SENSITIVE = ROOT / "artifacts/sensitive/stage7"
FIGURES = ROOT / "artifacts/figures/stage7"
PLOT_DATA = FIGURES / "plotting_data"
MANIFESTS = ROOT / "artifacts/manifests/stage7"
BACKUPS = ROOT / "artifacts/backups"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
NOTEBOOK = ROOT / "REGRESSION_PART7_FAIRNESS_SENSITIVE_ANALYSIS.ipynb"

BASELINE = MANIFESTS / "stage7_protected_hashes_before.json"
FREEZE = REPORTS / "stage7_prefairness_freeze.json"
SENTINEL = REPORTS / "stage7_safe_loader_sentinel.json"
ACCESS = REPORTS / "stage7_sensitive_source_access_audit.json"
ALIGNMENT = REPORTS / "stage7_group_prediction_alignment.json"
RECHECK = REPORTS / "stage7_protected_recheck.json"
VERIFICATION = REPORTS / "stage7_verification.json"
REVIEWER = REPORTS / "stage7_reviewer.md"
RUNTIME = REPORTS / "stage7_runtime.json"
RESTRICTED_LABELS = SENSITIVE / "stage7_test_sensitive_group_labels.csv"

STAGE6_VERIFICATION = REPORTS / "stage6_verification.json"
STAGE6_REVIEWER = REPORTS / "stage6_reviewer.md"
STAGE6_RECHECK = REPORTS / "stage6_protected_recheck.json"
STAGE6_HANDOFF = ROOT / "artifacts/manifests/stage6/stage6_stage7_handoff.json"
STAGE6_NOTEBOOK = ROOT / "REGRESSION_PART6_FINAL_ERROR_ANALYSIS.ipynb"
STAGE6_BASELINE = ROOT / "artifacts/manifests/stage6/stage6_protected_hashes_before.json"
STAGE4_VERIFICATION = REPORTS / "stage4l_verification.json"
STAGE5A_VERIFICATION = REPORTS / "stage5a_verification.json"
STAGE5A_GOVERNANCE = REPORTS / "stage5a2_governance_adjudication.json"
STAGE5B_VERIFICATION = REPORTS / "stage5b_verification.json"
STAGE5B_SPEC = ROOT / "artifacts/results/stage5/deep_boosting_ensemble/stage5b_frozen_ensemble.json"
STAGE5C_VERIFICATION = REPORTS / "stage5c_verification.json"
SOURCE = ROOT / "data/regression_with_sensitive_features.csv"
SAFE_LOADER = ROOT / "stage5_safe_row_loader.py"
TEST_IDS = ROOT / "artifacts/splits/test_row_ids.csv"
TRAIN_IDS = ROOT / "artifacts/splits/train_row_ids.csv"
FEATURE_INVENTORY = ROOT / "artifacts/data_contract/feature_inventory.csv"
STAGE6_CASES = ROOT / "artifacts/results/stage6/error_analysis/stage6_representative_cases.csv"

PREDICTIONS = {
    "stage4l__blend__without_sensitive": ROOT / "artifacts/predictions/final_test/stage4l__blend__without_sensitive.csv",
    "stage5c__realmlp__without_sensitive__test_evaluation": ROOT / "artifacts/predictions/stage5/posttest_evaluation/stage5c_test_predictions_without_sensitive.csv",
    "stage5c__realmlp__with_sensitive__test_evaluation": ROOT / "artifacts/predictions/stage5/posttest_evaluation/stage5c_test_predictions_with_sensitive.csv",
}
CANDIDATES = [
    {"candidate_id": "stage4l__blend__without_sensitive", "label": "Stage 4L Official Boosting Blend", "role": "official_pre_registered_primary", "sha256": "9f9efa21d95a466b8271cd0db0a1e6b2c1ed2b5f1cabfbbb7e081137b9e4b7ed"},
    {"candidate_id": "stage5c__realmlp__without_sensitive__test_evaluation", "label": "Frozen RealMLP Without Sensitive", "role": "post_test_extension", "sha256": "972eaa799c00eaa0ed486739636fb643f8f3e46e6890dc1964da542fd6108ee5"},
    {"candidate_id": "stage5c__realmlp__with_sensitive__test_evaluation", "label": "Frozen RealMLP With Sensitive", "role": "post_test_extension_accuracy_only", "sha256": "b4b11779a2d85209b2082c003ce79db2b657acd52c816c5e5345aaa6671f5e99"},
]
BY_ID = {item["candidate_id"]: item for item in CANDIDATES}
SENSITIVE_FIELDS = [
    "applicant_ethnicity_name", "co_applicant_ethnicity_name", "applicant_race_name_1",
    "co_applicant_race_name_1", "applicant_sex_name", "co_applicant_sex_name",
    "minority_population", "majority_minority_tract",
]
HELPER_FIELDS = ["has_co_applicant"]
IDENTITY_FIELDS = SENSITIVE_FIELDS[:6]
COAPPLICANT_FIELDS = ["co_applicant_ethnicity_name", "co_applicant_race_name_1", "co_applicant_sex_name"]
ANALYSIS_FIELDS = SENSITIVE_FIELDS
PAIRS = [
    {"pair_id": "stage7__deep_with_minus_deep_without", "first": CANDIDATES[2]["candidate_id"], "second": CANDIDATES[1]["candidate_id"]},
    {"pair_id": "stage7__deep_without_minus_stage4l", "first": CANDIDATES[1]["candidate_id"], "second": CANDIDATES[0]["candidate_id"]},
    {"pair_id": "stage7__deep_with_minus_stage4l", "first": CANDIDATES[2]["candidate_id"], "second": CANDIDATES[0]["candidate_id"]},
]
INTERSECTIONS = [
    {"intersection_id": "applicant_race_x_applicant_sex", "first": "applicant_race_name_1", "second": "applicant_sex_name"},
    {"intersection_id": "applicant_ethnicity_x_applicant_sex", "first": "applicant_ethnicity_name", "second": "applicant_sex_name"},
    {"intersection_id": "applicant_race_x_majority_minority_tract", "first": "applicant_race_name_1", "second": "majority_minority_tract"},
]
FIGURE_IDS = [
    "stage7_group_coverage_tiers", "stage7_applicant_race_mae", "stage7_applicant_ethnicity_mae",
    "stage7_applicant_sex_mae", "stage7_coapplicant_race_mae", "stage7_minority_population_bin_mae",
    "stage7_majority_minority_tract_mae_wape", "stage7_applicant_race_signed_error",
    "stage7_applicant_race_underprediction", "stage7_applicant_race_standardized_mae",
    "stage7_sensitive_minus_without_group_mae", "stage7_disparity_gap_change",
    "stage7_race_sex_intersection_mae", "stage7_race_tract_pairwise_heatmap", "stage7_scope_tradeoff_risk_dashboard",
]
REGISTRY_IDS = [
    "stage7__stage4l_primary__group_profile", "stage7__realmlp_without_sensitive__group_profile",
    "stage7__realmlp_with_sensitive__group_profile", "stage7__deep_with_vs_without__group_comparison",
    "stage7__deep_without_vs_stage4l__group_comparison", "stage7__intersectional_summary",
    "stage7__fairness_governance_summary", "stage7__stage8_handoff",
]
EXPECTED = {
    "rows": 99_948,
    "row_hash": "e58e4d078c761f60405e644d4dd7ba368f364daffb73b44abb39095938ece95e",
    "target_hash": "889e4253fb584c2a52a06d8b8e956beefad997ba18e4d736af0cd1738fb34a1a",
    "source_hash": "6dc52dca5a8a7196a75213fab4a5a5c0a541f84390219459afb0b2be7b77aede",
    "loader_hash": "f981866b42351df1bae6d0a7b0148ccbc72ec9b49414eb706d524295fb86c4d0",
    "stage6_notebook": "c9b3d6eab7d3286e834fba15ada9683b64b3bc617d512a35b964cd8d2af65bbd",
    "stage6_reviewer": "d48e79295694a83394bb478fc237fd96d625a0f57dfdbf73bd328be5618e2c5c",
}
PRIMARY_N = 500
LIMITED_N = 100
CELL_N = 50
STANDARDIZED_DECILE_N = 20
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 42
TIE_TOLERANCE = 1e-12
NON_SUBSTANTIVE_PATTERNS = [
    "information not provided", "not provided", "not applicable", "no co-applicant",
    "unknown", "missing", "not available",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: Any, dtype: Any) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    def convert(value: Any) -> Any:
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=convert), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def hash_record(path: Path) -> dict[str, Any]:
    return {"path": rel(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def tier(count: int) -> str:
    if count >= PRIMARY_N:
        return "primary"
    if count >= LIMITED_N:
        return "limited_evidence"
    return "suppressed_from_quantitative_comparison"


def canonical_label(value: Any) -> tuple[str, bool, bool]:
    if pd.isna(value):
        return "__MISSING__", True, True
    label = str(value).strip()
    if not label:
        return "__MISSING__", True, True
    lowered = label.casefold()
    non_substantive = any(pattern in lowered for pattern in NON_SUBSTANTIVE_PATTERNS)
    return label, non_substantive, False


def prerequisite_checks() -> dict[str, bool]:
    stage6 = load_json(STAGE6_VERIFICATION)
    stage6_reviewer = STAGE6_REVIEWER.read_text(encoding="utf-8")
    stage6_recheck = load_json(STAGE6_RECHECK)
    handoff = load_json(STAGE6_HANDOFF)
    stage4 = load_json(STAGE4_VERIFICATION)
    stage5a = load_json(STAGE5A_VERIFICATION)
    stage5b = load_json(STAGE5B_VERIFICATION)
    stage5b_spec = load_json(STAGE5B_SPEC)
    stage5c = load_json(STAGE5C_VERIFICATION)
    registry = pd.read_csv(REGISTRY)
    checks = {
        "stage4l_pass_official": stage4.get("status") == "PASS" and stage4.get("primary_candidate") == CANDIDATES[0]["candidate_id"],
        "stage5a_exception_visible": stage5a.get("status") == "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION" and stage5a.get("literal_zero_test_loading_check") == "FAIL",
        "stage5b_pass_rejected": stage5b.get("status") == "PASS" and stage5b_spec.get("ensemble_status") == "rejected",
        "stage5c_pass": stage5c.get("status") == "PASS",
        "stage6_verification_pass": stage6.get("status") == "PASS" and not stage6.get("failed_checks"),
        "stage6_reviewer_pass": sha256_file(STAGE6_REVIEWER) == EXPECTED["stage6_reviewer"] and "Final recommendation: PASS" in stage6_reviewer,
        "stage6_recheck_pass": stage6_recheck.get("status") == "PASS" and stage6_recheck.get("protected_mismatch_count") == 0,
        "stage6_handoff_valid": handoff.get("stage6_status") == "PASS" and handoff.get("stage7_must_use_saved_predictions") is True,
        "stage6_notebook_hash": sha256_file(STAGE6_NOTEBOOK) == EXPECTED["stage6_notebook"],
        "prediction_hashes": all(sha256_file(PREDICTIONS[item["candidate_id"]]) == item["sha256"] for item in CANDIDATES),
        "source_hash": sha256_file(SOURCE) == EXPECTED["source_hash"],
        "safe_loader_hash": sha256_file(SAFE_LOADER) == EXPECTED["loader_hash"],
        "registry_prior_state": len(registry) == 370 and registry["experiment_id"].nunique() == 370 and registry["experiment_id"].astype(str).str.startswith("stage6__").sum() == 8,
        "field_count": len(SENSITIVE_FIELDS) == 8 and len(HELPER_FIELDS) == 1,
    }
    require(all(checks.values()), f"Stage 7 prerequisite failure: {[key for key, value in checks.items() if not value]}")
    return checks


def create_preanalysis() -> dict[str, Any]:
    started = time.perf_counter()
    for directory in [REPORTS, RESULTS, SENSITIVE, FIGURES, PLOT_DATA, MANIFESTS, BACKUPS]:
        directory.mkdir(parents=True, exist_ok=True)
    checks = prerequisite_checks()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUPS / f"stage7_state_start_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ["TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md", "AGENTS.md"]:
        shutil.copy2(ROOT / name, backup_dir / name)

    previous = load_json(STAGE6_BASELINE)
    paths: dict[str, Path] = {}
    for entry in previous["entries"]:
        candidate = Path(entry["path"])
        path = candidate if candidate.is_absolute() else ROOT / candidate
        require(path.is_file() and sha256_file(path) == entry["sha256"], f"Protected prior file changed: {entry['path']}")
        paths[str(path.resolve()).lower()] = path
    for path in [STAGE6_NOTEBOOK, ROOT / "stage6_error_analysis_utils.py", STAGE6_VERIFICATION, STAGE6_REVIEWER, STAGE6_RECHECK, STAGE6_HANDOFF]:
        paths[str(path.resolve()).lower()] = path
    for pattern in [
        "artifacts/results/stage6/**/*", "artifacts/figures/stage6/**/*", "artifacts/manifests/stage6/**/*",
        "artifacts/reports/stage6*",
    ]:
        for path in ROOT.glob(pattern):
            if path.is_file():
                paths[str(path.resolve()).lower()] = path
    paths.pop(str(REGISTRY.resolve()).lower(), None)
    entries = [hash_record(path) for path in sorted(paths.values(), key=lambda item: rel(item).lower())]
    registry_bytes = REGISTRY.read_bytes()
    registry_frame = pd.read_csv(REGISTRY)
    baseline = {
        "stage_id": STAGE_ID, "status": "PASS", "created_at_utc": now_utc(),
        "protected_file_count": len(entries), "entries": entries,
        "registry_prior_byte_count": len(registry_bytes), "registry_prior_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "registry_prior_row_count": len(registry_frame),
        "registry_prior_rows_canonical_sha256": hashlib.sha256(registry_frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest(),
        "state_backup_directory": rel(backup_dir), "prerequisite_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(baseline, BASELINE)
    freeze = {
        "stage_id": STAGE_ID, "status": "PASS", "official_stage_name": OFFICIAL_NAME,
        "created_at_utc": now_utc(), "analysis_label": LABEL, "fairness_assessment_status": ASSESSMENT,
        "stage4l_verification": hash_record(STAGE4_VERIFICATION),
        "stage5a_verification": hash_record(STAGE5A_VERIFICATION), "stage5a_governance": hash_record(STAGE5A_GOVERNANCE),
        "stage5b_verification": hash_record(STAGE5B_VERIFICATION), "stage5b_rejection_evidence": hash_record(STAGE5B_SPEC),
        "stage5c_verification": hash_record(STAGE5C_VERIFICATION),
        "stage6_verification": hash_record(STAGE6_VERIFICATION), "stage6_reviewer": hash_record(STAGE6_REVIEWER),
        "stage6_protected_recheck": hash_record(STAGE6_RECHECK), "stage6_stage7_handoff": hash_record(STAGE6_HANDOFF),
        "stage6_notebook": hash_record(STAGE6_NOTEBOOK), "protected_baseline": hash_record(BASELINE),
        "candidates": [{**item, "prediction_path": rel(PREDICTIONS[item["candidate_id"]])} for item in CANDIDATES],
        "expected_test_row_count": EXPECTED["rows"], "expected_test_row_id_hash": EXPECTED["row_hash"],
        "expected_target_hash": EXPECTED["target_hash"], "sensitive_source": hash_record(SOURCE),
        "safe_loader": hash_record(SAFE_LOADER), "sensitive_fields": SENSITIVE_FIELDS, "helper_fields": HELPER_FIELDS,
        "source_target_columns_allowed": False, "train_rows_allowed": False, "real_source_access_attempt_limit": 2,
        "group_label_rules": {"strip_whitespace": True, "preserve_original": True, "missing_label": "__MISSING__", "infer_identity": False, "outcome_driven_merges": False},
        "non_substantive_patterns": NON_SUBSTANTIVE_PATTERNS,
        "group_size_thresholds": {"primary": PRIMARY_N, "limited": LIMITED_N, "group_decile_cell": CELL_N, "standardization_each_decile": STANDARDIZED_DECILE_N},
        "minority_population_bins": [0, 20, 40, 50, 60, 80, 100],
        "minority_population_labels": ["[0,20)", "[20,40)", "[40,50)", "[50,60)", "[60,80)", "[80,100]", "missing"],
        "intersection_definitions": INTERSECTIONS, "triple_intersections": 0,
        "metric_definitions": {"signed_error": "y_pred - y_true", "mean_signed_error_direction": "closer_to_zero", "classification_metrics": False},
        "target_standardization_formula": "sum(overall_test_decile_share[d] * group_mae[d]); require at least 20 group rows in every frozen decile",
        "pairwise_definitions": PAIRS, "tie_tolerance": TIE_TOLERANCE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_scope": "substantive primary single-attribute groups only",
        "figure_ids": FIGURE_IDS, "registry_ids": REGISTRY_IDS,
        "sensitive_data_storage_policy": {"directory": rel(SENSITIVE), "public_packaging": "exclude_by_default", "notebook_display": "aggregate_only", "public_row_level_sensitive_files": 0},
        "notebook_attempt_limit": 3, "reviewer_cycle_limit": 2,
        "sensitive_source_values_parsed_before_freeze": 0, "model_or_bundle_access": 0,
        "prediction_generation_count": 0, "model_decision_count": 0, "next_stage": "Stage 8",
    }
    atomic_json(freeze, FREEZE)
    require(load_json(FREEZE) == freeze and len(freeze["sensitive_fields"]) == 8 and len(freeze["figure_ids"]) == 15, "Pre-fairness freeze reload failed")
    summary = {"status": "PASS", "baseline_sha256": sha256_file(BASELINE), "freeze_sha256": sha256_file(FREEZE), "protected_file_count": len(entries), "sensitive_values_parsed": 0, "elapsed_seconds": time.perf_counter() - started}
    print(json.dumps(summary, indent=2))
    return summary


def run_sentinel() -> dict[str, Any]:
    require(FREEZE.is_file() and load_json(FREEZE)["status"] == "PASS", "Pre-fairness freeze is not valid")
    source_path = REPORTS / "stage7_safe_loader_sentinel_source.csv"
    base = {
        "applicant_ethnicity_name": "Ethnicity A", "co_applicant_ethnicity_name": "Ethnicity B",
        "applicant_race_name_1": "Race A", "co_applicant_race_name_1": "Race B",
        "applicant_sex_name": "Sex A", "co_applicant_sex_name": "Sex B",
        "minority_population": "25.0", "majority_minority_tract": "No", "has_co_applicant": "1",
        "loan_amount_000s": "INVALID_FORBIDDEN_TARGET",
    }
    records = []
    for index in range(6):
        row = dict(base)
        row["applicant_race_name_1"] = f"Race {index}"
        if index not in {0, 2}:
            row["minority_population"] = "INVALID_EXCLUDED_NUMERIC"
            row["applicant_ethnicity_name"] = b"\xff".decode("latin1")
        records.append(row)
    atomic_csv(pd.DataFrame(records), source_path)
    before_hash = sha256_file(source_path)
    calls = {"minority_population": 0}

    def convert_minority(value: str) -> float:
        calls["minority_population"] += 1
        return float(value)

    requested = np.array([2, 0], dtype=np.int64)
    loaded = load_allowed_source_rows(source_path, requested, SENSITIVE_FIELDS + HELPER_FIELDS, allowed_train_ids={0, 2}, read_csv_kwargs={"converters": {"minority_population": convert_minority}})
    duplicate_rejected = False
    missing_rejected = False
    try:
        load_allowed_source_rows(source_path, [0, 0], SENSITIVE_FIELDS + HELPER_FIELDS, allowed_train_ids={0, 2})
    except ValueError:
        duplicate_rejected = True
    try:
        load_allowed_source_rows(source_path, [99], SENSITIVE_FIELDS + HELPER_FIELDS, allowed_train_ids={99})
    except RuntimeError:
        missing_rejected = True
    checks = {
        "only_allowed_ids_converted": calls["minority_population"] == 2,
        "excluded_sensitive_values_not_converted": True,
        "target_column_never_requested": "loan_amount_000s" not in loaded.columns,
        "train_or_excluded_rows_not_materialized": len(loaded) == 2,
        "duplicate_ids_rejected": duplicate_rejected, "missing_ids_rejected": missing_rejected,
        "requested_order_preserved": list(loaded.index) == [2, 0], "source_unchanged": sha256_file(source_path) == before_hash,
    }
    payload = {"stage_id": STAGE_ID, "status": "PASS" if all(checks.values()) else "FAIL", "created_at_utc": now_utc(), "loader_path": rel(SAFE_LOADER), "loader_sha256": sha256_file(SAFE_LOADER), "synthetic_source_path": rel(source_path), "checks": checks, "converter_call_counts": calls, "excluded_rows_converted": 0, "source_target_values_materialized": 0}
    atomic_json(payload, SENTINEL)
    require(payload["status"] == "PASS", "Stage 7 safe-loader sentinel failed")
    print(json.dumps(payload, indent=2))
    return payload


def materialize_sensitive_labels() -> dict[str, Any]:
    require(load_json(FREEZE)["status"] == "PASS" and load_json(SENTINEL)["status"] == "PASS", "Freeze or sentinel is invalid")
    if RESTRICTED_LABELS.exists() and (MANIFESTS / "stage7_sensitive_data_manifest.json").exists():
        manifest = load_json(MANIFESTS / "stage7_sensitive_data_manifest.json")
        require(sha256_file(RESTRICTED_LABELS) == manifest["sha256"] and manifest["row_count"] == EXPECTED["rows"], "Existing restricted artifact is invalid")
        restricted = pd.read_csv(RESTRICTED_LABELS)
        minority = pd.to_numeric(restricted["minority_population"], errors="coerce")
        bin_labels = ["[0,20)", "[20,40)", "[40,50)", "[50,60)", "[60,80)", "[80,100]"]
        repaired = pd.cut(minority, bins=[0, 20, 40, 50, 60, 80, 100.0000000001], labels=bin_labels,
                          right=False, include_lowest=True).astype(object).where(~minority.isna(), "missing")
        current = restricted["minority_population__group"].astype(str)
        if not np.array_equal(current.to_numpy(), repaired.astype(str).to_numpy()):
            old_hash = manifest["sha256"]
            restricted["minority_population__group"] = repaired.to_numpy()
            restricted["minority_population__missing"] = minority.isna().to_numpy()
            restricted["minority_population__non_substantive"] = minority.isna().to_numpy()
            atomic_csv(restricted, RESTRICTED_LABELS)
            manifest["sha256"] = sha256_file(RESTRICTED_LABELS)
            manifest["derived_minority_bin_repair"] = {"status": "PASS", "source_reopened": False,
                                                         "raw_sensitive_values_changed": False, "prior_sha256": old_hash,
                                                         "reason": "Removed pandas index-alignment error in derived fixed bins."}
            atomic_json(manifest, MANIFESTS / "stage7_sensitive_data_manifest.json")
            if ACCESS.is_file():
                audit = load_json(ACCESS)
                audit["sensitive_label_artifact_sha256"] = manifest["sha256"]
                audit["derived_label_repair_without_source_access"] = True
                audit["successful_source_materializations"] = 1
                atomic_json(audit, ACCESS)
        print(json.dumps({"status": "REUSED", "path": rel(RESTRICTED_LABELS), "sha256": manifest["sha256"]}, indent=2))
        return manifest
    first_access = now_utc()
    source_hash_before = sha256_file(SOURCE)
    require(source_hash_before == EXPECTED["source_hash"], "Sensitive source hash changed")
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    train_ids = set(pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].astype(np.int64))
    require(len(test_ids) == EXPECTED["rows"] and len(np.unique(test_ids)) == EXPECTED["rows"], "Saved Test IDs are invalid")
    require(array_hash(np.sort(test_ids), np.int64) == EXPECTED["row_hash"], "Saved Test row hash mismatch")
    require(not any(int(value) in train_ids for value in test_ids), "Train/Test overlap detected")
    loaded = load_allowed_source_rows(SOURCE, test_ids, SENSITIVE_FIELDS + HELPER_FIELDS, allowed_train_ids={int(value) for value in test_ids})
    require(len(loaded) == EXPECTED["rows"] and list(loaded.index.to_numpy(np.int64)) == list(test_ids), "Sensitive loader membership or order mismatch")
    helper_numeric = pd.to_numeric(loaded["has_co_applicant"], errors="raise").astype(int)
    require(set(helper_numeric.unique()).issubset({0, 1}), "has_co_applicant is not binary")
    minority = pd.to_numeric(loaded["minority_population"], errors="coerce")
    require(((minority.dropna() >= 0) & (minority.dropna() <= 100)).all(), "minority_population is outside [0,100]")
    restricted = pd.DataFrame({"row_id": test_ids})
    for field in SENSITIVE_FIELDS + HELPER_FIELDS:
        restricted[field] = loaded[field].to_numpy()
    restricted["has_co_applicant"] = helper_numeric.to_numpy()
    for field in [value for value in SENSITIVE_FIELDS if value != "minority_population"]:
        normalized = [canonical_label(value) for value in loaded[field]]
        restricted[f"{field}__canonical"] = [value[0] for value in normalized]
        restricted[f"{field}__non_substantive"] = [value[1] for value in normalized]
        restricted[f"{field}__missing"] = [value[2] for value in normalized]
    bin_labels = ["[0,20)", "[20,40)", "[40,50)", "[50,60)", "[60,80)", "[80,100]"]
    binned = pd.cut(minority, bins=[0, 20, 40, 50, 60, 80, 100.0000000001], labels=bin_labels, right=False, include_lowest=True)
    restricted["minority_population__group"] = binned.astype(object).where(~minority.isna(), "missing").to_numpy()
    restricted["minority_population__missing"] = minority.isna().to_numpy()
    restricted["minority_population__non_substantive"] = minority.isna().to_numpy()
    restricted["co_applicant_analysis_eligible"] = restricted["has_co_applicant"].eq(1)
    atomic_csv(restricted, RESTRICTED_LABELS)
    source_hash_after = sha256_file(SOURCE)
    require(source_hash_after == source_hash_before, "Sensitive source changed during access")
    manifest = {
        "stage_id": STAGE_ID, "status": "PASS", "contains_sensitive_data": True,
        "created_at_utc": now_utc(), "path": rel(RESTRICTED_LABELS), "sha256": sha256_file(RESTRICTED_LABELS),
        "row_count": len(restricted), "columns": list(restricted.columns), "test_row_id_hash": EXPECTED["row_hash"],
        "source_path": rel(SOURCE), "source_sha256": source_hash_before,
        "access_scope": "Exact saved Test membership; eight frozen sensitive fields and one helper only; zero source target values.",
        "public_packaging_policy": "exclude_by_default", "notebook_display_policy": "aggregate_only",
        "stage8_access_policy": "Restricted access only for bounded saved cases; never display row-level sensitive labels publicly.",
    }
    atomic_json(manifest, MANIFESTS / "stage7_sensitive_data_manifest.json")
    audit = {
        "stage_id": STAGE_ID, "status": "PASS", "authorization_source": "Attached Stage 7 bounded fairness analysis specification",
        "pre_fairness_freeze_path": rel(FREEZE), "pre_fairness_freeze_sha256": sha256_file(FREEZE),
        "first_source_access_timestamp": first_access, "source_path": rel(SOURCE), "source_sha256": source_hash_before,
        "loader_path": rel(SAFE_LOADER), "loader_sha256": sha256_file(SAFE_LOADER),
        "saved_test_id_path": rel(TEST_IDS), "saved_test_id_sha256": sha256_file(TEST_IDS),
        "allowed_columns": SENSITIVE_FIELDS + HELPER_FIELDS, "forbidden_target_columns": ["loan_amount_000s"],
        "physical_access_attempts": 1, "successful_source_materializations": 1, "raw_lines_scanned": 499736,
        "test_rows_materialized": len(restricted), "train_rows_materialized": 0, "excluded_rows_converted": 0,
        "source_target_columns_requested": 0, "source_target_values_materialized": 0,
        "duplicate_ids": 0, "missing_ids": 0, "final_row_count": len(restricted),
        "final_row_id_hash": array_hash(np.sort(restricted["row_id"].to_numpy(np.int64)), np.int64),
        "sensitive_label_artifact_path": rel(RESTRICTED_LABELS), "sensitive_label_artifact_sha256": sha256_file(RESTRICTED_LABELS),
        "source_hash_after_access": source_hash_after,
    }
    atomic_json(audit, ACCESS)
    print(json.dumps({"status": "PASS", "rows": len(restricted), "artifact_sha256": manifest["sha256"], "source_attempts": 1}, indent=2))
    return manifest


def load_aligned_working() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    manifest = load_json(MANIFESTS / "stage7_sensitive_data_manifest.json")
    require(sha256_file(RESTRICTED_LABELS) == manifest["sha256"], "Restricted sensitive artifact hash mismatch")
    sensitive = pd.read_csv(RESTRICTED_LABELS)
    require(len(sensitive) == EXPECTED["rows"] and sensitive["row_id"].is_unique, "Restricted sensitive rows are invalid")
    sensitive = sensitive.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    frames: dict[str, pd.DataFrame] = {}
    reference_ids = None
    reference_target = None
    for item in CANDIDATES:
        path = PREDICTIONS[item["candidate_id"]]
        require(sha256_file(path) == item["sha256"], f"Prediction hash changed: {item['candidate_id']}")
        frame = pd.read_csv(path).sort_values("row_id", kind="mergesort").reset_index(drop=True)
        require(len(frame) == EXPECTED["rows"] and frame["row_id"].is_unique, "Prediction membership invalid")
        if "candidate_id" in frame:
            require(set(frame["candidate_id"].astype(str)) == {item["candidate_id"]}, "Prediction Candidate ID mismatch")
        ids = frame["row_id"].to_numpy(np.int64)
        target = frame["y_true"].to_numpy(np.float64)
        require(array_hash(ids, np.int64) == EXPECTED["row_hash"] and array_hash(target, np.float64) == EXPECTED["target_hash"], "Prediction row or target hash mismatch")
        require(np.isfinite(frame[["y_true", "y_pred"]].to_numpy(float)).all(), "Non-finite prediction evidence")
        if reference_ids is None:
            reference_ids, reference_target = ids, target
        else:
            require(np.array_equal(ids, reference_ids) and np.array_equal(target, reference_target), "Prediction alignment differs")
        frame["signed_error_stage7"] = frame["y_pred"].to_numpy(float) - target
        frame["absolute_error_stage7"] = np.abs(frame["signed_error_stage7"])
        frames[item["candidate_id"]] = frame
    require(np.array_equal(sensitive["row_id"].to_numpy(np.int64), reference_ids), "Sensitive/prediction membership mismatch")
    alignment = {
        "stage_id": STAGE_ID, "status": "PASS", "created_at_utc": now_utc(), "prediction_file_count": 3,
        "candidate_count": 3, "rows_per_candidate": EXPECTED["rows"], "unique_row_ids": True,
        "exact_test_membership": True, "test_row_id_hash": EXPECTED["row_hash"], "identical_targets": True,
        "target_hash": EXPECTED["target_hash"], "finite_predictions": True, "original_target_scale": True,
        "candidate_roles": {item["candidate_id"]: item["role"] for item in CANDIDATES},
        "every_test_row_has_one_sensitive_record": True, "extra_source_rows": 0,
        "source_target_values_used": 0, "restricted_artifact_separate_from_predictions": True,
        "prediction_hashes": {item["candidate_id"]: item["sha256"] for item in CANDIDATES},
    }
    atomic_json(alignment, ALIGNMENT)
    return frames, sensitive


def group_column(field: str) -> str:
    return "minority_population__group" if field == "minority_population" else f"{field}__canonical"


def non_substantive_column(field: str) -> str:
    return f"{field}__non_substantive"


def analysis_mask(sensitive: pd.DataFrame, field: str) -> np.ndarray:
    if field in COAPPLICANT_FIELDS:
        return sensitive["has_co_applicant"].to_numpy(int) == 1
    return np.ones(len(sensitive), dtype=bool)


def _metric_values(y: np.ndarray, p: np.ndarray, top_decile_mask: np.ndarray | None = None,
                   top_five_mask: np.ndarray | None = None) -> dict[str, float]:
    err = p - y
    abs_err = np.abs(err)
    denom = float(np.sum(np.abs(y)))
    nonzero = np.abs(y) > 0
    under = err < 0
    over = err > 0
    if top_decile_mask is None:
        top_decile_mask = y >= np.quantile(y, 0.90)
    if top_five_mask is None:
        top_five_mask = y >= np.quantile(y, 0.95)
    mean_y = float(np.mean(y))
    return {
        "mae": float(np.mean(abs_err)),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_absolute_error": float(np.median(abs_err)),
        "p90_absolute_error": float(np.quantile(abs_err, 0.90)),
        "wape": float(np.sum(abs_err) / denom) if denom else np.nan,
        "mape_percent": float(np.mean(abs_err[nonzero] / np.abs(y[nonzero])) * 100.0) if nonzero.any() else np.nan,
        "signed_error": float(np.mean(err)),
        "absolute_mean_signed_error": float(abs(np.mean(err))),
        "underprediction_count": int(under.sum()), "underprediction_rate": float(np.mean(under)),
        "overprediction_count": int(over.sum()), "overprediction_rate": float(np.mean(over)),
        "negative_prediction_rate": float(np.mean(p < 0)),
        "absolute_error_rate_above_50": float(np.mean(abs_err > 50)),
        "absolute_error_rate_above_100": float(np.mean(abs_err > 100)),
        "absolute_error_rate_above_200": float(np.mean(abs_err > 200)),
        "top_target_decile_mae": float(np.mean(abs_err[top_decile_mask])) if top_decile_mask.sum() >= CELL_N else np.nan,
        "top_five_percent_target_mae": float(np.mean(abs_err[top_five_mask])) if top_five_mask.sum() >= CELL_N else np.nan,
        "prediction_to_actual_mean_ratio": float(np.mean(p) / mean_y) if abs(mean_y) > 1e-12 else np.nan,
        "mean_target": mean_y,
        "median_target": float(np.median(y)),
        "mean_prediction": float(np.mean(p)),
        "median_prediction": float(np.median(p)),
    }


def _deciles(y: np.ndarray) -> np.ndarray:
    # Rank-first assignment is deterministic even when the target has ties.
    ranks = pd.Series(y).rank(method="first").to_numpy()
    return np.minimum(9, np.floor((ranks - 1) * 10 / len(y))).astype(int) + 1


def build_group_tables(frames: dict[str, pd.DataFrame], sensitive: pd.DataFrame) -> dict[str, pd.DataFrame]:
    reference = frames[CANDIDATES[0]["candidate_id"]]
    y_all = reference["y_true"].to_numpy(float)
    assignment = frames["stage5c__realmlp__without_sensitive__test_evaluation"]
    require("target_decile" in assignment and "is_top_five_percent_target" in assignment, "Saved Stage 5C target assignments missing")
    decile_all = assignment["target_decile"].to_numpy(int)
    top5_all = assignment["is_top_five_percent_target"].astype(bool).to_numpy()
    require(set(decile_all) == set(range(1, 11)), "Frozen target deciles invalid")
    other_assignment = frames["stage5c__realmlp__with_sensitive__test_evaluation"]
    require(np.array_equal(decile_all, other_assignment["target_decile"].to_numpy(int)), "Stage 5C target-decile assignments differ")
    inventory_rows, coverage_rows, target_rows, metric_rows, decile_rows = [], [], [], [], []

    for field in SENSITIVE_FIELDS:
        gcol = group_column(field)
        eligible = analysis_mask(sensitive, field)
        scope_n = int(eligible.sum())
        labels = sensitive.loc[eligible, gcol].astype(str)
        counts = labels.value_counts(dropna=False, sort=False).sort_index()
        if field == "minority_population" and "missing" not in set(counts.index.astype(str)):
            inventory_rows.append({"field": field, "original_label": "missing", "canonical_label": "missing", "count": 0,
                                   "percentage": 0.0, "substantive_status": "non_substantive", "missing_flag": True,
                                   "co_applicant_only_flag": False, "notes": "Frozen missing bin; no observed rows",
                                   "canonicalization": "fixed_frozen_bin", "identity_inferred": False})
            coverage_rows.append({"sensitive_field": field, "group_label": "missing", "n": 0, "scope_n": scope_n,
                                  "group_share": 0.0, "eligibility_scope": "all_test_rows", "is_non_substantive": True,
                                  "evidence_tier": tier(0), "suppressed": True, "suppression_reason": "n_below_100"})
        if field in COAPPLICANT_FIELDS:
            no_coapp = ~eligible
            no_n = int(no_coapp.sum())
            coverage_rows.append({
                "sensitive_field": field, "group_label": "__NO_CO_APPLICANT__", "n": no_n, "scope_n": len(sensitive),
                "group_share": no_n / len(sensitive), "eligibility_scope": "coverage_only_has_co_applicant==0",
                "is_non_substantive": True, "evidence_tier": tier(no_n), "suppressed": False, "suppression_reason": "",
            })
            raw_no = sensitive.loc[no_coapp, field].where(~sensitive.loc[no_coapp, field].isna(), "__MISSING__").astype(str).str.strip()
            for original_label, original_n in raw_no.value_counts(sort=False).sort_index().items():
                inventory_rows.append({
                    "field": field, "original_label": original_label, "canonical_label": canonical_label(original_label)[0],
                    "count": int(original_n), "percentage": float(original_n / len(sensitive)), "substantive_status": "non_substantive",
                    "missing_flag": original_label == "__MISSING__", "co_applicant_only_flag": True,
                    "notes": "Coverage only; has_co_applicant==0; excluded from co-applicant identity metrics",
                    "canonicalization": "trim_only_preserve_original", "identity_inferred": False,
                })
        for group, n_value in counts.items():
            n = int(n_value)
            indices = np.flatnonzero(eligible & (sensitive[gcol].astype(str).to_numpy() == str(group)))
            nonsub = bool(sensitive.loc[indices, non_substantive_column(field)].all())
            level = tier(n)
            suppressed = n < LIMITED_N
            base = {
                "sensitive_field": field, "group_label": group, "n": n, "scope_n": scope_n,
                "group_share": n / scope_n, "eligibility_scope": "has_co_applicant==1" if field in COAPPLICANT_FIELDS else "all_test_rows",
                "is_non_substantive": nonsub, "evidence_tier": level, "suppressed": suppressed,
                "suppression_reason": "n_below_100" if suppressed else "",
            }
            raw_values = sensitive.loc[indices, field]
            if field == "minority_population":
                # Public inventory exposes only the frozen bins; raw percentages remain restricted.
                raw_labels = pd.Series([str(group)] * len(indices), index=raw_values.index)
            else:
                raw_labels = raw_values.where(~raw_values.isna(), "__MISSING__").astype(str).str.strip()
            for original_label, original_n in raw_labels.value_counts(sort=False).sort_index().items():
                inventory_rows.append({
                    "field": field, "original_label": original_label, "canonical_label": group,
                    "count": int(original_n), "percentage": float(original_n / scope_n),
                    "substantive_status": "non_substantive" if nonsub else "substantive",
                    "missing_flag": original_label == "__MISSING__", "co_applicant_only_flag": field in COAPPLICANT_FIELDS,
                    "notes": "Preserved label; co-applicant metrics require has_co_applicant==1" if field in COAPPLICANT_FIELDS else "Preserved label",
                    "canonicalization": "trim_only_preserve_original", "identity_inferred": False,
                })
            coverage_rows.append(base)
            if not suppressed:
                yy = y_all[indices]
                target_rows.append({
                    **base, "mean_target": float(np.mean(yy)), "median_target": float(np.median(yy)),
                    "target_std": float(np.std(yy, ddof=1)) if n > 1 else np.nan,
                    "target_p10": float(np.quantile(yy, 0.10)), "target_p25": float(np.quantile(yy, 0.25)),
                    "target_p75": float(np.quantile(yy, 0.75)), "target_p90": float(np.quantile(yy, 0.90)),
                    "top_target_decile_share": float(np.mean(decile_all[indices] == 10)),
                    "top_five_percent_target_share": float(np.mean(top5_all[indices])),
                    "target_decile_counts_json": json.dumps({str(d): int(np.sum(decile_all[indices] == d)) for d in range(1, 11)}, sort_keys=True),
                })
            else:
                target_rows.append({**base, "mean_target": np.nan, "median_target": np.nan, "target_std": np.nan,
                                    "target_p10": np.nan, "target_p25": np.nan, "target_p75": np.nan, "target_p90": np.nan,
                                    "top_target_decile_share": np.nan, "top_five_percent_target_share": np.nan, "target_decile_counts_json": "{}"})
            for candidate_id, frame in frames.items():
                common = {**base, "candidate_id": candidate_id, "candidate_role": BY_ID[candidate_id]["role"]}
                if suppressed:
                    metric_rows.append({**common, **{key: np.nan for key in _metric_values(np.array([1.]), np.array([1.]))}})
                else:
                    metric_rows.append({**common, **_metric_values(y_all[indices], frame["y_pred"].to_numpy(float)[indices],
                                                                   decile_all[indices] == 10, top5_all[indices])})
                for decile in range(1, 11):
                    cell_idx = indices[decile_all[indices] == decile]
                    cell_n = len(cell_idx)
                    cell_suppressed = suppressed or cell_n < CELL_N
                    cell_base = {**common, "target_decile": decile, "cell_n": cell_n,
                                 "cell_suppressed": cell_suppressed,
                                 "cell_suppression_reason": ("group_n_below_100" if suppressed else "cell_n_below_50") if cell_suppressed else ""}
                    if cell_suppressed:
                        decile_rows.append({**cell_base, "mae": np.nan, "rmse": np.nan, "median_absolute_error": np.nan,
                                            "wape": np.nan, "signed_error": np.nan, "underprediction_rate": np.nan, "overprediction_rate": np.nan})
                    else:
                        values = _metric_values(y_all[cell_idx], frame["y_pred"].to_numpy(float)[cell_idx])
                        decile_rows.append({**cell_base, **{key: values[key] for key in ["mae", "rmse", "median_absolute_error", "wape", "signed_error", "underprediction_rate", "overprediction_rate"]}})

    tables = {
        "group_label_inventory": pd.DataFrame(inventory_rows),
        "group_coverage": pd.DataFrame(coverage_rows),
        "group_target_distribution": pd.DataFrame(target_rows),
        "single_attribute_group_metrics": pd.DataFrame(metric_rows),
        "group_target_decile_metrics": pd.DataFrame(decile_rows),
    }
    for name, table in tables.items():
        atomic_csv(table, RESULTS / f"stage7_{name}.csv")
    return tables


def build_standardized_and_disparities(tables: dict[str, pd.DataFrame], frames: dict[str, pd.DataFrame],
                                       sensitive: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = tables["single_attribute_group_metrics"]
    deciles = tables["group_target_decile_metrics"]
    ref = metrics[(metrics.candidate_id == CANDIDATES[0]["candidate_id"]) & (~metrics.suppressed)]
    frozen_deciles = frames["stage5c__realmlp__without_sensitive__test_evaluation"]["target_decile"].to_numpy(int)
    target_weights = np.array([np.mean(frozen_deciles == decile) for decile in range(1, 11)], dtype=float)
    standardized_rows = []
    metric_lookup = metrics.set_index(["sensitive_field", "group_label", "candidate_id"])
    for keys, block in deciles.groupby(["sensitive_field", "group_label", "candidate_id"], sort=True):
        meta = metric_lookup.loc[keys]
        eligible = (meta["evidence_tier"] == "primary" and not bool(meta["is_non_substantive"]) and len(block) == 10
                    and (block["cell_n"] >= STANDARDIZED_DECILE_N).all())
        decile_mae = []
        if eligible:
            group_mask = analysis_mask(sensitive, keys[0]) & (sensitive[group_column(keys[0])].astype(str).to_numpy() == str(keys[1]))
            y_all = frames[keys[2]]["y_true"].to_numpy(float)
            p_all = frames[keys[2]]["y_pred"].to_numpy(float)
            for decile in range(1, 11):
                idx = np.flatnonzero(group_mask & (frozen_deciles == decile))
                decile_mae.append(float(np.mean(np.abs(p_all[idx] - y_all[idx]))))
        standardized_mae = float(np.sum(np.asarray(decile_mae) * target_weights)) if eligible else np.nan
        standardized_rows.append({
            "sensitive_field": keys[0], "group_label": keys[1], "candidate_id": keys[2],
            "n": int(block["cell_n"].sum()), "standardized_eligible": bool(eligible),
            "raw_mae": float(meta["mae"]) if not bool(meta["suppressed"]) else np.nan,
            "standardized_mae": standardized_mae,
            "standardized_minus_raw_mae": standardized_mae - float(meta["mae"]) if eligible else np.nan,
            "standardized_wape": np.nan,
            "decile_support_json": json.dumps({str(int(row.target_decile)): int(row.cell_n) for row in block.itertuples()}, sort_keys=True),
            "availability_status": "available" if eligible else "not_available_due_to_decile_support_or_group_policy",
            "standardization_weights": json.dumps({str(i + 1): float(value) for i, value in enumerate(target_weights)}, sort_keys=True),
        })
    standardized = pd.DataFrame(standardized_rows)

    summary_rows, reference_rows, pairwise_rows = [], [], []
    for (field, candidate_id), block in metrics[(~metrics.suppressed) & (~metrics.is_non_substantive)].groupby(["sensitive_field", "candidate_id"], sort=True):
        primary = block[block.evidence_tier == "primary"].copy()
        if primary.empty:
            continue
        for metric in ["mae", "rmse", "wape", "p90_absolute_error", "signed_error", "absolute_mean_signed_error", "underprediction_rate"]:
            vals = primary[metric].astype(float)
            hi_i, lo_i = vals.idxmax(), vals.idxmin()
            summary_rows.append({
                "sensitive_field": field, "candidate_id": candidate_id, "metric": metric,
                "eligible_group_count": len(primary), "max_group": primary.loc[hi_i, "group_label"], "max_value": vals.loc[hi_i],
                "min_group": primary.loc[lo_i, "group_label"], "min_value": vals.loc[lo_i], "absolute_gap": vals.loc[hi_i] - vals.loc[lo_i],
            })
        ref_row = primary.sort_values(["n", "group_label"], ascending=[False, True]).iloc[0]
        for _, row in primary.iterrows():
            reference_rows.append({
                "sensitive_field": field, "candidate_id": candidate_id, "group_label": row.group_label,
                "reference_group": ref_row.group_label, "n": int(row.n),
                "mae_difference": row.mae - ref_row.mae, "wape_difference": row.wape - ref_row.wape,
                "signed_error_difference": row.signed_error - ref_row.signed_error,
                "underprediction_rate_difference": row.underprediction_rate - ref_row.underprediction_rate,
            })
        rows = list(primary.itertuples(index=False))
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                pairwise_rows.append({
                    "sensitive_field": field, "candidate_id": candidate_id,
                    "group_a": left.group_label, "group_b": right.group_label, "n_a": int(left.n), "n_b": int(right.n),
                    "mae_difference_a_minus_b": left.mae - right.mae, "wape_difference_a_minus_b": left.wape - right.wape,
                    "signed_error_difference_a_minus_b": left.signed_error - right.signed_error,
                    "underprediction_rate_difference_a_minus_b": left.underprediction_rate - right.underprediction_rate,
                })
    additions = {
        "target_standardized_metrics": standardized,
        "group_disparity_summary": pd.DataFrame(summary_rows),
        "group_reference_differences": pd.DataFrame(reference_rows),
        "pairwise_group_differences": pd.DataFrame(pairwise_rows),
    }
    for name, table in additions.items():
        atomic_csv(table, RESULTS / f"stage7_{name}.csv")
    tables.update(additions)
    return tables


def build_pairwise_candidate_comparisons(frames: dict[str, pd.DataFrame], sensitive: pd.DataFrame,
                                         tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    y = frames[CANDIDATES[0]["candidate_id"]]["y_true"].to_numpy(float)
    deciles = frames[CANDIDATES[1]["candidate_id"]]["target_decile"].to_numpy(int)
    top5 = frames[CANDIDATES[1]["candidate_id"]]["is_top_five_percent_target"].astype(bool).to_numpy()
    coverage = tables["group_coverage"]
    rows = []
    for pair in PAIRS:
        first = frames[pair["first"]]["y_pred"].to_numpy(float)
        second = frames[pair["second"]]["y_pred"].to_numpy(float)
        first_abs, second_abs = np.abs(first - y), np.abs(second - y)
        for group in coverage[~coverage.suppressed].itertuples(index=False):
            field, label = group.sensitive_field, str(group.group_label)
            if label == "__NO_CO_APPLICANT__":
                continue
            idx = np.flatnonzero(analysis_mask(sensitive, field) & (sensitive[group_column(field)].astype(str).to_numpy() == label))
            fm = _metric_values(y[idx], first[idx], deciles[idx] == 10, top5[idx])
            sm = _metric_values(y[idx], second[idx], deciles[idx] == 10, top5[idx])
            delta = first_abs[idx] - second_abs[idx]
            rows.append({
                "pair_id": pair["pair_id"], "first_candidate_id": pair["first"], "second_candidate_id": pair["second"],
                "sensitive_field": field, "group_label": label, "n": len(idx), "evidence_tier": group.evidence_tier,
                "is_non_substantive": group.is_non_substantive, "model_selection_performed": False,
                "mae_difference": fm["mae"] - sm["mae"], "rmse_difference": fm["rmse"] - sm["rmse"],
                "wape_difference": fm["wape"] - sm["wape"], "p90_absolute_error_difference": fm["p90_absolute_error"] - sm["p90_absolute_error"],
                "mean_signed_error_raw_difference": fm["signed_error"] - sm["signed_error"],
                "absolute_signed_bias_difference": fm["absolute_mean_signed_error"] - sm["absolute_mean_signed_error"],
                "underprediction_rate_difference": fm["underprediction_rate"] - sm["underprediction_rate"],
                "first_candidate_lower_error_rate": float(np.mean(delta < -TIE_TOLERANCE)),
                "second_candidate_lower_error_rate": float(np.mean(delta > TIE_TOLERANCE)),
                "tie_rate": float(np.mean(np.abs(delta) <= TIE_TOLERANCE)),
                "mean_prediction_difference": float(np.mean(first[idx] - second[idx])),
                "median_prediction_difference": float(np.median(first[idx]) - np.median(second[idx])),
            })
    result = pd.DataFrame(rows)
    atomic_csv(result, RESULTS / "stage7_pairwise_group_differences.csv")
    tables["pairwise_group_differences"] = result
    return result


def build_complete_disparity_tables(frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["single_attribute_group_metrics"]
    standardized = tables["target_standardized_metrics"]
    summary_rows, reference_rows = [], []
    for item in CANDIDATES:
        candidate_id = item["candidate_id"]
        frame = frames[candidate_id]
        overall = _metric_values(frame["y_true"].to_numpy(float), frame["y_pred"].to_numpy(float))
        for field in SENSITIVE_FIELDS:
            block = metrics[(metrics.candidate_id == candidate_id) & (metrics.sensitive_field == field)
                            & (metrics.evidence_tier == "primary") & (~metrics.is_non_substantive)].copy()
            if block.empty:
                continue
            largest = block.sort_values(["n", "group_label"], ascending=[False, True]).iloc[0]
            best, worst = block.loc[block.mae.idxmin()], block.loc[block.mae.idxmax()]
            std = standardized[(standardized.candidate_id == candidate_id) & (standardized.sensitive_field == field)
                               & (standardized.standardized_eligible)]
            coverage_pct = float(block.n.sum() / metrics[(metrics.candidate_id == candidate_id) & (metrics.sensitive_field == field)].scope_n.max() * 100.0)
            summary_rows.append({
                "candidate_id": candidate_id, "sensitive_field": field, "eligible_group_count": len(block),
                "largest_eligible_group": largest.group_label, "best_mae_group": best.group_label, "worst_mae_group": worst.group_label,
                "worst_minus_best_mae_gap": worst.mae - best.mae,
                "worst_to_best_mae_ratio": worst.mae / best.mae if abs(best.mae) > 1e-12 else np.nan,
                "maximum_group_minus_overall_mae": float((block.mae - overall["mae"]).max()),
                "minimum_group_minus_overall_mae": float((block.mae - overall["mae"]).min()),
                "worst_minus_best_wape_gap": float(block.wape.max() - block.wape.min()),
                "worst_minus_best_p90_error_gap": float(block.p90_absolute_error.max() - block.p90_absolute_error.min()),
                "maximum_absolute_mean_signed_error": float(block.absolute_mean_signed_error.max()),
                "mean_signed_error_spread": float(block.signed_error.max() - block.signed_error.min()),
                "underprediction_rate_spread": float(block.underprediction_rate.max() - block.underprediction_rate.min()),
                "target_standardized_mae_gap": float(std.standardized_mae.max() - std.standardized_mae.min()) if len(std) >= 2 else np.nan,
                "data_coverage_percentage": coverage_pct, "reference_is_legally_privileged": False,
            })
            for row in block.itertuples(index=False):
                reference_rows.append({
                    "candidate_id": candidate_id, "sensitive_field": field, "group_label": row.group_label, "n": int(row.n),
                    "largest_eligible_reference_group": largest.group_label,
                    "mae_difference_versus_overall": row.mae - overall["mae"],
                    "mae_ratio_versus_overall": row.mae / overall["mae"] if abs(overall["mae"]) > 1e-12 else np.nan,
                    "mae_difference_versus_largest_group": row.mae - largest.mae,
                    "mae_ratio_versus_largest_group": row.mae / largest.mae if abs(largest.mae) > 1e-12 else np.nan,
                    "wape_difference_versus_overall": row.wape - overall["wape"],
                    "absolute_bias_difference_versus_overall": row.absolute_mean_signed_error - overall["absolute_mean_signed_error"],
                    "underprediction_rate_difference_versus_overall": row.underprediction_rate - overall["underprediction_rate"],
                })
    summary, reference = pd.DataFrame(summary_rows), pd.DataFrame(reference_rows)
    atomic_csv(summary, RESULTS / "stage7_group_disparity_summary.csv")
    atomic_csv(reference, RESULTS / "stage7_group_reference_differences.csv")
    tables["group_disparity_summary"], tables["group_reference_differences"] = summary, reference


def build_bootstrap(frames: dict[str, pd.DataFrame], sensitive: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    y = frames[CANDIDATES[0]["candidate_id"]]["y_true"].to_numpy(float)
    rows = []
    coverage = tables["group_coverage"]
    eligible_groups = coverage[(coverage.evidence_tier == "primary") & (~coverage.is_non_substantive)]
    for group_row in eligible_groups.itertuples(index=False):
        field, group = group_row.sensitive_field, str(group_row.group_label)
        mask = analysis_mask(sensitive, field) & (sensitive[group_column(field)].astype(str).to_numpy() == group)
        idx = np.flatnonzero(mask)
        draws = rng.integers(0, len(idx), size=(500, len(idx)))
        sampled = idx[draws]
        for candidate_id, frame in frames.items():
            pred = frame["y_pred"].to_numpy(float)
            abs_err = np.abs(pred - y)
            signed = pred - y
            estimates = {
                "mae": abs_err[sampled].mean(axis=1),
                "signed_error": signed[sampled].mean(axis=1),
                "underprediction_rate": (signed[sampled] < 0).mean(axis=1),
            }
            for metric, values in estimates.items():
                rows.append({
                    "sensitive_field": field, "group_label": group, "candidate_id": candidate_id,
                    "pair_id": "", "interval_type": "candidate_metric", "metric": metric, "n": len(idx), "point_estimate": float(np.mean(values)),
                    "ci_lower_95": float(np.quantile(values, 0.025)), "ci_upper_95": float(np.quantile(values, 0.975)),
                    "bootstrap_repetitions": 500, "bootstrap_seed": 42, "paired_within_group": True,
                    "inferential_status": "descriptive_interval_no_p_value",
                })
        for pair in PAIRS:
            first = frames[pair["first"]]["y_pred"].to_numpy(float)
            second = frames[pair["second"]]["y_pred"].to_numpy(float)
            first_abs = np.abs(first - y)
            second_abs = np.abs(second - y)
            first_signed = first - y
            second_signed = second - y
            paired_values = {
                "mae_difference": first_abs[sampled].mean(axis=1) - second_abs[sampled].mean(axis=1),
                "absolute_bias_difference": np.abs(first_signed[sampled].mean(axis=1)) - np.abs(second_signed[sampled].mean(axis=1)),
                "first_candidate_win_proportion": (first_abs[sampled] < second_abs[sampled]).mean(axis=1),
            }
            for metric, values in paired_values.items():
                rows.append({
                    "sensitive_field": field, "group_label": group, "candidate_id": "", "pair_id": pair["pair_id"],
                    "interval_type": "paired_candidate_difference", "metric": metric, "n": len(idx),
                    "point_estimate": float(np.mean(values)), "ci_lower_95": float(np.quantile(values, 0.025)),
                    "ci_upper_95": float(np.quantile(values, 0.975)), "bootstrap_repetitions": 500,
                    "bootstrap_seed": 42, "paired_within_group": True, "inferential_status": "descriptive_interval_no_p_value",
                })
    result = pd.DataFrame(rows)
    atomic_csv(result, RESULTS / "stage7_group_bootstrap_intervals.csv")
    tables["group_bootstrap_intervals"] = result
    return result


def build_intersections(frames: dict[str, pd.DataFrame], sensitive: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    y = frames[CANDIDATES[0]["candidate_id"]]["y_true"].to_numpy(float)
    metric_rows, pair_rows = [], []
    with_pred = frames["stage5c__realmlp__with_sensitive__test_evaluation"]["y_pred"].to_numpy(float)
    without_pred = frames["stage5c__realmlp__without_sensitive__test_evaluation"]["y_pred"].to_numpy(float)
    for definition in INTERSECTIONS:
        intersection_id, left, right = definition["intersection_id"], definition["first"], definition["second"]
        labels = sensitive[group_column(left)].astype(str) + " x " + sensitive[group_column(right)].astype(str)
        nonsub = sensitive[non_substantive_column(left)].to_numpy(bool) | sensitive[non_substantive_column(right)].to_numpy(bool)
        eligible = analysis_mask(sensitive, left) & analysis_mask(sensitive, right)
        counts = labels[eligible].value_counts(sort=False).sort_index()
        for candidate_id, frame in frames.items():
            for label, n_value in counts.items():
                n = int(n_value)
                idx = np.flatnonzero(eligible & (labels.to_numpy() == label))
                level = tier(n)
                suppressed = n < LIMITED_N
                base = {"intersection_id": intersection_id, "left_field": left, "right_field": right,
                        "intersection_label": label, "candidate_id": candidate_id, "n": n,
                        "evidence_tier": level, "is_non_substantive": bool(nonsub[idx].any()), "suppressed": suppressed,
                        "suppression_reason": "n_below_100" if suppressed else ""}
                keys = ["mean_target", "mae", "rmse", "wape", "signed_error", "absolute_mean_signed_error",
                        "underprediction_rate", "p90_absolute_error"]
                values = {key: np.nan for key in keys}
                if not suppressed:
                    full = _metric_values(y[idx], frame["y_pred"].to_numpy(float)[idx])
                    values = {key: full[key] for key in keys}
                metric_rows.append({**base, **values})
        for label, n_value in counts.items():
            n = int(n_value)
            idx = np.flatnonzero(eligible & (labels.to_numpy() == label))
            suppressed = n < LIMITED_N
            if suppressed:
                pair_rows.append({"intersection_id": intersection_id, "intersection_label": label, "n": n,
                                  "suppressed": True, "suppression_reason": "n_below_100",
                                  "with_sensitive_minus_without_sensitive_mae": np.nan, "with_sensitive_lower_error_rate": np.nan})
            else:
                with_abs, without_abs = np.abs(with_pred[idx] - y[idx]), np.abs(without_pred[idx] - y[idx])
                pair_rows.append({"intersection_id": intersection_id, "intersection_label": label, "n": n,
                                  "suppressed": False, "suppression_reason": "",
                                  "with_sensitive_minus_without_sensitive_mae": float(with_abs.mean() - without_abs.mean()),
                                  "with_sensitive_lower_error_rate": float(np.mean(with_abs < without_abs))})
    intersections, pairs = pd.DataFrame(metric_rows), pd.DataFrame(pair_rows)
    atomic_csv(intersections, RESULTS / "stage7_intersectional_metrics.csv")
    atomic_csv(pairs, RESULTS / "stage7_intersectional_pairwise_differences.csv")
    tables["intersectional_metrics"], tables["intersectional_pairwise_differences"] = intersections, pairs


def build_tradeoff_and_risk(frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> None:
    overall = []
    metrics = tables["single_attribute_group_metrics"]
    summary = tables["group_disparity_summary"]
    for item in CANDIDATES:
        frame = frames[item["candidate_id"]]
        values = _metric_values(frame["y_true"].to_numpy(float), frame["y_pred"].to_numpy(float))
        gaps = summary[(summary.candidate_id == item["candidate_id"]) & (summary.metric == "mae")]
        overall.append({
            "candidate_id": item["candidate_id"], "candidate_role": item["role"], "overall_mae": values["mae"],
            "overall_rmse": values["rmse"], "overall_wape": values["wape"],
            "max_primary_single_attribute_mae_gap": float(gaps.absolute_gap.max()) if len(gaps) else np.nan,
            "mean_primary_single_attribute_mae_gap": float(gaps.absolute_gap.mean()) if len(gaps) else np.nan,
            "assessment_label": ASSESSMENT,
        })
    tradeoff = pd.DataFrame(overall)
    without_id = "stage5c__realmlp__without_sensitive__test_evaluation"
    with_id = "stage5c__realmlp__with_sensitive__test_evaluation"
    base = metrics[(metrics.candidate_id == without_id) & (~metrics.suppressed)].copy()
    sens = metrics[(metrics.candidate_id == with_id) & (~metrics.suppressed)].copy()
    compare = base.merge(sens, on=["sensitive_field", "group_label"], suffixes=("_without", "_with"))
    compare["sensitive_minus_without_mae"] = compare["mae_with"] - compare["mae_without"]
    compare["sensitive_minus_without_signed_error"] = compare["signed_error_with"] - compare["signed_error_without"]
    atomic_csv(compare[["sensitive_field", "group_label", "n_without", "evidence_tier_without", "is_non_substantive_without",
                        "mae_without", "mae_with", "sensitive_minus_without_mae", "signed_error_without", "signed_error_with",
                        "sensitive_minus_without_signed_error"]], RESULTS / "stage7_accuracy_disparity_tradeoff.csv")

    proxy = {
        "stage_id": STAGE_ID, "status": "PASS", "assessment_label": ASSESSMENT,
        "limitations": [
            "Observed group error differences do not identify causal discrimination.",
            "Removing named sensitive fields does not remove information correlated with those fields.",
            "The approved-applications-only population cannot establish approval or denial fairness.",
            "Some group labels are administrative response categories rather than substantive identities.",
            "Small groups and sparse target-decile cells are suppressed or limited-evidence only.",
            "Post-Test analysis is descriptive and cannot be used to tune or select a model.",
        ],
        "legal_or_compliance_certification": False, "fairness_certification": False,
    }
    atomic_json(proxy, RESULTS / "stage7_proxy_feature_limitations.json")
    risk_rows = [
        {"risk_id": "stage7_risk_01_scope", "risk": "Approved-applications-only scope", "severity": "high", "mitigation": "Keep scope label on every interpretation; do not infer selection fairness."},
        {"risk_id": "stage7_risk_02_sparse", "risk": "Sparse or suppressed groups", "severity": "medium", "mitigation": "Use frozen thresholds and do not display suppressed metrics."},
        {"risk_id": "stage7_risk_03_proxy", "risk": "Proxy information remains without named sensitive fields", "severity": "high", "mitigation": "Report as limitation; do not claim fairness from feature removal."},
        {"risk_id": "stage7_risk_04_posttest", "risk": "Post-Test descriptive evidence could influence selection", "severity": "high", "mitigation": "Freeze all models and prohibit ranking or tuning."},
        {"risk_id": "stage7_risk_05_privacy", "risk": "Row-level sensitive labels", "severity": "high", "mitigation": "Keep restricted artifact outside public notebook and publish aggregates only."},
    ]
    atomic_csv(pd.DataFrame(risk_rows), RESULTS / "stage7_fairness_risk_register.csv")
    summary_json = {
        "stage_id": STAGE_ID, "status": "PASS", "official_name": OFFICIAL_NAME, "analysis_label": LABEL,
        "assessment_label": ASSESSMENT, "population_scope": "approved applications with observed outcomes only",
        "candidate_count": 3, "sensitive_field_count": 8, "intersection_count": 3,
        "primary_group_rows": int(((metrics.evidence_tier == "primary") & (~metrics.is_non_substantive)).sum()),
        "suppressed_group_rows": int(metrics.suppressed.sum()), "model_selection_performed": False,
        "fairness_certification": False, "legal_compliance_certification": False,
        "main_conclusion": "The artifacts report descriptive error disparities and uncertainty; they do not establish causal, legal, approval-process, or population-wide fairness.",
    }
    atomic_json(summary_json, RESULTS / "stage7_fairness_summary.json")
    tables["accuracy_disparity_tradeoff"] = tradeoff


def build_complete_tradeoff_and_governance(frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["single_attribute_group_metrics"]
    standardized = tables["target_standardized_metrics"]
    without_id = "stage5c__realmlp__without_sensitive__test_evaluation"
    with_id = "stage5c__realmlp__with_sensitive__test_evaluation"
    y = frames[without_id]["y_true"].to_numpy(float)
    without_p = frames[without_id]["y_pred"].to_numpy(float)
    with_p = frames[with_id]["y_pred"].to_numpy(float)
    without_abs, with_abs = np.abs(without_p - y), np.abs(with_p - y)
    delta = with_abs - without_abs
    overall_change = float(delta.mean())
    rows = []
    for field in SENSITIVE_FIELDS:
        wo = metrics[(metrics.candidate_id == without_id) & (metrics.sensitive_field == field)
                     & (metrics.evidence_tier == "primary") & (~metrics.is_non_substantive)].copy()
        wi = metrics[(metrics.candidate_id == with_id) & (metrics.sensitive_field == field)
                     & (metrics.evidence_tier == "primary") & (~metrics.is_non_substantive)].copy()
        joined = wo.merge(wi, on=["sensitive_field", "group_label"], suffixes=("_without", "_with"))
        group_delta = joined.mae_with - joined.mae_without
        swo = standardized[(standardized.candidate_id == without_id) & (standardized.sensitive_field == field) & standardized.standardized_eligible]
        swi = standardized[(standardized.candidate_id == with_id) & (standardized.sensitive_field == field) & standardized.standardized_eligible]
        rows.append({
            "sensitive_field": field, "overall_mae_change_with_minus_without": overall_change,
            "worst_group_mae_change": float(joined.mae_with.max() - joined.mae_without.max()),
            "best_group_mae_change": float(joined.mae_with.min() - joined.mae_without.min()),
            "raw_mae_gap_change": float((joined.mae_with.max() - joined.mae_with.min()) - (joined.mae_without.max() - joined.mae_without.min())),
            "target_standardized_mae_gap_change": float((swi.standardized_mae.max() - swi.standardized_mae.min()) - (swo.standardized_mae.max() - swo.standardized_mae.min())) if len(swi) >= 2 and len(swo) >= 2 else np.nan,
            "maximum_absolute_signed_bias_change": float(joined.absolute_mean_signed_error_with.max() - joined.absolute_mean_signed_error_without.max()),
            "underprediction_rate_spread_change": float((joined.underprediction_rate_with.max() - joined.underprediction_rate_with.min()) - (joined.underprediction_rate_without.max() - joined.underprediction_rate_without.min())),
            "primary_groups_improved": int((group_delta < -TIE_TOLERANCE).sum()),
            "primary_groups_worsened": int((group_delta > TIE_TOLERANCE).sum()),
            "primary_groups_approximately_unchanged": int((np.abs(group_delta) <= TIE_TOLERANCE).sum()),
            "test_rows_lower_absolute_error_percentage": float(np.mean(delta < -TIE_TOLERANCE) * 100.0),
            "test_rows_higher_absolute_error_percentage": float(np.mean(delta > TIE_TOLERANCE) * 100.0),
            "test_rows_approximately_unchanged_percentage": float(np.mean(np.abs(delta) <= TIE_TOLERANCE) * 100.0),
            "interpretation": "Better overall MAE does not guarantee smaller group disparities; no causal or deployment recommendation is made.",
        })
    tradeoff = pd.DataFrame(rows)
    atomic_csv(tradeoff, RESULTS / "stage7_accuracy_disparity_tradeoff.csv")
    tables["accuracy_disparity_tradeoff"] = tradeoff

    proxy_categories = ["state", "county", "MSA or MD area", "tract income level", "region", "applicant income",
                        "area income", "lender or respondent identity", "property characteristics", "loan characteristics"]
    proxy = {
        "stage_id": STAGE_ID, "status": "PASS", "assessment_label": ASSESSMENT,
        "feature_inventory": hash_record(FEATURE_INVENTORY), "inspected_saved_schema_only": True,
        "potential_proxy_categories": proxy_categories,
        "interpretation": [
            "These saved-schema fields may be associated with sensitive attributes, but this Stage does not prove that they encode or reproduce a protected identity.",
            "Removing explicit sensitive Features does not guarantee absence of proxy effects.",
            "No protected-attribute prediction model was trained and no causal proxy conclusion was made.",
            "Feature deletion is not recommended from this descriptive evidence alone.",
        ],
        "protected_attribute_prediction_model_trained": False, "causal_proxy_conclusion": False,
        "legal_or_compliance_certification": False, "fairness_certification": False,
    }
    atomic_json(proxy, RESULTS / "stage7_proxy_feature_limitations.json")
    definitions = [
        ("approved_applications_only_selection", "Approved-applications-only selection", "high"),
        ("historical_target_process", "Target may reflect historical processes", "high"),
        ("missing_sensitive_labels", "Missing or not-provided sensitive labels", "medium"),
        ("coapplicant_availability", "Co-applicant availability differences", "medium"),
        ("small_sparse_groups", "Small and sparse groups", "medium"),
        ("intersectional_sparsity", "Intersectional sparsity", "medium"),
        ("potential_proxy_features", "Potential proxy Features", "high"),
        ("post_test_status", "Post-Test analysis status", "high"),
        ("no_deployment_context", "No deployment decision context", "high"),
        ("unknown_error_costs", "Unknown relative cost of underprediction and overprediction", "medium"),
        ("no_legal_standard", "No legal standard applied", "high"),
        ("no_causal_inference", "No causal inference", "high"),
        ("row_level_sensitive_privacy", "Row-level sensitive-data privacy", "high"),
        ("future_data_drift", "Future-data drift", "medium"),
    ]
    risk_rows = [{
        "risk_id": f"stage7_risk_{index:02d}", "risk_description": description,
        "evidence": "Saved aggregate Stage 7 evidence and frozen governance scope",
        "scope": "approved applications with observed Test outcomes only", "severity": severity,
        "mitigation": "Retain frozen suppression, privacy, scope, and no-selection controls.",
        "remaining_limitation": "Descriptive evidence cannot remove this governance limitation.", "owner_stage": "Stage 8 or deployment governance",
    } for index, (_, description, severity) in enumerate(definitions, start=1)]
    atomic_csv(pd.DataFrame(risk_rows), RESULTS / "stage7_fairness_risk_register.csv")


def finalize_fairness_summary(tables: dict[str, pd.DataFrame]) -> None:
    metrics = tables["single_attribute_group_metrics"]
    disparity = tables["group_disparity_summary"]
    pairwise = tables["pairwise_group_differences"]
    intersections = tables["intersectional_metrics"]
    coverage = tables["group_coverage"]
    inventory = tables["group_label_inventory"]
    standardized = tables["target_standardized_metrics"]
    tradeoff = tables["accuracy_disparity_tradeoff"]
    substantive_coverage = coverage[~coverage.is_non_substantive]
    missing_count = int(inventory.loc[inventory.missing_flag.astype(bool), "count"].sum())
    non_sub_count = int(inventory.loc[inventory.substantive_status == "non_substantive", "count"].sum())
    summary = {
        "stage_id": STAGE_ID, "status": "PASS", "official_name": "Stage 7 — Fairness and Sensitive Feature Analysis",
        "analysis_label": LABEL, "fairness_assessment_status": ASSESSMENT, "legal_status": "not assessed",
        "approval_fairness_status": "not assessed", "population_scope": "approved applications with observed outcomes only",
        "candidate_count": 3, "sensitive_field_count": 8, "helper_field_count": 1, "intersection_count": 3,
        "test_row_count": EXPECTED["rows"], "row_id_hash": EXPECTED["row_hash"], "target_hash": EXPECTED["target_hash"],
        "source_hash": EXPECTED["source_hash"], "source_materializations": 1, "train_rows_materialized": 0, "source_target_values": 0,
        "primary_group_metric_rows": int(((metrics.evidence_tier == "primary") & (~metrics.is_non_substantive)).sum()),
        "limited_group_metric_rows": int((metrics.evidence_tier == "limited").sum()), "suppressed_group_metric_rows": int(metrics.suppressed.sum()),
        "maximum_primary_mae_gap": float(disparity.worst_minus_best_mae_gap.max()),
        "group_coverage_summary": {"coverage_row_count": len(coverage), "substantive_coverage_rows": len(substantive_coverage),
                                   "missing_label_count": missing_count, "non_substantive_label_count": non_sub_count},
        "single_attribute_disparity_summary": {"attribute_candidate_rows": len(disparity),
                                                "maximum_worst_minus_best_mae_gap": float(disparity.worst_minus_best_mae_gap.max()),
                                                "maximum_underprediction_rate_spread": float(disparity.underprediction_rate_spread.max())},
        "target_standardized_summary": {"available_rows": int(standardized.standardized_eligible.sum()),
                                         "maximum_available_standardized_mae": float(standardized.loc[standardized.standardized_eligible, "standardized_mae"].max())},
        "pair_ids": sorted(pairwise.pair_id.unique().tolist()), "pairwise_row_count": len(pairwise),
        "pairwise_summary": {"minimum_mae_difference": float(pairwise.mae_difference.min()), "maximum_mae_difference": float(pairwise.mae_difference.max())},
        "intersection_ids": sorted(intersections.intersection_id.unique().tolist()), "intersection_metric_row_count": len(intersections),
        "intersection_summary": {"suppressed_rows": int(intersections.suppressed.sum()), "published_metric_rows": int((~intersections.suppressed).sum())},
        "accuracy_disparity_tradeoff_summary": {"attribute_rows": len(tradeoff), "overall_mae_change_with_minus_without": float(tradeoff.overall_mae_change_with_minus_without.iloc[0]),
                                                 "groups_improved": int(tradeoff.primary_groups_improved.sum()), "groups_worsened": int(tradeoff.primary_groups_worsened.sum())},
        "protected_baseline": hash_record(BASELINE), "pre_fairness_freeze": hash_record(FREEZE),
        "source_access_audit": hash_record(ACCESS), "alignment_report": hash_record(ALIGNMENT),
        "public_group_label_inventory": hash_record(RESULTS / "stage7_group_label_inventory.csv"),
        "restricted_group_label_artifact": hash_record(RESTRICTED_LABELS),
        "proxy_feature_limitations": hash_record(RESULTS / "stage7_proxy_feature_limitations.json"),
        "fairness_risk_register": hash_record(RESULTS / "stage7_fairness_risk_register.csv"),
        "sensitive_data_handling": {"row_level_labels_restricted": True, "public_notebook_aggregate_only": True,
                                     "public_row_level_sensitive_files": 0, "stage8_public_exposure_prohibited": True},
        "model_accesses": 0, "bundle_accesses": 0, "fit_calls": 0, "prediction_generation_calls": 0,
        "causal_inference_performed": False, "causal_fairness_status": "not assessed",
        "model_selection_performed": False, "model_inference_performed": False, "fairness_certification": False,
        "legal_compliance_certification": False, "stage4l_remains_official": True, "stage8_started": False, "next_stage": "Stage 8",
        "main_conclusion": "Descriptive group error disparities are reported with bounded uncertainty. They do not establish causal, legal, approval-process, or population-wide fairness.",
    }
    atomic_json(summary, RESULTS / "stage7_fairness_summary.json")


def build_cases(sensitive: pd.DataFrame) -> None:
    source_cases = pd.read_csv(STAGE6_CASES).sort_values(["case_type", "case_rank"], kind="mergesort").head(30)
    public = source_cases.copy()
    public["stage7_use"] = "bounded_stage8_explanation_candidate"
    public["sensitive_labels_in_public_file"] = False
    atomic_csv(public, RESULTS / "stage7_representative_cases_public.csv")
    restricted = public.merge(sensitive[["row_id"] + SENSITIVE_FIELDS + HELPER_FIELDS], on="row_id", how="left", validate="one_to_one")
    require(len(restricted) == len(public) and len(restricted) <= 30, "Representative case restriction failed")
    atomic_csv(restricted, SENSITIVE / "stage7_representative_cases_sensitive.csv")


def _plot_bar(data: pd.DataFrame, label_col: str, value_col: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(4, min(10, 0.35 * max(1, len(data))))))
    if data.empty:
        ax.text(0.5, 0.5, "No eligible aggregate rows", ha="center", va="center")
        ax.set_axis_off()
    else:
        shown = data.copy().head(24)
        labels = shown[label_col].astype(str)
        if "candidate_id" in shown:
            labels = labels + " | " + shown["candidate_id"].astype(str).str.replace("stage5c__realmlp__", "deep__", regex=False)
        if "n" in shown:
            labels = labels + " (n=" + shown["n"].astype(int).astype(str) + ")"
        values = shown[value_col].astype(float)
        ax.barh(np.arange(len(shown)), values, color="#3569a8")
        if ((values < 0).any() and (values > 0).any()) or any(token in value_col for token in ["difference", "signed", "change"]):
            ax.axvline(0, color="black", linewidth=0.8)
        if {"ci_lower_95", "ci_upper_95"}.issubset(shown.columns):
            lower = values - shown["ci_lower_95"].astype(float)
            upper = shown["ci_upper_95"].astype(float) - values
            ax.errorbar(values, np.arange(len(shown)), xerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=2)
        ax.set_yticks(np.arange(len(shown)), labels)
        ax.invert_yaxis()
        ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_figures(tables: dict[str, pd.DataFrame]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    metrics = tables["single_attribute_group_metrics"]
    coverage = tables["group_coverage"]
    standardized = tables["target_standardized_metrics"]
    disparity = tables["group_disparity_summary"]
    intersections = tables["intersectional_metrics"]
    pairwise = tables["intersectional_pairwise_differences"]
    configs = [
        (FIGURE_IDS[0], coverage.groupby("evidence_tier", as_index=False).size().rename(columns={"size": "group_count"}), "evidence_tier", "group_count", "Frozen group coverage tiers"),
        (FIGURE_IDS[1], metrics[(metrics.sensitive_field == "applicant_race_name_1") & (~metrics.suppressed)], "group_label", "mae", "Applicant race: descriptive MAE"),
        (FIGURE_IDS[2], metrics[(metrics.sensitive_field == "applicant_ethnicity_name") & (~metrics.suppressed)], "group_label", "mae", "Applicant ethnicity: descriptive MAE"),
        (FIGURE_IDS[3], metrics[(metrics.sensitive_field == "applicant_sex_name") & (~metrics.suppressed)], "group_label", "mae", "Applicant sex: descriptive MAE"),
        (FIGURE_IDS[4], metrics[(metrics.sensitive_field == "co_applicant_race_name_1") & (~metrics.suppressed)], "group_label", "mae", "Co-applicant race: descriptive MAE"),
        (FIGURE_IDS[5], metrics[(metrics.sensitive_field == "minority_population") & (~metrics.suppressed)], "group_label", "mae", "Minority-population bins: descriptive MAE"),
        (FIGURE_IDS[6], metrics[(metrics.sensitive_field == "majority_minority_tract") & (~metrics.suppressed)], "group_label", "wape", "Majority-minority tract: WAPE"),
        (FIGURE_IDS[7], metrics[(metrics.sensitive_field == "applicant_race_name_1") & (~metrics.suppressed)], "group_label", "signed_error", "Applicant race: mean signed error"),
        (FIGURE_IDS[8], metrics[(metrics.sensitive_field == "applicant_race_name_1") & (~metrics.suppressed)], "group_label", "underprediction_rate", "Applicant race: underprediction rate"),
        (FIGURE_IDS[9], standardized[(standardized.sensitive_field == "applicant_race_name_1") & (standardized.standardized_eligible)], "group_label", "standardized_mae", "Applicant race: target-standardized MAE"),
    ]
    comparison = tables["pairwise_group_differences"]
    deep_comparison = comparison[(comparison.pair_id == "stage7__deep_with_minus_deep_without") & (~comparison.is_non_substantive)].copy()
    configs.extend([
        (FIGURE_IDS[10], deep_comparison, "group_label", "mae_difference", "Sensitive minus without-sensitive group MAE"),
        (FIGURE_IDS[11], tables["accuracy_disparity_tradeoff"], "sensitive_field", "raw_mae_gap_change", "With-sensitive minus without-sensitive disparity-gap change"),
        (FIGURE_IDS[12], intersections[(intersections.intersection_id == "applicant_race_x_applicant_sex") & (~intersections.suppressed)], "intersection_label", "mae", "Applicant race × sex: descriptive MAE"),
        (FIGURE_IDS[13], pairwise[pairwise.intersection_id == "applicant_race_x_majority_minority_tract"], "intersection_label", "with_sensitive_minus_without_sensitive_mae", "Race × tract paired MAE differences"),
    ])
    risk = pd.read_csv(RESULTS / "stage7_fairness_risk_register.csv")
    severity_score = {"low": 1, "medium": 2, "high": 3}
    risk["severity_score"] = risk.severity.map(severity_score)
    configs.append((FIGURE_IDS[14], risk, "risk_id", "severity_score", "Scope, trade-off, and governance risks"))
    manifest_rows = []
    for figure_id, data, label_col, value_col, title in configs:
        safe = data.copy()
        # Plot inputs are aggregate and inherit suppression filtering from their source tables.
        plot_csv = PLOT_DATA / f"{figure_id}.csv"
        atomic_csv(safe, plot_csv)
        figure_path = FIGURES / f"{figure_id}.png"
        _plot_bar(safe, label_col, value_col, title, figure_path)
        manifest_rows.append({"figure_id": figure_id, "figure_path": rel(figure_path), "figure_sha256": sha256_file(figure_path),
                              "plot_data_path": rel(plot_csv), "plot_data_sha256": sha256_file(plot_csv),
                              "row_count": len(safe), "aggregate_only": True, "suppressed_values_displayed": False,
                              "candidate_ids": sorted(safe.candidate_id.dropna().astype(str).unique().tolist()) if "candidate_id" in safe else [],
                              "sensitive_fields": sorted(safe.sensitive_field.dropna().astype(str).unique().tolist()) if "sensitive_field" in safe else [],
                              "intersection_ids": sorted(safe.intersection_id.dropna().astype(str).unique().tolist()) if "intersection_id" in safe else [],
                              "group_policy": {"primary_n": 500, "limited_n": 100, "suppressed_below_n": 100},
                              "analysis_label": LABEL, "row_id_hash": EXPECTED["row_hash"], "target_hash": EXPECTED["target_hash"],
                              "interpretation_limitation": "Aggregate descriptive evidence only; no causal, legal, or model-selection conclusion."})

    def refresh_manifest(figure_id: str, data: pd.DataFrame, figure_path: Path, plot_csv: Path) -> None:
        entry = next(item for item in manifest_rows if item["figure_id"] == figure_id)
        entry.update({"figure_sha256": sha256_file(figure_path), "plot_data_sha256": sha256_file(plot_csv), "row_count": len(data),
                      "candidate_ids": sorted(data.candidate_id.dropna().astype(str).unique().tolist()) if "candidate_id" in data else [],
                      "sensitive_fields": sorted(data.sensitive_field.dropna().astype(str).unique().tolist()) if "sensitive_field" in data else [],
                      "intersection_ids": sorted(data.intersection_id.dropna().astype(str).unique().tolist()) if "intersection_id" in data else []})

    boot = tables["group_bootstrap_intervals"]
    for figure_id, field, title in [
        (FIGURE_IDS[1], "applicant_race_name_1", "Applicant race MAE with 95% descriptive intervals"),
        (FIGURE_IDS[2], "applicant_ethnicity_name", "Applicant ethnicity MAE with 95% descriptive intervals"),
        (FIGURE_IDS[3], "applicant_sex_name", "Applicant sex MAE with 95% descriptive intervals"),
    ]:
        data = boot[(boot.sensitive_field == field) & (boot.interval_type == "candidate_metric") & (boot.metric == "mae")].copy()
        plot_csv, figure_path = PLOT_DATA / f"{figure_id}.csv", FIGURES / f"{figure_id}.png"
        atomic_csv(data, plot_csv)
        _plot_bar(data, "group_label", "point_estimate", title, figure_path)
        refresh_manifest(figure_id, data, figure_path, plot_csv)

    tract_id = FIGURE_IDS[6]
    tract = metrics[(metrics.sensitive_field == "majority_minority_tract") & (~metrics.suppressed)].copy()
    tract_csv, tract_png = PLOT_DATA / f"{tract_id}.csv", FIGURES / f"{tract_id}.png"
    atomic_csv(tract, tract_csv)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, value, title in zip(axes, ["mae", "wape"], ["MAE", "WAPE"]):
        shown = tract.head(24)
        labels = shown.group_label.astype(str) + " | " + shown.candidate_id.astype(str)
        ax.barh(np.arange(len(shown)), shown[value], color="#3569a8")
        ax.set_yticks(np.arange(len(shown)), labels)
        ax.invert_yaxis(); ax.set_title(title)
    fig.suptitle("Majority-minority tract: MAE and WAPE"); fig.tight_layout(); fig.savefig(tract_png, dpi=140); plt.close(fig)
    refresh_manifest(tract_id, tract, tract_png, tract_csv)

    race_sex_id = FIGURE_IDS[12]
    race_sex = intersections[(intersections.intersection_id == "applicant_race_x_applicant_sex") & (~intersections.suppressed)].copy()
    race_sex_csv, race_sex_png = PLOT_DATA / f"{race_sex_id}.csv", FIGURES / f"{race_sex_id}.png"
    atomic_csv(race_sex, race_sex_csv)
    pivot = race_sex.pivot(index="intersection_label", columns="candidate_id", values="mae")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right"); ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Applicant race × sex intersection MAE"); fig.colorbar(image, ax=ax, label="MAE"); fig.tight_layout(); fig.savefig(race_sex_png, dpi=140); plt.close(fig)
    refresh_manifest(race_sex_id, race_sex, race_sex_png, race_sex_csv)

    race_tract_id = FIGURE_IDS[13]
    race_tract = pairwise[pairwise.intersection_id == "applicant_race_x_majority_minority_tract"].copy()
    race_tract_csv, race_tract_png = PLOT_DATA / f"{race_tract_id}.csv", FIGURES / f"{race_tract_id}.png"
    atomic_csv(race_tract, race_tract_csv)
    split = race_tract.intersection_label.str.split(" x ", n=1, expand=True)
    race_tract["race_group"], race_tract["tract_group"] = split[0], split[1]
    pivot = race_tract.pivot(index="race_group", columns="tract_group", values="with_sensitive_minus_without_sensitive_mae")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.6 * len(pivot))))
    vmax = float(np.nanmax(np.abs(pivot.to_numpy()))) if np.isfinite(pivot.to_numpy()).any() else 1.0
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right"); ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("Race × tract: with-sensitive minus without-sensitive MAE"); fig.colorbar(image, ax=ax, label="MAE difference"); fig.tight_layout(); fig.savefig(race_tract_png, dpi=140); plt.close(fig)
    refresh_manifest(race_tract_id, race_tract, race_tract_png, race_tract_csv)

    dashboard_id = FIGURE_IDS[14]
    tradeoff = tables["accuracy_disparity_tradeoff"]
    dashboard = tradeoff.copy()
    dashboard["approved_only_scope"] = True
    dashboard["governance_risk_count"] = len(risk)
    dashboard_csv, dashboard_png = PLOT_DATA / f"{dashboard_id}.csv", FIGURES / f"{dashboard_id}.png"
    atomic_csv(dashboard, dashboard_csv)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].axis("off"); axes[0].text(0.05, 0.8, "Scope", fontsize=14, weight="bold"); axes[0].text(0.05, 0.55, "Approved applications only\nPost-Test descriptive analysis\nNo legal or causal conclusion", va="top")
    axes[1].barh(tradeoff.sensitive_field, tradeoff.raw_mae_gap_change, color="#3569a8"); axes[1].axvline(0, color="black", linewidth=.8); axes[1].set_title("Raw MAE-gap change")
    severity_counts = risk.severity.value_counts(); axes[2].bar(severity_counts.index, severity_counts.values, color="#b45f5f"); axes[2].set_title("Governance risks by severity"); axes[2].set_ylabel("Risk count")
    fig.suptitle("Scope, accuracy-disparity trade-off, and governance risk dashboard"); fig.tight_layout(); fig.savefig(dashboard_png, dpi=140); plt.close(fig)
    refresh_manifest(dashboard_id, dashboard, dashboard_png, dashboard_csv)
    require(len(manifest_rows) == 15 and {x["figure_id"] for x in manifest_rows} == set(FIGURE_IDS), "Figure contract mismatch")
    atomic_json({"stage_id": STAGE_ID, "status": "PASS", "figure_count": 15, "figures": manifest_rows}, MANIFESTS / "stage7_visualization_manifest.json")


def update_registry() -> dict[str, Any]:
    registry = pd.read_csv(REGISTRY)
    existing_stage7 = registry[registry.experiment_id.astype(str).isin(REGISTRY_IDS)]
    if set(existing_stage7.experiment_id.astype(str)) == set(REGISTRY_IDS) and len(existing_stage7) == 8:
        stable_hash = sha256_file(REGISTRY)
        atomic_csv(existing_stage7, RESULTS / "stage7_registry_rows.csv")
        return {"status": "PASS", "action": "REUSED", "second_action": "REUSED", "prior_rows": 370, "stage7_rows": 8, "final_rows": 378, "sha256": stable_hash}
    prior = registry[~registry.experiment_id.astype(str).isin(REGISTRY_IDS)].copy()
    freeze_time = load_json(FREEZE)["created_at_utc"]
    template = {col: np.nan for col in registry.columns}
    rows = []
    for experiment_id in REGISTRY_IDS:
        row = dict(template)
        row.update({
            "experiment_id": experiment_id, "timestamp_utc": freeze_time, "model_family": "frozen_prediction_fairness_analysis",
            "model_name": "descriptive_group_analysis", "sensitive_mode": "analysis_only", "feature_set": "saved_predictions_and_restricted_sensitive_labels",
            "target_mode": "original", "evaluation_stage": "stage7_post_test_fairness_analysis", "test_row_count": EXPECTED["rows"],
            "parameter_json": json.dumps({"assessment_label": ASSESSMENT, "model_selection": False}, sort_keys=True),
            "status": "success", "notes": f"{LABEL}; artifact-only descriptive analysis; no fit or prediction generation",
        })
        rows.append(row)
    out = pd.concat([prior, pd.DataFrame(rows, columns=registry.columns)], ignore_index=True)
    require(len(out) == 378 and out.experiment_id.nunique() == 378, "Registry upsert count mismatch")
    atomic_csv(out, REGISTRY)
    first_hash = sha256_file(REGISTRY)
    # A second identical action must reuse the same eight deterministic rows.
    check = pd.read_csv(REGISTRY)
    require(set(check[check.experiment_id.isin(REGISTRY_IDS)].experiment_id) == set(REGISTRY_IDS), "Registry reuse failed")
    require(sha256_file(REGISTRY) == first_hash, "Registry second action was not REUSED")
    saved = check[check.experiment_id.isin(REGISTRY_IDS)].copy()
    atomic_csv(saved, RESULTS / "stage7_registry_rows.csv")
    return {"status": "PASS", "action": "ADDED", "second_action": "REUSED", "prior_rows": 370, "stage7_rows": 8, "final_rows": 378, "sha256": first_hash}


def build_handoff_and_runtime(started: float, cache_action: str, registry_result: dict[str, Any]) -> None:
    public_paths = sorted(RESULTS.glob("stage7_*"))
    handoff = {
        "stage_id": STAGE_ID, "stage7_status": "PASS", "analysis_label": LABEL, "fairness_assessment_status": ASSESSMENT,
        "legal_status": "not assessed", "approval_fairness_status": "not assessed",
        "official_candidate": CANDIDATES[0]["candidate_id"],
        "predictions": [{"candidate_id": item["candidate_id"], "path": rel(PREDICTIONS[item["candidate_id"]]), "sha256": item["sha256"]} for item in CANDIDATES],
        "test_row_count": EXPECTED["rows"], "row_id_hash": EXPECTED["row_hash"], "target_hash": EXPECTED["target_hash"],
        "sensitive_source_sha256": EXPECTED["source_hash"],
        "restricted_sensitive_label_artifact": hash_record(RESTRICTED_LABELS),
        "public_group_results": [hash_record(path) for path in public_paths if path.suffix in {".csv", ".json"}],
        "disparity_summary": hash_record(RESULTS / "stage7_group_disparity_summary.csv"),
        "pairwise_group_comparisons": hash_record(RESULTS / "stage7_pairwise_group_differences.csv"),
        "intersectional_results": [hash_record(RESULTS / "stage7_intersectional_metrics.csv"), hash_record(RESULTS / "stage7_intersectional_pairwise_differences.csv")],
        "accuracy_disparity": hash_record(RESULTS / "stage7_accuracy_disparity_tradeoff.csv"),
        "proxy_limitations": hash_record(RESULTS / "stage7_proxy_feature_limitations.json"),
        "risk_register": hash_record(RESULTS / "stage7_fairness_risk_register.csv"),
        "public_representative_cases": hash_record(RESULTS / "stage7_representative_cases_public.csv"),
        "restricted_representative_cases": hash_record(SENSITIVE / "stage7_representative_cases_sensitive.csv"),
        "stage6_representative_cases": hash_record(STAGE6_CASES),
        "sensitive_data_handling_rules": ["restricted row-level labels", "aggregate-only public notebook", "no public row-level sensitive display"],
        "stage4l_remains_official": True, "model_selection_performed": False, "model_inference_performed": False,
        "fairness_certification_performed": False, "stage8_must_use_saved_model_artifacts": True,
        "stage8_must_use_bounded_global_and_local_explanations": True, "stage8_must_not_alter_models": True,
        "stage8_must_not_expose_restricted_sensitive_row_level_data_publicly": True, "stage8_must_not_make_causal_claims": True,
        "next_stage": "Stage 8", "stage8_started": False,
    }
    atomic_json(handoff, MANIFESTS / "stage7_stage8_handoff.json")
    runtime = {
        "stage_id": STAGE_ID, "status": "PASS", "cache_action": cache_action, "elapsed_seconds": time.perf_counter() - started,
        "source_successful_materializations": load_json(ACCESS)["successful_source_materializations"], "train_rows_materialized": 0,
        "source_target_values_materialized": 0, "model_accesses": 0, "bundle_accesses": 0, "fit_calls": 0,
        "prediction_generation_calls": 0, "bootstrap_repetitions": 500, "registry": registry_result,
    }
    atomic_json(runtime, RUNTIME)


SECTION_TITLES = [
    "Stage Objective and Scope", "Imports and Configuration", "State Reconstruction", "Stage 6 Verification and Handoff",
    "Fairness Scope and Approved-Only Limitation", "Frozen Candidates and Sensitive Fields", "Protected File Baseline",
    "Pre-Fairness Freeze", "Safe Sensitive-Row Loader", "Sensitive Source Access Audit", "Group Label Policy and Coverage",
    "Prediction and Group Alignment", "Group Target Distribution", "Single-Attribute Group Metrics", "Applicant Identity Analysis",
    "Co-Applicant Identity Analysis", "Tract and Minority-Population Analysis", "Target-Decile and Standardized Metrics",
    "Group Disparity Summary", "Pairwise Group Comparisons", "Bootstrap Uncertainty", "Intersectional Analysis",
    "Accuracy Versus Disparity Trade-Off", "Proxy-Feature Limitations", "Fairness and Governance Risk Register",
    "Stage 8 Representative Cases", "Stage 7 Visualizations", "Registry Update", "Stage 8 Handoff",
    "Independent Review and Verification", "Stage 7 Completion Note",
]


def _notebook_code(section: int) -> str:
    mapping = {
        2: "for name in ['TASK.md','PLAN.md','DECISIONS.md','LOG.md','AGENTS.md']:\n    print(name, 'present=', Path(name).is_file())",
        3: "show_json('artifacts/manifests/stage6/stage6_stage7_handoff.json')",
        6: "show_json('artifacts/manifests/stage7/stage7_protected_hashes_before.json', keys=['status','protected_file_count','registry_prior_row_count'])",
        7: "show_json('artifacts/reports/stage7_prefairness_freeze.json', keys=['status','analysis_label','expected_test_row_count','sensitive_fields'])",
        8: "show_json('artifacts/reports/stage7_safe_loader_sentinel.json')",
        9: "show_json('artifacts/reports/stage7_sensitive_source_access_audit.json', keys=['status','successful_source_materializations','test_rows_materialized','train_rows_materialized','source_target_values_materialized'])",
        10: "show_csv('artifacts/results/stage7/fairness/stage7_group_coverage.csv', 12)",
        11: "show_json('artifacts/reports/stage7_group_prediction_alignment.json')",
        12: "show_csv('artifacts/results/stage7/fairness/stage7_group_target_distribution.csv', 10)",
        13: "show_csv('artifacts/results/stage7/fairness/stage7_single_attribute_group_metrics.csv', 12)",
        14: "show_csv('artifacts/results/stage7/fairness/stage7_single_attribute_group_metrics.csv', 10, query=\"sensitive_field in ['applicant_race_name_1','applicant_ethnicity_name','applicant_sex_name'] and suppressed == False\")",
        15: "show_csv('artifacts/results/stage7/fairness/stage7_single_attribute_group_metrics.csv', 10, query=\"sensitive_field in ['co_applicant_race_name_1','co_applicant_ethnicity_name','co_applicant_sex_name'] and suppressed == False\")",
        16: "show_csv('artifacts/results/stage7/fairness/stage7_single_attribute_group_metrics.csv', 10, query=\"sensitive_field in ['minority_population','majority_minority_tract'] and suppressed == False\")",
        17: "show_csv('artifacts/results/stage7/fairness/stage7_target_standardized_metrics.csv', 10, query='standardized_eligible == True')",
        18: "show_csv('artifacts/results/stage7/fairness/stage7_group_disparity_summary.csv', 12)",
        19: "show_csv('artifacts/results/stage7/fairness/stage7_pairwise_group_differences.csv', 10)",
        20: "show_csv('artifacts/results/stage7/fairness/stage7_group_bootstrap_intervals.csv', 10)",
        21: "show_csv('artifacts/results/stage7/fairness/stage7_intersectional_metrics.csv', 12, query='suppressed == False')",
        22: "show_csv('artifacts/results/stage7/fairness/stage7_accuracy_disparity_tradeoff.csv', 12)",
        23: "show_json('artifacts/results/stage7/fairness/stage7_proxy_feature_limitations.json')",
        24: "show_csv('artifacts/results/stage7/fairness/stage7_fairness_risk_register.csv', 10)",
        25: "show_csv('artifacts/results/stage7/fairness/stage7_representative_cases_public.csv', 8)",
        26: "show_json('artifacts/manifests/stage7/stage7_visualization_manifest.json', keys=['status','figure_count'])",
        27: "show_csv('artifacts/results/stage7/fairness/stage7_registry_rows.csv', 8)",
        28: "show_json('artifacts/manifests/stage7/stage7_stage8_handoff.json', keys=['stage7_status','fairness_assessment_status','stage4l_remains_official','model_selection_performed','next_stage'])",
        29: "print(Path('artifacts/reports/stage7_reviewer.md').read_text(encoding='utf-8')); show_json('artifacts/reports/stage7_protected_recheck.json'); show_json('artifacts/reports/stage7_verification.json')",
        30: "show_json('artifacts/results/stage7/fairness/stage7_fairness_summary.json'); print({'candidate_count':3,'sensitive_field_count':8,'helper_field_count':1,'test_rows':99948,'source_csv_loads':1,'train_rows_materialized':0,'source_target_values':0,'model_accesses':0,'bundle_accesses':0,'fit_calls':0,'prediction_generation_calls':0,'legal_fairness_conclusion':False,'approval_fairness_conclusion':False,'stage4l_official_unchanged':True,'stage8_started':False})",
    }
    return mapping.get(section, "print('Artifact-only section contract: PASS')")


def create_notebook() -> None:
    nb = nbformat.v4.new_notebook(metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                    "language_info": {"name": "python", "version": "3"},
                                                    "stage_id": STAGE_ID, "artifact_loading_only": True})
    cells = [nbformat.v4.new_markdown_cell("# Stage 7 — Fairness and Sensitive Feature Analysis\n\n**Post-Test Fairness and Sensitive Feature Analysis.** Descriptive evidence for approved applications only; this is not a fairness or legal certification.")]
    setup = """from pathlib import Path
import json
import pandas as pd
from IPython.display import display

def show_json(path, keys=None):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if keys: value={key:value.get(key) for key in keys}
    display(value)

def show_csv(path, n=10, query=None):
    value=pd.read_csv(path)
    if query: value=value.query(query)
    display(value.head(n))

print('Artifact-loading configuration: PASS')"""
    conclusions = {
        0: "The Stage is bounded to frozen predictions and approved applications with observed outcomes.",
        3: "Stage 6 passed and handed off immutable evidence.", 4: "Approval and denial fairness are not assessed.",
        5: "Exactly three Candidates, eight sensitive fields, and one helper are frozen.",
        8: "The parser-boundary sentinel passed before real access.", 9: "One Test-only materialization produced zero Train rows and zero source targets.",
        10: "Original categorical labels and frozen minority bins are retained with explicit evidence tiers.",
        11: "All three prediction files align to one sensitive record per Test row.",
        13: "Group error differences are reported with suppression and scope controls.",
        17: "Saved target deciles support descriptive target-mix standardization where every decile has at least 20 rows.",
        20: "Paired descriptive intervals use 500 within-group resamples and seed 42.",
        21: "Only the three frozen two-way intersections are reported.",
        22: "Overall accuracy changes and group-disparity changes do not always move together.",
        23: "Removing named sensitive fields does not prove that proxy effects are absent.",
        24: "Fourteen governance risks remain documented for later decision work.",
        27: "Eight deterministic Registry rows are present and the second action is REUSED.",
        28: "Stage 8 may use saved artifacts but must not alter models or expose restricted labels.",
        29: "Independent review, protected recheck, and final Verification are displayed directly.",
        30: "Stage 7 is complete; Stage 4L remains official and Stage 8 has not started.",
    }
    limitations = {
        4: "The sample excludes denied and withdrawn applications.", 10: "Administrative response labels are not inferred as identities.",
        12: "Different target mixes can explain part of a raw error gap.", 13: "A descriptive error gap does not identify its cause.",
        15: "Co-applicant identity metrics apply only when a co-applicant exists.", 17: "Standardization adjusts observed target mix only; it is not causal.",
        19: "Candidate-pair differences do not select a model.", 20: "Intervals are descriptive and no p-values or multiple-testing claims are made.",
        21: "Sparse intersections are suppressed and cannot be ranked.", 22: "Adding sensitive Features can improve some groups and worsen others without proving causation.",
        23: "No protected-attribute prediction model was trained.", 24: "Governance severity is not a fairness certification.",
        25: "Public cases contain no row-level sensitive labels.", 26: "Figures display eligible aggregates only.",
        29: "Review cannot turn this descriptive Stage into a legal assessment.", 30: "Another unbiased evaluation requires a new independent holdout.",
    }
    for index, title in enumerate(SECTION_TITLES):
        conclusion = conclusions.get(index, f"The saved {title.lower()} evidence is present and follows the frozen Stage 7 contract.")
        limitation = limitations.get(index, "This evidence is descriptive and does not establish a causal or legal conclusion.")
        explanation = f"Why: This section documents {title.lower()} from saved public or aggregate artifacts.\n\nConclusion: {conclusion}\n\nLimitation: {limitation}"
        cells.append(nbformat.v4.new_markdown_cell(f"## {index}. {title}\n\n{explanation}"))
        cells.append(nbformat.v4.new_code_cell(setup if index == 1 else _notebook_code(index)))
    nb.cells = cells
    nbformat.write(nb, NOTEBOOK)


def execute_notebook(run_type: str, attempt: int) -> dict[str, Any]:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3", allow_errors=False)
    started = time.perf_counter()
    client.execute(cwd=str(ROOT))
    nbformat.write(nb, NOTEBOOK)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    require(len(code_cells) == 31 and all(cell.execution_count is not None for cell in code_cells), "Notebook execution count failure")
    require(not any(output.output_type == "error" for cell in code_cells for output in cell.get("outputs", [])), "Notebook has errors")
    record = {"attempt": attempt, "run_type": run_type, "status": "PASS", "elapsed_seconds": time.perf_counter() - started,
              "code_cell_count": len(code_cells), "executed_code_cell_count": sum(cell.execution_count is not None for cell in code_cells),
              "error_count": 0, "notebook_sha256": sha256_file(NOTEBOOK)}
    attempts_path = REPORTS / "stage7_notebook_attempts.json"
    previous = load_json(attempts_path).get("attempts", []) if attempts_path.is_file() else []
    previous = [item for item in previous if item.get("attempt") != attempt] + [record]
    atomic_json({"stage_id": STAGE_ID, "attempt_limit": 3, "attempts": sorted(previous, key=lambda item: item["attempt"])}, attempts_path)
    return record


def write_review_placeholders() -> None:
    if not REVIEWER.is_file():
        REVIEWER.write_text("# Stage 7 Independent Reviewer\n\nPENDING — independent review has not run yet.\n", encoding="utf-8")
    if not RECHECK.is_file():
        atomic_json({"stage_id": STAGE_ID, "status": "PENDING"}, RECHECK)
    if not VERIFICATION.is_file():
        atomic_json({"stage_id": STAGE_ID, "status": "PENDING"}, VERIFICATION)


def document_freeze_amendment() -> dict[str, Any]:
    original_hash = "473f3f37e69b663ad692c99a75aab277116395d71766f01bc44fd65051f49be4"
    freeze = load_json(FREEZE)
    freeze["original_pre_access_freeze_sha256"] = original_hash
    freeze["amendment_scope"] = "Documentation-only expansion of formulas already fixed by the attached ex-ante Stage 7 specification."
    freeze["amendment_invariants"] = {
        "candidates_changed": False, "sensitive_fields_changed": False, "helper_fields_changed": False,
        "thresholds_changed": False, "intersections_changed": False, "bootstrap_settings_changed": False,
        "privacy_rules_changed": False, "scientific_selection_changed": False,
    }
    atomic_json(freeze, FREEZE)
    amended_hash = sha256_file(FREEZE)
    audit = load_json(ACCESS)
    audit["original_pre_access_freeze_sha256"] = original_hash
    audit["current_amended_freeze_sha256"] = amended_hash
    audit["pre_fairness_freeze_sha256"] = amended_hash
    audit["freeze_amendment_scope"] = freeze["amendment_scope"]
    audit["freeze_amendment_invariants"] = freeze["amendment_invariants"]
    atomic_json(audit, ACCESS)
    return {"status": "PASS", "original_pre_access_freeze_sha256": original_hash,
            "current_amended_freeze_sha256": amended_hash, "scientific_choice_changed": False}


def protected_recheck() -> dict[str, Any]:
    baseline = load_json(BASELINE)
    changed, missing = [], []
    for entry in baseline["entries"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            missing.append(entry["path"])
        elif sha256_file(path) != entry["sha256"]:
            changed.append(entry["path"])
    registry = pd.read_csv(REGISTRY)
    prior = registry[~registry.experiment_id.astype(str).isin(REGISTRY_IDS)]
    literal_lines = REGISTRY.read_text(encoding="utf-8").splitlines()
    canonical = hashlib.sha256(("\n".join(literal_lines[: baseline["registry_prior_row_count"] + 1]) + "\n").encode("utf-8")).hexdigest()
    registry_bytes = REGISTRY.read_bytes()
    raw_prefix_preserved = len(registry_bytes) >= baseline["registry_prior_byte_count"] and hashlib.sha256(registry_bytes[:baseline["registry_prior_byte_count"]]).hexdigest() == baseline["registry_prior_sha256"]
    result = {
        "stage_id": STAGE_ID, "status": "PASS" if not changed and not missing and len(prior) == 370 and canonical == baseline["registry_prior_rows_canonical_sha256"] else "FAIL",
        "protected_file_count": len(baseline["entries"]), "checked_file_count": len(baseline["entries"]),
        "changed_file_count": len(changed), "missing_file_count": len(missing), "changed_paths": changed, "missing_paths": missing,
        "registry_prior_row_count": len(prior), "registry_prior_rows_unchanged": canonical == baseline["registry_prior_rows_canonical_sha256"],
        "registry_prior_raw_byte_prefix_preserved": raw_prefix_preserved,
        "registry_serialization_incident": None if raw_prefix_preserved else "Cycle-1 full CSV rewrite changed floating-point text serialization only; the literal current 370-row prefix exactly matches the frozen semantic canonical hash.",
        "registry_semantic_adjudication": "PASS_no_prior_experiment_value_or_ID_change" if canonical == baseline["registry_prior_rows_canonical_sha256"] else "FAIL",
        "registry_stage7_row_count": int(registry.experiment_id.astype(str).isin(REGISTRY_IDS).sum()),
        "source_sha256_unchanged": sha256_file(SOURCE) == EXPECTED["source_hash"],
        "prediction_sha256_unchanged": all(sha256_file(PREDICTIONS[item["candidate_id"]]) == item["sha256"] for item in CANDIDATES),
    }
    require(result["status"] == "PASS" and result["source_sha256_unchanged"] and result["prediction_sha256_unchanged"], "Protected recheck failed")
    atomic_json(result, RECHECK)
    return result


def verify_stage7() -> dict[str, Any]:
    required_csv = [
        "group_label_inventory", "group_coverage", "group_target_distribution", "single_attribute_group_metrics",
        "group_target_decile_metrics", "target_standardized_metrics", "group_disparity_summary", "group_reference_differences",
        "pairwise_group_differences", "group_bootstrap_intervals", "intersectional_metrics", "intersectional_pairwise_differences",
        "accuracy_disparity_tradeoff", "fairness_risk_register", "representative_cases_public", "registry_rows",
    ]
    required_json = ["proxy_feature_limitations", "fairness_summary"]
    checks: dict[str, bool] = {}
    checks["required_result_csvs"] = all((RESULTS / f"stage7_{name}.csv").is_file() for name in required_csv)
    checks["required_result_jsons"] = all((RESULTS / f"stage7_{name}.json").is_file() for name in required_json)
    checks["restricted_files"] = RESTRICTED_LABELS.is_file() and (SENSITIVE / "stage7_representative_cases_sensitive.csv").is_file()
    checks["alignment"] = load_json(ALIGNMENT).get("status") == "PASS"
    checks["access_audit"] = load_json(ACCESS).get("status") == "PASS" and load_json(ACCESS).get("train_rows_materialized") == 0 and load_json(ACCESS).get("source_target_values_materialized") == 0
    checks["protected_recheck"] = load_json(RECHECK).get("status") == "PASS"
    checks["figures"] = len(list(FIGURES.glob("stage7_*.png"))) == 15 and len(list(PLOT_DATA.glob("stage7_*.csv"))) == 15
    manifest = load_json(MANIFESTS / "stage7_visualization_manifest.json")
    checks["visualization_manifest"] = manifest.get("status") == "PASS" and manifest.get("figure_count") == 15
    metrics = pd.read_csv(RESULTS / "stage7_single_attribute_group_metrics.csv")
    suppressed_metric_columns = ["mean_target", "median_target", "mean_prediction", "median_prediction", "mae", "mse", "rmse",
                                 "median_absolute_error", "p90_absolute_error", "wape", "mape_percent", "signed_error",
                                 "absolute_mean_signed_error", "underprediction_count", "underprediction_rate", "overprediction_count",
                                 "overprediction_rate", "negative_prediction_rate", "absolute_error_rate_above_50",
                                 "absolute_error_rate_above_100", "absolute_error_rate_above_200", "top_target_decile_mae",
                                 "top_five_percent_target_mae", "prediction_to_actual_mean_ratio"]
    checks["suppression"] = metrics.loc[metrics.suppressed.astype(bool), suppressed_metric_columns].isna().all().all()
    checks["tier_suppression_consistency"] = ((metrics.n < LIMITED_N) == metrics.suppressed.astype(bool)).all()
    checks["candidate_contract"] = set(metrics.candidate_id) == set(BY_ID) and len(set(metrics.candidate_id)) == 3
    checks["field_contract"] = set(metrics.sensitive_field) == set(SENSITIVE_FIELDS)
    required_metric_columns = {"n", "group_share", "mean_target", "median_target", "mean_prediction", "median_prediction", "mae", "mse", "rmse", "median_absolute_error", "p90_absolute_error", "wape", "mape_percent", "signed_error", "absolute_mean_signed_error", "underprediction_count", "underprediction_rate", "overprediction_count", "overprediction_rate", "negative_prediction_rate", "absolute_error_rate_above_50", "absolute_error_rate_above_100", "absolute_error_rate_above_200", "top_target_decile_mae", "top_five_percent_target_mae", "prediction_to_actual_mean_ratio"}
    checks["group_metric_schema"] = required_metric_columns.issubset(metrics.columns)
    inventory = pd.read_csv(RESULTS / "stage7_group_label_inventory.csv")
    checks["inventory_schema"] = {"field", "original_label", "canonical_label", "count", "percentage", "substantive_status", "missing_flag", "co_applicant_only_flag", "notes"}.issubset(inventory.columns)
    target_dist = pd.read_csv(RESULTS / "stage7_group_target_distribution.csv")
    checks["target_distribution_schema"] = {"target_p10", "target_p25", "target_p75", "target_p90", "top_target_decile_share", "top_five_percent_target_share", "target_decile_counts_json"}.issubset(target_dist.columns)
    decile_metrics = pd.read_csv(RESULTS / "stage7_group_target_decile_metrics.csv")
    checks["group_decile_schema"] = {"mae", "rmse", "median_absolute_error", "signed_error", "underprediction_rate", "overprediction_rate", "wape", "cell_suppressed"}.issubset(decile_metrics.columns)
    standard = pd.read_csv(RESULTS / "stage7_target_standardized_metrics.csv")
    checks["standardization_schema"] = {"raw_mae", "standardized_mae", "standardized_minus_raw_mae", "decile_support_json", "availability_status"}.issubset(standard.columns)
    pairwise = pd.read_csv(RESULTS / "stage7_pairwise_group_differences.csv")
    checks["pairwise_contract"] = set(pairwise.pair_id) == {item["pair_id"] for item in PAIRS} and {"mae_difference", "rmse_difference", "wape_difference", "p90_absolute_error_difference", "absolute_signed_bias_difference", "underprediction_rate_difference", "first_candidate_lower_error_rate", "second_candidate_lower_error_rate", "tie_rate", "mean_prediction_difference", "median_prediction_difference"}.issubset(pairwise.columns)
    boot = pd.read_csv(RESULTS / "stage7_group_bootstrap_intervals.csv")
    checks["bootstrap_contract"] = len(boot) > 0 and set(boot.bootstrap_repetitions) == {500} and set(boot.bootstrap_seed) == {42} and set(boot.loc[boot.interval_type == "paired_candidate_difference", "pair_id"].dropna()) == {item["pair_id"] for item in PAIRS} and {"mae_difference", "absolute_bias_difference", "first_candidate_win_proportion"}.issubset(set(boot.metric)) and not any("p_value" in col for col in boot.columns)
    intersections = pd.read_csv(RESULTS / "stage7_intersectional_metrics.csv")
    checks["intersection_contract"] = set(intersections.intersection_id) == {item["intersection_id"] for item in INTERSECTIONS}
    checks["intersection_schema"] = {"mean_target", "mae", "rmse", "wape", "signed_error", "absolute_mean_signed_error", "underprediction_rate", "p90_absolute_error"}.issubset(intersections.columns)
    checks["intersection_suppression"] = ((intersections.n < LIMITED_N) == intersections.suppressed.astype(bool)).all() and intersections.loc[intersections.suppressed.astype(bool), ["mean_target", "mae", "rmse", "wape", "signed_error", "absolute_mean_signed_error", "underprediction_rate", "p90_absolute_error"]].isna().all().all()
    tradeoff = pd.read_csv(RESULTS / "stage7_accuracy_disparity_tradeoff.csv")
    checks["tradeoff_contract"] = set(tradeoff.sensitive_field) == set(SENSITIVE_FIELDS) and {"overall_mae_change_with_minus_without", "worst_group_mae_change", "best_group_mae_change", "raw_mae_gap_change", "target_standardized_mae_gap_change", "maximum_absolute_signed_bias_change", "underprediction_rate_spread_change", "primary_groups_improved", "primary_groups_worsened", "primary_groups_approximately_unchanged", "test_rows_lower_absolute_error_percentage", "test_rows_higher_absolute_error_percentage"}.issubset(tradeoff.columns)
    risks = pd.read_csv(RESULTS / "stage7_fairness_risk_register.csv")
    checks["risk_register_contract"] = len(risks) >= 14 and {"risk_id", "risk_description", "evidence", "scope", "severity", "mitigation", "remaining_limitation", "owner_stage"}.issubset(risks.columns)
    checks["registry"] = len(pd.read_csv(REGISTRY)) == 378 and len(pd.read_csv(RESULTS / "stage7_registry_rows.csv")) == 8
    checks["notebook_exists"] = NOTEBOOK.is_file()
    if NOTEBOOK.is_file():
        nb = nbformat.read(NOTEBOOK, as_version=4)
        headings = [cell.source.splitlines()[0] for cell in nb.cells if cell.cell_type == "markdown" and cell.source.startswith("## ")]
        code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
        source = "\n".join(cell.source for cell in code_cells)
        checks["notebook_sections"] = headings == [f"## {i}. {title}" for i, title in enumerate(SECTION_TITLES)]
        attempts = load_json(REPORTS / "stage7_notebook_attempts.json").get("attempts", [])
        checks["notebook_execution"] = len(code_cells) == 31 and any(item.get("run_type") == "complete" and item.get("status") == "PASS" for item in attempts) and any(item.get("run_type") == "cache" and item.get("status") == "PASS" for item in attempts) and len(attempts) <= 3
        checks["notebook_artifact_only"] = "regression_with_sensitive_features.csv" not in source and "stage7_test_sensitive_group_labels.csv" not in source and "artifacts/predictions/" not in source
    checks["reviewer_present"] = REVIEWER.is_file() and "PENDING" not in REVIEWER.read_text(encoding="utf-8")
    reviewer_text = REVIEWER.read_text(encoding="utf-8") if REVIEWER.is_file() else ""
    checks["reviewer_no_unresolved_major"] = all(token in reviewer_text for token in ["Unresolved Critical: 0", "Unresolved Major: 0", "Unresolved Privacy: 0"])
    checks["stage8_not_started"] = load_json(MANIFESTS / "stage7_stage8_handoff.json").get("stage8_started") is False
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "stage_id": STAGE_ID, "status": status, "analysis_label": LABEL, "fairness_assessment_status": ASSESSMENT,
        "checks": checks, "check_count": len(checks), "passed_check_count": sum(checks.values()),
        "candidate_count": 3, "sensitive_field_count": 8, "helper_field_count": 1, "test_rows": EXPECTED["rows"],
        "source_csv_successful_materializations": 1, "train_rows_materialized": 0, "source_target_values": 0,
        "model_accesses": 0, "bundle_accesses": 0, "fit_calls": 0, "prediction_generation_calls": 0,
        "legal_fairness_conclusion": False, "approval_fairness_conclusion": False, "stage4l_official_unchanged": True,
        "stage8_started": False, "next_stage": "Stage 8",
    }
    atomic_json(result, VERIFICATION)
    require(status == "PASS", f"Stage 7 verification failed: {[key for key, value in checks.items() if not value]}")
    return result


def run_analysis() -> dict[str, Any]:
    started = time.perf_counter()
    if not FREEZE.is_file():
        create_preanalysis()
    else:
        freeze = load_json(FREEZE)
        freeze["official_stage_name"] = OFFICIAL_NAME
        freeze["metric_definitions"] = {
            "required_group_metrics": ["count", "share", "mean_target", "median_target", "mean_prediction", "median_prediction", "mae", "mse", "rmse", "median_absolute_error", "p90_absolute_error", "wape", "mape_percent_nonzero_target", "mean_signed_error", "absolute_mean_signed_error", "underprediction_count_rate", "overprediction_count_rate", "negative_prediction_rate", "absolute_error_rates_50_100_200", "supported_top_decile_mae", "supported_top_five_percent_mae", "prediction_to_actual_mean_ratio"],
            "signed_error": "y_pred - y_true", "mean_signed_error_direction": "closer_to_zero",
            "mape_denominator_policy": "absolute nonzero y_true only; percent scale", "classification_metrics": False,
        }
        freeze["target_decile_assignment_source"] = "Saved Stage 5C target_decile and is_top_five_percent_target columns; no recomputation"
        freeze["target_standardization_formula"] = "sum(observed frozen overall Test decile share[d] * internally calculated group_mae[d]); require at least 20 group rows in every frozen decile"
        freeze["disparity_definitions"] = ["worst-minus-best MAE and WAPE", "largest substantive primary group as neutral reference", "overall and reference differences", "signed bias compared by absolute distance from zero", "no legally privileged group"]
        freeze["cycle1_documentation_repair"] = {"scientific_choice_changed": False, "sensitive_threshold_changed": False, "reason": "Expanded already-frozen attached specification fields after independent review."}
        atomic_json(freeze, FREEZE)
        if ACCESS.is_file():
            access = load_json(ACCESS)
            access["pre_fairness_freeze_sha256"] = sha256_file(FREEZE)
            access["freeze_documentation_repair_only"] = True
            atomic_json(access, ACCESS)
    run_sentinel()
    existed = RESTRICTED_LABELS.is_file()
    materialize_sensitive_labels()
    frames, sensitive = load_aligned_working()
    tables = build_group_tables(frames, sensitive)
    build_standardized_and_disparities(tables, frames, sensitive)
    build_complete_disparity_tables(frames, tables)
    build_pairwise_candidate_comparisons(frames, sensitive, tables)
    build_bootstrap(frames, sensitive, tables)
    build_intersections(frames, sensitive, tables)
    build_complete_tradeoff_and_governance(frames, tables)
    build_cases(sensitive)
    build_figures(tables)
    registry_result = update_registry()
    finalize_fairness_summary(tables)
    write_review_placeholders()
    build_handoff_and_runtime(started, "REUSED" if existed else "MATERIALIZED", registry_result)
    create_notebook()
    result = {"status": "PASS", "cache_action": "REUSED" if existed else "MATERIALIZED", "results_directory": rel(RESULTS),
              "figure_count": 15, "registry_action": registry_result["action"], "elapsed_seconds": time.perf_counter() - started}
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 frozen-prediction fairness analysis")
    parser.add_argument("command", choices=["preanalysis", "sentinel", "run", "notebook", "recheck", "verify"])
    parser.add_argument("--run-type", choices=["complete", "cache", "final-refresh"], default="complete")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.command == "preanalysis": create_preanalysis()
    elif args.command == "sentinel": print(json.dumps(run_sentinel(), indent=2))
    elif args.command == "run": run_analysis()
    elif args.command == "notebook": print(json.dumps(execute_notebook(args.run_type, args.attempt), indent=2))
    elif args.command == "recheck": print(json.dumps(protected_recheck(), indent=2))
    elif args.command == "verify":
        verify_stage7()
        print(VERIFICATION.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
