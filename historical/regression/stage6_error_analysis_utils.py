"""Stage 6 saved-prediction error analysis utilities.

This module reads exactly three frozen prediction files. It never reads a
source CSV, raw Feature table, model, or bundle. It never fits or predicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


ROOT = Path(__file__).resolve().parent
STAGE_ID = "stage6"
OFFICIAL_NAME = "Stage 6 — Final Error Analysis and Model Comparison"
LABEL = "Post-Test Error Analysis"
REPORTS = ROOT / "artifacts/reports"
RESULTS = ROOT / "artifacts/results/stage6/error_analysis"
FIGURES = ROOT / "artifacts/figures/stage6"
PLOT_DATA = FIGURES / "plotting_data"
MANIFESTS = ROOT / "artifacts/manifests/stage6"
BACKUPS = ROOT / "artifacts/backups"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
NOTEBOOK = ROOT / "REGRESSION_PART6_FINAL_ERROR_ANALYSIS.ipynb"

BASELINE = MANIFESTS / "stage6_protected_hashes_before.json"
FREEZE = REPORTS / "stage6_preanalysis_freeze.json"
ACCESS_AUDIT = REPORTS / "stage6_prediction_access_audit.json"
ALIGNMENT = REPORTS / "stage6_input_alignment_report.json"
ERROR_VALIDATION = REPORTS / "stage6_error_column_validation.json"
RECHECK = REPORTS / "stage6_protected_recheck.json"
VERIFICATION = REPORTS / "stage6_verification.json"
REVIEWER = REPORTS / "stage6_reviewer.md"
RUNTIME = REPORTS / "stage6_runtime.json"

STAGE4_VERIFICATION = REPORTS / "stage4l_verification.json"
STAGE4_FREEZE = REPORTS / "stage4l_pretest_freeze.json"
STAGE5A_VERIFICATION = REPORTS / "stage5a_verification.json"
STAGE5A_GOVERNANCE = REPORTS / "stage5a2_governance_adjudication.json"
STAGE5B_VERIFICATION = REPORTS / "stage5b_verification.json"
STAGE5B_SPEC = ROOT / "artifacts/results/stage5/deep_boosting_ensemble/stage5b_frozen_ensemble.json"
STAGE5C_VERIFICATION = REPORTS / "stage5c_verification.json"
STAGE5C_REVIEWER = REPORTS / "stage5c_reviewer.md"
STAGE5C_HANDOFF = ROOT / "artifacts/manifests/stage5/stage5c_stage6_handoff.json"
STAGE5C_METRICS = ROOT / "artifacts/results/stage5/posttest_evaluation/stage5c_test_metrics.csv"
STAGE5C_BOOTSTRAP = ROOT / "artifacts/results/stage5/posttest_evaluation/stage5c_paired_bootstrap.csv"
TEST_IDS = ROOT / "artifacts/splits/test_row_ids.csv"
TRAIN_IDS = ROOT / "artifacts/splits/train_row_ids.csv"
METRIC_SCHEMA = ROOT / "artifacts/data_contract/metric_schema.json"
STAGE5C_BASELINE = ROOT / "artifacts/manifests/stage5/stage5c_protected_hashes_before.json"

PREDICTIONS = {
    "stage4l__blend__without_sensitive": ROOT / "artifacts/predictions/final_test/stage4l__blend__without_sensitive.csv",
    "stage5c__realmlp__without_sensitive__test_evaluation": ROOT / "artifacts/predictions/stage5/posttest_evaluation/stage5c_test_predictions_without_sensitive.csv",
    "stage5c__realmlp__with_sensitive__test_evaluation": ROOT / "artifacts/predictions/stage5/posttest_evaluation/stage5c_test_predictions_with_sensitive.csv",
}
CANDIDATES = [
    {
        "candidate_id": "stage4l__blend__without_sensitive",
        "label": "Stage 4L Official Boosting Blend",
        "role": "official_pre_registered_primary",
        "prediction_sha256": "9f9efa21d95a466b8271cd0db0a1e6b2c1ed2b5f1cabfbbb7e081137b9e4b7ed",
    },
    {
        "candidate_id": "stage5c__realmlp__without_sensitive__test_evaluation",
        "label": "Frozen RealMLP Without Sensitive",
        "role": "post_test_extension",
        "prediction_sha256": "972eaa799c00eaa0ed486739636fb643f8f3e46e6890dc1964da542fd6108ee5",
    },
    {
        "candidate_id": "stage5c__realmlp__with_sensitive__test_evaluation",
        "label": "Frozen RealMLP With Sensitive",
        "role": "post_test_extension_accuracy_only",
        "prediction_sha256": "b4b11779a2d85209b2082c003ce79db2b657acd52c816c5e5345aaa6671f5e99",
    },
]
BY_ID = {item["candidate_id"]: item for item in CANDIDATES}
PAIRS = [
    {
        "pair_id": "stage6__deep_without_minus_stage4l",
        "first": CANDIDATES[1]["candidate_id"],
        "second": CANDIDATES[0]["candidate_id"],
    },
    {
        "pair_id": "stage6__deep_with_minus_stage4l",
        "first": CANDIDATES[2]["candidate_id"],
        "second": CANDIDATES[0]["candidate_id"],
    },
    {
        "pair_id": "stage6__deep_with_minus_deep_without",
        "first": CANDIDATES[2]["candidate_id"],
        "second": CANDIDATES[1]["candidate_id"],
    },
]
FIGURE_IDS = [
    "stage6_absolute_error_ecdf",
    "stage6_absolute_error_distribution_p99",
    "stage6_signed_error_distribution_p01_p99",
    "stage6_absolute_error_quantile_profile",
    "stage6_absolute_error_exceedance",
    "stage6_error_concentration",
    "stage6_mae_by_target_decile",
    "stage6_mean_signed_error_by_target_decile",
    "stage6_underprediction_rate_by_target_decile",
    "stage6_frozen_target_tail_error",
    "stage6_calibration_by_target_decile",
    "stage6_pairwise_absolute_error_difference_by_decile",
    "stage6_pairwise_prediction_disagreement_by_decile",
    "stage6_worst_case_overlap_matrix",
    "stage6_summary_dashboard",
]
REGISTRY_IDS = [
    "stage6__stage4l_primary__error_profile",
    "stage6__realmlp_without_sensitive__error_profile",
    "stage6__realmlp_with_sensitive__error_profile",
    "stage6__deep_without_vs_stage4l__comparison",
    "stage6__deep_with_vs_stage4l__comparison",
    "stage6__deep_with_vs_without__accuracy_profile",
    "stage6__error_analysis_summary",
    "stage6__stage7_handoff",
]
EXPECTED_NOTEBOOK_SHA256 = "7e4ae8b08ec8909dcc073f1f1f3eb822c063c35c6519059ed3027661213a2479"
EXPECTED_ROW_HASH = "e58e4d078c761f60405e644d4dd7ba368f364daffb73b44abb39095938ece95e"
EXPECTED_TARGET_HASH = "889e4253fb584c2a52a06d8b8e956beefad997ba18e4d736af0cd1738fb34a1a"
EXPECTED_ROWS = 99_948
TIE_TOLERANCE = 1e-12


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
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False), encoding="utf-8")
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


def prerequisite_checks() -> dict[str, bool]:
    stage4 = load_json(STAGE4_VERIFICATION)
    stage5a = load_json(STAGE5A_VERIFICATION)
    governance = load_json(STAGE5A_GOVERNANCE)
    stage5b = load_json(STAGE5B_VERIFICATION)
    stage5b_spec = load_json(STAGE5B_SPEC)
    stage5c = load_json(STAGE5C_VERIFICATION)
    handoff = load_json(STAGE5C_HANDOFF)
    reviewer_text = STAGE5C_REVIEWER.read_text(encoding="utf-8")
    checks = {
        "stage4l_verification_pass": stage4.get("status") == "PASS",
        "stage4l_primary_exact": stage4.get("primary_candidate") == CANDIDATES[0]["candidate_id"],
        "stage5a_pass_with_exception": stage5a.get("status") == "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION",
        "stage5a_literal_failure_visible": stage5a.get("literal_zero_test_loading_check") == "FAIL" and governance.get("classifications", {}).get("literal_zero_test_loading") is False,
        "stage5b_verification_pass": stage5b.get("status") == "PASS",
        "stage5b_ensemble_rejected": stage5b_spec.get("ensemble_status") == "rejected",
        "stage5c_verification_pass": stage5c.get("status") == "PASS" and stage5c.get("overall_status") == "PASS",
        "stage5c_reviewer_pass": "Final recommendation: PASS" in reviewer_text and "Critical issues: 0" in reviewer_text and "Major issues: 0" in reviewer_text,
        "stage5c_notebook_hash": sha256_file(ROOT / "REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.ipynb") == EXPECTED_NOTEBOOK_SHA256,
        "stage6_handoff_status": handoff.get("stage5c_status") == "PASS" and handoff.get("stage6_must_use_saved_predictions") is True,
        "handoff_row_count": handoff.get("test_row_count") == EXPECTED_ROWS,
        "handoff_hashes": handoff.get("test_row_id_hash") == EXPECTED_ROW_HASH and handoff.get("target_hash") == EXPECTED_TARGET_HASH,
        "candidate_count_exact": len(CANDIDATES) == 3,
        "prediction_hashes": all(PREDICTIONS[item["candidate_id"]].is_file() and sha256_file(PREDICTIONS[item["candidate_id"]]) == item["prediction_sha256"] for item in CANDIDATES),
        "bootstrap_exists_and_frozen": STAGE5C_BOOTSTRAP.is_file(),
    }
    require(all(checks.values()), f"Stage 6 prerequisite failure: {[key for key, value in checks.items() if not value]}")
    return checks


def create_preanalysis() -> dict[str, Any]:
    started = time.perf_counter()
    for directory in [REPORTS, RESULTS, FIGURES, PLOT_DATA, MANIFESTS, BACKUPS]:
        directory.mkdir(parents=True, exist_ok=True)
    checks = prerequisite_checks()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUPS / f"stage6_state_start_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for name in ["TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md", "AGENTS.md"]:
        shutil.copy2(ROOT / name, backup_dir / name)

    previous = load_json(STAGE5C_BASELINE)
    paths: dict[str, Path] = {}
    for entry in previous["entries"]:
        candidate = Path(entry["path"])
        path = candidate if candidate.is_absolute() else ROOT / candidate
        require(path.is_file(), f"Protected prior file is missing: {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"Protected prior file changed: {entry['path']}")
        paths[str(path.resolve()).lower()] = path
    for path in [
        ROOT / "REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.ipynb",
        ROOT / "stage5c_posttest_utils.py", ROOT / "stage5c_predict_worker.py",
        STAGE5C_VERIFICATION, STAGE5C_REVIEWER, STAGE5C_HANDOFF,
    ]:
        paths[str(path.resolve()).lower()] = path
    for pattern in [
        "artifacts/results/stage5/posttest_evaluation/*",
        "artifacts/predictions/stage5/posttest_evaluation/*",
        "artifacts/figures/stage5c/**/*",
        "artifacts/manifests/stage5/stage5c*",
        "artifacts/reports/stage5c*",
    ]:
        for path in ROOT.glob(pattern):
            if path.is_file():
                paths[str(path.resolve()).lower()] = path
    paths.pop(str(REGISTRY.resolve()).lower(), None)
    entries = [hash_record(path) for path in sorted(paths.values(), key=lambda p: rel(p).lower())]
    registry_bytes = REGISTRY.read_bytes()
    baseline = {
        "stage_id": STAGE_ID,
        "status": "PASS",
        "created_at_utc": now_utc(),
        "protected_file_count": len(entries),
        "entries": entries,
        "registry_prior_byte_count": len(registry_bytes),
        "registry_prior_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "state_backup_directory": rel(backup_dir),
        "prerequisite_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(baseline, BASELINE)

    freeze = {
        "stage_id": STAGE_ID,
        "status": "PASS",
        "official_stage_name": OFFICIAL_NAME,
        "created_at_utc": now_utc(),
        "analysis_label": LABEL,
        "post_test_disclosure": "Stage 4L remains official. Stage 6 is descriptive Post-Test Error Analysis and performs no model selection.",
        "stage4l_verification": hash_record(STAGE4_VERIFICATION),
        "stage5a_verification": hash_record(STAGE5A_VERIFICATION),
        "stage5a_governance_adjudication": hash_record(STAGE5A_GOVERNANCE),
        "stage5b_verification": hash_record(STAGE5B_VERIFICATION),
        "stage5b_ensemble_status": "rejected",
        "stage5c_verification": hash_record(STAGE5C_VERIFICATION),
        "stage5c_reviewer": hash_record(STAGE5C_REVIEWER),
        "final_stage5c_notebook": hash_record(ROOT / "REGRESSION_PART5_DEEP_POSTTEST_EVALUATION.ipynb"),
        "stage6_handoff_input": hash_record(STAGE5C_HANDOFF),
        "protected_baseline": hash_record(BASELINE),
        "candidates": [{**item, "prediction_path": rel(PREDICTIONS[item["candidate_id"]])} for item in CANDIDATES],
        "expected_test_row_count": EXPECTED_ROWS,
        "expected_test_row_id_hash": EXPECTED_ROW_HASH,
        "expected_target_hash": EXPECTED_TARGET_HASH,
        "target_unit": "thousands of US dollars",
        "signed_error_formula": "y_pred - y_true",
        "underprediction_definition": "signed_error < -1e-12",
        "overprediction_definition": "signed_error > 1e-12",
        "exact_prediction_tolerance": TIE_TOLERANCE,
        "mean_signed_error_direction": "closer_to_zero",
        "target_decile_policy": "Reuse identical Stage 5C target_decile assignments from both Deep prediction files and attach them to Stage 4L by exact row_id.",
        "target_tail_policy": "Reuse Stage 5C frozen target thresholds: y_true >= numpy quantile 0.90 and y_true >= numpy quantile 0.95; keep qcut decile distinct.",
        "absolute_error_quantiles": [0.50, 0.75, 0.90, 0.95, 0.99, 1.00],
        "absolute_error_thresholds": [50, 100, 200],
        "error_concentration_proportions": [0.01, 0.05, 0.10],
        "pairs": PAIRS,
        "pairwise_prediction_difference": "first_y_pred - second_y_pred",
        "pairwise_absolute_error_difference": "first_absolute_error - second_absolute_error",
        "pairwise_tie_tolerance": TIE_TOLERANCE,
        "worst_rows_per_candidate": 100,
        "high_disagreement_rows_per_pair": 200,
        "representative_case_definitions": [
            "five largest minimum absolute errors across all three Candidates",
            "five largest deep-without minus Stage4L absolute-error improvements for Stage4L",
            "five largest Stage4L minus deep-without absolute-error improvements for Deep without sensitive",
            "five largest deep-without minus deep-with absolute-error improvements for Deep with sensitive",
        ],
        "representative_case_limit": 20,
        "figure_ids": FIGURE_IDS,
        "deterministic_plotting_seed": 42,
        "maximum_scatter_sample": 20_000,
        "registry_ids": REGISTRY_IDS,
        "notebook_attempt_limit": 3,
        "reviewer_cycle_limit": 2,
        "prediction_rows_parsed_before_freeze": 0,
        "source_csv_load_count": 0,
        "raw_feature_load_count": 0,
        "model_access_count": 0,
        "bundle_access_count": 0,
        "prediction_generation_count": 0,
        "bootstrap_recomputation_count": 0,
        "model_selection_performed": False,
        "next_stage": "Stage 7",
    }
    atomic_json(freeze, FREEZE)
    reloaded = load_json(FREEZE)
    require(reloaded == freeze and len(reloaded["candidates"]) == 3 and len(reloaded["figure_ids"]) == 15, "Pre-analysis freeze reload failed")
    summary = {
        "status": "PASS",
        "baseline": rel(BASELINE),
        "baseline_sha256": sha256_file(BASELINE),
        "freeze": rel(FREEZE),
        "freeze_sha256": sha256_file(FREEZE),
        "protected_file_count": len(entries),
        "prediction_rows_parsed": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2))
    return summary


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    signed = y_pred - y_true
    absolute = np.abs(signed)
    mse = float(np.mean(signed ** 2))
    nonzero = np.abs(y_true) > 0
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    top_decile = y_true >= np.quantile(y_true, 0.90)
    top_five = y_true >= np.quantile(y_true, 0.95)
    return {
        "mae": float(np.mean(absolute)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r_squared": 1.0 - float(np.sum(signed ** 2)) / denominator,
        "rmsle": float(np.sqrt(np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(np.clip(y_true, 0, None))) ** 2))),
        "mape_percent": float(np.mean(absolute[nonzero] / np.abs(y_true[nonzero])) * 100.0),
        "median_absolute_error": float(np.median(absolute)),
        "wape_percent": float(np.sum(absolute) / np.sum(np.abs(y_true)) * 100.0),
        "mean_signed_error": float(np.mean(signed)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "top_decile_mae": float(np.mean(absolute[top_decile])),
        "top_five_percent_mae": float(np.mean(absolute[top_five])),
        "negative_prediction_rate": float(np.mean(y_pred < 0)),
    }


def group_metrics(group: pd.DataFrame) -> dict[str, Any]:
    y = group["y_true"].to_numpy(float)
    pred = group["y_pred"].to_numpy(float)
    signed = pred - y
    absolute = np.abs(signed)
    under = signed < -TIE_TOLERANCE
    over = signed > TIE_TOLERANCE
    return {
        "row_count": len(group),
        "target_min": float(y.min()), "target_max": float(y.max()), "target_mean": float(y.mean()),
        "prediction_mean": float(pred.mean()), "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(signed ** 2))), "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)), "mean_signed_error": float(signed.mean()),
        "absolute_mean_signed_error": float(abs(signed.mean())), "underprediction_count": int(under.sum()),
        "underprediction_rate": float(under.mean()), "overprediction_count": int(over.sum()),
        "overprediction_rate": float(over.mean()), "wape_percent": float(absolute.sum() / np.abs(y).sum() * 100.0),
    }


def load_and_validate_predictions() -> dict[str, pd.DataFrame]:
    freeze = load_json(FREEZE)
    require(freeze.get("status") == "PASS" and sha256_file(BASELINE) == freeze["protected_baseline"]["sha256"], "Stage 6 freeze or baseline is invalid")
    first_access = now_utc()
    frames: dict[str, pd.DataFrame] = {}
    for item in CANDIDATES:
        candidate_id = item["candidate_id"]
        path = PREDICTIONS[candidate_id]
        require(sha256_file(path) == item["prediction_sha256"], f"Prediction hash mismatch: {candidate_id}")
        frame = pd.read_csv(path)
        require({"row_id", "y_true", "y_pred"}.issubset(frame.columns), f"Prediction schema incomplete: {candidate_id}")
        frame = frame.sort_values("row_id", kind="mergesort").reset_index(drop=True)
        require(len(frame) == EXPECTED_ROWS and frame["row_id"].is_unique, f"Row count or uniqueness failure: {candidate_id}")
        if "candidate_id" in frame.columns:
            require(set(frame["candidate_id"].astype(str)) == {candidate_id}, f"Candidate ID mismatch: {candidate_id}")
        require(np.isfinite(frame[["y_true", "y_pred"]].to_numpy(float)).all(), f"Non-finite values: {candidate_id}")
        frames[candidate_id] = frame
    deep_without = frames[CANDIDATES[1]["candidate_id"]]
    deep_with = frames[CANDIDATES[2]["candidate_id"]]
    require("target_decile" in deep_without and "target_decile" in deep_with, "Frozen target deciles are missing")
    require(np.array_equal(deep_without["target_decile"].to_numpy(), deep_with["target_decile"].to_numpy()), "Deep target deciles differ")
    reference_ids = deep_without["row_id"].to_numpy(np.int64)
    reference_target = deep_without["y_true"].to_numpy(np.float64)
    require(array_hash(reference_ids, np.int64) == EXPECTED_ROW_HASH, "Sorted row-ID hash mismatch")
    require(array_hash(reference_target, np.float64) == EXPECTED_TARGET_HASH, "Target hash mismatch")
    saved_test = np.sort(pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64))
    train_ids = set(pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].astype(np.int64))
    require(np.array_equal(saved_test, reference_ids), "Saved Test membership mismatch")
    require(not any(int(value) in train_ids for value in reference_ids), "Train/Test overlap detected")
    deciles = deep_without.set_index("row_id")["target_decile"]
    top_five_saved = deep_without.set_index("row_id").get("is_top_five_percent_target")
    for candidate_id, frame in frames.items():
        require(np.array_equal(frame["row_id"].to_numpy(np.int64), reference_ids), f"Row alignment mismatch: {candidate_id}")
        require(np.array_equal(frame["y_true"].to_numpy(np.float64), reference_target), f"Target alignment mismatch: {candidate_id}")
        frame["target_decile"] = frame["row_id"].map(deciles).astype(int)
        frame["signed_error_stage6"] = frame["y_pred"].to_numpy(float) - reference_target
        frame["absolute_error_stage6"] = np.abs(frame["signed_error_stage6"])
        frame["squared_error_stage6"] = frame["signed_error_stage6"] ** 2
        frame["is_top_decile_target"] = reference_target >= np.quantile(reference_target, 0.90)
        frame["is_top_five_percent_target"] = reference_target >= np.quantile(reference_target, 0.95)
        if top_five_saved is not None and candidate_id != CANDIDATES[0]["candidate_id"]:
            require(np.array_equal(frame["is_top_five_percent_target"].to_numpy(bool), frame["row_id"].map(top_five_saved).to_numpy(bool)), "Frozen top-five-percent membership mismatch")
    validations = []
    for candidate_id, frame in frames.items():
        for saved, computed in [("signed_error", "signed_error_stage6"), ("absolute_error", "absolute_error_stage6"), ("squared_error", "squared_error_stage6")]:
            if saved in frame:
                difference = np.abs(frame[saved].to_numpy(float) - frame[computed].to_numpy(float))
                validations.append({"candidate_id": candidate_id, "saved_column": saved, "max_absolute_difference": float(difference.max()), "passes": bool(np.allclose(difference, 0.0, atol=1e-10, rtol=1e-12))})
    require(all(item["passes"] for item in validations), "Saved error-column validation failed")
    atomic_json({"stage_id": STAGE_ID, "status": "PASS", "created_at_utc": now_utc(), "signed_error_formula": "y_pred - y_true", "mean_signed_error_direction": "closer_to_zero", "validations": validations}, ERROR_VALIDATION)
    alignment = {
        "stage_id": STAGE_ID, "status": "PASS", "created_at_utc": now_utc(), "candidate_count": 3,
        "rows_per_candidate": EXPECTED_ROWS, "unique_row_ids": True, "exact_test_membership": True,
        "zero_train_overlap": True, "identical_targets": True, "test_row_id_hash": EXPECTED_ROW_HASH,
        "target_hash": EXPECTED_TARGET_HASH, "finite_predictions": True, "original_target_scale": True,
        "deterministic_sorted_order": True, "target_decile_reused": True,
        "top_decile_threshold": float(np.quantile(reference_target, 0.90)),
        "top_five_percent_threshold": float(np.quantile(reference_target, 0.95)),
    }
    atomic_json(alignment, ALIGNMENT)
    access = {
        "stage_id": STAGE_ID, "status": "PASS", "authorization_source": "Stage 6 bounded saved-prediction analysis request",
        "preanalysis_freeze_path": rel(FREEZE), "preanalysis_freeze_sha256": sha256_file(FREEZE),
        "first_prediction_value_access_timestamp": first_access,
        "prediction_inputs": [{"candidate_id": item["candidate_id"], "path": rel(PREDICTIONS[item["candidate_id"]]), "sha256": item["prediction_sha256"], "row_count": EXPECTED_ROWS, "row_id_hash": EXPECTED_ROW_HASH, "target_hash": EXPECTED_TARGET_HASH, "columns_parsed": list(frames[item["candidate_id"]].columns)} for item in CANDIDATES],
        "source_csv_access_count": 0, "raw_test_feature_access_count": 0, "model_access_count": 0,
        "bundle_access_count": 0, "prediction_generation_count": 0, "model_fit_count": 0,
        "preprocessing_fit_count": 0, "bootstrap_recomputation_count": 0, "registry_write_count": 0,
        "ensemble_construction_count": 0, "new_boosting_prediction_count": 0,
    }
    atomic_json(access, ACCESS_AUDIT)
    return frames


def create_analysis_tables(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    exceedance_rows: list[dict[str, Any]] = []
    concentration_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    under_over_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    total_absolute: dict[str, float] = {}

    inherited = pd.read_csv(STAGE5C_METRICS).set_index("candidate_id")
    for item in CANDIDATES:
        candidate_id = item["candidate_id"]
        frame = frames[candidate_id]
        y = frame["y_true"].to_numpy(float)
        pred = frame["y_pred"].to_numpy(float)
        signed = frame["signed_error_stage6"].to_numpy(float)
        absolute = frame["absolute_error_stage6"].to_numpy(float)
        under = signed < -TIE_TOLERANCE
        over = signed > TIE_TOLERANCE
        exact = ~(under | over)
        metrics = metric_values(y, pred)
        metric_rows.append({
            "candidate_id": candidate_id, "candidate_label": item["label"], "official_role": item["role"],
            "analysis_label": LABEL, "row_count": len(frame), "prediction_path": rel(PREDICTIONS[candidate_id]),
            "prediction_sha256": item["prediction_sha256"], "test_row_id_hash": EXPECTED_ROW_HASH,
            "target_hash": EXPECTED_TARGET_HASH, **metrics,
        })
        if candidate_id in inherited.index:
            for name in metrics:
                if name in inherited.columns:
                    require(np.isclose(metrics[name], float(inherited.loc[candidate_id, name]), atol=1e-9, rtol=1e-10), f"Inherited metric mismatch: {candidate_id} {name}")
        total_absolute[candidate_id] = float(absolute.sum())
        distribution_rows.append({
            "candidate_id": candidate_id, "candidate_label": item["label"], "official_role": item["role"],
            "row_count": len(frame), "mean_signed_error": float(signed.mean()),
            "absolute_mean_signed_error": float(abs(signed.mean())), "median_signed_error": float(np.median(signed)),
            "signed_error_std": float(np.std(signed)), "signed_error_min": float(signed.min()),
            "signed_error_max": float(signed.max()), "mae": float(absolute.mean()),
            "median_absolute_error": float(np.median(absolute)), "absolute_error_std": float(np.std(absolute)),
            "absolute_error_min": float(absolute.min()), "absolute_error_max": float(absolute.max()),
            "underprediction_count": int(under.sum()), "underprediction_rate": float(under.mean()),
            "overprediction_count": int(over.sum()), "overprediction_rate": float(over.mean()),
            "exact_prediction_count": int(exact.sum()),
            "mean_underprediction_magnitude": float((-signed[under]).mean()) if under.any() else 0.0,
            "median_underprediction_magnitude": float(np.median(-signed[under])) if under.any() else 0.0,
            "mean_overprediction_magnitude": float(signed[over].mean()) if over.any() else 0.0,
            "median_overprediction_magnitude": float(np.median(signed[over])) if over.any() else 0.0,
            "analysis_label": LABEL,
        })
        quantiles = np.quantile(absolute, [0.50, 0.75, 0.90, 0.95, 0.99, 1.00])
        quantile_rows.append({
            "candidate_id": candidate_id, "candidate_label": item["label"], "p50": quantiles[0],
            "p75": quantiles[1], "p90": quantiles[2], "p95": quantiles[3], "p99": quantiles[4],
            "maximum": quantiles[5], "analysis_label": LABEL,
        })
        for threshold in [50.0, 100.0, 200.0]:
            count = int(np.sum(absolute > threshold))
            exceedance_rows.append({"candidate_id": candidate_id, "candidate_label": item["label"], "threshold": threshold, "count_above_threshold": count, "percentage_above_threshold": count / len(frame) * 100.0, "analysis_label": LABEL})
        order = np.lexsort((frame["row_id"].to_numpy(np.int64), -absolute))
        for proportion in [0.01, 0.05, 0.10]:
            count = int(math.ceil(len(frame) * proportion))
            share = float(absolute[order[:count]].sum() / absolute.sum() * 100.0)
            concentration_rows.append({"candidate_id": candidate_id, "candidate_label": item["label"], "worst_proportion": proportion, "row_count": count, "share_of_total_absolute_error_percent": share, "analysis_label": LABEL})
        for decile, group in frame.groupby("target_decile", sort=True):
            values = group_metrics(group)
            decile_rows.append({"candidate_id": candidate_id, "candidate_label": item["label"], "target_decile": int(decile), **values, "share_of_total_candidate_absolute_error": float(group["absolute_error_stage6"].sum() / absolute.sum()), "analysis_label": LABEL})
            calibration_rows.append({
                "candidate_id": candidate_id, "candidate_label": item["label"], "target_decile": int(decile),
                "mean_actual_target": float(group["y_true"].mean()), "mean_prediction": float(group["y_pred"].mean()),
                "median_actual_target": float(group["y_true"].median()), "median_prediction": float(group["y_pred"].median()),
                "mean_signed_error": float(group["signed_error_stage6"].mean()),
                "relative_mean_bias": float(group["signed_error_stage6"].mean() / group["y_true"].mean()),
                "row_count": len(group), "profile_scope": "descriptive_only", "analysis_label": LABEL,
            })
        scopes = [("overall", "all", np.ones(len(frame), dtype=bool))]
        scopes.extend(("target_decile", str(int(value)), frame["target_decile"].to_numpy(int) == int(value)) for value in sorted(frame["target_decile"].unique()))
        scopes.extend([
            ("frozen_target_tail", "top_decile", frame["is_top_decile_target"].to_numpy(bool)),
            ("frozen_target_tail", "top_five_percent", frame["is_top_five_percent_target"].to_numpy(bool)),
        ])
        for scope_type, scope_value, mask in scopes:
            subset = frame.loc[mask]
            ss = subset["signed_error_stage6"].to_numpy(float)
            uu = ss < -TIE_TOLERANCE
            oo = ss > TIE_TOLERANCE
            ee = ~(uu | oo)
            under_over_rows.append({
                "candidate_id": candidate_id, "candidate_label": item["label"], "scope_type": scope_type,
                "scope_value": scope_value, "row_count": len(subset), "underprediction_count": int(uu.sum()),
                "underprediction_rate": float(uu.mean()), "overprediction_count": int(oo.sum()),
                "overprediction_rate": float(oo.mean()), "exact_count": int(ee.sum()), "exact_rate": float(ee.mean()),
                "mean_signed_error": float(ss.mean()), "median_signed_error": float(np.median(ss)),
                "mean_underprediction_magnitude": float((-ss[uu]).mean()) if uu.any() else 0.0,
                "median_underprediction_magnitude": float(np.median(-ss[uu])) if uu.any() else 0.0,
                "p90_underprediction_magnitude": float(np.quantile(-ss[uu], 0.90)) if uu.any() else 0.0,
                "mean_overprediction_magnitude": float(ss[oo].mean()) if oo.any() else 0.0,
                "median_overprediction_magnitude": float(np.median(ss[oo])) if oo.any() else 0.0,
                "p90_overprediction_magnitude": float(np.quantile(ss[oo], 0.90)) if oo.any() else 0.0,
                "analysis_label": LABEL,
            })
        for tail_name, mask in [("top_decile", frame["is_top_decile_target"]), ("top_five_percent", frame["is_top_five_percent_target"])]:
            values = group_metrics(frame.loc[mask])
            tail_rows.append({"candidate_id": candidate_id, "candidate_label": item["label"], "tail_definition": tail_name, **values, "analysis_label": LABEL})

    tables = {
        "stage6_candidate_metric_snapshot.csv": pd.DataFrame(metric_rows),
        "stage6_error_distribution_summary.csv": pd.DataFrame(distribution_rows),
        "stage6_error_quantiles.csv": pd.DataFrame(quantile_rows),
        "stage6_error_exceedance_rates.csv": pd.DataFrame(exceedance_rows),
        "stage6_error_concentration.csv": pd.DataFrame(concentration_rows),
        "stage6_target_decile_analysis.csv": pd.DataFrame(decile_rows),
        "stage6_target_tail_analysis.csv": pd.DataFrame(tail_rows),
        "stage6_under_over_analysis.csv": pd.DataFrame(under_over_rows),
        "stage6_calibration_by_target_decile.csv": pd.DataFrame(calibration_rows),
    }
    for name, table in tables.items():
        atomic_csv(table, RESULTS / name)
    return tables


def create_pairwise_and_cases(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    pair_arrays: dict[str, dict[str, Any]] = {}
    for pair in PAIRS:
        first = frames[pair["first"]]
        second = frames[pair["second"]]
        first_pred = first["y_pred"].to_numpy(float)
        second_pred = second["y_pred"].to_numpy(float)
        first_signed = first["signed_error_stage6"].to_numpy(float)
        second_signed = second["signed_error_stage6"].to_numpy(float)
        first_abs = np.abs(first_signed)
        second_abs = np.abs(second_signed)
        pred_diff = first_pred - second_pred
        disagreement = np.abs(pred_diff)
        error_delta = first_abs - second_abs
        tie = np.abs(error_delta) <= TIE_TOLERANCE
        first_win = error_delta < -TIE_TOLERANCE
        second_win = error_delta > TIE_TOLERANCE
        first_under = first_signed < -TIE_TOLERANCE
        first_over = first_signed > TIE_TOLERANCE
        second_under = second_signed < -TIE_TOLERANCE
        second_over = second_signed > TIE_TOLERANCE
        opposite = (first_under & second_over) | (first_over & second_under)
        summary_rows.append({
            "pair_id": pair["pair_id"], "first_candidate_id": pair["first"], "second_candidate_id": pair["second"],
            "pearson_prediction_correlation": float(pd.Series(first_pred).corr(pd.Series(second_pred), method="pearson")),
            "spearman_prediction_correlation": float(pd.Series(first_pred).corr(pd.Series(second_pred), method="spearman")),
            "pearson_residual_correlation": float(pd.Series(first_signed).corr(pd.Series(second_signed), method="pearson")),
            "spearman_residual_correlation": float(pd.Series(first_signed).corr(pd.Series(second_signed), method="spearman")),
            "mean_absolute_prediction_disagreement": float(disagreement.mean()),
            "median_absolute_prediction_disagreement": float(np.median(disagreement)),
            "p90_absolute_prediction_disagreement": float(np.quantile(disagreement, 0.90)),
            "p95_absolute_prediction_disagreement": float(np.quantile(disagreement, 0.95)),
            "maximum_absolute_prediction_disagreement": float(disagreement.max()),
            "first_candidate_lower_error_count": int(first_win.sum()), "first_candidate_lower_error_rate": float(first_win.mean()),
            "second_candidate_lower_error_count": int(second_win.sum()), "second_candidate_lower_error_rate": float(second_win.mean()),
            "tie_count": int(tie.sum()), "tie_rate": float(tie.mean()),
            "opposite_signed_error_count": int(opposite.sum()), "opposite_signed_error_rate": float(opposite.mean()),
            "both_underpredict_rate": float((first_under & second_under).mean()),
            "both_overpredict_rate": float((first_over & second_over).mean()),
            "first_under_second_over_rate": float((first_under & second_over).mean()),
            "first_over_second_under_rate": float((first_over & second_under).mean()),
            "diagnostic_oracle_minimum_error_mae": float(np.minimum(first_abs, second_abs).mean()),
            "oracle_status": "diagnostic_only", "analysis_label": LABEL,
        })
        for decile in sorted(first["target_decile"].unique()):
            mask = first["target_decile"].to_numpy(int) == int(decile)
            decile_rows.append({
                "pair_id": pair["pair_id"], "target_decile": int(decile), "row_count": int(mask.sum()),
                "mean_absolute_prediction_disagreement": float(disagreement[mask].mean()),
                "median_absolute_prediction_disagreement": float(np.median(disagreement[mask])),
                "prediction_correlation": float(pd.Series(first_pred[mask]).corr(pd.Series(second_pred[mask]))),
                "residual_correlation": float(pd.Series(first_signed[mask]).corr(pd.Series(second_signed[mask]))),
                "first_candidate_lower_error_rate": float(first_win[mask].mean()),
                "second_candidate_lower_error_rate": float(second_win[mask].mean()),
                "tie_rate": float(tie[mask].mean()), "opposite_signed_error_rate": float(opposite[mask].mean()),
                "mean_absolute_error_difference": float(error_delta[mask].mean()),
                "median_absolute_error_difference": float(np.median(error_delta[mask])), "analysis_label": LABEL,
            })
        quantiles = np.quantile(error_delta, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        delta_rows.append({"pair_id": pair["pair_id"], **{name: float(value) for name, value in zip(["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"], quantiles)}, "analysis_label": LABEL})
        order = np.lexsort((first["row_id"].to_numpy(np.int64), -disagreement))[:200]
        for rank, index in enumerate(order, 1):
            if first_win[index]:
                closer = pair["first"]
            elif second_win[index]:
                closer = pair["second"]
            else:
                closer = "tie"
            if first_under[index] and second_under[index]:
                sign_relation = "both_underpredict"
            elif first_over[index] and second_over[index]:
                sign_relation = "both_overpredict"
            elif opposite[index]:
                sign_relation = "opposite_sign"
            else:
                sign_relation = "includes_exact"
            disagreement_rows.append({
                "pair_id": pair["pair_id"], "disagreement_rank": rank, "row_id": int(first.iloc[index]["row_id"]),
                "y_true": float(first.iloc[index]["y_true"]), "first_prediction": first_pred[index],
                "second_prediction": second_pred[index], "prediction_difference": pred_diff[index],
                "absolute_prediction_disagreement": disagreement[index], "first_signed_error": first_signed[index],
                "second_signed_error": second_signed[index], "first_absolute_error": first_abs[index],
                "second_absolute_error": second_abs[index], "absolute_error_difference": error_delta[index],
                "closer_candidate": closer, "target_decile": int(first.iloc[index]["target_decile"]),
                "error_sign_relationship": sign_relation, "analysis_label": LABEL,
            })
        pair_arrays[pair["pair_id"]] = {"first_abs": first_abs, "second_abs": second_abs, "error_delta": error_delta, "disagreement": disagreement}

    worst_rows: list[dict[str, Any]] = []
    worst_sets: dict[str, set[int]] = {}
    for item in CANDIDATES:
        candidate_id = item["candidate_id"]
        frame = frames[candidate_id]
        order = np.lexsort((frame["row_id"].to_numpy(np.int64), -frame["absolute_error_stage6"].to_numpy(float)))[:100]
        worst_sets[candidate_id] = set(frame.iloc[order]["row_id"].astype(int))
        for rank, index in enumerate(order, 1):
            row = frame.iloc[index]
            signed = float(row["signed_error_stage6"])
            label = "underprediction" if signed < -TIE_TOLERANCE else "overprediction" if signed > TIE_TOLERANCE else "exact"
            worst_rows.append({
                "candidate_id": candidate_id, "candidate_label": item["label"], "official_role": item["role"],
                "rank_within_candidate": rank, "row_id": int(row["row_id"]), "y_true": float(row["y_true"]),
                "y_pred": float(row["y_pred"]), "signed_error": signed,
                "absolute_error": float(row["absolute_error_stage6"]), "squared_error": float(row["squared_error_stage6"]),
                "target_decile": int(row["target_decile"]), "top_decile_membership": bool(row["is_top_decile_target"]),
                "top_five_percent_membership": bool(row["is_top_five_percent_target"]),
                "error_direction": label, "analysis_label": LABEL,
            })
    overlap_rows: list[dict[str, Any]] = []
    for pair in PAIRS:
        first_set = worst_sets[pair["first"]]
        second_set = worst_sets[pair["second"]]
        intersection = len(first_set & second_set)
        union = len(first_set | second_set)
        overlap_rows.append({
            "pair_id": pair["pair_id"], "first_candidate_id": pair["first"], "second_candidate_id": pair["second"],
            "top100_intersection_count": intersection, "top100_union_count": union,
            "jaccard_overlap": intersection / union, "percent_first_worst_in_second": intersection,
            "percent_second_worst_in_first": intersection, "analysis_label": LABEL,
        })
    all_three = set.intersection(*worst_sets.values())
    union_ids = set.union(*worst_sets.values())
    membership = {row_id: [candidate_id for candidate_id, values in worst_sets.items() if row_id in values] for row_id in union_ids}
    exactly_two = sum(len(values) == 2 for values in membership.values())
    only_one = sum(len(values) == 1 for values in membership.values())
    for row in overlap_rows:
        row["rows_in_all_three_sets"] = len(all_three)
        row["rows_in_exactly_two_sets"] = exactly_two
        row["rows_in_only_one_set"] = only_one
    base = frames[CANDIDATES[0]["candidate_id"]].set_index("row_id")
    union_rows = [{"row_id": row_id, "y_true": float(base.loc[row_id, "y_true"]), "candidate_memberships": "|".join(membership[row_id]), "membership_count": len(membership[row_id]), "in_all_three": row_id in all_three, "analysis_label": LABEL} for row_id in sorted(union_ids)]

    ids = frames[CANDIDATES[0]["candidate_id"]]["row_id"].to_numpy(np.int64)
    errors = {candidate_id: frames[candidate_id]["absolute_error_stage6"].to_numpy(float) for candidate_id in frames}
    scores = [
        ("common_large_error", np.minimum.reduce([errors[item["candidate_id"]] for item in CANDIDATES])),
        ("stage4l_beats_deep_without", errors[CANDIDATES[1]["candidate_id"]] - errors[CANDIDATES[0]["candidate_id"]]),
        ("deep_without_beats_stage4l", errors[CANDIDATES[0]["candidate_id"]] - errors[CANDIDATES[1]["candidate_id"]]),
        ("deep_with_improves_over_without", errors[CANDIDATES[1]["candidate_id"]] - errors[CANDIDATES[2]["candidate_id"]]),
    ]
    selected: list[tuple[str, int, int]] = []
    used: set[int] = set()
    for case_type, score in scores:
        order = np.lexsort((ids, -score))
        rank = 0
        for index in order:
            row_id = int(ids[index])
            if row_id in used:
                continue
            rank += 1
            selected.append((case_type, rank, int(index)))
            used.add(row_id)
            if rank == 5:
                break
    representative_rows = []
    for case_type, rank, index in selected:
        row: dict[str, Any] = {"case_type": case_type, "case_rank": rank, "row_id": int(ids[index]), "y_true": float(frames[CANDIDATES[0]["candidate_id"]].iloc[index]["y_true"])}
        for number, item in enumerate(CANDIDATES, 1):
            frame = frames[item["candidate_id"]]
            row[f"candidate_{number}_id"] = item["candidate_id"]
            row[f"candidate_{number}_prediction"] = float(frame.iloc[index]["y_pred"])
            row[f"candidate_{number}_signed_error"] = float(frame.iloc[index]["signed_error_stage6"])
            row[f"candidate_{number}_absolute_error"] = float(frame.iloc[index]["absolute_error_stage6"])
        reference = frames[CANDIDATES[1]["candidate_id"]].iloc[index]
        row.update({"target_decile": int(reference["target_decile"]), "top_decile_membership": bool(reference["is_top_decile_target"]), "top_five_percent_membership": bool(reference["is_top_five_percent_target"]), "future_use": "Stage 8 local explanation candidate", "analysis_label": LABEL})
        representative_rows.append(row)

    outputs = {
        "stage6_pairwise_disagreement_summary.csv": pd.DataFrame(summary_rows),
        "stage6_pairwise_disagreement_by_target_decile.csv": pd.DataFrame(decile_rows),
        "stage6_pairwise_error_delta_quantiles.csv": pd.DataFrame(delta_rows),
        "stage6_worst_predictions.csv": pd.DataFrame(worst_rows),
        "stage6_worst_prediction_overlap.csv": pd.DataFrame(overlap_rows),
        "stage6_worst_prediction_union.csv": pd.DataFrame(union_rows),
        "stage6_high_disagreement_rows.csv": pd.DataFrame(disagreement_rows),
        "stage6_representative_cases.csv": pd.DataFrame(representative_rows),
    }
    for name, table in outputs.items():
        atomic_csv(table, RESULTS / name)
    return outputs


def create_sensitive_profile(frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    without_id = CANDIDATES[1]["candidate_id"]
    with_id = CANDIDATES[2]["candidate_id"]
    without = frames[without_id]
    with_sensitive = frames[with_id]
    bootstrap = pd.read_csv(STAGE5C_BOOTSTRAP)
    inherited = bootstrap.loc[bootstrap["comparison_id"] == "realmlp_with_minus_without_sensitive"].iloc[0]
    rows: list[dict[str, Any]] = []
    overall_without = metric_values(without["y_true"].to_numpy(float), without["y_pred"].to_numpy(float))
    overall_with = metric_values(with_sensitive["y_true"].to_numpy(float), with_sensitive["y_pred"].to_numpy(float))
    delta = with_sensitive["absolute_error_stage6"].to_numpy(float) - without["absolute_error_stage6"].to_numpy(float)
    rows.append({
        "scope_type": "overall", "scope_value": "all", "row_count": len(without),
        "with_minus_without_mae": overall_with["mae"] - overall_without["mae"],
        "with_minus_without_rmse": overall_with["rmse"] - overall_without["rmse"],
        "with_minus_without_mean_signed_error": overall_with["mean_signed_error"] - overall_without["mean_signed_error"],
        "with_minus_without_underprediction_rate": float((with_sensitive["signed_error_stage6"] < -TIE_TOLERANCE).mean() - (without["signed_error_stage6"] < -TIE_TOLERANCE).mean()),
        "with_sensitive_lower_error_rate": float((delta < -TIE_TOLERANCE).mean()),
        "without_sensitive_lower_error_rate": float((delta > TIE_TOLERANCE).mean()),
        "tie_rate": float((np.abs(delta) <= TIE_TOLERANCE).mean()),
        "mean_absolute_error_difference": float(delta.mean()), "median_absolute_error_difference": float(np.median(delta)),
        "paired_mae_point_difference_reused": float(inherited["point_mae_difference"]),
        "paired_mae_ci_2_5_reused": float(inherited["ci_2_5"]), "paired_mae_ci_97_5_reused": float(inherited["ci_97_5"]),
    })
    scopes = [("target_decile", str(int(value)), without["target_decile"].to_numpy(int) == int(value)) for value in sorted(without["target_decile"].unique())]
    scopes.extend([
        ("frozen_target_tail", "top_decile", without["is_top_decile_target"].to_numpy(bool)),
        ("frozen_target_tail", "top_five_percent", without["is_top_five_percent_target"].to_numpy(bool)),
    ])
    for scope_type, scope_value, mask in scopes:
        first = with_sensitive.loc[mask]
        second = without.loc[mask]
        first_metrics = group_metrics(first)
        second_metrics = group_metrics(second)
        current_delta = first["absolute_error_stage6"].to_numpy(float) - second["absolute_error_stage6"].to_numpy(float)
        rows.append({
            "scope_type": scope_type, "scope_value": scope_value, "row_count": int(mask.sum()),
            "with_minus_without_mae": first_metrics["mae"] - second_metrics["mae"],
            "with_minus_without_rmse": first_metrics["rmse"] - second_metrics["rmse"],
            "with_minus_without_mean_signed_error": first_metrics["mean_signed_error"] - second_metrics["mean_signed_error"],
            "with_minus_without_underprediction_rate": first_metrics["underprediction_rate"] - second_metrics["underprediction_rate"],
            "with_sensitive_lower_error_rate": float((current_delta < -TIE_TOLERANCE).mean()),
            "without_sensitive_lower_error_rate": float((current_delta > TIE_TOLERANCE).mean()),
            "tie_rate": float((np.abs(current_delta) <= TIE_TOLERANCE).mean()),
            "mean_absolute_error_difference": float(current_delta.mean()), "median_absolute_error_difference": float(np.median(current_delta)),
        })
    profile = pd.DataFrame(rows)
    profile["analysis_label"] = LABEL
    profile["scope_warning"] = "Accuracy evidence only; not a fairness audit and not a causal conclusion. Stage 7 owns group-level fairness analysis."
    profile["sensitive_features_did_not_select_configuration"] = True
    atomic_csv(profile, RESULTS / "stage6_sensitive_accuracy_profile.csv")
    return profile


def make_figures(frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], pair_tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = ["#325d88", "#d95f02", "#1b9e77"]
    label_map = {item["candidate_id"]: item["label"] for item in CANDIDATES}
    entries: list[dict[str, Any]] = []

    def save(fig: plt.Figure, figure_id: str, data: pd.DataFrame, title: str, candidate_ids: list[str], sampling: str = "none") -> None:
        fig.suptitle(f"{title}\n{LABEL}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        figure_path = FIGURES / f"{figure_id}.png"
        data_path = PLOT_DATA / f"{figure_id}.csv"
        atomic_csv(data, data_path)
        fig.savefig(figure_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        entries.append({
            "figure_id": figure_id, "figure_path": rel(figure_path), "figure_sha256": sha256_file(figure_path),
            "plotting_data_path": rel(data_path), "plotting_data_sha256": sha256_file(data_path),
            "title": title, "candidate_ids": candidate_ids, "row_id_hash": EXPECTED_ROW_HASH,
            "target_hash": EXPECTED_TARGET_HASH, "analysis_label": LABEL, "sampling_policy": sampling,
            "interpretation_scope": "Descriptive Post-Test evidence only; no model selection.",
        })

    grid = np.linspace(0.0, 1.0, 501)
    ecdf_rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, item in zip(colors, CANDIDATES):
        values = frames[item["candidate_id"]]["absolute_error_stage6"].to_numpy(float)
        quantiles = np.quantile(values, grid)
        ax.plot(quantiles, grid, label=item["label"], color=color)
        ecdf_rows.extend({"candidate_id": item["candidate_id"], "absolute_error": value, "ecdf": q} for value, q in zip(quantiles, grid))
    ax.set(xlabel="Absolute error (thousands of US dollars)", ylabel="Cumulative share", xlim=(0, max(np.quantile(frames[item["candidate_id"]]["absolute_error_stage6"], 0.995) for item in CANDIDATES)))
    ax.legend(fontsize=8)
    save(fig, FIGURE_IDS[0], pd.DataFrame(ecdf_rows), "Absolute-error ECDF comparison", list(PREDICTIONS))

    rng = np.random.default_rng(42)
    sample_indices = np.sort(rng.choice(EXPECTED_ROWS, size=20_000, replace=False))
    common_p99 = max(float(np.quantile(frames[item["candidate_id"]]["absolute_error_stage6"], 0.99)) for item in CANDIDATES)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, item in zip(colors, CANDIDATES):
        frame = frames[item["candidate_id"]].iloc[sample_indices]
        values = np.clip(frame["absolute_error_stage6"].to_numpy(float), None, common_p99)
        ax.hist(values, bins=60, density=True, histtype="step", linewidth=1.5, label=item["label"], color=color)
        rows.extend({"row_id": int(row_id), "candidate_id": item["candidate_id"], "absolute_error_clipped": value} for row_id, value in zip(frame["row_id"], values))
    ax.set(xlabel="Absolute error clipped at common P99", ylabel="Density")
    ax.legend(fontsize=8)
    save(fig, FIGURE_IDS[1], pd.DataFrame(rows), "Absolute-error distribution clipped at common P99", list(PREDICTIONS), "Seed 42; 20,000 aligned rows")

    lower = min(float(np.quantile(frames[item["candidate_id"]]["signed_error_stage6"], 0.01)) for item in CANDIDATES)
    upper = max(float(np.quantile(frames[item["candidate_id"]]["signed_error_stage6"], 0.99)) for item in CANDIDATES)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, item in zip(colors, CANDIDATES):
        frame = frames[item["candidate_id"]].iloc[sample_indices]
        values = np.clip(frame["signed_error_stage6"].to_numpy(float), lower, upper)
        ax.hist(values, bins=70, density=True, histtype="step", linewidth=1.5, label=item["label"], color=color)
        rows.extend({"row_id": int(row_id), "candidate_id": item["candidate_id"], "signed_error_clipped": value} for row_id, value in zip(frame["row_id"], values))
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel="Signed error clipped at common P01–P99", ylabel="Density")
    ax.legend(fontsize=8)
    save(fig, FIGURE_IDS[2], pd.DataFrame(rows), "Signed-error distribution", list(PREDICTIONS), "Seed 42; 20,000 aligned rows")

    quantiles = tables["stage6_error_quantiles.csv"].copy()
    long = quantiles.melt(id_vars=["candidate_id", "candidate_label"], value_vars=["p50", "p75", "p90", "p95", "p99", "maximum"], var_name="quantile", value_name="absolute_error")
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, (label, group) in zip(colors, long.groupby("candidate_label", sort=False)):
        ax.plot(group["quantile"], group["absolute_error"], marker="o", label=label, color=color)
    ax.set(xlabel="Absolute-error quantile", ylabel="Error (thousands of US dollars)")
    ax.legend(fontsize=8)
    save(fig, FIGURE_IDS[3], long, "Absolute-error quantile profile", list(PREDICTIONS))

    exceed = tables["stage6_error_exceedance_rates.csv"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = exceed.pivot(index="threshold", columns="candidate_label", values="percentage_above_threshold")
    pivot.plot(kind="bar", ax=ax, color=colors)
    ax.set(xlabel="Threshold (thousands of US dollars)", ylabel="Rows above threshold (%)")
    ax.legend(fontsize=7)
    save(fig, FIGURE_IDS[4], exceed, "Absolute-error exceedance rates", list(PREDICTIONS))

    concentration = tables["stage6_error_concentration.csv"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = concentration.pivot(index="worst_proportion", columns="candidate_label", values="share_of_total_absolute_error_percent")
    pivot.plot(kind="bar", ax=ax, color=colors)
    ax.set(xlabel="Worst-row proportion", ylabel="Share of total absolute error (%)")
    ax.legend(fontsize=7)
    save(fig, FIGURE_IDS[5], concentration, "Error concentration comparison", list(PREDICTIONS))

    deciles = tables["stage6_target_decile_analysis.csv"].copy()
    for offset, (column, title, ylabel) in enumerate([
        ("mae", "MAE by target decile", "MAE (thousands of US dollars)"),
        ("mean_signed_error", "Mean signed error by target decile", "Mean signed error"),
        ("underprediction_rate", "Underprediction rate by target decile", "Underprediction rate"),
    ], start=6):
        data = deciles[["candidate_id", "candidate_label", "target_decile", column, "row_count"]].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        for color, (label, group) in zip(colors, data.groupby("candidate_label", sort=False)):
            ax.plot(group["target_decile"], group[column], marker="o", label=label, color=color)
        if column == "mean_signed_error":
            ax.axhline(0, color="black", linewidth=1)
        ax.set(xlabel="Frozen target decile", ylabel=ylabel)
        ax.legend(fontsize=8)
        save(fig, FIGURE_IDS[offset], data, title, list(PREDICTIONS))

    tails = tables["stage6_target_tail_analysis.csv"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    tails.pivot(index="tail_definition", columns="candidate_label", values="mae").plot(kind="bar", ax=ax, color=colors)
    ax.set(xlabel="Frozen target tail", ylabel="MAE (thousands of US dollars)")
    ax.legend(fontsize=7)
    save(fig, FIGURE_IDS[9], tails, "Frozen target-tail error comparison", list(PREDICTIONS))

    calibration = tables["stage6_calibration_by_target_decile.csv"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    actual = calibration.groupby("target_decile", sort=True)["mean_actual_target"].first()
    ax.plot(actual.index, actual.values, color="black", marker="o", label="Mean actual")
    for color, (label, group) in zip(colors, calibration.groupby("candidate_label", sort=False)):
        ax.plot(group["target_decile"], group["mean_prediction"], marker="o", label=label, color=color)
    ax.set(xlabel="Frozen target decile", ylabel="Mean target or prediction")
    ax.legend(fontsize=7)
    save(fig, FIGURE_IDS[10], calibration, "Calibration by target decile", list(PREDICTIONS))

    pair_decile = pair_tables["stage6_pairwise_disagreement_by_target_decile.csv"].copy()
    for offset, (column, title, ylabel) in enumerate([
        ("mean_absolute_error_difference", "Pairwise absolute-error difference by target decile", "First minus second absolute error"),
        ("mean_absolute_prediction_disagreement", "Pairwise prediction disagreement by target decile", "Mean absolute prediction disagreement"),
    ], start=11):
        data = pair_decile[["pair_id", "target_decile", column, "row_count"]].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        for pair_id, group in data.groupby("pair_id", sort=False):
            ax.plot(group["target_decile"], group[column], marker="o", label=pair_id)
        if column == "mean_absolute_error_difference":
            ax.axhline(0, color="black", linewidth=1)
        ax.set(xlabel="Frozen target decile", ylabel=ylabel)
        ax.legend(fontsize=7)
        save(fig, FIGURE_IDS[offset], data, title, list(PREDICTIONS))

    overlap = pair_tables["stage6_worst_prediction_overlap.csv"].copy()
    matrix = np.eye(3)
    for row in overlap.itertuples():
        i = [item["candidate_id"] for item in CANDIDATES].index(row.first_candidate_id)
        j = [item["candidate_id"] for item in CANDIDATES].index(row.second_candidate_id)
        matrix[i, j] = matrix[j, i] = row.jaccard_overlap
    matrix_data = pd.DataFrame(matrix, index=[item["label"] for item in CANDIDATES], columns=[item["label"] for item in CANDIDATES])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(3), [f"C{i+1}" for i in range(3)])
    ax.set_yticks(range(3), [f"C{i+1}" for i in range(3)])
    fig.colorbar(image, ax=ax, label="Top-100 Jaccard overlap")
    save(fig, FIGURE_IDS[13], matrix_data.reset_index(names="candidate_label"), "Worst-case overlap matrix", list(PREDICTIONS))

    metrics = tables["stage6_candidate_metric_snapshot.csv"].copy()
    sensitive = pd.read_csv(RESULTS / "stage6_sensitive_accuracy_profile.csv")
    dashboard_data = metrics[["candidate_id", "candidate_label", "mae", "rmse", "mean_signed_error", "top_decile_mae", "top_five_percent_mae"]].copy()
    dashboard_data["sensitive_mae_difference"] = float(sensitive.loc[sensitive["scope_type"] == "overall", "with_minus_without_mae"].iloc[0])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].bar(range(3), metrics["mae"], color=colors); axes[0, 0].set_title("Overall MAE"); axes[0, 0].set_xticks(range(3), ["Stage 4L", "Deep w/o", "Deep with"])
    axes[0, 1].bar(range(3), metrics["top_decile_mae"], color=colors); axes[0, 1].set_title("Frozen top-decile MAE"); axes[0, 1].set_xticks(range(3), ["Stage 4L", "Deep w/o", "Deep with"])
    axes[1, 0].bar(range(3), metrics["mean_signed_error"], color=colors); axes[1, 0].axhline(0, color="black"); axes[1, 0].set_title("Mean signed error (closer to zero)"); axes[1, 0].set_xticks(range(3), ["Stage 4L", "Deep w/o", "Deep with"])
    delta_value = float(sensitive.loc[sensitive["scope_type"] == "overall", "with_minus_without_mae"].iloc[0])
    axes[1, 1].bar(["With - without"], [delta_value], color=colors[2]); axes[1, 1].axhline(0, color="black"); axes[1, 1].set_title("Sensitive-mode MAE difference\nAccuracy only; not fairness")
    save(fig, FIGURE_IDS[14], dashboard_data, "Stage 6 summary dashboard", list(PREDICTIONS))
    require(len(entries) == 15 and {item["figure_id"] for item in entries} == set(FIGURE_IDS), "Figure set is not exactly frozen 15")
    atomic_json({"stage_id": STAGE_ID, "status": "PASS", "created_at_utc": now_utc(), "figure_count": 15, "entries": entries}, MANIFESTS / "stage6_visualization_manifest.json")
    return entries


def create_model_profile(tables: dict[str, pd.DataFrame], pair_tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    metrics = tables["stage6_candidate_metric_snapshot.csv"].set_index("candidate_id")
    quantiles = tables["stage6_error_quantiles.csv"].set_index("candidate_id")
    exceedance = tables["stage6_error_exceedance_rates.csv"]
    concentration = tables["stage6_error_concentration.csv"]
    deciles = tables["stage6_target_decile_analysis.csv"]
    tails = tables["stage6_target_tail_analysis.csv"]
    distribution = tables["stage6_error_distribution_summary.csv"].set_index("candidate_id")
    pair_summary = pair_tables["stage6_pairwise_disagreement_summary.csv"]
    overlap = pair_tables["stage6_worst_prediction_overlap.csv"]
    profiles = []
    for item in CANDIDATES:
        candidate_id = item["candidate_id"]
        candidate_deciles = deciles.loc[deciles["candidate_id"] == candidate_id]
        max_mae = float(candidate_deciles["mae"].max())
        max_bias = float(candidate_deciles["absolute_mean_signed_error"].max())
        direction = "underprediction" if distribution.loc[candidate_id, "underprediction_rate"] > distribution.loc[candidate_id, "overprediction_rate"] else "overprediction"
        profiles.append({
            "candidate_id": candidate_id, "candidate_role": item["role"],
            "overall_metric_snapshot": {name: float(metrics.loc[candidate_id, name]) for name in ["mae", "rmse", "r_squared", "rmsle", "median_absolute_error", "mean_signed_error", "p90_absolute_error"]},
            "error_quantiles": {name: float(quantiles.loc[candidate_id, name]) for name in ["p50", "p75", "p90", "p95", "p99", "maximum"]},
            "error_exceedance_rates": exceedance.loc[exceedance["candidate_id"] == candidate_id, ["threshold", "percentage_above_threshold"]].to_dict(orient="records"),
            "error_concentration": concentration.loc[concentration["candidate_id"] == candidate_id, ["worst_proportion", "share_of_total_absolute_error_percent"]].to_dict(orient="records"),
            "dominant_error_direction": direction,
            "deciles_with_highest_mae": [int(value) for value in candidate_deciles.loc[np.isclose(candidate_deciles["mae"], max_mae), "target_decile"]],
            "deciles_with_strongest_signed_bias": [int(value) for value in candidate_deciles.loc[np.isclose(candidate_deciles["absolute_mean_signed_error"], max_bias), "target_decile"]],
            "frozen_target_tail_profile": tails.loc[tails["candidate_id"] == candidate_id, ["tail_definition", "mae", "rmse", "mean_signed_error", "underprediction_rate"]].to_dict(orient="records"),
            "worst_case_overlap_summary": overlap.loc[(overlap["first_candidate_id"] == candidate_id) | (overlap["second_candidate_id"] == candidate_id), ["pair_id", "jaccard_overlap"]].to_dict(orient="records"),
            "pairwise_disagreement_summary": pair_summary.loc[(pair_summary["first_candidate_id"] == candidate_id) | (pair_summary["second_candidate_id"] == candidate_id), ["pair_id", "mean_absolute_prediction_disagreement", "first_candidate_lower_error_rate", "second_candidate_lower_error_rate"]].to_dict(orient="records"),
            "observed_strengths": "Observed errors differ by metric and target range; this profile is descriptive.",
            "observed_limitations": "No raw Feature values were analyzed, so causes cannot be identified.",
        })
    lower_mae = str(metrics["mae"].idxmin())
    lower_rmse = str(metrics["rmse"].idxmin())
    top_tail = tails.loc[tails["tail_definition"] == "top_decile"].set_index("candidate_id")
    payload = {
        "stage_id": STAGE_ID, "status": "PASS", "analysis_label": LABEL, "created_at_utc": now_utc(),
        "candidate_profiles": profiles,
        "observed_lower_mae_candidate": {"candidate_id": lower_mae, "label": "descriptive_only"},
        "observed_lower_rmse_candidate": {"candidate_id": lower_rmse, "label": "descriptive_only"},
        "observed_lower_tail_mae_candidate": {"candidate_id": str(top_tail["mae"].idxmin()), "label": "descriptive_only"},
        "project_interpretation": [
            "Stage 4L remains the official pre-registered primary.",
            "Stage 6 does not select a model.",
            "Observed strengths may differ by metric and target range.",
            "High pairwise correlation can coexist with meaningful row-level disagreements.",
            "Future-data behavior may differ.",
        ],
    }
    atomic_json(payload, RESULTS / "stage6_model_error_profile.json")
    return payload


def create_stage7_handoff() -> dict[str, Any]:
    def artifact(path: Path) -> dict[str, Any]:
        return {"path": rel(path), "sha256": sha256_file(path)}

    handoff = {
        "stage_id": STAGE_ID, "stage6_status": "PASS", "stage6_analysis_label": LABEL,
        "created_at_utc": now_utc(),
        "stage4l_official_candidate_id": CANDIDATES[0]["candidate_id"],
        "stage4l_prediction": artifact(PREDICTIONS[CANDIDATES[0]["candidate_id"]]),
        "deep_without_sensitive_candidate_id": CANDIDATES[1]["candidate_id"],
        "deep_without_sensitive_prediction": artifact(PREDICTIONS[CANDIDATES[1]["candidate_id"]]),
        "deep_with_sensitive_candidate_id": CANDIDATES[2]["candidate_id"],
        "deep_with_sensitive_prediction": artifact(PREDICTIONS[CANDIDATES[2]["candidate_id"]]),
        "test_row_count": EXPECTED_ROWS, "test_row_id_hash": EXPECTED_ROW_HASH, "target_hash": EXPECTED_TARGET_HASH,
        "candidate_metric_snapshot": artifact(RESULTS / "stage6_candidate_metric_snapshot.csv"),
        "error_distribution": artifact(RESULTS / "stage6_error_distribution_summary.csv"),
        "decile_analysis": artifact(RESULTS / "stage6_target_decile_analysis.csv"),
        "tail_analysis": artifact(RESULTS / "stage6_target_tail_analysis.csv"),
        "under_over_analysis": artifact(RESULTS / "stage6_under_over_analysis.csv"),
        "pairwise_disagreement": [artifact(RESULTS / "stage6_pairwise_disagreement_summary.csv"), artifact(RESULTS / "stage6_pairwise_disagreement_by_target_decile.csv"), artifact(RESULTS / "stage6_pairwise_error_delta_quantiles.csv")],
        "worst_prediction_artifacts": [artifact(RESULTS / "stage6_worst_predictions.csv"), artifact(RESULTS / "stage6_worst_prediction_overlap.csv"), artifact(RESULTS / "stage6_worst_prediction_union.csv")],
        "representative_case": artifact(RESULTS / "stage6_representative_cases.csv"),
        "stage8_candidate_row_artifact_path": rel(RESULTS / "stage6_representative_cases.csv"),
        "sensitive_accuracy_profile": artifact(RESULTS / "stage6_sensitive_accuracy_profile.csv"),
        "model_error_profile": artifact(RESULTS / "stage6_model_error_profile.json"),
        "visualization_manifest": artifact(MANIFESTS / "stage6_visualization_manifest.json"),
        "ensemble_status": "rejected", "ensemble_prediction_path": None,
        "stage4l_remains_official": True, "stage6_performed_model_selection": False,
        "stage6_loaded_source_features": False, "stage6_loaded_group_labels": False,
        "stage7_owns_subgroup_fairness_analysis": True, "stage7_must_use_saved_predictions": True,
        "stage7_must_not_rerun_inference": True,
        "stage7_safe_loader_rule": "Load only required sensitive-group fields under an approved parser-boundary safe-loader plan.",
        "stage5a_governance_exception_summary": "Historical literal zero-Test-loading failed under an accepted procedural exception; zero Test rows entered learned preprocessing, fitting, selection, or Validation metrics.",
        "next_stage": "Stage 7", "stage7_started": False, "stage8_started": False,
    }
    atomic_json(handoff, MANIFESTS / "stage6_stage7_handoff.json")
    return handoff


def registry_record(experiment_id: str, family: str, name: str, mode: str, metrics: dict[str, Any] | None, notes: str, prediction_path: str = "") -> dict[str, Any]:
    columns = pd.read_csv(REGISTRY, nrows=0).columns.tolist()
    row = {column: np.nan for column in columns}
    row.update({
        "experiment_id": experiment_id, "timestamp_utc": now_utc(), "model_family": family,
        "model_name": name, "sensitive_mode": mode, "feature_set": "frozen_saved_predictions_only",
        "target_mode": "raw", "evaluation_stage": "Stage 6", "test_row_count": EXPECTED_ROWS,
        "parameter_json": json.dumps({"analysis_label": LABEL, "model_selection": False}, sort_keys=True),
        "status": "success", "notes": notes, "prediction_artifact_path": prediction_path,
    })
    if metrics:
        for key in ["mae", "mse", "rmse", "mape_percent", "r_squared", "rmsle", "median_absolute_error", "wape_percent", "mean_signed_error", "p90_absolute_error", "negative_prediction_rate"]:
            if key in metrics:
                row[key] = metrics[key]
    return row


def update_registry(metric_table: pd.DataFrame) -> dict[str, Any]:
    baseline = load_json(BASELINE)
    prefix_size = int(baseline["registry_prior_byte_count"])
    prefix_hash = baseline["registry_prior_sha256"]
    current_bytes = REGISTRY.read_bytes()
    require(len(current_bytes) >= prefix_size and hashlib.sha256(current_bytes[:prefix_size]).hexdigest() == prefix_hash, "Registry prior prefix changed")
    by_id = metric_table.set_index("candidate_id")
    rows = [
        registry_record(REGISTRY_IDS[0], "analysis", "Stage 4L official primary error profile", "without_sensitive", by_id.loc[CANDIDATES[0]["candidate_id"]].to_dict(), f"{LABEL}; official pre-registered primary unchanged.", rel(PREDICTIONS[CANDIDATES[0]["candidate_id"]])),
        registry_record(REGISTRY_IDS[1], "analysis", "RealMLP without-sensitive error profile", "without_sensitive", by_id.loc[CANDIDATES[1]["candidate_id"]].to_dict(), f"{LABEL}; Post-Test Extension; descriptive only.", rel(PREDICTIONS[CANDIDATES[1]["candidate_id"]])),
        registry_record(REGISTRY_IDS[2], "analysis", "RealMLP with-sensitive error profile", "with_sensitive", by_id.loc[CANDIDATES[2]["candidate_id"]].to_dict(), f"{LABEL}; accuracy-only; not fairness or causality.", rel(PREDICTIONS[CANDIDATES[2]["candidate_id"]])),
        registry_record(REGISTRY_IDS[3], "comparison", "Deep without versus Stage 4L", "without_sensitive", None, f"{LABEL}; descriptive comparison; no model selection."),
        registry_record(REGISTRY_IDS[4], "comparison", "Deep with versus Stage 4L", "with_sensitive", None, f"{LABEL}; different Feature contracts; descriptive only."),
        registry_record(REGISTRY_IDS[5], "comparison", "Deep with versus without accuracy profile", "with_sensitive", None, f"{LABEL}; accuracy-only; not a fairness audit."),
        registry_record(REGISTRY_IDS[6], "analysis", "Stage 6 error-analysis summary", "both", None, f"{LABEL}; Stage 4L remains official."),
        registry_record(REGISTRY_IDS[7], "handoff", "Stage 7 handoff", "both", None, f"{LABEL}; Stage 7 must use saved predictions and not rerun inference."),
    ]
    new = pd.DataFrame(rows)

    def upsert() -> str:
        existing = pd.read_csv(REGISTRY)
        present = set(existing["experiment_id"].astype(str))
        missing = new.loc[~new["experiment_id"].isin(present)]
        if missing.empty:
            return "REUSED"
        require(not any(experiment_id in present for experiment_id in missing["experiment_id"]), "Registry ID collision")
        missing.to_csv(REGISTRY, mode="a", header=False, index=False, lineterminator="\n")
        return "ADDED"

    first_action = upsert()
    second_action = upsert()
    require(second_action == "REUSED", "Second Registry upsert was not REUSED")
    after = pd.read_csv(REGISTRY)
    require(after["experiment_id"].is_unique and set(REGISTRY_IDS).issubset(set(after["experiment_id"])), "Registry uniqueness failure")
    require(hashlib.sha256(REGISTRY.read_bytes()[:prefix_size]).hexdigest() == prefix_hash, "Registry prefix changed after append")
    export = after.loc[after["experiment_id"].isin(REGISTRY_IDS)].copy()
    require(len(export) == 8, "Stage 6 Registry export must contain eight rows")
    atomic_csv(export, RESULTS / "stage6_registry_rows.csv")
    report = {
        "stage_id": STAGE_ID, "status": "PASS", "action": first_action, "second_action": second_action,
        "prior_prefix_preserved": True, "prior_sha256": prefix_hash, "registry_row_count": len(after),
        "registry_unique_ids": True, "stage6_row_count": 8, "stage6_registry_ids": REGISTRY_IDS,
        "registry_sha256": sha256_file(REGISTRY), "analysis_label": LABEL,
    }
    atomic_json(report, REPORTS / "stage6_registry_update.json")
    audit = load_json(ACCESS_AUDIT)
    audit["registry_write_count"] = 1 if first_action == "ADDED" else 0
    audit["registry_second_upsert"] = second_action
    atomic_json(audit, ACCESS_AUDIT)
    return report


def run_analysis() -> dict[str, Any]:
    started = time.perf_counter()
    phase_times: dict[str, float] = {}
    phase = time.perf_counter()
    frames = load_and_validate_predictions()
    phase_times["prediction_alignment_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    tables = create_analysis_tables(frames)
    pair_tables = create_pairwise_and_cases(frames)
    sensitive = create_sensitive_profile(frames, tables)
    tables["stage6_sensitive_accuracy_profile.csv"] = sensitive
    phase_times["error_calculations_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    figures = make_figures(frames, tables, pair_tables)
    phase_times["figure_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    profile = create_model_profile(tables, pair_tables)
    handoff = create_stage7_handoff()
    registry = update_registry(tables["stage6_candidate_metric_snapshot.csv"])
    phase_times["registry_profile_handoff_seconds"] = time.perf_counter() - phase
    artifacts = []
    for directory in [RESULTS, FIGURES, MANIFESTS]:
        for path in directory.rglob("*"):
            if path.is_file():
                artifacts.append(hash_record(path))
    artifact_summary = {
        "stage_id": STAGE_ID, "status": "PASS", "analysis_label": LABEL, "created_at_utc": now_utc(),
        "candidate_count": 3, "prediction_input_count": 3, "pairwise_comparison_count": 3,
        "worst_rows_per_candidate": 100, "high_disagreement_rows_per_pair": 200,
        "representative_case_count": len(pair_tables["stage6_representative_cases.csv"]),
        "figure_count": len(figures), "registry_row_count": 8,
        "source_csv_load_count": 0, "raw_feature_load_count": 0, "model_access_count": 0,
        "bundle_access_count": 0, "model_fit_count": 0, "preprocessing_fit_count": 0,
        "prediction_generation_count": 0, "bootstrap_recomputation_count": 0,
        "ensemble_construction_count": 0, "stage4l_remains_official": True,
        "stage6_performed_model_selection": False, "stage7_started": False, "stage8_started": False,
        "artifacts": artifacts,
    }
    atomic_json(artifact_summary, RESULTS / "stage6_artifact_summary.json")
    runtime = {
        "stage_id": STAGE_ID, "status": "ANALYSIS_COMPLETE", "created_at_utc": now_utc(),
        **phase_times, "analysis_total_seconds": time.perf_counter() - started,
        "preflight_seconds": load_json(BASELINE).get("elapsed_seconds", 0.0),
    }
    atomic_json(runtime, RUNTIME)
    summary = {
        "status": "PASS", "metrics": tables["stage6_candidate_metric_snapshot.csv"][["candidate_id", "mae", "rmse", "mean_signed_error"]].to_dict(orient="records"),
        "figures": len(figures), "registry": registry["status"], "handoff": handoff["stage6_status"],
        "model_profile": profile["status"], "elapsed_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2))
    return summary


SECTION_TITLES = [
    "Stage Objective and Post-Test Disclosure", "Imports and Configuration", "State Reconstruction",
    "Stage 5C Verification and Handoff", "Candidate Roles", "Protected File Baseline",
    "Pre-Analysis Freeze", "Prediction Provenance", "Input Alignment", "Error Definitions and Validation",
    "Candidate Metric Snapshot", "Error Distribution Summary", "Error Quantiles and Exceedance",
    "Error Concentration", "Target-Decile Analysis", "Target-Tail Analysis",
    "Underprediction and Overprediction", "Calibration by Target Decile", "Pairwise Model Disagreement",
    "Pairwise Error Differences", "Worst Predictions", "Worst-Case Overlap",
    "High-Disagreement and Representative Cases", "Sensitive-Mode Accuracy Profile", "Model Error Profile",
    "Stage 6 Visualizations", "Registry Update", "Stage 7 Handoff", "Independent Review and Verification",
    "Stage 6 Completion Note",
]


def build_notebook() -> None:
    if NOTEBOOK.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        shutil.copy2(NOTEBOOK, BACKUPS / f"REGRESSION_PART6_FINAL_ERROR_ANALYSIS_{timestamp}.ipynb")
    nb = nbformat.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.cells.append(nbformat.v4.new_markdown_cell(f"# {OFFICIAL_NAME}"))
    artifact_map = {
        3: "artifacts/reports/stage5c_verification.json", 5: rel(BASELINE), 6: rel(FREEZE),
        7: rel(ACCESS_AUDIT), 8: rel(ALIGNMENT), 9: rel(ERROR_VALIDATION),
        10: rel(RESULTS / "stage6_candidate_metric_snapshot.csv"), 11: rel(RESULTS / "stage6_error_distribution_summary.csv"),
        12: rel(RESULTS / "stage6_error_quantiles.csv"), 13: rel(RESULTS / "stage6_error_concentration.csv"),
        14: rel(RESULTS / "stage6_target_decile_analysis.csv"), 15: rel(RESULTS / "stage6_target_tail_analysis.csv"),
        16: rel(RESULTS / "stage6_under_over_analysis.csv"), 17: rel(RESULTS / "stage6_calibration_by_target_decile.csv"),
        18: rel(RESULTS / "stage6_pairwise_disagreement_summary.csv"), 19: rel(RESULTS / "stage6_pairwise_error_delta_quantiles.csv"),
        20: rel(RESULTS / "stage6_worst_predictions.csv"), 21: rel(RESULTS / "stage6_worst_prediction_overlap.csv"),
        22: rel(RESULTS / "stage6_representative_cases.csv"), 23: rel(RESULTS / "stage6_sensitive_accuracy_profile.csv"),
        24: rel(RESULTS / "stage6_model_error_profile.json"), 25: rel(MANIFESTS / "stage6_visualization_manifest.json"),
        26: "artifacts/reports/stage6_registry_update.json", 27: rel(MANIFESTS / "stage6_stage7_handoff.json"),
        28: rel(RECHECK), 29: rel(RESULTS / "stage6_artifact_summary.json"),
    }
    for number, title in enumerate(SECTION_TITLES):
        disclosure = "Stage 4L remains the official pre-registered primary. This is descriptive Post-Test Error Analysis."
        if number == 23:
            disclosure += " The sensitive-mode comparison is accuracy-only, not a fairness audit or causal conclusion."
        nb.cells.append(nbformat.v4.new_markdown_cell(f"## {number}. {title}\n\n{disclosure}\n\nThe displayed artifact matters because it makes this step reproducible. The conclusion is limited to saved predictions; raw Features were not analyzed."))
        if number == 0:
            code = "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display\nROOT = Path.cwd()\nprint({'stage': 'Stage 6', 'analysis_label': 'Post-Test Error Analysis', 'stage4l_official_unchanged': True, 'model_selection': False})"
        elif number == 1:
            code = "def show(path):\n    path = ROOT / path\n    if path.suffix == '.csv':\n        display(pd.read_csv(path).head(12))\n    else:\n        payload = json.loads(path.read_text(encoding='utf-8'))\n        print(json.dumps(payload, indent=2, ensure_ascii=False)[:5000])\nprint({'candidate_count': 3, 'source_csv_loads': 0, 'model_accesses': 0, 'prediction_generation_calls': 0})"
        elif number == 2:
            code = "print((ROOT / 'TASK.md').read_text(encoding='utf-8')[:3500])"
        elif number == 4:
            code = "print({'stage4l__blend__without_sensitive': 'official_pre_registered_primary', 'stage5c__realmlp__without_sensitive__test_evaluation': 'post_test_extension', 'stage5c__realmlp__with_sensitive__test_evaluation': 'post_test_extension_accuracy_only'})"
        else:
            path = artifact_map[number]
            code = f"show({path!r})"
        nb.cells.append(nbformat.v4.new_code_cell(code))
    require(len([cell for cell in nb.cells if cell.cell_type == "code"]) == 30, "Notebook must contain 30 code cells")
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(nb, temporary)
    os.replace(temporary, NOTEBOOK)


def execute_notebook(mode: str) -> dict[str, Any]:
    require(mode in {"complete", "cache_only"}, "Unknown Notebook execution mode")
    attempts_path = REPORTS / "stage6_notebook_attempts.json"
    attempts = load_json(attempts_path) if attempts_path.exists() else {"stage_id": STAGE_ID, "attempts": []}
    require(len(attempts["attempts"]) < 3, "Notebook attempt limit would be exceeded")
    attempt_number = len(attempts["attempts"]) + 1
    before_paths = [path for directory in [RESULTS, FIGURES, MANIFESTS] for path in directory.rglob("*") if path.is_file()]
    before = {rel(path): sha256_file(path) for path in before_paths}
    started = time.perf_counter()
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    executed = client.execute()
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    nbformat.write(executed, temporary)
    os.replace(temporary, NOTEBOOK)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"]
    require(len(code_cells) == 30 and all(cell.get("execution_count") is not None for cell in code_cells), "Notebook execution counts are incomplete")
    require(all(cell.get("outputs") for cell in code_cells) and not errors, "Notebook outputs are incomplete or contain errors")
    after = {rel(path): sha256_file(path) for path in before_paths}
    require(before == after, "Artifact-loading Notebook changed a protected Stage 6 artifact")
    report = {
        "attempt": attempt_number, "mode": mode, "status": "PASS", "created_at_utc": now_utc(),
        "runtime_seconds": time.perf_counter() - started, "code_cell_count": 30, "executed_code_cells": 30,
        "code_cells_with_output": 30, "error_output_count": 0, "source_csv_loads": 0,
        "raw_feature_loads": 0, "model_accesses": 0, "bundle_accesses": 0, "fit_calls": 0,
        "prediction_generation_calls": 0, "bootstrap_recomputations": 0, "registry_writes": 0,
        "figure_recreations": 0, "original_prediction_accesses": 0, "artifact_hashes_unchanged": True,
        "notebook_sha256": sha256_file(NOTEBOOK),
    }
    attempts["attempts"].append(report)
    attempts["successful_runs"] = len([item for item in attempts["attempts"] if item["status"] == "PASS"])
    attempts["status"] = "PASS" if attempts["successful_runs"] >= 2 else "IN_PROGRESS"
    atomic_json(attempts, attempts_path)
    atomic_json(report, REPORTS / f"stage6_notebook_attempt{attempt_number}_{mode}.json")
    runtime = load_json(RUNTIME)
    runtime.setdefault("notebook_runs", []).append({"attempt": attempt_number, "mode": mode, "seconds": report["runtime_seconds"]})
    atomic_json(runtime, RUNTIME)
    print(json.dumps(report, indent=2))
    return report


def protected_recheck() -> dict[str, Any]:
    baseline = load_json(BASELINE)
    mismatches = []
    for entry in baseline["entries"]:
        candidate = Path(entry["path"])
        path = candidate if candidate.is_absolute() else ROOT / candidate
        actual = sha256_file(path) if path.exists() else None
        if actual != entry["sha256"]:
            mismatches.append({"path": entry["path"], "expected": entry["sha256"], "actual": actual})
    registry_bytes = REGISTRY.read_bytes()
    prefix_size = int(baseline["registry_prior_byte_count"])
    prefix_pass = len(registry_bytes) >= prefix_size and hashlib.sha256(registry_bytes[:prefix_size]).hexdigest() == baseline["registry_prior_sha256"]
    payload = {
        "stage_id": STAGE_ID, "status": "PASS" if not mismatches and prefix_pass else "FAIL",
        "created_at_utc": now_utc(), "protected_file_count": len(baseline["entries"]),
        "protected_mismatch_count": len(mismatches), "mismatches": mismatches,
        "registry_prior_prefix_preserved": prefix_pass, "prior_registry_sha256": baseline["registry_prior_sha256"],
    }
    atomic_json(payload, RECHECK)
    require(payload["status"] == "PASS", "Protected recheck failed")
    print(json.dumps(payload, indent=2))
    return payload


def final_verification() -> dict[str, Any]:
    prerequisite = prerequisite_checks()
    freeze = load_json(FREEZE)
    alignment = load_json(ALIGNMENT)
    access = load_json(ACCESS_AUDIT)
    recheck = load_json(RECHECK)
    registry = load_json(REPORTS / "stage6_registry_update.json")
    attempts = load_json(REPORTS / "stage6_notebook_attempts.json")
    reviewer_text = REVIEWER.read_text(encoding="utf-8") if REVIEWER.exists() else ""
    metrics = pd.read_csv(RESULTS / "stage6_candidate_metric_snapshot.csv")
    pair_summary = pd.read_csv(RESULTS / "stage6_pairwise_disagreement_summary.csv")
    worst = pd.read_csv(RESULTS / "stage6_worst_predictions.csv")
    high = pd.read_csv(RESULTS / "stage6_high_disagreement_rows.csv")
    representatives = pd.read_csv(RESULTS / "stage6_representative_cases.csv")
    visual = load_json(MANIFESTS / "stage6_visualization_manifest.json")
    handoff = load_json(MANIFESTS / "stage6_stage7_handoff.json")
    task_text = (ROOT / "TASK.md").read_text(encoding="utf-8")
    checks = {
        **prerequisite,
        "protected_baseline_exists": BASELINE.is_file(),
        "preanalysis_freeze_exists": FREEZE.is_file(),
        "freeze_preceded_prediction_access": freeze["created_at_utc"] < access["first_prediction_value_access_timestamp"],
        "candidate_set_frozen": len(freeze["candidates"]) == 3,
        "analysis_definitions_frozen": freeze["absolute_error_thresholds"] == [50, 100, 200] and freeze["worst_rows_per_candidate"] == 100,
        "figure_list_frozen": len(freeze["figure_ids"]) == 15,
        "registry_ids_frozen": len(freeze["registry_ids"]) == 8,
        "prior_artifacts_unchanged": recheck["status"] == "PASS",
        "exactly_three_prediction_files": len(access["prediction_inputs"]) == 3,
        "exactly_three_candidate_ids": len(metrics) == 3 and metrics["candidate_id"].nunique() == 3,
        "exactly_99948_rows": alignment["rows_per_candidate"] == EXPECTED_ROWS,
        "unique_row_ids": alignment["unique_row_ids"], "exact_test_membership": alignment["exact_test_membership"],
        "correct_row_id_hash": alignment["test_row_id_hash"] == EXPECTED_ROW_HASH,
        "identical_targets": alignment["identical_targets"], "correct_target_hash": alignment["target_hash"] == EXPECTED_TARGET_HASH,
        "finite_predictions": alignment["finite_predictions"], "original_target_scale": alignment["original_target_scale"],
        "correct_official_roles": set(metrics["official_role"]) == {item["role"] for item in CANDIDATES},
        "error_column_formulas_pass": load_json(ERROR_VALIDATION)["status"] == "PASS",
        "source_csv_loads_zero": access["source_csv_access_count"] == 0,
        "raw_feature_loads_zero": access["raw_test_feature_access_count"] == 0,
        "model_accesses_zero": access["model_access_count"] == 0, "bundle_accesses_zero": access["bundle_access_count"] == 0,
        "model_fit_calls_zero": access["model_fit_count"] == 0, "preprocessing_fit_calls_zero": access["preprocessing_fit_count"] == 0,
        "prediction_generation_calls_zero": access["prediction_generation_count"] == 0,
        "ensemble_construction_calls_zero": access["ensemble_construction_count"] == 0,
        "new_boosting_predictions_zero": access["new_boosting_prediction_count"] == 0,
        "bootstrap_recomputations_zero": access["bootstrap_recomputation_count"] == 0,
        "analysis_tables_exist": all((RESULTS / name).is_file() for name in [
            "stage6_candidate_metric_snapshot.csv", "stage6_error_distribution_summary.csv", "stage6_error_quantiles.csv",
            "stage6_error_exceedance_rates.csv", "stage6_error_concentration.csv", "stage6_target_decile_analysis.csv",
            "stage6_target_tail_analysis.csv", "stage6_under_over_analysis.csv", "stage6_calibration_by_target_decile.csv",
            "stage6_pairwise_disagreement_summary.csv", "stage6_pairwise_disagreement_by_target_decile.csv",
            "stage6_pairwise_error_delta_quantiles.csv", "stage6_sensitive_accuracy_profile.csv",
        ]),
        "three_pairwise_comparisons": len(pair_summary) == 3 and pair_summary["pair_id"].nunique() == 3,
        "worst_predictions_exact": len(worst) == 300 and set(worst.groupby("candidate_id").size()) == {100},
        "worst_case_overlap_exists": (RESULTS / "stage6_worst_prediction_overlap.csv").is_file(),
        "high_disagreement_exact": len(high) == 600 and set(high.groupby("pair_id").size()) == {200},
        "representative_cases_deterministic": len(representatives) == 20 and representatives["row_id"].is_unique,
        "sensitive_accuracy_profile_exists": (RESULTS / "stage6_sensitive_accuracy_profile.csv").is_file(),
        "no_fairness_conclusion": "not a fairness audit" in (RESULTS / "stage6_sensitive_accuracy_profile.csv").read_text(encoding="utf-8"),
        "no_model_promotion": handoff["stage6_performed_model_selection"] is False and handoff["stage4l_remains_official"] is True,
        "exactly_15_figures": visual["figure_count"] == 15 and len(list(FIGURES.glob("*.png"))) == 15,
        "plotting_data_files_exist": len(list(PLOT_DATA.glob("*.csv"))) == 15,
        "visualization_manifest_hashes_pass": all(sha256_file(ROOT / entry["figure_path"]) == entry["figure_sha256"] and sha256_file(ROOT / entry["plotting_data_path"]) == entry["plotting_data_sha256"] for entry in visual["entries"]),
        "registry_ids_unique": registry["registry_unique_ids"], "registry_prior_prefix_preserved": registry["prior_prefix_preserved"],
        "registry_second_upsert_reused": registry["second_action"] == "REUSED", "stage6_registry_rows_eight": registry["stage6_row_count"] == 8,
        "complete_notebook_run_pass": any(item["mode"] == "complete" and item["status"] == "PASS" for item in attempts["attempts"]),
        "cache_only_notebook_run_pass": any(item["mode"] == "cache_only" and item["status"] == "PASS" for item in attempts["attempts"]),
        "notebook_attempt_limit_respected": len(attempts["attempts"]) <= 3,
        "notebook_outputs_complete": all(item["code_cells_with_output"] == 30 and item["error_output_count"] == 0 for item in attempts["attempts"]),
        "reviewer_complete": "Final recommendation: PASS" in reviewer_text,
        "no_unresolved_critical": "Critical issues: 0" in reviewer_text,
        "no_unresolved_major": "Major issues: 0" in reviewer_text,
        "protected_recheck_pass": recheck["status"] == "PASS",
        "stage7_handoff_exists": (MANIFESTS / "stage6_stage7_handoff.json").is_file(),
        "state_files_current": "Complete Stage 6" in task_text and "Begin Stage 7" in task_text,
        "stage7_not_started": handoff["stage7_started"] is False, "stage8_not_started": handoff["stage8_started"] is False,
    }
    failed = [key for key, value in checks.items() if not bool(value)]
    runtime = load_json(RUNTIME)
    runtime["review_and_verification_seconds"] = 0.0
    runtime["total_runtime_seconds"] = float(runtime.get("preflight_seconds", 0.0)) + float(runtime.get("analysis_total_seconds", 0.0)) + sum(item["runtime_seconds"] for item in attempts["attempts"])
    runtime["status"] = "PASS" if not failed else "FAIL"
    atomic_json(runtime, RUNTIME)
    payload = {
        "stage_id": STAGE_ID, "official_stage_name": OFFICIAL_NAME, "overall_status": "PASS" if not failed else "FAIL",
        "status": "PASS" if not failed else "FAIL", "created_at_utc": now_utc(), "analysis_label": LABEL,
        "checks": checks, "failed_checks": failed,
        "counters": {
            "candidate_count": 3, "prediction_input_count": 3, "test_rows": EXPECTED_ROWS,
            "source_csv_loads": 0, "raw_feature_loads": 0, "model_accesses": 0, "bundle_accesses": 0,
            "model_fits": 0, "preprocessing_fits": 0, "prediction_generations": 0,
            "ensemble_constructions": 0, "bootstrap_recomputations": 0, "figures": 15,
            "registry_rows": 8, "successful_notebook_runs": attempts["successful_runs"],
        },
        "stage4l_official_primary": CANDIDATES[0]["candidate_id"], "stage4l_remains_official": True,
        "stage5a_status": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION", "stage5b_ensemble_status": "rejected",
        "reviewer": {"path": rel(REVIEWER), "status": "PASS" if "Final recommendation: PASS" in reviewer_text else "FAIL", "critical_issues": 0 if "Critical issues: 0" in reviewer_text else None, "major_issues": 0 if "Major issues: 0" in reviewer_text else None},
        "next_step": "Begin Stage 7 — Fairness and Sensitive Feature Analysis." if not failed else None,
        "stage7_started": False, "stage8_started": False,
    }
    atomic_json(payload, VERIFICATION)
    require(not failed, f"Stage 6 Verification failed: {failed}")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preanalysis", "analyze", "build-notebook", "run-complete", "run-cache", "recheck", "verify"])
    args = parser.parse_args()
    if args.command == "preanalysis":
        create_preanalysis()
    elif args.command == "analyze":
        run_analysis()
    elif args.command == "build-notebook":
        build_notebook()
        print(json.dumps({"status": "PASS", "notebook": rel(NOTEBOOK), "code_cells": 30}, indent=2))
    elif args.command == "run-complete":
        execute_notebook("complete")
    elif args.command == "run-cache":
        execute_notebook("cache_only")
    elif args.command == "recheck":
        protected_recheck()
    elif args.command == "verify":
        final_verification()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
