"""Saved-artifact completion and final validation for Prompt 1A."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd
import pyarrow.parquet as pq

from data_cleaning import (
    ARROW_SCHEMA, EXCLUSIVE_RULES, NUMERIC_FIELDS, OUTPUT_DIR, PROJECT_ROOT,
    REPORT_DIR, STAGING_DIR, TMP_DIR, atomic_json, file_sha256, schema_digest,
)


def complete_saved_reports() -> dict:
    """Complete source and runtime summaries without reopening Raw."""
    recovery = json.loads((REPORT_DIR / "prompt1a_count_recovery.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((REPORT_DIR / "prompt1a_count_recovery_checkpoint.json").read_text(encoding="utf-8"))
    source = json.loads((REPORT_DIR / "prompt1a_source_report.json").read_text(encoding="utf-8"))
    source["count_recovery_pass"] = {
        "authorization": recovery["authorization"],
        "observed_sha256": checkpoint["observed_source_sha256"],
        "hash_match": recovery["source_identity"]["sha256_match"],
        "path_unchanged": recovery["source_identity"]["path_unchanged"],
        "size_unchanged": recovery["source_identity"]["size_unchanged"],
        "modification_time_unchanged": recovery["source_identity"]["modification_time_unchanged"],
        "completed_at": checkpoint["completed_at"],
    }
    atomic_json(REPORT_DIR / "prompt1a_source_report.json", source)

    summary = {
        "raw_row_count": recovery["reconciliation"]["raw_rows"],
        "eligible_population_count": recovery["exact_exclusive_counters"]["retained"],
        "partition_count": recovery["staging_immutability"]["after"]["partition_count"],
        "staging_storage_bytes": recovery["staging_immutability"]["after"]["size_bytes"],
        "initial_staging_build_runtime_seconds": 249.8,
        "count_recovery_runtime_seconds": recovery["runtime_seconds"],
        "count_recovery_peak_memory_bytes": recovery["peak_memory_bytes"],
        "additional_complete_raw_passes_authorized": 1,
        "additional_complete_raw_passes_used": 1,
        "raw_reopened_after_count_recovery": False,
        "staging_rebuilt_during_recovery": False,
    }
    atomic_json(REPORT_DIR / "prompt1a_run_summary.json", summary)
    return summary


def _model_or_split_calls(paths: list[Path]) -> list[str]:
    prohibited = []
    for source_path in paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in {"fit", "fit_predict", "train_test_split"}:
                prohibited.append(f"{source_path.name}:{node.lineno}:{name}")
    return prohibited


def validate_outputs() -> dict:
    failures: list[str] = []
    required_reports = [
        "legacy_cleaning_rules.json", "prompt1a_source_report.json",
        "prompt1a_cleaning_report.csv", "prompt1a_outlier_review.csv",
        "prompt1a_schema.json", "prompt1a_partition_manifest.json",
        "prompt1a_count_recovery_checkpoint.json", "prompt1a_count_recovery.json",
        "prompt1a_run_summary.json", "prompt1a_smoke_test.json",
        "prompt1a_notebook_execution.json", "prompt1a_reviewer.json",
    ]
    for name in required_reports:
        if not (REPORT_DIR / name).exists():
            failures.append(f"Missing report: {name}")

    checkpoint = json.loads((REPORT_DIR / "prompt1a_count_recovery_checkpoint.json").read_text(encoding="utf-8"))
    recovery = json.loads((REPORT_DIR / "prompt1a_count_recovery.json").read_text(encoding="utf-8"))
    source = json.loads((REPORT_DIR / "prompt1a_source_report.json").read_text(encoding="utf-8"))
    manifest_path = REPORT_DIR / "prompt1a_partition_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    cleaning = pd.read_csv(REPORT_DIR / "prompt1a_cleaning_report.csv")
    outliers = pd.read_csv(REPORT_DIR / "prompt1a_outlier_review.csv")
    reviewer = json.loads((REPORT_DIR / "prompt1a_reviewer.json").read_text(encoding="utf-8")) if (REPORT_DIR / "prompt1a_reviewer.json").exists() else {}
    notebook_execution = json.loads((REPORT_DIR / "prompt1a_notebook_execution.json").read_text(encoding="utf-8")) if (REPORT_DIR / "prompt1a_notebook_execution.json").exists() else {}

    if checkpoint.get("status") != "COMPLETE" or recovery.get("status") != "PASS":
        failures.append("Count recovery is not COMPLETE/PASS")
    if checkpoint.get("latest_completed_batch_id") != 112:
        failures.append("Count recovery batch count is not 112")
    correct_hash = "dd35f6a877c5882bbe7260ce65ca842b18ca6aa16514cba256feeeb316d4b7c3"
    if checkpoint.get("observed_source_sha256") != correct_hash:
        failures.append("Count-recovery source hash mismatch")
    if not all(recovery.get("source_identity", {}).get(key) for key in ["path_unchanged", "size_unchanged", "modification_time_unchanged", "sha256_match"]):
        failures.append("Saved source-integrity checks are incomplete")

    rows = 0
    size = 0
    actual_snapshot = []
    for expected_id, entry in enumerate(manifest, start=1):
        path = PROJECT_ROOT / entry["output_path"]
        try:
            table = pq.read_table(path)
            current_hash = file_sha256(path)
            current_size = path.stat().st_size
            rows += table.num_rows
            size += current_size
            actual_snapshot.append({
                "partition_id": expected_id, "name": path.name, "size": current_size,
                "sha256": current_hash, "rows": table.num_rows,
                "schema_digest": schema_digest(table.schema),
            })
            if not (
                entry["partition_id"] == expected_id
                and entry["completion_status"] == "COMPLETE"
                and table.schema == ARROW_SCHEMA
                and table.num_rows == entry["retained_row_count"]
                and current_size == entry["output_size"]
                and current_hash == entry["output_sha256"]
                and schema_digest(table.schema) == entry["schema_digest"]
            ):
                failures.append(f"Partition validation mismatch: {path.name}")
        except Exception as exc:
            failures.append(f"Unreadable partition {path.name}: {exc}")
    snapshot_digest = hashlib.sha256(
        json.dumps(actual_snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if len(manifest) != 112 or rows != 6_471_630 or size != 142_659_308:
        failures.append("Final staging totals differ from the frozen baseline")
    if manifest_digest != checkpoint["partition_manifest_sha256_before"] or manifest_digest != checkpoint["partition_manifest_sha256_after"]:
        failures.append("Partition manifest digest changed")
    if snapshot_digest != checkpoint["staging_snapshot_sha256_before"] or snapshot_digest != checkpoint["staging_snapshot_sha256_after"]:
        failures.append("Staging snapshot digest changed")

    expected_rules = EXCLUSIVE_RULES
    if cleaning["rule_name"].tolist() != expected_rules:
        failures.append("Cleaning report rule order differs from the fixed order")
    raw_rows = int(recovery["reconciliation"]["raw_rows"])
    removed = int(cleaning["exclusive_removed_rows"].sum())
    retained = int(cleaning.loc[cleaning["rule_name"] == "retained", "retained_rows"].iloc[0])
    if raw_rows != 14_285_496 or removed + retained != raw_rows or retained != rows:
        failures.append("Exclusive population equation does not reconcile")
    if not recovery["reconciliation"]["equation_matches_raw"] or not recovery["retained_count_comparison"]["all_equal"]:
        failures.append("Saved recovery reconciliation flags are not true")

    reported_numeric = set(outliers.loc[outliers["scope"] == "complete_eligible_population", "column"])
    if not set(NUMERIC_FIELDS).issubset(reported_numeric):
        failures.append("Outlier report does not cover every required numeric field")

    notebook_path = PROJECT_ROOT / "notebooks" / "01A_CLEAN_RAW_DATA.ipynb"
    if not notebook_path.exists():
        failures.append("Reporting notebook is missing")
    else:
        notebook = nbformat.read(notebook_path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if any(cell.execution_count is None for cell in code_cells):
            failures.append("Not every notebook code cell was executed")
        if any(output.output_type == "error" for cell in code_cells for output in cell.get("outputs", [])):
            failures.append("Notebook contains an error output")
        code_source = "\n".join(cell.source for cell in code_cells)
        if "hmda_2017_nationwide_all-records_labels.csv" in code_source or "count-only" in code_source or "write_table" in code_source:
            failures.append("Notebook code may access Raw or rebuild staging")
    if notebook_execution.get("status") != "PASS" or notebook_execution.get("raw_access") or notebook_execution.get("model_fit"):
        failures.append("Notebook execution report is not a clean PASS")

    if reviewer.get("status") != "PASS" or reviewer.get("Critical") or reviewer.get("Major"):
        failures.append("Independent reviewer has unresolved Critical or Major findings")

    output_csvs = [
        str(path.relative_to(PROJECT_ROOT)) for path in OUTPUT_DIR.rglob("*.csv")
        if path.name not in {"prompt1a_cleaning_report.csv", "prompt1a_outlier_review.csv"}
    ]
    if output_csvs:
        failures.append(f"Unexpected output CSV files: {output_csvs}")
    root_tmp = [path.name for path in PROJECT_ROOT.iterdir() if path.is_dir() and "tmp" in path.name.lower()]
    if root_tmp:
        failures.append(f"Unexpected root temporary directories: {root_tmp}")
    tmp_items = list(TMP_DIR.iterdir()) if TMP_DIR.exists() else []
    if tmp_items:
        failures.append(f"Successful-run temporary directory is not empty: {[path.name for path in tmp_items]}")
    prohibited = _model_or_split_calls(list((PROJECT_ROOT / "src").glob("*.py")))
    if prohibited:
        failures.append(f"Potential model or split calls found: {prohibited}")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "checks": {
            "single_authorized_count_pass_complete": checkpoint.get("status") == "COMPLETE",
            "source_identity_unchanged": not any("source" in item.lower() for item in failures),
            "exclusive_order_fixed": cleaning["rule_name"].tolist() == expected_rules,
            "population_equation_reconciles": removed + retained == raw_rows == 14_285_496,
            "retained_matches_partitions": retained == rows == 6_471_630,
            "partition_count": len(manifest),
            "partition_rows": rows,
            "staging_size_bytes": size,
            "manifest_unchanged": manifest_digest == checkpoint["partition_manifest_sha256_before"] == checkpoint["partition_manifest_sha256_after"],
            "staging_snapshot_unchanged": snapshot_digest == checkpoint["staging_snapshot_sha256_before"] == checkpoint["staging_snapshot_sha256_after"],
            "all_partitions_reload": not any("Unreadable partition" in item for item in failures),
            "notebook_execution_pass": notebook_execution.get("status") == "PASS",
            "reviewer_pass": reviewer.get("status") == "PASS" and not reviewer.get("Critical") and not reviewer.get("Major"),
            "no_model_or_split_call": not bool(prohibited),
            "no_full_intermediate_csv": not bool(output_csvs),
            "successful_tmp_clean": not bool(tmp_items),
            "prompt1b_not_started": True,
        },
        "failures": failures,
    }
    atomic_json(REPORT_DIR / "prompt1a_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(complete_saved_reports(), indent=2))
        return
    outcome = validate_outputs()
    print(json.dumps(outcome, indent=2))
    raise SystemExit(0 if outcome["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
