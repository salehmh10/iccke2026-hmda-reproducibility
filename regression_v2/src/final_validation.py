"""Final saved-artifact verification and DATA_READY closure for Prompt 1B."""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import nbformat
import numpy as np
import pyarrow.parquet as pq

from feature_engineering import ENGINEERED_FEATURES, NUMERIC_ENGINEERED_FEATURES, duckdb_feature_expressions
from final_dataset_builder import (
    AUDIT_ONLY_FIELDS,
    DATA_DIR,
    LEGACY_KEY_FIELDS,
    NUMERIC_FIELDS,
    PROJECT_ROOT,
    PROJECTS_ROOT,
    REPORT_DIR,
    SENSITIVE_FIELDS,
    TMP_DIR,
    atomic_json,
    file_sha256,
    legacy_equality_sql,
    quoted,
)


DEVELOPMENT_PATH = DATA_DIR / "development.parquet"
IID_FEATURES_PATH = DATA_DIR / "iid_holdout_features.parquet"
IID_TARGETS_PATH = DATA_DIR / "iid_holdout_targets.parquet"


def _load_json(name: str) -> dict[str, Any]:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def _model_calls() -> list[str]:
    found = []
    for path in (PROJECT_ROOT / "src").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
            if name in {"fit", "fit_predict", "predict", "train"}:
                found.append(f"{path.name}:{node.lineno}:{name}")
    return found


def _legacy_select(path: Path) -> str:
    values = []
    for field in LEGACY_KEY_FIELDS:
        source = f'trim("{field}")'
        if field in NUMERIC_FIELDS:
            values.append(f'round(CAST({source} AS DOUBLE), 10) AS "{field}"')
        else:
            values.append(f'CAST({source} AS VARCHAR) AS "{field}"')
    escaped = path.as_posix().replace("'", "''")
    return f"SELECT {', '.join(values)} FROM read_csv('{escaped}', header=true, all_varchar=true) GROUP BY ALL"


def validate_saved_outputs(write_report: bool = True) -> dict[str, Any]:
    """Prove every final Prompt 1B condition from saved artifacts."""
    failures: list[str] = []
    required_reports = [
        "prompt1a_verification.json", "prompt1b_deduplication_report.json",
        "prompt1b_conflict_report.json", "prompt1b_legacy_exclusion_report.json",
        "feature_roles.json", "split_manifest.json", "data_build_report.json",
        "prompt1b_notebook_execution.json", "prompt1b_reviewer.json",
    ]
    for name in required_reports:
        if not (REPORT_DIR / name).exists():
            failures.append(f"Missing report: {name}")
    for path in [DEVELOPMENT_PATH, IID_FEATURES_PATH, IID_TARGETS_PATH]:
        if not path.exists():
            failures.append(f"Missing final data file: {path.name}")
    if failures:
        result = {"status": "FAIL", "checks": {}, "failures": failures}
        if write_report:
            atomic_json(REPORT_DIR / "final_verification.json", result)
        return result

    prompt1a = _load_json("prompt1a_verification.json")
    dedup = _load_json("prompt1b_deduplication_report.json")
    conflict = _load_json("prompt1b_conflict_report.json")
    legacy = _load_json("prompt1b_legacy_exclusion_report.json")
    roles = _load_json("feature_roles.json")
    split = _load_json("split_manifest.json")
    build = _load_json("data_build_report.json")
    notebook_execution = _load_json("prompt1b_notebook_execution.json")
    reviewer = _load_json("prompt1b_reviewer.json")

    development = pq.read_table(DEVELOPMENT_PATH)
    iid_features = pq.read_table(IID_FEATURES_PATH)
    iid_targets = pq.read_table(IID_TARGETS_PATH)
    development_rows = development.num_rows
    iid_rows = iid_features.num_rows
    target_rows = iid_targets.num_rows
    iid_target_columns = iid_targets.schema.names
    target_absent = not any(name in iid_features.schema.names for name in [
        "loan_amount_000s", "target", "y", "log1p_loan_amount_target",
        "log1p_loan_amount_for_eda", "loan_amount_dollars_for_eda",
    ])
    key_order_equal = iid_features["row_hash"].equals(iid_targets["row_hash"])

    connection = duckdb.connect()
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='4GB'")
    connection.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
    development_sql = DEVELOPMENT_PATH.as_posix().replace("'", "''")
    features_sql = IID_FEATURES_PATH.as_posix().replace("'", "''")
    targets_sql = IID_TARGETS_PATH.as_posix().replace("'", "''")
    connection.execute(f"CREATE VIEW development AS SELECT * FROM read_parquet('{development_sql}')")
    connection.execute(f"CREATE VIEW iid_features AS SELECT * FROM read_parquet('{features_sql}')")
    connection.execute(f"CREATE VIEW iid_targets AS SELECT * FROM read_parquet('{targets_sql}')")
    connection.execute("CREATE VIEW iid_complete AS SELECT f.*, t.loan_amount_000s FROM iid_features f JOIN iid_targets t USING (row_hash)")
    role_counts = dict(connection.execute("SELECT development_role, count(*) FROM development GROUP BY 1").fetchall())
    development_row_unique = connection.execute("SELECT count(*) - count(DISTINCT row_hash) FROM development").fetchone()[0]
    iid_row_unique = connection.execute("SELECT count(*) - count(DISTINCT row_hash) FROM iid_features").fetchone()[0]
    development_record_duplicates = connection.execute("SELECT count(*) - count(DISTINCT record_hash) FROM development").fetchone()[0]
    iid_record_duplicates = connection.execute("SELECT count(*) - count(DISTINCT record_hash) FROM iid_features").fetchone()[0]
    cross_row_overlap = connection.execute("SELECT count(*) FROM (SELECT row_hash FROM development INTERSECT SELECT row_hash FROM iid_features)").fetchone()[0]
    cross_record_overlap = connection.execute("SELECT count(*) FROM (SELECT record_hash FROM development INTERSECT SELECT record_hash FROM iid_features)").fetchone()[0]
    selected_conflicts = connection.execute("""
        SELECT count(*) FROM (
            SELECT row_hash FROM (
                SELECT row_hash, loan_amount_000s FROM development
                UNION ALL SELECT row_hash, loan_amount_000s FROM iid_complete
            ) GROUP BY row_hash HAVING count(DISTINCT loan_amount_000s) > 1
        )
    """).fetchone()[0]

    legacy_path = PROJECTS_ROOT / "main" / "hmda_regression_approved_500k.csv"
    connection.execute(f"CREATE TABLE legacy_keys AS {_legacy_select(legacy_path)}")
    dev_legacy_match = legacy_equality_sql("d", "l")
    iid_legacy_match = legacy_equality_sql("i", "l")
    development_legacy_overlap = connection.execute(
        f"SELECT count(*) FROM development d WHERE EXISTS (SELECT 1 FROM legacy_keys l WHERE {dev_legacy_match})"
    ).fetchone()[0]
    iid_legacy_overlap = connection.execute(
        f"SELECT count(*) FROM iid_complete i WHERE EXISTS (SELECT 1 FROM legacy_keys l WHERE {iid_legacy_match})"
    ).fetchone()[0]

    formula_checks = {}
    expression_map = {item.rsplit(" AS ", 1)[1]: item.rsplit(" AS ", 1)[0] for item in duckdb_feature_expressions()}
    for feature in ENGINEERED_FEATURES:
        expression = expression_map[feature]
        if feature in NUMERIC_ENGINEERED_FEATURES:
            predicate = f'"{feature}" IS DISTINCT FROM ({expression})'
        else:
            predicate = f'"{feature}" IS DISTINCT FROM ({expression})'
        mismatches = connection.execute(f"""
            SELECT sum(mismatches) FROM (
                SELECT count(*) FILTER (WHERE {predicate}) mismatches FROM development
                UNION ALL SELECT count(*) FILTER (WHERE {predicate}) FROM iid_features
            )
        """).fetchone()[0]
        formula_checks[feature] = {"status": "PASS" if mismatches == 0 else "FAIL", "mismatches": int(mismatches)}
    infinity_count = 0
    unexpected_null_count = 0
    for feature in NUMERIC_ENGINEERED_FEATURES:
        nonfinite, nulls = connection.execute(f"""
            SELECT sum(nonfinite_count), sum(null_count) FROM (
                SELECT count(*) FILTER (WHERE NOT isfinite("{feature}")) AS nonfinite_count,
                       count(*) FILTER (WHERE "{feature}" IS NULL) AS null_count FROM development
                UNION ALL
                SELECT count(*) FILTER (WHERE NOT isfinite("{feature}")),
                       count(*) FILTER (WHERE "{feature}" IS NULL) FROM iid_features
            )
        """).fetchone()
        infinity_count += int(nonfinite or 0)
        unexpected_null_count += int(nulls or 0)
    for feature in [item for item in ENGINEERED_FEATURES if item not in NUMERIC_ENGINEERED_FEATURES]:
        nulls = connection.execute(f"SELECT (SELECT count(*) FROM development WHERE \"{feature}\" IS NULL) + (SELECT count(*) FROM iid_features WHERE \"{feature}\" IS NULL)").fetchone()[0]
        unexpected_null_count += int(nulls)
    connection.close()

    contract_checks = {}
    for name, fields in roles["contracts"].items():
        values = set(fields)
        contract_checks[name] = {
            "audit_fields_absent": not bool(values & set(AUDIT_ONLY_FIELDS)),
            "target_absent": "loan_amount_000s" not in values,
        }
    default_sensitive_absent = all(
        not (set(roles["contracts"][name]) & set(SENSITIVE_FIELDS))
        for name in ["main_without_sensitive_without_lender", "main_without_sensitive_with_lender"]
    )
    no_lender_excludes_respondent = "respondent_id" not in roles["contracts"]["main_without_sensitive_without_lender"]
    model_calls = _model_calls()
    root_tmp = [path.name for path in PROJECT_ROOT.iterdir() if path.is_dir() and "tmp" in path.name.lower()]
    tmp_items = [path.name for path in TMP_DIR.iterdir()] if TMP_DIR.exists() else []
    notebook = nbformat.read(PROJECT_ROOT / "notebooks" / "01B_BUILD_FINAL_DATASETS.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    notebook_errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]

    checks = {
        "prompt1a_verification_pass": prompt1a.get("status") == "PASS",
        "prompt1a_staging_rows": build["prompt1a_handoff"]["staged_rows"],
        "raw_access_count": build.get("raw_access_count"),
        "development_rows": development_rows,
        "development_train_rows": int(role_counts.get("train", 0)),
        "development_validation_rows": int(role_counts.get("validation", 0)),
        "iid_feature_rows": iid_rows,
        "iid_target_rows": target_rows,
        "development_iid_row_hash_overlap": int(cross_row_overlap),
        "development_row_hash_duplicates": int(development_row_unique),
        "iid_row_hash_duplicates": int(iid_row_unique),
        "development_record_hash_duplicates": int(development_record_duplicates),
        "iid_record_hash_duplicates": int(iid_record_duplicates),
        "cross_set_record_hash_overlap": int(cross_record_overlap),
        "selected_contradictory_target_groups": int(selected_conflicts),
        "development_legacy_overlap": int(development_legacy_overlap),
        "iid_legacy_overlap": int(iid_legacy_overlap),
        "engineered_feature_checks": formula_checks,
        "engineered_infinity_count": infinity_count,
        "unexpected_engineered_null_count": unexpected_null_count,
        "iid_target_absent_from_features": target_absent,
        "iid_target_columns": iid_target_columns,
        "iid_feature_target_keys_equal": key_order_equal,
        "iid_feature_target_order_equal": key_order_equal,
        "default_main_sensitive_fields_absent": default_sensitive_absent,
        "no_lender_contract_excludes_respondent_id": no_lender_excludes_respondent,
        "contract_checks": contract_checks,
        "all_parquet_files_reload": True,
        "zstd_compression": all(
            {pq.ParquetFile(path).metadata.row_group(0).column(0).compression} == {"ZSTD"}
            for path in [DEVELOPMENT_PATH, IID_FEATURES_PATH, IID_TARGETS_PATH]
        ),
        "no_random_root_tmp_directory": not root_tmp,
        "prompt1b_tmp_clean": not tmp_items,
        "notebook_code_cells": len(code_cells),
        "notebook_all_code_cells_executed": all(cell.execution_count is not None for cell in code_cells),
        "notebook_error_count": len(notebook_errors),
        "notebook_execution_pass": notebook_execution.get("status") == "PASS",
        "model_fit_count": 0 if not model_calls else len(model_calls),
        "reviewer_status": reviewer.get("status"),
        "reviewer_unresolved_critical": len(reviewer.get("Critical", [])),
        "reviewer_unresolved_major": len(reviewer.get("Major", [])),
        "deduplication_report_status": dedup.get("status"),
        "conflict_report_status": conflict.get("status"),
        "legacy_report_status": legacy.get("status"),
        "split_manifest_status": split.get("status"),
    }
    required_truths = [
        checks["prompt1a_verification_pass"], checks["prompt1a_staging_rows"] == 6_471_630,
        checks["raw_access_count"] == 0, development_rows == 500_000,
        checks["development_train_rows"] == 400_000, checks["development_validation_rows"] == 100_000,
        iid_rows == target_rows == 75_000, cross_row_overlap == development_row_unique == iid_row_unique == 0,
        development_record_duplicates == iid_record_duplicates == cross_record_overlap == 0,
        selected_conflicts == development_legacy_overlap == iid_legacy_overlap == 0,
        all(item["status"] == "PASS" for item in formula_checks.values()), infinity_count == 0,
        unexpected_null_count == 0, target_absent,
        iid_target_columns == ["row_hash", "loan_amount_000s"], key_order_equal,
        default_sensitive_absent, no_lender_excludes_respondent,
        all(item["audit_fields_absent"] and item["target_absent"] for item in contract_checks.values()),
        checks["zstd_compression"], not root_tmp, not tmp_items,
        checks["notebook_all_code_cells_executed"], len(notebook_errors) == 0,
        notebook_execution.get("status") == "PASS", not model_calls,
        reviewer.get("status") == "PASS", not reviewer.get("Critical"), not reviewer.get("Major"),
        all(item.get("status") == "PASS" for item in [dedup, conflict, legacy, split, build]),
    ]
    if not all(required_truths):
        failures.append("One or more required final verification predicates are false")
    result = {"status": "PASS" if not failures else "FAIL", "checks": checks, "failures": failures}
    if write_report:
        atomic_json(REPORT_DIR / "final_verification.json", result)
    return result


def finalize() -> dict[str, Any]:
    """Write PASS verification and then create DATA_READY as the last report."""
    result = validate_saved_outputs(write_report=True)
    if result["status"] != "PASS":
        raise RuntimeError(f"Final verification failed: {result['failures']}")
    build = _load_json("data_build_report.json")
    legacy = _load_json("prompt1b_legacy_exclusion_report.json")
    dedup = _load_json("prompt1b_deduplication_report.json")
    conflict = _load_json("prompt1b_conflict_report.json")
    ready = {
        "status": "PASS",
        "prompt1a_verification_path": "outputs/reports/prompt1a_verification.json",
        "prompt1a_staging_row_count": 6_471_630,
        "deduplicated_row_count": dedup["rows_after_exact_deduplication"],
        "conflict_excluded_row_count": conflict["excluded_row_count"],
        "legacy_excluded_row_count": legacy["v2_rows_excluded_after_conflict_removal"],
        "development_path": "outputs/data/development.parquet", "development_rows": 500_000,
        "train_rows": 400_000, "validation_rows": 100_000,
        "iid_features_path": "outputs/data/iid_holdout_features.parquet",
        "iid_targets_path": "outputs/data/iid_holdout_targets.parquet", "iid_rows": 75_000,
        "development_iid_overlap": 0, "legacy_overlap": 0,
        "selected_conflict_groups": 0,
        "duplicate_result": {
            "development_record_hash_duplicates": 0, "iid_record_hash_duplicates": 0,
            "cross_set_record_hash_overlap": 0,
        },
        "feature_role_path": "outputs/reports/feature_roles.json",
        "final_verification_path": "outputs/reports/final_verification.json",
        "raw_access_count": 0, "model_fit_count": 0,
        "next_step": "Begin Prompt 2 - Regression V2 Baselines and Boosting Models.",
        "build_runtime_seconds": build["runtime_seconds"],
    }
    atomic_json(REPORT_DIR / "DATA_READY.json", ready)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["check", "finalize"])
    args = parser.parse_args()
    result = validate_saved_outputs(write_report=False) if args.command == "check" else finalize()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
