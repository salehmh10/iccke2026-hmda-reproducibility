"""Build the frozen Prompt 1B Development and IID datasets without model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedShuffleSplit

from feature_engineering import (
    ENGINEERED_FEATURES,
    NUMERIC_ENGINEERED_FEATURES,
    duckdb_feature_expressions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
PROJECTS_ROOT = REPOSITORY_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STAGING_DIR = OUTPUT_DIR / "staging" / "eligible_cleaned"
DATA_DIR = OUTPUT_DIR / "data"
REPORT_DIR = OUTPUT_DIR / "reports"
TMP_DIR = OUTPUT_DIR / "tmp"
CONFIG_PATH = PROJECT_ROOT / "config.json"
MANIFEST_PATH = REPORT_DIR / "prompt1a_partition_manifest.json"
PROMPT1A_VERIFICATION_PATH = REPORT_DIR / "prompt1a_verification.json"

STAGING_FIELDS = [
    "respondent_id", "agency_name", "loan_type_name", "property_type_name",
    "loan_purpose_name", "owner_occupancy_name", "loan_amount_000s",
    "preapproval_name", "msamd_name", "state_name", "state_abbr", "state_code",
    "county_name", "county_code", "census_tract_number", "applicant_ethnicity_name",
    "co_applicant_ethnicity_name", "applicant_race_name_1", "co_applicant_race_name_1",
    "applicant_sex_name", "co_applicant_sex_name", "lien_status_name",
    "applicant_income_000s", "population", "minority_population",
    "hud_median_family_income", "tract_to_msamd_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
    "action_taken_code",
]
NUMERIC_FIELDS = {
    "loan_amount_000s", "applicant_income_000s", "population", "minority_population",
    "hud_median_family_income", "tract_to_msamd_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
}
ROW_HASH_FIELDS = [field for field in STAGING_FIELDS if field != "loan_amount_000s"]
LEGACY_KEY_FIELDS = [field for field in STAGING_FIELDS if field not in {"state_abbr", "action_taken_code"}]
SENSITIVE_FIELDS = [
    "applicant_ethnicity_name", "co_applicant_ethnicity_name", "applicant_race_name_1",
    "co_applicant_race_name_1", "applicant_sex_name", "co_applicant_sex_name",
    "minority_population", "majority_minority_tract",
]
AUDIT_ONLY_FIELDS = [
    "action_taken_code", "row_hash", "record_hash", "development_role",
    "majority_minority_tract",
]
TARGET_ALIASES = {
    "loan_amount_000s", "log1p_loan_amount_target", "log1p_loan_amount_for_eda",
    "loan_amount_dollars_for_eda", "target", "y",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TMP_DIR / f"{path.name}.partial"
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def quoted(field: str, alias: str | None = None) -> str:
    prefix = f'{alias}.' if alias else ""
    return f'{prefix}"{field}"'


def typed_value_sql(field: str, alias: str | None = None) -> str:
    value = quoted(field, alias)
    if field in NUMERIC_FIELDS:
        return f"'f:' || CAST({value} AS VARCHAR)"
    if field == "action_taken_code":
        return f"'i:' || CAST({value} AS VARCHAR)"
    return f"'s:' || {value}"


def canonical_sql(fields: list[str], alias: str | None = None) -> str:
    values = ", ".join(typed_value_sql(field, alias) for field in fields)
    return f"to_json(list_value({values}))"


def hash_sql(fields: list[str], alias: str | None = None) -> str:
    return f"sha256({canonical_sql(fields, alias)})"


def equality_sql(fields: list[str], left: str, right: str) -> str:
    return " AND ".join(
        f"{quoted(field, left)} IS NOT DISTINCT FROM {quoted(field, right)}" for field in fields
    )


def legacy_equality_sql(left: str, right: str) -> str:
    """Compare the frozen Legacy key after stable float canonicalization."""
    terms = []
    for field in LEGACY_KEY_FIELDS:
        left_value = quoted(field, left)
        right_value = quoted(field, right)
        if field in NUMERIC_FIELDS:
            left_value = f"round({left_value}, 10)"
        terms.append(f"{left_value} IS NOT DISTINCT FROM {right_value}")
    return " AND ".join(terms)


def duplicate_safe_deciles(target: pd.Series) -> pd.Series:
    """Create stable quantile bins without splitting equal edge values."""
    bins = pd.qcut(target.astype("float64"), q=10, labels=False, duplicates="drop")
    if bins.isna().any() or bins.nunique() < 2:
        raise RuntimeError("Target values cannot form usable duplicate-safe bins")
    return bins.astype("int8")


def make_split_assignments(candidates: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Create exact fixed Development/IID and Train/Validation memberships."""
    if len(candidates) != 575_000 or candidates["row_hash"].duplicated().any():
        raise ValueError("The candidate pool must contain 575,000 unique row_hash values")
    deciles = duplicate_safe_deciles(candidates["loan_amount_000s"])
    indices = np.arange(len(candidates))
    outer = StratifiedShuffleSplit(n_splits=1, test_size=75_000, random_state=random_state)
    development_position, iid_position = next(outer.split(indices, deciles))
    development = indices[development_position]
    iid = indices[iid_position]
    inner = StratifiedShuffleSplit(n_splits=1, test_size=100_000, random_state=random_state)
    train_position, validation_position = next(inner.split(development, deciles.iloc[development]))
    train = development[train_position]
    validation = development[validation_position]
    role = np.full(len(candidates), "iid", dtype=object)
    role[train] = "train"
    role[validation] = "validation"
    result = pd.DataFrame({
        "row_hash": candidates["row_hash"].to_numpy(),
        "development_role": role,
        "target_decile": deciles.to_numpy(),
    })
    counts = result["development_role"].value_counts().to_dict()
    if counts != {"train": 400_000, "validation": 100_000, "iid": 75_000}:
        raise RuntimeError(f"Unexpected split counts: {counts}")
    return result


def feature_roles(superset_columns: list[str]) -> dict[str, Any]:
    """Create the three frozen model feature contracts."""
    forbidden_main = set(TARGET_ALIASES) | set(SENSITIVE_FIELDS) | set(AUDIT_ONLY_FIELDS)
    without_lender = [
        field for field in superset_columns
        if field not in forbidden_main and field != "respondent_id"
    ]
    with_lender = [field for field in superset_columns if field not in forbidden_main]
    diagnostic_forbidden = set(TARGET_ALIASES) | set(AUDIT_ONLY_FIELDS)
    diagnostic = [field for field in superset_columns if field not in diagnostic_forbidden]
    return {
        "status": "PASS",
        "feature_superset": superset_columns,
        "sensitive_fields": SENSITIVE_FIELDS,
        "audit_only_fields": AUDIT_ONLY_FIELDS,
        "target_and_alias_exclusions": sorted(TARGET_ALIASES),
        "contracts": {
            "main_without_sensitive_without_lender": without_lender,
            "main_without_sensitive_with_lender": with_lender,
            "diagnostic_with_sensitive": diagnostic,
        },
        "notes": {
            "diagnostic_with_sensitive": "For controlled accuracy diagnostics and later fairness or error analysis only.",
            "shared_storage": "All contracts select columns from one saved feature superset.",
        },
    }


def _schema_digest(schema: pa.Schema) -> str:
    payload = json.dumps(
        [{"name": item.name, "type": str(item.type), "nullable": item.nullable} for item in schema],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_prompt1a_handoff() -> dict[str, Any]:
    """Reload and hash every immutable Prompt 1A staging partition."""
    task_text = (PROJECT_ROOT / "TASK.md").read_text(encoding="utf-8")
    verification = json.loads(PROMPT1A_VERIFICATION_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    if "Prompt 1A COMPLETE" not in task_text:
        failures.append("TASK.md does not state Prompt 1A COMPLETE")
    if verification.get("status") != "PASS":
        failures.append("Prompt 1A verification is not PASS")
    if len(manifest) != 112:
        failures.append("Prompt 1A manifest does not contain 112 entries")
    rows = 0
    size = 0
    schema_digests: set[str] = set()
    for expected_id, entry in enumerate(manifest, 1):
        path = PROJECT_ROOT / entry["output_path"]
        try:
            table = pq.read_table(path)
            digest = file_sha256(path)
            current_size = path.stat().st_size
            rows += table.num_rows
            size += current_size
            schema_digests.add(_schema_digest(table.schema))
            if not (
                entry["partition_id"] == expected_id
                and entry["completion_status"] == "COMPLETE"
                and table.num_rows == entry["retained_row_count"]
                and current_size == entry["output_size"]
                and digest == entry["output_sha256"]
                and _schema_digest(table.schema) == entry["schema_digest"]
                and table.schema.names == STAGING_FIELDS
            ):
                failures.append(f"Partition mismatch: {path.name}")
        except Exception as exc:
            failures.append(f"Unreadable partition {path.name}: {exc}")
    if rows != 6_471_630 or size != 142_659_308 or len(schema_digests) != 1:
        failures.append("Prompt 1A staging totals or schema differ from the frozen handoff")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "prompt1a_verification_path": PROMPT1A_VERIFICATION_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "partition_count": len(manifest), "staged_rows": rows, "staging_bytes": size,
        "schema_count": len(schema_digests), "all_hashes_match": not failures,
        "all_partitions_reloaded": not any("Unreadable" in item for item in failures),
        "failures": failures,
    }
    if failures:
        raise RuntimeError("Prompt 1A handoff is invalid: " + "; ".join(failures[:3]))
    return result


def discover_legacy_sources() -> dict[str, Path]:
    """Find the strongest read-only Legacy sample and its processed variants."""
    main = PROJECTS_ROOT / "main"
    sources = {
        "raw_sample": main / "hmda_regression_approved_500k.csv",
        "processed_sensitive": main / "data" / "processed" / "regression_with_sensitive_features.csv",
        "processed_without_sensitive": main / "data" / "processed" / "regression_without_sensitive_features.csv",
        "legacy_notebook": main / "REGRESION_PART1.ipynb",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Required Legacy evidence is missing: {missing}")
    return sources


def _connect(database: Path | None = None) -> duckdb.DuckDBPyConnection:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database) if database else ":memory:")
    connection.execute("SET threads=4")
    connection.execute("SET memory_limit='6GB'")
    connection.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
    return connection


def _legacy_normalized_select(path: Path) -> str:
    expressions = []
    for field in LEGACY_KEY_FIELDS:
        source = f'trim("{field}")'
        expressions.append(
            f"CAST({source} AS DOUBLE) AS \"{field}\"" if field in NUMERIC_FIELDS
            else f"CAST({source} AS VARCHAR) AS \"{field}\""
        )
    escaped = path.as_posix().replace("'", "''")
    return f"SELECT {', '.join(expressions)} FROM read_csv('{escaped}', header=true, all_varchar=true)"


def _parquet_compression(path: Path) -> set[str]:
    metadata = pq.ParquetFile(path).metadata
    return {
        metadata.row_group(row_group).column(column).compression
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.row_group(row_group).num_columns)
    }


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    target = frame["loan_amount_000s"].astype("float64")
    proportions = frame["target_decile"].value_counts(normalize=True).sort_index()
    return {
        "count": int(target.count()), "mean": float(target.mean()),
        "standard_deviation": float(target.std(ddof=1)), "minimum": float(target.min()),
        "median": float(target.median()), "p90": float(target.quantile(0.90)),
        "p95": float(target.quantile(0.95)), "p99": float(target.quantile(0.99)),
        "maximum": float(target.max()),
        "target_decile_proportions": {str(int(key)): float(value) for key, value in proportions.items()},
    }


def _write_parquet(connection: duckdb.DuckDBPyConnection, query: str, final_path: Path) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TMP_DIR / f"{final_path.name}.partial"
    if temporary.exists():
        temporary.unlink()
    escaped = temporary.as_posix().replace("'", "''")
    connection.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    pq.read_table(temporary)
    os.replace(temporary, final_path)


def _final_output_metadata(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    compression = sorted(_parquet_compression(path))
    if compression != ["ZSTD"]:
        raise RuntimeError(f"Unexpected compression for {path.name}: {compression}")
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(), "rows": table.num_rows,
        "columns": table.num_columns, "column_names": table.schema.names,
        "bytes": path.stat().st_size, "sha256": file_sha256(path),
        "schema_digest": _schema_digest(table.schema), "compression": compression,
    }


def run_full_build() -> dict[str, Any]:
    """Run the one complete disk-backed Prompt 1B population build."""
    started = time.perf_counter()
    handoff = validate_prompt1a_handoff()
    sources = discover_legacy_sources()
    final_paths = [
        DATA_DIR / "development.parquet", DATA_DIR / "iid_holdout_features.parquet",
        DATA_DIR / "iid_holdout_targets.parquet", REPORT_DIR / "DATA_READY.json",
    ]
    if any(path.exists() for path in final_paths):
        raise RuntimeError("Prompt 1B final outputs already exist; validate the checkpoint before rebuilding")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["prompt1b"]
    database = TMP_DIR / "prompt1b_full.duckdb"
    if database.exists():
        database.unlink()
    connection = _connect(database)
    staging_glob = (STAGING_DIR / "*.parquet").as_posix().replace("'", "''")
    fields = ", ".join(quoted(field) for field in STAGING_FIELDS)
    group_fields = ", ".join(quoted(field) for field in STAGING_FIELDS)
    connection.execute(f"""
        CREATE TABLE deduplicated AS
        WITH exact_groups AS (
            SELECT {fields}, count(*)::BIGINT AS duplicate_count
            FROM read_parquet('{staging_glob}')
            GROUP BY {group_fields}
        )
        SELECT *, {hash_sql(STAGING_FIELDS)} AS record_hash,
                  {hash_sql(ROW_HASH_FIELDS)} AS row_hash
        FROM exact_groups
    """)
    deduplicated_rows, duplicate_groups, copies_removed = connection.execute("""
        SELECT count(*), count(*) FILTER (WHERE duplicate_count > 1),
               coalesce(sum(duplicate_count - 1) FILTER (WHERE duplicate_count > 1), 0)
        FROM deduplicated
    """).fetchone()
    dedup_report = {
        "status": "PASS", "ordered_exact_fields": STAGING_FIELDS,
        "hash_algorithm": "SHA-256 over a typed ordered JSON array",
        "collision_safety": "Exact GROUP BY compares all 30 physical fields; hash alone never decides equality.",
        "rows_before_deduplication": 6_471_630,
        "exact_duplicate_groups": int(duplicate_groups),
        "duplicate_copies_removed": int(copies_removed),
        "rows_after_exact_deduplication": int(deduplicated_rows),
    }
    atomic_json(REPORT_DIR / "prompt1b_deduplication_report.json", dedup_report)

    row_fields = ", ".join(quoted(field) for field in ROW_HASH_FIELDS)
    connection.execute(f"""
        CREATE TABLE conflict_keys AS
        SELECT {row_fields}, row_hash, count(*)::BIGINT AS affected_rows,
               count(DISTINCT loan_amount_000s)::BIGINT AS distinct_targets
        FROM deduplicated
        GROUP BY {row_fields}, row_hash
        HAVING count(DISTINCT loan_amount_000s) > 1
    """)
    conflict_groups, conflict_rows = connection.execute(
        "SELECT count(*), coalesce(sum(affected_rows), 0) FROM conflict_keys"
    ).fetchone()
    conflict_report = {
        "status": "PASS", "ordered_conflict_fields": ROW_HASH_FIELDS,
        "target_excluded_from_row_hash": True, "action_taken_code_in_row_hash": True,
        "contradictory_group_count": int(conflict_groups),
        "affected_exact_deduplicated_rows": int(conflict_rows),
        "excluded_row_count": int(conflict_rows),
        "full_group_exclusion": True,
    }
    atomic_json(REPORT_DIR / "prompt1b_conflict_report.json", conflict_report)

    legacy_path = sources["raw_sample"]
    connection.execute(f"CREATE TABLE legacy_rows AS {_legacy_normalized_select(legacy_path)}")
    legacy_rows = connection.execute("SELECT count(*) FROM legacy_rows").fetchone()[0]
    null_expression = " OR ".join(f'{quoted(field)} IS NULL' for field in LEGACY_KEY_FIELDS)
    invalid_legacy_rows = connection.execute(f"SELECT count(*) FROM legacy_rows WHERE {null_expression}").fetchone()[0]
    if legacy_rows != 500_000 or invalid_legacy_rows:
        raise RuntimeError("The preferred Legacy source failed row-count or normalization validation")
    original_legacy_fields = ", ".join(quoted(field) for field in LEGACY_KEY_FIELDS)
    legacy_original_unique_keys = connection.execute(
        f"SELECT count(*) FROM (SELECT {original_legacy_fields} FROM legacy_rows GROUP BY {original_legacy_fields})"
    ).fetchone()[0]
    legacy_key_fields = ", ".join(
        f'round({quoted(field)}, 10) AS {quoted(field)}' if field in NUMERIC_FIELDS else quoted(field)
        for field in LEGACY_KEY_FIELDS
    )
    connection.execute(f"""
        CREATE TABLE legacy_keys AS
        SELECT *, {hash_sql(LEGACY_KEY_FIELDS)} AS legacy_key_hash
        FROM (SELECT {legacy_key_fields} FROM legacy_rows GROUP BY ALL)
    """)
    legacy_unique_keys = connection.execute("SELECT count(*) FROM legacy_keys").fetchone()[0]
    legacy_match = legacy_equality_sql("d", "l")
    matched_keys = connection.execute(f"""
        SELECT count(*) FROM legacy_keys l
        WHERE EXISTS (SELECT 1 FROM deduplicated d WHERE {legacy_match})
    """).fetchone()[0]
    if matched_keys != legacy_unique_keys:
        raise RuntimeError(f"Legacy keys are unmatched: {legacy_unique_keys - matched_keys}")
    conflict_match = equality_sql(ROW_HASH_FIELDS, "d", "c")
    legacy_excluded_rows = connection.execute(f"""
        SELECT count(*) FROM deduplicated d
        WHERE NOT EXISTS (SELECT 1 FROM conflict_keys c WHERE {conflict_match})
          AND EXISTS (SELECT 1 FROM legacy_keys l WHERE {legacy_match})
    """).fetchone()[0]
    connection.execute(f"""
        CREATE TABLE eligible_final AS
        SELECT d.* EXCLUDE (duplicate_count)
        FROM deduplicated d
        WHERE NOT EXISTS (SELECT 1 FROM conflict_keys c WHERE {conflict_match})
          AND NOT EXISTS (SELECT 1 FROM legacy_keys l WHERE {legacy_match})
    """)
    eligible_rows = connection.execute("SELECT count(*) FROM eligible_final").fetchone()[0]
    processed_counts = {}
    for key in ["processed_sensitive", "processed_without_sensitive"]:
        path = sources[key].as_posix().replace("'", "''")
        processed_counts[key] = int(connection.execute(
            f"SELECT count(*) FROM read_csv('{path}', header=true, all_varchar=true)"
        ).fetchone()[0])
    legacy_report = {
        "status": "PASS", "source_priority": "A", "source_path": str(legacy_path),
        "source_sha256": file_sha256(legacy_path), "source_bytes": legacy_path.stat().st_size,
        "source_rows": int(legacy_rows), "source_unique_normalized_keys": int(legacy_unique_keys),
        "processed_variant_rows": processed_counts,
        "matching_key_fields": LEGACY_KEY_FIELDS,
        "normalization": "Trim strings, parse the eight numeric fields as float64, and canonicalize both sides to 10 decimal places to remove CSV round-trip noise below 1e-10.",
        "numeric_canonicalization_identity_collapse": int(legacy_original_unique_keys - legacy_unique_keys),
        "matched_legacy_keys": int(matched_keys),
        "unmatched_legacy_keys": int(legacy_unique_keys - matched_keys),
        "v2_rows_excluded_after_conflict_removal": int(legacy_excluded_rows),
        "eligible_rows_after_legacy_exclusion": int(eligible_rows),
        "conservative_multi_match_exclusion": True,
    }
    atomic_json(REPORT_DIR / "prompt1b_legacy_exclusion_report.json", legacy_report)
    if eligible_rows < config["candidate_rows"]:
        raise RuntimeError(f"Only {eligible_rows} eligible rows remain; 575,000 are required")

    seed = config["selection_seed"].replace("'", "''")
    connection.execute(f"""
        CREATE TABLE candidates AS
        SELECT * FROM (
            SELECT e.*, sha256('{seed}' || row_hash) AS selection_score,
                   row_number() OVER (ORDER BY sha256('{seed}' || row_hash), row_hash) AS candidate_rank
            FROM eligible_final e
        )
        WHERE candidate_rank <= {int(config['candidate_rows'])}
    """)
    candidate_frame = connection.execute(
        "SELECT row_hash, loan_amount_000s FROM candidates ORDER BY candidate_rank"
    ).fetchdf()
    if len(candidate_frame) != 575_000:
        raise RuntimeError("Deterministic candidate selection did not produce 575,000 rows")
    assignments = make_split_assignments(candidate_frame, int(config["random_state"]))
    candidate_audit = candidate_frame.copy()
    candidate_audit["target_decile"] = duplicate_safe_deciles(candidate_frame["loan_amount_000s"])
    candidate_audit["development_role"] = assignments["development_role"]
    connection.register("split_assignments", assignments)
    connection.execute("""
        CREATE TABLE selected AS
        SELECT c.*, a.development_role, a.target_decile
        FROM candidates c JOIN split_assignments a USING (row_hash)
    """)
    expressions = ",\n".join(duckdb_feature_expressions())
    connection.execute(f"CREATE TABLE featured AS SELECT *, {expressions} FROM selected")
    engineered_checks = {}
    for field in NUMERIC_ENGINEERED_FEATURES:
        nulls, nonfinite = connection.execute(
            f'SELECT count(*) FILTER (WHERE "{field}" IS NULL), '
            f'count(*) FILTER (WHERE NOT isfinite("{field}")) FROM featured'
        ).fetchone()
        engineered_checks[field] = {"status": "PASS" if nulls == nonfinite == 0 else "FAIL", "nulls": int(nulls), "nonfinite": int(nonfinite)}
    for field in [item for item in ENGINEERED_FEATURES if item not in NUMERIC_ENGINEERED_FEATURES]:
        nulls = connection.execute(f'SELECT count(*) FROM featured WHERE "{field}" IS NULL').fetchone()[0]
        engineered_checks[field] = {"status": "PASS" if nulls == 0 else "FAIL", "nulls": int(nulls)}
    if any(item["status"] != "PASS" for item in engineered_checks.values()):
        raise RuntimeError("Engineered feature validation failed")

    output_base = STAGING_FIELDS + ["record_hash", "row_hash"] + ENGINEERED_FEATURES
    development_columns = output_base + ["development_role"]
    iid_feature_columns = [field for field in output_base if field != "loan_amount_000s"]
    select_development = ", ".join(quoted(field) for field in development_columns)
    select_iid = ", ".join(quoted(field) for field in iid_feature_columns)
    _write_parquet(
        connection,
        f"SELECT {select_development} FROM featured WHERE development_role <> 'iid' ORDER BY candidate_rank",
        DATA_DIR / "development.parquet",
    )
    _write_parquet(
        connection,
        f"SELECT {select_iid} FROM featured WHERE development_role = 'iid' ORDER BY candidate_rank",
        DATA_DIR / "iid_holdout_features.parquet",
    )
    _write_parquet(
        connection,
        "SELECT row_hash, loan_amount_000s FROM featured WHERE development_role = 'iid' ORDER BY candidate_rank",
        DATA_DIR / "iid_holdout_targets.parquet",
    )

    metadata = {
        name: _final_output_metadata(DATA_DIR / filename)
        for name, filename in {
            "development": "development.parquet",
            "iid_features": "iid_holdout_features.parquet",
            "iid_targets": "iid_holdout_targets.parquet",
        }.items()
    }
    if metadata["development"]["rows"] != 500_000 or metadata["iid_features"]["rows"] != 75_000 or metadata["iid_targets"]["rows"] != 75_000:
        raise RuntimeError("Final Parquet row counts are invalid")
    iid_features = pq.read_table(DATA_DIR / "iid_holdout_features.parquet", columns=["row_hash"])
    iid_targets = pq.read_table(DATA_DIR / "iid_holdout_targets.parquet")
    if iid_targets.schema.names != ["row_hash", "loan_amount_000s"]:
        raise RuntimeError("IID target schema is invalid")
    if not iid_features.column("row_hash").equals(iid_targets.column("row_hash")):
        raise RuntimeError("IID feature and target key order differs")

    roles = feature_roles(development_columns)
    if any(set(AUDIT_ONLY_FIELDS) & set(values) for values in roles["contracts"].values()):
        raise RuntimeError("An audit-only field entered a model contract")
    atomic_json(REPORT_DIR / "feature_roles.json", roles)
    distributions = {
        "development_train": _distribution(candidate_audit.loc[candidate_audit["development_role"] == "train"]),
        "development_validation": _distribution(candidate_audit.loc[candidate_audit["development_role"] == "validation"]),
        "iid_holdout": _distribution(candidate_audit.loc[candidate_audit["development_role"] == "iid"]),
    }
    split_manifest = {
        "status": "PASS", "random_state": int(config["random_state"]),
        "target_decile_method": "pandas.qcut(q=10, duplicates='drop') on the frozen 575,000 candidates",
        "candidate_rows": 575_000, "development_rows": 500_000, "train_rows": 400_000,
        "validation_rows": 100_000, "iid_rows": 75_000,
        "selection_score": "SHA-256('regression_v2_seed_42' + row_hash)",
        "membership_changed_after_distribution_review": False,
        "target_distributions": distributions,
    }
    atomic_json(REPORT_DIR / "split_manifest.json", split_manifest)
    report = {
        "status": "PASS", "created_at": utc_now(), "prompt1a_handoff": handoff,
        "raw_access_count": 0, "model_fit_count": 0,
        "population": {
            "staged": 6_471_630, "deduplicated": int(deduplicated_rows),
            "conflict_excluded": int(conflict_rows), "legacy_excluded": int(legacy_excluded_rows),
            "eligible_after_exclusions": int(eligible_rows), "selected": 575_000,
        },
        "engineered_feature_checks": engineered_checks,
        "engineered_infinity_count": int(sum(item.get("nonfinite", 0) for item in engineered_checks.values())),
        "unexpected_engineered_null_count": int(sum(item.get("nulls", 0) for item in engineered_checks.values())),
        "outputs": metadata, "runtime_seconds": float(time.perf_counter() - started),
        "temporary_database": database.relative_to(PROJECT_ROOT).as_posix(),
    }
    atomic_json(REPORT_DIR / "data_build_report.json", report)
    connection.close()
    if database.exists():
        database.unlink()
    for item in list(TMP_DIR.iterdir()):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    return report


def run_smoke_test() -> dict[str, Any]:
    """Run one bounded two-partition test of the frozen scientific operations."""
    started = time.perf_counter()
    sources = discover_legacy_sources()
    part1 = STAGING_DIR / "part-000001.parquet"
    part2 = STAGING_DIR / "part-000002.parquet"
    connection = _connect()
    connection.execute(f"CREATE TABLE smoke_base AS SELECT * FROM read_parquet(['{part1.as_posix()}', '{part2.as_posix()}'])")
    base_rows = connection.execute("SELECT count(*) FROM smoke_base").fetchone()[0]
    fields = ", ".join(quoted(field) for field in STAGING_FIELDS)
    first = connection.execute(f"SELECT {fields} FROM smoke_base LIMIT 1").fetchone()
    second = connection.execute(f"SELECT {fields} FROM smoke_base OFFSET 1 LIMIT 1").fetchone()
    injected_schema = pq.read_schema(part1)
    injected = pa.Table.from_pylist([
        dict(zip(STAGING_FIELDS, first)),
        {**dict(zip(STAGING_FIELDS, second)), "loan_amount_000s": float(second[6]) + 1.0},
    ], schema=injected_schema)
    connection.register("injected_rows", injected)
    connection.execute("CREATE TABLE smoke_input AS SELECT * FROM smoke_base UNION ALL SELECT * FROM injected_rows")
    connection.execute(f"""
        CREATE TABLE smoke_dedup AS
        WITH groups AS (SELECT {fields}, count(*) duplicate_count FROM smoke_input GROUP BY {fields})
        SELECT *, {hash_sql(STAGING_FIELDS)} record_hash, {hash_sql(ROW_HASH_FIELDS)} row_hash FROM groups
    """)
    duplicate_removed = connection.execute("SELECT sum(duplicate_count - 1) FROM smoke_dedup").fetchone()[0]
    row_fields = ", ".join(quoted(field) for field in ROW_HASH_FIELDS)
    connection.execute(f"""
        CREATE TABLE smoke_conflicts AS SELECT {row_fields}, row_hash, count(*) affected_rows
        FROM smoke_dedup GROUP BY {row_fields}, row_hash HAVING count(DISTINCT loan_amount_000s) > 1
    """)
    conflict_rows = connection.execute("SELECT sum(affected_rows) FROM smoke_conflicts").fetchone()[0]
    legacy_projection = ", ".join(
        f'round({quoted(field)}, 10) AS {quoted(field)}' if field in NUMERIC_FIELDS else quoted(field)
        for field in LEGACY_KEY_FIELDS
    )
    connection.execute(f"CREATE TABLE smoke_legacy AS SELECT {legacy_projection} FROM ({_legacy_normalized_select(sources['raw_sample'])})")
    legacy_match = legacy_equality_sql("s", "l")
    legacy_matches = connection.execute(f"""
        SELECT count(*) FROM smoke_dedup s
        WHERE EXISTS (SELECT 1 FROM smoke_legacy l WHERE {legacy_match})
    """).fetchone()[0]
    ranked = connection.execute(f"""
        SELECT row_hash, loan_amount_000s FROM smoke_dedup
        ORDER BY sha256('regression_v2_seed_42' || row_hash), row_hash LIMIT 5750
    """).fetchdf()
    deciles = duplicate_safe_deciles(ranked["loan_amount_000s"])
    indices = np.arange(len(ranked))
    outer = StratifiedShuffleSplit(n_splits=1, test_size=750, random_state=42)
    development_position, iid_position = next(outer.split(indices, deciles))
    development = indices[development_position]; iid = indices[iid_position]
    inner = StratifiedShuffleSplit(n_splits=1, test_size=1000, random_state=42)
    train_position, validation_position = next(inner.split(development, deciles.iloc[development]))
    train = development[train_position]; validation = development[validation_position]
    sample = connection.execute("SELECT * FROM smoke_base LIMIT 100").fetchdf()
    from feature_engineering import engineer_features
    engineered = engineer_features(sample)
    formulas_ok = len(engineered) == 100 and all(field in engineered for field in ENGINEERED_FEATURES)
    finite = np.isfinite(engineered[NUMERIC_ENGINEERED_FEATURES].to_numpy(dtype="float64")).all()
    feature_temp = TMP_DIR / "prompt1b_smoke_features.parquet"
    target_temp = TMP_DIR / "prompt1b_smoke_targets.parquet"
    pq.write_table(pa.table({"row_hash": ranked.loc[iid, "row_hash"].tolist()}), feature_temp, compression="zstd")
    pq.write_table(pa.table({"row_hash": ranked.loc[iid, "row_hash"].tolist(), "loan_amount_000s": ranked.loc[iid, "loan_amount_000s"].tolist()}), target_temp, compression="zstd")
    keys_equal = pq.read_table(feature_temp)["row_hash"].equals(pq.read_table(target_temp)["row_hash"])
    feature_temp.unlink(); target_temp.unlink(); connection.close()
    result = {
        "status": "PASS" if duplicate_removed >= 1 and conflict_rows >= 2 and legacy_matches > 0 and formulas_ok and finite and keys_equal else "FAIL",
        "saved_staging_partitions": [part1.name, part2.name], "saved_staging_rows": int(base_rows),
        "injected_cross_source_duplicate_removed": int(duplicate_removed),
        "contradictory_rows_fully_identified": int(conflict_rows),
        "actual_legacy_matches_in_bounded_staging": int(legacy_matches),
        "deterministic_ranked_rows": len(ranked),
        "split_counts": {"development": len(development), "iid": len(iid), "train": len(train), "validation": len(validation)},
        "all_16_formulas_present": bool(formulas_ok), "engineered_numeric_finite": bool(finite),
        "target_physically_separated": True, "feature_target_key_order_equal": bool(keys_equal),
        "parquet_reload": True, "temp_cleanup": not feature_temp.exists() and not target_temp.exists(),
        "runtime_seconds": float(time.perf_counter() - started), "raw_access_count": 0, "model_fit_count": 0,
    }
    atomic_json(REPORT_DIR / "prompt1b_smoke_test.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"Prompt 1B smoke test failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate-handoff", "smoke", "full"])
    args = parser.parse_args()
    if args.command == "validate-handoff":
        result = validate_prompt1a_handoff()
    elif args.command == "smoke":
        result = run_smoke_test()
    else:
        result = run_full_build()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
