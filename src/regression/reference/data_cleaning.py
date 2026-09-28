"""Prompt 1A bounded-memory HMDA cleaning and Parquet staging build."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from source_reader import discover_raw_csv, iter_csv_batches, read_header


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STAGING_DIR = OUTPUT_DIR / "staging" / "eligible_cleaned"
REPORT_DIR = OUTPUT_DIR / "reports"
TMP_DIR = OUTPUT_DIR / "tmp"
CONFIG_PATH = PROJECT_ROOT / "config.json"
COUNT_CHECKPOINT_PATH = REPORT_DIR / "prompt1a_count_recovery_checkpoint.json"
COUNT_RECOVERY_PATH = REPORT_DIR / "prompt1a_count_recovery.json"

EXCLUSIVE_RULES = [
    "malformed_csv_structure",
    "ineligible_action",
    "required_missing_like",
    "invalid_numeric_parse_or_nonfinite",
    "invalid_numeric_range",
    "invalid_categorical_code",
    "retained",
]

MISSING_LIKE = {"", "nan", "none", "null", "na", "n/a", ".", "-", "--"}

REQUIRED_FIELDS = [
    "applicant_income_000s", "msamd_name", "msamd", "tract_to_msamd_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
    "census_tract_number", "population", "minority_population",
    "hud_median_family_income", "county_name", "county_code", "state_name",
    "state_abbr", "state_code", "loan_amount_000s",
]

NUMERIC_FIELDS = [
    "loan_amount_000s", "applicant_income_000s", "population",
    "minority_population", "hud_median_family_income", "tract_to_msamd_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
]

POSITIVE_FIELDS = set(NUMERIC_FIELDS) - {"minority_population"}

ALLOWED_CODES = {
    "agency_code": {"1", "2", "3", "5", "7", "9"},
    "loan_type": {"1", "2", "3", "4"},
    "property_type": {"1", "2", "3"},
    "loan_purpose": {"1", "2", "3"},
    "owner_occupancy": {"1", "2", "3"},
    "preapproval": {"1", "2", "3"},
    "action_taken": {"1", "2", "8"},
    "applicant_ethnicity": {"1", "2", "3", "4"},
    "co_applicant_ethnicity": {"1", "2", "3", "4", "5"},
    "applicant_race_1": {"1", "2", "3", "4", "5", "6", "7"},
    "co_applicant_race_1": {"1", "2", "3", "4", "5", "6", "7", "8"},
    "applicant_sex": {"1", "2", "3", "4"},
    "co_applicant_sex": {"1", "2", "3", "4", "5"},
    "hoepa_status": {"1", "2"},
    "lien_status": {"1", "2", "3", "4"},
}

RETAINED_STRING_FIELDS = [
    "respondent_id", "agency_name", "loan_type_name", "property_type_name",
    "loan_purpose_name", "owner_occupancy_name", "preapproval_name", "msamd_name",
    "state_name", "state_abbr", "state_code", "county_name", "county_code",
    "census_tract_number", "applicant_ethnicity_name", "co_applicant_ethnicity_name",
    "applicant_race_name_1", "co_applicant_race_name_1", "applicant_sex_name",
    "co_applicant_sex_name", "lien_status_name",
]

RETAINED_FIELDS = (
    RETAINED_STRING_FIELDS[:6]
    + ["loan_amount_000s"]
    + RETAINED_STRING_FIELDS[6:]
    + [
        "applicant_income_000s", "population", "minority_population",
        "hud_median_family_income", "tract_to_msamd_income",
        "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
        "action_taken_code",
    ]
)

SOURCE_FIELDS = list(dict.fromkeys(
    [field for field in RETAINED_FIELDS if field != "action_taken_code"]
    + REQUIRED_FIELDS
    + list(ALLOWED_CODES)
))

ARROW_SCHEMA = pa.schema([
    pa.field(name, pa.float64() if name in NUMERIC_FIELDS else pa.int8() if name == "action_taken_code" else pa.string(), nullable=False)
    for name in RETAINED_FIELDS
])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def is_missing_like(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin(MISSING_LIKE)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].astype("string").str.strip()
    return result


def deterministic_partition_name(partition_id: int) -> str:
    if partition_id < 1:
        raise ValueError("partition_id must be positive")
    return f"part-{partition_id:06d}.parquet"


def schema_digest(schema: pa.Schema = ARROW_SCHEMA) -> str:
    payload = json.dumps(
        [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record(stats: dict[tuple[str, str], int], rule: str, column: str, count: int) -> None:
    stats[(rule, column)] += int(count)


def clean_batch(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], int]]:
    missing_columns = sorted(set(SOURCE_FIELDS) - set(raw_frame.columns))
    if missing_columns:
        raise KeyError(f"Required raw columns are absent: {missing_columns}")

    frame = normalize_frame(raw_frame[SOURCE_FIELDS])
    stats: dict[tuple[str, str], int] = defaultdict(int)

    eligible = frame["action_taken"].isin(ALLOWED_CODES["action_taken"])
    _record(stats, "ineligible_action", "action_taken", int((~eligible).sum()))
    _record(stats, "ineligible_action", "__all__", int((~eligible).sum()))
    frame = frame.loc[eligible].copy()

    missing_masks = {column: is_missing_like(frame[column]) for column in REQUIRED_FIELDS}
    for column, mask in missing_masks.items():
        _record(stats, "required_missing_like", column, int(mask.sum()))
    missing_any = np.logical_or.reduce([mask.to_numpy() for mask in missing_masks.values()])
    _record(stats, "required_missing_like", "__all__", int(missing_any.sum()))
    frame = frame.loc[~missing_any].copy()

    parsed: dict[str, pd.Series] = {}
    parse_masks: dict[str, pd.Series] = {}
    for column in NUMERIC_FIELDS:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        bad = numeric.isna() | ~np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
        parsed[column] = numeric.astype("float64")
        parse_masks[column] = pd.Series(bad, index=frame.index)
        _record(stats, "invalid_numeric_parse_or_nonfinite", column, int(parse_masks[column].sum()))
    parse_any = np.logical_or.reduce([mask.to_numpy() for mask in parse_masks.values()])
    _record(stats, "invalid_numeric_parse_or_nonfinite", "__all__", int(parse_any.sum()))
    frame = frame.loc[~parse_any].copy()
    parsed = {column: values.loc[frame.index] for column, values in parsed.items()}

    range_masks: dict[str, pd.Series] = {}
    for column, numeric in parsed.items():
        if column == "minority_population":
            bad = (numeric < 0) | (numeric > 100)
        else:
            bad = numeric <= 0
        range_masks[column] = bad
        _record(stats, "invalid_numeric_range", column, int(bad.sum()))
    range_any = np.logical_or.reduce([mask.to_numpy() for mask in range_masks.values()])
    _record(stats, "invalid_numeric_range", "__all__", int(range_any.sum()))
    frame = frame.loc[~range_any].copy()
    parsed = {column: values.loc[frame.index] for column, values in parsed.items()}

    code_masks: dict[str, pd.Series] = {}
    for column, allowed in ALLOWED_CODES.items():
        missing = is_missing_like(frame[column])
        bad = (~missing) & (~frame[column].isin(allowed))
        code_masks[column] = bad
        _record(stats, "invalid_categorical_code", column, int(bad.sum()))
    code_any = np.logical_or.reduce([mask.to_numpy() for mask in code_masks.values()])
    _record(stats, "invalid_categorical_code", "__all__", int(code_any.sum()))
    frame = frame.loc[~code_any].copy()

    output = frame[[field for field in RETAINED_FIELDS if field not in NUMERIC_FIELDS and field != "action_taken_code"]].copy()
    for column in NUMERIC_FIELDS:
        output[column] = parsed[column].loc[frame.index].astype("float64")
    output["action_taken_code"] = frame["action_taken"].astype("int8")
    output = output[RETAINED_FIELDS].reset_index(drop=True)
    return output, stats


def frame_to_table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(frame, schema=ARROW_SCHEMA, preserve_index=False, safe=True)


def atomic_json(path: Path, value: Any) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp = TMP_DIR / f"{path.name}.partial"
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_partition(table: pa.Table, partition_id: int) -> dict[str, Any]:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    name = deterministic_partition_name(partition_id)
    final_path = STAGING_DIR / name
    temp_path = TMP_DIR / f"{name}.partial"
    pq.write_table(table, temp_path, compression="zstd", use_dictionary=True)
    reloaded = pq.read_table(temp_path)
    if reloaded.schema != ARROW_SCHEMA or reloaded.num_rows != table.num_rows:
        raise RuntimeError(f"Partition validation failed before commit: {name}")
    del reloaded
    gc.collect()
    os.replace(temp_path, final_path)
    return {
        "output_path": final_path.relative_to(PROJECT_ROOT).as_posix(),
        "output_size": final_path.stat().st_size,
        "output_sha256": file_sha256(final_path),
        "schema_digest": schema_digest(),
        "completion_status": "COMPLETE",
    }


def legacy_rules_report() -> dict[str, Any]:
    return {
        "legacy_notebooks": [
            "../main/DATA_CLEANING_FOR_14M.ipynb",
            "../main/REGRESION_PART1.ipynb",
        ],
        "columns_removed": [
            "edit_status_name", "edit_status", "sequence_number", "application_date_indicator",
            "applicant_race_name_2", "applicant_race_2", "applicant_race_name_3", "applicant_race_3",
            "applicant_race_name_4", "applicant_race_4", "applicant_race_name_5", "applicant_race_5",
            "co_applicant_race_name_2", "co_applicant_race_2", "co_applicant_race_name_3", "co_applicant_race_3",
            "co_applicant_race_name_4", "co_applicant_race_4", "co_applicant_race_name_5", "co_applicant_race_5",
            "denial_reason_name_1", "denial_reason_1", "denial_reason_name_2", "denial_reason_2",
            "denial_reason_name_3", "denial_reason_3", "rate_spread", "purchaser_type_name",
            "purchaser_type", "action_taken_name", "hoepa_status_name",
        ],
        "required_fields": REQUIRED_FIELDS,
        "missing_like_tokens_casefolded": sorted(MISSING_LIKE),
        "meaningful_labels_not_missing": [
            "Information not provided by applicant in mail, Internet, or telephone application",
            "Information not provided", "Not applicable", "No co-applicant",
        ],
        "valid_numeric_ranges": {
            **{column: {"minimum_exclusive": 0, "maximum": None} for column in sorted(POSITIVE_FIELDS)},
            "minority_population": {"minimum_inclusive": 0, "maximum_inclusive": 100},
        },
        "valid_categorical_codes": {column: sorted(values) for column, values in ALLOWED_CODES.items()},
        "final_retained_base_fields": RETAINED_FIELDS,
        "retained_field_reasons": {
            "loan_amount_000s": "Regression target in thousands of U.S. dollars",
            "respondent_id": "Lender identity and later lender ablation",
            "action_taken_code": "Audit-only proof of eligible regression action",
            "sensitive_name_fields_and_minority_population": "Later aggregate sensitive diagnostics",
            "geographic_fields": "Legacy matching, geography, and validated feature engineering",
            "other_base_fields": "Legacy exact comparison and validated feature engineering inputs",
        },
        "outlier_handling": "Retain finite valid extremes; no IQR or Z-score deletion, capping, or winsorization.",
        "duplicate_handling": "Deferred to Prompt 1B. The legacy notebooks do not define one consistent consumed global comparator.",
        "target_definition": {"field": "loan_amount_000s", "unit": "thousands of U.S. dollars"},
        "prompt1a_refinements": [
            "Only the 16 explicitly required fields cause missing-like row removal.",
            "Non-finite numeric values are rejected.",
            "Only action_taken codes 1, 2, and 8 are eligible.",
        ],
    }


def source_report(raw_path: Path, config: dict[str, Any], end_hash: str | None = None) -> dict[str, Any]:
    stat = raw_path.stat()
    zip_path = DATA_DIR / config["raw_zip_name"]
    return {
        "relative_path": raw_path.relative_to(PROJECT_ROOT).as_posix(),
        "absolute_path": str(raw_path),
        "size_bytes": stat.st_size,
        "modification_time_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256_baseline": config["source_sha256"],
        "sha256_processing_pass": end_hash,
        "hash_match": end_hash == config["source_sha256"] if end_hash else None,
        "header": read_header(raw_path),
        "encoding": "UTF-8 (no BOM detected; strict sampled decoding passed)",
        "delimiter": ",",
        "zip_representation": {
            "relative_path": zip_path.relative_to(PROJECT_ROOT).as_posix(),
            "size_bytes": zip_path.stat().st_size if zip_path.exists() else None,
            "read_for_processing": False,
        },
    }


def validate_header(header: list[str]) -> None:
    missing = sorted(set(SOURCE_FIELDS) - set(header))
    if missing:
        raise RuntimeError(f"Required raw columns are absent: {missing}")


def aggregate_stats(total: dict[tuple[str, str], int], batch: dict[tuple[str, str], int]) -> None:
    for key, value in batch.items():
        total[key] += value


def count_cleaning_batch(raw_frame: pd.DataFrame) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Count one batch by calling the exact staging cleaning function."""
    cleaned, stats = clean_batch(raw_frame)
    exclusive = {
        "malformed_csv_structure": 0,
        "ineligible_action": int(stats.get(("ineligible_action", "__all__"), 0)),
        "required_missing_like": int(stats.get(("required_missing_like", "__all__"), 0)),
        "invalid_numeric_parse_or_nonfinite": int(stats.get(("invalid_numeric_parse_or_nonfinite", "__all__"), 0)),
        "invalid_numeric_range": int(stats.get(("invalid_numeric_range", "__all__"), 0)),
        "invalid_categorical_code": int(stats.get(("invalid_categorical_code", "__all__"), 0)),
        "retained": int(len(cleaned)),
    }
    diagnostics: dict[str, dict[str, int]] = defaultdict(dict)
    for (rule, column), count in stats.items():
        if column != "__all__":
            diagnostics[rule][column] = int(count)
    return exclusive, dict(diagnostics)


def add_nested_counts(total: dict[str, dict[str, int]], batch: dict[str, dict[str, int]]) -> None:
    for rule, columns in batch.items():
        target = total.setdefault(rule, {})
        for column, count in columns.items():
            target[column] = int(target.get(column, 0) + count)


def reconcile_exclusive_counts(raw_rows: int, counts: dict[str, int], expected_retained: int) -> dict[str, Any]:
    values = {rule: int(counts.get(rule, 0)) for rule in EXCLUSIVE_RULES}
    if any(value < 0 for value in values.values()):
        raise ValueError("Exclusive counters must be non-negative")
    accounted = sum(values.values())
    return {
        "raw_rows": int(raw_rows),
        "accounted_rows": accounted,
        "equation_matches_raw": accounted == int(raw_rows),
        "retained_expected": int(expected_retained),
        "retained_observed": values["retained"],
        "retained_matches_expected": values["retained"] == int(expected_retained),
    }


def staging_snapshot() -> dict[str, Any]:
    manifest_path = REPORT_DIR / "prompt1a_partition_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    actual = []
    errors = []
    total_rows = 0
    total_size = 0
    for expected_id, entry in enumerate(manifest, start=1):
        path = PROJECT_ROOT / entry["output_path"]
        try:
            metadata = pq.read_metadata(path)
            current_schema_digest = schema_digest(pq.read_schema(path))
            current_hash = file_sha256(path)
            current_size = path.stat().st_size
            current_rows = metadata.num_rows
            total_rows += current_rows
            total_size += current_size
            actual.append({
                "partition_id": expected_id,
                "name": path.name,
                "size": current_size,
                "sha256": current_hash,
                "rows": current_rows,
                "schema_digest": current_schema_digest,
            })
            if not (
                entry["partition_id"] == expected_id
                and entry["completion_status"] == "COMPLETE"
                and entry["retained_row_count"] == current_rows
                and entry["output_size"] == current_size
                and entry["output_sha256"] == current_hash
                and entry["schema_digest"] == current_schema_digest
            ):
                errors.append(path.name)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    snapshot_payload = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "staging_snapshot_sha256": hashlib.sha256(snapshot_payload).hexdigest(),
        "partition_count": len(manifest),
        "row_count": total_rows,
        "size_bytes": total_size,
        "errors": errors,
    }


def checkpoint_template(raw_path: Path, config: dict[str, Any], staging_before: dict[str, Any]) -> dict[str, Any]:
    now = utc_now()
    return {
        "recovery_run_id": "prompt1a_count_recovery_20260805",
        "source_path": raw_path.relative_to(PROJECT_ROOT).as_posix(),
        "source_size": raw_path.stat().st_size,
        "expected_source_sha256": config["source_sha256"],
        "source_modification_time": datetime.fromtimestamp(raw_path.stat().st_mtime, timezone.utc).isoformat(),
        "batch_size_or_block_size": config["batch_block_size_bytes"],
        "latest_completed_batch_id": 0,
        "cumulative_input_rows": 0,
        "cumulative_malformed_csv_structure": 0,
        "cumulative_ineligible_action": 0,
        "cumulative_required_missing_like": 0,
        "cumulative_invalid_numeric_parse_or_nonfinite": 0,
        "cumulative_invalid_numeric_range": 0,
        "cumulative_invalid_categorical_code": 0,
        "cumulative_retained": 0,
        "non_exclusive_diagnostic_counts": {},
        "partition_manifest_sha256_before": staging_before["manifest_sha256"],
        "staging_snapshot_sha256_before": staging_before["staging_snapshot_sha256"],
        "status": "IN_PROGRESS",
        "started_at": now,
        "updated_at": now,
        "completed_at": None,
    }


def update_count_checkpoint(
    checkpoint: dict[str, Any],
    batch_id: int,
    parsed_rows: int,
    malformed_rows: int,
    exclusive: dict[str, int],
    diagnostics: dict[str, dict[str, int]],
) -> dict[str, Any]:
    checkpoint["latest_completed_batch_id"] = int(batch_id)
    checkpoint["cumulative_input_rows"] = int(parsed_rows + malformed_rows)
    checkpoint["cumulative_malformed_csv_structure"] = int(malformed_rows)
    for rule in EXCLUSIVE_RULES[1:]:
        key = f"cumulative_{rule}"
        checkpoint[key] = int(exclusive.get(rule, 0))
    checkpoint["non_exclusive_diagnostic_counts"] = diagnostics
    checkpoint["updated_at"] = utc_now()
    atomic_json(COUNT_CHECKPOINT_PATH, checkpoint)
    reloaded = json.loads(COUNT_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if reloaded["latest_completed_batch_id"] != batch_id:
        raise RuntimeError("Count checkpoint batch ID reload failed")
    for rule in EXCLUSIVE_RULES:
        if reloaded[f"cumulative_{rule}"] != checkpoint[f"cumulative_{rule}"]:
            raise RuntimeError(f"Count checkpoint reload failed for {rule}")
    return reloaded


def write_cleaning_report(raw_rows: int, counts: dict[str, int]) -> None:
    remaining = int(raw_rows)
    rows = []
    notes = {
        "malformed_csv_structure": "Structurally malformed CSV rows; first exclusive rule.",
        "ineligible_action": "Action code is not 1, 2, or 8.",
        "required_missing_like": "At least one of the 16 required fields is missing-like.",
        "invalid_numeric_parse_or_nonfinite": "Required numeric parsing failed or produced infinity.",
        "invalid_numeric_range": "A numeric value violates the fixed positive or percentage range.",
        "invalid_categorical_code": "A present categorical code is outside its fixed allowed set.",
    }
    for order, rule in enumerate(EXCLUSIVE_RULES[:-1], start=1):
        removed = int(counts[rule])
        remaining -= removed
        rows.append({
            "rule_order": order,
            "rule_name": rule,
            "exclusive_removed_rows": removed,
            "retained_rows": 0,
            "percentage_of_raw_rows": removed / raw_rows * 100,
            "cumulative_rows_remaining": remaining,
            "notes": notes[rule],
        })
    rows.append({
        "rule_order": 7,
        "rule_name": "retained",
        "exclusive_removed_rows": 0,
        "retained_rows": int(counts["retained"]),
        "percentage_of_raw_rows": counts["retained"] / raw_rows * 100,
        "cumulative_rows_remaining": int(counts["retained"]),
        "notes": "Complete cleaned eligible population; no deduplication applied.",
    })
    pd.DataFrame(rows).to_csv(REPORT_DIR / "prompt1a_cleaning_report.csv", index=False)


def run_count_only() -> None:
    if COUNT_CHECKPOINT_PATH.exists():
        prior = json.loads(COUNT_CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if prior.get("status") == "COMPLETE":
            print("A COMPLETE count-recovery checkpoint already exists; Raw pass skipped.")
            return
        raise RuntimeError("An incomplete count-recovery checkpoint exists; no additional Raw pass is authorized")

    config = load_config()
    expected_hash = config["source_sha256"]
    if len(expected_hash) != 64:
        raise RuntimeError("Configured source SHA-256 is not 64 characters")
    raw_path = discover_raw_csv(DATA_DIR)
    source_stat = raw_path.stat()
    expected_mtime = datetime.fromisoformat(config["source_mtime_utc"].replace("Z", "+00:00")).timestamp()
    if source_stat.st_size != config["source_size_bytes"] or source_stat.st_mtime != expected_mtime:
        raise RuntimeError("Raw source size or modification time differs from the protected baseline")
    source_evidence = json.loads((REPORT_DIR / "prompt1a_source_report.json").read_text(encoding="utf-8"))
    if source_evidence.get("sha256_processing_pass") != expected_hash or not source_evidence.get("hash_match"):
        raise RuntimeError("Existing verified source hash evidence does not match config")
    validate_header(source_evidence["header"])

    staging_before = staging_snapshot()
    if staging_before != {
        "manifest_sha256": "cc9d62df99a1322bc1afa544efa33f64bd41ec15386a9831e54270028bc320e1",
        "staging_snapshot_sha256": "2d90c202ebf01b3881906f846d5359721b3d9a98ad64a9dad9b6f5525ea98316",
        "partition_count": 112,
        "row_count": 6_471_630,
        "size_bytes": 142_659_308,
        "errors": [],
    }:
        raise RuntimeError(f"Staging preflight differs from the recorded recovery baseline: {staging_before}")

    checkpoint = checkpoint_template(raw_path, config, staging_before)
    atomic_json(COUNT_CHECKPOINT_PATH, checkpoint)
    checkpoint = json.loads(COUNT_CHECKPOINT_PATH.read_text(encoding="utf-8"))

    invalid_counter: dict[str, int] = {}
    batches, hashing_reader = iter_csv_batches(
        raw_path, SOURCE_FIELDS, config["batch_block_size_bytes"], invalid_counter, True
    )
    started = time.perf_counter()
    parsed_rows = 0
    cumulative_exclusive = {rule: 0 for rule in EXCLUSIVE_RULES}
    cumulative_diagnostics: dict[str, dict[str, int]] = {}
    peak_memory_bytes = None
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None

    for batch_id, batch in enumerate(batches, start=1):
        raw_frame = batch.to_pandas()
        parsed_rows += len(raw_frame)
        batch_exclusive, batch_diagnostics = count_cleaning_batch(raw_frame)
        for rule, count in batch_exclusive.items():
            cumulative_exclusive[rule] += int(count)
        add_nested_counts(cumulative_diagnostics, batch_diagnostics)
        malformed_rows = int(invalid_counter.get("structurally_malformed_rows", 0))
        checkpoint = update_count_checkpoint(
            checkpoint, batch_id, parsed_rows, malformed_rows,
            cumulative_exclusive, cumulative_diagnostics,
        )
        if process is not None:
            rss = int(process.memory_info().rss)
            peak_memory_bytes = rss if peak_memory_bytes is None else max(peak_memory_bytes, rss)
        print(
            f"Count batch {batch_id:03d}: input={len(raw_frame):,}, "
            f"cumulative_input={checkpoint['cumulative_input_rows']:,}, "
            f"cumulative_retained={checkpoint['cumulative_retained']:,}",
            flush=True,
        )
        del raw_frame

    if hashing_reader is None:
        raise RuntimeError("Count-only pass did not use the integrated hashing stream")
    observed_hash = hashing_reader.hexdigest
    hashing_reader.close()
    runtime_seconds = time.perf_counter() - started

    malformed_rows = int(invalid_counter.get("structurally_malformed_rows", 0))
    cumulative_exclusive["malformed_csv_structure"] = malformed_rows
    checkpoint["cumulative_input_rows"] = int(parsed_rows + malformed_rows)
    checkpoint["cumulative_malformed_csv_structure"] = malformed_rows
    for rule in EXCLUSIVE_RULES[1:]:
        checkpoint[f"cumulative_{rule}"] = int(cumulative_exclusive[rule])

    current_stat = raw_path.stat()
    source_integrity = {
        "path_unchanged": raw_path.relative_to(PROJECT_ROOT).as_posix() == checkpoint["source_path"],
        "size_unchanged": current_stat.st_size == checkpoint["source_size"],
        "modification_time_unchanged": datetime.fromtimestamp(current_stat.st_mtime, timezone.utc).isoformat() == checkpoint["source_modification_time"],
        "observed_sha256": observed_hash,
        "expected_sha256": expected_hash,
        "sha256_match": observed_hash == expected_hash,
    }
    staging_after = staging_snapshot()
    staging_immutable = staging_after == staging_before
    reconciliation = reconcile_exclusive_counts(
        checkpoint["cumulative_input_rows"], cumulative_exclusive, staging_before["row_count"]
    )
    all_pass = (
        checkpoint["cumulative_input_rows"] == 14_285_496
        and reconciliation["equation_matches_raw"]
        and reconciliation["retained_matches_expected"]
        and staging_before["row_count"] == 6_471_630
        and staging_immutable
        and not staging_after["errors"]
        and all(source_integrity.values())
    )

    checkpoint.update({
        "observed_source_sha256": observed_hash,
        "partition_manifest_sha256_after": staging_after["manifest_sha256"],
        "staging_snapshot_sha256_after": staging_after["staging_snapshot_sha256"],
        "runtime_seconds": runtime_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "status": "COMPLETE" if all_pass else "FAILED_RECONCILIATION",
        "updated_at": utc_now(),
        "completed_at": utc_now(),
    })
    atomic_json(COUNT_CHECKPOINT_PATH, checkpoint)

    recovery = {
        "authorization": "Prompt 1A-R: exactly one additional complete Raw pass for counting and integrated source verification only",
        "recovery_run_id": checkpoint["recovery_run_id"],
        "source_identity": source_integrity,
        "batch_count": checkpoint["latest_completed_batch_id"],
        "exact_exclusive_counters": {rule: int(checkpoint[f"cumulative_{rule}"]) for rule in EXCLUSIVE_RULES},
        "non_exclusive_diagnostic_counts": checkpoint["non_exclusive_diagnostic_counts"],
        "reconciliation": reconciliation,
        "retained_count_comparison": {
            "count_pass": checkpoint["cumulative_retained"],
            "partition_total": staging_after["row_count"],
            "expected": 6_471_630,
            "all_equal": checkpoint["cumulative_retained"] == staging_after["row_count"] == 6_471_630,
        },
        "staging_immutability": {
            "before": staging_before,
            "after": staging_after,
            "unchanged": staging_immutable,
        },
        "runtime_seconds": runtime_seconds,
        "peak_memory_bytes": peak_memory_bytes,
        "status": "PASS" if all_pass else "FAIL",
    }
    atomic_json(COUNT_RECOVERY_PATH, recovery)
    write_cleaning_report(checkpoint["cumulative_input_rows"], recovery["exact_exclusive_counters"])
    if not all_pass:
        raise RuntimeError(f"Count-only reconciliation failed; saved recovery evidence: {recovery}")
    print(json.dumps(recovery, indent=2))


def stats_to_frame(stats: dict[tuple[str, str], int], raw_rows: int, eligible_rows: int) -> pd.DataFrame:
    order = {
        "structurally_malformed_csv": 1,
        "ineligible_action": 2,
        "required_missing_like": 3,
        "invalid_numeric_parse_or_nonfinite": 4,
        "invalid_numeric_range": 5,
        "invalid_categorical_code": 6,
    }
    rows = []
    for (rule, column), count in sorted(stats.items(), key=lambda item: (order.get(item[0][0], 99), item[0][1])):
        rows.append({
            "rule_order": order.get(rule, 99),
            "rule": rule,
            "column": column,
            "detected_rows": int(count),
            "count_type": "exclusive_rule_total" if column == "__all__" else "per_column_detection_may_overlap",
        })
    removed = sum(row["detected_rows"] for row in rows if row["column"] == "__all__")
    rows.extend([
        {"rule_order": 0, "rule": "population_flow", "column": "raw_rows", "detected_rows": raw_rows, "count_type": "flow_count"},
        {"rule_order": 7, "rule": "population_flow", "column": "eligible_cleaned_rows", "detected_rows": eligible_rows, "count_type": "flow_count"},
        {"rule_order": 8, "rule": "population_flow", "column": "all_exclusive_removals", "detected_rows": removed, "count_type": "flow_count"},
    ])
    return pd.DataFrame(rows).sort_values(["rule_order", "rule", "column"]).reset_index(drop=True)


def write_schema_report() -> None:
    reasons = legacy_rules_report()["retained_field_reasons"]
    fields = []
    for field in ARROW_SCHEMA:
        if field.name == "loan_amount_000s":
            reason = reasons["loan_amount_000s"]
        elif field.name == "respondent_id":
            reason = reasons["respondent_id"]
        elif field.name == "action_taken_code":
            reason = reasons["action_taken_code"]
        elif field.name in {"applicant_ethnicity_name", "co_applicant_ethnicity_name", "applicant_race_name_1", "co_applicant_race_name_1", "applicant_sex_name", "co_applicant_sex_name", "minority_population"}:
            reason = reasons["sensitive_name_fields_and_minority_population"]
        elif field.name in {"msamd_name", "state_name", "state_abbr", "state_code", "county_name", "county_code", "census_tract_number"}:
            reason = reasons["geographic_fields"]
        else:
            reason = reasons["other_base_fields"]
        fields.append({"name": field.name, "type": str(field.type), "nullable": field.nullable, "reason": reason})
    atomic_json(REPORT_DIR / "prompt1a_schema.json", {"schema_digest": schema_digest(), "field_count": len(fields), "fields": fields})


def build_outlier_report() -> None:
    parquet_glob = (STAGING_DIR / "*.parquet").as_posix()
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET memory_limit='4GB'")
    connection.execute(f"SET temp_directory='{TMP_DIR.as_posix()}'")
    probabilities = [0.001, 0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995, 0.999]
    labels = ["p0_1", "p0_5", "p1", "p5", "p25", "median", "p75", "p95", "p99", "p99_5", "p99_9"]
    rows: list[dict[str, Any]] = []
    for column in NUMERIC_FIELDS:
        query = f"""
            SELECT count(*) AS n, min({column}) AS minimum,
                   quantile_cont({column}, {probabilities}) AS qs,
                   max({column}) AS maximum, avg({column}) AS mean,
                   stddev_samp({column}) AS std
            FROM read_parquet(?)
        """
        n, minimum, quantiles, maximum, mean, std = connection.execute(query, [parquet_glob]).fetchone()
        metrics = {"count": n, "minimum": minimum, **dict(zip(labels, quantiles)), "maximum": maximum, "mean": mean, "std": std}
        for metric, value in metrics.items():
            rows.append({"column": column, "scope": "complete_eligible_population", "metric": metric, "value": value})
        equal_one = connection.execute(f"SELECT count(*) FROM read_parquet(?) WHERE {column} = 1", [parquet_glob]).fetchone()[0]
        if column in {"loan_amount_000s", "applicant_income_000s"}:
            rows.append({"column": column, "scope": "complete_eligible_population", "metric": "count_equal_1", "value": equal_one})
            top = connection.execute(
                f"SELECT count(*), min(v), median(v), max(v), avg(v), stddev_samp(v) FROM (SELECT {column} AS v FROM read_parquet(?) ORDER BY {column} DESC LIMIT 100)",
                [parquet_glob],
            ).fetchone()
            for metric, value in zip(["count", "minimum", "median", "maximum", "mean", "std"], top):
                rows.append({"column": column, "scope": "largest_100_values", "metric": metric, "value": value})
    connection.close()
    pd.DataFrame(rows).to_csv(REPORT_DIR / "prompt1a_outlier_review.csv", index=False)


def run_smoke(max_rows: int) -> None:
    if not 1 <= max_rows <= 25_000:
        raise ValueError("Smoke test max_rows must be between 1 and 25,000")
    config = load_config()
    raw_path = discover_raw_csv(DATA_DIR)
    validate_header(read_header(raw_path))
    invalid = {}
    batches, stream = iter_csv_batches(raw_path, SOURCE_FIELDS, min(config["batch_block_size_bytes"], 16 * 1024 * 1024), invalid, False)
    frames = []
    remaining = max_rows
    for batch in batches:
        take = min(remaining, batch.num_rows)
        frames.append(batch.slice(0, take).to_pandas())
        remaining -= take
        if remaining == 0:
            break
    if stream:
        stream.close()
    raw = pd.concat(frames, ignore_index=True)
    cleaned, _ = clean_batch(raw)
    table = frame_to_table(cleaned)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    smoke_path = TMP_DIR / "prompt1a_smoke.parquet"
    pq.write_table(table, smoke_path, compression="zstd")
    reloaded = pq.read_table(smoke_path)
    if reloaded.schema != ARROW_SCHEMA or reloaded.num_rows != table.num_rows:
        raise RuntimeError("Smoke Parquet reload or schema check failed")
    smoke_path.unlink()
    report = {
        "status": "PASS",
        "max_rows": max_rows,
        "input_rows": len(raw),
        "retained_rows": len(cleaned),
        "schema_digest": schema_digest(),
        "parquet_reload": True,
        "temporary_file_removed": not smoke_path.exists(),
        "completed_at_utc": utc_now(),
    }
    atomic_json(REPORT_DIR / "prompt1a_smoke_test.json", report)
    print(json.dumps(report, indent=2))


def run_full() -> None:
    started = time.perf_counter()
    config = load_config()
    raw_path = discover_raw_csv(DATA_DIR)
    header = read_header(raw_path)
    validate_header(header)
    stat = raw_path.stat()
    if stat.st_size != config["source_size_bytes"]:
        raise RuntimeError("Raw source size differs from the protected baseline")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT_DIR / "legacy_cleaning_rules.json", legacy_rules_report())
    atomic_json(REPORT_DIR / "prompt1a_source_report.json", source_report(raw_path, config))
    write_schema_report()

    manifest_path = REPORT_DIR / "prompt1a_partition_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    if existing_manifest and all(entry.get("completion_status") == "COMPLETE" for entry in existing_manifest):
        expected = sorted(STAGING_DIR.glob("part-*.parquet"))
        if len(expected) == len(existing_manifest):
            all_valid = True
            for entry in existing_manifest:
                path = PROJECT_ROOT / entry["output_path"]
                try:
                    metadata = pq.read_metadata(path)
                    all_valid &= (
                        path.stat().st_size == entry["output_size"]
                        and file_sha256(path) == entry["output_sha256"]
                        and schema_digest(pq.read_schema(path)) == entry["schema_digest"]
                        and metadata.num_rows == entry["retained_row_count"]
                    )
                except Exception:
                    all_valid = False
            required = [
                REPORT_DIR / "prompt1a_cleaning_report.csv",
                REPORT_DIR / "prompt1a_outlier_review.csv",
                REPORT_DIR / "prompt1a_schema.json",
            ]
            if all_valid and all(path.exists() for path in required):
                print("All completed partitions and core reports validate; full raw scan skipped.")
                return

    existing_by_id = {int(item["partition_id"]): item for item in existing_manifest if item.get("completion_status") == "COMPLETE"}
    manifest: list[dict[str, Any]] = []
    stats: dict[tuple[str, str], int] = defaultdict(int)
    invalid_counter: dict[str, int] = {}
    batches, hashing_reader = iter_csv_batches(
        raw_path, SOURCE_FIELDS, config["batch_block_size_bytes"], invalid_counter, True
    )
    source_row = 1
    retained_total = 0
    for partition_id, batch in enumerate(batches, start=1):
        raw_frame = batch.to_pandas()
        input_count = len(raw_frame)
        cleaned, batch_stats = clean_batch(raw_frame)
        aggregate_stats(stats, batch_stats)
        table = frame_to_table(cleaned)
        retained_total += table.num_rows
        source_end = source_row + input_count - 1

        existing = existing_by_id.get(partition_id)
        reuse = False
        if existing:
            path = PROJECT_ROOT / existing["output_path"]
            reuse = (
                existing["source_batch_range"] == f"{source_row}-{source_end}"
                and existing["input_row_count"] == input_count
                and existing["retained_row_count"] == table.num_rows
                and path.exists()
                and path.stat().st_size == existing["output_size"]
                and file_sha256(path) == existing["output_sha256"]
                and schema_digest(pq.read_schema(path)) == existing["schema_digest"]
                and pq.read_metadata(path).num_rows == table.num_rows
            )
        details = existing if reuse else write_partition(table, partition_id)
        entry = {
            "partition_id": partition_id,
            "source_batch_range": f"{source_row}-{source_end}",
            "input_row_count": input_count,
            "retained_row_count": table.num_rows,
            "output_path": details["output_path"],
            "output_size": details["output_size"],
            "output_sha256": details["output_sha256"],
            "schema_digest": details["schema_digest"],
            "completion_status": "COMPLETE",
        }
        manifest.append(entry)
        atomic_json(manifest_path, manifest)
        print(f"Partition {partition_id:03d}: input={input_count:,}, retained={table.num_rows:,}, reused={reuse}", flush=True)
        source_row = source_end + 1
        del raw_frame, cleaned, table

    if hashing_reader is None:
        raise RuntimeError("Full processing pass did not create a hashing reader")
    processing_hash = hashing_reader.hexdigest
    hashing_reader.close()
    malformed = invalid_counter.get("structurally_malformed_rows", 0)
    _record(stats, "structurally_malformed_csv", "__all__", malformed)
    raw_rows = (source_row - 1) + malformed
    hash_checkpoint = {
        "baseline_sha256": config["source_sha256"],
        "processing_pass_sha256": processing_hash,
        "match": processing_hash == config["source_sha256"],
    }
    atomic_json(REPORT_DIR / "prompt1a_processing_hash_checkpoint.json", hash_checkpoint)
    if not hash_checkpoint["match"]:
        raise RuntimeError("Raw source SHA-256 changed during processing")

    current_stat = raw_path.stat()
    if current_stat.st_size != config["source_size_bytes"]:
        raise RuntimeError("Raw source size changed during processing")
    final_source_report = source_report(raw_path, config, processing_hash)
    atomic_json(REPORT_DIR / "prompt1a_source_report.json", final_source_report)

    cleaning = stats_to_frame(stats, raw_rows, retained_total)
    cleaning.to_csv(REPORT_DIR / "prompt1a_cleaning_report.csv", index=False)
    build_outlier_report()

    elapsed = time.perf_counter() - started
    storage = sum(item["output_size"] for item in manifest)
    run_summary = {
        "started_at_utc": datetime.fromtimestamp(time.time() - elapsed, timezone.utc).isoformat(),
        "completed_at_utc": utc_now(),
        "runtime_seconds": elapsed,
        "raw_row_count": raw_rows,
        "eligible_population_count": retained_total,
        "partition_count": len(manifest),
        "staging_storage_bytes": storage,
        "source_reads_in_workflow": 2,
        "source_read_note": "One baseline hash pass and one processing pass with an integrated end hash.",
    }
    atomic_json(REPORT_DIR / "prompt1a_run_summary.json", run_summary)

    for extra in STAGING_DIR.glob("part-*.parquet"):
        try:
            part_id = int(extra.stem.split("-")[1])
        except ValueError:
            continue
        if part_id > len(manifest):
            raise RuntimeError(f"Unexpected stale staging partition: {extra.name}")

    for child in TMP_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(json.dumps(run_summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--max-rows", type=int, default=25_000)
    subparsers.add_parser("full")
    subparsers.add_parser("count-only")
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "smoke":
        run_smoke(args.max_rows)
    elif args.command == "count-only":
        run_count_only()
    else:
        run_full()


if __name__ == "__main__":
    main()
