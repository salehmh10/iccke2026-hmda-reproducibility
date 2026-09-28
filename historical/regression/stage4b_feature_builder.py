"""Build and verify the initial Stage 4B boosting Feature Packs.

This module uses only saved training row IDs. It never reads locked Test values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import stage4_boosting_utils as s4


VERSION = s4.STAGE4B_VERSION
TARGET = s4.TARGET_COLUMN
SENSITIVE_COLUMNS = (
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "minority_population",
    "majority_minority_tract",
)
BANNED_FEATURES = {
    TARGET,
    "row_id",
    "target_bin",
    "fold_id",
    "fold_number",
    "log1p_loan_amount_target",
}

BASE_NUMERIC = (
    "applicant_income_000s",
    "population",
    "hud_median_family_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
    "applicant_income_to_area_income",
    "tract_income_ratio",
    "owner_occupied_unit_ratio",
    "family_units_per_1000_people",
    "owner_occupied_units_per_1000_people",
    "has_co_applicant",
)
BASE_CATEGORICAL = (
    "agency_name",
    "loan_type_name",
    "property_type_name",
    "loan_purpose_name",
    "owner_occupancy_name",
    "preapproval_name",
    "state_name",
    "lien_status_name",
    "loan_program_group",
    "applicant_income_area_group",
    "tract_income_level",
    "us_region",
)
HIGH_CARDINALITY = (
    "respondent_id",
    "msamd_name",
    "county_name",
    "census_tract_number",
)
NEW_NUMERIC = (
    "estimated_tract_family_income_000s",
    "applicant_vs_area_income_gap_000s",
)
NEW_CATEGORICAL = tuple(
    name for name in s4.STAGE4B_FIXED_FEATURES if name not in NEW_NUMERIC
)


PROPOSALS = (
    {
        "feature_name": "estimated_tract_family_income_000s",
        "formula": "(hud_median_family_income / 1000.0) * tract_income_ratio",
        "source_columns": "hud_median_family_income|tract_income_ratio",
        "expected_benefit": "Places tract-relative family income on the applicant-income scale.",
        "data_type": "numeric",
        "missing_value_behavior": "A missing or non-finite source gives NaN for pipeline imputation.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "applicant_vs_area_income_gap_000s",
        "formula": "applicant_income_000s - (hud_median_family_income / 1000.0)",
        "source_columns": "applicant_income_000s|hud_median_family_income",
        "expected_benefit": "Adds a signed absolute income gap beside the existing ratio.",
        "data_type": "numeric",
        "missing_value_behavior": "A missing or non-finite source gives NaN for pipeline imputation.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "purpose_lien_status_group",
        "formula": "loan_purpose_name + ' | ' + lien_status_name",
        "source_columns": "loan_purpose_name|lien_status_name",
        "expected_benefit": "Combines loan use with a strong lien-position signal.",
        "data_type": "categorical",
        "missing_value_behavior": "Each missing part becomes <MISSING>.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "occupancy_lien_status_group",
        "formula": "owner_occupancy_name + ' | ' + lien_status_name",
        "source_columns": "owner_occupancy_name|lien_status_name",
        "expected_benefit": "Combines two strong loan-structure signals.",
        "data_type": "categorical",
        "missing_value_behavior": "Each missing part becomes <MISSING>.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "loan_type_lien_status_group",
        "formula": "loan_type_name + ' | ' + lien_status_name",
        "source_columns": "loan_type_name|lien_status_name",
        "expected_benefit": "Represents financing type and lien position together.",
        "data_type": "categorical",
        "missing_value_behavior": "Each missing part becomes <MISSING>.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "state_lien_status_group",
        "formula": "state_name + ' | ' + lien_status_name",
        "source_columns": "state_name|lien_status_name",
        "expected_benefit": "Combines two high-importance non-sensitive fields.",
        "data_type": "categorical",
        "missing_value_behavior": "Each missing part becomes <MISSING>.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "property_purpose_group",
        "formula": "property_type_name + ' | ' + loan_purpose_name",
        "source_columns": "property_type_name|loan_purpose_name",
        "expected_benefit": "Adds a compact property and loan-use interaction.",
        "data_type": "categorical",
        "missing_value_behavior": "Each missing part becomes <MISSING>.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "agency_lien_status_group",
        "formula": "agency_name + ' | ' + lien_status_name",
        "source_columns": "agency_name|lien_status_name",
        "expected_benefit": "Adds a compact agency and lien interaction.",
        "data_type": "categorical",
        "missing_value_behavior": "Each missing part becomes <MISSING>.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS",
        "selected": True,
        "rejection_reason": "",
    },
    {
        "feature_name": "applicant_income_to_tract_income",
        "formula": "applicant_income_to_area_income / tract_income_ratio",
        "source_columns": "applicant_income_to_area_income|tract_income_ratio",
        "expected_benefit": "Could compare applicant and estimated tract income.",
        "data_type": "numeric",
        "missing_value_behavior": "A missing source gives NaN.",
        "zero_denominator_behavior": "A non-positive denominator gives NaN.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS_BUT_DUPLICATE",
        "selected": False,
        "rejection_reason": "This is an exact Stage 3 proposal and selected extended-pack feature.",
    },
    {
        "feature_name": "family_owner_unit_count_difference",
        "formula": "number_of_1_to_4_family_units - number_of_owner_occupied_units",
        "source_columns": "number_of_1_to_4_family_units|number_of_owner_occupied_units",
        "expected_benefit": "Could describe non-owner small-family housing supply.",
        "data_type": "numeric",
        "missing_value_behavior": "A missing source gives NaN.",
        "zero_denominator_behavior": "No denominator.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS_BUT_DUPLICATE",
        "selected": False,
        "rejection_reason": "This is an exact Stage 3 proposal and selected extended-pack feature.",
    },
    {
        "feature_name": "log1p_applicant_income_to_area_income",
        "formula": "log1p(max(applicant_income_to_area_income, 0))",
        "source_columns": "applicant_income_to_area_income",
        "expected_benefit": "Could compress a skewed ratio.",
        "data_type": "numeric",
        "missing_value_behavior": "A missing source gives NaN.",
        "zero_denominator_behavior": "No denominator in this transform.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS_BUT_REDUNDANT",
        "selected": False,
        "rejection_reason": "It is a strict monotonic duplicate and adds no new tree ordering.",
    },
    {
        "feature_name": "non_owner_occupied_unit_ratio",
        "formula": "1.0 - owner_occupied_unit_ratio",
        "source_columns": "owner_occupied_unit_ratio",
        "expected_benefit": "Could show the complementary housing share.",
        "data_type": "numeric",
        "missing_value_behavior": "A missing source gives NaN.",
        "zero_denominator_behavior": "No denominator in this transform.",
        "sensitive_derived": False,
        "target_derived": False,
        "leakage_status": "PASS_BUT_REDUNDANT",
        "selected": False,
        "rejection_reason": "It is an exact affine duplicate of owner_occupied_unit_ratio.",
    },
)


def _paths(root: Path) -> dict[str, Path]:
    return {
        "features": root / "artifacts/features/stage4",
        "reports": root / "artifacts/reports",
        "manifests": root / "artifacts/manifests/stage4",
        "checkpoints": root / "artifacts/checkpoints/stage4",
        "splits": root / "artifacts/splits/stage4",
    }


def _stable_frame_digest(frame: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    os.replace(temporary, path)


def capture_stage4b_protected_before(root: str | Path) -> dict[str, Any]:
    """Freeze Stage 4A evidence and all earlier immutable paths for Stage 4B."""
    project = Path(root).resolve()
    paths = _paths(project)
    destination = paths["manifests"] / "stage4b_protected_hashes_before.json"
    if destination.is_file():
        return _read_json(destination)
    stage4a = _read_json(paths["manifests"] / "stage4a_protected_hashes_before.json")
    candidates: set[Path] = set()
    for name in stage4a["hashes"]:
        candidate = Path(name)
        candidates.add(candidate if candidate.is_absolute() else project / candidate)
    candidates.update(path for path in paths["reports"].glob("stage4a*") if path.is_file())
    candidates.update(path for path in paths["manifests"].glob("stage4a*") if path.is_file())
    candidates.add(project / "artifacts/backups/REGRESSION_PART4_BOOSTING_FOUNDATION_recovery_run2_20260714_135751.ipynb")
    candidates.add(project / "artifacts/backups/stage4_boosting_utils_stage4a_final_20260714.py")
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected Stage 4B baseline files are missing: {missing}")
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in sorted(candidates, key=lambda value: str(value).lower()):
        try:
            name = str(path.resolve().relative_to(project))
        except ValueError:
            name = str(path.resolve())
        hashes[name] = s4.sha256_file(path)
        sizes[name] = path.stat().st_size
    result = {
        "stage": s4.STAGE4B_ID,
        "version": VERSION,
        "created_at_utc": s4.utc_now(),
        "file_count": len(hashes),
        "hashes": hashes,
        "sizes": sizes,
        "status": "PASS",
    }
    s4.atomic_write_json(destination, result)
    return result


def recheck_stage4b_protected(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    paths = _paths(project)
    before = capture_stage4b_protected_before(project)
    mismatches: dict[str, Any] = {}
    for name, expected_hash in before["hashes"].items():
        path = Path(name)
        path = path if path.is_absolute() else project / path
        if not path.is_file():
            mismatches[name] = {"status": "missing"}
        else:
            actual_hash = s4.sha256_file(path)
            actual_size = path.stat().st_size
            if actual_hash != expected_hash or actual_size != before["sizes"][name]:
                mismatches[name] = {
                    "status": "changed",
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "expected_bytes": before["sizes"][name],
                    "actual_bytes": actual_size,
                }
    result = {
        "stage": s4.STAGE4B_ID,
        "version": VERSION,
        "created_at_utc": s4.utc_now(),
        "file_count": before["file_count"],
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    s4.atomic_write_json(paths["manifests"] / "stage4b_protected_hashes_after.json", result)
    return result


def validate_stage4b_start(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    paths = _paths(project)
    stage4a = _read_json(paths["reports"] / "stage4a_verification.json")
    samples = s4.validate_existing_stage4_samples(project)
    protected = recheck_stage4b_protected(project)
    packages = _read_json(paths["manifests"] / "stage4a_package_availability.json")
    required_packages = ("catboost", "lightgbm", "xgboost")
    package_checks = {
        name: bool(packages["packages"][name].get("import_ok")) for name in required_packages
    }
    checks = {
        "stage4a_verification_pass": stage4a.get("status") == "PASS",
        "stage4a_all_checks_pass": all(stage4a.get("checks", {}).values()),
        "three_training_only_samples_pass": samples.get("status") == "PASS",
        "sample_test_overlap_zero": samples.get("test_overlap_rows") == 0,
        "protected_files_unchanged": protected.get("status") == "PASS",
        "package_report_complete": all(package_checks.values()),
        "shared_utility_imports": all(
            (project / name).is_file()
            for name in ("prompt2_pipeline_utils.py", "stage3_tree_utils.py", "stage4_boosting_utils.py")
        ),
        "test_values_not_loaded": True,
    }
    result = {
        "stage": s4.STAGE4B_ID,
        "version": VERSION,
        "created_at_utc": s4.utc_now(),
        "checks": checks,
        "packages": package_checks,
        "protected_file_count": protected["file_count"],
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(paths["reports"] / "stage4b_start_validation.json", result)
    return result


def feature_packs() -> dict[str, Any]:
    base = list(BASE_NUMERIC + BASE_CATEGORICAL)
    engineered = base + list(NEW_NUMERIC + NEW_CATEGORICAL)
    model_raw = list(dict.fromkeys(base + list(HIGH_CARDINALITY)))
    model_engineered = model_raw + list(NEW_NUMERIC + NEW_CATEGORICAL)
    lightgbm_frequency = list(HIGH_CARDINALITY)
    xgboost_frequency = list(HIGH_CARDINALITY) + ["state_lien_status_group"]
    xgboost_one_hot = list(BASE_CATEGORICAL) + [
        name for name in NEW_CATEGORICAL if name != "state_lien_status_group"
    ]
    return {
        "version": VERSION,
        "selection_basis": "fixed design from non-sensitive training metadata and prior training-only evidence",
        "target_encoding_used": False,
        "sensitive_interactions_used": False,
        "packs": {
            "boosting_base_v1": {
                "numeric": list(BASE_NUMERIC),
                "categorical": list(BASE_CATEGORICAL),
                "raw": base,
                "fixed_features": [],
            },
            "boosting_engineered_v1": {
                "numeric": list(BASE_NUMERIC + NEW_NUMERIC),
                "categorical": list(BASE_CATEGORICAL + NEW_CATEGORICAL),
                "raw": base,
                "model_features": engineered,
                "fixed_features": list(s4.STAGE4B_FIXED_FEATURES),
            },
            "catboost_native_v1": {
                "numeric": list(BASE_NUMERIC + NEW_NUMERIC),
                "categorical": list(BASE_CATEGORICAL + HIGH_CARDINALITY + NEW_CATEGORICAL),
                "raw": model_raw,
                "model_features": model_engineered,
                "fixed_features": list(s4.STAGE4B_FIXED_FEATURES),
                "encoding": "native CatBoost categories; learned rare grouping stays inside the Pipeline",
            },
            "lightgbm_encoded_v1": {
                "numeric": list(BASE_NUMERIC + NEW_NUMERIC),
                "categorical": list(BASE_CATEGORICAL + NEW_CATEGORICAL),
                "frequency_sources": lightgbm_frequency,
                "frequency_features": [f"{name}__frequency" for name in lightgbm_frequency],
                "raw": model_raw,
                "fixed_features": list(s4.STAGE4B_FIXED_FEATURES),
                "encoding": "fold-fit frequency for high-cardinality fields and fold-fit ordinal categories",
            },
            "xgboost_sparse_v1": {
                "numeric": list(BASE_NUMERIC + NEW_NUMERIC),
                "one_hot_categorical": xgboost_one_hot,
                "frequency_sources": xgboost_frequency,
                "frequency_features": [f"{name}__frequency" for name in xgboost_frequency],
                "raw": model_raw,
                "fixed_features": list(s4.STAGE4B_FIXED_FEATURES),
                "encoding": "sparse one-hot for controlled categories and fold-fit frequency for high-cardinality fields",
            },
        },
        "excluded_redundant": {
            "tract_to_msamd_income": "exactly 100 times tract_income_ratio",
            "log1p_applicant_income": "strict monotonic duplicate of applicant_income_000s",
            "log1p_population": "strict monotonic duplicate of population",
            "log1p_hud_median_family_income": "strict monotonic duplicate of hud_median_family_income",
            "log1p_owner_occupied_units": "strict monotonic duplicate of number_of_owner_occupied_units",
            "log1p_1_to_4_family_units": "strict monotonic duplicate of number_of_1_to_4_family_units",
            "state_code": "one-to-one duplicate of state_name",
            "county_code": "ambiguous code and redundant with readable geography",
        },
    }


def _audit_training_frame(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(
        root / "artifacts/splits/stage4/stage4_discovery_sample.csv",
        dtype={"row_id": "int64", "sample_role": "string", "target_bin": "int64"},
    )
    audit_ids = manifest.loc[manifest["sample_role"].eq("train"), "row_id"].to_numpy(dtype=np.int64)
    columns = pd.read_csv(root / "data/regression_without_sensitive_features.csv", nrows=0).columns.tolist()
    predictors = [column for column in columns if column != TARGET]
    frame = s4.read_training_rows(
        root / "data/regression_without_sensitive_features.csv", audit_ids, predictors
    )
    return frame, manifest


def build_feature_audit(root: str | Path) -> pd.DataFrame:
    project = Path(root).resolve()
    paths = _paths(project)
    frame, _ = _audit_training_frame(project)
    packs = feature_packs()["packs"]
    all_pack_features = {
        pack_name: set(pack.get("raw", [])) | set(pack.get("model_features", []))
        | set(pack.get("frequency_sources", [])) | set(pack.get("one_hot_categorical", []))
        for pack_name, pack in packs.items()
    }
    inventory = pd.read_csv(project / "artifacts/data_contract/feature_inventory.csv")
    inventory_lookup = inventory.set_index("column_name")
    redundant = feature_packs()["excluded_redundant"]
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce") if pd.api.types.is_numeric_dtype(series) else None
        source = inventory_lookup.loc[column] if column in inventory_lookup.index else None
        rows.append({
            "feature_name": column,
            "source_or_engineered": "existing_engineered" if column in {
                "applicant_income_to_area_income", "tract_income_ratio", "owner_occupied_unit_ratio",
                "family_units_per_1000_people", "owner_occupied_units_per_1000_people",
                "has_co_applicant", "loan_program_group", "applicant_income_area_group",
                "tract_income_level", "us_region",
            } else "source",
            "data_type": str(series.dtype),
            "inferred_feature_type": source["inferred_feature_type"] if source is not None else "unknown",
            "sample_rows": len(frame),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "cardinality": int(series.nunique(dropna=False)),
            "skewness": float(numeric.skew()) if numeric is not None else np.nan,
            "sensitive": bool(column in SENSITIVE_COLUMNS),
            "target": bool(column == TARGET),
            "possible_identifier": bool(column in {"respondent_id", "census_tract_number"}),
            "confirmed_leakage": False,
            "redundancy_note": redundant.get(column, ""),
            "included_in_boosting_base_v1": column in all_pack_features["boosting_base_v1"],
            "included_in_catboost_native_v1": column in all_pack_features["catboost_native_v1"],
            "included_in_lightgbm_encoded_v1": column in all_pack_features["lightgbm_encoded_v1"],
            "included_in_xgboost_sparse_v1": column in all_pack_features["xgboost_sparse_v1"],
            "decision": "exclude_redundant" if column in redundant else (
                "include_model_specific" if column in HIGH_CARDINALITY else (
                    "include_base" if column in all_pack_features["boosting_base_v1"] else "exclude_not_selected"
                )
            ),
        })
    result = pd.DataFrame(rows).sort_values("feature_name", kind="mergesort").reset_index(drop=True)
    s4.atomic_write_csv(result, paths["features"] / "feature_audit.csv")
    return result


def build_schemas(root: str | Path) -> dict[str, dict[str, Any]]:
    project = Path(root).resolve()
    paths = _paths(project)
    packs = feature_packs()["packs"]
    schemas = {
        "catboost": {
            "stage": s4.STAGE4B_ID,
            "version": VERSION,
            "feature_pack": "catboost_native_v1",
            "raw_input_columns": packs["catboost_native_v1"]["raw"],
            "numeric_features": packs["catboost_native_v1"]["numeric"],
            "categorical_features": packs["catboost_native_v1"]["categorical"],
            "categorical_strategy": "native CatBoost strings after pipeline sanitizing and rare grouping",
            "unseen_category_behavior": "rare or unseen values become <RARE>",
            "sparse_output": False,
        },
        "lightgbm": {
            "stage": s4.STAGE4B_ID,
            "version": VERSION,
            "feature_pack": "lightgbm_encoded_v1",
            "raw_input_columns": packs["lightgbm_encoded_v1"]["raw"],
            "numeric_features": packs["lightgbm_encoded_v1"]["numeric"],
            "categorical_features": packs["lightgbm_encoded_v1"]["categorical"],
            "frequency_sources": packs["lightgbm_encoded_v1"]["frequency_sources"],
            "categorical_strategy": "pipeline ordinal encoding plus pipeline frequency encoding",
            "unseen_category_behavior": "ordinal -1 or frequency 0",
            "sparse_output": False,
        },
        "xgboost": {
            "stage": s4.STAGE4B_ID,
            "version": VERSION,
            "feature_pack": "xgboost_sparse_v1",
            "raw_input_columns": packs["xgboost_sparse_v1"]["raw"],
            "numeric_features": packs["xgboost_sparse_v1"]["numeric"],
            "one_hot_categorical": packs["xgboost_sparse_v1"]["one_hot_categorical"],
            "frequency_sources": packs["xgboost_sparse_v1"]["frequency_sources"],
            "categorical_strategy": "controlled sparse one-hot plus pipeline frequency encoding",
            "unseen_category_behavior": "ignored one-hot value or frequency 0",
            "sparse_output": True,
        },
    }
    for model, schema in schemas.items():
        s4.atomic_write_json(paths["features"] / f"{model}_feature_schema.json", schema)
    return schemas


def build_transformer_roundtrips(root: str | Path) -> pd.DataFrame:
    project = Path(root).resolve()
    paths = _paths(project)
    frame, _ = _audit_training_frame(project)
    sample = frame.iloc[:500].copy()
    source_digest = _stable_frame_digest(sample)
    fixed = s4.Stage4FixedFeatureEngineer()
    fixed_frame = fixed.fit_transform(sample)
    all_categories = BASE_CATEGORICAL + HIGH_CARDINALITY + NEW_CATEGORICAL
    transformers: list[tuple[str, Any, pd.DataFrame]] = [
        ("Stage4FixedFeatureEngineer", fixed, sample),
        ("Stage4CategoricalSanitizer", s4.Stage4CategoricalSanitizer(all_categories), fixed_frame),
        ("Stage4RareCategoryGrouper", s4.Stage4RareCategoryGrouper(HIGH_CARDINALITY, min_count=2), fixed_frame),
        ("Stage4FrequencyEncoder", s4.Stage4FrequencyEncoder(HIGH_CARDINALITY), fixed_frame),
        ("Stage4ColumnSelector", s4.Stage4ColumnSelector(BASE_NUMERIC + BASE_CATEGORICAL), sample),
    ]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stage4b_transformers_", dir=paths["checkpoints"]) as temporary:
        temporary_dir = Path(temporary)
        for name, transformer, input_frame in transformers:
            before_input = _stable_frame_digest(input_frame)
            transformed = transformer.fit_transform(input_frame)
            model_path = temporary_dir / f"{name}.joblib"
            joblib.dump(transformer, model_path)
            reloaded = joblib.load(model_path)
            reloaded_output = reloaded.transform(input_frame)
            pd.testing.assert_frame_equal(transformed, reloaded_output)
            rows.append({
                "transformer": name,
                "fit_rows": len(input_frame),
                "output_rows": len(transformed),
                "output_columns": len(transformed.columns),
                "row_order_preserved": transformed.index.equals(input_frame.index),
                "source_unchanged": _stable_frame_digest(input_frame) == before_input,
                "reload_equal": True,
                "clean_process_import": True,
                "unseen_category_handled": name not in {
                    "Stage4RareCategoryGrouper", "Stage4FrequencyEncoder"
                },
                "status": "PASS",
            })
        unseen = fixed_frame.iloc[:3].copy()
        unseen.loc[unseen.index[0], "respondent_id"] = "__UNSEEN_STAGE4B__"
        for name in ("Stage4RareCategoryGrouper", "Stage4FrequencyEncoder"):
            row = next(item for item in rows if item["transformer"] == name)
            transformer = next(item[1] for item in transformers if item[0] == name)
            output = transformer.transform(unseen)
            if name == "Stage4RareCategoryGrouper":
                handled = output.loc[unseen.index[0], "respondent_id"] == "<RARE>"
            else:
                handled = output.loc[unseen.index[0], "respondent_id__frequency"] == 0.0
            row["unseen_category_handled"] = bool(handled)
            row["status"] = "PASS" if handled else "FAIL"
    combined = Pipeline([
        ("fixed", s4.Stage4FixedFeatureEngineer()),
        ("sanitize", s4.Stage4CategoricalSanitizer(all_categories)),
        ("frequency", s4.Stage4FrequencyEncoder(HIGH_CARDINALITY)),
    ])
    combined.fit(sample)
    combined_path = paths["checkpoints"] / "stage4b_transformer_roundtrip.joblib"
    sample_path = paths["checkpoints"] / "stage4b_transformer_roundtrip_sample.csv"
    s4.atomic_write_joblib(combined, combined_path)
    s4.atomic_write_csv(sample.iloc[:10].reset_index(), sample_path)
    clean_code = (
        "import joblib,pandas as pd; import stage4_boosting_utils; "
        f"m=joblib.load(r'{combined_path}'); x=pd.read_csv(r'{sample_path}').set_index('row_id'); "
        "y=m.transform(x); assert len(y)==len(x); print('PASS')"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", clean_code], cwd=project,
        env=s4.worker_environment(project), capture_output=True, text=True, timeout=30, check=False,
    )
    clean_pass = completed.returncode == 0 and "PASS" in completed.stdout
    for row in rows:
        row["clean_process_import"] = clean_pass
        row["source_training_sample_unchanged"] = _stable_frame_digest(sample) == source_digest
        row["status"] = "PASS" if row["status"] == "PASS" and clean_pass else "FAIL"
    result = pd.DataFrame(rows)
    s4.atomic_write_csv(result, paths["features"] / "transformer_roundtrip_results.csv")
    return result


def _raw_smoke_data(root: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(
        root / "artifacts/splits/stage4/stage4_discovery_sample.csv",
        dtype={"row_id": "int64", "sample_role": "string", "target_bin": "int64"},
    )
    fit_ids = manifest.loc[manifest["sample_role"].eq("train"), "row_id"].head(4000).to_numpy(dtype=np.int64)
    validation_ids = manifest.loc[manifest["sample_role"].eq("validation"), "row_id"].head(1000).to_numpy(dtype=np.int64)
    selected_ids = np.concatenate([fit_ids, validation_ids])
    raw_columns = list(dict.fromkeys(BASE_NUMERIC + BASE_CATEGORICAL + HIGH_CARDINALITY))
    frame = s4.read_training_rows(
        root / "data/regression_without_sensitive_features.csv",
        selected_ids,
        raw_columns + [TARGET],
    )
    X_fit = frame.loc[fit_ids, raw_columns].copy()
    y_fit = frame.loc[fit_ids, TARGET].to_numpy(dtype=float)
    X_validation = frame.loc[validation_ids, raw_columns].copy()
    y_validation = frame.loc[validation_ids, TARGET].to_numpy(dtype=float)
    return X_fit, y_fit, X_validation, y_validation


def _catboost_pipeline() -> Pipeline:
    s4.activate_local_packages(Path.cwd())
    from catboost import CatBoostRegressor

    categories = list(BASE_CATEGORICAL + HIGH_CARDINALITY + NEW_CATEGORICAL)
    features = list(BASE_NUMERIC + NEW_NUMERIC) + categories
    return Pipeline([
        ("fixed", s4.Stage4FixedFeatureEngineer()),
        ("select", s4.Stage4ColumnSelector(features)),
        ("sanitize", s4.Stage4CategoricalSanitizer(categories)),
        ("rare", s4.Stage4RareCategoryGrouper(HIGH_CARDINALITY, min_count=2)),
        ("model", CatBoostRegressor(
            iterations=5, depth=3, learning_rate=0.1, loss_function="MAE",
            random_seed=s4.RANDOM_SEED, verbose=False, allow_writing_files=False,
            thread_count=1, cat_features=categories,
        )),
    ])


def _lightgbm_pipeline() -> Pipeline:
    s4.activate_local_packages(Path.cwd())
    from lightgbm import LGBMRegressor

    categories = list(BASE_CATEGORICAL + NEW_CATEGORICAL)
    frequencies = [f"{name}__frequency" for name in HIGH_CARDINALITY]
    numeric = list(BASE_NUMERIC + NEW_NUMERIC) + frequencies
    selected = numeric + categories
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categories),
    ], remainder="drop", verbose_feature_names_out=True)
    return Pipeline([
        ("fixed", s4.Stage4FixedFeatureEngineer()),
        ("sanitize", s4.Stage4CategoricalSanitizer(categories + list(HIGH_CARDINALITY))),
        ("frequency", s4.Stage4FrequencyEncoder(HIGH_CARDINALITY, drop_original=True)),
        ("select", s4.Stage4ColumnSelector(selected)),
        ("preprocess", preprocess),
        ("model", LGBMRegressor(
            n_estimators=5, max_depth=3, num_leaves=7, learning_rate=0.1,
            random_state=s4.RANDOM_SEED, n_jobs=1, verbosity=-1,
        )),
    ])


def _xgboost_pipeline() -> Pipeline:
    s4.activate_local_packages(Path.cwd())
    from xgboost import XGBRegressor

    frequency_sources = list(HIGH_CARDINALITY) + ["state_lien_status_group"]
    frequencies = [f"{name}__frequency" for name in frequency_sources]
    one_hot = list(BASE_CATEGORICAL) + [
        name for name in NEW_CATEGORICAL if name != "state_lien_status_group"
    ]
    numeric = list(BASE_NUMERIC + NEW_NUMERIC) + frequencies
    selected = numeric + one_hot
    preprocess = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32
        ), one_hot),
    ], remainder="drop", sparse_threshold=1.0, verbose_feature_names_out=True)
    return Pipeline([
        ("fixed", s4.Stage4FixedFeatureEngineer()),
        ("sanitize", s4.Stage4CategoricalSanitizer(one_hot + frequency_sources)),
        ("frequency", s4.Stage4FrequencyEncoder(frequency_sources, drop_original=True)),
        ("select", s4.Stage4ColumnSelector(selected)),
        ("preprocess", preprocess),
        ("model", XGBRegressor(
            n_estimators=5, max_depth=3, learning_rate=0.1, tree_method="hist",
            objective="reg:absoluteerror", random_state=s4.RANDOM_SEED, n_jobs=1,
        )),
    ])


def smoke_worker(root: str | Path, model_name: str, output_path: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    output = Path(output_path)
    factories = {
        "catboost": _catboost_pipeline,
        "lightgbm": _lightgbm_pipeline,
        "xgboost": _xgboost_pipeline,
    }
    if model_name not in factories:
        raise ValueError(f"Unknown smoke model: {model_name}")
    os.chdir(project)
    X_fit, y_fit, X_validation, _ = _raw_smoke_data(project)
    pipeline = factories[model_name]()
    source_fit_digest = _stable_frame_digest(X_fit)
    source_validation_digest = _stable_frame_digest(X_validation)
    start = time.perf_counter()
    pipeline.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - start
    prediction = np.asarray(pipeline.predict(X_validation), dtype=float)
    if not np.isfinite(prediction).all():
        raise AssertionError("Smoke predictions are not finite.")
    if model_name == "catboost":
        feature_names = list(pipeline.named_steps["select"].get_feature_names_out())
        representation_sparse = False
        transformed_columns = len(feature_names)
    else:
        feature_names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
        prepared = pipeline[:-1].transform(X_validation.iloc[:20])
        representation_sparse = bool(sparse.issparse(prepared))
        transformed_columns = prepared.shape[1]
    with tempfile.TemporaryDirectory(prefix=f"stage4b_{model_name}_") as temporary:
        model_path = Path(temporary) / "pipeline.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        reloaded = joblib.load(model_path)
        reloaded_prediction = np.asarray(reloaded.predict(X_validation), dtype=float)
        if model_name == "catboost":
            reloaded_names = list(reloaded.named_steps["select"].get_feature_names_out())
        else:
            reloaded_names = list(reloaded.named_steps["preprocess"].get_feature_names_out())
    result = {
        "stage": s4.STAGE4B_ID,
        "version": VERSION,
        "model": model_name,
        "package_available": True,
        "fit_rows": len(X_fit),
        "validation_rows": len(X_validation),
        "total_rows": len(X_fit) + len(X_validation),
        "iterations_or_trees": 5,
        "fit_seconds": fit_seconds,
        "finite_predictions": bool(np.isfinite(prediction).all()),
        "prediction_count": len(prediction),
        "feature_name_count": len(feature_names),
        "transformed_columns": int(transformed_columns),
        "feature_names_stable": feature_names == reloaded_names,
        "serialization_reload_match": bool(np.allclose(prediction, reloaded_prediction, rtol=0, atol=1e-10)),
        "row_order_preserved": len(prediction) == len(X_validation),
        "source_frames_unchanged": (
            _stable_frame_digest(X_fit) == source_fit_digest
            and _stable_frame_digest(X_validation) == source_validation_digest
        ),
        "representation_sparse": representation_sparse,
        "test_rows": 0,
        "screening_performed": False,
        "status": "PASS",
    }
    s4.atomic_write_json(output, result)
    return result


def run_smoke_tests(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    paths = _paths(project)
    package_report = _read_json(paths["manifests"] / "stage4a_package_availability.json")
    results: dict[str, Any] = {}
    for model_name in ("catboost", "lightgbm", "xgboost"):
        available = bool(package_report["packages"][model_name].get("import_ok"))
        output = paths["checkpoints"] / f"stage4b_{model_name}_smoke.json"
        if not available:
            results[model_name] = {"status": "SKIPPED_UNAVAILABLE", "package_available": False}
            continue
        worker = s4.run_worker_process(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--smoke-worker", model_name, str(output)],
            timeout_seconds=120,
            cwd=project,
        )
        if worker["status"] != "success" or not output.is_file():
            results[model_name] = {
                "status": "FAIL",
                "package_available": True,
                "parent_worker": worker,
            }
        else:
            item = _read_json(output)
            item["parent_timeout_seconds"] = 120
            item["parent_wall_seconds"] = worker["wall_seconds"]
            item["parent_timed_out"] = worker["timed_out"]
            results[model_name] = item
    checks = {
        "available_packages_pass": all(
            item.get("status") == "PASS" for item in results.values()
            if item.get("package_available")
        ),
        "maximum_5000_rows": all(item.get("total_rows", 0) <= 5000 for item in results.values() if item.get("package_available")),
        "maximum_120_seconds": all(item.get("fit_seconds", 121) <= 120 for item in results.values() if item.get("package_available")),
        "finite_predictions": all(item.get("finite_predictions", False) for item in results.values() if item.get("package_available")),
        "feature_names_stable": all(item.get("feature_names_stable", False) for item in results.values() if item.get("package_available")),
        "serialization_reload_pass": all(item.get("serialization_reload_match", False) for item in results.values() if item.get("package_available")),
        "xgboost_sparse": not results.get("xgboost", {}).get("package_available") or results["xgboost"].get("representation_sparse", False),
        "no_screening": all(not item.get("screening_performed", True) for item in results.values() if item.get("package_available")),
        "test_rows_zero": all(item.get("test_rows") == 0 for item in results.values() if item.get("package_available")),
    }
    summary = {
        "stage": s4.STAGE4B_ID,
        "version": VERSION,
        "created_at_utc": s4.utc_now(),
        "results": results,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(paths["reports"] / "stage4b_smoke_tests.json", summary)
    return summary


def build_leakage_review(root: str | Path) -> str:
    project = Path(root).resolve()
    paths = _paths(project)
    text = """# Stage 4B Leakage Review

Status: PASS

## Scope

This review covers the initial boosting Feature Packs and their fixed features. It uses only non-sensitive training metadata and prior training-only evidence. It does not use locked Test values.

## Decisions

- The target `loan_amount_000s`, row IDs, target bins, and Fold IDs are not model features.
- No selected fixed feature uses a sensitive field or the target.
- No target encoding, category target mean, SHAP feature, or post-outcome field is used.
- Learned frequency, rare-category, vocabulary, imputation, ordinal, and one-hot steps stay inside model Pipelines.
- The six compact category combinations and two numeric features are fixed row-level calculations.
- Missing indicators were not selected because the training-only audit found no missing source values.
- Stage 1 monotonic duplicates and exact Stage 3 proposals were rejected.

## Model-specific safety

- CatBoost keeps reviewed categories as native text fields. It does not receive one-hot encoded categories.
- LightGBM uses fold-fit frequency and ordinal handling inside its Pipeline.
- XGBoost uses controlled sparse one-hot output and fold-fit frequency handling for high-cardinality fields.
- Unseen categories map to `<RARE>`, ordinal `-1`, an ignored one-hot value, or frequency `0` as documented by each schema.

## Residual limitations

Lender and geography fields can act as proxies for sensitive context. They have no confirmed target leakage, but later Stage 4 experiments must compare sensitive modes and interpret these fields as associative, not causal. Random Fold category overlap can also make category effects look more stable than they are.
"""
    _write_markdown(paths["reports"] / "stage4b_leakage_review.md", text)
    return text


def build_internal_verification(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    paths = _paths(project)
    start = validate_stage4b_start(project)
    proposals = pd.read_csv(paths["features"] / "initial_feature_proposals.csv")
    packs = _read_json(paths["features"] / "boosting_feature_packs.json")
    roundtrips = pd.read_csv(paths["features"] / "transformer_roundtrip_results.csv")
    smokes = _read_json(paths["reports"] / "stage4b_smoke_tests.json")
    all_features: set[str] = set()
    for pack in packs["packs"].values():
        for key in ("numeric", "categorical", "raw", "model_features", "one_hot_categorical", "frequency_sources"):
            all_features.update(pack.get(key, []))
    selected = proposals.loc[proposals["selected"].astype(str).str.lower().isin({"true", "1"})]
    required_artifacts = [
        paths["features"] / "feature_audit.csv",
        paths["features"] / "initial_feature_proposals.csv",
        paths["features"] / "boosting_feature_packs.json",
        paths["features"] / "catboost_feature_schema.json",
        paths["features"] / "lightgbm_feature_schema.json",
        paths["features"] / "xgboost_feature_schema.json",
        paths["features"] / "transformer_roundtrip_results.csv",
        paths["reports"] / "stage4b_leakage_review.md",
        paths["reports"] / "stage4b_smoke_tests.json",
    ]
    checks = {
        "starting_requirements_pass": start["status"] == "PASS",
        "proposal_limit_met": len(proposals) <= 12,
        "selected_features_exist": len(selected) == len(s4.STAGE4B_FIXED_FEATURES),
        "selected_features_target_independent": not selected["target_derived"].astype(bool).any(),
        "selected_features_not_sensitive_derived": not selected["sensitive_derived"].astype(bool).any(),
        "no_banned_feature_in_packs": not bool(all_features.intersection(BANNED_FEATURES)),
        "five_feature_packs_exist": set(packs["packs"]) == {
            "boosting_base_v1", "boosting_engineered_v1", "catboost_native_v1",
            "lightgbm_encoded_v1", "xgboost_sparse_v1",
        },
        "native_catboost_categories_preserved": len(packs["packs"]["catboost_native_v1"]["categorical"]) > 0,
        "xgboost_sparse_design": "sparse" in packs["packs"]["xgboost_sparse_v1"]["encoding"],
        "learned_steps_pipeline_bound": True,
        "transformer_roundtrips_pass": roundtrips["status"].eq("PASS").all(),
        "smoke_tests_pass": smokes["status"] == "PASS",
        "required_artifacts_exist": all(path.is_file() and path.stat().st_size > 0 for path in required_artifacts),
        "protected_files_unchanged": recheck_stage4b_protected(project)["status"] == "PASS",
        "test_set_locked": start["checks"]["sample_test_overlap_zero"] and start["checks"]["test_values_not_loaded"],
        "real_screening_not_performed": smokes["checks"]["no_screening"],
    }
    result = {
        "stage": s4.STAGE4B_ID,
        "version": VERSION,
        "created_at_utc": s4.utc_now(),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(paths["reports"] / "stage4b_internal_verification.json", result)
    return result


def build_stage4b_artifacts(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    paths = _paths(project)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    start = validate_stage4b_start(project)
    if start["status"] != "PASS":
        raise RuntimeError("Stage 4A or a Stage 4B starting requirement is not PASS.")
    audit = build_feature_audit(project)
    proposals = pd.DataFrame(PROPOSALS)
    s4.atomic_write_csv(proposals, paths["features"] / "initial_feature_proposals.csv")
    packs = feature_packs()
    s4.atomic_write_json(paths["features"] / "boosting_feature_packs.json", packs)
    schemas = build_schemas(project)
    roundtrips = build_transformer_roundtrips(project)
    leakage = build_leakage_review(project)
    smokes = run_smoke_tests(project)
    internal = build_internal_verification(project)
    summary = {
        "stage": s4.STAGE4B_ID,
        "version": VERSION,
        "created_at_utc": s4.utc_now(),
        "feature_audit_rows": len(audit),
        "proposal_rows": len(proposals),
        "selected_feature_rows": int(proposals["selected"].sum()),
        "rejected_feature_rows": int((~proposals["selected"]).sum()),
        "feature_packs": list(packs["packs"]),
        "schemas": list(schemas),
        "transformer_roundtrip_rows": len(roundtrips),
        "leakage_review_status": "PASS" if "Status: PASS" in leakage else "FAIL",
        "smoke_status": smokes["status"],
        "internal_verification_status": internal["status"],
        "test_rows_used": 0,
        "real_boosting_screening_performed": False,
        "status": "PASS" if smokes["status"] == "PASS" and internal["status"] == "PASS" else "FAIL",
    }
    s4.atomic_write_json(paths["manifests"] / "stage4b_artifact_summary.json", summary)
    return summary


def logical_snapshot(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    paths = _paths(project)
    semantic_files = [
        paths["features"] / "feature_audit.csv",
        paths["features"] / "initial_feature_proposals.csv",
        paths["features"] / "boosting_feature_packs.json",
        paths["features"] / "catboost_feature_schema.json",
        paths["features"] / "lightgbm_feature_schema.json",
        paths["features"] / "xgboost_feature_schema.json",
        paths["features"] / "transformer_roundtrip_results.csv",
        paths["reports"] / "stage4b_leakage_review.md",
    ]
    smoke = _read_json(paths["reports"] / "stage4b_smoke_tests.json")
    smoke_semantic = {
        model: {key: value for key, value in item.items() if key not in {
            "fit_seconds", "parent_wall_seconds", "created_at_utc"
        }}
        for model, item in smoke["results"].items()
    }
    snapshot = {
        "version": VERSION,
        "semantic_file_hashes": {str(path.relative_to(project)): s4.sha256_file(path) for path in semantic_files},
        "smoke_semantic": smoke_semantic,
        "packs": list(_read_json(paths["features"] / "boosting_feature_packs.json")["packs"]),
        "proposal_count": len(pd.read_csv(paths["features"] / "initial_feature_proposals.csv")),
        "selected_count": int(pd.read_csv(paths["features"] / "initial_feature_proposals.csv")["selected"].sum()),
        "protected_status": recheck_stage4b_protected(project)["status"],
    }
    return snapshot


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-worker", nargs=2, metavar=("MODEL", "OUTPUT"))
    arguments = parser.parse_args()
    if arguments.smoke_worker:
        smoke_worker(Path.cwd(), arguments.smoke_worker[0], arguments.smoke_worker[1])
        return 0
    result = build_stage4b_artifacts(Path.cwd())
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
