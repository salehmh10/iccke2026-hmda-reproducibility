"""Focused tests for the frozen Prompt 1B data build."""

from __future__ import annotations

import ast
import json
import shutil
import sys
import unittest
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from feature_engineering import ENGINEERED_FEATURES, NUMERIC_ENGINEERED_FEATURES, engineer_features
from final_dataset_builder import (
    AUDIT_ONLY_FIELDS,
    LEGACY_KEY_FIELDS,
    NUMERIC_FIELDS,
    ROW_HASH_FIELDS,
    STAGING_FIELDS,
    canonical_sql,
    duplicate_safe_deciles,
    equality_sql,
    feature_roles,
    hash_sql,
    make_split_assignments,
)


def sample_row(target: float = 200.0) -> dict:
    row = {field: f"value_{field}" for field in STAGING_FIELDS}
    for field in NUMERIC_FIELDS:
        row[field] = 25.0
    row.update({
        "loan_amount_000s": target,
        "applicant_income_000s": 100.0,
        "population": 2000.0,
        "minority_population": 50.0,
        "hud_median_family_income": 50_000.0,
        "tract_to_msamd_income": 80.0,
        "number_of_owner_occupied_units": 400.0,
        "number_of_1_to_4_family_units": 500.0,
        "action_taken_code": 1,
        "co_applicant_sex_name": "No co-applicant",
        "loan_type_name": "FHA-insured",
        "state_name": "New York",
    })
    return row


class FinalDatasetBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = PROJECT_ROOT / "outputs" / "tmp" / "test_final_dataset_build"
        cls.tmp.mkdir(parents=True, exist_ok=True)
        candidates = pd.DataFrame({
            "row_hash": [f"{index:064x}" for index in range(575_000)],
            "loan_amount_000s": (np.arange(575_000) % 1000 + 1).astype("float64"),
        })
        cls.assignments = make_split_assignments(candidates)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmp.exists():
            shutil.rmtree(cls.tmp)

    def test_frozen_field_order_and_hash_field_order(self) -> None:
        self.assertEqual(len(STAGING_FIELDS), 30)
        self.assertEqual(ROW_HASH_FIELDS, [field for field in STAGING_FIELDS if field != "loan_amount_000s"])
        self.assertEqual(LEGACY_KEY_FIELDS, [field for field in STAGING_FIELDS if field not in {"state_abbr", "action_taken_code"}])
        self.assertLess(canonical_sql(STAGING_FIELDS).find("respondent_id"), canonical_sql(STAGING_FIELDS).find("agency_name"))

    def test_record_and_row_hash_are_deterministic(self) -> None:
        frame = pd.DataFrame([sample_row(), sample_row()])
        connection = duckdb.connect()
        connection.register("rows", frame)
        record = connection.execute(f"SELECT {hash_sql(STAGING_FIELDS)} FROM rows").fetchall()
        row_hash = connection.execute(f"SELECT {hash_sql(ROW_HASH_FIELDS)} FROM rows").fetchall()
        self.assertEqual(record[0], record[1])
        self.assertEqual(row_hash[0], row_hash[1])
        self.assertEqual(len(record[0][0]), 64)

    def test_target_changes_record_hash_but_not_row_hash(self) -> None:
        frame = pd.DataFrame([sample_row(200.0), sample_row(201.0)])
        connection = duckdb.connect()
        connection.register("rows", frame)
        result = connection.execute(
            f"SELECT {hash_sql(STAGING_FIELDS)}, {hash_sql(ROW_HASH_FIELDS)} FROM rows"
        ).fetchall()
        self.assertNotEqual(result[0][0], result[1][0])
        self.assertEqual(result[0][1], result[1][1])

    def test_duplicate_group_spanning_partitions_uses_exact_fields(self) -> None:
        schema = pa.Table.from_pylist([sample_row()]).schema
        part_a = self.tmp / "part-a.parquet"
        part_b = self.tmp / "part-b.parquet"
        pq.write_table(pa.Table.from_pylist([sample_row()], schema=schema), part_a)
        pq.write_table(pa.Table.from_pylist([sample_row(), sample_row(300.0)], schema=schema), part_b)
        connection = duckdb.connect()
        fields = ", ".join(f'"{field}"' for field in STAGING_FIELDS)
        groups = connection.execute(
            f"SELECT count(*) n FROM read_parquet(['{part_a.as_posix()}','{part_b.as_posix()}']) GROUP BY {fields} ORDER BY n DESC"
        ).fetchall()
        self.assertEqual(groups, [(2,), (1,)])

    def test_contradictory_group_is_removed_in_full(self) -> None:
        rows = pd.DataFrame([sample_row(200.0), sample_row(201.0), {**sample_row(300.0), "respondent_id": "other"}])
        connection = duckdb.connect()
        connection.register("rows", rows)
        row_fields = ", ".join(f'"{field}"' for field in ROW_HASH_FIELDS)
        connection.execute(f"CREATE TABLE keyed AS SELECT *, {hash_sql(ROW_HASH_FIELDS)} row_hash FROM rows")
        connection.execute(f"CREATE TABLE conflicts AS SELECT {row_fields}, row_hash FROM keyed GROUP BY {row_fields}, row_hash HAVING count(DISTINCT loan_amount_000s)>1")
        match = equality_sql(ROW_HASH_FIELDS, "k", "c")
        kept = connection.execute(f"SELECT loan_amount_000s FROM keyed k WHERE NOT EXISTS (SELECT 1 FROM conflicts c WHERE {match})").fetchall()
        self.assertEqual(kept, [(300.0,)])

    def test_legacy_normalization_and_antijoin(self) -> None:
        current = pd.DataFrame([{field: sample_row()[field] for field in LEGACY_KEY_FIELDS}])
        legacy = current.copy()
        legacy["respondent_id"] = "  " + legacy["respondent_id"] + "  "
        legacy["loan_amount_000s"] = legacy["loan_amount_000s"].astype(str)
        for field in NUMERIC_FIELDS & set(LEGACY_KEY_FIELDS):
            legacy[field] = legacy[field].astype(str)
        connection = duckdb.connect()
        connection.register("current", current)
        connection.register("legacy_raw", legacy)
        expressions = [
            f'CAST(trim("{field}") AS DOUBLE) AS "{field}"' if field in NUMERIC_FIELDS
            else f'CAST(trim("{field}") AS VARCHAR) AS "{field}"'
            for field in LEGACY_KEY_FIELDS
        ]
        connection.execute(f"CREATE TABLE legacy AS SELECT {', '.join(expressions)} FROM legacy_raw")
        match = equality_sql(LEGACY_KEY_FIELDS, "c", "l")
        count = connection.execute(f"SELECT count(*) FROM current c WHERE NOT EXISTS (SELECT 1 FROM legacy l WHERE {match})").fetchone()[0]
        self.assertEqual(count, 0)

    def test_selection_score_is_deterministic(self) -> None:
        connection = duckdb.connect()
        hashes = pd.DataFrame({"row_hash": ["b" * 64, "a" * 64, "c" * 64]})
        connection.register("rows", hashes)
        query = "SELECT row_hash FROM rows ORDER BY sha256('regression_v2_seed_42' || row_hash), row_hash"
        self.assertEqual(connection.execute(query).fetchall(), connection.execute(query).fetchall())

    def test_candidate_and_split_sizes_are_exact(self) -> None:
        counts = self.assignments["development_role"].value_counts().to_dict()
        self.assertEqual(len(self.assignments), 575_000)
        self.assertEqual(counts, {"train": 400_000, "validation": 100_000, "iid": 75_000})

    def test_split_memberships_are_disjoint(self) -> None:
        grouped = self.assignments.groupby("row_hash")["development_role"].nunique()
        self.assertEqual(int((grouped != 1).sum()), 0)
        self.assertEqual(self.assignments["row_hash"].nunique(), 575_000)

    def test_duplicate_safe_deciles_keep_equal_targets_together(self) -> None:
        target = pd.Series(np.repeat(np.arange(1, 101), 10))
        bins = duplicate_safe_deciles(target)
        self.assertEqual(pd.DataFrame({"target": target, "bin": bins}).groupby("target")["bin"].nunique().max(), 1)

    def test_all_16_engineered_formulas(self) -> None:
        result = engineer_features(pd.DataFrame([sample_row()]))
        self.assertEqual(len(ENGINEERED_FEATURES), 16)
        self.assertTrue(set(ENGINEERED_FEATURES).issubset(result.columns))
        self.assertAlmostEqual(result.loc[0, "log1p_applicant_income"], np.log1p(100.0))
        self.assertAlmostEqual(result.loc[0, "log1p_population"], np.log1p(2000.0))
        self.assertAlmostEqual(result.loc[0, "log1p_hud_median_family_income"], np.log1p(50_000.0))
        self.assertAlmostEqual(result.loc[0, "log1p_owner_occupied_units"], np.log1p(400.0))
        self.assertAlmostEqual(result.loc[0, "log1p_1_to_4_family_units"], np.log1p(500.0))
        self.assertAlmostEqual(result.loc[0, "applicant_income_to_area_income"], 2.0)
        self.assertAlmostEqual(result.loc[0, "tract_income_ratio"], 0.8)
        self.assertAlmostEqual(result.loc[0, "owner_occupied_unit_ratio"], 0.8)
        self.assertAlmostEqual(result.loc[0, "family_units_per_1000_people"], 250.0)
        self.assertAlmostEqual(result.loc[0, "owner_occupied_units_per_1000_people"], 200.0)
        self.assertEqual(result.loc[0, "has_co_applicant"], 0)
        self.assertEqual(result.loc[0, "loan_program_group"], "Government backed")
        self.assertEqual(result.loc[0, "applicant_income_area_group"], "High")
        self.assertEqual(result.loc[0, "tract_income_level"], "Low")
        self.assertEqual(result.loc[0, "us_region"], "Northeast")
        self.assertEqual(result.loc[0, "majority_minority_tract"], "Majority minority")

    def test_division_by_zero_has_no_infinity(self) -> None:
        row = sample_row()
        row["hud_median_family_income"] = 0.0
        row["population"] = 0.0
        row["number_of_1_to_4_family_units"] = 0.0
        result = engineer_features(pd.DataFrame([row]))
        values = result[["applicant_income_to_area_income", "owner_occupied_unit_ratio", "family_units_per_1000_people", "owner_occupied_units_per_1000_people"]].to_numpy(dtype="float64")
        self.assertFalse(np.isinf(values).any())
        self.assertTrue(np.isnan(values).all())

    def test_valid_engineered_numeric_values_are_finite(self) -> None:
        result = engineer_features(pd.DataFrame([sample_row()]))
        self.assertTrue(np.isfinite(result[NUMERIC_ENGINEERED_FEATURES].to_numpy(dtype="float64")).all())

    def test_feature_roles_exclude_sensitive_target_audit_and_lender(self) -> None:
        superset = STAGING_FIELDS + ["record_hash", "row_hash"] + ENGINEERED_FEATURES + ["development_role"]
        roles = feature_roles(superset)["contracts"]
        no_lender = set(roles["main_without_sensitive_without_lender"])
        with_lender = set(roles["main_without_sensitive_with_lender"])
        self.assertNotIn("respondent_id", no_lender)
        self.assertIn("respondent_id", with_lender)
        self.assertFalse(set(AUDIT_ONLY_FIELDS) & no_lender)
        self.assertFalse(set(AUDIT_ONLY_FIELDS) & with_lender)
        self.assertNotIn("loan_amount_000s", no_lender)
        self.assertNotIn("applicant_sex_name", no_lender)

    def test_iid_target_absence_and_key_order_equality(self) -> None:
        features = pa.table({"row_hash": ["a", "b"], "feature": [1.0, 2.0]})
        targets = pa.table({"row_hash": ["a", "b"], "loan_amount_000s": [10.0, 20.0]})
        feature_path = self.tmp / "features.parquet"
        target_path = self.tmp / "targets.parquet"
        pq.write_table(features, feature_path, compression="zstd")
        pq.write_table(targets, target_path, compression="zstd")
        reloaded_features = pq.read_table(feature_path)
        reloaded_targets = pq.read_table(target_path)
        self.assertNotIn("loan_amount_000s", reloaded_features.schema.names)
        self.assertEqual(reloaded_targets.schema.names, ["row_hash", "loan_amount_000s"])
        self.assertTrue(reloaded_features["row_hash"].equals(reloaded_targets["row_hash"]))
        self.assertEqual(reloaded_features.num_rows, 2)

    def test_raw_and_model_fit_counts_are_zero_by_static_contract(self) -> None:
        tree = ast.parse((SRC_DIR / "final_dataset_builder.py").read_text(encoding="utf-8"))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in {"fit", "fit_predict", "predict", "train"}:
                    calls.append(name)
        self.assertEqual(calls, [])
        self.assertNotIn("hmda_2017_nationwide_all-records_labels.csv", (SRC_DIR / "final_dataset_builder.py").read_text(encoding="utf-8"))

    def test_no_random_root_tmp_folder(self) -> None:
        root_tmp = [path.name for path in PROJECT_ROOT.iterdir() if path.is_dir() and "tmp" in path.name.lower()]
        self.assertEqual(root_tmp, [])


if __name__ == "__main__":
    unittest.main()
