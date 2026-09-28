"""Focused Prompt 1A cleaning tests using the standard library runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
import ast
import hashlib
import inspect
import json
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_cleaning import (  # noqa: E402
    ALLOWED_CODES, ARROW_SCHEMA, EXCLUSIVE_RULES, NUMERIC_FIELDS, RETAINED_FIELDS,
    SOURCE_FIELDS, clean_batch, count_cleaning_batch, deterministic_partition_name,
    frame_to_table, is_missing_like, reconcile_exclusive_counts,
    update_count_checkpoint,
    run_count_only,
)
import data_cleaning  # noqa: E402


def valid_row() -> dict[str, str]:
    row = {field: "value" for field in SOURCE_FIELDS}
    row.update({
        "respondent_id": " 00123 ", "agency_name": " CFPB ",
        "loan_type_name": "Conventional", "property_type_name": "One-to-four family",
        "loan_purpose_name": "Home purchase", "owner_occupancy_name": "Owner occupied",
        "preapproval_name": "Not applicable", "action_taken": "1", "msamd_name": "Metro",
        "msamd": "12345", "state_name": "New York", "state_abbr": "NY", "state_code": "36",
        "county_name": "Albany", "county_code": "001", "census_tract_number": "0025.00",
        "applicant_ethnicity_name": "Information not provided",
        "co_applicant_ethnicity_name": "Not applicable", "applicant_race_name_1": "White",
        "co_applicant_race_name_1": "No co-applicant", "applicant_sex_name": "Female",
        "co_applicant_sex_name": "No co-applicant", "lien_status_name": "Secured by first lien",
        "loan_amount_000s": "250", "applicant_income_000s": "80", "population": "3500",
        "minority_population": "0", "hud_median_family_income": "70000",
        "tract_to_msamd_income": "95.5", "number_of_owner_occupied_units": "1000",
        "number_of_1_to_4_family_units": "1500",
    })
    for column, values in ALLOWED_CODES.items():
        if column != "action_taken":
            row[column] = sorted(values)[0]
    return row


class RawCleaningTests(unittest.TestCase):
    def clean_rows(self, rows):
        return clean_batch(pd.DataFrame(rows))[0]

    def test_missing_like_normalization(self):
        series = pd.Series([" ", " NaN ", "N/A", ".", "--", "data"])
        self.assertEqual(is_missing_like(series).tolist(), [True, True, True, True, True, False])

    def test_meaningful_text_labels_are_preserved(self):
        row = valid_row()
        output = self.clean_rows([row])
        self.assertEqual(len(output), 1)
        self.assertEqual(output.loc[0, "applicant_ethnicity_name"], "Information not provided")
        self.assertEqual(output.loc[0, "co_applicant_sex_name"], "No co-applicant")

    def test_positive_value_enforcement_and_extreme_preservation(self):
        zero = valid_row(); zero["loan_amount_000s"] = "0"
        negative = valid_row(); negative["applicant_income_000s"] = "-1"
        extreme = valid_row(); extreme["loan_amount_000s"] = "999999999999"
        output = self.clean_rows([zero, negative, extreme])
        self.assertEqual(len(output), 1)
        self.assertEqual(output.loc[0, "loan_amount_000s"], 999999999999.0)

    def test_minority_population_boundaries(self):
        rows = []
        for value in ["0", "100", "-0.1", "100.1"]:
            row = valid_row(); row["minority_population"] = value; rows.append(row)
        output = self.clean_rows(rows)
        self.assertEqual(output["minority_population"].tolist(), [0.0, 100.0])

    def test_nonfinite_and_parse_failure_rejection(self):
        rows = []
        for value in ["inf", "-inf", "NaN", "bad", "25"]:
            row = valid_row(); row["population"] = value; rows.append(row)
        output = self.clean_rows(rows)
        self.assertEqual(len(output), 1)

    def test_categorical_allowed_codes_and_optional_missing(self):
        good = valid_row()
        bad = valid_row(); bad["agency_code"] = "4"
        optional_missing = valid_row(); optional_missing["agency_code"] = " -- "
        output = self.clean_rows([good, bad, optional_missing])
        self.assertEqual(len(output), 2)

    def test_population_filter(self):
        rows = []
        for action in ["1", "2", "8", "3", "7", ""]:
            row = valid_row(); row["action_taken"] = action; rows.append(row)
        output = self.clean_rows(rows)
        self.assertEqual(output["action_taken_code"].tolist(), [1, 2, 8])

    def test_string_trimming_and_identifier_preservation(self):
        output = self.clean_rows([valid_row()])
        self.assertEqual(output.loc[0, "respondent_id"], "00123")
        self.assertEqual(output.loc[0, "census_tract_number"], "0025.00")

    def test_stable_numeric_dtypes_and_schema(self):
        output = self.clean_rows([valid_row()])
        for column in NUMERIC_FIELDS:
            self.assertEqual(str(output[column].dtype), "float64")
        self.assertEqual(str(output["action_taken_code"].dtype), "int8")
        self.assertEqual(frame_to_table(output).schema, ARROW_SCHEMA)
        self.assertEqual(list(output.columns), RETAINED_FIELDS)
        self.assertNotIn("loan_approved", output.columns)

    def test_required_column_check(self):
        row = valid_row(); del row["msamd"]
        with self.assertRaises(KeyError):
            clean_batch(pd.DataFrame([row]))

    def test_deterministic_partition_naming(self):
        self.assertEqual(deterministic_partition_name(1), "part-000001.parquet")
        self.assertEqual(deterministic_partition_name(42), "part-000042.parquet")
        with self.assertRaises(ValueError):
            deterministic_partition_name(0)

    def test_parquet_reload(self):
        table = frame_to_table(self.clean_rows([valid_row()]))
        tmp_parent = ROOT / "outputs" / "tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent) as folder:
            path = Path(folder) / "test.parquet"
            pq.write_table(table, path, compression="zstd")
            reloaded = pq.read_table(path)
            self.assertEqual(reloaded.schema, ARROW_SCHEMA)
            self.assertEqual(reloaded.num_rows, 1)

    def test_exclusive_first_failure_order_and_no_double_counting(self):
        ineligible = valid_row(); ineligible["action_taken"] = "3"; ineligible["population"] = "bad"
        missing = valid_row(); missing["msamd"] = " -- "; missing["population"] = "bad"
        parse = valid_row(); parse["population"] = "bad"; parse["agency_code"] = "4"
        numeric_range = valid_row(); numeric_range["population"] = "0"; numeric_range["agency_code"] = "4"
        category = valid_row(); category["agency_code"] = "4"
        retained = valid_row()
        rows = [ineligible, missing, parse, numeric_range, category, retained]
        exclusive, _ = count_cleaning_batch(pd.DataFrame(rows))
        self.assertEqual(exclusive, {
            "malformed_csv_structure": 0,
            "ineligible_action": 1,
            "required_missing_like": 1,
            "invalid_numeric_parse_or_nonfinite": 1,
            "invalid_numeric_range": 1,
            "invalid_categorical_code": 1,
            "retained": 1,
        })
        self.assertEqual(sum(exclusive.values()), len(rows))

    def test_per_column_diagnostics_can_overlap_without_exclusive_double_count(self):
        row = valid_row()
        row["population"] = "0"
        row["loan_amount_000s"] = "0"
        exclusive, diagnostics = count_cleaning_batch(pd.DataFrame([row]))
        self.assertEqual(exclusive["invalid_numeric_range"], 1)
        self.assertEqual(diagnostics["invalid_numeric_range"]["population"], 1)
        self.assertEqual(diagnostics["invalid_numeric_range"]["loan_amount_000s"], 1)

    def test_checkpoint_atomic_write_reload_and_cumulative_update(self):
        checkpoint = {
            "latest_completed_batch_id": 0,
            "cumulative_input_rows": 0,
            **{f"cumulative_{rule}": 0 for rule in EXCLUSIVE_RULES},
            "non_exclusive_diagnostic_counts": {},
            "updated_at": "",
        }
        exclusive = {rule: 0 for rule in EXCLUSIVE_RULES}
        exclusive["ineligible_action"] = 2
        exclusive["retained"] = 3
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs" / "tmp") as folder:
            target = Path(folder) / "checkpoint.json"
            with mock.patch.object(data_cleaning, "COUNT_CHECKPOINT_PATH", target):
                updated = update_count_checkpoint(
                    checkpoint, 1, 5, 0, exclusive,
                    {"invalid_numeric_range": {"population": 2}},
                )
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), updated)
            self.assertEqual(updated["latest_completed_batch_id"], 1)
            self.assertEqual(updated["cumulative_input_rows"], 5)
            self.assertEqual(updated["cumulative_ineligible_action"], 2)
            self.assertEqual(updated["cumulative_retained"], 3)

    def test_reconciliation_and_retained_comparison(self):
        counts = {rule: 0 for rule in EXCLUSIVE_RULES}
        counts.update({"ineligible_action": 4, "required_missing_like": 3, "retained": 3})
        result = reconcile_exclusive_counts(10, counts, 3)
        self.assertTrue(result["equation_matches_raw"])
        self.assertTrue(result["retained_matches_expected"])
        self.assertFalse(reconcile_exclusive_counts(11, counts, 3)["equation_matches_raw"])

    def test_correct_64_character_source_hash(self):
        config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        expected = "dd35f6a877c5882bbe7260ce65ca842b18ca6aa16514cba256feeeb316d4b7c3"
        self.assertEqual(config["source_sha256"], expected)
        self.assertEqual(len(config["source_sha256"]), 64)

    def test_count_batch_performs_no_parquet_write_and_staging_is_unchanged(self):
        manifest_path = ROOT / "outputs" / "reports" / "prompt1a_partition_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_partition = ROOT / manifest[0]["output_path"]
        before = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            hashlib.sha256(first_partition.read_bytes()).hexdigest(),
            first_partition.stat().st_size,
        )
        with mock.patch.object(data_cleaning.pq, "write_table", side_effect=AssertionError("Parquet write attempted")):
            exclusive, _ = count_cleaning_batch(pd.DataFrame([valid_row()]))
        after = (
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            hashlib.sha256(first_partition.read_bytes()).hexdigest(),
            first_partition.stat().st_size,
        )
        self.assertEqual(exclusive["retained"], 1)
        self.assertEqual(before, after)

    def test_count_only_mode_reuses_clean_batch_and_has_no_partition_writer(self):
        counter_source = inspect.getsource(count_cleaning_batch)
        runner_source = inspect.getsource(run_count_only)
        self.assertIn("clean_batch(raw_frame)", counter_source)
        self.assertNotIn("write_partition(", runner_source)
        self.assertNotIn("pq.write_table", runner_source)

    def test_no_random_root_tmp_and_no_model_fit_call(self):
        root_tmp = [path for path in ROOT.iterdir() if path.is_dir() and "tmp" in path.name.lower()]
        self.assertEqual(root_tmp, [])
        prohibited = []
        for source_path in (ROOT / "src").glob("*.py"):
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in {"fit", "fit_predict", "train_test_split"}:
                    prohibited.append((source_path.name, node.lineno, name))
        self.assertEqual(prohibited, [])


if __name__ == "__main__":
    unittest.main()
