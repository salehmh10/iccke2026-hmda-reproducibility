"""Focused, fit-free tests for Regression V2 Prompt 2."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from metrics import compute_regression_metrics, inverse_target, tail_membership, transform_target
from model_bundles import ModelBundle, atomic_joblib_dump, load_bundle
from preprocessing import (
    TrainFittedCategoricalEncoder,
    build_linear_compact_v2,
    make_xgb_preprocessor,
)
from prompt2_modeling import (
    deterministic_candidate_id,
    freeze_feature_contracts,
    load_feature_roles,
    select_family_candidate,
    validate_data_ready,
)


NUMERIC_MODEL_FIELDS = {
    "applicant_income_000s",
    "population",
    "hud_median_family_income",
    "tract_to_msamd_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
    "log1p_applicant_income",
    "log1p_population",
    "log1p_hud_median_family_income",
    "log1p_owner_occupied_units",
    "log1p_1_to_4_family_units",
    "applicant_income_to_area_income",
    "tract_income_ratio",
    "owner_occupied_unit_ratio",
    "family_units_per_1000_people",
    "owner_occupied_units_per_1000_people",
    "has_co_applicant",
}


def _fit_free_linear_components():
    """Return importable bundle components without calling a model fit."""
    preprocessor = FunctionTransformer(validate=False)
    model = LinearRegression()
    model.coef_ = np.array([1.0], dtype=np.float64)
    model.intercept_ = 0.0
    model.n_features_in_ = 1
    model.feature_names_in_ = np.array(["signal"], dtype=object)
    return preprocessor, model


def _synthetic_contract_frame(feature_roles: dict, rows: int = 4) -> pd.DataFrame:
    """Create a small raw frame with every primary and lender field."""
    contracts = feature_roles["contracts"]
    fields = list(dict.fromkeys(
        contracts["main_without_sensitive_without_lender"]
        + contracts["main_without_sensitive_with_lender"]
    ))
    data: dict[str, object] = {}
    for field in fields:
        if field in NUMERIC_MODEL_FIELDS:
            data[field] = np.arange(1, rows + 1, dtype=np.float64)
        else:
            data[field] = [f"{field}_{index % 2}" for index in range(rows)]
    return pd.DataFrame(data)


def _feature_names(value) -> set[str]:
    """Normalize the supported compact-pack return shapes for assertions."""
    if isinstance(value, dict):
        names: list[str] = []
        for key in ("feature_names", "numeric_features", "categorical_features"):
            names.extend(value.get(key, []))
        return set(names)
    if hasattr(value, "feature_names"):
        return set(value.feature_names)
    if isinstance(value, (list, tuple, set)):
        return set(value)
    raise AssertionError(f"Unsupported feature-pack result: {type(value)!r}")


def _literal_loader_paths(source_path: Path) -> list[str]:
    """Return literal paths passed directly to file-loading calls."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    loader_names = {"open", "read_csv", "read_parquet", "read_table", "ParquetFile"}
    paths: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        called = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if called not in loader_names:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            paths.append(first.value.lower().replace("\\", "/"))
    return paths


class Prompt2ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.feature_roles = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "feature_roles.json").read_text(encoding="utf-8")
        )
        # Read only the permitted Development columns, once for this test process.
        development_path = PROJECT_ROOT / "outputs" / "data" / "development.parquet"
        cls.development_keys = pq.read_table(
            development_path,
            columns=["row_hash", "record_hash", "development_role", "loan_amount_000s"],
        ).to_pandas()
        cls.tmp = PROJECT_ROOT / "outputs" / "tmp" / "prompt2" / "test_prompt2_modeling"
        cls.tmp.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmp.exists():
            shutil.rmtree(cls.tmp)

    def test_data_ready_handoff_validation(self) -> None:
        result = validate_data_ready(PROJECT_ROOT)
        ready = result["data_ready"]
        self.assertEqual(ready["status"], "PASS")
        self.assertEqual(ready["development_rows"], 500_000)
        self.assertEqual(ready["train_rows"], 400_000)
        self.assertEqual(ready["validation_rows"], 100_000)
        self.assertEqual(ready["development_iid_overlap"], 0)
        self.assertEqual(ready["legacy_overlap"], 0)
        self.assertEqual(ready["selected_conflict_groups"], 0)
        self.assertEqual(result["final_verification"]["status"], "PASS")

    def test_development_role_counts_and_target_validity(self) -> None:
        frame = self.development_keys
        self.assertEqual(len(frame), 500_000)
        self.assertEqual(frame["development_role"].value_counts().to_dict(), {"train": 400_000, "validation": 100_000})
        self.assertTrue(np.isfinite(frame["loan_amount_000s"].to_numpy(dtype=np.float64)).all())

    def test_train_validation_membership_is_disjoint(self) -> None:
        frame = self.development_keys
        self.assertFalse(frame["row_hash"].duplicated().any())
        self.assertFalse(frame["record_hash"].duplicated().any())
        roles_per_hash = frame.groupby("row_hash", sort=False)["development_role"].nunique()
        self.assertEqual(int((roles_per_hash > 1).sum()), 0)

    def test_saved_feature_contracts_exclude_target_audit_and_sensitive_fields(self) -> None:
        roles = load_feature_roles(PROJECT_ROOT)
        forbidden = set(roles["target_and_alias_exclusions"]) | set(roles["audit_only_fields"]) | set(roles["sensitive_fields"])
        for name in ("main_without_sensitive_without_lender", "main_without_sensitive_with_lender"):
            self.assertFalse(forbidden & set(roles["contracts"][name]))

    def test_lender_contract_diff_is_exactly_respondent_id(self) -> None:
        contracts = self.feature_roles["contracts"]
        no_lender = contracts["main_without_sensitive_without_lender"]
        with_lender = contracts["main_without_sensitive_with_lender"]
        self.assertNotIn("respondent_id", no_lender)
        self.assertIn("respondent_id", with_lender)
        self.assertEqual(["respondent_id"] + no_lender, with_lender)

    def test_frozen_contracts_preserve_prompt1b_boundaries(self) -> None:
        train = _synthetic_contract_frame(self.feature_roles)
        frozen = freeze_feature_contracts(train, self.feature_roles)
        self.assertEqual(
            frozen["main_without_sensitive_without_lender"],
            self.feature_roles["contracts"]["main_without_sensitive_without_lender"],
        )
        self.assertEqual(
            frozen["main_without_sensitive_with_lender"],
            self.feature_roles["contracts"]["main_without_sensitive_with_lender"],
        )

    def test_linear_compact_v2_construction(self) -> None:
        rows = 101
        train = _synthetic_contract_frame(self.feature_roles, rows=rows)
        train["applicant_income_000s"] = np.arange(rows, dtype=np.float64)
        train["log1p_applicant_income"] = np.log1p(np.arange(rows, dtype=np.float64))
        train["has_co_applicant"] = np.arange(rows) % 2
        train["loan_type_name"] = ["A" if index % 2 else "B" for index in range(rows)]
        train["state_name"] = [f"state_{index % 50}" for index in range(rows)]
        train["agency_name"] = [f"agency_{index}" for index in range(rows)]
        train["msamd_name"] = ["metro"] * rows
        train["county_name"] = ["county"] * rows
        train["county_code"] = ["001"] * rows
        train["census_tract_number"] = ["0001"] * rows
        names = _feature_names(build_linear_compact_v2(train, self.feature_roles))
        self.assertTrue({"applicant_income_000s", "log1p_applicant_income", "has_co_applicant", "loan_type_name", "state_name"} <= names)
        self.assertFalse({"respondent_id", "msamd_name", "county_name", "county_code", "census_tract_number", "agency_name"} & names)

    def test_category_encoder_is_train_fitted_and_handles_unknowns(self) -> None:
        train = pd.DataFrame({
            "low": ["a", "b", "a", None],
            "high": ["h1", "h2", "h3", "h4"],
        })
        validation = pd.DataFrame({
            "low": ["validation_only", None],
            "high": ["validation_only", None],
        })
        encoder = TrainFittedCategoricalEncoder(high_cardinality_threshold=2)
        encoder.fit(train)
        first = np.asarray(encoder.transform(validation), dtype=np.float64)
        second = np.asarray(encoder.transform(validation), dtype=np.float64)
        self.assertEqual(first.shape, (2, 2))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)
        learned = repr(vars(encoder))
        self.assertNotIn("validation_only", learned)

    def test_xgboost_preprocessor_stays_sparse_and_handles_unknowns(self) -> None:
        train = pd.DataFrame({
            "numeric": [1.0, np.nan, 3.0, 4.0],
            "low": ["a", "b", "a", None],
            "high": ["h1", "h2", "h3", "h4"],
        })
        validation = pd.DataFrame({"numeric": [np.nan], "low": ["new"], "high": ["never_seen"]})
        preprocessor = make_xgb_preprocessor(list(train.columns), high_cardinality_threshold=2)
        transformed_train = preprocessor.fit_transform(train)
        transformed_validation = preprocessor.transform(validation)
        self.assertTrue(sparse.issparse(transformed_train))
        self.assertTrue(sparse.issparse(transformed_validation))
        self.assertTrue(np.isfinite(transformed_validation.data).all())

    def test_target_transform_and_inverse_are_exact(self) -> None:
        target = np.array([0.0, 1.0, 10.0, 1_000.0])
        np.testing.assert_array_equal(transform_target(target, "raw"), target)
        np.testing.assert_allclose(inverse_target(transform_target(target, "log1p"), "log1p"), target, rtol=0, atol=1e-12)
        with self.assertRaises(ValueError):
            transform_target(target, "sqrt")

    def test_metric_formulas_and_signed_error_direction(self) -> None:
        actual = np.array([10.0, 20.0, 100.0, 200.0])
        predicted = np.array([8.0, 25.0, 90.0, 210.0])
        metrics = compute_regression_metrics(actual, predicted)
        error = predicted - actual
        self.assertAlmostEqual(metrics["mae"], 6.75)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(57.25))
        self.assertAlmostEqual(metrics["median_absolute_error"], 7.5)
        self.assertAlmostEqual(metrics["p90_absolute_error"], 10.0)
        self.assertAlmostEqual(metrics["mean_signed_error"], 0.75)
        self.assertEqual(metrics["negative_prediction_count"], 0)
        self.assertAlmostEqual(metrics["mean_signed_error"], float(np.mean(error)))
        under = compute_regression_metrics(np.array([10.0, 20.0]), np.array([9.0, 18.0]))
        self.assertLess(under["mean_signed_error"], 0.0)

    def test_rmsle_only_clips_negative_predictions(self) -> None:
        actual = np.array([1.0, 2.0])
        predicted = np.array([-3.0, 4.0])
        metrics = compute_regression_metrics(actual, predicted)
        self.assertAlmostEqual(metrics["mae"], 3.0)
        self.assertEqual(metrics["negative_prediction_count"], 1)
        expected = np.sqrt(np.mean((np.log1p([0.0, 4.0]) - np.log1p(actual)) ** 2))
        self.assertAlmostEqual(metrics["rmsle"], expected)

    def test_tail_membership_uses_fixed_validation_targets(self) -> None:
        actual = np.arange(1.0, 101.0)
        decile = tail_membership(actual, 0.90)
        five_percent = tail_membership(actual, 0.95)
        self.assertEqual(int(decile.sum()), 10)
        self.assertEqual(int(five_percent.sum()), 5)
        self.assertTrue(np.all(actual[decile] >= np.quantile(actual, 0.90)))

    def test_family_selection_relative_tie_rule(self) -> None:
        records = [
            {"status": "COMPLETE", "candidate_id": "lower_mae", "mae": 100.0, "rmse": 120.0, "top_decile_mae": 300.0, "fitted_iterations": 200, "bundle_size_bytes": 2000, "fit_time_seconds": 20.0, "target_mode": "log1p"},
            {"status": "COMPLETE", "candidate_id": "relative_tie_lower_rmse", "mae": 100.2, "rmse": 110.0, "top_decile_mae": 310.0, "fitted_iterations": 200, "bundle_size_bytes": 2000, "fit_time_seconds": 20.0, "target_mode": "raw"},
        ]
        self.assertEqual(select_family_candidate(records)["candidate_id"], "relative_tie_lower_rmse")

        tail_tie = [
            {"status": "COMPLETE", "candidate_id": "a", "mae": 100.0, "rmse": 110.0, "top_decile_mae": 300.0, "fitted_iterations": 100, "bundle_size_bytes": 1000, "fit_time_seconds": 10.0, "target_mode": "raw"},
            {"status": "COMPLETE", "candidate_id": "b", "mae": 100.1, "rmse": 110.1, "top_decile_mae": 250.0, "fitted_iterations": 200, "bundle_size_bytes": 2000, "fit_time_seconds": 20.0, "target_mode": "log1p"},
        ]
        self.assertEqual(select_family_candidate(tail_tie)["candidate_id"], "b")

    def test_deterministic_candidate_ids_ignore_parameter_order(self) -> None:
        first = deterministic_candidate_id("xgboost", "anchor", "main", "raw", {"depth": 6, "eta": 0.05}, 42)
        second = deterministic_candidate_id("xgboost", "anchor", "main", "raw", {"eta": 0.05, "depth": 6}, 42)
        changed = deterministic_candidate_id("xgboost", "anchor", "main", "raw", {"eta": 0.05, "depth": 6}, 43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(first, deterministic_candidate_id("xgboost", "anchor", "main", "raw", {"depth": 6, "eta": 0.05}, 42))

    def test_model_bundle_serialization_and_prediction_row_order(self) -> None:
        preprocessor, model = _fit_free_linear_components()
        bundle = ModelBundle(
            model_id="synthetic_bundle",
            family="synthetic",
            feature_names=["signal"],
            feature_contract_name="synthetic_contract",
            target_mode="raw",
            preprocessor=preprocessor,
            model=model,
            package_versions={"numpy": np.__version__},
            model_parameters={},
            selected_best_iteration=None,
            development_source_sha256="0" * 64,
            train_row_hash_digest="1" * 64,
            validation_row_hash_digest="2" * 64,
        )
        path = self.tmp / "synthetic_bundle.joblib"
        atomic_joblib_dump(bundle, path)
        reloaded = load_bundle(path)
        raw = pd.DataFrame({"unused": [9.0, 9.0, 9.0], "signal": [3.0, 1.0, 2.0]}, index=[30, 10, 20])
        expected = np.array([3.0, 1.0, 2.0])
        np.testing.assert_array_equal(reloaded.predict(raw), expected)

    def test_model_bundle_clean_process_prediction_equality(self) -> None:
        preprocessor, model = _fit_free_linear_components()
        bundle = ModelBundle(
            model_id="clean_process_bundle",
            family="synthetic",
            feature_names=["signal"],
            feature_contract_name="synthetic_contract",
            target_mode="raw",
            preprocessor=preprocessor,
            model=model,
            package_versions={"numpy": np.__version__},
            model_parameters={},
            selected_best_iteration=None,
            development_source_sha256="0" * 64,
            train_row_hash_digest="1" * 64,
            validation_row_hash_digest="2" * 64,
        )
        path = self.tmp / "clean_process_bundle.joblib"
        atomic_joblib_dump(bundle, path)
        reference = bundle.predict(pd.DataFrame({"signal": [5.0, 2.0, 8.0]}))
        code = (
            "import json, joblib, pandas as pd; "
            f"b=joblib.load({str(path)!r}); "
            "print(json.dumps(b.predict(pd.DataFrame({'signal':[5.0,2.0,8.0]})).tolist()))"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT / "tests"), str(SRC_DIR)])
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        clean = np.asarray(json.loads(completed.stdout.strip()), dtype=np.float64)
        self.assertLessEqual(float(np.max(np.abs(reference - clean))), 1e-7)

    def test_prompt2_python_has_no_literal_iid_or_raw_loader(self) -> None:
        for name in ("prompt2_modeling.py", "preprocessing.py", "metrics.py", "model_bundles.py"):
            path = SRC_DIR / name
            for literal in _literal_loader_paths(path):
                self.assertNotIn("iid_holdout", literal, msg=f"{name} directly loads IID: {literal}")
                self.assertFalse(literal.endswith(".csv") or literal.endswith(".zip"), msg=f"{name} directly loads Raw: {literal}")

    def test_notebook_is_artifact_only_when_present(self) -> None:
        notebook_path = PROJECT_ROOT / "notebooks" / "02_BASELINES_AND_BOOSTING.ipynb"
        if not notebook_path.exists():
            self.skipTest("Prompt 2 reporting notebook is created after full fits")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        tree = ast.parse(code)
        prohibited_calls = {"fit", "fit_transform", "fit_predict", "train"}
        found = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in prohibited_calls
        }
        self.assertFalse(found, msg=f"Notebook contains model/preprocessing fit calls: {sorted(found)}")
        lowered = code.lower().replace("\\", "/")
        self.assertNotIn("iid_holdout_features.parquet", lowered)
        self.assertNotIn("iid_holdout_targets.parquet", lowered)
        self.assertNotIn("hmda_2017_nationwide_all-records_labels.csv", lowered)
        self.assertNotIn("hmda_2017_nationwide_all-records_labels.zip", lowered)

    def test_no_random_temporary_directory_at_regression_v2_root(self) -> None:
        suspicious = [
            path.name
            for path in PROJECT_ROOT.iterdir()
            if path.is_dir() and (path.name.lower().startswith("tmp") or path.name.lower().startswith("temp"))
        ]
        self.assertEqual(suspicious, [])


if __name__ == "__main__":
    unittest.main()
