"""Focused, fit-free verification for Regression V2 Prompt 3.

The suite may read the frozen Development file and saved Prompt 2 artifacts.
It never opens Raw or IID data and never calls a Deep model fit.  Files made by
tests stay below ``outputs/tmp/prompt3/tests``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
TEST_TMP = PROJECT_ROOT / "outputs" / "tmp" / "prompt3" / "tests"

# The project-local Deep environment reuses the base Python data packages.
_BASE_SITE = Path(sys.base_prefix) / "Lib" / "site-packages"
if _BASE_SITE.is_dir() and str(_BASE_SITE) not in sys.path:
    sys.path.append(str(_BASE_SITE))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyRegressor

import prompt3_deep_models as p3
from deep_bundles import (
    BundleMetadata,
    FTTransformerBundle,
    RealMLPBundle,
    load_bundle,
)
from deep_metrics import (
    compute_regression_metrics,
    inverse_target,
    paired_mae_bootstrap,
    select_deep_anchor,
    select_family_candidate,
    tail_membership,
    transform_target,
)
from deep_preprocessing import (
    FTPreprocessor,
    MISSING_TOKEN,
    RealMLPPreprocessor,
    duplicate_safe_deciles,
    make_ft_internal_split,
    ordered_digest,
)


PROMPT3_SOURCE = (SRC_DIR / "prompt3_deep_models.py").read_text(encoding="utf-8")
PROMPT3_TREE = ast.parse(PROMPT3_SOURCE)
EXPECTED_DEVELOPMENT_SHA256 = (
    "0ed232397be3ec4de1483c594954dce7b4704b375ca295397d899323dc4f0b6b"
)
EXPECTED_TRAIN_DIGEST = (
    "26265d75d8fa35d7417e2a9fb2888b9f625e6d19123cd7a2972974f403a30166"
)
EXPECTED_VALIDATION_DIGEST = (
    "676b577233627c8b237d214a29eeb798e099d028583c8b73068b151a2a204290"
)
EXPECTED_SCHEMA_DIGEST = (
    "9d7fc383dd63af96bc274d769920eba0741459dd4ec118fd897c9f1e27d47f39"
)
EXPECTED_PREDICTION_COLUMNS = [
    "row_hash",
    "y_true",
    "y_pred",
    "model_id",
    "family",
    "feature_contract",
    "target_mode",
]


def _fresh_directory(name: str) -> Path:
    """Return a clean test directory that is proven to be below TEST_TMP."""
    path = (TEST_TMP / name).resolve()
    path.relative_to(TEST_TMP.resolve())
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _function_node(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in PROMPT3_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate(candidate_id: str, **updates) -> dict:
    result = {
        "candidate_id": candidate_id,
        "mae": 100.0,
        "rmse": 120.0,
        "top_decile_mae": 300.0,
        "top_five_percent_mae": 400.0,
        "fit_time_seconds": 10.0,
        "bundle_size_bytes": 1000,
        "simplicity_rank": 0,
    }
    result.update(updates)
    return result


def _metadata(family: str, feature_names: list[str], target_mode: str) -> BundleMetadata:
    return BundleMetadata(
        model_id=f"tiny_{family}",
        family=family,
        feature_names=feature_names,
        feature_contract_name=p3.PRIMARY_CONTRACT,
        target_mode=target_mode,
        model_configuration={"test_only": True},
        package_versions={},
        development_source_sha256=EXPECTED_DEVELOPMENT_SHA256,
        train_row_hash_digest=EXPECTED_TRAIN_DIGEST,
        validation_row_hash_digest=EXPECTED_VALIDATION_DIGEST,
        training_seed=42,
        selected_epoch=1,
        device="cpu float32",
    )


def setUpModule() -> None:
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    for name in ("TMP", "TEMP", "MPLCONFIGDIR", "TORCH_HOME", "XDG_CACHE_HOME", "HF_HOME"):
        destination = TEST_TMP / "cache" / name.lower()
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(destination)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def tearDownModule() -> None:
    if TEST_TMP.exists():
        TEST_TMP.resolve().relative_to(
            (PROJECT_ROOT / "outputs" / "tmp" / "prompt3").resolve()
        )
        shutil.rmtree(TEST_TMP)


class Prompt3HandoffAndBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        development = PROJECT_ROOT / p3.DEVELOPMENT_RELATIVE
        cls.development_path = development
        cls.keys = pq.read_table(
            development,
            columns=["row_hash", "record_hash", "development_role", p3.TARGET],
        ).to_pandas()
        cls.train = cls.keys.loc[cls.keys["development_role"].eq("train")]
        cls.validation = cls.keys.loc[cls.keys["development_role"].eq("validation")]
        cls.roles = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "feature_roles.json").read_text(
                encoding="utf-8"
            )
        )
        cls.design = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "prompt3_frozen_design.json").read_text(
                encoding="utf-8"
            )
        )

    def test_prompt2_handoff_and_development_identity(self) -> None:
        handoff = p3.validate_prompt2_handoff(PROJECT_ROOT)
        self.assertEqual(handoff["status"], "PASS")
        self.assertTrue(all(report["status"] == "PASS" for report in handoff["reports"].values()))
        task = (PROJECT_ROOT / "TASK.md").read_text(encoding="utf-8")
        self.assertTrue(
            any(
                state in task
                for state in (
                    "Prompt 3 IN PROGRESS",
                    "Prompt 3 BLOCKED",
                    "Prompt 3R IN PROGRESS",
                    "Prompt 3 COMPLETE",
                )
            )
        )

        self.assertEqual(_sha256(self.development_path), EXPECTED_DEVELOPMENT_SHA256)
        self.assertEqual(len(self.keys), 500_000)
        self.assertEqual(len(self.train), 400_000)
        self.assertEqual(len(self.validation), 100_000)
        self.assertTrue(self.keys["row_hash"].is_unique)
        self.assertTrue(self.keys["record_hash"].is_unique)
        self.assertTrue(np.isfinite(self.keys[p3.TARGET].to_numpy(np.float64)).all())
        self.assertFalse(set(self.train["row_hash"]) & set(self.validation["row_hash"]))
        self.assertEqual(ordered_digest(self.train["row_hash"]), EXPECTED_TRAIN_DIGEST)
        self.assertEqual(
            ordered_digest(self.validation["row_hash"]), EXPECTED_VALIDATION_DIGEST
        )
        schema = pq.read_schema(self.development_path)
        schema_payload = [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in schema
        ]
        self.assertEqual(p3.canonical_digest(schema_payload), EXPECTED_SCHEMA_DIGEST)

    def test_exact_no_sensitive_no_lender_contract(self) -> None:
        contract = self.roles["contracts"][p3.PRIMARY_CONTRACT]
        forbidden = (
            set(self.roles["sensitive_fields"])
            | set(self.roles["audit_only_fields"])
            | set(self.roles["target_and_alias_exclusions"])
            | {"respondent_id"}
        )
        self.assertEqual(len(contract), 35)
        self.assertEqual(len(set(contract)), 35)
        self.assertFalse(forbidden & set(contract))
        self.assertTrue(set(contract).issubset(pq.read_schema(self.development_path).names))
        self.assertEqual(self.design["feature_contract"]["features"], contract)
        self.assertEqual(len(self.design["feature_contract"]["numeric_features"]), 17)
        self.assertEqual(len(self.design["feature_contract"]["categorical_features"]), 18)

    def test_selected_prompt2_predictions_are_exactly_aligned(self) -> None:
        manifest = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "prompt2_prediction_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        entries = {entry["key"]: entry for entry in manifest["artifacts"]}
        self.assertEqual(manifest["validation_row_hash_digest"], EXPECTED_VALIDATION_DIGEST)
        self.assertEqual(set(p3.PROMPT2_PREDICTIONS), {
            "lasso", "histgradientboosting", "catboost", "lightgbm", "xgboost"
        })
        expected_hashes = self.validation["row_hash"].astype(str).reset_index(drop=True)
        expected_target = self.validation[p3.TARGET].to_numpy(dtype=np.float64)
        for family, relative in p3.PROMPT2_PREDICTIONS.items():
            with self.subTest(family=family):
                entry = entries[family]
                self.assertEqual(entry["path"], relative.as_posix())
                path = PROJECT_ROOT / relative
                self.assertEqual(_sha256(path), entry["sha256"])
                frame = pd.read_parquet(path)
                self.assertEqual(list(frame.columns), EXPECTED_PREDICTION_COLUMNS)
                self.assertEqual(len(frame), 100_000)
                self.assertTrue(frame["row_hash"].astype(str).reset_index(drop=True).equals(expected_hashes))
                self.assertTrue(np.array_equal(frame["y_true"].to_numpy(np.float64), expected_target))
                self.assertTrue(np.isfinite(frame["y_pred"].to_numpy(np.float64)).all())
                self.assertNotIn("with_lender.parquet", relative.name)

    def test_read_and_write_guards_close_raw_iid_and_outside_paths(self) -> None:
        development = p3.guard_read_path(PROJECT_ROOT, p3.DEVELOPMENT_RELATIVE)
        self.assertEqual(development, self.development_path.resolve())
        for prohibited in (
            "data/raw.csv",
            "outputs/data/iid_holdout_features.parquet",
            "outputs/data/iid_holdout_targets.parquet",
            REPOSITORY_ROOT / "AGENTS.md",
        ):
            with self.subTest(path=str(prohibited)), self.assertRaises(PermissionError):
                p3.guard_read_path(PROJECT_ROOT, prohibited)
        with self.assertRaises(PermissionError):
            p3.guard_write_path(PROJECT_ROOT, "data/changed.csv")
        with self.assertRaises(PermissionError):
            p3.guard_write_path(PROJECT_ROOT, REPOSITORY_ROOT / "outside.json")
        allowed = p3.guard_write_path(PROJECT_ROOT, "outputs/tmp/prompt3/allowed.json")
        allowed.relative_to((PROJECT_ROOT / "outputs" / "tmp" / "prompt3").resolve())

    def test_frozen_design_matches_current_code_and_state(self) -> None:
        current = p3._load_design(PROJECT_ROOT, require_smoke=False)
        self.assertEqual(current["status"], "FROZEN")
        self.assertEqual(current["development_source"]["schema_digest"], EXPECTED_SCHEMA_DIGEST)
        self.assertEqual(current["development_source"]["development_sha256"], EXPECTED_DEVELOPMENT_SHA256)
        self.assertEqual(current["development_source"]["raw_access_count"], 0)
        self.assertEqual(current["development_source"]["iid_feature_access_count"], 0)
        self.assertEqual(current["development_source"]["iid_target_access_count"], 0)

    def test_no_forbidden_prompt3_artifact_names_or_root_temp_directory(self) -> None:
        model_root = PROJECT_ROOT / p3.MODELS_RELATIVE
        prediction_root = PROJECT_ROOT / "outputs" / "predictions" / "prompt3"
        names = [
            path.name.lower()
            for base in (model_root, prediction_root)
            if base.exists()
            for path in base.rglob("*")
        ]
        self.assertFalse(any("ensemble" in name for name in names))
        self.assertFalse(any("tail_gate" in name or "tail_specialist" in name for name in names))
        self.assertFalse(any("final_article" in name or "final_project" in name for name in names))
        self.assertFalse(any("iid" in name for name in names))
        suspicious = [
            path.name
            for path in PROJECT_ROOT.iterdir()
            if path.is_dir() and path.name.lower().startswith(("tmp", "temp"))
        ]
        self.assertEqual(suspicious, [])


class Prompt3PreprocessingAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        development = pq.read_table(
            PROJECT_ROOT / p3.DEVELOPMENT_RELATIVE,
            columns=["row_hash", "development_role", p3.TARGET],
        ).to_pandas()
        cls.train = development.loc[development["development_role"].eq("train")]
        cls.validation = development.loc[development["development_role"].eq("validation")]
        cls.design = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "prompt3_frozen_design.json").read_text(
                encoding="utf-8"
            )
        )

    def test_duplicate_safe_deciles_and_small_split_are_deterministic(self) -> None:
        target = np.repeat(np.arange(1.0, 11.0), 10)
        bins = duplicate_safe_deciles(target)
        grouped = pd.DataFrame({"target": target, "bin": bins}).groupby("target")["bin"].nunique()
        self.assertEqual(int(grouped.max()), 1)
        hashes = [f"train-{index:03d}" for index in range(100)]
        external = [f"external-{index:03d}" for index in range(20)]
        first = make_ft_internal_split(
            target,
            hashes,
            external_validation_row_hashes=external,
            fit_rows=80,
            early_stopping_rows=20,
            random_state=42,
        )
        second = make_ft_internal_split(
            target,
            hashes,
            external_validation_row_hashes=external,
            fit_rows=80,
            early_stopping_rows=20,
            random_state=42,
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[2]["fit_rows"], 80)
        self.assertEqual(first[2]["early_stopping_rows"], 20)
        self.assertEqual(first[2]["internal_overlap"], 0)
        self.assertEqual(first[2]["external_validation_overlap"], 0)

    def test_full_ft_internal_split_has_exact_frozen_membership(self) -> None:
        fit_index, stop_index, audit = make_ft_internal_split(
            self.train[p3.TARGET],
            self.train["row_hash"],
            external_validation_row_hashes=self.validation["row_hash"],
        )
        self.assertEqual(len(fit_index), 360_000)
        self.assertEqual(len(stop_index), 40_000)
        self.assertEqual(np.intersect1d(fit_index, stop_index).size, 0)
        self.assertEqual(len(np.union1d(fit_index, stop_index)), 400_000)
        self.assertEqual(audit, self.design["ft_internal_split"])

    def test_realmlp_preprocessor_is_train_only_and_preserves_sources(self) -> None:
        train = pd.DataFrame({
            "numeric": [1.0, np.nan, 3.0, 100.0],
            "category": ["a", "b", None, "a"],
        })
        validation = pd.DataFrame({
            "numeric": [np.nan, 1_000_000.0],
            "category": ["validation_only", None],
        })
        train_before = train.copy(deep=True)
        validation_before = validation.copy(deep=True)
        preprocessor = RealMLPPreprocessor(
            ["numeric", "category"], ["numeric"], ["category"]
        )
        preprocessor.fit(train, row_hashes=["a", "b", "c", "d"])
        transformed = preprocessor.transform(validation)
        self.assertEqual(preprocessor.numeric_medians_["numeric"], 3.0)
        self.assertEqual(preprocessor.categorical_vocabularies_["category"], ["a", "b"])
        self.assertNotIn("validation_only", preprocessor.categorical_vocabularies_["category"])
        self.assertEqual(float(transformed.loc[0, "numeric"]), 3.0)
        self.assertEqual(transformed.loc[0, "category"], "validation_only")
        self.assertEqual(transformed.loc[1, "category"], MISSING_TOKEN)
        self.assertEqual(transformed["numeric"].dtype, np.float32)
        self.assertEqual(preprocessor.evidence()["fit_row_count"], 4)
        self.assertEqual(preprocessor.evidence()["fit_row_hash_digest"], ordered_digest(["a", "b", "c", "d"]))
        assert_frame_equal(train, train_before)
        assert_frame_equal(validation, validation_before)

    def test_ft_preprocessor_train_only_missing_unknown_and_dtypes(self) -> None:
        train = pd.DataFrame({
            "numeric": [1.0, np.nan, 3.0, 100.0],
            "category": ["a", "b", None, "a"],
        })
        validation = pd.DataFrame({
            "numeric": [np.nan, 1_000_000.0, 2.0],
            "category": ["validation_only", None, "a"],
        })
        train_before = train.copy(deep=True)
        validation_before = validation.copy(deep=True)
        preprocessor = FTPreprocessor(
            ["numeric", "category"], ["numeric"], ["category"], n_quantiles=1000
        )
        preprocessor.fit(train, row_hashes=["a", "b", "c", "d"])
        numeric, categorical = preprocessor.transform(validation)
        self.assertEqual(preprocessor.numeric_medians_["numeric"], 3.0)
        self.assertEqual(preprocessor.categorical_vocabularies_["category"], {"a": 2, "b": 3})
        self.assertNotIn("validation_only", preprocessor.categorical_vocabularies_["category"])
        self.assertEqual(preprocessor.cardinalities_, [4])
        self.assertEqual(categorical[:, 0].tolist(), [1, 0, 2])
        self.assertEqual(numeric.dtype, np.float32)
        self.assertEqual(categorical.dtype, np.int64)
        self.assertTrue(np.isfinite(numeric).all())
        self.assertEqual(preprocessor.quantile_transformer_.n_quantiles_, 4)
        self.assertEqual(preprocessor.quantile_transformer_.output_distribution, "normal")
        self.assertIsNone(preprocessor.quantile_transformer_.subsample)
        evidence = preprocessor.evidence()
        self.assertEqual((evidence["missing_index"], evidence["unknown_index"], evidence["known_start_index"]), (0, 1, 2))
        assert_frame_equal(train, train_before)
        assert_frame_equal(validation, validation_before)

    def test_ft_split_rejects_duplicates_and_external_overlap(self) -> None:
        target = np.repeat(np.arange(1.0, 11.0), 10)
        hashes = [f"h-{index}" for index in range(100)]
        duplicates = hashes.copy()
        duplicates[-1] = duplicates[0]
        with self.assertRaises(ValueError):
            make_ft_internal_split(target, duplicates, fit_rows=80, early_stopping_rows=20)
        with self.assertRaises(ValueError):
            make_ft_internal_split(
                target,
                hashes,
                external_validation_row_hashes=[hashes[0]],
                fit_rows=80,
                early_stopping_rows=20,
            )


class Prompt3MetricsAndSelectionTests(unittest.TestCase):
    def test_target_transform_inverse_and_rmsle_clipping(self) -> None:
        target = np.array([0.0, 1.0, 10.0, 1_000.0])
        np.testing.assert_array_equal(transform_target(target, "raw"), target)
        np.testing.assert_allclose(
            inverse_target(transform_target(target, "log1p"), "log1p"),
            target,
            rtol=0,
            atol=1e-12,
        )
        actual = np.array([1.0, 2.0])
        predicted = np.array([-3.0, 4.0])
        metrics = compute_regression_metrics(actual, predicted)
        self.assertEqual(metrics["negative_prediction_count"], 1)
        self.assertEqual(metrics["mae"], 3.0)
        expected_rmsle = np.sqrt(np.mean((np.log1p([0.0, 4.0]) - np.log1p(actual)) ** 2))
        self.assertAlmostEqual(metrics["rmsle"], expected_rmsle)

    def test_metric_formulas_tails_and_signed_error_direction(self) -> None:
        actual = np.arange(1.0, 101.0)
        predicted = actual.copy()
        predicted[-10:] -= np.arange(1.0, 11.0)
        predicted[0] = -2.0
        metrics = compute_regression_metrics(
            actual,
            predicted,
            fit_time_seconds=2.0,
            prediction_time_seconds=0.5,
            model_size_bytes=123,
            bundle_size_bytes=456,
        )
        error = predicted - actual
        absolute = np.abs(error)
        decile = actual >= np.quantile(actual, 0.90)
        five = actual >= np.quantile(actual, 0.95)
        self.assertAlmostEqual(metrics["mae"], float(np.mean(absolute)))
        self.assertAlmostEqual(metrics["rmse"], float(np.sqrt(np.mean(error**2))))
        self.assertAlmostEqual(metrics["mean_signed_error"], float(np.mean(error)))
        self.assertAlmostEqual(metrics["top_decile_mae"], float(np.mean(absolute[decile])))
        self.assertAlmostEqual(metrics["top_five_percent_mae"], float(np.mean(absolute[five])))
        self.assertAlmostEqual(metrics["top_decile_signed_error"], float(np.mean(error[decile])))
        self.assertAlmostEqual(metrics["top_five_percent_signed_error"], float(np.mean(error[five])))
        self.assertAlmostEqual(metrics["top_decile_underprediction_rate"], float(np.mean(error[decile] < 0)))
        self.assertAlmostEqual(metrics["top_five_percent_underprediction_rate"], float(np.mean(error[five] < 0)))
        self.assertLess(metrics["mean_signed_error"], 0.0)
        self.assertEqual((metrics["model_size_bytes"], metrics["bundle_size_bytes"]), (123, 456))
        self.assertEqual(int(tail_membership(actual, 0.90).sum()), 10)
        self.assertEqual(int(tail_membership(actual, 0.95).sum()), 5)

    def test_family_selection_exact_relative_tie_boundaries(self) -> None:
        exact = [
            _candidate("lower_mae", mae=100.0, rmse=130.0),
            _candidate("exact_boundary", mae=100.25, rmse=110.0),
        ]
        self.assertEqual(select_family_candidate(exact)["candidate_id"], "exact_boundary")
        above = [
            _candidate("lower_mae", mae=100.0, rmse=130.0),
            _candidate("above_boundary", mae=100.250001, rmse=110.0),
        ]
        self.assertEqual(select_family_candidate(above)["candidate_id"], "lower_mae")
        rmse_exact = [
            _candidate("tail_a", mae=100.0, rmse=100.0, top_decile_mae=300.0),
            _candidate("tail_b", mae=100.1, rmse=100.25, top_decile_mae=250.0),
        ]
        self.assertEqual(select_family_candidate(rmse_exact)["candidate_id"], "tail_b")
        rmse_above = [
            _candidate("rmse_best", mae=100.0, rmse=100.0, top_decile_mae=300.0),
            _candidate("rmse_out", mae=100.1, rmse=100.250001, top_decile_mae=200.0),
        ]
        self.assertEqual(select_family_candidate(rmse_above)["candidate_id"], "rmse_best")

    def test_family_selection_final_tie_order(self) -> None:
        fields = [
            ("top_decile_mae", 299.0),
            ("top_five_percent_mae", 399.0),
            ("fit_time_seconds", 9.0),
            ("bundle_size_bytes", 999),
            ("simplicity_rank", -1),
        ]
        for field, better in fields:
            with self.subTest(field=field):
                first = _candidate("a")
                second = _candidate("b", **{field: better})
                # Keep all earlier tie fields equal, so this field decides.
                self.assertEqual(select_family_candidate([first, second])["candidate_id"], "b")

    def test_deep_anchor_requires_both_families_and_uses_frozen_order(self) -> None:
        real = {**_candidate("real"), "family": "realmlp"}
        ft = {**_candidate("ft", mae=100.25, rmse=110.0), "family": "fttransformer"}
        self.assertEqual(select_deep_anchor([real, ft])["candidate_id"], "ft")
        ft["mae"] = 100.250001
        self.assertEqual(select_deep_anchor([real, ft])["candidate_id"], "real")
        with self.assertRaises(ValueError):
            select_deep_anchor([real])

    def test_paired_bootstrap_is_aligned_deterministic_and_frozen(self) -> None:
        actual = np.arange(1.0, 101.0)
        real = actual + np.sin(actual)
        ft = actual + np.cos(actual)
        first = paired_mae_bootstrap(actual, real, ft)
        second = paired_mae_bootstrap(actual, real, ft)
        self.assertEqual(first, second)
        self.assertEqual(first["n_rows"], 100)
        self.assertEqual(first["n_resamples"], 300)
        self.assertEqual(first["random_state"], 42)
        self.assertAlmostEqual(
            first["realmlp_win_proportion"] + first["ft_win_proportion"] + first["tie_proportion"],
            1.0,
        )
        with self.assertRaises(ValueError):
            paired_mae_bootstrap(actual, real[:-1], ft)


class Prompt3DesignAndStaticSafetyTests(unittest.TestCase):
    def test_exact_candidates_architecture_and_fit_budget(self) -> None:
        self.assertEqual([row["candidate_id"] for row in p3.REALMLP_CANDIDATES], [
            "realmlp_raw_pdrop015", "realmlp_raw_pdrop020"
        ])
        self.assertEqual([row["p_drop"] for row in p3.REALMLP_CANDIDATES], [0.15, 0.20])
        self.assertTrue(all(row["target_mode"] == "raw" for row in p3.REALMLP_CANDIDATES))
        self.assertEqual([row["candidate_id"] for row in p3.FT_CANDIDATES], [
            "fttransformer_log1p_wd1e5", "fttransformer_log1p_wd1e4"
        ])
        self.assertEqual([row["weight_decay"] for row in p3.FT_CANDIDATES], [1e-5, 1e-4])
        self.assertTrue(all(row["target_mode"] == "log1p" for row in p3.FT_CANDIDATES))
        self.assertEqual(len(p3.ALL_CANDIDATES), 4)
        self.assertEqual(p3.MAX_SCIENTIFIC_FITS, 5)
        self.assertEqual(p3.NUM_WORKERS, 0)

        real = p3.REALMLP_PARAMETERS
        for key, expected in {
            "device": "cpu", "random_state": 42, "n_cv": 1, "n_refit": 1,
            "n_repeats": 1, "val_fraction": 0.10, "n_epochs": 30,
            "use_early_stopping": True, "use_best_mean_epoch_for_cv": True,
        }.items():
            self.assertEqual(real[key], expected)
        ft = p3.FT_SCIENTIFIC_PARAMETERS
        for key, expected in {
            "d_token": 64, "n_blocks": 3, "attention_n_heads": 8,
            "attention_dropout": 0.20, "ffn_hidden_factor": 2.0,
            "ffn_dropout": 0.10, "residual_dropout": 0.0,
            "optimizer": "AdamW", "learning_rate": 1e-4,
            "gradient_clip_norm": 1.0, "maximum_epochs": 50,
            "early_stopping_patience": 8, "random_state": 42, "num_workers": 0,
        }.items():
            self.assertEqual(ft[key], expected)
        architecture = p3._ft_architecture(17, [4, 5])
        self.assertEqual(architecture["n_cont_features"], 17)
        self.assertEqual(architecture["cat_cardinalities"], [4, 5])
        self.assertEqual(architecture["d_block"], 64)
        self.assertEqual(architecture["n_blocks"], 3)
        self.assertEqual(architecture["attention_n_heads"], 8)

    def test_realmlp_external_validation_is_absent_from_model_fit_ast(self) -> None:
        function = _function_node("_fit_realmlp_candidate")
        model_fit_calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
        ]
        self.assertEqual(len(model_fit_calls), 1)
        call = model_fit_calls[0]
        self.assertEqual(ast.unparse(call.args[0]), "x_train")
        self.assertEqual(ast.unparse(call.args[1]), "y_train")
        self.assertEqual({item.arg for item in call.keywords}, {"cat_col_names"})
        names = {node.id for node in ast.walk(call) if isinstance(node, ast.Name)}
        self.assertFalse({"validation", "x_validation", "y_validation"} & names)
        source = ast.unparse(function)
        self.assertIn("preprocessor.fit_transform(train[features]", source)
        self.assertIn("preprocessor.transform(validation[features])", source)
        self.assertIn("_save_realmlp_metadata_evidence", source)
        self.assertIn("fitted_object_persisted_before_parsing", source)

    def test_realmlp_selected_epoch_uses_official_fit_params(self) -> None:
        fake_type = type("FakePyTabKitEstimator", (), {})
        fake_type.__module__ = "pytabkit.testing"
        model = fake_type()
        model.fit_params_ = {"stop_epoch": {"mae": 7}}
        epoch, evidence = p3._realmlp_selected_epoch(model)
        self.assertEqual(epoch, 7)
        self.assertEqual(evidence["final_selected_epoch"], 7)
        model.fit_params_ = {"stop_epoch": 4}
        epoch, _ = p3._realmlp_selected_epoch(model)
        self.assertEqual(epoch, 4)
        for invalid in ({}, {"stop_epoch": 0}, {"stop_epoch": 31}):
            model.fit_params_ = invalid
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                p3._realmlp_selected_epoch(model)

    def test_realmlp_metadata_inventory_is_bounded_and_omits_arrays(self) -> None:
        fake_type = type("FakePyTabKitEstimator", (), {})
        fake_type.__module__ = "pytabkit.testing"
        model = fake_type()
        model.fit_params_ = {
            "stop_epoch": {"mae": np.int64(2)},
            "large": np.arange(100_000, dtype=np.float32),
            "nested": [{"best_epoch": 2}],
        }
        inventory = p3._metadata_inventory(model, maximum_depth=10)
        array_rows = [row for row in inventory["records"] if row.get("large_value_omitted")]
        self.assertEqual(len(array_rows), 1)
        self.assertEqual(array_rows[0]["shape"], [100_000])
        self.assertNotIn("scalar_value", array_rows[0])
        self.assertFalse(inventory["large_arrays_and_tensors_serialized"])
        self.assertLessEqual(max(row["depth"] for row in inventory["records"]), 10)

    def test_realmlp_multiple_agreeing_paths_pass_and_conflicts_fail(self) -> None:
        fake_type = type("FakePyTabKitEstimator", (), {})
        fake_type.__module__ = "pytabkit.testing"
        model = fake_type()
        model.fit_params_ = {"stop_epoch": {"mae": 2.0}}
        model.alg_interface_ = SimpleNamespace(fit_params=[{"stop_epoch": {"mae": 2}}])
        evidence = p3._realmlp_epoch_evidence(model, maximum_epochs=2)
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["unique_normalized_epochs"], [2])
        model.alg_interface_.fit_params = [{"stop_epoch": {"mae": 1}}]
        evidence = p3._realmlp_epoch_evidence(model, maximum_epochs=2)
        self.assertEqual(evidence["status"], "FAIL")
        self.assertTrue(evidence["conflict"])
        self.assertIsNone(evidence["final_selected_epoch"])

    def test_realmlp_generic_max_epoch_and_unproved_zero_are_rejected(self) -> None:
        fake_type = type("FakePyTabKitEstimator", (), {})
        fake_type.__module__ = "pytabkit.testing"
        model = fake_type()
        model.max_epochs = 2
        model.n_epochs = 2
        evidence = p3._realmlp_epoch_evidence(model, maximum_epochs=2)
        self.assertEqual(evidence["status"], "FAIL")
        self.assertFalse(evidence["accepted_paths"])
        self.assertGreaterEqual(len(evidence["rejected_paths"]), 2)
        model.stop_epoch = 0
        evidence = p3._realmlp_epoch_evidence(model, maximum_epochs=2)
        self.assertEqual(evidence["status"], "FAIL")
        self.assertIn("zero is invalid", evidence["rejected_paths"][-1]["rejection_reason"])

    def test_realmlp_object_first_order_and_saved_parser_are_fit_free(self) -> None:
        smoke_source = ast.unparse(_function_node("_smoke_realmlp"))
        persistence = smoke_source.index("fitted_realmlp_smoke.joblib")
        completion = smoke_source.index("fit_completed.json")
        parsing = smoke_source.index("_complete_saved_realmlp_smoke")
        self.assertLess(persistence, completion)
        self.assertLess(completion, parsing)
        recovery = _function_node("_complete_saved_realmlp_smoke")
        fit_calls = [
            node
            for node in ast.walk(recovery)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
        ]
        self.assertEqual(fit_calls, [])
        self.assertIn("scientific_fit_count': 0", ast.unparse(recovery))

    def test_atomic_estimator_persistence_reloads_without_fit(self) -> None:
        directory = _fresh_directory("realmlp_object_first")
        estimator = SimpleNamespace(
            alg_interface_=SimpleNamespace(fit_params=[{"stop_epoch": {"mae": 2}}]),
            x_converter_=SimpleNamespace(),
        )
        path = p3._atomic_joblib(PROJECT_ROOT, directory / "estimator.joblib", estimator)
        self.assertGreater(path.stat().st_size, 0)
        loaded = joblib.load(path)
        self.assertTrue(hasattr(loaded, "alg_interface_"))
        parser_source = ast.unparse(_function_node("_save_realmlp_metadata_evidence"))
        self.assertNotIn(".fit(", parser_source)

    def test_ft_internal_selection_and_full_train_refit_are_isolated_static(self) -> None:
        candidate_source = ast.unparse(_function_node("_fit_ft_candidate"))
        self.assertIn("internal_fit = train.iloc[fit_index]", candidate_source)
        self.assertIn("internal_stop = train.iloc[stop_index]", candidate_source)
        self.assertIn("preprocessor.fit_transform(internal_fit[features]", candidate_source)
        self.assertIn("preprocessor.transform(internal_stop[features])", candidate_source)
        self.assertIn("preprocessor.transform(validation[features])", candidate_source)
        self.assertIn("stop_arrays=stop_arrays", candidate_source)
        self.assertIn("y_stop_original=internal_stop[TARGET]", candidate_source)

        refit_source = ast.unparse(_function_node("refit_models"))
        self.assertIn("if len(results) != 4", refit_source)
        self.assertIn("preprocessor.fit_transform(train[features]", refit_source)
        self.assertIn("preprocessor.transform(validation[features])", refit_source)
        self.assertIn("maximum_epochs=selected_epoch", refit_source)
        self.assertIn("stop_arrays=None", refit_source)
        self.assertIn("y_stop_original=None", refit_source)
        self.assertIn("patience=None", refit_source)
        self.assertIn("'early_stopping': False", refit_source)
        self.assertIn("'early_stopping_rows': 0", refit_source)

    def test_attempt_ledger_enforces_initial_plus_one_retry(self) -> None:
        workspace = _fresh_directory("attempt_ledger") / "fake_regression_v2"
        workspace.mkdir()
        operation = "scientific:tiny_candidate"
        first = p3._start_attempt(workspace, operation, maximum=2)
        p3._finish_attempt(workspace, operation, first, status="FAILED", error=RuntimeError("technical"))
        second = p3._start_attempt(workspace, operation, maximum=2)
        p3._finish_attempt(workspace, operation, second, status="PASS")
        with self.assertRaises(RuntimeError):
            p3._start_attempt(workspace, operation, maximum=2)
        ledger = json.loads(
            (workspace / "outputs" / "reports" / "prompt3_attempt_ledger.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual([row["status"] for row in ledger["operations"][operation]], ["FAILED", "PASS"])

    def test_attempt_fit_and_reviewer_invariants_are_present_static(self) -> None:
        fit_source = ast.unparse(_function_node("fit_candidates"))
        smoke_source = ast.unparse(_function_node("run_smoke"))
        verify_source = ast.unparse(_function_node("verify_prompt3"))
        self.assertIn("maximum=2", fit_source)
        self.assertIn("for candidate in candidates", fit_source)
        self.assertNotIn("ThreadPool", fit_source)
        self.assertNotIn("ProcessPool", fit_source)
        self.assertGreaterEqual(smoke_source.count("maximum=2"), 2)
        self.assertIn("prompt3_reviewer.json", verify_source)
        self.assertIn("reviewer_unresolved_critical", verify_source)
        self.assertIn("reviewer_unresolved_major", verify_source)
        config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))["prompt3"]
        self.assertEqual(config["max_scientific_fits"], 5)
        self.assertEqual(config["max_technical_retry_per_candidate"], 1)
        self.assertEqual(config["max_smoke_attempts_per_family"], 2)
        self.assertEqual(config["historical_realmlp_smoke_attempts"], 2)
        self.assertEqual(config["additional_realmlp_smoke_attempts_authorized"], 1)
        self.assertEqual(config["max_realmlp_smoke_attempts"], 3)
        self.assertEqual(config["max_metadata_only_repairs_after_saved_smoke"], 2)
        self.assertEqual(config["max_notebook_attempts"], 2)
        self.assertEqual(config["reviewer_cycles"], 1)

    def test_stage5_environment_is_required_and_available(self) -> None:
        p3._assert_stage5_environment()
        executable = Path(sys.executable).resolve().as_posix().lower()
        self.assertIn("/artifacts/environment/stage5_env/", executable)


class Prompt3BundlePredictionAndNotebookTests(unittest.TestCase):
    def test_realmlp_bundle_round_trip_and_clean_process_prediction(self) -> None:
        directory = _fresh_directory("realmlp_bundle")
        train = pd.DataFrame({"numeric": [1.0, np.nan, 3.0], "category": ["a", "b", None]})
        preprocessor = RealMLPPreprocessor(
            ["numeric", "category"], ["numeric"], ["category"]
        ).fit(train, row_hashes=["a", "b", "c"])
        model = DummyRegressor(strategy="constant", constant=7.0)
        model.constant_ = np.array([[7.0]])
        model.n_outputs_ = 1
        bundle = RealMLPBundle(
            metadata=_metadata("realmlp", ["numeric", "category"], "raw"),
            preprocessor=preprocessor,
            model=model,
        )
        joblib.dump(bundle, directory / "bundle.joblib", compress=3)
        (directory / "manifest.json").write_text(
            json.dumps({"status": "COMPLETE", "family": "realmlp", "artifact": "bundle.joblib"}),
            encoding="utf-8",
        )
        frame = pd.DataFrame({"category": ["unseen", None], "numeric": [np.nan, 5.0], "unused": [1, 2]})
        expected = load_bundle(directory).predict(frame)
        np.testing.assert_array_equal(expected, [7.0, 7.0])

        code = (
            "import json,sys,pandas as pd; "
            "sys.path.insert(0,sys.argv[1]); "
            "from deep_bundles import load_bundle; "
            "b=load_bundle(sys.argv[2]); "
            "x=pd.DataFrame({'category':['unseen',None],'numeric':[None,5.0],'unused':[1,2]}); "
            "print(json.dumps(b.predict(x).tolist()))"
        )
        environment = os.environ.copy()
        completed = subprocess.run(
            [sys.executable, "-c", code, str(SRC_DIR), str(directory)],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        clean = np.asarray(json.loads(completed.stdout.strip()), dtype=np.float64)
        self.assertLessEqual(float(np.max(np.abs(clean - expected))), 1e-6)

    def test_ft_bundle_state_round_trip_without_training(self) -> None:
        import torch

        directory = _fresh_directory("ft_bundle")
        train = pd.DataFrame({"numeric": [1.0, 2.0, 3.0, 4.0], "category": ["a", "b", None, "a"]})
        preprocessor = FTPreprocessor(
            ["numeric", "category"], ["numeric"], ["category"], n_quantiles=4
        ).fit(train, row_hashes=["a", "b", "c", "d"])
        architecture = p3._ft_architecture(1, preprocessor.cardinalities_)
        model = p3._build_ft_model(architecture, "cpu")
        torch.save(model.state_dict(), directory / "model_state.pt")
        joblib.dump(
            {
                "metadata": _metadata("fttransformer", ["numeric", "category"], "log1p"),
                "preprocessor": preprocessor,
                "architecture": architecture,
                "prediction_batch_size": 2,
            },
            directory / "bundle.joblib",
            compress=3,
        )
        (directory / "manifest.json").write_text(
            json.dumps({
                "status": "COMPLETE", "family": "fttransformer",
                "artifact": "bundle.joblib", "state_dict": "model_state.pt"
            }),
            encoding="utf-8",
        )
        frame = pd.DataFrame({"category": ["unseen", None, "a"], "numeric": [np.nan, 2.0, 3.0]})
        first = load_bundle(directory).predict(frame)
        second = load_bundle(directory).predict(frame)
        self.assertEqual(first.shape, (3,))
        self.assertTrue(np.isfinite(first).all())
        self.assertLessEqual(float(np.max(np.abs(first - second))), 1e-6)

    def test_prediction_frame_schema_alignment_and_failures(self) -> None:
        validation = pq.read_table(
            PROJECT_ROOT / p3.DEVELOPMENT_RELATIVE,
            columns=["row_hash", "development_role", p3.TARGET],
            filters=[("development_role", "=", "validation")],
        ).to_pandas()
        frame = p3._prediction_frame(
            validation["row_hash"],
            validation[p3.TARGET],
            np.zeros(len(validation), dtype=np.float64),
            model_id="tiny",
            family="realmlp",
            target_mode="raw",
        )
        self.assertEqual(list(frame.columns), EXPECTED_PREDICTION_COLUMNS)
        self.assertEqual(len(frame), 100_000)
        self.assertTrue(frame["row_hash"].is_unique)
        self.assertTrue(np.isfinite(frame["y_pred"]).all())
        p3._validate_prediction_alignment([frame, frame.copy()])
        changed_order = frame.copy()
        changed_order.loc[[0, 1], "row_hash"] = changed_order.loc[[1, 0], "row_hash"].to_numpy()
        with self.assertRaises(RuntimeError):
            p3._validate_prediction_alignment([frame, changed_order])
        changed_target = frame.copy()
        changed_target.loc[0, "y_true"] += 1.0
        with self.assertRaises(RuntimeError):
            p3._validate_prediction_alignment([frame, changed_target])
        with self.assertRaises(RuntimeError):
            p3._prediction_frame(
                ["a"], [1.0], [np.nan], model_id="bad", family="realmlp", target_mode="raw"
            )

    def test_atomic_prediction_parquet_is_zstd_and_reloadable(self) -> None:
        workspace = _fresh_directory("parquet") / "fake_regression_v2"
        workspace.mkdir()
        frame = pd.DataFrame({"row_hash": ["a", "b"], "y_true": [1.0, 2.0], "y_pred": [1.5, 2.5]})
        path = p3.atomic_parquet(workspace, "outputs/tmp/prompt3/tiny.parquet", frame)
        reloaded = pd.read_parquet(path)
        assert_frame_equal(frame, reloaded)
        parquet = pq.ParquetFile(path)
        codecs = {
            parquet.metadata.row_group(group).column(column).compression.upper()
            for group in range(parquet.metadata.num_row_groups)
            for column in range(parquet.metadata.row_group(group).num_columns)
        }
        self.assertEqual(codecs, {"ZSTD"})

    def test_notebook_builder_code_is_artifact_only_static(self) -> None:
        function = _function_node("build_notebook")
        code_cells: list[str] = []
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "new_code_cell"
                and node.args
            ):
                code_cells.append(ast.literal_eval(node.args[0]))
        self.assertGreaterEqual(len(code_cells), 20)
        code = "\n".join(code_cells)
        tree = ast.parse(code)
        prohibited = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"fit", "fit_transform", "fit_predict", "train"}
        }
        self.assertEqual(prohibited, set())
        lowered = code.lower().replace("\\", "/")
        self.assertNotIn("iid_holdout_features.parquet", lowered)
        self.assertNotIn("iid_holdout_targets.parquet", lowered)
        self.assertNotIn("hmda_2017_nationwide_all-records_labels.csv", lowered)
        self.assertNotIn("realmlp_td_regressor", lowered)
        self.assertNotIn("fttransformer(", lowered)

    def test_executed_notebook_is_fit_free_when_present(self) -> None:
        notebook_path = PROJECT_ROOT / p3.NOTEBOOK_RELATIVE
        if not notebook_path.exists():
            self.skipTest("Prompt 3 reporting notebook is created after scientific fits")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        tree = ast.parse(code)
        prohibited = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"fit", "fit_transform", "fit_predict", "train"}
        }
        self.assertEqual(prohibited, set())
        errors = [
            output
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        unexecuted = [
            cell
            for cell in notebook.get("cells", [])
            if cell.get("cell_type") == "code" and cell.get("execution_count") is None
        ]
        self.assertEqual(errors, [])
        self.assertEqual(unexecuted, [])


if __name__ == "__main__":
    unittest.main()
