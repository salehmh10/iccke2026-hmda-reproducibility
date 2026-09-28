"""Focused, fit-free verification for Regression V2 Prompt 4A.

The suite uses synthetic arrays for scientific formulas and reads only saved
Prompt 2/3 prediction and report artifacts for integration checks. It never
opens Raw, the ``data`` directory, Development, or either IID Parquet file.
Files made by tests stay below ``outputs/tmp/prompt4a/tests``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
TEST_TMP = PROJECT_ROOT / "outputs" / "tmp" / "prompt4a" / "tests"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import ensemble_utils as ensembles
import prompt4_metrics as metrics
import prompt4a_experiments as p4a
import tail_models as tails


EXPECTED_VALIDATION_DIGEST = (
    "676b577233627c8b237d214a29eeb798e099d028583c8b73068b151a2a204290"
)
EXPECTED_FEATURE_CONTRACT = "main_without_sensitive_without_lender"
EXPECTED_PREDICTIONS = {
    "lasso": Path("outputs/predictions/prompt2/validation/selected_lasso.parquet"),
    "histgradientboosting": Path(
        "outputs/predictions/prompt2/validation/selected_histgradientboosting.parquet"
    ),
    "catboost": Path(
        "outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet"
    ),
    "lightgbm": Path(
        "outputs/predictions/prompt2/validation/selected_lightgbm_without_lender.parquet"
    ),
    "xgboost": Path(
        "outputs/predictions/prompt2/validation/selected_xgboost_without_lender.parquet"
    ),
    "realmlp": Path("outputs/predictions/prompt3/validation/selected_realmlp.parquet"),
    "fttransformer": Path(
        "outputs/predictions/prompt3/validation/selected_fttransformer.parquet"
    ),
}
EXPECTED_REGRESSION_COLUMNS = [
    "row_hash",
    "y_true",
    "y_pred",
    "candidate_id",
    "candidate_type",
    "global_base",
    "selection_or_audit_role",
]
EXPECTED_GATE_COLUMNS = [
    "row_hash",
    "operational_tail_true",
    "p_tail",
    "predicted_tail_035",
    "predicted_tail_050",
    "predicted_tail_065",
    "selection_or_audit_role",
]


def _ordered_digest(values) -> str:
    text = "\n".join(pd.Series(values, copy=False).astype(str).tolist())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fresh_directory(name: str) -> Path:
    path = (TEST_TMP / name).resolve()
    path.relative_to(TEST_TMP.resolve())
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_tree(module) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _function_node(module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _source_tree(module).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _synthetic_tail_frame(rows: int = 3) -> pd.DataFrame:
    values: dict[str, object] = {}
    for name in tails.PRIMARY_FEATURE_NAMES:
        if name in tails.NUMERIC_FEATURE_NAMES:
            values[name] = np.arange(1, rows + 1, dtype=np.float64)
        else:
            values[name] = [f"{name}_{index % 2}" for index in range(rows)]
    return pd.DataFrame(values)


class _FitFreeRegressionModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[tails.NUMERIC_FEATURE_NAMES[0]].to_numpy(dtype=np.float64)


def setUpModule() -> None:
    TEST_TMP.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def tearDownModule() -> None:
    if TEST_TMP.exists():
        TEST_TMP.resolve().relative_to(
            (PROJECT_ROOT / "outputs" / "tmp" / "prompt4a").resolve()
        )
        shutil.rmtree(TEST_TMP)


class Prompt4AHandoffAndBoundaryTests(unittest.TestCase):
    def test_prompt3_handoff_is_pass_and_state_is_resumable(self) -> None:
        handoff = p4a.validate_prompt3_handoff(PROJECT_ROOT)
        self.assertEqual(handoff["status"], "PASS")
        task = (PROJECT_ROOT / "TASK.md").read_text(encoding="utf-8")
        self.assertTrue(
            any(
                state in task
                for state in (
                    "Prompt 3 COMPLETE",
                    "Prompt 4A IN PROGRESS",
                    "Prompt 4A BLOCKED",
                    "Prompt 4A COMPLETE",
                )
            )
        )
        for name in (
            "PROMPT2_READY.json",
            "prompt2_verification.json",
            "PROMPT3_READY.json",
            "prompt3_verification.json",
            "prompt2_prediction_manifest.json",
            "prompt3_prediction_manifest.json",
            "feature_roles.json",
        ):
            self.assertTrue((PROJECT_ROOT / "outputs" / "reports" / name).is_file())

    def test_exact_primary_contract_excludes_target_sensitive_and_lender(self) -> None:
        roles = json.loads(
            (PROJECT_ROOT / "outputs" / "reports" / "feature_roles.json").read_text(
                encoding="utf-8"
            )
        )
        features = roles["contracts"][EXPECTED_FEATURE_CONTRACT]
        forbidden = (
            set(roles["sensitive_fields"])
            | set(roles["audit_only_fields"])
            | set(roles["target_and_alias_exclusions"])
            | {"respondent_id", "operational_tail"}
        )
        self.assertEqual(len(features), 35)
        self.assertEqual(len(set(features)), 35)
        self.assertFalse(forbidden & set(features))

    def test_seven_saved_predictions_are_exactly_aligned(self) -> None:
        frames: list[pd.DataFrame] = []
        for family, relative in EXPECTED_PREDICTIONS.items():
            with self.subTest(family=family):
                path = PROJECT_ROOT / relative
                self.assertTrue(path.is_file())
                frame = pd.read_parquet(path)
                self.assertEqual(len(frame), 100_000)
                self.assertTrue(frame["row_hash"].is_unique)
                self.assertTrue(np.isfinite(frame["y_true"].to_numpy(np.float64)).all())
                self.assertTrue(np.isfinite(frame["y_pred"].to_numpy(np.float64)).all())
                frames.append(frame)

        reference_hash = frames[0]["row_hash"].astype(str).to_numpy()
        reference_target = frames[0]["y_true"].to_numpy(np.float64)
        self.assertEqual(_ordered_digest(reference_hash), EXPECTED_VALIDATION_DIGEST)
        for frame in frames[1:]:
            np.testing.assert_array_equal(
                frame["row_hash"].astype(str).to_numpy(), reference_hash
            )
            np.testing.assert_array_equal(
                frame["y_true"].to_numpy(np.float64), reference_target
            )

    def test_read_and_write_guards_close_protected_and_outside_paths(self) -> None:
        allowed = p4a.guard_read_path(PROJECT_ROOT, next(iter(EXPECTED_PREDICTIONS.values())))
        allowed.relative_to(PROJECT_ROOT.resolve())
        for prohibited in (
            "data/raw.csv",
            "outputs/data/iid_holdout_features.parquet",
            "outputs/data/iid_holdout_targets.parquet",
            REPOSITORY_ROOT / "outside.json",
        ):
            with self.subTest(path=str(prohibited)), self.assertRaises(PermissionError):
                p4a.guard_read_path(PROJECT_ROOT, prohibited)
        with self.assertRaises(PermissionError):
            p4a.guard_write_path(PROJECT_ROOT, "data/changed.csv")
        with self.assertRaises(PermissionError):
            p4a.guard_write_path(PROJECT_ROOT, REPOSITORY_ROOT / "outside.json")
        writable = p4a.guard_write_path(
            PROJECT_ROOT, "outputs/tmp/prompt4a/tests/allowed.json"
        )
        writable.relative_to(PROJECT_ROOT.resolve())

    def test_no_forbidden_final_or_iid_prediction_artifact(self) -> None:
        self.assertFalse((PROJECT_ROOT / "outputs" / "reports" / "FINAL_PRE_IID_FREEZE.json").exists())
        prediction_root = PROJECT_ROOT / "outputs" / "predictions"
        forbidden = [
            path
            for path in prediction_root.rglob("*")
            if path.is_file() and "iid" in path.name.lower()
        ]
        self.assertEqual(forbidden, [])


class Prompt4AMetricTests(unittest.TestCase):
    def test_standard_and_operational_tail_definitions_remain_distinct(self) -> None:
        actual = np.array([1.0, 2.0, 9.0, 10.0, 10.0, 11.0])
        standard = metrics.quantile_membership(actual, 0.90)
        operational = metrics.operational_tail_membership(actual, 10.0)
        np.testing.assert_array_equal(operational, [False, False, False, False, False, True])
        self.assertTrue(standard[-1])
        self.assertFalse(operational[3])
        self.assertFalse(operational[4])

    def test_metric_formulas_bottom_tail_boundary_and_signed_error(self) -> None:
        actual = np.arange(1.0, 101.0)
        error = np.where(actual < 90.0, 1.0, -10.0)
        predicted = actual + error
        result = metrics.compute_regression_metrics(actual, predicted)
        top10 = actual >= np.quantile(actual, 0.90)
        top05 = actual >= np.quantile(actual, 0.95)
        boundary = (actual >= np.quantile(actual, 0.85)) & (
            actual <= np.quantile(actual, 0.95)
        )
        absolute = np.abs(error)
        self.assertAlmostEqual(result["bottom_90_mae"], float(np.mean(absolute[~top10])))
        self.assertAlmostEqual(result["top_decile_mae"], float(np.mean(absolute[top10])))
        self.assertAlmostEqual(
            result["top_five_percent_mae"], float(np.mean(absolute[top05]))
        )
        self.assertAlmostEqual(
            result["p85_to_p95_boundary_mae"], float(np.mean(absolute[boundary]))
        )
        self.assertAlmostEqual(result["mean_signed_error"], float(np.mean(error)))
        self.assertLess(result["top_decile_signed_error"], 0.0)
        self.assertEqual(result["top_decile_underprediction_rate"], 1.0)

    def test_rmsle_clips_only_predictions_and_rejects_invalid_inputs(self) -> None:
        result = metrics.compute_regression_metrics([1.0, 2.0], [-3.0, 4.0])
        expected = np.sqrt(np.mean((np.log1p([0.0, 4.0]) - np.log1p([1.0, 2.0])) ** 2))
        self.assertAlmostEqual(result["rmsle"], expected)
        self.assertEqual(result["negative_prediction_count"], 1)
        with self.assertRaises(ValueError):
            metrics.compute_regression_metrics([1.0], [np.nan])
        with self.assertRaises(ValueError):
            metrics.compute_regression_metrics([-1.0], [0.0])

    def test_preliminary_ranking_uses_relative_ties_and_complexity(self) -> None:
        rows = [
            {
                "candidate_id": "lowest_mae",
                "mae": 100.0,
                "rmse": 120.0,
                "top_decile_mae": 300.0,
                "top_five_percent_mae": 400.0,
                "inference_complexity": 2,
            },
            {
                "candidate_id": "relative_tie",
                "mae": 100.25,
                "rmse": 110.0,
                "top_decile_mae": 250.0,
                "top_five_percent_mae": 350.0,
                "inference_complexity": 3,
            },
        ]
        self.assertEqual(metrics.select_reference(rows)["candidate_id"], "relative_tie")
        rows[1]["mae"] = 100.2500001
        self.assertEqual(metrics.select_reference(rows)["candidate_id"], "lowest_mae")

    def test_provisional_acceptance_pass_partial_and_fail(self) -> None:
        base = {
            "mae": 100.0,
            "rmse": 120.0,
            "bottom_90_mae": 50.0,
            "top_decile_mae": 300.0,
            "top_decile_signed_error": -100.0,
            "top_decile_underprediction_rate": 0.8,
        }
        passed = {
            "mae": 99.0,
            "rmse": 120.2,
            "bottom_90_mae": 50.1,
            "top_decile_mae": 291.0,
            "top_decile_signed_error": -90.0,
            "top_decile_underprediction_rate": 0.7,
        }
        failed = {
            "mae": 101.0,
            "rmse": 121.0,
            "bottom_90_mae": 51.0,
            "top_decile_mae": 300.0,
            "top_decile_signed_error": -110.0,
            "top_decile_underprediction_rate": 0.8,
        }
        self.assertEqual(
            metrics.provisional_acceptance(passed, base)["provisional_acceptance_status"],
            "PASS",
        )
        partial = dict(failed, mae=99.0)
        self.assertEqual(
            metrics.provisional_acceptance(partial, base)["provisional_acceptance_status"],
            "PARTIAL",
        )
        self.assertEqual(
            metrics.provisional_acceptance(failed, base)["provisional_acceptance_status"],
            "FAIL",
        )

    def test_paired_bootstrap_is_aligned_deterministic_and_frozen(self) -> None:
        actual = np.arange(1.0, 101.0)
        reference = actual + np.where(np.arange(100) % 2, 2.0, -2.0)
        candidate = actual + np.where(np.arange(100) % 2, 1.0, -1.0)
        first = metrics.paired_mae_bootstrap(actual, reference, candidate)
        second = metrics.paired_mae_bootstrap(actual, reference, candidate)
        self.assertEqual(first, second)
        self.assertEqual(first["n_resamples"], 300)
        self.assertEqual(first["random_state"], 42)
        self.assertAlmostEqual(first["mae_difference"], -1.0)
        self.assertEqual(first["win_proportion"], 1.0)
        with self.assertRaises(ValueError):
            metrics.paired_mae_bootstrap(actual, reference[:-1], candidate)


class Prompt4AEnsembleTests(unittest.TestCase):
    EXPECTED_FIXED = {
        "ens_boost_equal": (
            ["catboost", "lightgbm", "xgboost"],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ),
        "ens_boost_cat050": (
            ["catboost", "lightgbm", "xgboost"],
            [0.50, 0.25, 0.25],
        ),
        "ens_boost_cat060": (
            ["catboost", "lightgbm", "xgboost"],
            [0.60, 0.20, 0.20],
        ),
        "ens_cat_ft_equal": (["catboost", "fttransformer"], [0.50, 0.50]),
        "ens_cat_realmlp_equal": (["catboost", "realmlp"], [0.50, 0.50]),
        "ens_cat_lgb_ft": (
            ["catboost", "lightgbm", "fttransformer"],
            [0.40, 0.20, 0.40],
        ),
        "ens_cat_ft_realmlp": (
            ["catboost", "fttransformer", "realmlp"],
            [0.40, 0.40, 0.20],
        ),
        "ens_boost_ft_equal": (
            ["catboost", "lightgbm", "xgboost", "fttransformer"],
            [0.25, 0.25, 0.25, 0.25],
        ),
    }

    def test_exact_no_fit_candidates_weights_and_grids(self) -> None:
        expected_ids = set(self.EXPECTED_FIXED) | {
            "ens_convex_boosting",
            "ens_convex_boosting_deep",
        }
        self.assertEqual(set(ensembles.NO_FIT_ENSEMBLES), expected_ids)
        self.assertEqual(len(ensembles.NO_FIT_ENSEMBLES), 10)
        for candidate_id, (members, weights) in self.EXPECTED_FIXED.items():
            with self.subTest(candidate_id=candidate_id):
                definition = ensembles.NO_FIT_ENSEMBLES[candidate_id]
                self.assertEqual(definition["kind"], "fixed")
                self.assertEqual(list(definition["members"]), members)
                np.testing.assert_allclose(definition["weights"], weights, rtol=0, atol=1e-15)
                self.assertAlmostEqual(float(np.sum(definition["weights"])), 1.0)
        self.assertEqual(
            list(ensembles.NO_FIT_ENSEMBLES["ens_convex_boosting"]["members"]),
            ["catboost", "lightgbm", "xgboost"],
        )
        self.assertEqual(
            list(ensembles.NO_FIT_ENSEMBLES["ens_convex_boosting_deep"]["members"]),
            ["catboost", "lightgbm", "xgboost", "realmlp", "fttransformer"],
        )
        self.assertEqual(tuple(ensembles.HARD_THRESHOLDS), (0.35, 0.50, 0.65))
        self.assertEqual(tuple(ensembles.ALPHA_GRID), (0.50, 0.75, 1.00))

    def test_validation_selection_audit_split_is_exact_and_deterministic(self) -> None:
        target = np.repeat(np.arange(1.0, 11.0), 10)
        hashes = np.array([f"validation-{index:03d}" for index in range(100)])
        first = ensembles.deterministic_validation_split(
            target,
            hashes,
            random_state=42,
            selection_rows=70,
            audit_rows=30,
        )
        second = ensembles.deterministic_validation_split(
            target,
            hashes,
            random_state=42,
            selection_rows=70,
            audit_rows=30,
        )
        selection, audit, evidence = first
        np.testing.assert_array_equal(selection, second[0])
        np.testing.assert_array_equal(audit, second[1])
        self.assertEqual(evidence, second[2])
        self.assertEqual(len(selection), 70)
        self.assertEqual(len(audit), 30)
        self.assertEqual(np.intersect1d(selection, audit).size, 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([selection, audit])), np.arange(100)
        )
        self.assertEqual(_ordered_digest(hashes[selection]), evidence["selection_row_hash_digest"])
        self.assertEqual(_ordered_digest(hashes[audit]), evidence["audit_row_hash_digest"])

    def test_weighted_ensemble_formula_and_convex_constraints(self) -> None:
        matrix = np.array(
            [
                [1.0, 4.0, 7.0],
                [2.0, 5.0, 8.0],
                [3.0, 6.0, 9.0],
                [4.0, 7.0, 10.0],
            ]
        )
        weights = np.array([0.50, 0.25, 0.25])
        np.testing.assert_allclose(
            ensembles.apply_ensemble(matrix, weights), matrix @ weights, rtol=0, atol=1e-15
        )
        with self.assertRaises(ValueError):
            ensembles.apply_ensemble(matrix, [0.5, -0.1, 0.6])
        with self.assertRaises(ValueError):
            ensembles.apply_ensemble(matrix, [0.5, 0.2, 0.2])

        target = matrix[:, 0] * 0.7 + matrix[:, 1] * 0.3
        first = ensembles.optimize_convex_mae(matrix, target)
        second = ensembles.optimize_convex_mae(matrix, target)
        self.assertEqual(first["status"], "COMPLETE")
        np.testing.assert_allclose(first["weights"], second["weights"], rtol=0, atol=1e-12)
        learned = np.asarray(first["weights"], dtype=np.float64)
        self.assertTrue(np.all(learned >= 0.0))
        self.assertAlmostEqual(float(np.sum(learned)), 1.0, places=10)
        self.assertEqual(first["solver_starts"], 1)

    def test_prediction_alignment_rejects_order_and_target_changes(self) -> None:
        def frame(prediction: list[float]) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "row_hash": ["a", "b", "c"],
                    "y_true": [1.0, 2.0, 3.0],
                    "y_pred": prediction,
                    "model_id": "tiny",
                    "family": "tiny",
                    "feature_contract": EXPECTED_FEATURE_CONTRACT,
                    "target_mode": "raw",
                }
            )

        first = frame([1.1, 2.1, 3.1])
        second = frame([0.9, 1.9, 2.9])
        evidence = ensembles.validate_prediction_alignment(
            {"first": first, "second": second}, expected_rows=3
        )
        self.assertEqual(evidence["status"], "PASS")
        self.assertTrue(evidence["exact_row_order"])
        self.assertTrue(evidence["exact_target_equality"])
        changed_order = second.copy()
        changed_order.loc[[0, 1], "row_hash"] = changed_order.loc[[1, 0], "row_hash"].to_numpy()
        with self.assertRaises(ValueError):
            ensembles.validate_prediction_alignment(
                {"first": first, "second": changed_order}, expected_rows=3
            )
        changed_target = second.copy()
        changed_target.loc[0, "y_true"] += 1.0
        with self.assertRaises(ValueError):
            ensembles.validate_prediction_alignment(
                {"first": first, "second": changed_target}, expected_rows=3
            )

    def test_hard_routing_and_soft_mixture_formulas(self) -> None:
        global_prediction = np.array([10.0, 20.0, 30.0, 40.0])
        specialist = np.array([100.0, 200.0, 300.0, 400.0])
        probability = np.array([0.34, 0.35, 0.64, 0.65])
        np.testing.assert_array_equal(
            ensembles.hard_route(global_prediction, specialist, probability, 0.35),
            [10.0, 200.0, 300.0, 400.0],
        )
        expected = global_prediction + 0.75 * probability * (
            specialist - global_prediction
        )
        np.testing.assert_allclose(
            ensembles.soft_mix(global_prediction, specialist, probability, 0.75),
            expected,
            rtol=0,
            atol=1e-12,
        )

    def test_correlation_report_is_complete_for_seven_models(self) -> None:
        base = np.arange(20.0)
        frame = pd.DataFrame(
            {
                name: base * (index + 1) + (index % 2)
                for index, name in enumerate(EXPECTED_PREDICTIONS)
            }
        )
        report = ensembles.correlation_report(frame)
        self.assertEqual(
            set(report["metric"]),
            {"pearson_correlation", "spearman_correlation", "mean_absolute_difference"},
        )
        self.assertEqual(len(report), 3 * 7 * 7)
        self.assertEqual(set(report["model_a"]), set(EXPECTED_PREDICTIONS))
        self.assertEqual(set(report["model_b"]), set(EXPECTED_PREDICTIONS))


class Prompt4ATailDefinitionAndIsolationTests(unittest.TestCase):
    def test_gate_specialist_and_weighted_configurations_are_exact(self) -> None:
        expected_gate = {
            "loss_function": "Logloss",
            "eval_metric": "PRAUC",
            "iterations": 1500,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 10,
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "thread_count": 4,
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        expected_specialist = {
            "loss_function": "MAE",
            "iterations": 2000,
            "depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 20,
            "random_strength": 1,
            "random_seed": 42,
            "thread_count": 4,
            "early_stopping_rounds": 100,
            "verbose": False,
        }
        for key, value in expected_gate.items():
            self.assertEqual(tails.GATE_PARAMETERS[key], value)
        for key, value in expected_specialist.items():
            self.assertEqual(tails.SPECIALIST_PARAMETERS[key], value)
        self.assertEqual(tuple(tails.TAIL_WEIGHTS), (2, 4))
        configuration = tails.WEIGHTED_PARAMETERS
        self.assertEqual(configuration["iterations"], 2000)
        self.assertEqual(configuration["random_seed"], 42)
        self.assertEqual(configuration["thread_count"], 4)
        self.assertNotIn("early_stopping_rounds", configuration)
        self.assertEqual(set(tails.TAIL_WEIGHTED_CONFIGS), {2, 4})

    def test_operational_tail_and_sample_weight_formulas(self) -> None:
        actual = np.array([9.0, 10.0, 10.0, 11.0, 20.0])
        labels = metrics.operational_tail_membership(actual, 10.0)
        np.testing.assert_array_equal(labels, [False, False, False, True, True])
        np.testing.assert_array_equal(tails.operational_tail_labels(actual, 10.0), labels)
        np.testing.assert_array_equal(
            tails.make_tail_weights(actual, 10.0, 2), [1.0, 1.0, 1.0, 2.0, 2.0]
        )
        np.testing.assert_array_equal(
            tails.make_tail_weights(actual, 10.0, 4), [1.0, 1.0, 1.0, 4.0, 4.0]
        )
        with self.assertRaises(ValueError):
            tails.make_tail_weights(actual, 10.0, 3)

    def test_internal_tail_split_is_deterministic_disjoint_and_train_only(self) -> None:
        target = np.repeat(np.arange(1.0, 11.0), 10)
        hashes = np.array([f"train-{index:03d}" for index in range(100)])
        external = np.array([f"validation-{index:03d}" for index in range(20)])
        q90_train = float(np.quantile(target, 0.90))
        first = tails.make_internal_tail_split(
            target,
            hashes,
            q90_train,
            external,
            random_state=42,
            fit_rows=90,
            stop_rows=10,
        )
        second = tails.make_internal_tail_split(
            target,
            hashes,
            q90_train,
            external,
            random_state=42,
            fit_rows=90,
            stop_rows=10,
        )
        fit_index, stop_index, evidence = first
        np.testing.assert_array_equal(fit_index, second[0])
        np.testing.assert_array_equal(stop_index, second[1])
        self.assertEqual(evidence, second[2])
        self.assertEqual(len(fit_index), 90)
        self.assertEqual(len(stop_index), 10)
        self.assertEqual(np.intersect1d(fit_index, stop_index).size, 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([fit_index, stop_index])), np.arange(100)
        )
        self.assertEqual(evidence["validation_overlap_rows"], 0)
        self.assertEqual(_ordered_digest(hashes[fit_index]), evidence["fit_row_hash_digest"])
        self.assertEqual(_ordered_digest(hashes[stop_index]), evidence["stop_row_hash_digest"])
        with self.assertRaises(ValueError):
            tails.make_internal_tail_split(
                target,
                hashes,
                q90_train,
                [hashes[0]],
                random_state=42,
                fit_rows=90,
                stop_rows=10,
            )

    def test_preprocessor_preserves_sources_and_frozen_feature_roles(self) -> None:
        source = _synthetic_tail_frame()
        source.loc[1, list(tails.NUMERIC_FEATURE_NAMES)] = np.nan
        source.loc[1, list(tails.CATEGORICAL_FEATURE_NAMES)] = None
        before = source.copy(deep=True)
        preprocessor = tails.ExplicitCatBoostPreprocessor().fit(source)
        transformed = preprocessor.transform(source)
        pd.testing.assert_frame_equal(source, before)
        self.assertEqual(list(transformed.columns), list(tails.PRIMARY_FEATURE_NAMES))
        self.assertEqual(len(preprocessor.cat_feature_indices_), 18)
        self.assertEqual(len(preprocessor.numeric_features_), 17)
        self.assertFalse(
            {"loan_amount_000s", "operational_tail", "respondent_id"}
            & set(preprocessor.feature_names_in_)
        )
        self.assertEqual(
            transformed.loc[1, tails.CATEGORICAL_FEATURE_NAMES[0]], tails.MISSING_SENTINEL
        )
        self.assertTrue(
            all(
                transformed[name].dtype == np.float64
                for name in tails.NUMERIC_FEATURE_NAMES
            )
        )

    def test_fit_isolation_rejects_validation_overlap_and_wrong_full_refit(self) -> None:
        authorized = ["train-a", "train-b", "train-c", "train-d"]
        evidence = tails.validate_fit_isolation(
            ["train-a", "train-b"],
            authorized,
            stop_row_hashes=["train-c", "train-d"],
            validation_row_hashes=["validation-a"],
        )
        self.assertEqual(evidence["status"], "PASS")
        self.assertEqual(evidence["fit_stop_overlap"], 0)
        self.assertEqual(evidence["validation_overlap"], 0)
        with self.assertRaises(ValueError):
            tails.validate_fit_isolation(
                ["train-a", "validation-a"],
                authorized,
                validation_row_hashes=["validation-a"],
            )
        with self.assertRaises(ValueError):
            tails.validate_fit_isolation(
                ["train-a", "train-b"], authorized, require_full_authorized_fit=True
            )

    def test_fit_free_regression_bundle_round_trip(self) -> None:
        raw = _synthetic_tail_frame()
        preprocessor = tails.ExplicitCatBoostPreprocessor().fit(raw)
        metadata = tails.TailModelMetadata(
            model_id="tiny_tail_bundle",
            model_role="test_only",
            feature_names=list(tails.PRIMARY_FEATURE_NAMES),
            feature_contract=tails.PRIMARY_FEATURE_CONTRACT,
            model_configuration={"test_only": True},
            seed=42,
            selected_iteration=1,
            training_row_count=3,
            train_membership_digest="1" * 64,
            q90_train=10.0,
            target_definition="raw loan_amount_000s",
            package_versions={},
        )
        bundle = tails.RegressionBundle(metadata, preprocessor, _FitFreeRegressionModel())
        workspace = _fresh_directory("bundle") / "regression_v2"
        destination = workspace / "outputs" / "models" / "prompt4a" / "tiny"
        manifest = tails.save_regression_bundle(bundle, destination)
        self.assertEqual(manifest["status"], "COMPLETE")
        reloaded = tails.load_regression_bundle(destination)
        np.testing.assert_array_equal(reloaded.predict(raw), bundle.predict(raw))
        self.assertEqual(reloaded.metadata, metadata)

    def test_fit_primitives_keep_selection_stop_and_full_refit_roles_static(self) -> None:
        gate_source = ast.unparse(_function_node(tails, "fit_gate"))
        regression_source = ast.unparse(_function_node(tails, "fit_regressor"))
        self.assertIn("catboost", gate_source.lower())
        self.assertIn("catboost", regression_source.lower())
        self.assertNotIn("ThreadPool", gate_source)
        self.assertNotIn("ProcessPool", gate_source)
        self.assertNotIn("ThreadPool", regression_source)
        self.assertNotIn("ProcessPool", regression_source)
        for source in (gate_source, regression_source):
            self.assertNotIn("respondent_id", source)
            self.assertNotIn("iid_holdout", source.lower())

    def test_prompt4a_fit_calls_are_bounded_to_the_six_frozen_roles(self) -> None:
        # Smoke fits are explicitly outside the scientific-fit budget. Inspect
        # only the scientific runner so this check does not miscount them.
        function = _function_node(p4a, "fit_tail_models")
        source = ast.unparse(function)
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id in {"fit_gate", "fit_regressor"})
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"fit", "fit_transform", "partial_fit"}
                )
            )
        ]
        self.assertEqual(len(calls), 5)
        self.assertIn("for weight in TAIL_WEIGHTS", source)
        self.assertIn("set(ledger['completed_roles']) != set(SCIENTIFIC_FIT_ROLES)", source)
        metadata_source = ast.unparse(_function_node(p4a, "_bundle_metadata"))
        self.assertIn("'cache_identity': identity", metadata_source)
        self.assertEqual(len(p4a.SCIENTIFIC_FIT_ROLES), 6)


class Prompt4ADesignAndStaticSafetyTests(unittest.TestCase):
    def test_fit_budget_roles_config_and_no_parallel_execution(self) -> None:
        self.assertEqual(p4a.MAX_SCIENTIFIC_FITS, 6)
        self.assertEqual(
            tuple(p4a.SCIENTIFIC_FIT_ROLES),
            (
                "gate_selection",
                "gate_full_refit",
                "specialist_selection",
                "specialist_full_refit",
                "tail_weighted_catboost_w2",
                "tail_weighted_catboost_w4",
            ),
        )
        config = json.loads((PROJECT_ROOT / "config.json").read_text(encoding="utf-8"))[
            "prompt4a"
        ]
        self.assertEqual(config["max_scientific_fits"], 6)
        self.assertEqual(config["max_technical_retry_per_fit"], 1)
        self.assertEqual(config["max_smoke_attempts"], 2)
        self.assertEqual(config["max_notebook_attempts"], 2)
        self.assertEqual(config["reviewer_cycles"], 1)
        self.assertEqual(config["n_threads"], 4)
        source = Path(p4a.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ThreadPoolExecutor", source)
        self.assertNotIn("ProcessPoolExecutor", source)

    def test_notebook_builder_is_fit_free_and_artifact_only_static(self) -> None:
        if not hasattr(p4a, "build_notebook"):
            self.skipTest("Notebook builder is added after the experiment core")
        function = _function_node(p4a, "build_notebook")
        code_cells: list[str] = []
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "new_code_cell"
                and node.args
            ):
                try:
                    code_cells.append(ast.literal_eval(node.args[0]))
                except (ValueError, TypeError):
                    continue
        if not code_cells:
            self.skipTest("Notebook cells are assembled by a separate fit-free helper")
        code = "\n".join(code_cells)
        tree = ast.parse(code)
        prohibited = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"fit", "fit_transform", "fit_predict", "partial_fit", "train"}
        }
        self.assertEqual(prohibited, set())
        lowered = code.lower().replace("\\", "/")
        for text in (
            "iid_holdout_features.parquet",
            "iid_holdout_targets.parquet",
            "outputs/data/development.parquet",
            "catboostclassifier",
            "catboostregressor",
        ):
            self.assertNotIn(text, lowered)


class Prompt4AArtifactTests(unittest.TestCase):
    def test_frozen_design_matches_exact_counts_and_grids_when_present(self) -> None:
        path = PROJECT_ROOT / "outputs" / "reports" / "prompt4a_frozen_design.json"
        if not path.exists():
            self.skipTest("Prompt 4A frozen design has not been created")
        design = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(design["status"], "FROZEN")
        serialized = json.dumps(design, sort_keys=True)
        for value in ("0.35", "0.5", "0.65", "0.75", "1.0"):
            self.assertIn(value, serialized)
        self.assertEqual(design["fit_budget"]["max_scientific_fits"], 6)
        self.assertFalse(design.get("final_model_selected", False))
        self.assertFalse(design.get("final_model_frozen", False))

    def test_prompt4a_predictions_are_aligned_zstd_when_manifest_present(self) -> None:
        path = PROJECT_ROOT / "outputs" / "reports" / "prompt4a_prediction_manifest.json"
        if not path.exists():
            self.skipTest("Prompt 4A predictions have not been completed")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "PASS")
        artifacts = manifest["artifacts"]
        self.assertEqual(len(artifacts), 26)
        reference_hash: np.ndarray | None = None
        for entry in artifacts:
            artifact_path = PROJECT_ROOT / entry["path"]
            frame = pd.read_parquet(artifact_path)
            self.assertEqual(len(frame), 100_000)
            self.assertTrue(frame["row_hash"].is_unique)
            current_hash = frame["row_hash"].astype(str).to_numpy()
            if reference_hash is None:
                reference_hash = current_hash
            else:
                np.testing.assert_array_equal(current_hash, reference_hash)
            if "p_tail" in frame.columns:
                self.assertEqual(list(frame.columns), EXPECTED_GATE_COLUMNS)
                self.assertTrue(np.isfinite(frame["p_tail"].to_numpy(np.float64)).all())
                self.assertTrue(frame["p_tail"].between(0.0, 1.0).all())
            else:
                self.assertEqual(list(frame.columns), EXPECTED_REGRESSION_COLUMNS)
                self.assertTrue(np.isfinite(frame[["y_true", "y_pred"]]).all().all())
            parquet = pq.ParquetFile(artifact_path)
            codecs = {
                parquet.metadata.row_group(group).column(column).compression.upper()
                for group in range(parquet.metadata.num_row_groups)
                for column in range(parquet.metadata.row_group(group).num_columns)
            }
            self.assertEqual(codecs, {"ZSTD"})

    def test_executed_notebook_is_fit_free_and_inline_when_present(self) -> None:
        path = PROJECT_ROOT / "notebooks" / "04A_INITIAL_ENSEMBLE_AND_TAIL_EXPERIMENTS.ipynb"
        if not path.exists():
            self.skipTest("Prompt 4A reporting notebook has not been created")
        notebook = json.loads(path.read_text(encoding="utf-8"))
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
            and node.func.attr
            in {"fit", "fit_transform", "fit_predict", "partial_fit", "train"}
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
        images = [
            output
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
            if "image/png" in output.get("data", {})
        ]
        displays = [
            output
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
            if output.get("output_type") in {"display_data", "execute_result"}
        ]
        self.assertEqual(errors, [])
        self.assertEqual(unexecuted, [])
        self.assertGreaterEqual(len(images), 1)
        self.assertGreaterEqual(len(displays), 10)


if __name__ == "__main__":
    unittest.main()
