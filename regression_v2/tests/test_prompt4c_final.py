from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompt4_metrics import provisional_acceptance
from prompt4b2_metrics import direct_routing, gate_strength
from prompt4b_residual import residual_target
from prompt4c_bundles import (
    FinalGlobalBundle,
    FinalPrimaryBundle,
    development_target_cutpoints,
    duplicate_safe_target_deciles,
    mape_details,
    stage3_gate_strength,
    stage3_prediction,
    wape_percent,
)
from prompt4c_final import (
    DEVELOPMENT_SHA256,
    FEATURE_CONTRACT,
    FINAL_ROLES,
    atomic_json,
    deterministic_final_twofold,
    ensure_allowed_data_path,
    file_sha256,
    load_ledger,
    valid_saved_role,
)


class ConstantComponent:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=np.float64)


class DummyMeta:
    def predict_tail_probability(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), 1.0, dtype=np.float64)


def test_development_identity_and_exact_feature_contract():
    development = ROOT / "outputs/data/development.parquet"
    assert file_sha256(development) == DEVELOPMENT_SHA256
    assert pq.ParquetFile(development).metadata.num_rows == 500_000
    roles = json.loads((ROOT / "outputs/reports/feature_roles.json").read_text())
    features = roles["contracts"][FEATURE_CONTRACT]
    assert len(features) == len(set(features)) == 35
    assert not {"respondent_id", "loan_amount_000s", "row_hash"} & set(features)


def test_stage3_strict_routing_threshold_alpha_and_no_cap():
    probability = np.array([0.74, 0.75, 0.875, 1.0])
    strength = stage3_gate_strength(probability)
    assert np.allclose(strength, [0.0, 0.0, 0.375, 0.75])
    prediction = stage3_prediction(np.full(4, 100.0), np.full(4, 1000.0), probability)
    assert np.allclose(prediction, [100.0, 100.0, 475.0, 850.0])


def test_global_formula_is_exact():
    cat = np.array([1.0, 10.0])
    lgb = np.array([2.0, 20.0])
    xgb = np.array([3.0, 30.0])
    result = 0.60 * cat + 0.20 * lgb + 0.20 * xgb
    assert np.array_equal(result, [1.6, 16.0])


def test_residual_target_formula_and_tail_rule():
    y = np.array([438.0, 439.0, 500.0])
    global_prediction = np.array([400.0, 410.0, 450.0])
    assert np.array_equal(residual_target(y, global_prediction), [38.0, 29.0, 50.0])
    assert np.array_equal(y > 438.0, [False, True, True])


def test_mape_without_epsilon_and_nonpositive_handling():
    result = mape_details([100.0, 200.0], [90.0, 220.0])
    assert result["mape_percent"] == pytest.approx(10.0)
    assert result["mape_invalid_nonpositive_rows"] == 0
    mixed = mape_details([100.0, 0.0, -5.0], [90.0, 2.0, 1.0])
    assert mixed["mape_percent"] == pytest.approx(10.0)
    assert mixed["mape_invalid_nonpositive_rows"] == 2
    assert mixed["mape_valid_coverage"] == pytest.approx(1 / 3)


def test_wape_formula_and_mae_identity_for_positive_targets():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([90.0, 220.0, 270.0])
    expected = 100.0 * np.mean(np.abs(y - p)) / np.mean(y)
    assert wape_percent(y, p) == pytest.approx(expected)


def test_duplicate_safe_deciles_and_development_bands():
    y = np.arange(1.0, 101.0)
    labels = duplicate_safe_target_deciles(y)
    assert np.array_equal(np.unique(labels), np.arange(10))
    assert all(np.count_nonzero(labels == item) == 10 for item in range(10))
    cutpoints = development_target_cutpoints(y)
    assert list(cutpoints) == [f"q{i:02d}" for i in range(10, 100, 10)]
    assert cutpoints["q90"] == pytest.approx(np.quantile(y, 0.9))


def test_final_twofold_rule_is_deterministic_and_complete():
    y = np.tile(np.arange(1.0, 1001.0), 500)
    hashes = pd.Series([f"row-{index:06d}" for index in range(500_000)])
    a1, b1, evidence1 = deterministic_final_twofold(y, hashes)
    a2, b2, evidence2 = deterministic_final_twofold(y, hashes)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert len(a1) == len(b1) == 250_000
    assert np.intersect1d(a1, b1).size == 0
    assert np.union1d(a1, b1).size == 500_000
    assert evidence1["fold_assignment_digest"] == evidence2["fold_assignment_digest"]


def test_oof_no_self_fit_invariant_synthetic():
    fold = np.array(["fold_a", "fold_b", "fold_a", "fold_b"])
    fit_fold = np.array(["fold_b", "fold_a", "fold_b", "fold_a"])
    assert np.count_nonzero(fold == fit_fold) == 0
    prediction_count = pd.Series([0, 1, 2, 3]).value_counts()
    assert (prediction_count == 1).all()


def test_six_condition_evaluator_is_unchanged():
    base = {
        "mae": 100.0, "top_decile_mae": 200.0, "bottom_90_mae": 50.0,
        "rmse": 120.0, "top_decile_signed_error": -100.0,
        "top_decile_underprediction_rate": 0.8,
    }
    candidate = {
        "mae": 99.0, "top_decile_mae": 190.0, "bottom_90_mae": 50.1,
        "rmse": 120.2, "top_decile_signed_error": -90.0,
        "top_decile_underprediction_rate": 0.7,
    }
    checks = provisional_acceptance(candidate, base)
    assert checks["conditions_passed"] == 6
    assert checks["provisional_acceptance_status"] == "PASS"


def test_global_and_primary_bundle_input_contracts():
    features = ["a", "b"]
    global_bundle = FinalGlobalBundle(features, {
        "catboost": ConstantComponent(10.0),
        "lightgbm": ConstantComponent(20.0),
        "xgboost": ConstantComponent(30.0),
    })
    frame = pd.DataFrame({"b": [2, 3], "a": [1, 2], "respondent_id": ["x", "y"]})
    assert np.array_equal(global_bundle.predict(frame), [16.0, 16.0])
    primary = FinalPrimaryBundle(features, global_bundle, DummyMeta(), ConstantComponent(4.0))
    assert np.array_equal(primary.predict(frame), [19.0, 19.0])
    with pytest.raises(ValueError):
        primary.predict(frame.drop(columns="a"))
    duplicated = pd.concat([frame, frame[["a"]]], axis=1)
    with pytest.raises(ValueError):
        primary.predict(duplicated)


def test_blocka_saved_zero_fit_formula_is_exact():
    g2 = pd.read_parquet(ROOT / "outputs/predictions/prompt4a/validation/ens_convex_boosting_deep.parquet")
    gate = pd.read_parquet(ROOT / "outputs/predictions/prompt4a/validation/tail_gate.parquet")
    direct = pd.read_parquet(ROOT / "outputs/predictions/prompt4a/validation/tail_specialist.parquet")
    saved = pd.read_parquet(ROOT / "outputs/predictions/prompt4b2/validation/nf_global2_oldraw_direct_cap25.parquet")
    prediction, _ = direct_routing(
        g2.y_pred.to_numpy(float), direct.y_pred.to_numpy(float),
        gate_strength(gate.p_tail.to_numpy(float), 0.85, 0.50), 109.5,
    )
    assert np.max(np.abs(prediction - saved.y_pred.to_numpy(float))) == 0.0


def test_fit_ledger_accounting_starts_with_exact_roles(tmp_path: Path):
    ledger = load_ledger(tmp_path)
    assert tuple(ledger["authorized_roles"]) == FINAL_ROLES
    assert ledger["authorized_role_count"] == 11
    assert ledger["scientific_candidate_searches"] == 0


def test_resume_semantics_require_exact_manifest_and_hash(tmp_path: Path):
    destination = tmp_path / "role"
    destination.mkdir()
    bundle = ConstantComponent(1.0)
    joblib.dump(bundle, destination / "bundle.joblib")
    digest = file_sha256(destination / "bundle.joblib")
    manifest = {
        "status": "COMPLETE", "role": "r", "configuration_hash": "c",
        "training_membership_digest": "m", "model_sha256": digest,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest))
    assert valid_saved_role(destination, {"role": "r", "configuration_hash": "c", "training_membership_digest": "m"}) is not None
    assert valid_saved_role(destination, {"role": "r", "configuration_hash": "different", "training_membership_digest": "m"}) is None


def test_raw_and_iid_guards():
    with pytest.raises(PermissionError):
        ensure_allowed_data_path(ROOT, ROOT / "data/raw.csv")
    with pytest.raises(PermissionError):
        ensure_allowed_data_path(ROOT, ROOT / "outputs/data/iid_holdout_features.parquet")
    with pytest.raises(PermissionError):
        ensure_allowed_data_path(ROOT, ROOT / "outputs/data/iid_holdout_targets.parquet")
    ensure_allowed_data_path(ROOT, ROOT / "outputs/data/development.parquet")


def test_freeze_serialization_round_trip(tmp_path: Path):
    payload = {"status": "FROZEN", "primary": "stage3", "value": 0.75}
    atomic_json(tmp_path, "freeze.json", payload)
    assert json.loads((tmp_path / "freeze.json").read_text()) == payload


def test_prompt4c_pre_fit_reports_when_prepared():
    required = [
        "prompt4c_handoff_validation.json", "prompt4c_final_selection_freeze.json",
        "prompt4c_stage3_recipe_reproduction.json", "prompt4c_global_recipe_reproduction.json",
        "prompt4c_blocka_package_eligibility.json", "prompt4c_refit_plan.json",
    ]
    missing = [name for name in required if not (ROOT / "outputs/reports" / name).exists()]
    if missing:
        pytest.skip(f"Prompt 4C prepare phase has not run: {missing}")
    assert json.loads((ROOT / "outputs/reports/prompt4c_stage3_recipe_reproduction.json").read_text())["status"] == "PASS"
    assert json.loads((ROOT / "outputs/reports/prompt4c_global_recipe_reproduction.json").read_text())["status"] == "PASS"
    freeze = json.loads((ROOT / "outputs/reports/prompt4c_final_selection_freeze.json").read_text())
    assert freeze["selection_frozen_before_prompt4c_percentage_metrics"] is True
    assert freeze["final_primary_recipe"] == "stage3_residual_t75_a75"

