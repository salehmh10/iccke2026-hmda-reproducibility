"""Focused synthetic and frozen-contract tests for Prompt 4B3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompt4b3_beat import (
    CANDIDATE_ID,
    COSTAWARE_ID,
    FEATURE_COUNT,
    GLOBAL_FEATURE,
    LINEAR_ID,
    PROPOSAL_FEATURE,
    BeatClassifierBundle,
    alpha_costaware,
    alpha_linear,
    beat_target,
    classifier_config_from_anchor,
    cost_asymmetry,
    fixed_fraction_diagnostics,
    load_bundle,
    make_feature_frame,
    policy_prediction,
    reproduce_benefit,
    resumable_model_status,
    save_bundle,
    select_policy,
    six_condition_evaluator,
    validate_feature_contract,
    validate_oof_membership,
    validate_read_path,
)


def clean_features() -> list[str]:
    roles = json.loads((ROOT / "outputs/reports/feature_roles.json").read_text(encoding="utf-8"))
    return roles["contracts"]["main_without_sensitive_without_lender"]


class FakePreprocessor:
    def transform(self, frame):
        return frame

    def evidence(self):
        return {"type": "FakePreprocessor", "feature_count": FEATURE_COUNT}


class FakeProbabilityModel:
    def predict_proba(self, frame):
        probability = np.full(len(frame), 0.25)
        return np.column_stack([1.0 - probability, probability])

    def save_model(self, path, format="cbm"):
        Path(path).write_bytes(b"synthetic-prompt4b3-model")


def test_beat_formula_and_zero_benefit_behavior():
    benefit = np.array([-2.0, 0.0, 1e-12, 5.0])
    assert beat_target(benefit).tolist() == [0, 0, 1, 1]


def test_benefit_reproduction():
    y = np.array([130.0, 80.0, 100.0])
    global_prediction = np.array([100.0, 100.0, 100.0])
    proposal = np.array([20.0, 0.0, 50.0])
    expected = np.abs(y - global_prediction) - np.abs(y - (global_prediction + proposal))
    assert np.array_equal(reproduce_benefit(y, global_prediction, proposal), expected)


def test_exact_37_feature_order_and_no_prohibited_inference_feature():
    base = clean_features()
    contract = validate_feature_contract(base)
    assert len(contract) == 37 and contract[-2:] == [GLOBAL_FEATURE, PROPOSAL_FEATURE]
    frame = pd.DataFrame({name: [1] for name in base})
    result = make_feature_frame(frame, base, [100.0], [10.0])
    assert result.columns.tolist() == contract
    with pytest.raises(ValueError):
        validate_feature_contract(base[:-1] + ["p_tail"])


def test_membership_alignment_and_zero_self_fit():
    hashes = pd.Series([f"row-{index}" for index in range(400_000)])
    benefit = pd.DataFrame({"row_hash": hashes})
    residual = pd.DataFrame(
        {"row_hash": hashes, "exactly_one_oof_prediction": True, "self_fit": False}
    )
    evidence = validate_oof_membership(hashes, benefit, residual)
    assert evidence["train_rows"] == 400_000 and evidence["zero_self_fit_rows"] == 0
    residual.loc[0, "self_fit"] = True
    with pytest.raises(ValueError, match="self-fit"):
        validate_oof_membership(hashes, benefit, residual)


def test_p0_formula():
    result = cost_asymmetry(np.array([10.0, 20.0, -30.0, -10.0, 0.0]))
    assert result["G"] == 15.0 and result["D"] == 20.0
    assert result["p0"] == 20.0 / 35.0


def test_alpha_bounds_and_exact_prediction_formulas():
    probability = np.array([-0.2, 0.25, 0.75, 1.2])
    linear = alpha_linear(probability)
    cost = alpha_costaware(probability, 0.5)
    assert np.array_equal(linear, [0.0, 0.25, 0.75, 1.0])
    assert np.array_equal(cost, [0.0, 0.0, 0.5, 1.0])
    global_prediction = np.full(4, 100.0)
    proposal = np.full(4, 20.0)
    assert np.array_equal(policy_prediction(global_prediction, proposal, cost), global_prediction + cost * proposal)


def test_six_condition_evaluator_pass_partial_fail():
    reference = {
        "mae": 10.0, "rmse": 12.0, "top_decile_mae": 20.0, "bottom_90_mae": 8.0,
        "top_decile_signed_error": -5.0, "top_decile_underprediction_rate": 0.8,
    }
    passed = dict(reference, mae=9.0, top_decile_mae=19.0, top_decile_signed_error=-4.0, top_decile_underprediction_rate=0.7)
    partial = dict(reference, mae=9.0)
    failed = dict(reference, mae=11.0, rmse=13.0, top_decile_mae=21.0, bottom_90_mae=9.0, top_decile_signed_error=-6.0, top_decile_underprediction_rate=0.9)
    assert six_condition_evaluator(passed, reference)["conditions_passed"] == 6
    assert six_condition_evaluator(partial, reference)["provisional_acceptance_status"] == "PARTIAL"
    assert six_condition_evaluator(failed, reference)["provisional_acceptance_status"] == "FAIL"


def test_candidate_selection_tie_rules():
    rows = [
        {"candidate_id": LINEAR_ID, "conditions_passed": 5, "mae": 9.0, "top_decile_mae": 20.0},
        {"candidate_id": COSTAWARE_ID, "conditions_passed": 5, "mae": 9.0, "top_decile_mae": 20.0},
    ]
    assert select_policy(rows) == COSTAWARE_ID
    rows[0]["conditions_passed"] = 6
    assert select_policy(rows) == LINEAR_ID


def test_fixed_routed_fraction_diagnostics():
    labels = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 1])
    benefit = np.arange(-5, 5, dtype=float)
    score = np.arange(10, dtype=float)
    result = fixed_fraction_diagnostics(labels, benefit, score, fractions=(0.2,))
    assert result.loc[0, "routed_rows"] == 2
    assert result.loc[0, "precision"] == 0.5
    assert result.loc[0, "mean_realized_benefit"] == 3.5


def test_classifier_configuration_is_frozen_and_unweighted():
    config = classifier_config_from_anchor(
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10, "random_strength": 1}
    )
    assert config["loss_function"] == "Logloss" and config["iterations"] == 1500
    assert config["random_seed"] == 42 and config["thread_count"] <= 4 and config["task_type"] == "CPU"
    assert not {"class_weights", "auto_class_weights", "early_stopping_rounds", "use_best_model"}.intersection(config)


def test_iid_and_raw_path_guards():
    assert validate_read_path(ROOT, "outputs/data/development.parquet").name == "development.parquet"
    for path in ("outputs/data/iid_holdout_features.parquet", "outputs/data/iid_holdout_targets.parquet", "data/source.csv"):
        with pytest.raises(PermissionError):
            validate_read_path(ROOT, path)


def test_model_bundle_serialization_contract_and_strict_input_order():
    contract = validate_feature_contract(clean_features())
    bundle = BeatClassifierBundle(FakePreprocessor(), FakeProbabilityModel(), contract, {"candidate_id": CANDIDATE_ID})
    destination = ROOT / "outputs/tmp/prompt4b3/unit_bundle"
    save_bundle(bundle, destination)
    reloaded = load_bundle(destination)
    frame = pd.DataFrame({name: [0] for name in contract})
    assert reloaded.predict_probability(frame).tolist() == [0.25]
    with pytest.raises(ValueError, match="exact ordered"):
        reloaded.predict_probability(frame.loc[:, list(reversed(contract))])


def test_cache_resume_semantics():
    destination = ROOT / "outputs/tmp/prompt4b3/unit_bundle"
    assert resumable_model_status(destination, {"physical_attempts": []}) == "REUSE_VALID_MODEL"
    missing = ROOT / "outputs/tmp/prompt4b3/missing_bundle"
    assert resumable_model_status(missing, {"physical_attempts": []}) == "START_FIRST_ATTEMPT"
    assert resumable_model_status(missing, {"physical_attempts": [{"status": "IN_PROGRESS"}]}) == "INSPECT_ACTIVE_ATTEMPT"


def test_candidate_ids_are_exact():
    assert CANDIDATE_ID == "prompt4b3__beat_classifier__main37_fixed"
    assert {LINEAR_ID, COSTAWARE_ID} == {"prompt4b3__beat_linear", "prompt4b3__beat_costaware"}
