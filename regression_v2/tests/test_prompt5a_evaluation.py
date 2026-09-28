"""Focused synthetic and immutable-hash tests for Prompt 5A.

These tests never open either original IID Parquet file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.prompt5a_evaluation import (
    BLOCKA_ID,
    FREEZE_SHA,
    GLOBAL_MODEL_ID,
    GLOBAL_SHA,
    PERMITTED_MODELS,
    PRIMARY_MODEL_ID,
    PRIMARY_SHA,
    _aligned_saved_frame,
    assign_development_bands,
    assert_exact_model_set,
    assert_prediction_allowed,
    assert_target_access_allowed,
    duplicate_safe_target_deciles,
    metric_row,
    paired_bootstrap,
    prohibit_training_operation,
    row_digest,
    schema_details,
    sha256,
    six_condition_rows,
    static_no_training_check,
    validate_lock_schema,
)


ROOT = Path(__file__).resolve().parents[1]


def frozen() -> dict:
    return json.loads((ROOT / "outputs/reports/FINAL_PRE_IID_FREEZE.json").read_text(encoding="utf-8"))


def test_01_freeze_hash_is_exact() -> None:
    assert sha256(ROOT / "outputs/reports/FINAL_PRE_IID_FREEZE.json") == FREEZE_SHA


def test_02_bundle_hashes_are_exact() -> None:
    assert sha256(ROOT / "outputs/models/final_pre_iid/primary_stage3/bundle.joblib") == PRIMARY_SHA
    assert sha256(ROOT / "outputs/models/final_pre_iid/global_comparator/bundle.joblib") == GLOBAL_SHA


def test_03_model_set_is_exactly_two() -> None:
    assert_exact_model_set(PERMITTED_MODELS)
    with pytest.raises(PermissionError):
        assert_exact_model_set((*PERMITTED_MODELS, "third_model"))


def test_04_block_a_is_excluded() -> None:
    state = frozen()
    assert state["historical_blocka"]["iid_eligible"] is False
    assert BLOCKA_ID not in [item["model_id"] for item in state["iid_protocol"]["permitted_models"]]


def test_05_feature_contract_is_exact_and_target_free() -> None:
    state = frozen()
    roles = json.loads((ROOT / "outputs/reports/feature_roles.json").read_text(encoding="utf-8"))
    features = state["project_identity"]["features"]
    assert len(features) == len(set(features)) == 35
    assert features == roles["contracts"]["main_without_sensitive_without_lender"]
    assert "loan_amount_000s" not in features and "row_hash" not in features


def test_06_prediction_lock_schema() -> None:
    lock = {
        "status": "PASS_PREDICTIONS_LOCKED_BEFORE_TARGET_ACCESS", "created_at_utc": "x",
        "authorization_id": "x", "freeze_sha256": "x", "primary_bundle_sha256": "x",
        "global_bundle_sha256": "x", "iid_feature_snapshot_row_digest": "x",
        "predictions": {PRIMARY_MODEL_ID: {}, GLOBAL_MODEL_ID: {}},
        "feature_access_successful_read_count": 1, "target_access_successful_read_count": 0,
    }
    validate_lock_schema(lock)
    bad = dict(lock); bad.pop("freeze_sha256")
    with pytest.raises(ValueError):
        validate_lock_schema(bad)


def test_07_target_alignment_logic(tmp_path: Path) -> None:
    rows = pd.Series(["r1", "r2", "r3"])
    for relative, frame in (
        ("outputs/data/post_iid/iid_model_features_snapshot.parquet", pd.DataFrame({"row_hash": rows})),
        ("outputs/data/post_iid/iid_target_snapshot.parquet", pd.DataFrame({"row_hash": rows.iloc[[2,0,1]].reset_index(drop=True), "loan_amount_000s": [30.,10.,20.]})),
        (f"outputs/predictions/prompt5a/iid/{PRIMARY_MODEL_ID}.parquet", pd.DataFrame({"row_hash": rows, "prediction": [11.,19.,31.], "global_base_prediction": [10.,20.,30.], "meta_gate_probability": [.1,.8,.9], "residual_specialist_proposal": [1.,2.,3.], "routing_strength": [0.,.1,.2], "routing_condition_activated": [False,True,True], "applied_residual_correction": [0.,.2,.6]})),
        (f"outputs/predictions/prompt5a/iid/{GLOBAL_MODEL_ID}.parquet", pd.DataFrame({"row_hash": rows, "prediction": [10.,20.,30.]})),
    ):
        path = tmp_path / relative; path.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(path, index=False)
    aligned, _ = _aligned_saved_frame(tmp_path)
    assert aligned["row_hash"].tolist() == rows.tolist()
    assert aligned["y_true"].tolist() == [10.,20.,30.]


def test_08_mae_and_rmse() -> None:
    result = metric_row([10., 20.], [12., 16.])
    assert result["mae"] == pytest.approx(3.0)
    assert result["rmse"] == pytest.approx(np.sqrt(10.0))


def test_09_mape_positive_only_without_epsilon() -> None:
    result = metric_row([0., 10., 20.], [5., 11., 18.])
    assert result["mape_percent"] == pytest.approx(10.0)
    assert result["mape_invalid_nonpositive_rows"] == 1
    assert result["mape_valid_coverage"] == pytest.approx(2 / 3)


def test_10_wape() -> None:
    result = metric_row([10., 20.], [12., 16.])
    assert result["wape_percent"] == pytest.approx(20.0)


def test_11_duplicate_safe_deciles() -> None:
    y = np.arange(1, 101, dtype=float)
    deciles = duplicate_safe_target_deciles(y)
    assert set(deciles) == set(range(10))
    assert np.bincount(deciles).tolist() == [10] * 10


def test_12_frozen_band_construction() -> None:
    cutpoints = {f"q{i:02d}": float(i) for i in range(10, 100, 10)}
    result = assign_development_bands([1., 10., 11., 90., 91.], cutpoints)
    assert result.tolist() == [1, 1, 2, 9, 10]


def test_13_six_condition_evaluator() -> None:
    global_metrics = {"mae": 10., "top_decile_mae": 100., "bottom_90_mae": 5., "rmse": 20., "top_decile_signed_error": -20., "top_decile_underprediction_rate": .8}
    primary = {"mae": 9., "top_decile_mae": 96., "bottom_90_mae": 5.01, "rmse": 20.04, "top_decile_signed_error": -19., "top_decile_underprediction_rate": .7}
    rows, summary = six_condition_rows(primary, global_metrics)
    assert len(rows) == 6 and summary["conditions_passed"] == 6


def test_14_bootstrap_is_deterministic() -> None:
    y = np.arange(1, 101, dtype=float)
    p = y + np.sin(y)
    g = y + np.cos(y)
    first, samples1 = paired_bootstrap(y, p, g, n_resamples=10, seed=42)
    second, samples2 = paired_bootstrap(y, p, g, n_resamples=10, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert all(np.array_equal(samples1[key], samples2[key]) for key in samples1)


def test_15_snapshot_schema(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.parquet"
    pd.DataFrame({"row_hash": ["a"], "x": [1.0]}).to_parquet(path, index=False)
    schema = schema_details(path)
    assert schema["columns"] == ["row_hash", "x"]
    assert set(schema["types"]) == {"row_hash", "x"}


def test_16_no_training_guard() -> None:
    with pytest.raises(PermissionError, match="BLOCKED_POST_IID_FIT_ATTEMPT"):
        prohibit_training_operation("fit")


def test_17_source_has_no_training_calls() -> None:
    assert static_no_training_check(ROOT / "src/prompt5a_evaluation.py")


def test_18_no_target_before_lock_guard() -> None:
    ledger = {"feature_successful_content_reads": 1, "target_successful_content_reads": 0}
    with pytest.raises(PermissionError):
        assert_target_access_allowed(ledger, False)


def test_19_no_prediction_after_target_guard() -> None:
    ledger = {"target_successful_content_reads": 1, "prediction_models": 0}
    with pytest.raises(PermissionError):
        assert_prediction_allowed(ledger)


def test_20_row_digest_is_order_sensitive() -> None:
    assert row_digest(["a", "b"]) != row_digest(["b", "a"])

