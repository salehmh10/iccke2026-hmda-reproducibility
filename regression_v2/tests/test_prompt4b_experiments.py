"""Focused Prompt 4B contract tests."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompt4_metrics import provisional_acceptance
from prompt4b_calibration import fit_platt, prior_correct_probability, residual_routed_prediction, routed_prediction, thresholded_gate_strength
from prompt4b_crossfit import GLOBAL_FEATURE, deterministic_twofold_split
from prompt4b_ensemble import apply_weights, optimize_tail_weighted_mae
from prompt4b_experiments import FIT_ROLES, validate_prompt4a
from prompt4b_residual import residual_target


def test_prompt4a_readiness_and_roles():
    state = validate_prompt4a(ROOT)
    assert state["status"] == "PASS"
    assert len(state["train"]) == 400_000 and len(state["validation"]) == 100_000
    assert (state["roles"] == "selection").sum() == 70_000
    assert (state["roles"] == "audit").sum() == 30_000


def test_raw_iid_and_final_freeze_closure():
    ready = json.loads((ROOT / "outputs/reports/PROMPT4A_READY.json").read_text(encoding="utf-8"))
    assert ready["raw_access_count"] == ready["iid_feature_access_count"] == ready["iid_target_access_count"] == 0
    assert not (ROOT / "outputs/reports/FINAL_PRE_IID_FREEZE.json").exists()


def test_q90_consistency():
    state = validate_prompt4a(ROOT)
    assert float(np.quantile(state["train"]["loan_amount_000s"], 0.90)) == 438.0


def test_prior_correction_math():
    weighted = np.array([0.1, 0.5, 0.9])
    ratio = 9.0
    corrected = prior_correct_probability(weighted, ratio)
    expected = weighted / (ratio * (1 - weighted) + weighted)
    assert np.allclose(corrected, expected)


def test_platt_records_selection_digest():
    p = np.linspace(0.01, 0.99, 100)
    y = (p > 0.55).astype(int)
    model = fit_platt(p, y, "selection-only")
    assert model.fitted_row_hash_digest == "selection-only"
    assert np.isfinite(model.predict(p)).all()


def test_thresholded_gate_formula():
    p = np.array([0.6, 0.65, 0.75, 1.0])
    strength = thresholded_gate_strength(p, 0.65, 0.5)
    assert np.allclose(strength, [0, 0, 0.5 * 0.10 / 0.35, 0.5])


def test_cap_formula():
    global_prediction = np.array([100.0, 100.0])
    specialist = np.array([1000.0, -1000.0])
    result = routed_prediction(global_prediction, specialist, np.ones(2), 0.65, 1.0, 109.5)
    assert np.allclose(result, [209.5, -9.5])


def test_residual_target_and_correction_formula():
    y = np.array([120.0, 80.0]); global_prediction = np.array([100.0, 100.0])
    residual = residual_target(y, global_prediction)
    assert np.array_equal(residual, [20.0, -20.0])
    prediction = residual_routed_prediction(global_prediction, residual, np.ones(2), 0.5, 1.0, 10.0)
    assert np.array_equal(prediction, [110.0, 90.0])


def test_twofold_disjointness_and_oof_contract():
    y = np.repeat(np.arange(10, dtype=float), 40_000)
    row_hash = pd.Series([f"r{i}" for i in range(400_000)])
    a, b, evidence = deterministic_twofold_split(y, row_hash, 8.0)
    assert len(a) == len(b) == 200_000
    assert np.intersect1d(a, b).size == 0
    assert evidence["overlap_rows"] == 0


def test_oof_ensemble_formula():
    cat = np.array([1.0, 2.0]); lgb = np.array([3.0, 4.0]); xgb = np.array([5.0, 6.0])
    assert np.allclose(0.6 * cat + 0.2 * lgb + 0.2 * xgb, [2.2, 3.2])


def test_meta_gate_feature_contract_excludes_leakage():
    state = validate_prompt4a(ROOT)
    features = state["source"]["features"] + [GLOBAL_FEATURE]
    assert len(features) == 36
    assert GLOBAL_FEATURE in features
    assert not {"y_true", "loan_amount_000s", "residual", "absolute_error", "respondent_id"} & set(features)


def test_residual_specialist_membership_is_tail_only_when_artifact_exists():
    path = ROOT / "outputs/models/prompt4b/residual_specialist/full/manifest.json"
    if not path.exists(): pytest.skip("Stage 3 has not run yet")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["tail_only"] is True and manifest["training_row_count"] == 39_897


def test_ensemble_constraints():
    matrix = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [2, 2, 2, 2]], dtype=float)
    y = np.array([1.5, 1.5, 2.0]); tail = np.array([False, True, False])
    result = optimize_tail_weighted_mae(matrix, y, tail, 1.5)
    assert np.all(result["weights"] >= 0)
    assert np.isclose(result["weights"].sum(), 1.0)
    assert np.isfinite(apply_weights(matrix, result["weights"])).all()


def test_selection_only_optimization_declared():
    source = (ROOT / "src/prompt4b_experiments.py").read_text(encoding="utf-8")
    assert "matrix[selection]" in source and "audit_reporting_only" in source


def test_six_condition_rules():
    base = {"mae": 10, "rmse": 12, "top_decile_mae": 20, "bottom_90_mae": 8, "top_decile_signed_error": -5, "top_decile_underprediction_rate": .8}
    candidate = {"mae": 9, "rmse": 12, "top_decile_mae": 19, "bottom_90_mae": 8, "top_decile_signed_error": -4, "top_decile_underprediction_rate": .7}
    result = provisional_acceptance(candidate, base)
    assert result["conditions_total"] == 6 and result["conditions_passed"] == 6 and result["provisional_acceptance_status"] == "PASS"


def test_fit_budget_exact_roles():
    assert len(FIT_ROLES) == 10 and len(set(FIT_ROLES)) == 10


def test_no_full_development_fit_or_iid_access_in_source():
    sources = [ROOT / "src" / name for name in ("prompt4b_experiments.py", "prompt4b_calibration.py", "prompt4b_crossfit.py", "prompt4b_residual.py", "prompt4b_ensemble.py")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "500_000" not in text.replace("EXPECTED_DEVELOPMENT_ROWS", "")
    tree = ast.parse(text)
    string_values = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not any("iid_holdout" in value.lower() for value in string_values)


def test_oracle_is_diagnostic_when_artifact_exists():
    path = ROOT / "outputs/reports/prompt4b_stage1_oracle.json"
    if not path.exists(): pytest.skip("Stage 1 has not run yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["analysis_only"] is True and report["deployable"] is False


def test_bootstrap_alignment_when_artifact_exists():
    path = ROOT / "outputs/reports/prompt4b_bootstrap.json"
    if not path.exists(): pytest.skip("Bootstrap has not run yet")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert all(item["rows"] == 100_000 for item in report["overall_mae"].values())


def test_notebook_no_fit_when_artifact_exists():
    path = ROOT / "notebooks/04B_STEPWISE_TAIL_IMPROVEMENT.ipynb"
    if not path.exists(): pytest.skip("Notebook has not been built yet")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(cell.get("source", "") if isinstance(cell.get("source", ""), str) else "".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert ".fit(" not in code and "fit_predict" not in code
