"""Focused, no-fit contract tests for Prompt 4B2.

The tests use small synthetic arrays, saved JSON/CSV evidence, and AST checks.
They never read Development, Raw, or IID parquet data and never fit a model.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPORTS = ROOT / "outputs" / "reports"
sys.path.insert(0, str(SRC))

from prompt4_metrics import provisional_acceptance
from prompt4b2_metrics import SIX_CONDITIONS, direct_routing, gate_strength, routed_residual, selection_rank


EXPECTED_NO_FIT_IDS = {
    "nf_oldraw_residual_symcap25",
    "nf_oldplatt_residual",
    "nf_oldraw_residual_positive_cap25",
    "nf_meta_residual_positive_cap25",
    "nf_consensus_residual_positive_cap25",
    "nf_meta_residual_positive_gamma15",
    "nf_meta_residual_positive_gamma20",
    "nf_global_convex_boosting_deep",
    "nf_global2_oldraw_direct_cap25",
}
EXPECTED_QUANTILE_IDS = {"quantile_residual_q60", "quantile_residual_q65"}
EXPECTED_BENEFIT_THRESHOLDS = [0.0, 5.0, 10.0]
EXPECTED_FIT_ROLES = {
    "quantile_residual_q60",
    "quantile_residual_q65",
    "oof_residual_foldA",
    "oof_residual_foldB",
    "benefit_router_selection",
    "benefit_router_full_refit",
}
FORBIDDEN_ROUTER_FEATURES = {
    "y_true",
    "loan_amount_000s",
    "target_decile",
    "operational_tail",
    "realized_residual",
    "absolute_error",
    "realized_benefit",
    "row_hash",
    "record_hash",
    "respondent_id",
    "p_raw",
    "p_meta_gate",
    "specialist_realized_error",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(name: str) -> str:
    path = SRC / name
    if not path.exists():
        pytest.skip(f"Expected interface is not available yet: {path.name}")
    return path.read_text(encoding="utf-8")


def _module(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"Expected interface is not available yet: {name}")


def _candidate_ids_from_source(text: str, prefix: str) -> set[str]:
    return set(re.findall(rf"\b{re.escape(prefix)}[a-zA-Z0-9_]+\b", text))


def _read_literal_paths(source: str) -> list[str]:
    """Collect literal paths passed to common file-reading calls."""
    tree = ast.parse(source)
    paths: list[str] = []
    readers = {"read_csv", "read_parquet", "read_table", "read_text", "read_bytes", "open"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in readers or not node.args:
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            paths.append(argument.value.lower().replace("\\", "/"))
    return paths


def test_prompt4b_readiness_and_saved_membership_contract():
    ready = _json(REPORTS / "PROMPT4B_READY.json")
    verification = _json(REPORTS / "prompt4b_verification.json")
    design = _json(REPORTS / "prompt4b_frozen_design.json")
    source = design["source"]
    split = design["validation_split"]
    assert ready["status"] == verification["status"] == "PASS"
    assert source["rows"] == 500_000
    assert source["train_rows"] == 400_000 and source["validation_rows"] == 100_000
    assert split["selection_rows"] == 70_000 and split["audit_rows"] == 30_000
    assert split["selection_audit_overlap"] == 0
    assert design["q90_train"] == 438.0
    assert len(source["features"]) == source["feature_count"] == 35
    assert source["feature_contract"] == "main_without_sensitive_without_lender"


def test_raw_iid_final_refit_and_freeze_closure():
    ready = _json(REPORTS / "PROMPT4B_READY.json")
    verification = _json(REPORTS / "prompt4b_verification.json")
    assert ready["raw_access_count"] == 0
    assert ready["iid_feature_access_count"] == ready["iid_target_access_count"] == 0
    assert ready["iid_prediction_count"] == 0
    assert verification["full_development_refit_count"] == 0
    assert verification["final_model_selected"] is False
    assert verification["final_model_frozen"] is False
    assert not (REPORTS / "FINAL_PRE_IID_FREEZE.json").exists()


def test_q90_and_cap25_formula():
    q90 = float(_json(REPORTS / "prompt4b_frozen_design.json")["q90_train"])
    assert q90 == 438.0
    assert 0.25 * q90 == 109.5


def test_exact_nine_no_fit_candidate_ids_and_no_tenth_candidate():
    module = _module("prompt4b2_experiments")
    assert set(module.NO_FIT_CANDIDATES) == EXPECTED_NO_FIT_IDS
    assert len(module.NO_FIT_CANDIDATES) == len(set(module.NO_FIT_CANDIDATES)) == 9


def test_gate_strength_old_raw_nonlinear_and_nonnegative():
    probability = np.array([0.2, 0.75, 0.85, 0.925, 1.0])
    linear = gate_strength(probability, threshold=0.85, alpha=0.50, gamma=1.0)
    expected = 0.50 * np.maximum(probability - 0.85, 0.0) / 0.15
    assert np.allclose(linear, expected)
    assert np.all(linear >= 0)
    meta = np.array([0.70, 0.75, 0.875, 1.0])
    for gamma in (1.5, 2.0):
        expected_gamma = 0.75 * np.power(np.maximum(meta - 0.75, 0.0) / 0.25, gamma)
        assert np.allclose(gate_strength(meta, 0.75, 0.75, gamma), expected_gamma)


def test_positive_only_asymmetric_cap_and_symmetric_residual():
    global_prediction = np.full(4, 100.0)
    residual = np.array([-500.0, -20.0, 20.0, 500.0])
    strength = np.ones(4)
    positive_prediction, positive_correction = routed_residual(
        global_prediction, residual, strength, lower_cap=0.0, upper_cap=109.5, positive_only=True
    )
    assert np.array_equal(positive_correction, [0.0, 0.0, 20.0, 109.5])
    assert np.array_equal(positive_prediction, global_prediction + positive_correction)
    _, symmetric = routed_residual(
        global_prediction, residual, strength, lower_cap=-109.5, upper_cap=109.5
    )
    assert np.array_equal(symmetric, [-109.5, -20.0, 20.0, 109.5])


def test_consensus_strength_is_elementwise_minimum():
    probability_old = np.array([0.85, 0.90, 1.0])
    probability_meta = np.array([1.0, 0.90, 0.75])
    old = gate_strength(probability_old, 0.85, 0.50, 1.0)
    meta = gate_strength(probability_meta, 0.75, 0.75, 1.0)
    consensus = np.minimum(old, meta)
    assert np.array_equal(consensus, np.minimum(old, meta))
    assert np.all(consensus <= old) and np.all(consensus <= meta)


def test_global2_direct_formula_and_unsafe_combinations_are_absent():
    g2 = np.array([100.0, 100.0])
    direct = np.array([500.0, -500.0])
    prediction, correction = direct_routing(g2, direct, np.ones(2), cap=109.5)
    assert np.array_equal(correction, [109.5, -109.5])
    assert np.array_equal(prediction, g2 + correction)
    ids = _candidate_ids_from_source(_source("prompt4b2_experiments.py"), "nf_")
    assert {item for item in ids if item.startswith("nf_global2_")} == {
        "nf_global2_oldraw_direct_cap25"
    }


def test_platt_parameters_are_reused_from_prompt4b_selection_ranking():
    candidates = pd.read_csv(REPORTS / "prompt4b_stage1_candidates.csv")
    acceptance = pd.read_csv(REPORTS / "prompt4b_stage1_acceptance.csv")
    platt = candidates.loc[candidates["probability_source"] == "platt"]
    ranked = selection_rank(platt, acceptance.loc[acceptance["candidate_id"].isin(platt["candidate_id"])])
    assert ranked[0] == "stage1_platt_t65_a50"
    frozen = platt.loc[(platt["scope"] == "selection") & (platt["candidate_id"] == ranked[0])].iloc[0]
    assert frozen["threshold"] == 0.65 and frozen["alpha"] == 0.50
    assert not bool(frozen["capped"]) and pd.isna(frozen["cap"])


def test_quantile_contract_has_only_q60_q65_and_fixed_791_iterations():
    module = _module("prompt4b2_quantile")
    experiments = _module("prompt4b2_experiments")
    assert set(experiments.QUANTILE_ALPHAS) == set(module.ALLOWED_QUANTILE_ALPHAS) == {0.60, 0.65}
    assert experiments.QUANTILE_ITERATIONS == module.QUANTILE_ITERATIONS == 791
    assert {f"quantile_residual_q{int(alpha * 100)}" for alpha in module.ALLOWED_QUANTILE_ALPHAS} == EXPECTED_QUANTILE_IDS
    for alpha in module.ALLOWED_QUANTILE_ALPHAS:
        parameters = module.quantile_parameters(alpha)
        assert parameters["loss_function"] == f"Quantile:alpha={alpha:.2f}"
        assert parameters["iterations"] == 791
        assert parameters["random_seed"] == 42 and parameters["thread_count"] <= 4
        assert "early_stopping_rounds" not in parameters
    with pytest.raises(ValueError):
        module.quantile_parameters(0.70)
    config = _json(ROOT / "config.json")["prompt4b2"]
    assert config["quantile_alphas"] == [0.60, 0.65]


def test_quantile_tail_only_membership_and_oof_residual_target():
    module = _module("prompt4b2_quantile")
    frame = pd.DataFrame(
        {
            "clean_feature": [1.0, 2.0, 3.0],
            "global_prediction_feature": [400.0, 400.0, 400.0],
        }
    )
    features, residual, mask = module.prepare_tail_residual_fit_data(
        frame,
        np.array([438.0, 439.0, 500.0]),
        np.array([400.0, 400.0, 400.0]),
        ["clean_feature"],
        438.0,
    )
    assert mask.tolist() == [False, True, True]
    assert len(features) == 2
    assert np.array_equal(residual, [39.0, 100.0])
    altered = frame.copy()
    altered["global_prediction_feature"] = [401.0, 400.0, 400.0]
    with pytest.raises(ValueError, match="OOF Global"):
        module.prepare_tail_residual_fit_data(
            altered,
            np.array([438.0, 439.0, 500.0]),
            np.array([400.0, 400.0, 400.0]),
            ["clean_feature"],
            438.0,
        )


def test_quantile_and_oof_residual_targets_use_leakage_safe_global_oof():
    quantile = _source("prompt4b2_quantile.py").lower()
    benefit = _source("prompt4b2_benefit.py").lower()
    combined = quantile + "\n" + benefit
    assert "global_oof_prediction" in combined
    assert "loan_amount_000s" in combined or "y_true" in combined
    assert "residual_oof" in benefit
    assert "791" in combined


def test_twofold_oof_contract_is_frozen_and_leakage_safe():
    crossfit = _json(REPORTS / "prompt4b_stage2_crossfit.json")
    split = crossfit["split"]
    assert split["fold_a_rows"] == split["fold_b_rows"] == 200_000
    assert split["overlap_rows"] == 0
    assert crossfit["rows"] == 400_000
    assert crossfit["exactly_one_oof_per_row"] is True
    assert crossfit["zero_self_fit_rows"] is True
    source = _source("prompt4b2_benefit.py").lower()
    assert "fold_a" in source and "fold_b" in source
    assert "residual_oof" in source

    experiment_source = _source("prompt4b2_experiments.py")
    tree = ast.parse(experiment_source)
    complementary_prediction_mask = False
    self_fit_false = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "predict_mask" for target in node.targets):
                complementary_prediction_mask |= (
                    isinstance(node.value, ast.UnaryOp)
                    and isinstance(node.value.op, ast.Invert)
                    and isinstance(node.value.operand, ast.Name)
                    and node.value.operand.id == "fit_mask"
                )
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                self_fit_false |= (
                    isinstance(key, ast.Constant)
                    and key.value == "self_fit"
                    and isinstance(value, ast.Constant)
                    and value.value is False
                )
    assert complementary_prediction_mask and self_fit_false


def test_benefit_proposal_and_target_formula_on_synthetic_rows():
    y_true = np.array([130.0, 80.0, 100.0, 300.0])
    global_oof = np.array([100.0, 100.0, 100.0, 100.0])
    residual_oof = np.array([20.0, -30.0, 500.0, 500.0])
    proposal = np.clip(np.maximum(residual_oof, 0.0), 0.0, 109.5)
    benefit = np.abs(y_true - global_oof) - np.abs(y_true - (global_oof + proposal))
    assert np.array_equal(proposal, [20.0, 0.0, 109.5, 109.5])
    assert np.array_equal(benefit, [20.0, 0.0, -109.5, 109.5])

    module = _module("prompt4b2_benefit")
    implementation_proposal = module.positive_only_capped_proposal(residual_oof, 109.5)
    implementation_benefit = module.benefit_target(y_true, global_oof, implementation_proposal)
    assert np.array_equal(implementation_proposal, proposal)
    assert np.array_equal(implementation_benefit, benefit)


def test_benefit_router_feature_contract_is_exactly_37_and_target_free():
    roles = _json(REPORTS / "feature_roles.json")
    clean = roles["contracts"]["main_without_sensitive_without_lender"]
    features = clean + ["global_prediction_feature", "proposed_residual_correction"]
    assert len(clean) == 35 and len(features) == 37 and len(set(features)) == 37
    assert not FORBIDDEN_ROUTER_FEATURES.intersection(features)
    module = _module("prompt4b2_benefit")
    assert module.GLOBAL_FEATURE == "global_prediction_feature"
    assert module.PROPOSED_CORRECTION_FEATURE == "proposed_residual_correction"
    base = pd.DataFrame({name: [index] for index, name in enumerate(clean)})
    routed = module.make_benefit_feature_frame(base, clean, [100.0], [10.0])
    assert routed.columns.tolist() == features and routed.shape == (1, 37)
    assert not FORBIDDEN_ROUTER_FEATURES.intersection(routed.columns)


def test_benefit_router_thresholds_are_exact_and_strict():
    thresholds = _json(ROOT / "config.json")["prompt4b2"]["benefit_thresholds"]
    assert thresholds == EXPECTED_BENEFIT_THRESHOLDS
    predicted_benefit = np.array([-1.0, 0.0, 0.01, 5.0, 5.01, 10.0, 10.01])
    selected = {threshold: predicted_benefit > threshold for threshold in thresholds}
    assert selected[0.0].tolist() == [False, False, True, True, True, True, True]
    assert selected[5.0].tolist() == [False, False, False, False, True, True, True]
    assert selected[10.0].tolist() == [False, False, False, False, False, False, True]
    module = _module("prompt4b2_benefit")
    diagnostics = module.benefit_threshold_diagnostics(
        np.full(7, 200.0),
        np.full(7, 100.0),
        np.full(7, 10.0),
        predicted_benefit,
        150.0,
    )
    assert diagnostics["benefit_threshold"].tolist() == EXPECTED_BENEFIT_THRESHOLDS
    assert diagnostics["selected_row_count"].tolist() == [5, 3, 1]
    with pytest.raises(ValueError, match="exactly 0, 5, and 10"):
        module.benefit_threshold_diagnostics(
            np.full(7, 200.0),
            np.full(7, 100.0),
            np.full(7, 10.0),
            predicted_benefit,
            150.0,
            thresholds=(0.0, 5.0),
        )


def test_benefit_router_configuration_and_internal_split():
    module = _module("prompt4b2_benefit")
    assert module.BENEFIT_THRESHOLDS == tuple(EXPECTED_BENEFIT_THRESHOLDS)
    parameters = module.BENEFIT_ROUTER_PARAMETERS
    assert parameters == {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 1500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 10,
        "random_strength": 1,
        "random_seed": 42,
        "thread_count": 4,
        "early_stopping_rounds": 100,
        "verbose": False,
    }
    fixed = module.fixed_benefit_router_config(321)
    assert fixed["iterations"] == 321
    assert "early_stopping_rounds" not in fixed
    source = _source("prompt4b2_benefit.py") + "\n" + _source("prompt4b2_experiments.py")
    for token in ("CatBoostRegressor", "RMSE", "1500", "depth", "learning_rate", "l2_leaf_reg"):
        assert token in source
    for token in ("360_000", "40_000", "400_000"):
        assert token in source or token.replace("_", "") in source
    assert "100" in source and "random_seed" in source and "thread_count" in source


def test_selection_ranking_ignores_audit_and_complete_validation():
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "scope": "selection", "mae": 9.0, "bottom_90_mae": 8.0, "top_decile_mae": 18.0},
            {"candidate_id": "b", "scope": "selection", "mae": 9.5, "bottom_90_mae": 7.0, "top_decile_mae": 17.0},
            {"candidate_id": "a", "scope": "audit", "mae": 1000.0, "bottom_90_mae": 1000.0, "top_decile_mae": 1000.0},
            {"candidate_id": "b", "scope": "audit", "mae": 0.0, "bottom_90_mae": 0.0, "top_decile_mae": 0.0},
        ]
    )
    acceptance = pd.DataFrame(
        [
            {"candidate_id": "a", "scope": "selection", "conditions_passed": 5},
            {"candidate_id": "b", "scope": "selection", "conditions_passed": 5},
            {"candidate_id": "a", "scope": "audit", "conditions_passed": 0},
            {"candidate_id": "b", "scope": "audit", "conditions_passed": 6},
        ]
    )
    assert selection_rank(candidates, acceptance) == ["a", "b"]


def test_six_condition_rubric_is_unchanged():
    assert len(SIX_CONDITIONS) == 6 and len(set(SIX_CONDITIONS)) == 6
    reference = {
        "mae": 10.0,
        "rmse": 12.0,
        "top_decile_mae": 20.0,
        "bottom_90_mae": 8.0,
        "top_decile_signed_error": -5.0,
        "top_decile_underprediction_rate": 0.8,
    }
    candidate = {
        "mae": 9.0,
        "rmse": 12.0,
        "top_decile_mae": 19.0,
        "bottom_90_mae": 8.0,
        "top_decile_signed_error": -4.0,
        "top_decile_underprediction_rate": 0.7,
    }
    result = provisional_acceptance(candidate, reference)
    assert result["conditions_total"] == result["conditions_passed"] == 6
    assert result["provisional_acceptance_status"] == "PASS"


def test_scientific_fit_budget_is_exactly_six_roles():
    config = _json(ROOT / "config.json")["prompt4b2"]
    assert config["max_scientific_fits"] == 6
    source = _source("prompt4b2_experiments.py")
    observed = {role for role in EXPECTED_FIT_ROLES if role in source}
    assert observed == EXPECTED_FIT_ROLES


def test_sources_do_not_read_raw_or_iid_and_do_not_define_final_fit():
    paths = [SRC / name for name in (
        "prompt4b2_experiments.py",
        "prompt4b2_benefit.py",
        "prompt4b2_quantile.py",
        "prompt4b2_metrics.py",
    )]
    existing = [path for path in paths if path.exists()]
    if len(existing) < 4:
        pytest.skip("Expected Prompt 4B2 source interfaces are not all available yet")
    literal_reads = []
    for path in existing:
        literal_reads.extend(_read_literal_paths(path.read_text(encoding="utf-8")))
    assert not any("iid_holdout" in path for path in literal_reads)
    assert not any("hmda_2017" in path or path.startswith("data/") for path in literal_reads)
    for path in existing:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            target_names = {target.id for target in targets if isinstance(target, ast.Name)}
            if target_names.intersection({"final_model_selected", "final_model_frozen"}):
                assert not (isinstance(value, ast.Constant) and value.value is True)


def test_frozen_design_contract_when_artifact_exists():
    path = REPORTS / "prompt4b2_frozen_design.json"
    if not path.exists():
        pytest.skip("Prompt 4B2 frozen design has not been written yet")
    design = _json(path)
    assert design["status"] == "FROZEN"
    encoded = json.dumps(design, sort_keys=True)
    assert all(candidate_id in encoded for candidate_id in EXPECTED_NO_FIT_IDS)
    assert all(candidate_id in encoded for candidate_id in EXPECTED_QUANTILE_IDS)
    assert design.get("max_scientific_fits", design.get("scientific_fit_budget")) == 6
    assert design.get("benefit_thresholds") == EXPECTED_BENEFIT_THRESHOLDS
    assert design.get("final_model_selected") is False
    assert design.get("final_model_frozen") is False


def test_notebook_is_artifact_only_and_zero_fit_when_available():
    path = ROOT / "notebooks" / "04B2_BENEFIT_AWARE_SELECTIVE_CORRECTION.ipynb"
    if not path.exists():
        pytest.skip("Prompt 4B2 reporting notebook has not been built yet")
    notebook = _json(path)
    code = "\n".join(
        cell.get("source", "") if isinstance(cell.get("source", ""), str) else "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(code)
    call_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "fit" not in call_names and "fit_predict" not in call_names and "train" not in call_names
    assert "read_parquet" in call_names or "read_csv" in call_names or "read_text" in call_names
