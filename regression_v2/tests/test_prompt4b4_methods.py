import joblib
import numpy as np
import pytest

from src.prompt4b4_methods import (
    IMrObjective, SERAObjective, denseweight_direct_from_density, fit_imr_gmm,
    fixed_substitution_predictions, imr_gradient, lds_kernel_window, lds_weights,
    scientific_resume_action, select_mae_challenger, select_tailaware,
    sera_bruteforce_sigma, sera_sigma, six_condition_evaluator,
    validate_feature_contract,
)


def reference():
    return {"mae": 100.0, "top_decile_mae": 200.0, "bottom_90_mae": 50.0,
            "rmse": 150.0, "top_decile_signed_error": -100.0,
            "top_decile_underprediction_rate": 0.8}


def candidate(**changes):
    value = reference()
    value.update({"mae": 99.0, "top_decile_mae": 194.0, "bottom_90_mae": 50.125,
                  "rmse": 150.375, "top_decile_signed_error": -99.0,
                  "top_decile_underprediction_rate": 0.79})
    value.update(changes)
    return value


def test_feature_contract_guards():
    validate_feature_contract([f"f{i}" for i in range(35)])
    with pytest.raises(ValueError): validate_feature_contract([f"f{i}" for i in range(34)])
    with pytest.raises(ValueError): validate_feature_contract(["respondent_id"] + [f"f{i}" for i in range(34)])


def test_denseweight_direct_normalization_and_positivity():
    weights = denseweight_direct_from_density([0.0, 0.5, 1.0])
    assert np.isclose(weights.mean(), 1.0)
    assert np.all(weights > 0)


def test_sera_exact_trapezoid_and_derivatives():
    phi = np.array([0.0, 0.123, 0.5, 0.999, 1.0])
    sigma = sera_sigma(phi)
    assert np.allclose(sigma, sera_bruteforce_sigma(phi), atol=1e-15, rtol=0)
    actual = np.array([2., 2., 2., 2., 2.])
    pred = np.array([1., 3., 2., 4., 0.])
    grad, hess = SERAObjective(sigma)(actual, pred)
    assert np.allclose(grad, 2*sigma*(pred-actual))
    assert np.allclose(hess, 2*sigma)
    assert grad[2] == 0
    assert np.all(np.diff(sigma) >= 0)


def test_lds_source_pattern():
    window = lds_kernel_window()
    assert len(window) == 5 and np.isclose(window.max(), 1.0) and np.allclose(window, window[::-1])
    weights, bins, counts, effective = lds_weights([1., 1., 2., 10.])
    assert np.isclose(weights.mean(), 1.0) and np.all(weights > 0)
    assert bins.tolist() == [1, 1, 2, 10] and counts[1] == 2 and effective.shape == counts.shape


def test_imr_gradient_shape_finite_and_wrapper():
    y = np.array([0., 1., 3.]); pred = np.array([0.2, 0.8, 2.5])
    means=np.array([0., 2.]); weights=np.array([0.7, 0.3]); variances=np.array([1., 2.])
    grad, hess = imr_gradient(y, pred, means, weights, variances)
    grad2, hess2 = IMrObjective(means, weights, variances)(y, pred)
    assert grad.shape == hess.shape == y.shape and np.isfinite(grad).all() and np.all(hess == 1)
    assert np.array_equal(grad, grad2) and np.array_equal(hess, hess2)


def test_imr_prior_is_fitted_only_from_supplied_training_labels():
    train=np.r_[np.linspace(1,10,50),np.linspace(100,120,50)]
    model, means, weights, variances=fit_imr_gmm(train)
    assert model.n_features_in_ == 1 and len(means) == len(weights) == len(variances) == 6
    assert means.max() <= train.max() and means.min() >= train.min()


def test_six_condition_edges_and_directions():
    assert six_condition_evaluator(candidate(), reference())["conditions_passed"] == 6
    assert six_condition_evaluator(candidate(top_decile_mae=194.000001), reference())["conditions_passed"] == 5
    assert six_condition_evaluator({"mae":101,"top_decile_mae":201,"bottom_90_mae":51,"rmse":151,
        "top_decile_signed_error":-101,"top_decile_underprediction_rate":.81}, reference())["conditions_passed"] == 0
    assert six_condition_evaluator(candidate(top_decile_mae=194.0), reference())["C2_top_decile_mae_improves_3pct"]
    assert six_condition_evaluator(candidate(bottom_90_mae=50.125), reference())["C3_bottom_90_mae_worsens_at_most_0_25pct"]
    assert six_condition_evaluator(candidate(rmse=150.375), reference())["C4_rmse_worsens_at_most_0_25pct"]
    assert not six_condition_evaluator(candidate(top_decile_signed_error=-101), reference())["C5_top_decile_signed_error_closer_to_zero"]
    assert not six_condition_evaluator(candidate(top_decile_underprediction_rate=.8), reference())["C6_top_decile_underprediction_rate_decreases"]


def test_selection_rankings():
    rows=[{"candidate_id":"a","conditions_passed":5,"mae":10,"top_decile_mae":20,"rmse":30,"complexity":1},
          {"candidate_id":"b","conditions_passed":6,"mae":11,"top_decile_mae":19,"rmse":29,"complexity":1}]
    assert select_tailaware(rows) == "b"
    assert select_mae_challenger(rows) == "a"


def test_fixed_component_substitutions_are_exact():
    dense=np.array([1.,2.]); sera=np.array([3.,4.]); imr=np.array([5.,6.]); lds=np.array([7.,8.])
    cat=np.array([10.,20.]); lgb=np.array([30.,40.]); xgb=np.array([50.,60.])
    got=fixed_substitution_predictions(dense,sera,imr,lds,cat,lgb,xgb)
    assert np.array_equal(got["prompt4b4__sub_densecat"],.6*dense+.2*lgb+.2*xgb)
    assert np.array_equal(got["prompt4b4__sub_seraxgb"],.6*cat+.2*lgb+.2*sera)
    assert np.array_equal(got["prompt4b4__sub_imrgb"],.6*cat+.2*lgb+.2*imr)
    assert np.array_equal(got["prompt4b4__sub_ldslgb"],.6*cat+.2*lds+.2*xgb)


def test_objective_serialization(tmp_path):
    path=tmp_path/"objectives.joblib"
    objects=[SERAObjective(np.array([.1,.5,1.])),IMrObjective(np.array([0.,2.]),np.array([.7,.3]),np.array([1.,2.]))]
    joblib.dump(objects,path); restored=joblib.load(path)
    assert np.array_equal(restored[0].coefficient,objects[0].coefficient)
    assert np.array_equal(restored[1].means,objects[1].means)


def test_resume_logic_never_repeats_valid_fit():
    passed=[{"category":"scientific_candidate","candidate_id":"c","status":"PASS"}]
    assert scientific_resume_action(passed,"c",True,True) == "REUSE"
    with pytest.raises(RuntimeError): scientific_resume_action(passed,"c",False,True)
    failed=[{"category":"scientific_candidate","candidate_id":"c","status":"TECHNICAL_FAILURE"}]*2
    with pytest.raises(RuntimeError): scientific_resume_action(failed,"c",False,False)
