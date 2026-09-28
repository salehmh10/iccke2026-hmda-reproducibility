import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.features import (
    AdvancedFinancialFeatureEngineer,
    build_advanced_feature_pipeline,
    build_feature_pipeline,
)
from src.features.financial import CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_amount_000s": [200.0, 120.0, 350.0, 90.0, 180.0, 250.0],
            "applicant_income_000s": [100.0, 60.0, 175.0, 0.0, 90.0, 125.0],
            "hud_median_family_income": [80_000.0, 60_000.0, 70_000.0, 50_000.0, 75_000.0, 85_000.0],
            "tract_to_msamd_income": [80.0, 100.0, 120.0, 70.0, 95.0, 110.0],
            "agency_name": ["A", "A", "B", "B", "A", "B"],
            "loan_type_name": ["Conventional", "FHA", "Conventional", "FHA", "VA", "VA"],
            "property_type_name": ["One", "One", "Two", "One", "Two", "One"],
            "loan_purpose_name": ["Purchase", "Refi", "Purchase", "Refi", "Purchase", "Refi"],
            "owner_occupancy_name": ["Owner", "Owner", "Investor", "Owner", "Investor", "Owner"],
            "preapproval_name": ["Requested", "Not requested", "Requested", "Not requested", "Requested", "Not requested"],
            "lien_status_name": ["First", "First", "Junior", "First", "Junior", "First"],
            "loan_approved": [1, 0, 1, 0, 1, 0],
            "target_denied": [0, 1, 0, 1, 0, 1],
        }
    )


def test_v2_numeric_formulas_are_finite_or_explicitly_missing() -> None:
    engineered = AdvancedFinancialFeatureEngineer(
        families=("numeric_relative", "nonlinear")
    ).fit_transform(_frame())
    assert engineered.loc[0, "loan_to_area_median"] == pytest.approx(2.5)
    assert engineered.loc[0, "applicant_income_to_tract_income"] == pytest.approx(
        100.0 / 64.0
    )
    assert engineered.loc[0, "loan_income_normalized_gap"] == pytest.approx(1.0 / 3.0)
    assert engineered.loc[0, "loan_to_income_gt_2"] == 0.0
    assert engineered.loc[3, "loan_to_tract_income"] == pytest.approx(90.0 / 35.0)


def test_category_statistics_use_fit_rows_and_unseen_fallback() -> None:
    train = _frame().iloc[:4].copy()
    held = _frame().iloc[[4]].copy()
    held.loc[:, "loan_type_name"] = "Never seen"
    transformer = AdvancedFinancialFeatureEngineer(
        families=("category_context", "onehot_numeric"), min_category_count=2
    ).fit(train)
    transformed = transformer.transform(held)
    relative = transformed.loc[
        held.index[0], "loan_amount_000s__by__loan_type_name__relative_median"
    ]
    global_median = np.median(train["loan_amount_000s"])
    assert relative == pytest.approx(
        (held.iloc[0]["loan_amount_000s"] - global_median) / global_median
    )
    assert transformed.loc[held.index[0], "loan_type_name__rare_or_unseen"] == 1.0
    interaction_columns = [
        column
        for column in transformed
        if "__x__loan_type_name__" in column
    ]
    assert interaction_columns
    assert (transformed.loc[held.index[0], interaction_columns] == 0.0).all()


def test_empty_family_contract_matches_v1_preencoding_columns() -> None:
    engineered = AdvancedFinancialFeatureEngineer(families=()).fit_transform(_frame())
    assert list(engineered.columns) == [*MODEL_NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def test_empty_family_pipeline_is_numerically_identical_to_v1() -> None:
    frame = _frame()
    v1 = build_feature_pipeline().fit_transform(frame)
    v2_baseline = build_advanced_feature_pipeline(families=()).fit_transform(frame)
    np.testing.assert_allclose(v1.toarray(), v2_baseline.toarray(), rtol=0.0, atol=0.0)


def test_target_changes_cannot_change_v2_features() -> None:
    first = _frame()
    changed = first.copy()
    changed["loan_approved"] = 1 - changed["loan_approved"]
    changed["target_denied"] = 1 - changed["target_denied"]
    left = AdvancedFinancialFeatureEngineer().fit_transform(first)
    right = AdvancedFinancialFeatureEngineer().fit_transform(changed)
    pd.testing.assert_frame_equal(left, right)
    assert "loan_approved" not in left
    assert "target_denied" not in left


def test_advanced_pipeline_is_sparse_finite_and_handles_unseen_categories() -> None:
    train = _frame().iloc[:5].copy()
    held = _frame().iloc[[5]].copy()
    held.loc[:, "loan_purpose_name"] = "Unseen purpose"
    pipeline = build_advanced_feature_pipeline(min_category_count=2)
    train_matrix = pipeline.fit_transform(train)
    held_matrix = pipeline.transform(held)
    assert sparse.issparse(train_matrix)
    assert sparse.issparse(held_matrix)
    assert train_matrix.shape[1] == held_matrix.shape[1]
    assert np.isfinite(train_matrix.data).all()
    assert np.isfinite(held_matrix.data).all()


def test_unknown_family_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown V2 feature families"):
        AdvancedFinancialFeatureEngineer(families=("blind_polynomial",)).fit(_frame())
