from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.features import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FEATURES,
    MODEL_NUMERIC_FEATURES,
    FinancialFeatureEngineer,
    build_feature_pipeline,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_amount_000s": [200.0, 100.0, np.nan, 80.0],
            "applicant_income_000s": [100.0, 0.0, 50.0, 40.0],
            "population": [1_000.0, 2_000.0, 1_500.0, 800.0],
            "hud_median_family_income": [80_000.0, 0.0, 100_000.0, 60_000.0],
            "tract_to_msamd_income": [80.0, 120.0, 100.0, 90.0],
            "number_of_owner_occupied_units": [400.0, 0.0, 300.0, 200.0],
            "number_of_1_to_4_family_units": [800.0, 0.0, 600.0, 250.0],
            "agency_name": ["A", "B", "A", "B"],
            "loan_type_name": ["Conventional", "FHA", "Conventional", "VA"],
            "property_type_name": ["House", "House", "Manufactured", "House"],
            "loan_purpose_name": ["Purchase", "Refinance", "Purchase", "Improve"],
            "owner_occupancy_name": ["Owner", "Owner", "Non-owner", "Owner"],
            "preapproval_name": ["Requested", "Not applicable", np.nan, "Not requested"],
            "lien_status_name": ["First", "First", "Subordinate", "None"],
            # Every prohibited family is deliberately present in the input.
            "loan_approved": [1, 0, 1, 0],
            "target_denied": [0, 1, 0, 1],
            "respondent_id": ["r1", "r2", "r3", "r4"],
            "applicant_ethnicity_name": ["x"] * 4,
            "co_applicant_ethnicity_name": ["x"] * 4,
            "applicant_race_name_1": ["x"] * 4,
            "co_applicant_race_name_1": ["x"] * 4,
            "applicant_sex_name": ["x"] * 4,
            "co_applicant_sex_name": ["x"] * 4,
            "msamd_name": ["m"] * 4,
            "state_name": ["s"] * 4,
            "state_code": [1] * 4,
            "county_name": ["c"] * 4,
            "county_code": [2] * 4,
            "census_tract_number": ["0001.00"] * 4,
            "minority_population": [25.0] * 4,
        }
    )


def test_financial_formulas_and_zero_denominators() -> None:
    engineered = FinancialFeatureEngineer().fit_transform(_frame())

    assert engineered.loc[0, "loan_to_income"] == pytest.approx(2.0)
    assert np.isnan(engineered.loc[1, "loan_to_income"])
    assert engineered.loc[0, "log1p_loan_amount_000s"] == pytest.approx(
        np.log1p(200.0)
    )
    assert engineered.loc[0, "log1p_applicant_income_000s"] == pytest.approx(
        np.log1p(100.0)
    )
    assert engineered.loc[0, "tract_to_msamd_income_ratio"] == pytest.approx(0.8)
    assert engineered.loc[0, "applicant_income_to_area_median"] == pytest.approx(
        1.25
    )
    assert np.isnan(engineered.loc[1, "applicant_income_to_area_median"])
    assert engineered.loc[1, "applicant_income_zero"] == 1.0
    assert engineered.loc[1, "hud_median_family_income_zero"] == 1.0


def test_modeling_whitelist_excludes_target_protected_id_and_geography() -> None:
    engineered = FinancialFeatureEngineer().fit_transform(_frame())

    assert list(engineered.columns) == [*MODEL_NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    assert not EXCLUDED_FEATURES.intersection(engineered.columns)


def test_missing_required_schema_fails_closed() -> None:
    incomplete = _frame().drop(columns="applicant_income_000s")
    with pytest.raises(ValueError, match="applicant_income_000s"):
        FinancialFeatureEngineer().fit(incomplete)


def test_pipeline_is_train_fit_only_sparse_finite_and_shape_stable() -> None:
    train = _frame().iloc[:3].copy()
    validation = _frame().iloc[[3]].copy()
    validation.loc[:, "agency_name"] = "UNSEEN_AGENCY"
    validation.loc[:, "loan_amount_000s"] = 10_000.0

    pipeline = build_feature_pipeline()
    train_matrix = pipeline.fit_transform(train)
    imputer = pipeline.named_steps["preprocess"].named_transformers_[
        "numeric"
    ].named_steps["imputer"]
    loan_index = list(MODEL_NUMERIC_FEATURES).index("loan_amount_000s")
    learned_train_median = imputer.statistics_[loan_index]

    validation_matrix = pipeline.transform(validation)

    assert learned_train_median == pytest.approx(150.0)
    assert imputer.statistics_[loan_index] == learned_train_median
    assert sparse.issparse(train_matrix)
    assert sparse.issparse(validation_matrix)
    assert train_matrix.shape[1] == validation_matrix.shape[1]
    assert np.isfinite(train_matrix.data).all()
    assert np.isfinite(validation_matrix.data).all()


def test_transform_is_deterministic_and_target_independent() -> None:
    first = _frame()
    changed_target = first.copy()
    changed_target["loan_approved"] = 1 - changed_target["loan_approved"]
    changed_target["target_denied"] = 1 - changed_target["target_denied"]

    left = build_feature_pipeline().fit_transform(first)
    right = build_feature_pipeline().fit_transform(changed_target)

    assert left.shape == right.shape
    assert (left != right).nnz == 0
