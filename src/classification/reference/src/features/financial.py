"""Schema-aware, row-local financial features for the HMDA extract.

The transformer in this module deliberately learns no distributional statistic.
It can therefore be placed before a train-fitted sklearn preprocessor without
creating cross-split leakage.  Its whitelist also prevents audit-only columns
from accidentally reaching a model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


SOURCE_NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_amount_000s",
    "applicant_income_000s",
    "hud_median_family_income",
    "tract_to_msamd_income",
)

# Only low-cardinality, application-time categorical fields are admitted.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "agency_name",
    "loan_type_name",
    "property_type_name",
    "loan_purpose_name",
    "owner_occupancy_name",
    "preapproval_name",
    "lien_status_name",
)

# Raw percentage and count columns used only to create a more interpretable
# conversion/ratio are not also emitted.  This avoids exact linear duplicates.
PASSTHROUGH_NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_amount_000s",
    "applicant_income_000s",
)

ENGINEERED_NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_to_income",
    "log1p_loan_amount_000s",
    "log1p_applicant_income_000s",
    "tract_to_msamd_income_ratio",
    "applicant_income_to_area_median",
    "applicant_income_zero",
    "hud_median_family_income_zero",
)

MODEL_NUMERIC_FEATURES: tuple[str, ...] = (
    *PASSTHROUGH_NUMERIC_FEATURES,
    *ENGINEERED_NUMERIC_FEATURES,
)

REQUIRED_SOURCE_FEATURES: tuple[str, ...] = (
    *SOURCE_NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
)

# This explicit deny list is documentation as well as a testable safety guard.
# The transformer emits a whitelist, so other unknown columns are excluded too.
EXCLUDED_FEATURES: frozenset[str] = frozenset(
    {
        # Raw and analytical targets / likely target aliases.
        "loan_approved",
        "target_denied",
        "action_taken",
        "action_taken_name",
        # Direct entity identifier.
        "respondent_id",
        # Protected attributes retained for fairness audit only.
        "applicant_ethnicity_name",
        "co_applicant_ethnicity_name",
        "applicant_race_name_1",
        "co_applicant_race_name_1",
        "applicant_sex_name",
        "co_applicant_sex_name",
        # Geography labels/codes and the explicit minority-composition proxy.
        "msamd_name",
        "state_name",
        "state_code",
        "county_name",
        "county_code",
        "census_tract_number",
        "minority_population",
        # Fine-grained tract counts can jointly behave like a tract fingerprint.
        "population",
        "number_of_owner_occupied_units",
        "number_of_1_to_4_family_units",
    }
)


def _safe_nonnegative_numeric(series: pd.Series) -> pd.Series:
    """Coerce to float and turn impossible negative magnitudes into missing."""

    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.mask(values < 0.0)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide where the denominator is finite and strictly positive.

    Undefined ratios are represented as NaN for the downstream train-fitted
    imputer; infinity is never emitted.
    """

    numerator_values = numerator.to_numpy(dtype="float64", na_value=np.nan)
    denominator_values = denominator.to_numpy(dtype="float64", na_value=np.nan)
    output = np.full(numerator_values.shape, np.nan, dtype="float64")
    valid = (
        np.isfinite(numerator_values)
        & np.isfinite(denominator_values)
        & (denominator_values > 0.0)
    )
    np.divide(numerator_values, denominator_values, out=output, where=valid)
    return pd.Series(output, index=numerator.index, dtype="float64")


class FinancialFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create row-local financial features and enforce the modeling whitelist.

    Parameters
    ----------
    strict:
        If true, fail with a clear schema error when any required source column
        is absent.  This prevents a misspelled or silently changed schema from
        changing the experiment's feature contract.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "FinancialFeatureEngineer requires a pandas DataFrame so that "
                "the HMDA schema can be validated."
            )
        missing = sorted(set(REQUIRED_SOURCE_FEATURES).difference(X.columns))
        if self.strict and missing:
            raise ValueError(f"Missing required HMDA feature columns: {missing}")

    def fit(self, X: pd.DataFrame, y: object = None) -> "FinancialFeatureEngineer":
        """Validate schema; no values or distributional statistics are learned."""

        self._validate_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.output_features_ = np.asarray(
            (*MODEL_NUMERIC_FEATURES, *CATEGORICAL_FEATURES), dtype=object
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the fixed, leakage-screened modeling frame."""

        check_is_fitted(self, "output_features_")
        self._validate_input(X)

        numeric = {
            name: _safe_nonnegative_numeric(
                X[name]
                if name in X
                else pd.Series(np.nan, index=X.index, dtype="float64")
            )
            for name in SOURCE_NUMERIC_FEATURES
        }

        loan = numeric["loan_amount_000s"]
        income = numeric["applicant_income_000s"]
        area_income_000s = numeric["hud_median_family_income"] / 1_000.0

        output = pd.DataFrame(index=X.index)
        for name in PASSTHROUGH_NUMERIC_FEATURES:
            output[name] = numeric[name]

        # Both loan and applicant income are reported in $1,000 units, so their
        # units cancel. This is a requested-balance / annual-income proxy, not DTI.
        output["loan_to_income"] = _safe_ratio(loan, income)
        output["log1p_loan_amount_000s"] = np.log1p(loan)
        output["log1p_applicant_income_000s"] = np.log1p(income)

        # HMDA reports this field as a percentage of MSA/MD median income.
        output["tract_to_msamd_income_ratio"] = (
            numeric["tract_to_msamd_income"] / 100.0
        )
        output["applicant_income_to_area_median"] = _safe_ratio(
            income, area_income_000s
        )

        # Explicit denominator-state indicators preserve the distinction between
        # an undefined ratio and an ordinary value after median imputation.
        output["applicant_income_zero"] = income.eq(0.0).astype("float64")
        output["hud_median_family_income_zero"] = numeric[
            "hud_median_family_income"
        ].eq(0.0).astype("float64")

        for name in CATEGORICAL_FEATURES:
            output[name] = (
                X[name].astype("object")
                if name in X
                else pd.Series(np.nan, index=X.index, dtype="object")
            )

        # Keep exact order stable for model serialization and split consistency.
        return output.loc[:, self.output_features_].copy()

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        """Return the stable pre-encoding feature names."""

        check_is_fitted(self, "output_features_")
        return self.output_features_.copy()
