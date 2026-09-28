"""Leakage-safe, synergy-aware feature candidates for feature generation V2.

The V1 transformer remains untouched so its frozen serialized artifacts keep
their original executable contract.  This module creates an independent V2
pipeline.  Row-local formulas need no fit statistics; robust scaling and
category-conditioned statistics are fitted only from the rows passed to
``fit`` and have explicit unseen-category fallbacks.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from .financial import (
    CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    FinancialFeatureEngineer,
)


FEATURE_FAMILIES: tuple[str, ...] = (
    "numeric_relative",
    "nonlinear",
    "category_context",
    "onehot_numeric",
)

ROBUST_SOURCE_FEATURES: tuple[str, ...] = (
    "loan_amount_000s",
    "applicant_income_000s",
    "loan_to_income",
    "tract_to_msamd_income_ratio",
    "applicant_income_to_area_median",
)

# These pairs are deliberately low-cardinality and application-time. Agency is
# excluded from conditioned baselines/interactions to avoid strengthening
# lender memorization. All statistics are X-only; the target is never used.
CATEGORY_CONTEXT_PAIRS: tuple[tuple[str, str], ...] = (
    ("loan_type_name", "loan_amount_000s"),
    ("loan_type_name", "loan_to_income"),
    ("property_type_name", "loan_amount_000s"),
    ("property_type_name", "loan_to_income"),
    ("loan_purpose_name", "loan_amount_000s"),
    ("loan_purpose_name", "applicant_income_000s"),
    ("loan_purpose_name", "loan_to_income"),
    ("owner_occupancy_name", "applicant_income_000s"),
    ("owner_occupancy_name", "loan_to_income"),
    ("preapproval_name", "loan_amount_000s"),
    ("preapproval_name", "loan_to_income"),
    ("lien_status_name", "loan_amount_000s"),
    ("lien_status_name", "loan_to_income"),
)

ONEHOT_NUMERIC_PAIRS: tuple[tuple[str, str], ...] = (
    ("loan_type_name", "loan_amount_000s"),
    ("loan_type_name", "loan_to_income"),
    ("property_type_name", "loan_amount_000s"),
    ("property_type_name", "loan_to_income"),
    ("loan_purpose_name", "loan_amount_000s"),
    ("loan_purpose_name", "loan_to_income"),
    ("owner_occupancy_name", "applicant_income_000s"),
    ("owner_occupancy_name", "loan_to_income"),
    ("preapproval_name", "loan_amount_000s"),
    ("lien_status_name", "loan_amount_000s"),
    ("lien_status_name", "loan_to_income"),
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator_values = pd.to_numeric(numerator, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    denominator_values = pd.to_numeric(denominator, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    output = np.full(numerator_values.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator_values)
        & np.isfinite(denominator_values)
        & (denominator_values > 0.0)
    )
    np.divide(numerator_values, denominator_values, out=output, where=valid)
    return pd.Series(output, index=numerator.index, dtype="float64")


def _normalized_difference(left: pd.Series, right: pd.Series) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    denominator = np.abs(left_values) + np.abs(right_values) + 1e-9
    output = (left_values - right_values) / denominator
    output[~np.isfinite(left_values) | ~np.isfinite(right_values)] = np.nan
    return pd.Series(output, index=left.index, dtype="float64")


def _category_key(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("__MISSING__")


def _level_slug(value: str) -> str:
    readable = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:28] or "level"
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:6]
    return f"{readable}_{suffix}"


class AdvancedFinancialFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create controlled V2 candidate families without using the target.

    Parameters
    ----------
    families:
        Any subset of ``FEATURE_FAMILIES``. An empty tuple reproduces the V1
        pre-encoding feature frame.
    min_category_count:
        Training-row support below which a fitted category is marked rare.
    strict:
        Preserve V1 fail-closed schema validation.
    """

    def __init__(
        self,
        *,
        families: Sequence[str] = FEATURE_FAMILIES,
        min_category_count: int = 500,
        strict: bool = True,
    ) -> None:
        self.families = tuple(families)
        self.min_category_count = min_category_count
        self.strict = strict

    def fit(self, X: pd.DataFrame, y: object = None) -> "AdvancedFinancialFeatureEngineer":
        unknown = sorted(set(self.families).difference(FEATURE_FAMILIES))
        if unknown:
            raise ValueError(f"unknown V2 feature families: {unknown}")
        if self.min_category_count < 1:
            raise ValueError("min_category_count must be positive")

        self.base_ = FinancialFeatureEngineer(strict=self.strict).fit(X)
        frame = self.base_.transform(X)
        self.category_support_: dict[str, dict[str, int]] = {}
        self.category_levels_: dict[str, tuple[str, ...]] = {}
        if "category_context" in self.families or "onehot_numeric" in self.families:
            for category in CATEGORICAL_FEATURES:
                keys = _category_key(frame[category])
                counts = keys.value_counts(dropna=False).sort_index()
                self.category_support_[category] = {
                    str(level): int(count) for level, count in counts.items()
                }
                self.category_levels_[category] = tuple(sorted(self.category_support_[category]))

        self.category_statistics_: dict[tuple[str, str], dict[str, object]] = {}
        if "category_context" in self.families:
            for category, numeric in CATEGORY_CONTEXT_PAIRS:
                keys = _category_key(frame[category])
                values = pd.to_numeric(frame[numeric], errors="coerce").astype("float64")
                grouped = pd.DataFrame({"key": keys, "value": values}).groupby(
                    "key", observed=False
                )["value"]
                medians = grouped.median()
                q25 = grouped.quantile(0.25)
                q75 = grouped.quantile(0.75)
                global_median = float(values.median())
                global_iqr = float(values.quantile(0.75) - values.quantile(0.25))
                if not np.isfinite(global_median):
                    global_median = 0.0
                if not np.isfinite(global_iqr) or global_iqr <= 1e-9:
                    global_iqr = 1.0
                self.category_statistics_[(category, numeric)] = {
                    "median": {str(key): float(value) for key, value in medians.items()},
                    "iqr": {
                        str(key): float(max(q75.loc[key] - q25.loc[key], 1e-9))
                        for key in medians.index
                    },
                    "global_median": global_median,
                    "global_iqr": global_iqr,
                }

        self.robust_statistics_: dict[str, tuple[float, float]] = {}
        if "nonlinear" in self.families:
            for numeric in ROBUST_SOURCE_FEATURES:
                values = pd.to_numeric(frame[numeric], errors="coerce").astype("float64")
                median = float(values.median())
                iqr = float(values.quantile(0.75) - values.quantile(0.25))
                if not np.isfinite(median):
                    median = 0.0
                if not np.isfinite(iqr) or iqr <= 1e-9:
                    iqr = 1.0
                self.robust_statistics_[numeric] = (median, iqr)

        transformed = self._transform_frame(frame, X)
        self.output_features_ = np.asarray(transformed.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def _transform_frame(self, frame: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        output: dict[str, object] = {
            name: frame[name].copy() for name in MODEL_NUMERIC_FEATURES
        }
        loan = frame["loan_amount_000s"].astype("float64")
        income = frame["applicant_income_000s"].astype("float64")
        lti = frame["loan_to_income"].astype("float64")
        area_income = pd.to_numeric(
            X["hud_median_family_income"], errors="coerce"
        ).astype("float64").mask(lambda values: values < 0.0) / 1_000.0
        tract_income = area_income * frame["tract_to_msamd_income_ratio"].astype("float64")

        if "numeric_relative" in self.families:
            output["loan_to_area_median"] = _safe_ratio(loan, area_income)
            output["loan_to_tract_income"] = _safe_ratio(loan, tract_income)
            output["applicant_income_to_tract_income"] = _safe_ratio(income, tract_income)
            output["loan_minus_income_000s"] = loan - income
            output["abs_loan_minus_income_000s"] = (loan - income).abs()
            output["loan_income_normalized_gap"] = _normalized_difference(loan, income)
            output["loan_share_of_loan_plus_income"] = _safe_ratio(loan, loan + income)
            output["applicant_minus_area_income_000s"] = income - area_income
            output["applicant_minus_tract_income_000s"] = income - tract_income
            output["tract_minus_area_income_000s"] = tract_income - area_income

        if "nonlinear" in self.families:
            output["sqrt_loan_amount_000s"] = np.sqrt(loan.clip(lower=0.0))
            output["sqrt_applicant_income_000s"] = np.sqrt(income.clip(lower=0.0))
            output["log1p_loan_to_income"] = np.log1p(lti.clip(lower=0.0))
            output["log1p_applicant_income_to_area_median"] = np.log1p(
                frame["applicant_income_to_area_median"].astype("float64").clip(lower=0.0)
            )
            output["tract_income_ratio_abs_deviation"] = (
                frame["tract_to_msamd_income_ratio"].astype("float64") - 1.0
            ).abs()
            output["loan_to_income_gt_2"] = lti.gt(2.0).astype("float64")
            output["loan_to_income_gt_3"] = lti.gt(3.0).astype("float64")
            output["applicant_income_below_area"] = income.lt(area_income).astype("float64")
            output["tract_income_below_msa"] = frame[
                "tract_to_msamd_income_ratio"
            ].astype("float64").lt(1.0).astype("float64")
            output["high_lti_low_tract_income"] = (
                lti.gt(2.0)
                & frame["tract_to_msamd_income_ratio"].astype("float64").lt(1.0)
            ).astype("float64")
            for numeric, (median, iqr) in self.robust_statistics_.items():
                output[f"{numeric}__robust_z"] = (
                    frame[numeric].astype("float64") - median
                ) / iqr

        if "category_context" in self.families:
            row_count = float(sum(next(iter(self.category_support_.values())).values()))
            for category in CATEGORICAL_FEATURES:
                keys = _category_key(frame[category])
                support = self.category_support_[category]
                counts = keys.map(support).fillna(0.0).astype("float64")
                output[f"{category}__frequency"] = counts / row_count
                output[f"{category}__rare_or_unseen"] = (
                    counts < float(self.min_category_count)
                ).astype("float64")
            for category, numeric in CATEGORY_CONTEXT_PAIRS:
                keys = _category_key(frame[category])
                values = frame[numeric].astype("float64")
                stats = self.category_statistics_[(category, numeric)]
                median = keys.map(stats["median"]).fillna(stats["global_median"]).astype("float64")
                iqr = keys.map(stats["iqr"]).fillna(stats["global_iqr"]).astype("float64")
                prefix = f"{numeric}__by__{category}"
                output[f"{prefix}__relative_median"] = (
                    values - median
                ) / (median.abs() + 1e-9)
                output[f"{prefix}__robust_z"] = (values - median) / iqr.clip(lower=1e-9)

        if "onehot_numeric" in self.families:
            for category, numeric in ONEHOT_NUMERIC_PAIRS:
                keys = _category_key(frame[category])
                values = frame[numeric].astype("float64")
                for level in self.category_levels_[category]:
                    name = f"{numeric}__x__{category}__{_level_slug(level)}"
                    output[name] = np.where(keys.eq(level), values, 0.0)

        for category in CATEGORICAL_FEATURES:
            output[category] = frame[category].astype("object")
        return pd.DataFrame(output, index=frame.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "output_features_")
        frame = self.base_.transform(X)
        output = self._transform_frame(frame, X)
        missing = sorted(set(self.output_features_).difference(output.columns))
        if missing:
            raise RuntimeError(f"V2 transform lost fitted output columns: {missing}")
        return output.loc[:, self.output_features_].copy()

    def get_feature_names_out(
        self, input_features: Sequence[str] | None = None
    ) -> np.ndarray:
        check_is_fitted(self, "output_features_")
        return self.output_features_.copy()


def _one_hot_encoder() -> OneHotEncoder:
    common = {"handle_unknown": "ignore", "dtype": np.float64}
    try:
        return OneHotEncoder(sparse_output=True, **common)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(sparse=True, **common)


def build_advanced_feature_pipeline(
    *,
    families: Sequence[str] = FEATURE_FAMILIES,
    min_category_count: int = 500,
    strict_schema: bool = True,
) -> Pipeline:
    """Build an unfitted V2 pipeline with train-fitted dynamic selectors."""

    numeric_pipeline = Pipeline(
        steps=(
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scaler", StandardScaler()),
        )
    )
    categorical_pipeline = Pipeline(
        steps=(
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        )
    )
    preprocessor = ColumnTransformer(
        transformers=(
            (
                "numeric",
                numeric_pipeline,
                make_column_selector(dtype_include=np.number),
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(dtype_exclude=np.number),
            ),
        ),
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    return Pipeline(
        steps=(
            (
                "financial_v2",
                AdvancedFinancialFeatureEngineer(
                    families=families,
                    min_category_count=min_category_count,
                    strict=strict_schema,
                ),
            ),
            ("preprocess", preprocessor),
        )
    )
