"""Train-fitted preprocessing used by Regression V2 Prompt 2 models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MISSING_SENTINEL = "__MISSING__"
EXCLUDED_LINEAR_CATEGORICAL = {
    "respondent_id",
    "msamd_name",
    "county_name",
    "county_code",
    "census_tract_number",
}


def _frame(X, feature_names: Iterable[str]) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Preprocessing requires a pandas DataFrame with named columns.")
    names = list(feature_names)
    missing = sorted(set(names) - set(X.columns))
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return X.loc[:, names]


def _is_numeric(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series.dtype))


def build_linear_compact_v2(train_df: pd.DataFrame, feature_roles: dict) -> dict:
    """Freeze the compact Lasso feature pack from Train cardinalities only."""
    contract = list(feature_roles["contracts"]["main_without_sensitive_without_lender"])
    sensitive = set(feature_roles["sensitive_fields"])
    audit = set(feature_roles["audit_only_fields"])
    target = set(feature_roles["target_and_alias_exclusions"])
    allowed = [c for c in contract if c not in sensitive | audit | target]
    numeric = [c for c in allowed if _is_numeric(train_df[c])]
    categorical = [
        c
        for c in allowed
        if not _is_numeric(train_df[c])
        and c not in EXCLUDED_LINEAR_CATEGORICAL
        and int(train_df[c].nunique(dropna=True)) <= 100
    ]
    features = numeric + categorical
    if "respondent_id" in features:
        raise AssertionError("linear_compact_v2 must exclude respondent_id")
    return {
        "name": "linear_compact_v2",
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "train_cardinality_limit": 100,
    }


def make_lasso_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, list(numeric_features)), ("categorical", categorical, list(categorical_features))],
        sparse_threshold=1.0,
        remainder="drop",
    )


class TrainFittedCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Median + frequency/ordinal encoder fitted only on the supplied Train rows."""

    def __init__(self, feature_names: list[str] | None = None, high_cardinality_threshold: int = 100):
        self.feature_names = feature_names
        self.high_cardinality_threshold = high_cardinality_threshold

    def fit(self, X: pd.DataFrame, y=None):
        names = list(self.feature_names or X.columns)
        frame = _frame(X, names)
        self.feature_names_in_ = names
        self.numeric_features_ = [c for c in names if _is_numeric(frame[c])]
        categorical = [c for c in names if c not in self.numeric_features_]
        cardinality = {c: int(frame[c].nunique(dropna=True)) for c in categorical}
        self.high_cardinality_features_ = [c for c in categorical if cardinality[c] > self.high_cardinality_threshold]
        self.ordinal_features_ = [c for c in categorical if c not in self.high_cardinality_features_]
        self.numeric_medians_ = {
            c: float(pd.to_numeric(frame[c], errors="coerce").median()) for c in self.numeric_features_
        }
        self.frequency_maps_ = {}
        for c in self.high_cardinality_features_:
            values = frame[c].fillna(MISSING_SENTINEL).astype(str)
            self.frequency_maps_[c] = (values.value_counts(dropna=False) / len(values)).to_dict()
        self.ordinal_maps_ = {}
        for c in self.ordinal_features_:
            values = frame[c].fillna(MISSING_SENTINEL).astype(str)
            categories = sorted(values.unique().tolist())
            self.ordinal_maps_[c] = {value: index for index, value in enumerate(categories)}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = _frame(X, self.feature_names_in_)
        arrays: list[np.ndarray] = []
        for c in self.numeric_features_:
            values = pd.to_numeric(frame[c], errors="coerce").fillna(self.numeric_medians_[c])
            arrays.append(values.to_numpy(dtype=np.float32, copy=False))
        for c in self.high_cardinality_features_:
            values = frame[c].fillna(MISSING_SENTINEL).astype(str)
            arrays.append(values.map(self.frequency_maps_[c]).fillna(0.0).to_numpy(dtype=np.float32))
        for c in self.ordinal_features_:
            values = frame[c].fillna(MISSING_SENTINEL).astype(str)
            arrays.append(values.map(self.ordinal_maps_[c]).fillna(-1).to_numpy(dtype=np.float32))
        if not arrays:
            return np.empty((len(frame), 0), dtype=np.float32)
        return np.column_stack(arrays).astype(np.float32, copy=False)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            self.numeric_features_ + self.high_cardinality_features_ + self.ordinal_features_, dtype=object
        )


def make_dense_tree_preprocessor(feature_names: list[str], high_cardinality_threshold: int = 100):
    return TrainFittedCategoricalEncoder(list(feature_names), high_cardinality_threshold)


class SparseXGBPreprocessor(BaseEstimator, TransformerMixin):
    """Sparse one-hot preprocessing with Train-fitted frequency encoding."""

    def __init__(self, feature_names: list[str] | None = None, high_cardinality_threshold: int = 100):
        self.feature_names = feature_names
        self.high_cardinality_threshold = high_cardinality_threshold

    def fit(self, X: pd.DataFrame, y=None):
        names = list(self.feature_names or X.columns)
        frame = _frame(X, names)
        self.feature_names_in_ = names
        self.numeric_features_ = [c for c in names if _is_numeric(frame[c])]
        categorical = [c for c in names if c not in self.numeric_features_]
        cardinality = {c: int(frame[c].nunique(dropna=True)) for c in categorical}
        self.high_cardinality_features_ = [c for c in categorical if cardinality[c] > self.high_cardinality_threshold]
        self.low_cardinality_features_ = [c for c in categorical if c not in self.high_cardinality_features_]
        self.numeric_medians_ = {
            c: float(pd.to_numeric(frame[c], errors="coerce").median()) for c in self.numeric_features_
        }
        self.frequency_maps_ = {}
        for c in self.high_cardinality_features_:
            values = frame[c].fillna(MISSING_SENTINEL).astype(str)
            self.frequency_maps_[c] = (values.value_counts(dropna=False) / len(values)).to_dict()
        self.one_hot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        low = frame[self.low_cardinality_features_].fillna(MISSING_SENTINEL).astype(str)
        self.one_hot_.fit(low)
        return self

    def transform(self, X: pd.DataFrame):
        frame = _frame(X, self.feature_names_in_)
        dense_columns: list[np.ndarray] = []
        for c in self.numeric_features_:
            values = pd.to_numeric(frame[c], errors="coerce").fillna(self.numeric_medians_[c])
            dense_columns.append(values.to_numpy(dtype=np.float32, copy=False))
        for c in self.high_cardinality_features_:
            values = frame[c].fillna(MISSING_SENTINEL).astype(str)
            dense_columns.append(values.map(self.frequency_maps_[c]).fillna(0.0).to_numpy(dtype=np.float32))
        if dense_columns:
            dense = sparse.csr_matrix(np.column_stack(dense_columns), dtype=np.float32)
        else:
            dense = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
        low = frame[self.low_cardinality_features_].fillna(MISSING_SENTINEL).astype(str)
        encoded = self.one_hot_.transform(low).tocsr()
        return sparse.hstack([dense, encoded], format="csr", dtype=np.float32)


def make_xgb_preprocessor(feature_names: list[str], high_cardinality_threshold: int = 100):
    return SparseXGBPreprocessor(list(feature_names), high_cardinality_threshold)


class CatBoostFramePreprocessor(BaseEstimator, TransformerMixin):
    """Preserve numeric columns and normalize categorical values as strings."""

    def __init__(self, feature_names: list[str] | None = None):
        self.feature_names = feature_names

    def fit(self, X: pd.DataFrame, y=None):
        names = list(self.feature_names or X.columns)
        frame = _frame(X, names)
        self.feature_names_in_ = names
        self.numeric_features_ = [c for c in names if _is_numeric(frame[c])]
        self.categorical_features_ = [c for c in names if c not in self.numeric_features_]
        self.cat_feature_indices_ = [names.index(c) for c in self.categorical_features_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = _frame(X, self.feature_names_in_).copy()
        for c in self.categorical_features_:
            frame[c] = frame[c].fillna(MISSING_SENTINEL).astype(str)
        for c in self.numeric_features_:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(np.float64)
        return frame
