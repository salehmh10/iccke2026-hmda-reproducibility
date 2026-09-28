"""Train-only preprocessing for the Prompt 3 Deep tabular models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer


SEED = 42
MISSING_TOKEN = "__PROMPT3_MISSING__"


def ordered_digest(values: Iterable[Any]) -> str:
    """Hash ordered identifiers without changing their order."""
    text = "\n".join(pd.Series(values, copy=False).astype(str).tolist())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_named_frame(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a named view after checking the complete feature contract."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Deep preprocessing requires a pandas DataFrame.")
    names = list(columns)
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique.")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return frame.loc[:, names]


def split_feature_types(
    train_frame: pd.DataFrame, feature_names: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Freeze numeric and categorical roles from Train dtypes only."""
    selected = select_named_frame(train_frame, feature_names)
    numeric = [name for name in selected if pd.api.types.is_numeric_dtype(selected[name].dtype)]
    categorical = [name for name in selected if name not in numeric]
    if len(numeric) + len(categorical) != selected.shape[1]:
        raise AssertionError("Feature role split is incomplete.")
    return numeric, categorical


def duplicate_safe_deciles(y: Any) -> np.ndarray:
    """Create deterministic target-decile labels while allowing tied edges."""
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Target deciles need finite, non-empty targets.")
    labels = pd.qcut(pd.Series(values), q=10, labels=False, duplicates="drop")
    result = labels.to_numpy(dtype=np.int16, na_value=-1)
    if (result < 0).any() or np.unique(result).size < 2:
        raise ValueError("Target values cannot support a stratified internal split.")
    return result


def make_ft_internal_split(
    y_train: Any,
    train_row_hashes: Iterable[Any],
    *,
    external_validation_row_hashes: Iterable[Any] | None = None,
    fit_rows: int = 360_000,
    early_stopping_rows: int = 40_000,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create the frozen FT split using only Development Train membership."""
    target = np.asarray(y_train, dtype=np.float64).reshape(-1)
    hashes = pd.Series(train_row_hashes, copy=False).astype(str).to_numpy()
    expected = fit_rows + early_stopping_rows
    if target.size != expected or hashes.size != expected:
        raise ValueError(f"FT split requires exactly {expected:,} Train rows.")
    if pd.Series(hashes).duplicated().any():
        raise ValueError("Train row hashes must be unique.")
    bins = duplicate_safe_deciles(target)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=early_stopping_rows, random_state=random_state
    )
    fit_index, stop_index = next(splitter.split(np.zeros(expected, dtype=np.int8), bins))
    fit_index = np.sort(fit_index.astype(np.int64, copy=False))
    stop_index = np.sort(stop_index.astype(np.int64, copy=False))
    if fit_index.size != fit_rows or stop_index.size != early_stopping_rows:
        raise AssertionError("FT internal role counts are incorrect.")
    if np.intersect1d(fit_index, stop_index).size:
        raise AssertionError("FT internal roles overlap.")

    fit_hashes = hashes[fit_index]
    stop_hashes = hashes[stop_index]
    external_overlap = 0
    if external_validation_row_hashes is not None:
        external = set(pd.Series(external_validation_row_hashes, copy=False).astype(str))
        external_overlap = len((set(fit_hashes) | set(stop_hashes)) & external)
        if external_overlap:
            raise ValueError("FT internal rows overlap external Validation.")
    audit = {
        "random_state": int(random_state),
        "stratification": "duplicate-safe target deciles",
        "fit_rows": int(fit_index.size),
        "early_stopping_rows": int(stop_index.size),
        "internal_overlap": 0,
        "external_validation_overlap": int(external_overlap),
        "fit_row_hash_digest": ordered_digest(fit_hashes),
        "early_stopping_row_hash_digest": ordered_digest(stop_hashes),
    }
    return fit_index, stop_index, audit


def _categorical_strings(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    return values.fillna(MISSING_TOKEN).astype(str)


@dataclass
class RealMLPPreprocessor:
    """Train medians plus stable string categories for official RealMLP."""

    feature_names: list[str]
    numeric_features: list[str] | None = None
    categorical_features: list[str] | None = None
    numeric_medians_: dict[str, float] = field(default_factory=dict, init=False)
    categorical_vocabularies_: dict[str, list[str]] = field(default_factory=dict, init=False)
    fit_row_count_: int = field(default=0, init=False)
    fit_row_hash_digest_: str | None = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        train_frame: pd.DataFrame,
        y: Any = None,
        *,
        row_hashes: Iterable[Any] | None = None,
    ) -> "RealMLPPreprocessor":
        selected = select_named_frame(train_frame, self.feature_names)
        inferred_numeric, inferred_categorical = split_feature_types(selected, self.feature_names)
        numeric = list(self.numeric_features) if self.numeric_features is not None else inferred_numeric
        categorical = (
            list(self.categorical_features)
            if self.categorical_features is not None
            else inferred_categorical
        )
        if set(numeric).intersection(categorical) or set(numeric + categorical) != set(self.feature_names):
            raise ValueError("RealMLP feature roles must partition the feature contract.")
        self.numeric_features_ = numeric
        self.categorical_features_ = categorical
        self.numeric_medians_ = {}
        for name in numeric:
            values = pd.to_numeric(selected[name], errors="coerce")
            median = float(values.median())
            if not np.isfinite(median):
                raise ValueError(f"Numeric feature has no finite Train median: {name}")
            self.numeric_medians_[name] = median
        self.categorical_vocabularies_ = {}
        for name in categorical:
            values = _categorical_strings(selected[name])
            known = sorted(value for value in values.unique().tolist() if value != MISSING_TOKEN)
            self.categorical_vocabularies_[name] = known
        self.fit_row_count_ = len(selected)
        if row_hashes is not None:
            hashes = list(row_hashes)
            if len(hashes) != len(selected):
                raise ValueError("Preprocessing row hashes do not match Train row count.")
            self.fit_row_hash_digest_ = ordered_digest(hashes)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise RuntimeError("RealMLPPreprocessor is not fitted.")
        result = select_named_frame(frame, self.feature_names).copy()
        for name in self.numeric_features_:
            result[name] = (
                pd.to_numeric(result[name], errors="coerce")
                .fillna(self.numeric_medians_[name])
                .astype(np.float32)
            )
        for name in self.categorical_features_:
            result[name] = _categorical_strings(result[name])
        return result

    def fit_transform(self, train_frame: pd.DataFrame, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        return self.fit(train_frame, y, **fit_params).transform(train_frame)

    def evidence(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("RealMLPPreprocessor is not fitted.")
        return {
            "fit_row_count": self.fit_row_count_,
            "fit_row_hash_digest": self.fit_row_hash_digest_,
            "numeric_features": list(self.numeric_features_),
            "categorical_features": list(self.categorical_features_),
            "numeric_medians": dict(self.numeric_medians_),
            "categorical_vocabulary_sizes": {
                name: len(values) for name, values in self.categorical_vocabularies_.items()
            },
            "missing_token": MISSING_TOKEN,
            "unknown_policy": "Official RealMLP unknown category maps through its Train-fitted encoder.",
        }


@dataclass
class FTPreprocessor:
    """Median/quantile numeric data and indexed categorical data for FT."""

    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    n_quantiles: int = 1000
    random_state: int = SEED
    numeric_medians_: dict[str, float] = field(default_factory=dict, init=False)
    quantile_transformer_: QuantileTransformer | None = field(default=None, init=False)
    categorical_vocabularies_: dict[str, dict[str, int]] = field(default_factory=dict, init=False)
    cardinalities_: list[int] = field(default_factory=list, init=False)
    fit_row_count_: int = field(default=0, init=False)
    fit_row_hash_digest_: str | None = field(default=None, init=False)
    is_fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if set(self.numeric_features).intersection(self.categorical_features):
            raise ValueError("FT numeric and categorical roles overlap.")
        if set(self.numeric_features + self.categorical_features) != set(self.feature_names):
            raise ValueError("FT feature roles must partition the feature contract.")

    def fit(
        self,
        train_frame: pd.DataFrame,
        y: Any = None,
        *,
        row_hashes: Iterable[Any] | None = None,
    ) -> "FTPreprocessor":
        selected = select_named_frame(train_frame, self.feature_names)
        self.numeric_medians_ = {}
        numeric_arrays: list[np.ndarray] = []
        for name in self.numeric_features:
            values = pd.to_numeric(selected[name], errors="coerce")
            median = float(values.median())
            if not np.isfinite(median):
                raise ValueError(f"Numeric feature has no finite Train median: {name}")
            self.numeric_medians_[name] = median
            numeric_arrays.append(values.fillna(median).to_numpy(dtype=np.float64, copy=False))
        numeric = (
            np.column_stack(numeric_arrays)
            if numeric_arrays
            else np.empty((len(selected), 0), dtype=np.float64)
        )
        self.quantile_transformer_ = None
        if numeric.shape[1]:
            self.quantile_transformer_ = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, len(selected)),
                output_distribution="normal",
                random_state=self.random_state,
                subsample=None,
                copy=True,
            )
            self.quantile_transformer_.fit(numeric)

        self.categorical_vocabularies_ = {}
        self.cardinalities_ = []
        for name in self.categorical_features:
            values = _categorical_strings(selected[name])
            known = sorted(value for value in values.unique().tolist() if value != MISSING_TOKEN)
            mapping = {value: index + 2 for index, value in enumerate(known)}
            self.categorical_vocabularies_[name] = mapping
            self.cardinalities_.append(len(mapping) + 2)
        self.fit_row_count_ = len(selected)
        if row_hashes is not None:
            hashes = list(row_hashes)
            if len(hashes) != len(selected):
                raise ValueError("Preprocessing row hashes do not match fit row count.")
            self.fit_row_hash_digest_ = ordered_digest(hashes)
        self.is_fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError("FTPreprocessor is not fitted.")
        selected = select_named_frame(frame, self.feature_names)
        numeric_arrays = [
            pd.to_numeric(selected[name], errors="coerce")
            .fillna(self.numeric_medians_[name])
            .to_numpy(dtype=np.float64, copy=False)
            for name in self.numeric_features
        ]
        numeric = (
            np.column_stack(numeric_arrays)
            if numeric_arrays
            else np.empty((len(selected), 0), dtype=np.float64)
        )
        if self.quantile_transformer_ is not None:
            numeric = self.quantile_transformer_.transform(numeric)
        numeric = np.ascontiguousarray(numeric, dtype=np.float32)

        categorical_arrays: list[np.ndarray] = []
        for name in self.categorical_features:
            values = _categorical_strings(selected[name])
            mapping = self.categorical_vocabularies_[name]
            # Pandas/Arrow may expose a read-only view. The reserved missing
            # index is assigned below, so require a private writable array.
            encoded = values.map(mapping).fillna(1).to_numpy(dtype=np.int64, copy=True)
            encoded[values.to_numpy() == MISSING_TOKEN] = 0
            categorical_arrays.append(encoded)
        categorical = (
            np.column_stack(categorical_arrays)
            if categorical_arrays
            else np.empty((len(selected), 0), dtype=np.int64)
        )
        return numeric, np.ascontiguousarray(categorical, dtype=np.int64)

    def fit_transform(self, train_frame: pd.DataFrame, y: Any = None, **fit_params: Any) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(train_frame, y, **fit_params).transform(train_frame)

    def evidence(self) -> dict[str, Any]:
        if not self.is_fitted_:
            raise RuntimeError("FTPreprocessor is not fitted.")
        return {
            "fit_row_count": self.fit_row_count_,
            "fit_row_hash_digest": self.fit_row_hash_digest_,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "n_quantiles": min(self.n_quantiles, self.fit_row_count_),
            "output_distribution": "normal",
            "quantile_subsample": None,
            "random_state": self.random_state,
            "numeric_medians": dict(self.numeric_medians_),
            "categorical_vocabulary_sizes": {
                name: len(mapping) for name, mapping in self.categorical_vocabularies_.items()
            },
            "cardinalities": list(self.cardinalities_),
            "missing_index": 0,
            "unknown_index": 1,
            "known_start_index": 2,
        }
