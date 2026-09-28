"""Train-fitted sklearn preprocessing for engineered HMDA features."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .financial import (
    CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    FinancialFeatureEngineer,
)


def _one_hot_encoder() -> OneHotEncoder:
    """Construct a sparse encoder across supported sklearn keyword versions."""

    common = {"handle_unknown": "ignore", "dtype": np.float64}
    try:
        return OneHotEncoder(sparse_output=True, **common)
    except TypeError:  # pragma: no cover - compatibility with sklearn < 1.2
        return OneHotEncoder(sparse=True, **common)


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted sparse-compatible numeric/categorical preprocessor.

    Median values, scaling parameters, missingness-indicator availability, and
    categorical vocabulary are learned only when the caller fits this object.
    Callers must therefore pass the training split to ``fit``/``fit_transform``
    and use only ``transform`` for validation and test.
    """

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
    return ColumnTransformer(
        transformers=(
            ("numeric", numeric_pipeline, list(MODEL_NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ),
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def build_feature_pipeline(*, strict_schema: bool = True) -> Pipeline:
    """Build the complete unfitted feature pipeline.

    The row-local feature step learns no statistics. The second step must be fit
    on training rows only and is then reused unchanged for validation/test.
    """

    return Pipeline(
        steps=(
            ("financial", FinancialFeatureEngineer(strict=strict_schema)),
            ("preprocess", build_preprocessor()),
        )
    )
