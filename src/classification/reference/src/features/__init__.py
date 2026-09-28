"""Schema-aware financial features and preprocessing."""
"""Leakage-safe feature construction for the HMDA benchmark."""

from .financial import (
    CATEGORICAL_FEATURES,
    ENGINEERED_NUMERIC_FEATURES,
    EXCLUDED_FEATURES,
    MODEL_NUMERIC_FEATURES,
    FinancialFeatureEngineer,
)
from .preprocessing import build_feature_pipeline, build_preprocessor
from .advanced import (
    CATEGORY_CONTEXT_PAIRS,
    FEATURE_FAMILIES,
    ONEHOT_NUMERIC_PAIRS,
    AdvancedFinancialFeatureEngineer,
    build_advanced_feature_pipeline,
)

__all__ = [
    "CATEGORICAL_FEATURES",
    "ENGINEERED_NUMERIC_FEATURES",
    "EXCLUDED_FEATURES",
    "MODEL_NUMERIC_FEATURES",
    "FinancialFeatureEngineer",
    "build_feature_pipeline",
    "build_preprocessor",
    "CATEGORY_CONTEXT_PAIRS",
    "FEATURE_FAMILIES",
    "ONEHOT_NUMERIC_PAIRS",
    "AdvancedFinancialFeatureEngineer",
    "build_advanced_feature_pipeline",
]
