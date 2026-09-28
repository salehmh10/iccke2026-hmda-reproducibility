"""Importable pipeline utilities for Prompt 2 linear-family models."""

from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, GammaRegressor, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler


def category_to_object(values):
    """Convert pandas categories to objects while keeping missing values."""
    if isinstance(values, pd.DataFrame):
        converted = values.astype(object)
        return converted.where(pd.notna(converted), np.nan)
    array = np.asarray(values, dtype=object)
    array[pd.isna(array)] = np.nan
    return array


def canonical_json(value) -> str:
    """Return stable compact JSON for IDs and configuration comparisons."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def configuration_digest(value, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def deterministic_experiment_id(
    model_name: str,
    sensitive_mode: str,
    target_mode: str,
    evaluation_stage: str,
    fold_number,
    configuration: dict,
    version: str = "linear_compact_v1",
) -> str:
    fold_text = "na" if fold_number is None else str(fold_number)
    digest = configuration_digest({"version": version, "configuration": configuration})
    return (
        f"p2__{model_name}__{sensitive_mode}__{target_mode}__"
        f"{evaluation_stage}__fold-{fold_text}__cfg-{digest}"
    )


def make_one_hot_encoder():
    """Build a sparse version-compatible encoder with rare-level control."""
    kwargs = {
        "handle_unknown": "infrequent_if_exist",
        "min_frequency": 0.001,
        "max_categories": 60,
        "dtype": np.float64,
    }
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = True
    else:  # pragma: no cover - for older scikit-learn only
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)


def make_preprocessor(numeric_features, categorical_features, scaler_name: str):
    """Return a new unfitted linear-family preprocessor."""
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {scaler_name}")
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("to_object", FunctionTransformer(category_to_object, validate=False, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def make_estimator(configuration: dict, fit_overrides: dict | None = None):
    """Return a new unfitted estimator from a frozen configuration."""
    config = deepcopy(configuration)
    if fit_overrides:
        config.update(fit_overrides)
    model_name = config["model_name"]
    if model_name == "dummy_median":
        return DummyRegressor(strategy="median")
    if model_name == "linear_regression":
        return LinearRegression(tol=1e-6, n_jobs=1)
    if model_name == "ridge":
        return Ridge(alpha=float(config["alpha"]), solver="lsqr", tol=1e-4, max_iter=1000)
    if model_name == "lasso":
        return Lasso(
            alpha=float(config["alpha"]), max_iter=int(config.get("max_iter", 5000)),
            tol=float(config.get("tol", 1e-4)), selection="cyclic",
        )
    if model_name == "elastic_net":
        return ElasticNet(
            alpha=float(config["alpha"]), l1_ratio=float(config["l1_ratio"]),
            max_iter=int(config.get("max_iter", 5000)), tol=float(config.get("tol", 1e-4)),
            selection="cyclic",
        )
    if model_name == "gamma_regressor":
        return GammaRegressor(
            alpha=float(config["alpha"]), solver="lbfgs",
            max_iter=int(config.get("max_iter", 500)), tol=float(config.get("tol", 1e-5)),
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_complete_pipeline(configuration, numeric_features, categorical_features, fit_overrides=None):
    """Build a fresh complete raw-DataFrame-to-prediction object."""
    preprocessor = make_preprocessor(numeric_features, categorical_features, configuration["scaler"])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", make_estimator(configuration, fit_overrides=fit_overrides)),
    ])
    if configuration["target_mode"] == "log1p":
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=True,
        )
    return pipeline


def fitted_pipeline(model):
    """Return the fitted inner Pipeline for raw or transformed targets."""
    if isinstance(model, TransformedTargetRegressor):
        return model.regressor_
    return model


def fitted_estimator(model):
    return fitted_pipeline(model).named_steps["regressor"]


def transformed_feature_names(model):
    return fitted_pipeline(model).named_steps["preprocessor"].get_feature_names_out()

