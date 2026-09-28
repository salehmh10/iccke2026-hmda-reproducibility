"""Fixed Quantile Residual Specialist primitives for Prompt 4B2.

This module contains model primitives only. It does not discover project
files, load Development or IID data, select a candidate, or fit on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    from .prompt4b_crossfit import (
        GLOBAL_FEATURE,
        MetaPreprocessor,
        load_experimental_bundle,
        save_experimental_bundle,
    )
    from .prompt4b_residual import residual_target
except ImportError:  # Direct use with regression_v2/src on sys.path.
    from prompt4b_crossfit import (
        GLOBAL_FEATURE,
        MetaPreprocessor,
        load_experimental_bundle,
        save_experimental_bundle,
    )
    from prompt4b_residual import residual_target


SEED = 42
THREAD_COUNT = 4
QUANTILE_ITERATIONS = 791
ALLOWED_QUANTILE_ALPHAS = (0.60, 0.65)

_BASE_PARAMETERS: dict[str, Any] = {
    "iterations": QUANTILE_ITERATIONS,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20,
    "random_strength": 1,
    "random_seed": SEED,
    "thread_count": THREAD_COUNT,
    "verbose": False,
}


def validate_quantile_alpha(alpha: float) -> float:
    """Return one of the two authorized Prompt 4B2 Quantile levels."""
    value = float(alpha)
    if not any(np.isclose(value, allowed, rtol=0.0, atol=1e-12) for allowed in ALLOWED_QUANTILE_ALPHAS):
        raise ValueError("Prompt 4B2 authorizes only Quantile alpha 0.60 or 0.65.")
    return next(allowed for allowed in ALLOWED_QUANTILE_ALPHAS if np.isclose(value, allowed, rtol=0.0, atol=1e-12))


def quantile_parameters(alpha: float) -> dict[str, Any]:
    """Build the exact fixed CatBoost configuration for one Quantile role."""
    value = validate_quantile_alpha(alpha)
    return {
        **_BASE_PARAMETERS,
        "loss_function": f"Quantile:alpha={value:.2f}",
    }


QUANTILE_PARAMETERS = {
    alpha: quantile_parameters(alpha) for alpha in ALLOWED_QUANTILE_ALPHAS
}


def operational_tail_mask(y_true: Any, q90_train: float) -> np.ndarray:
    """Return the frozen operational Tail rule: target strictly above q90."""
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    threshold = float(q90_train)
    if target.size == 0 or not np.isfinite(target).all():
        raise ValueError("Operational Tail targets must be non-empty and finite.")
    if not np.isfinite(threshold):
        raise ValueError("q90_train must be finite.")
    return target > threshold


def prepare_tail_residual_fit_data(
    frame: pd.DataFrame,
    y_true: Any,
    global_oof_prediction: Any,
    base_features: Sequence[str],
    q90_train: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create an explicit Tail-only working copy and its leakage-safe target.

    ``frame`` must already contain the saved OOF Global prediction in
    ``global_prediction_feature``. The returned Boolean mask refers to the
    original row order and proves which rows entered the Tail-only fit.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Quantile residual fitting requires a pandas DataFrame.")
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    global_prediction = np.asarray(global_oof_prediction, dtype=np.float64).reshape(-1)
    if len(frame) != target.size or target.shape != global_prediction.shape:
        raise ValueError("Quantile residual inputs must be row-aligned.")
    names = list(base_features) + [GLOBAL_FEATURE]
    if len(names) != len(set(names)):
        raise ValueError("Quantile residual feature names must be unique.")
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing Quantile residual features: {missing}")
    mask = operational_tail_mask(target, q90_train)
    if not np.any(mask):
        raise ValueError("The operational Tail contains no fitting rows.")
    expected_global = pd.to_numeric(frame[GLOBAL_FEATURE], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(expected_global).all() or not np.array_equal(expected_global, global_prediction):
        raise ValueError("global_prediction_feature must equal the aligned saved OOF Global prediction.")
    features = frame.loc[mask, names].copy()
    residual = residual_target(target, global_prediction)[mask]
    return features, residual, mask


def fit_quantile_residual_model(
    X_tail: pd.DataFrame,
    residual_tail: Any,
    base_features: Sequence[str],
    *,
    alpha: float,
) -> tuple[Any, MetaPreprocessor, int]:
    """Fit one fixed 791-iteration Quantile model on preselected Tail rows.

    Callers should use :func:`prepare_tail_residual_fit_data` or the combined
    :func:`fit_tail_quantile_residual_model` to construct the Tail-only rows.
    There is no parameter override or early stopping in this bounded design.
    """
    from catboost import CatBoostRegressor

    if not isinstance(X_tail, pd.DataFrame):
        raise TypeError("Quantile residual fitting requires a pandas DataFrame.")
    target = np.asarray(residual_tail, dtype=np.float64).reshape(-1)
    if len(X_tail) == 0 or len(X_tail) != target.size or not np.isfinite(target).all():
        raise ValueError("Quantile residual Tail rows and targets are invalid.")
    names = list(base_features)
    missing = sorted(set(names + [GLOBAL_FEATURE]) - set(X_tail.columns))
    if missing:
        raise ValueError(f"Missing Quantile residual features: {missing}")
    config = quantile_parameters(alpha)
    if config["iterations"] != QUANTILE_ITERATIONS or config["random_seed"] != SEED:
        raise RuntimeError("The fixed Quantile scientific configuration changed.")
    if int(config["thread_count"]) > THREAD_COUNT:
        raise RuntimeError("Quantile CatBoost thread count exceeds four.")
    preprocessor = MetaPreprocessor(names).fit(X_tail)
    transformed = preprocessor.transform(X_tail)
    model = CatBoostRegressor(
        **config,
        allow_writing_files=False,
        task_type="CPU",
    )
    model.fit(
        transformed,
        target,
        cat_features=preprocessor.cat_feature_indices_,
        verbose=False,
    )
    selected = int(model.tree_count_)
    if selected != QUANTILE_ITERATIONS:
        raise RuntimeError("Quantile Residual Specialist did not fit exactly 791 iterations.")
    return model, preprocessor, selected


def fit_tail_quantile_residual_model(
    frame: pd.DataFrame,
    y_true: Any,
    global_oof_prediction: Any,
    base_features: Sequence[str],
    q90_train: float,
    *,
    alpha: float,
) -> tuple[Any, MetaPreprocessor, int, np.ndarray]:
    """Prepare and fit one authorized Tail-only Quantile Residual model."""
    X_tail, residual_tail, mask = prepare_tail_residual_fit_data(
        frame,
        y_true,
        global_oof_prediction,
        base_features,
        q90_train,
    )
    model, preprocessor, selected = fit_quantile_residual_model(
        X_tail,
        residual_tail,
        base_features,
        alpha=alpha,
    )
    return model, preprocessor, selected, mask


@dataclass
class QuantileResidualBundle:
    """Joblib-reloadable Quantile Residual Specialist inference bundle."""

    preprocessor: MetaPreprocessor
    model: Any
    metadata: dict[str, Any]

    @property
    def alpha(self) -> float:
        return validate_quantile_alpha(self.metadata.get("quantile_alpha"))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        _ = self.alpha
        prediction = np.asarray(
            self.model.predict(self.preprocessor.transform(frame)),
            dtype=np.float64,
        ).reshape(-1)
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise RuntimeError("Quantile Residual Specialist produced invalid predictions.")
        return prediction


QuantileResidualSpecialistBundle = QuantileResidualBundle


def save_quantile_residual_bundle(
    bundle: QuantileResidualBundle,
    destination: str | Path,
) -> dict[str, Any]:
    """Save a Quantile bundle through the existing experimental convention."""
    if not isinstance(bundle, QuantileResidualBundle):
        raise TypeError("save_quantile_residual_bundle requires a QuantileResidualBundle.")
    _ = bundle.alpha
    return save_experimental_bundle(bundle, Path(destination), dict(bundle.metadata))


def load_quantile_residual_bundle(
    destination: str | Path,
) -> tuple[QuantileResidualBundle, dict[str, Any]]:
    """Hash-check and reload one saved Quantile Residual bundle."""
    bundle, manifest = load_experimental_bundle(Path(destination))
    if not isinstance(bundle, QuantileResidualBundle):
        raise TypeError("Saved artifact is not a QuantileResidualBundle.")
    _ = bundle.alpha
    return bundle, manifest


def quantile_residual_diagnostics(
    y_true: Any,
    global_prediction: Any,
    predicted_residual: Any,
) -> dict[str, Any]:
    """Summarize a proposed Quantile residual correction on aligned rows."""
    actual_residual = residual_target(y_true, global_prediction)
    prediction = np.asarray(predicted_residual, dtype=np.float64).reshape(-1)
    if prediction.shape != actual_residual.shape or prediction.size == 0 or not np.isfinite(prediction).all():
        raise ValueError("Quantile residual diagnostics require finite aligned predictions.")
    error = prediction - actual_residual
    correlation = (
        float(np.corrcoef(prediction, actual_residual)[0, 1])
        if prediction.size > 1
        and np.std(prediction) > 0
        and np.std(actual_residual) > 0
        else np.nan
    )
    absolute_error = np.abs(error)
    return {
        "rows": int(prediction.size),
        "residual_mae": float(np.mean(absolute_error)),
        "residual_rmse": float(np.sqrt(np.mean(error**2))),
        "residual_signed_error": float(np.mean(error)),
        "median_residual_error": float(np.median(error)),
        "positive_correction_rate": float(np.mean(prediction > 0.0)),
        "predicted_true_residual_correlation": correlation,
        "correct_correction_direction": float(
            np.mean(np.sign(prediction) == np.sign(actual_residual))
        ),
        "p90_absolute_residual_error": float(np.quantile(absolute_error, 0.90)),
    }


residual_diagnostics = quantile_residual_diagnostics
