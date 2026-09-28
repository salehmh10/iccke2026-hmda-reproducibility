"""Shared original-scale metrics for Regression V2 Prompt 2."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


VALID_TARGET_MODES = {"raw", "log1p"}


def transform_target(y, mode: str) -> np.ndarray:
    """Transform a non-negative target without changing its row order."""
    values = np.asarray(y, dtype=np.float64)
    if mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {mode}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Target values must be finite and non-negative.")
    return values.copy() if mode == "raw" else np.log1p(values)


def inverse_target(z, mode: str) -> np.ndarray:
    """Return predictions on the original target scale."""
    values = np.asarray(z, dtype=np.float64)
    if mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {mode}")
    return values.copy() if mode == "raw" else np.expm1(values)


def tail_membership(y_true, quantile: float) -> np.ndarray:
    """Return a fixed target-tail mask using Validation targets only."""
    values = np.asarray(y_true, dtype=np.float64)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    if not np.isfinite(values).all():
        raise ValueError("Tail membership requires finite target values.")
    threshold = float(np.quantile(values, quantile))
    return values >= threshold


def compute_regression_metrics(
    y_true,
    y_pred,
    *,
    fit_time_seconds: float = 0.0,
    prediction_time_seconds: float = 0.0,
    bundle_size_bytes: int = 0,
) -> dict:
    """Calculate every Prompt 2 metric on the original target scale."""
    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError("Target and prediction shapes differ.")
    if actual.ndim != 1 or actual.size == 0:
        raise ValueError("Metrics require non-empty one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics require finite targets and predictions.")

    error = predicted - actual
    absolute_error = np.abs(error)
    top_decile = tail_membership(actual, 0.90)
    top_five_percent = tail_membership(actual, 0.95)
    rmsle_pred = np.clip(predicted, 0.0, None)
    rmsle = float(np.sqrt(np.mean((np.log1p(rmsle_pred) - np.log1p(actual)) ** 2)))

    def tail_values(mask: np.ndarray) -> tuple[float, float]:
        return float(np.mean(absolute_error[mask])), float(np.mean(error[mask] < 0))

    top_decile_mae, top_decile_under = tail_values(top_decile)
    top_five_mae, top_five_under = tail_values(top_five_percent)
    negative_count = int(np.sum(predicted < 0))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "rmsle": rmsle,
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "mean_signed_error": float(np.mean(error)),
        "negative_prediction_count": negative_count,
        "negative_prediction_rate": float(negative_count / actual.size),
        "top_decile_mae": top_decile_mae,
        "top_five_percent_mae": top_five_mae,
        "top_decile_underprediction_rate": top_decile_under,
        "top_five_percent_underprediction_rate": top_five_under,
        "fit_time_seconds": float(fit_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
        "bundle_size_bytes": int(bundle_size_bytes),
    }
