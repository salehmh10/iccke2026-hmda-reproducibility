"""Pure metric and comparison functions for Prompt 4A.

This module has no file access and no model-fit code. All values are on the
original target scale, measured in thousands of U.S. dollars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RELATIVE_TIE = 0.0025


def _aligned(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Targets and predictions must be aligned and non-empty.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite.")
    if (actual < 0.0).any():
        raise ValueError("True targets must be non-negative.")
    return actual, predicted


def quantile_membership(y_true: Any, quantile: float) -> np.ndarray:
    """Return inclusive target-quantile membership within one evaluated set."""
    values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Quantile membership needs finite, non-empty targets.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between zero and one.")
    return values >= float(np.quantile(values, quantile))


def operational_tail_membership(y_true: Any, q90_train: float) -> np.ndarray:
    """Apply the frozen Train-derived operational Tail definition."""
    values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    threshold = float(q90_train)
    if values.size == 0 or not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Operational Tail inputs must be finite and non-empty.")
    return values > threshold


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    """Calculate all required Prompt 4A original-scale regression metrics."""
    actual, predicted = _aligned(y_true, y_pred)
    error = predicted - actual
    absolute = np.abs(error)
    top10 = quantile_membership(actual, 0.90)
    top05 = quantile_membership(actual, 0.95)
    bottom90 = ~top10
    q85 = float(np.quantile(actual, 0.85))
    q95 = float(np.quantile(actual, 0.95))
    boundary = (actual >= q85) & (actual <= q95)
    negative = predicted < 0.0

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(values[mask]))

    return {
        "n_rows": int(actual.size),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "rmsle": float(
            np.sqrt(np.mean((np.log1p(np.clip(predicted, 0.0, None)) - np.log1p(actual)) ** 2))
        ),
        "median_absolute_error": float(np.median(absolute)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "mean_signed_error": float(np.mean(error)),
        "negative_prediction_count": int(np.count_nonzero(negative)),
        "negative_prediction_rate": float(np.mean(negative)),
        "bottom_90_mae": masked_mean(absolute, bottom90),
        "top_decile_mae": masked_mean(absolute, top10),
        "top_five_percent_mae": masked_mean(absolute, top05),
        "top_decile_signed_error": masked_mean(error, top10),
        "top_five_percent_signed_error": masked_mean(error, top05),
        "top_decile_underprediction_rate": masked_mean(error < 0.0, top10),
        "top_five_percent_underprediction_rate": masked_mean(error < 0.0, top05),
        "p85_to_p95_boundary_mae": masked_mean(absolute, boundary),
        "top_decile_rows": int(np.count_nonzero(top10)),
        "top_five_percent_rows": int(np.count_nonzero(top05)),
        "boundary_rows": int(np.count_nonzero(boundary)),
    }


def compute_operational_tail_metrics(
    y_true: Any, y_pred: Any, q90_train: float
) -> dict[str, float | int]:
    """Report regression behavior on the frozen operational Tail and body."""
    actual, predicted = _aligned(y_true, y_pred)
    error = predicted - actual
    absolute = np.abs(error)
    tail = operational_tail_membership(actual, q90_train)
    body = ~tail

    def mean_or_nan(values: np.ndarray, mask: np.ndarray) -> float:
        return float(np.mean(values[mask])) if np.any(mask) else float("nan")

    return {
        "operational_tail_rows": int(np.count_nonzero(tail)),
        "operational_body_rows": int(np.count_nonzero(body)),
        "operational_tail_mae": mean_or_nan(absolute, tail),
        "operational_body_mae": mean_or_nan(absolute, body),
        "operational_tail_signed_error": mean_or_nan(error, tail),
        "operational_body_signed_error": mean_or_nan(error, body),
        "operational_tail_underprediction_rate": mean_or_nan(error < 0.0, tail),
        "operational_body_underprediction_rate": mean_or_nan(error < 0.0, body),
    }


def _relative_pool(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    best = min(float(row[field]) for row in rows)
    scale = max(abs(best), np.finfo(np.float64).eps)
    return [row for row in rows if (float(row[field]) - best) / scale <= RELATIVE_TIE]


def select_reference(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen MAE/RMSE/Tail/complexity preliminary ranking rule."""
    rows = [dict(row) for row in candidates if str(row.get("status", "COMPLETE")) != "UNAVAILABLE"]
    if not rows:
        raise ValueError("At least one available Candidate is required.")
    required = {"candidate_id", "mae", "rmse", "top_decile_mae", "top_five_percent_mae"}
    if any(required - row.keys() for row in rows):
        raise ValueError("Candidate results do not contain the frozen ranking metrics.")
    pool = _relative_pool(rows, "mae")
    if len(pool) > 1:
        pool = sorted(pool, key=lambda row: float(row["rmse"]))
        best_rmse = float(pool[0]["rmse"])
        scale = max(abs(best_rmse), np.finfo(np.float64).eps)
        pool = [row for row in pool if (float(row["rmse"]) - best_rmse) / scale <= RELATIVE_TIE]
    return min(
        pool,
        key=lambda row: (
            float(row["top_decile_mae"]),
            float(row["top_five_percent_mae"]),
            int(row.get("inference_complexity", 999)),
            str(row["candidate_id"]),
        ),
    )


def provisional_acceptance(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the six frozen diagnostic conditions against the own Global base."""
    checks = {
        "complete_mae_lower": float(candidate["mae"]) < float(base["mae"]),
        "top_decile_mae_improves_3pct": float(candidate["top_decile_mae"])
        <= float(base["top_decile_mae"]) * 0.97,
        "bottom_90_mae_worsens_at_most_0_25pct": float(candidate["bottom_90_mae"])
        <= float(base["bottom_90_mae"]) * 1.0025,
        "rmse_worsens_at_most_0_25pct": float(candidate["rmse"])
        <= float(base["rmse"]) * 1.0025,
        "top_decile_signed_error_closer_to_zero": abs(float(candidate["top_decile_signed_error"]))
        < abs(float(base["top_decile_signed_error"])),
        "top_decile_underprediction_rate_decreases": float(
            candidate["top_decile_underprediction_rate"]
        )
        < float(base["top_decile_underprediction_rate"]),
    }
    passed = int(sum(checks.values()))
    status = "PASS" if passed == len(checks) else "PARTIAL" if passed > 0 else "FAIL"
    return {**checks, "conditions_passed": passed, "conditions_total": len(checks), "provisional_acceptance_status": status}


def paired_mae_bootstrap(
    y_true: Any,
    reference_prediction: Any,
    candidate_prediction: Any,
    *,
    n_resamples: int = 300,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Return candidate-minus-reference MAE evidence from paired resamples."""
    actual, reference = _aligned(y_true, reference_prediction)
    candidate = np.asarray(candidate_prediction, dtype=np.float64).reshape(-1)
    if candidate.shape != actual.shape or not np.isfinite(candidate).all():
        raise ValueError("Candidate predictions must be finite and aligned.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    pointwise = np.abs(candidate - actual) - np.abs(reference - actual)
    rng = np.random.default_rng(random_state)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, actual.size, size=actual.size)
        differences[index] = float(np.mean(pointwise[sample]))
    return {
        "rows": int(actual.size),
        "n_resamples": int(n_resamples),
        "random_state": int(random_state),
        "mae_difference": float(np.mean(pointwise)),
        "percentile_2_5": float(np.quantile(differences, 0.025)),
        "median": float(np.median(differences)),
        "percentile_97_5": float(np.quantile(differences, 0.975)),
        "win_proportion": float(np.mean(differences < 0.0)),
    }
