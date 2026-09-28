"""Original-scale metrics and frozen selection rules for Prompt 3.

The functions in this module do not read data or write files.  They are kept
small so the modeling code and the independent tests use the same formulas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


VALID_TARGET_MODES = {"raw", "log1p"}
RELATIVE_TIE = 0.0025


def transform_target(y: Any, mode: str) -> np.ndarray:
    """Return a finite target in the requested scientific target mode."""
    values = np.asarray(y, dtype=np.float64).reshape(-1)
    if mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {mode}")
    if values.size == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Targets must be finite, non-negative, and non-empty.")
    return values.copy() if mode == "raw" else np.log1p(values)


def inverse_target(values: Any, mode: str) -> np.ndarray:
    """Return predictions on the original loan-amount scale."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if mode not in VALID_TARGET_MODES:
        raise ValueError(f"Unsupported target mode: {mode}")
    result = array.copy() if mode == "raw" else np.expm1(array)
    if not np.isfinite(result).all():
        raise ValueError("Inverse target values are not finite.")
    return result


def tail_membership(y_true: Any, quantile: float) -> np.ndarray:
    """Use the Validation target distribution to define an inclusive tail."""
    values = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Tail membership needs finite, non-empty targets.")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one.")
    return values >= float(np.quantile(values, quantile))


def compute_regression_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    fit_time_seconds: float = 0.0,
    prediction_time_seconds: float = 0.0,
    model_size_bytes: int = 0,
    bundle_size_bytes: int = 0,
) -> dict[str, float | int]:
    """Calculate every Prompt 3 metric on the original target scale."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("Targets and predictions must be aligned and non-empty.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite.")
    if (actual < 0).any():
        raise ValueError("RMSLE requires non-negative true targets.")

    error = predicted - actual
    absolute_error = np.abs(error)
    decile = tail_membership(actual, 0.90)
    five_percent = tail_membership(actual, 0.95)
    negative_count = int(np.count_nonzero(predicted < 0.0))
    rmsle_prediction = np.clip(predicted, 0.0, None)

    def tail_values(mask: np.ndarray) -> tuple[float, float, float]:
        return (
            float(np.mean(absolute_error[mask])),
            float(np.mean(error[mask])),
            float(np.mean(error[mask] < 0.0)),
        )

    decile_mae, decile_signed, decile_under = tail_values(decile)
    five_mae, five_signed, five_under = tail_values(five_percent)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "rmsle": float(
            np.sqrt(np.mean((np.log1p(rmsle_prediction) - np.log1p(actual)) ** 2))
        ),
        "median_absolute_error": float(np.median(absolute_error)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "mean_signed_error": float(np.mean(error)),
        "negative_prediction_count": negative_count,
        "negative_prediction_rate": float(negative_count / actual.size),
        "top_decile_mae": decile_mae,
        "top_five_percent_mae": five_mae,
        "top_decile_signed_error": decile_signed,
        "top_five_percent_signed_error": five_signed,
        "top_decile_underprediction_rate": decile_under,
        "top_five_percent_underprediction_rate": five_under,
        "fit_time_seconds": float(fit_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
        "model_size_bytes": int(model_size_bytes),
        "bundle_size_bytes": int(bundle_size_bytes),
    }


def _relative_pool(rows: list[dict[str, Any]], field: str, tolerance: float) -> list[dict[str, Any]]:
    best = min(float(row[field]) for row in rows)
    scale = max(abs(best), np.finfo(np.float64).eps)
    return [row for row in rows if (float(row[field]) - best) / scale <= tolerance]


def select_family_candidate(
    candidates: Sequence[Mapping[str, Any]], *, relative_tie: float = RELATIVE_TIE
) -> dict[str, Any]:
    """Apply the frozen Prompt 3 within-family selection rule.

    ``simplicity_rank`` is optional.  A lower value means simpler
    regularization. Candidate ID is only a deterministic last safeguard.
    """
    rows = [dict(row) for row in candidates]
    if not rows:
        raise ValueError("At least one Candidate is required.")
    required = {
        "candidate_id",
        "mae",
        "rmse",
        "top_decile_mae",
        "top_five_percent_mae",
        "fit_time_seconds",
        "bundle_size_bytes",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Candidate result is missing fields: {sorted(missing)}")

    pool = _relative_pool(rows, "mae", relative_tie)
    if len(pool) > 1:
        pool = _relative_pool(pool, "rmse", relative_tie)
    return min(
        pool,
        key=lambda row: (
            float(row["top_decile_mae"]),
            float(row["top_five_percent_mae"]),
            float(row["fit_time_seconds"]),
            int(row["bundle_size_bytes"]),
            int(row.get("simplicity_rank", 0)),
            str(row["candidate_id"]),
        ),
    )


def select_deep_anchor(
    representatives: Sequence[Mapping[str, Any]], *, relative_tie: float = RELATIVE_TIE
) -> dict[str, Any]:
    """Select the descriptive Deep anchor without discarding either family."""
    rows = [dict(row) for row in representatives]
    if len(rows) != 2 or {str(r.get("family")) for r in rows} != {
        "realmlp",
        "fttransformer",
    }:
        raise ValueError("Deep-anchor selection requires one representative from each family.")
    pool = _relative_pool(rows, "mae", relative_tie)
    if len(pool) > 1:
        pool = _relative_pool(pool, "rmse", relative_tie)
    return min(
        pool,
        key=lambda row: (
            float(row["top_decile_mae"]),
            float(row["top_five_percent_mae"]),
            float(row["fit_time_seconds"]),
            str(row["candidate_id"]),
        ),
    )


def paired_mae_bootstrap(
    y_true: Any,
    realmlp_prediction: Any,
    ft_prediction: Any,
    *,
    n_resamples: int = 300,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Compare aligned Deep predictions with a descriptive paired bootstrap."""
    actual = np.asarray(y_true, dtype=np.float64).reshape(-1)
    realmlp = np.asarray(realmlp_prediction, dtype=np.float64).reshape(-1)
    ft = np.asarray(ft_prediction, dtype=np.float64).reshape(-1)
    if actual.shape != realmlp.shape or actual.shape != ft.shape or actual.size == 0:
        raise ValueError("Paired bootstrap arrays must have the same non-zero length.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")
    if not np.isfinite(np.column_stack([actual, realmlp, ft])).all():
        raise ValueError("Paired bootstrap arrays must be finite.")

    realmlp_error = np.abs(realmlp - actual)
    ft_error = np.abs(ft - actual)
    rng = np.random.default_rng(random_state)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, actual.size, size=actual.size)
        differences[index] = float(np.mean(realmlp_error[sample] - ft_error[sample]))

    return {
        "n_rows": int(actual.size),
        "n_resamples": int(n_resamples),
        "random_state": int(random_state),
        "realmlp_minus_ft_mae_difference": float(np.mean(realmlp_error - ft_error)),
        "percentile_2_5": float(np.quantile(differences, 0.025)),
        "median": float(np.median(differences)),
        "percentile_97_5": float(np.quantile(differences, 0.975)),
        "realmlp_win_proportion": float(np.mean(differences < 0.0)),
        "ft_win_proportion": float(np.mean(differences > 0.0)),
        "tie_proportion": float(np.mean(differences == 0.0)),
    }
