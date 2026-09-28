"""Metrics and deterministic routing formulas for Prompt 4B2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    from .prompt4_metrics import (
        compute_operational_tail_metrics,
        compute_regression_metrics,
        provisional_acceptance,
    )
except ImportError:
    from prompt4_metrics import (
        compute_operational_tail_metrics,
        compute_regression_metrics,
        provisional_acceptance,
    )


SIX_CONDITIONS = (
    "overall_mae_improves",
    "top_decile_mae_improves_3pct",
    "bottom_90_mae_within_0p25pct",
    "rmse_within_0p25pct",
    "top_decile_signed_error_closer_to_zero",
    "top_decile_underprediction_rate_decreases",
)


def gate_strength(probability: Any, threshold: float, alpha: float, gamma: float = 1.0) -> np.ndarray:
    """Return non-negative normalized Gate strength."""
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or not 0 <= threshold < 1 or alpha < 0 or gamma <= 0:
        raise ValueError("Gate inputs are invalid.")
    normalized = np.maximum(values - float(threshold), 0.0) / (1.0 - float(threshold))
    return float(alpha) * np.power(normalized, float(gamma))


def routed_residual(
    global_prediction: Any,
    residual_prediction: Any,
    strength: Any,
    *,
    lower_cap: float,
    upper_cap: float,
    positive_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a routed residual and return final prediction and correction."""
    global_values = np.asarray(global_prediction, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual_prediction, dtype=np.float64).reshape(-1)
    routed = np.asarray(strength, dtype=np.float64).reshape(-1)
    if global_values.shape != residual.shape or global_values.shape != routed.shape:
        raise ValueError("Routing inputs are not aligned.")
    if positive_only:
        residual = np.maximum(residual, 0.0)
    correction = np.clip(routed * residual, float(lower_cap), float(upper_cap))
    prediction = global_values + correction
    if not np.isfinite(prediction).all():
        raise ValueError("Routed prediction is not finite.")
    return prediction, correction


def direct_routing(
    global_prediction: Any,
    specialist_prediction: Any,
    strength: Any,
    cap: float,
) -> tuple[np.ndarray, np.ndarray]:
    global_values = np.asarray(global_prediction, dtype=np.float64).reshape(-1)
    specialist = np.asarray(specialist_prediction, dtype=np.float64).reshape(-1)
    return routed_residual(
        global_values,
        specialist - global_values,
        strength,
        lower_cap=-float(cap),
        upper_cap=float(cap),
    )


def scope_masks(roles: Any) -> tuple[tuple[str, np.ndarray], ...]:
    values = np.asarray(roles, dtype=object).reshape(-1)
    return (
        ("selection", values == "selection"),
        ("audit", values == "audit"),
        ("complete_validation", np.ones(values.size, dtype=bool)),
    )


def metric_rows(
    candidate_id: str,
    block: str,
    y_true: Any,
    prediction: Any,
    roles: Any,
    q90_train: float,
    **extra: Any,
) -> list[dict[str, Any]]:
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    role_values = np.asarray(roles, dtype=object).reshape(-1)
    if target.shape != pred.shape or target.shape != role_values.shape:
        raise ValueError("Metric inputs are not aligned.")
    rows: list[dict[str, Any]] = []
    for scope, mask in scope_masks(role_values):
        rows.append(
            {
                "candidate_id": candidate_id,
                "block": block,
                "scope": scope,
                **extra,
                **compute_regression_metrics(target[mask], pred[mask]),
                **compute_operational_tail_metrics(target[mask], pred[mask], q90_train),
            }
        )
    return rows


def reference_by_scope(rows: pd.DataFrame, candidate_id: str) -> dict[str, dict[str, Any]]:
    subset = rows.loc[rows["candidate_id"] == candidate_id]
    if set(subset["scope"]) != {"selection", "audit", "complete_validation"}:
        raise ValueError("Reference rows are incomplete.")
    return {str(row["scope"]): row.to_dict() for _, row in subset.iterrows()}


def acceptance_rows(candidate_rows: pd.DataFrame, reference: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in candidate_rows.iterrows():
        checks = provisional_acceptance(row.to_dict(), reference[str(row["scope"])])
        rows.append({"candidate_id": row["candidate_id"], "scope": row["scope"], **checks})
    return pd.DataFrame(rows)


def selection_rank(candidate_rows: pd.DataFrame, acceptance: pd.DataFrame) -> list[str]:
    """Apply the frozen Prompt 4B ranking and tie rule using Selection only."""
    selected = candidate_rows.loc[candidate_rows["scope"] == "selection"].merge(
        acceptance.loc[acceptance["scope"] == "selection", ["candidate_id", "conditions_passed"]],
        on="candidate_id",
        validate="one_to_one",
    )
    ordered = selected.sort_values(
        ["conditions_passed", "mae", "bottom_90_mae", "top_decile_mae", "candidate_id"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    return ordered["candidate_id"].astype(str).tolist()


def paired_bootstrap(
    y_true: Any,
    candidate: Any,
    reference: Any,
    top_decile_mask: Any,
    *,
    resamples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired descriptive bootstrap for overall and fixed-tail MAE differences."""
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    tail = np.asarray(top_decile_mask, dtype=bool).reshape(-1)
    if y.shape != cand.shape or y.shape != ref.shape or y.shape != tail.shape:
        raise ValueError("Bootstrap inputs are not aligned.")
    candidate_loss = np.abs(cand - y)
    reference_loss = np.abs(ref - y)
    rng = np.random.default_rng(seed)
    overall = np.empty(resamples, dtype=np.float64)
    top = np.empty(resamples, dtype=np.float64)
    tail_index = np.flatnonzero(tail)
    for index in range(resamples):
        draw = rng.integers(0, y.size, y.size)
        top_draw = tail_index[rng.integers(0, tail_index.size, tail_index.size)]
        overall[index] = np.mean(candidate_loss[draw] - reference_loss[draw])
        top[index] = np.mean(candidate_loss[top_draw] - reference_loss[top_draw])

    def summarize(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_difference": float(np.mean(values)),
            "median_difference": float(np.median(values)),
            "percentile_2_5": float(np.quantile(values, 0.025)),
            "percentile_97_5": float(np.quantile(values, 0.975)),
            "win_proportion": float(np.mean(values < 0)),
        }

    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "difference_definition": "candidate MAE minus reference MAE; negative favors candidate",
        "overall_mae_difference": summarize(overall),
        "fixed_complete_validation_top_decile_mae_difference": summarize(top),
        "adaptive_development_limitation": "Descriptive only; the interval does not remove adaptive-selection bias.",
    }
