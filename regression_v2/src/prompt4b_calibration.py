"""Calibration and routing utilities for Prompt 4B.

The functions in this module are deterministic and do not read files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


EPSILON = 1e-6


def clipped_logit(probability: Any, epsilon: float = EPSILON) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Probability values must be finite and non-empty.")
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def prior_correct_probability(probability: Any, positive_to_negative_weight_ratio: float) -> np.ndarray:
    """Invert binary class weighting from weighted to unweighted posterior odds."""
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    ratio = float(positive_to_negative_weight_ratio)
    if values.size == 0 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Probability values must be finite and within [0, 1].")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("The class-weight ratio must be finite and positive.")
    denominator = ratio * (1.0 - values) + values
    corrected = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
    return np.clip(corrected, 0.0, 1.0)


def expected_calibration_error(y_true: Any, probability: Any, bins: int = 10) -> float:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    if labels.shape != values.shape or labels.size == 0 or not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Calibration inputs must be aligned binary labels and probabilities.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(values, edges[1:-1], right=False), bins - 1)
    result = 0.0
    for bin_id in range(bins):
        mask = index == bin_id
        if np.any(mask):
            result += float(np.mean(mask)) * abs(float(np.mean(values[mask])) - float(np.mean(labels[mask])))
    return float(result)


def reliability_rows(y_true: Any, probability: Any, bins: int = 10) -> list[dict[str, Any]]:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(values, edges[1:-1], right=False), bins - 1)
    rows = []
    for bin_id in range(bins):
        mask = index == bin_id
        rows.append({
            "probability_bin": bin_id,
            "lower": float(edges[bin_id]),
            "upper": float(edges[bin_id + 1]),
            "row_count": int(np.count_nonzero(mask)),
            "mean_probability": float(np.mean(values[mask])) if np.any(mask) else np.nan,
            "observed_prevalence": float(np.mean(labels[mask])) if np.any(mask) else np.nan,
        })
    return rows


def calibration_summary(y_true: Any, probability: Any) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    return {
        "pr_auc": float(average_precision_score(labels, values)),
        "roc_auc": float(roc_auc_score(labels, values)),
        "brier_score": float(brier_score_loss(labels, values)),
        "calibration_error": expected_calibration_error(labels, values),
        "mean_probability": float(np.mean(values)),
        "observed_prevalence": float(np.mean(labels)),
    }


@dataclass
class PlattCalibrator:
    model: LogisticRegression
    fitted_row_hash_digest: str
    coefficient: float
    intercept: float
    seed: int = 42

    def predict(self, probability: Any) -> np.ndarray:
        logits = clipped_logit(probability).reshape(-1, 1)
        values = self.model.predict_proba(logits)[:, 1]
        if not np.isfinite(values).all():
            raise RuntimeError("Platt calibration produced non-finite probabilities.")
        return values


def fit_platt(probability: Any, y_true: Any, fitted_row_hash_digest: str) -> PlattCalibrator:
    labels = np.asarray(y_true, dtype=np.int8).reshape(-1)
    logits = clipped_logit(probability).reshape(-1, 1)
    if labels.size != logits.shape[0] or np.unique(labels).size != 2:
        raise ValueError("Platt calibration needs aligned Selection rows with both classes.")
    model = LogisticRegression(random_state=42, solver="lbfgs", max_iter=1000)
    model.fit(logits, labels)
    return PlattCalibrator(
        model=model,
        fitted_row_hash_digest=str(fitted_row_hash_digest),
        coefficient=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def thresholded_gate_strength(probability: Any, threshold: float, alpha: float) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    threshold = float(threshold)
    alpha = float(alpha)
    if not 0 <= threshold < 1 or not 0 <= alpha <= 1:
        raise ValueError("Threshold and alpha are outside their valid ranges.")
    strength = np.zeros_like(values)
    routed = values > threshold
    strength[routed] = alpha * (values[routed] - threshold) / (1.0 - threshold)
    return strength


def routed_prediction(global_prediction: Any, specialist_prediction: Any, probability: Any, threshold: float, alpha: float, cap: float | None = None) -> np.ndarray:
    global_values = np.asarray(global_prediction, dtype=np.float64).reshape(-1)
    specialist = np.asarray(specialist_prediction, dtype=np.float64).reshape(-1)
    strength = thresholded_gate_strength(probability, threshold, alpha)
    if not (global_values.shape == specialist.shape == strength.shape):
        raise ValueError("Routing inputs must be aligned.")
    correction = strength * (specialist - global_values)
    if cap is not None:
        cap_value = float(cap)
        if not np.isfinite(cap_value) or cap_value <= 0:
            raise ValueError("Correction cap must be finite and positive.")
        correction = np.clip(correction, -cap_value, cap_value)
    return global_values + correction


def residual_routed_prediction(global_prediction: Any, residual_prediction: Any, probability: Any, threshold: float, alpha: float, cap: float | None = None) -> np.ndarray:
    global_values = np.asarray(global_prediction, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual_prediction, dtype=np.float64).reshape(-1)
    strength = thresholded_gate_strength(probability, threshold, alpha)
    if not (global_values.shape == residual.shape == strength.shape):
        raise ValueError("Residual routing inputs must be aligned.")
    correction = strength * residual
    if cap is not None:
        correction = np.clip(correction, -float(cap), float(cap))
    return global_values + correction
