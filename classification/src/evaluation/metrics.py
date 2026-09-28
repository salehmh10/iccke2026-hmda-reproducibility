"""Binary evaluation utilities with denial encoded as the positive class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdResult:
    """Validation-only threshold selection result."""

    threshold: float
    objective: float
    metrics: dict[str, Any]


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_binary(
    y_true: np.ndarray,
    denial_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return required discrimination, class, and calibration metrics.

    Label 0 is Approval and label 1 is Denial. Probabilities are clipped only
    for log-loss numerical stability; ranking and threshold metrics use the
    original finite values.
    """

    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    probability = np.asarray(denial_probability, dtype=np.float64).reshape(-1)
    if y.shape != probability.shape:
        raise ValueError("y_true and denial_probability must have equal shape")
    if not np.isfinite(probability).all():
        raise ValueError("denial_probability contains NaN or infinity")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y_true must contain only 0=Approval and 1=Denial")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")

    prediction = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    approval_recall = _safe_divide(tn, tn + fp)
    denial_recall = _safe_divide(tp, tp + fn)
    approval_precision = _safe_divide(tn, tn + fn)
    denial_precision = _safe_divide(tp, tp + fp)
    unique_predictions = int(np.unique(prediction).size)
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(y.size),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1_denial": float(f1_score(y, prediction, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "recall_denial": denial_recall,
        "recall_approval": approval_recall,
        "precision_denial": denial_precision,
        "precision_approval": approval_precision,
        "specificity": approval_recall,
        "predicted_positive_rate": float(prediction.mean()),
        "predicted_negative_rate": float(1.0 - prediction.mean()),
        "unique_prediction_classes": unique_predictions,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    result["degenerate"] = bool(
        unique_predictions < 2
        or min(approval_recall, denial_recall) < 0.01
        or (result["balanced_accuracy"] <= 0.501 and abs(result["mcc"]) < 0.01)
    )
    return result


def optimize_threshold(
    y_true: np.ndarray,
    denial_probability: np.ndarray,
    grid_size: int = 199,
) -> ThresholdResult:
    """Choose a threshold on validation data by MCC, then balanced accuracy.

    The scalar objective keeps MCC primary and adds a small balanced-accuracy
    tie-breaker. Candidate bounds avoid thresholds that trivially predict one
    class unless every model output is constant.
    """

    probability = np.asarray(denial_probability, dtype=np.float64).reshape(-1)
    if grid_size < 9:
        raise ValueError("grid_size must be at least 9")
    quantiles = np.linspace(0.005, 0.995, grid_size)
    candidates = np.unique(np.concatenate(([0.5], np.quantile(probability, quantiles))))
    candidates = candidates[(candidates > 0.0) & (candidates < 1.0)]
    if candidates.size == 0:
        metrics = evaluate_binary(y_true, probability, threshold=0.5)
        return ThresholdResult(0.5, float(metrics["mcc"]), metrics)

    best: ThresholdResult | None = None
    for threshold in candidates:
        y = np.asarray(y_true, dtype=np.int8).reshape(-1)
        prediction = probability >= threshold
        positive = y == 1
        negative = ~positive
        tp = int(np.count_nonzero(prediction & positive))
        fp = int(np.count_nonzero(prediction & negative))
        fn = int(np.count_nonzero(~prediction & positive))
        tn = int(np.count_nonzero(~prediction & negative))
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = _safe_divide(tp * tn - fp * fn, float(denominator))
        balanced = 0.5 * (_safe_divide(tp, tp + fn) + _safe_divide(tn, tn + fp))
        score = float(mcc + 1e-3 * balanced)
        # Full rank/calibration metrics are computed once after selection.
        candidate = ThresholdResult(float(threshold), score, {})
        if best is None or candidate.objective > best.objective:
            best = candidate
    assert best is not None
    return ThresholdResult(
        threshold=best.threshold,
        objective=best.objective,
        metrics=evaluate_binary(y_true, probability, threshold=best.threshold),
    )


def calibration_table(
    y_true: np.ndarray,
    denial_probability: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Return a serializable quantile-binned calibration curve."""

    observed, predicted = calibration_curve(
        np.asarray(y_true), np.asarray(denial_probability), n_bins=n_bins, strategy="quantile"
    )
    return [
        {"mean_predicted_probability": float(p), "observed_denial_rate": float(o)}
        for o, p in zip(observed, predicted, strict=True)
    ]
