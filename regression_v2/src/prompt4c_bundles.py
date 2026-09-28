"""Frozen Prompt 4C metrics and final inference bundles.

This module contains no data access and no fitting.  It keeps the final
prediction formulas small enough to test independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


GLOBAL_WEIGHTS = {"catboost": 0.60, "lightgbm": 0.20, "xgboost": 0.20}
FINAL_GATE_THRESHOLD = 0.75
FINAL_CORRECTION_ALPHA = 0.75


def select_feature_frame(frame: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Select the exact named contract and reject ambiguous input columns."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Final bundle prediction requires a pandas DataFrame.")
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].astype(str).tolist()
        raise ValueError(f"Duplicate input columns are not allowed: {duplicated}")
    names = list(feature_names)
    if len(names) != len(set(names)):
        raise ValueError("The frozen feature contract contains duplicate names.")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return frame.loc[:, names]


def stage3_gate_strength(
    probability: Any,
    threshold: float = FINAL_GATE_THRESHOLD,
    alpha: float = FINAL_CORRECTION_ALPHA,
) -> np.ndarray:
    """Recover the strict Prompt 4B normalized routing strength."""
    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    threshold = float(threshold)
    alpha = float(alpha)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Gate probabilities must be finite and within [0, 1].")
    if not 0.0 <= threshold < 1.0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("Gate threshold or alpha is invalid.")
    strength = np.zeros_like(values)
    routed = values > threshold
    strength[routed] = alpha * (values[routed] - threshold) / (1.0 - threshold)
    return strength


def stage3_prediction(
    global_prediction: Any,
    residual_prediction: Any,
    probability: Any,
    threshold: float = FINAL_GATE_THRESHOLD,
    alpha: float = FINAL_CORRECTION_ALPHA,
) -> np.ndarray:
    """Apply the uncapped, sign-preserving final Stage 3 formula."""
    global_values = np.asarray(global_prediction, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual_prediction, dtype=np.float64).reshape(-1)
    strength = stage3_gate_strength(probability, threshold, alpha)
    if not (global_values.shape == residual.shape == strength.shape):
        raise ValueError("Stage 3 prediction inputs are not aligned.")
    result = global_values + strength * residual
    if not np.isfinite(result).all():
        raise ValueError("Stage 3 prediction is not finite.")
    return result


def mape_details(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    """Calculate ordinary positive-target MAPE without an epsilon."""
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("MAPE inputs must be aligned and non-empty.")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("MAPE inputs must be finite.")
    valid = target > 0.0
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        value = np.nan
    else:
        value = float(100.0 * np.mean(np.abs(target[valid] - prediction[valid]) / target[valid]))
    return {
        "mape_percent": value,
        "mape_invalid_nonpositive_rows": int(target.size - valid_count),
        "mape_valid_rows": valid_count,
        "mape_valid_coverage": float(valid_count / target.size),
    }


def wape_percent(y_true: Any, y_pred: Any) -> float:
    """Calculate frozen WAPE: 100 * sum absolute error / sum target."""
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("WAPE inputs must be aligned and non-empty.")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("WAPE inputs must be finite.")
    denominator = float(np.sum(target))
    if denominator <= 0.0:
        raise ValueError("WAPE needs a strictly positive target sum.")
    return float(100.0 * np.sum(np.abs(target - prediction)) / denominator)


def duplicate_safe_target_deciles(y_true: Any) -> np.ndarray:
    """Use the existing project qcut convention for true-target deciles."""
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if target.size == 0 or not np.isfinite(target).all():
        raise ValueError("Target deciles need finite, non-empty targets.")
    labels = pd.qcut(pd.Series(target), 10, labels=False, duplicates="drop")
    result = labels.to_numpy(dtype=np.int16, na_value=-1)
    if (result < 0).any() or np.unique(result).size != 10:
        raise ValueError("The frozen reporting contract requires ten target deciles.")
    return result


def development_target_cutpoints(y_true: Any) -> dict[str, float]:
    """Freeze q10 through q90 using NumPy's project-standard linear quantile."""
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if target.size == 0 or not np.isfinite(target).all():
        raise ValueError("Target cutpoints need finite, non-empty targets.")
    levels = np.arange(0.1, 1.0, 0.1)
    values = np.quantile(target, levels, method="linear")
    return {f"q{int(level * 100):02d}": float(value) for level, value in zip(levels, values)}


@dataclass
class FinalGlobalBundle:
    """Frozen 60/20/20 full-Development Global comparator."""

    feature_names: list[str]
    components: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: dict(GLOBAL_WEIGHTS))

    def __post_init__(self) -> None:
        if self.weights != GLOBAL_WEIGHTS or set(self.components) != set(GLOBAL_WEIGHTS):
            raise ValueError("Final Global components or weights are not frozen correctly.")

    def predict_components(self, raw_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        selected = select_feature_frame(raw_frame, self.feature_names)
        result = {
            name: np.asarray(self.components[name].predict(selected), dtype=np.float64).reshape(-1)
            for name in GLOBAL_WEIGHTS
        }
        if any(values.shape != (len(selected),) or not np.isfinite(values).all() for values in result.values()):
            raise RuntimeError("A final Global component produced invalid predictions.")
        return result

    def predict(self, raw_frame: pd.DataFrame) -> np.ndarray:
        components = self.predict_components(raw_frame)
        result = sum(self.weights[name] * components[name] for name in GLOBAL_WEIGHTS)
        if result.shape != (len(raw_frame),) or not np.isfinite(result).all():
            raise RuntimeError("Final Global prediction is invalid.")
        return result


@dataclass
class FinalPrimaryBundle:
    """Complete features-only Stage 3 inference contract."""

    feature_names: list[str]
    global_bundle: FinalGlobalBundle
    meta_gate_bundle: Any
    residual_bundle: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    gate_threshold: float = FINAL_GATE_THRESHOLD
    correction_alpha: float = FINAL_CORRECTION_ALPHA

    def predict_details(self, raw_frame: pd.DataFrame) -> dict[str, np.ndarray]:
        selected = select_feature_frame(raw_frame, self.feature_names)
        global_prediction = self.global_bundle.predict(selected)
        meta_frame = selected.copy()
        meta_frame["global_prediction_feature"] = global_prediction
        probability = np.asarray(
            self.meta_gate_bundle.predict_tail_probability(meta_frame), dtype=np.float64
        ).reshape(-1)
        residual = np.asarray(self.residual_bundle.predict(meta_frame), dtype=np.float64).reshape(-1)
        strength = stage3_gate_strength(probability, self.gate_threshold, self.correction_alpha)
        prediction = stage3_prediction(
            global_prediction,
            residual,
            probability,
            self.gate_threshold,
            self.correction_alpha,
        )
        return {
            "global_prediction": global_prediction,
            "meta_gate_probability": probability,
            "residual_prediction": residual,
            "routing_strength": strength,
            "prediction": prediction,
        }

    def predict(self, raw_frame: pd.DataFrame) -> np.ndarray:
        return self.predict_details(raw_frame)["prediction"]


def load_final_bundle(path: str | Path) -> FinalGlobalBundle | FinalPrimaryBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, (FinalGlobalBundle, FinalPrimaryBundle)):
        raise TypeError(f"Unexpected final bundle type: {type(bundle).__name__}")
    return bundle

