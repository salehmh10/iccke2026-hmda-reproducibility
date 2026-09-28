"""Residual Tail Specialist utilities for Prompt 4B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from .prompt4b_crossfit import MetaPreprocessor
except ImportError:
    from prompt4b_crossfit import MetaPreprocessor


RESIDUAL_PARAMETERS = {
    "loss_function": "MAE",
    "iterations": 2000,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20,
    "random_strength": 1,
    "random_seed": 42,
    "thread_count": 4,
    "early_stopping_rounds": 100,
    "verbose": False,
}


def residual_target(y_true: Any, global_oof_prediction: Any) -> np.ndarray:
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    global_prediction = np.asarray(global_oof_prediction, dtype=np.float64).reshape(-1)
    if target.shape != global_prediction.shape or not np.isfinite(target).all() or not np.isfinite(global_prediction).all():
        raise ValueError("Residual target inputs must be finite and aligned.")
    return target - global_prediction


@dataclass
class ResidualSpecialistBundle:
    preprocessor: MetaPreprocessor
    model: Any
    metadata: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.model.predict(self.preprocessor.transform(frame)), dtype=np.float64).reshape(-1)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise RuntimeError("Residual Specialist produced invalid predictions.")
        return values


def fit_residual_model(X_fit: pd.DataFrame, y_fit: Any, base_features: list[str], *, X_stop: pd.DataFrame | None = None, y_stop: Any | None = None, parameters: dict[str, Any] | None = None):
    from catboost import CatBoostRegressor

    target = np.asarray(y_fit, dtype=np.float64).reshape(-1)
    if len(X_fit) != target.size or not np.isfinite(target).all():
        raise ValueError("Residual fit rows and targets are invalid.")
    config = dict(RESIDUAL_PARAMETERS if parameters is None else parameters)
    early_stopping = config.pop("early_stopping_rounds", None)
    preprocessor = MetaPreprocessor(base_features).fit(X_fit)
    model = CatBoostRegressor(**config, allow_writing_files=False, task_type="CPU")
    kwargs = {"cat_features": preprocessor.cat_feature_indices_, "verbose": False}
    use_stop = X_stop is not None
    if use_stop:
        stop_target = np.asarray(y_stop, dtype=np.float64).reshape(-1)
        if len(X_stop) != stop_target.size or not np.isfinite(stop_target).all():
            raise ValueError("Residual stop rows are invalid.")
        kwargs.update({"eval_set": (preprocessor.transform(X_stop), stop_target), "use_best_model": True})
        if early_stopping is not None:
            kwargs["early_stopping_rounds"] = int(early_stopping)
    elif early_stopping is not None:
        raise ValueError("A fixed residual refit must not use early stopping.")
    model.fit(preprocessor.transform(X_fit), target, **kwargs)
    if use_stop and int(model.get_best_iteration()) >= 0:
        selected = int(model.get_best_iteration()) + 1
    else:
        selected = int(model.tree_count_)
    return model, preprocessor, selected


def residual_diagnostics(y_true: Any, global_prediction: Any, predicted_residual: Any) -> dict[str, Any]:
    actual_residual = residual_target(y_true, global_prediction)
    prediction = np.asarray(predicted_residual, dtype=np.float64).reshape(-1)
    error = prediction - actual_residual
    correlation = float(np.corrcoef(prediction, actual_residual)[0, 1]) if prediction.size > 1 and np.std(prediction) > 0 and np.std(actual_residual) > 0 else np.nan
    return {
        "rows": int(prediction.size),
        "residual_mae": float(np.mean(np.abs(error))),
        "residual_rmse": float(np.sqrt(np.mean(error ** 2))),
        "residual_signed_error": float(np.mean(error)),
        "predicted_true_residual_correlation": correlation,
        "correct_correction_direction": float(np.mean(np.sign(prediction) == np.sign(actual_residual))),
        "p90_absolute_residual_error": float(np.quantile(np.abs(error), 0.90)),
    }
