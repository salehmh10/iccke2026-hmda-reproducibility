"""Deterministic no-fit ensemble optimization for Prompt 4B."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def apply_weights(prediction_matrix: Any, weights: Any) -> np.ndarray:
    matrix = np.asarray(prediction_matrix, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != values.size:
        raise ValueError("Prediction matrix and weights are not aligned.")
    if np.any(values < -1e-10) or not np.isclose(np.sum(values), 1.0, atol=1e-7):
        raise ValueError("Ensemble weights must be non-negative and sum to one.")
    result = matrix @ values
    if not np.isfinite(result).all():
        raise RuntimeError("Ensemble prediction is not finite.")
    return result


def optimize_tail_weighted_mae(prediction_matrix: Any, y_true: Any, operational_tail: Any, lambda_tail: float) -> dict[str, Any]:
    matrix = np.asarray(prediction_matrix, dtype=np.float64)
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    tail = np.asarray(operational_tail, dtype=bool).reshape(-1)
    if matrix.shape[0] != target.size or tail.size != target.size:
        raise ValueError("Tail-weighted optimization inputs are not aligned.")
    sample_weight = np.where(tail, float(lambda_tail), 1.0)
    objective = lambda w: float(np.mean(sample_weight * np.abs(target - matrix @ w)))
    start = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    result = minimize(objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * matrix.shape[1], constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}], options={"ftol": 1e-10, "maxiter": 1000, "disp": False})
    weights = np.clip(result.x, 0.0, None)
    weights /= weights.sum()
    return {"status": "COMPLETE" if result.success else "SOLVER_WARNING", "weights": weights, "objective": objective(weights), "solver_message": str(result.message), "lambda_tail": float(lambda_tail)}


def optimize_body_constrained(prediction_matrix: Any, y_true: Any, operational_tail: Any, base_prediction: Any, tolerance: float = 0.0025) -> dict[str, Any]:
    """Minimize Tail MAE with deterministic large penalties for body/overall violations."""
    matrix = np.asarray(prediction_matrix, dtype=np.float64)
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    tail = np.asarray(operational_tail, dtype=bool).reshape(-1)
    base = np.asarray(base_prediction, dtype=np.float64).reshape(-1)
    body = ~tail
    base_mae = float(np.mean(np.abs(target - base)))
    base_body = float(np.mean(np.abs(target[body] - base[body])))
    max_mae = base_mae * (1.0 + tolerance)
    max_body = base_body * (1.0 + tolerance)
    penalty_scale = 10_000.0

    def components(weights):
        prediction = matrix @ weights
        mae = float(np.mean(np.abs(target - prediction)))
        body_mae = float(np.mean(np.abs(target[body] - prediction[body])))
        tail_mae = float(np.mean(np.abs(target[tail] - prediction[tail])))
        return tail_mae, mae, body_mae

    def objective(weights):
        tail_mae, mae, body_mae = components(weights)
        return tail_mae + penalty_scale * max(0.0, mae - max_mae) ** 2 + penalty_scale * max(0.0, body_mae - max_body) ** 2

    start = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    result = minimize(objective, start, method="SLSQP", bounds=[(0.0, 1.0)] * matrix.shape[1], constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}], options={"ftol": 1e-10, "maxiter": 1500, "disp": False})
    weights = np.clip(result.x, 0.0, None); weights /= weights.sum()
    tail_mae, mae, body_mae = components(weights)
    return {"status": "COMPLETE" if result.success else "SOLVER_WARNING", "weights": weights, "objective": objective(weights), "tail_mae": tail_mae, "mae": mae, "body_mae": body_mae, "max_mae": max_mae, "max_body_mae": max_body, "constraints_satisfied": bool(mae <= max_mae + 1e-6 and body_mae <= max_body + 1e-6), "penalty_scale": penalty_scale, "solver_message": str(result.message)}
