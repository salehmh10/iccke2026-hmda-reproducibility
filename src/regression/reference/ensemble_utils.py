"""Pure deterministic ensemble helpers for Regression V2 Prompt 4A.

This module performs no file access and no model fitting.  It operates only on
already aligned Validation predictions supplied by its caller.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


SEED = 42
EXPECTED_VALIDATION_ROWS = 100_000
VALIDATION_SELECTION_ROWS = 70_000
VALIDATION_AUDIT_ROWS = 30_000
MODEL_PREDICTION_COLUMN = "y_pred"
EXPECTED_PREDICTION_COLUMNS = (
    "row_hash",
    "y_true",
    MODEL_PREDICTION_COLUMN,
    "model_id",
    "family",
    "feature_contract",
    "target_mode",
)
HARD_ROUTING_THRESHOLDS = (0.35, 0.50, 0.65)
SOFT_MIXTURE_ALPHAS = (0.50, 0.75, 1.00)


NO_FIT_ENSEMBLE_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "ens_boost_equal",
        "candidate_type": "fixed",
        "members": ("catboost", "lightgbm", "xgboost"),
        "weights": {"catboost": 1.0 / 3.0, "lightgbm": 1.0 / 3.0, "xgboost": 1.0 / 3.0},
    },
    {
        "candidate_id": "ens_boost_cat050",
        "candidate_type": "fixed",
        "members": ("catboost", "lightgbm", "xgboost"),
        "weights": {"catboost": 0.50, "lightgbm": 0.25, "xgboost": 0.25},
    },
    {
        "candidate_id": "ens_boost_cat060",
        "candidate_type": "fixed",
        "members": ("catboost", "lightgbm", "xgboost"),
        "weights": {"catboost": 0.60, "lightgbm": 0.20, "xgboost": 0.20},
    },
    {
        "candidate_id": "ens_cat_ft_equal",
        "candidate_type": "fixed",
        "members": ("catboost", "fttransformer"),
        "weights": {"catboost": 0.50, "fttransformer": 0.50},
    },
    {
        "candidate_id": "ens_cat_realmlp_equal",
        "candidate_type": "fixed",
        "members": ("catboost", "realmlp"),
        "weights": {"catboost": 0.50, "realmlp": 0.50},
    },
    {
        "candidate_id": "ens_cat_lgb_ft",
        "candidate_type": "fixed",
        "members": ("catboost", "lightgbm", "fttransformer"),
        "weights": {"catboost": 0.40, "lightgbm": 0.20, "fttransformer": 0.40},
    },
    {
        "candidate_id": "ens_cat_ft_realmlp",
        "candidate_type": "fixed",
        "members": ("catboost", "fttransformer", "realmlp"),
        "weights": {"catboost": 0.40, "fttransformer": 0.40, "realmlp": 0.20},
    },
    {
        "candidate_id": "ens_boost_ft_equal",
        "candidate_type": "fixed",
        "members": ("catboost", "lightgbm", "xgboost", "fttransformer"),
        "weights": {"catboost": 0.25, "lightgbm": 0.25, "xgboost": 0.25, "fttransformer": 0.25},
    },
    {
        "candidate_id": "ens_convex_boosting",
        "candidate_type": "convex",
        "members": ("catboost", "lightgbm", "xgboost"),
        "weights": None,
    },
    {
        "candidate_id": "ens_convex_boosting_deep",
        "candidate_type": "convex",
        "members": ("catboost", "lightgbm", "xgboost", "realmlp", "fttransformer"),
        "weights": None,
    },
)

# Stable orchestration-facing names.  The mapping form keeps the frozen order
# and uses positional weights so it can operate directly on a NumPy matrix.
HARD_THRESHOLDS = HARD_ROUTING_THRESHOLDS
ALPHA_GRID = SOFT_MIXTURE_ALPHAS
NO_FIT_ENSEMBLES: dict[str, dict[str, Any]] = {
    item["candidate_id"]: {
        "kind": item["candidate_type"],
        "members": list(item["members"]),
        "weights": (
            [float(item["weights"][name]) for name in item["members"]]
            if item["weights"] is not None
            else None
        ),
    }
    for item in NO_FIT_ENSEMBLE_DEFINITIONS
}


def _float_vector(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite, non-empty vector.")
    return result


def _bool_vector(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    if result.dtype != np.bool_:
        unique = set(pd.Series(result).dropna().tolist())
        if not unique.issubset({False, True, 0, 1}) or pd.isna(result).any():
            raise ValueError(f"{name} must contain only Boolean values.")
    return result.astype(bool, copy=False)


def _require_same_length(named: Mapping[str, np.ndarray]) -> int:
    lengths = {name: int(values.size) for name, values in named.items()}
    if not lengths or len(set(lengths.values())) != 1:
        raise ValueError(f"All supplied vectors must have identical lengths: {lengths}")
    return next(iter(lengths.values()))


def _ordered_digest(values: Iterable[Any]) -> str:
    text = "\n".join(pd.Series(values, copy=False).astype(str).tolist())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_no_fit_ensemble_definitions() -> list[dict[str, Any]]:
    """Return independent copies of the exact ten frozen ensemble definitions."""
    definitions = deepcopy(list(NO_FIT_ENSEMBLE_DEFINITIONS))
    if len(definitions) != 10 or len({item["candidate_id"] for item in definitions}) != 10:
        raise AssertionError("Prompt 4A must define exactly ten unique no-fit ensembles.")
    for item in definitions:
        if tuple(item["members"]) != tuple(dict.fromkeys(item["members"])):
            raise AssertionError(f"Duplicate ensemble member: {item['candidate_id']}")
        if item["candidate_type"] == "fixed":
            weights = item["weights"]
            if set(weights) != set(item["members"]):
                raise AssertionError(f"Fixed weights do not match members: {item['candidate_id']}")
            if any(float(weight) < 0.0 for weight in weights.values()):
                raise AssertionError(f"Negative frozen weight: {item['candidate_id']}")
            if not np.isclose(sum(map(float, weights.values())), 1.0, atol=1e-12, rtol=0.0):
                raise AssertionError(f"Frozen weights do not sum to one: {item['candidate_id']}")
        elif item["candidate_type"] != "convex" or item["weights"] is not None:
            raise AssertionError(f"Unknown no-fit ensemble definition: {item}")
    return definitions


def duplicate_safe_deciles(y_true: Any) -> np.ndarray:
    """Create deterministic target-decile labels while allowing tied edges."""
    actual = _float_vector(y_true, name="y_true")
    labels = pd.qcut(pd.Series(actual), q=10, labels=False, duplicates="drop")
    result = labels.to_numpy(dtype=np.int16, na_value=-1)
    if (result < 0).any() or np.unique(result).size < 2:
        raise ValueError("Target values cannot support duplicate-safe decile stratification.")
    return result


def make_validation_selection_audit_split(
    y_true: Any,
    row_hashes: Iterable[Any],
    *,
    selection_rows: int = VALIDATION_SELECTION_ROWS,
    audit_rows: int = VALIDATION_AUDIT_ROWS,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the frozen 70k/30k split while preserving source row order.

    The first two arrays are sorted source positions.  The third is a role
    vector aligned to the original Validation order.
    """
    actual = _float_vector(y_true, name="y_true")
    hashes = pd.Series(row_hashes, copy=False)
    if hashes.isna().any():
        raise ValueError("Validation row_hash cannot contain missing values.")
    hash_values = hashes.astype(str).to_numpy()
    if hash_values.size != actual.size:
        raise ValueError("Validation targets and row hashes must have identical lengths.")
    if pd.Series(hash_values).duplicated().any():
        raise ValueError("Validation row_hash must be unique.")
    if selection_rows <= 0 or audit_rows <= 0 or selection_rows + audit_rows != actual.size:
        raise ValueError("Selection and audit counts must be positive and cover Validation exactly.")

    strata = duplicate_safe_deciles(actual)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=selection_rows,
        test_size=audit_rows,
        random_state=random_state,
    )
    selection_index, audit_index = next(
        splitter.split(np.zeros(actual.size, dtype=np.int8), strata)
    )
    selection_index = np.sort(selection_index.astype(np.int64, copy=False))
    audit_index = np.sort(audit_index.astype(np.int64, copy=False))
    if selection_index.size != selection_rows or audit_index.size != audit_rows:
        raise AssertionError("Validation selection/audit row counts are incorrect.")
    if np.intersect1d(selection_index, audit_index).size:
        raise AssertionError("Validation selection and audit positions overlap.")

    roles = np.full(actual.size, "", dtype=object)
    roles[selection_index] = "selection"
    roles[audit_index] = "audit"
    if set(roles.tolist()) != {"selection", "audit"}:
        raise AssertionError("Validation selection/audit roles are incomplete.")

    report = {
        "status": "PASS",
        "random_state": int(random_state),
        "stratification": "duplicate-safe target deciles",
        "source_validation_rows": int(actual.size),
        "selection_rows": int(selection_index.size),
        "audit_rows": int(audit_index.size),
        "selection_audit_overlap": 0,
        "source_validation_row_hash_digest": _ordered_digest(hash_values),
        "selection_row_hash_digest": _ordered_digest(hash_values[selection_index]),
        "audit_row_hash_digest": _ordered_digest(hash_values[audit_index]),
        "decile_label_count": int(np.unique(strata).size),
    }
    return selection_index, audit_index, roles, report


def deterministic_validation_split(
    y_true: Any,
    row_hashes: Iterable[Any],
    *,
    selection_rows: int = VALIDATION_SELECTION_ROWS,
    audit_rows: int = VALIDATION_AUDIT_ROWS,
    random_state: int = SEED,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Orchestration wrapper for the frozen deterministic Validation split."""
    selection, audit, _, evidence = make_validation_selection_audit_split(
        y_true,
        row_hashes,
        selection_rows=selection_rows,
        audit_rows=audit_rows,
        random_state=random_state,
    )
    return selection, audit, evidence


def validate_prediction_alignment(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    expected_rows: int = EXPECTED_VALIDATION_ROWS,
    expected_row_hash: Iterable[Any] | None = None,
    expected_y_true: Any | None = None,
) -> dict[str, Any]:
    """Validate exact row order, targets, logical schema, and finite predictions."""
    if not prediction_frames:
        raise ValueError("At least one prediction frame is required.")
    if len(prediction_frames) != len(set(prediction_frames)):
        raise ValueError("Prediction source names must be unique.")

    reference_name = next(iter(prediction_frames))
    reference = prediction_frames[reference_name]
    if not isinstance(reference, pd.DataFrame):
        raise TypeError(f"Prediction source {reference_name!r} is not a DataFrame.")
    reference_hash: np.ndarray | None = None
    reference_target: np.ndarray | None = None
    sources: list[dict[str, Any]] = []

    for name, frame in prediction_frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"Prediction source {name!r} is not a DataFrame.")
        if tuple(frame.columns) != EXPECTED_PREDICTION_COLUMNS:
            raise ValueError(
                f"Prediction source {name!r} has the wrong logical schema: {list(frame.columns)}"
            )
        if len(frame) != expected_rows:
            raise ValueError(
                f"Prediction source {name!r} has {len(frame):,} rows; expected {expected_rows:,}."
            )
        if frame["row_hash"].isna().any() or frame["row_hash"].duplicated().any():
            raise ValueError(f"Prediction source {name!r} has missing or duplicate row_hash values.")
        row_hash = frame["row_hash"].astype(str).to_numpy()
        y_true = _float_vector(frame["y_true"], name=f"{name}.y_true")
        y_pred = _float_vector(frame[MODEL_PREDICTION_COLUMN], name=f"{name}.y_pred")
        _require_same_length({"row_hash": row_hash, "y_true": y_true, "y_pred": y_pred})
        for metadata_column in EXPECTED_PREDICTION_COLUMNS[3:]:
            if frame[metadata_column].isna().any() or frame[metadata_column].nunique() != 1:
                raise ValueError(
                    f"Prediction source {name!r} must have one non-missing {metadata_column} value."
                )

        if reference_hash is None:
            reference_hash = row_hash
            reference_target = y_true
        elif not np.array_equal(reference_hash, row_hash):
            raise ValueError(f"Prediction source {name!r} has different row_hash order.")
        elif not np.array_equal(reference_target, y_true):
            raise ValueError(f"Prediction source {name!r} has different y_true values.")

        sources.append(
            {
                "source": str(name),
                "rows": int(len(frame)),
                "row_hash_unique": True,
                "row_order_equal": True,
                "target_equal": True,
                "finite_predictions": True,
                "row_hash_digest": _ordered_digest(row_hash),
                "model_id": str(frame["model_id"].iloc[0]),
                "family": str(frame["family"].iloc[0]),
                "feature_contract": str(frame["feature_contract"].iloc[0]),
                "target_mode": str(frame["target_mode"].iloc[0]),
            }
        )

    assert reference_hash is not None and reference_target is not None
    if expected_row_hash is not None:
        expected_hash = pd.Series(expected_row_hash, copy=False)
        if expected_hash.isna().any() or not np.array_equal(
            reference_hash, expected_hash.astype(str).to_numpy()
        ):
            raise ValueError("Saved predictions do not match the expected Validation row order.")
    if expected_y_true is not None and not np.array_equal(
        reference_target, _float_vector(expected_y_true, name="expected_y_true")
    ):
        raise ValueError("Saved prediction targets do not match expected Validation targets.")

    return {
        "status": "PASS",
        "prediction_source_count": int(len(sources)),
        "rows_per_source": int(expected_rows),
        "validation_row_hash_digest": _ordered_digest(reference_hash),
        "exact_row_order": True,
        "exact_target_equality": True,
        "all_predictions_finite": True,
        "logical_schema": list(EXPECTED_PREDICTION_COLUMNS),
        "sources": sources,
    }


def build_aligned_prediction_table(
    prediction_frames: Mapping[str, pd.DataFrame],
    *,
    selection_or_audit_role: Sequence[str] | None = None,
    expected_rows: int = EXPECTED_VALIDATION_ROWS,
    expected_row_hash: Iterable[Any] | None = None,
    expected_y_true: Any | None = None,
) -> pd.DataFrame:
    """Create one aligned table without modifying any supplied source frame."""
    validate_prediction_alignment(
        prediction_frames,
        expected_rows=expected_rows,
        expected_row_hash=expected_row_hash,
        expected_y_true=expected_y_true,
    )
    first = next(iter(prediction_frames.values()))
    aligned = pd.DataFrame(
        {
            "row_hash": first["row_hash"].astype(str).to_numpy(copy=True),
            "y_true": first["y_true"].to_numpy(dtype=np.float64, copy=True),
        }
    )
    for name, frame in prediction_frames.items():
        if name in aligned.columns:
            raise ValueError(f"Prediction source name conflicts with an aligned-table column: {name}")
        aligned[str(name)] = frame[MODEL_PREDICTION_COLUMN].to_numpy(dtype=np.float64, copy=True)
    if selection_or_audit_role is not None:
        roles = np.asarray(selection_or_audit_role, dtype=object).reshape(-1)
        if roles.size != len(aligned) or not set(roles.tolist()).issubset({"selection", "audit"}):
            raise ValueError("selection_or_audit_role must align and contain only selection/audit.")
        aligned["selection_or_audit_role"] = roles.copy()
    return aligned


def weighted_ensemble_prediction(
    predictions: Mapping[str, Any],
    weights: Mapping[str, float],
    *,
    require_nonnegative: bool = True,
    sum_tolerance: float = 1e-10,
) -> np.ndarray:
    """Apply an exact original-scale weighted ensemble formula."""
    if not weights:
        raise ValueError("At least one ensemble weight is required.")
    missing = [name for name in weights if name not in predictions]
    if missing:
        raise ValueError(f"Missing component predictions: {missing}")
    numeric_weights = {name: float(weight) for name, weight in weights.items()}
    if not np.isfinite(list(numeric_weights.values())).all():
        raise ValueError("Ensemble weights must be finite.")
    if require_nonnegative and any(weight < 0.0 for weight in numeric_weights.values()):
        raise ValueError("Ensemble weights must be non-negative.")
    if not np.isclose(sum(numeric_weights.values()), 1.0, atol=sum_tolerance, rtol=0.0):
        raise ValueError("Ensemble weights must sum to one.")

    vectors = {
        name: _float_vector(predictions[name], name=f"predictions[{name!r}]")
        for name in numeric_weights
    }
    _require_same_length(vectors)
    result = np.zeros(next(iter(vectors.values())).shape, dtype=np.float64)
    for name, weight in numeric_weights.items():
        result += weight * vectors[name]
    if not np.isfinite(result).all():
        raise ValueError("Weighted ensemble produced non-finite predictions.")
    return result


def apply_ensemble(prediction_matrix: Any, weights: Any) -> np.ndarray:
    """Apply nonnegative sum-one weights to an ``n_rows x n_models`` matrix."""
    matrix = np.asarray(prediction_matrix, dtype=np.float64)
    weight_vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("prediction_matrix must be a non-empty two-dimensional array.")
    if not np.isfinite(matrix).all() or not np.isfinite(weight_vector).all():
        raise ValueError("Predictions and weights must be finite.")
    if matrix.shape[1] != weight_vector.size:
        raise ValueError("One weight is required for every prediction column.")
    if (weight_vector < 0.0).any():
        raise ValueError("Ensemble weights must be non-negative.")
    if not np.isclose(weight_vector.sum(), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("Ensemble weights must sum to one.")
    result = matrix @ weight_vector
    if not np.isfinite(result).all():
        raise ValueError("Weighted ensemble produced non-finite predictions.")
    return result


def optimize_convex_mae_weights(
    y_true: Any,
    predictions: Mapping[str, Any],
    *,
    model_order: Sequence[str] | None = None,
    maxiter: int = 1_000,
    ftol: float = 1e-12,
) -> dict[str, Any]:
    """Run one deterministic equal-start SLSQP fit for nonnegative MAE weights.

    Any solver error or invalid result returns ``status='UNAVAILABLE'``.  No
    alternative optimizer, random restart, or fallback ensemble is used.
    """
    names = list(model_order or predictions.keys())
    base = {
        "solver_method": "SLSQP",
        "solver_start_count": 1,
        "initialization": "equal_weight",
        "objective": "selection_mae",
        "model_order": names,
    }
    try:
        if len(names) < 2 or len(names) != len(set(names)):
            raise ValueError("Convex optimization requires at least two unique models.")
        if set(names) != set(predictions):
            raise ValueError("model_order must contain each supplied prediction exactly once.")
        actual = _float_vector(y_true, name="y_true")
        vectors = {
            name: _float_vector(predictions[name], name=f"predictions[{name!r}]")
            for name in names
        }
        _require_same_length({"y_true": actual, **vectors})
        matrix = np.column_stack([vectors[name] for name in names])
        initial = np.full(len(names), 1.0 / len(names), dtype=np.float64)

        from scipy.optimize import minimize

        def objective(weight_vector: np.ndarray) -> float:
            return float(np.mean(np.abs(matrix @ weight_vector - actual)))

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * len(names),
            constraints=(
                {"type": "eq", "fun": lambda weight_vector: float(np.sum(weight_vector) - 1.0)},
            ),
            options={"maxiter": int(maxiter), "ftol": float(ftol), "disp": False},
        )
        weights = np.asarray(result.x, dtype=np.float64).reshape(-1)
        valid = (
            bool(result.success)
            and weights.size == len(names)
            and np.isfinite(weights).all()
            and bool((weights >= 0.0).all())
            and bool((weights <= 1.0).all())
            and bool(np.isclose(weights.sum(), 1.0, atol=1e-10, rtol=0.0))
        )
        if not valid:
            return {
                **base,
                "status": "UNAVAILABLE",
                "weights": None,
                "sum_weights": None,
                "selection_mae": None,
                "solver_success": bool(result.success),
                "solver_status": int(result.status),
                "solver_message": str(result.message),
                "solver_iterations": int(getattr(result, "nit", 0)),
            }

        weight_map = {name: float(weights[index]) for index, name in enumerate(names)}
        prediction = weighted_ensemble_prediction(vectors, weight_map, sum_tolerance=1e-10)
        return {
            **base,
            "status": "COMPLETE",
            "weights": weight_map,
            "sum_weights": float(weights.sum()),
            "selection_mae": float(np.mean(np.abs(prediction - actual))),
            "solver_success": True,
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "solver_iterations": int(getattr(result, "nit", 0)),
        }
    except Exception as error:  # A failed convex Candidate must remain unavailable.
        return {
            **base,
            "status": "UNAVAILABLE",
            "weights": None,
            "sum_weights": None,
            "selection_mae": None,
            "solver_success": False,
            "solver_status": None,
            "solver_message": f"{type(error).__name__}: {error}",
            "solver_iterations": 0,
        }


def optimize_convex_mae(
    prediction_matrix: Any,
    y_true: Any,
    *,
    maxiter: int = 1_000,
    ftol: float = 1e-12,
) -> dict[str, Any]:
    """Matrix-oriented SLSQP wrapper used by the Prompt 4A orchestrator."""
    try:
        matrix = np.asarray(prediction_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            raise ValueError("Convex optimization needs an n_rows x n_models matrix with at least two models.")
        names = [f"model_{index}" for index in range(matrix.shape[1])]
        result = optimize_convex_mae_weights(
            y_true,
            {name: matrix[:, index] for index, name in enumerate(names)},
            model_order=names,
            maxiter=maxiter,
            ftol=ftol,
        )
        if result["status"] != "COMPLETE":
            return {
                **result,
                "status": "UNAVAILABLE",
                "weights": None,
                "solver_starts": 1,
                "objective": None,
            }
        weights = [float(result["weights"][name]) for name in names]
        return {
            **result,
            "status": "COMPLETE",
            "weights": weights,
            "solver_starts": 1,
            "objective": float(result["selection_mae"]),
        }
    except Exception as error:
        return {
            "status": "UNAVAILABLE",
            "weights": None,
            "solver_method": "SLSQP",
            "solver_starts": 1,
            "solver_status": None,
            "solver_message": f"{type(error).__name__}: {error}",
            "objective": None,
        }


def hard_routing_prediction(
    global_prediction: Any,
    tail_specialist_prediction: Any,
    p_tail: Any,
    threshold: float,
) -> np.ndarray:
    """Use the Specialist exactly where ``p_tail >= threshold``."""
    threshold = float(threshold)
    if threshold not in HARD_ROUTING_THRESHOLDS:
        raise ValueError(f"threshold must be one of {HARD_ROUTING_THRESHOLDS}.")
    global_values = _float_vector(global_prediction, name="global_prediction")
    specialist = _float_vector(tail_specialist_prediction, name="tail_specialist_prediction")
    probabilities = _float_vector(p_tail, name="p_tail")
    _require_same_length(
        {"global_prediction": global_values, "tail_specialist_prediction": specialist, "p_tail": probabilities}
    )
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("p_tail must lie in [0, 1].")
    return np.where(probabilities >= threshold, specialist, global_values)


def hard_route(
    global_prediction: Any,
    tail_specialist_prediction: Any,
    p_tail: Any,
    threshold: float,
) -> np.ndarray:
    """Compatibility name for :func:`hard_routing_prediction`."""
    return hard_routing_prediction(
        global_prediction, tail_specialist_prediction, p_tail, threshold
    )


def soft_mixture_prediction(
    global_prediction: Any,
    tail_specialist_prediction: Any,
    p_tail: Any,
    alpha: float,
) -> np.ndarray:
    """Apply ``global + alpha * p_tail * (specialist - global)`` exactly."""
    alpha = float(alpha)
    if alpha not in SOFT_MIXTURE_ALPHAS:
        raise ValueError(f"alpha must be one of {SOFT_MIXTURE_ALPHAS}.")
    global_values = _float_vector(global_prediction, name="global_prediction")
    specialist = _float_vector(tail_specialist_prediction, name="tail_specialist_prediction")
    probabilities = _float_vector(p_tail, name="p_tail")
    _require_same_length(
        {"global_prediction": global_values, "tail_specialist_prediction": specialist, "p_tail": probabilities}
    )
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("p_tail must lie in [0, 1].")
    result = global_values + alpha * probabilities * (specialist - global_values)
    if not np.isfinite(result).all():
        raise ValueError("Soft mixture produced non-finite predictions.")
    return result


def soft_mix(
    global_prediction: Any,
    tail_specialist_prediction: Any,
    p_tail: Any,
    alpha: float,
) -> np.ndarray:
    """Compatibility name for :func:`soft_mixture_prediction`."""
    return soft_mixture_prediction(
        global_prediction, tail_specialist_prediction, p_tail, alpha
    )


def prediction_diversity_matrices(predictions: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Return full Pearson, Spearman, and mean-absolute-difference matrices."""
    if len(predictions) < 2:
        raise ValueError("Prediction diversity needs at least two models.")
    names = list(predictions)
    if len(names) != len(set(names)):
        raise ValueError("Prediction names must be unique.")
    vectors = {
        name: _float_vector(predictions[name], name=f"predictions[{name!r}]") for name in names
    }
    _require_same_length(vectors)
    frame = pd.DataFrame(vectors, columns=names)
    pearson = frame.corr(method="pearson")
    spearman = frame.corr(method="spearman")
    mean_difference = pd.DataFrame(index=names, columns=names, dtype=np.float64)
    for left in names:
        for right in names:
            mean_difference.loc[left, right] = float(np.mean(np.abs(vectors[left] - vectors[right])))
    for matrix in (pearson, spearman, mean_difference):
        matrix.index.name = "model"
        matrix.columns.name = "model"
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mean_absolute_prediction_difference": mean_difference,
    }


def prediction_correlation_report(predictions: Mapping[str, Any]) -> pd.DataFrame:
    """Return a deterministic long-form full diversity report."""
    matrices = prediction_diversity_matrices(predictions)
    names = list(predictions)
    rows = []
    for left in names:
        for right in names:
            rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "pearson": float(matrices["pearson"].loc[left, right]),
                    "spearman": float(matrices["spearman"].loc[left, right]),
                    "mean_absolute_prediction_difference": float(
                        matrices["mean_absolute_prediction_difference"].loc[left, right]
                    ),
                }
            )
    return pd.DataFrame(rows)


def correlation_report(predictions: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Return three long-form diversity metrics for every ordered model pair."""
    source = (
        {str(column): predictions[column].to_numpy(dtype=np.float64) for column in predictions.columns}
        if isinstance(predictions, pd.DataFrame)
        else predictions
    )
    wide = prediction_correlation_report(source)
    metric_columns = (
        ("pearson", "pearson_correlation"),
        ("spearman", "spearman_correlation"),
        ("mean_absolute_prediction_difference", "mean_absolute_difference"),
    )
    rows: list[dict[str, Any]] = []
    for record in wide.to_dict("records"):
        for column, metric in metric_columns:
            rows.append(
                {
                    "model_a": record["model_a"],
                    "model_b": record["model_b"],
                    "metric": metric,
                    "value": float(record[column]),
                }
            )
    return pd.DataFrame(rows, columns=["model_a", "model_b", "metric", "value"])


def routing_summary(
    p_tail: Any,
    operational_tail_true: Any,
    threshold: float,
) -> dict[str, float | int]:
    """Summarize hard routing; false routing rate is the body-row false-positive rate."""
    threshold = float(threshold)
    if threshold not in HARD_ROUTING_THRESHOLDS:
        raise ValueError(f"threshold must be one of {HARD_ROUTING_THRESHOLDS}.")
    probabilities = _float_vector(p_tail, name="p_tail")
    truth = _bool_vector(operational_tail_true, name="operational_tail_true")
    _require_same_length({"p_tail": probabilities, "operational_tail_true": truth})
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("p_tail must lie in [0, 1].")
    routed = probabilities >= threshold
    body = ~truth
    true_tail_count = int(np.count_nonzero(truth))
    body_count = int(np.count_nonzero(body))
    true_tail_routed_count = int(np.count_nonzero(routed & truth))
    false_routed_count = int(np.count_nonzero(routed & body))
    return {
        "threshold": threshold,
        "rows": int(probabilities.size),
        "routed_row_count": int(np.count_nonzero(routed)),
        "routed_percentage": float(np.mean(routed) * 100.0),
        "true_tail_count": true_tail_count,
        "true_tail_routed_count": true_tail_routed_count,
        "true_tail_recall": float(true_tail_routed_count / true_tail_count) if true_tail_count else float("nan"),
        "body_count": body_count,
        "false_routed_count": false_routed_count,
        "false_routing_rate": float(false_routed_count / body_count) if body_count else float("nan"),
    }


def _correction_group(correction: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = correction[mask]
    if values.size == 0:
        return {
            f"{prefix}_rows": 0,
            f"{prefix}_mean_signed_correction": float("nan"),
            f"{prefix}_positive_correction_count": 0,
            f"{prefix}_negative_correction_count": 0,
            f"{prefix}_zero_correction_count": 0,
            f"{prefix}_positive_correction_rate": float("nan"),
            f"{prefix}_negative_correction_rate": float("nan"),
            f"{prefix}_zero_correction_rate": float("nan"),
        }
    positive = values > 0.0
    negative = values < 0.0
    zero = values == 0.0
    return {
        f"{prefix}_rows": int(values.size),
        f"{prefix}_mean_signed_correction": float(np.mean(values)),
        f"{prefix}_positive_correction_count": int(np.count_nonzero(positive)),
        f"{prefix}_negative_correction_count": int(np.count_nonzero(negative)),
        f"{prefix}_zero_correction_count": int(np.count_nonzero(zero)),
        f"{prefix}_positive_correction_rate": float(np.mean(positive)),
        f"{prefix}_negative_correction_rate": float(np.mean(negative)),
        f"{prefix}_zero_correction_rate": float(np.mean(zero)),
    }


def correction_summary(
    global_prediction: Any,
    final_prediction: Any,
    operational_tail_true: Any,
) -> dict[str, float | int]:
    """Summarize ``final - global`` corrections overall and by Tail/body truth."""
    global_values = _float_vector(global_prediction, name="global_prediction")
    final_values = _float_vector(final_prediction, name="final_prediction")
    truth = _bool_vector(operational_tail_true, name="operational_tail_true")
    _require_same_length(
        {"global_prediction": global_values, "final_prediction": final_values, "operational_tail_true": truth}
    )
    correction = final_values - global_values
    absolute = np.abs(correction)
    return {
        "rows": int(correction.size),
        "mean_absolute_correction": float(np.mean(absolute)),
        "p95_absolute_correction": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_correction": float(np.max(absolute)),
        "mean_signed_correction": float(np.mean(correction)),
        **_correction_group(correction, ~truth, "body"),
        **_correction_group(correction, truth, "tail"),
    }


__all__ = [
    "SEED",
    "EXPECTED_VALIDATION_ROWS",
    "VALIDATION_SELECTION_ROWS",
    "VALIDATION_AUDIT_ROWS",
    "MODEL_PREDICTION_COLUMN",
    "EXPECTED_PREDICTION_COLUMNS",
    "HARD_ROUTING_THRESHOLDS",
    "SOFT_MIXTURE_ALPHAS",
    "NO_FIT_ENSEMBLE_DEFINITIONS",
    "HARD_THRESHOLDS",
    "ALPHA_GRID",
    "NO_FIT_ENSEMBLES",
    "get_no_fit_ensemble_definitions",
    "duplicate_safe_deciles",
    "make_validation_selection_audit_split",
    "deterministic_validation_split",
    "validate_prediction_alignment",
    "build_aligned_prediction_table",
    "weighted_ensemble_prediction",
    "apply_ensemble",
    "optimize_convex_mae_weights",
    "optimize_convex_mae",
    "hard_routing_prediction",
    "hard_route",
    "soft_mixture_prediction",
    "soft_mix",
    "prediction_diversity_matrices",
    "prediction_correlation_report",
    "correlation_report",
    "routing_summary",
    "correction_summary",
]
