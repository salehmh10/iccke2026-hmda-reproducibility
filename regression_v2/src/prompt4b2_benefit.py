"""Leakage-safe Expected-Benefit Router utilities for Prompt 4B2.

This module defines formulas, model factories, reloadable bundles, and
diagnostics. Importing it never reads data and never fits a model.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

try:
    from .preprocessing import CatBoostFramePreprocessor
    from .prompt4_metrics import compute_regression_metrics
except ImportError:
    from preprocessing import CatBoostFramePreprocessor
    from prompt4_metrics import compute_regression_metrics


SEED = 42
GLOBAL_FEATURE = "global_prediction_feature"
PROPOSED_CORRECTION_FEATURE = "proposed_residual_correction"
BENEFIT_TARGET = "benefit_oof"
BASE_FEATURE_COUNT = 35
BENEFIT_ROUTER_FEATURE_COUNT = 37
BENEFIT_THRESHOLDS = (0.0, 5.0, 10.0)
# Frozen strict routes: predicted_benefit > 0, predicted_benefit > 5,
# and predicted_benefit > 10. Equality never selects a row.
OOF_FOLD_LABELS = ("fold_a", "fold_b")
BENEFIT_SELECTION_FIT_ROWS = 360_000
BENEFIT_SELECTION_STOP_ROWS = 40_000
BENEFIT_FULL_REFIT_ROWS = 400_000

BENEFIT_ROUTER_PARAMETERS: dict[str, Any] = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10,
    "random_strength": 1,
    "random_seed": SEED,
    "thread_count": 4,
    "early_stopping_rounds": 100,
    "verbose": False,
}

# These fields are valid for diagnostics or row alignment, but never as router
# inputs. The clean 35-feature contract is checked separately by the caller.
FORBIDDEN_ROUTER_FEATURES = frozenset(
    {
        "loan_amount_000s",
        "y_true",
        "target_decile",
        "operational_tail",
        "realized_residual",
        "absolute_error",
        "realized_benefit",
        BENEFIT_TARGET,
        "row_hash",
        "record_hash",
        "respondent_id",
        "old_gate_score",
        "meta_gate_score",
        "specialist_realized_error",
    }
)


def _finite_vector(values: Any, name: str, *, allow_empty: bool = False) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if (not allow_empty and result.size == 0) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite, non-empty vector.")
    return result


def _aligned_vectors(**values: Any) -> dict[str, np.ndarray]:
    result = {name: _finite_vector(value, name) for name, value in values.items()}
    sizes = {array.size for array in result.values()}
    if len(sizes) != 1:
        raise ValueError("Benefit arrays must be exactly row-aligned.")
    return result


def positive_only_capped_proposal(residual_prediction: Any, cap25: float) -> np.ndarray:
    """Return ``clip(max(residual_prediction, 0), 0, cap25)`` exactly."""
    residual = _finite_vector(residual_prediction, "residual_prediction")
    cap = float(cap25)
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError("cap25 must be finite and positive.")
    return np.clip(np.maximum(residual, 0.0), 0.0, cap)


# The longer name mirrors the frozen design language.
proposed_residual_correction = positive_only_capped_proposal


def benefit_target(
    y_true: Any,
    global_oof_prediction: Any,
    proposed_correction: Any,
) -> np.ndarray:
    """Calculate OOF absolute-error reduction from a fixed correction.

    The exact formula is
    ``abs(y - G_OOF) - abs(y - (G_OOF + proposal_oof))``.
    Positive values mean the proposal helps; negative values mean it harms.
    """
    arrays = _aligned_vectors(
        y_true=y_true,
        global_oof_prediction=global_oof_prediction,
        proposed_correction=proposed_correction,
    )
    y = arrays["y_true"]
    global_prediction = arrays["global_oof_prediction"]
    proposal = arrays["proposed_correction"]
    if np.any(proposal < 0.0):
        raise ValueError("The Prompt 4B2 benefit proposal must be positive-only.")
    return np.abs(y - global_prediction) - np.abs(y - (global_prediction + proposal))


compute_benefit_target = benefit_target


def build_benefit_training_table(
    row_hash: Sequence[Any],
    y_true: Any,
    global_oof_prediction: Any,
    residual_oof_prediction: Any,
    cap25: float,
) -> pd.DataFrame:
    """Build the minimal non-sensitive OOF benefit artifact."""
    hashes = pd.Series(row_hash, copy=False).astype(str).reset_index(drop=True)
    arrays = _aligned_vectors(
        y_true=y_true,
        global_oof_prediction=global_oof_prediction,
        residual_oof_prediction=residual_oof_prediction,
    )
    if len(hashes) != arrays["y_true"].size or hashes.duplicated().any():
        raise ValueError("Benefit training rows need aligned, unique row_hash values.")
    proposal = positive_only_capped_proposal(arrays["residual_oof_prediction"], cap25)
    benefit = benefit_target(arrays["y_true"], arrays["global_oof_prediction"], proposal)
    return pd.DataFrame(
        {
            "row_hash": hashes,
            "global_oof_prediction": arrays["global_oof_prediction"],
            "residual_oof_prediction": arrays["residual_oof_prediction"],
            PROPOSED_CORRECTION_FEATURE: proposal,
            BENEFIT_TARGET: benefit,
        }
    )


def validate_benefit_feature_contract(base_features: Sequence[str]) -> list[str]:
    """Validate and return the exact 35 + 2 router feature names."""
    names = [str(name) for name in base_features]
    if len(names) != BASE_FEATURE_COUNT or len(set(names)) != BASE_FEATURE_COUNT:
        raise ValueError("Benefit Router requires exactly 35 unique clean base features.")
    forbidden = sorted(set(names) & FORBIDDEN_ROUTER_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden Benefit Router features: {forbidden}")
    if GLOBAL_FEATURE in names or PROPOSED_CORRECTION_FEATURE in names:
        raise ValueError("Router-added features must not already be base features.")
    contract = names + [GLOBAL_FEATURE, PROPOSED_CORRECTION_FEATURE]
    if len(contract) != BENEFIT_ROUTER_FEATURE_COUNT:
        raise RuntimeError("Benefit Router feature contract is not 37 columns.")
    return contract


def make_benefit_feature_frame(
    base_frame: pd.DataFrame,
    base_features: Sequence[str],
    global_prediction: Any,
    proposed_correction: Any,
) -> pd.DataFrame:
    """Create the inference-available 37-column Benefit Router frame."""
    if not isinstance(base_frame, pd.DataFrame):
        raise TypeError("base_frame must be a pandas DataFrame.")
    contract = validate_benefit_feature_contract(base_features)
    missing = [name for name in contract[:BASE_FEATURE_COUNT] if name not in base_frame]
    if missing:
        raise ValueError(f"Missing clean Benefit Router features: {missing}")
    arrays = _aligned_vectors(
        global_prediction=global_prediction,
        proposed_correction=proposed_correction,
    )
    if len(base_frame) != arrays["global_prediction"].size:
        raise ValueError("Benefit Router feature inputs are not row-aligned.")
    if np.any(arrays["proposed_correction"] < 0.0):
        raise ValueError("The proposed residual correction must be positive-only.")
    frame = base_frame.loc[:, contract[:BASE_FEATURE_COUNT]].copy()
    frame[GLOBAL_FEATURE] = arrays["global_prediction"]
    frame[PROPOSED_CORRECTION_FEATURE] = arrays["proposed_correction"]
    return frame.loc[:, contract]


@dataclass
class BenefitPreprocessor:
    """CatBoost frame preprocessing for the exact 37-feature contract."""

    base_features: list[str]
    global_feature: str = GLOBAL_FEATURE
    proposal_feature: str = PROPOSED_CORRECTION_FEATURE

    def fit(self, frame: pd.DataFrame):
        if self.global_feature != GLOBAL_FEATURE or self.proposal_feature != PROPOSED_CORRECTION_FEATURE:
            raise ValueError("Benefit Router added-feature names are frozen.")
        names = validate_benefit_feature_contract(self.base_features)
        self.transformer_ = CatBoostFramePreprocessor(names).fit(frame[names])
        self.feature_names_in_ = names
        self.cat_feature_indices_ = list(self.transformer_.cat_feature_indices_)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "transformer_"):
            raise RuntimeError("BenefitPreprocessor must be fitted before transform.")
        return self.transformer_.transform(frame[self.feature_names_in_])

    def evidence(self) -> dict[str, Any]:
        if not hasattr(self, "transformer_"):
            raise RuntimeError("BenefitPreprocessor has no fitted evidence.")
        return {
            "type": type(self).__name__,
            "feature_names": list(self.feature_names_in_),
            "feature_count": len(self.feature_names_in_),
            "global_feature": self.global_feature,
            "proposal_feature": self.proposal_feature,
            "cat_feature_indices": list(self.cat_feature_indices_),
        }


@dataclass
class BenefitRouterBundle:
    """Reloadable experimental bundle that predicts expected MAE reduction."""

    preprocessor: BenefitPreprocessor
    model: Any
    metadata: dict[str, Any]

    def predict_expected_benefit(self, frame: pd.DataFrame) -> np.ndarray:
        prediction = np.asarray(
            self.model.predict(self.preprocessor.transform(frame)), dtype=np.float64
        ).reshape(-1)
        if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
            raise RuntimeError("Benefit Router produced invalid predictions.")
        return prediction

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_expected_benefit(frame)


def fixed_benefit_router_config(selected_iteration: int) -> dict[str, Any]:
    """Create the no-early-stopping configuration for the 400k refit."""
    iteration = int(selected_iteration)
    if iteration < 1:
        raise ValueError("selected_iteration must be positive.")
    config = dict(BENEFIT_ROUTER_PARAMETERS)
    config.pop("early_stopping_rounds", None)
    config["iterations"] = iteration
    config["verbose"] = False
    return config


def fit_benefit_router(
    X_fit: pd.DataFrame,
    y_fit: Any,
    base_features: Sequence[str],
    *,
    X_stop: pd.DataFrame | None = None,
    y_stop: Any | None = None,
    parameters: Mapping[str, Any] | None = None,
    fixed_iterations: int | None = None,
) -> tuple[Any, BenefitPreprocessor, int]:
    """Fit one authorized selection model or fixed-iteration refit.

    A stop set selects the iteration with RMSE and early stopping. Without a
    stop set, callers must provide a fixed configuration made by
    :func:`fixed_benefit_router_config`.
    """
    from catboost import CatBoostRegressor

    target = _finite_vector(y_fit, "benefit target")
    if len(X_fit) != target.size:
        raise ValueError("Benefit fit rows and targets are not aligned.")
    names = validate_benefit_feature_contract(base_features)
    if list(X_fit.columns) != names:
        raise ValueError("Benefit fit frame must contain the exact ordered 37-feature contract.")
    if fixed_iterations is not None:
        if X_stop is not None or y_stop is not None:
            raise ValueError("fixed_iterations cannot be used with a stop set.")
        if parameters is not None:
            raise ValueError("Use either parameters or fixed_iterations, not both.")
        config = fixed_benefit_router_config(fixed_iterations)
    else:
        config = dict(BENEFIT_ROUTER_PARAMETERS if parameters is None else parameters)
    early_stopping = config.pop("early_stopping_rounds", None)
    preprocessor = BenefitPreprocessor(list(base_features)).fit(X_fit)
    model = CatBoostRegressor(**config, allow_writing_files=False, task_type="CPU")
    fit_kwargs: dict[str, Any] = {
        "cat_features": preprocessor.cat_feature_indices_,
        "verbose": False,
    }
    use_stop = X_stop is not None or y_stop is not None
    if use_stop:
        if X_stop is None or y_stop is None:
            raise ValueError("Benefit stop inputs must be supplied together.")
        stop_target = _finite_vector(y_stop, "benefit stop target")
        if len(X_stop) != stop_target.size or list(X_stop.columns) != names:
            raise ValueError("Benefit stop rows must use the exact aligned feature contract.")
        if early_stopping is None:
            raise ValueError("The Benefit selection fit must keep early stopping = 100.")
        fit_kwargs.update(
            {
                "eval_set": (preprocessor.transform(X_stop), stop_target),
                "use_best_model": True,
                "early_stopping_rounds": int(early_stopping),
            }
        )
    elif early_stopping is not None:
        raise ValueError("A fixed Benefit Router refit must not use early stopping.")
    model.fit(preprocessor.transform(X_fit), target, **fit_kwargs)
    best = int(model.get_best_iteration()) if use_stop else -1
    selected = best + 1 if best >= 0 else int(model.tree_count_)
    if selected < 1:
        raise RuntimeError("Benefit Router did not produce a valid iteration count.")
    return model, preprocessor, selected


def fit_benefit_router_selection(
    X_fit: pd.DataFrame,
    y_fit: Any,
    X_stop: pd.DataFrame,
    y_stop: Any,
    base_features: Sequence[str],
) -> tuple[Any, BenefitPreprocessor, int]:
    return fit_benefit_router(
        X_fit, y_fit, base_features, X_stop=X_stop, y_stop=y_stop,
        parameters=BENEFIT_ROUTER_PARAMETERS,
    )


def fit_benefit_router_fixed_refit(
    X_train: pd.DataFrame,
    y_train: Any,
    base_features: Sequence[str],
    selected_iteration: int,
) -> tuple[Any, BenefitPreprocessor, int]:
    return fit_benefit_router(
        X_train, y_train, base_features,
        parameters=fixed_benefit_router_config(selected_iteration),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_bundle_metadata(bundle: BenefitRouterBundle) -> None:
    metadata = bundle.metadata
    required = {
        "feature_contract",
        "model_configuration",
        "selected_iteration",
        "seed",
        "training_membership_digest",
        "development_source_sha256",
        "package_versions",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Benefit Router bundle metadata is incomplete: {missing}")
    contract = metadata["feature_contract"]
    if list(contract) != list(bundle.preprocessor.feature_names_in_) or len(contract) != 37:
        raise ValueError("Benefit Router bundle metadata has the wrong feature contract.")
    if int(metadata["seed"]) != SEED or int(metadata["selected_iteration"]) < 1:
        raise ValueError("Benefit Router bundle seed or selected iteration is invalid.")
    if not str(metadata["training_membership_digest"]) or not str(metadata["development_source_sha256"]):
        raise ValueError("Benefit Router bundle source and membership digests are required.")
    if not isinstance(metadata["package_versions"], Mapping):
        raise ValueError("Benefit Router package versions must be a mapping.")


def save_benefit_router_bundle(
    bundle: BenefitRouterBundle, destination: str | Path
) -> dict[str, Any]:
    """Atomically save and reload-check an experimental Benefit Router."""
    if not isinstance(bundle, BenefitRouterBundle):
        raise TypeError("save_benefit_router_bundle requires a BenefitRouterBundle.")
    _validate_bundle_metadata(bundle)
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "bundle.joblib"
    temporary = directory / "bundle.joblib.tmp"
    joblib.dump(bundle, temporary, compress=3)
    reloaded = joblib.load(temporary)
    if not isinstance(reloaded, BenefitRouterBundle):
        raise RuntimeError("Benefit Router temporary bundle reload failed.")
    _validate_bundle_metadata(reloaded)
    os.replace(temporary, artifact)
    manifest = {
        "status": "COMPLETE",
        "bundle_type": "benefit_router",
        "bundle_format_version": 1,
        "artifact": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "metadata": bundle.metadata,
        "preprocessing": bundle.preprocessor.evidence(),
    }
    manifest_temp = directory / "manifest.json.tmp"
    manifest_temp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    json.loads(manifest_temp.read_text(encoding="utf-8"))
    os.replace(manifest_temp, directory / "manifest.json")
    return manifest


def load_benefit_router_bundle(destination: str | Path) -> BenefitRouterBundle:
    directory = Path(destination)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("bundle_type") != "benefit_router":
        raise RuntimeError("Benefit Router manifest is incomplete or incompatible.")
    artifact = directory / manifest["artifact"]
    if _sha256(artifact) != manifest.get("artifact_sha256"):
        raise RuntimeError("Benefit Router artifact hash does not match its manifest.")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, BenefitRouterBundle):
        raise TypeError("Benefit Router bundle payload has the wrong type.")
    _validate_bundle_metadata(bundle)
    if bundle.metadata != manifest.get("metadata"):
        raise RuntimeError("Benefit Router bundle metadata does not match its manifest.")
    return bundle


def _group_benefit(labels: Sequence[Any], benefit: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"group": labels, "benefit": benefit})
    rows: list[dict[str, Any]] = []
    for group, part in frame.groupby("group", observed=True, sort=True):
        values = part["benefit"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "group": str(group),
                "rows": int(values.size),
                "mean_benefit": float(np.mean(values)),
                "median_benefit": float(np.median(values)),
                "positive_benefit_proportion": float(np.mean(values > 0.0)),
                "negative_benefit_proportion": float(np.mean(values < 0.0)),
            }
        )
    return rows


def benefit_target_diagnostics(
    y_true: Any,
    proposed_correction: Any,
    benefit_oof: Any,
    cap25: float,
) -> dict[str, Any]:
    """Describe the frozen benefit target before router fitting."""
    arrays = _aligned_vectors(
        y_true=y_true,
        proposed_correction=proposed_correction,
        benefit_oof=benefit_oof,
    )
    y = arrays["y_true"]
    proposal = arrays["proposed_correction"]
    benefit = arrays["benefit_oof"]
    cap = float(cap25)
    if not np.isfinite(cap) or cap <= 0.0 or np.any(proposal < 0.0) or np.any(proposal > cap):
        raise ValueError("Benefit diagnostics require proposals inside [0, cap25].")
    positive = benefit > 0.0
    negative = benefit < 0.0
    zero = benefit == 0.0
    # Rank-first qcut gives ten deterministic, near-equal diagnostic groups
    # even when the target has tied values. These labels never enter a model.
    target_decile = pd.qcut(
        pd.Series(y).rank(method="first"), 10, labels=False
    ).astype(int)
    edges = [-np.inf, 0.0, 0.25 * cap, 0.50 * cap, 0.75 * cap, cap]
    labels = ["zero", "positive_0_25", "positive_25_50", "positive_50_75", "positive_75_100"]
    correction_bin = pd.cut(proposal, bins=edges, labels=labels, include_lowest=True)
    return {
        "rows": int(benefit.size),
        "mean_benefit": float(np.mean(benefit)),
        "median_benefit": float(np.median(benefit)),
        "positive_benefit_proportion": float(np.mean(positive)),
        "negative_benefit_proportion": float(np.mean(negative)),
        "zero_benefit_proportion": float(np.mean(zero)),
        "mean_positive_benefit": float(np.mean(benefit[positive])) if np.any(positive) else None,
        "mean_negative_benefit": float(np.mean(benefit[negative])) if np.any(negative) else None,
        "p05_benefit": float(np.quantile(benefit, 0.05)),
        "p50_benefit": float(np.quantile(benefit, 0.50)),
        "p95_benefit": float(np.quantile(benefit, 0.95)),
        "benefit_by_target_decile": _group_benefit(target_decile, benefit),
        "benefit_by_proposed_correction_magnitude_bin": _group_benefit(correction_bin, benefit),
        "target_decile_is_diagnostic_only": True,
    }


def router_quality_diagnostics(
    realized_benefit: Any, predicted_benefit: Any
) -> dict[str, Any]:
    """Report regression and sign quality for expected-benefit predictions."""
    arrays = _aligned_vectors(
        realized_benefit=realized_benefit,
        predicted_benefit=predicted_benefit,
    )
    actual = arrays["realized_benefit"]
    predicted = arrays["predicted_benefit"]
    error = predicted - actual
    actual_rank = pd.Series(actual).rank(method="average").to_numpy(float)
    predicted_rank = pd.Series(predicted).rank(method="average").to_numpy(float)
    if actual.size > 1 and np.std(actual_rank) > 0.0 and np.std(predicted_rank) > 0.0:
        spearman = float(np.corrcoef(actual_rank, predicted_rank)[0, 1])
    else:
        spearman = None
    return {
        "rows": int(actual.size),
        "benefit_mae": float(np.mean(np.abs(error))),
        "benefit_rmse": float(np.sqrt(np.mean(error**2))),
        "spearman_correlation": spearman,
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(actual))),
    }


def realized_benefit(
    y_true: Any, global_prediction: Any, proposed_correction: Any
) -> np.ndarray:
    """Calculate Validation benefit for evaluation only."""
    return benefit_target(y_true, global_prediction, proposed_correction)


def benefit_threshold_diagnostics(
    y_true: Any,
    global_prediction: Any,
    proposed_correction: Any,
    predicted_benefit: Any,
    q90_train: float,
    *,
    thresholds: Sequence[float] = BENEFIT_THRESHOLDS,
) -> pd.DataFrame:
    """Evaluate exactly B0, B5, and B10 with strict ``>`` routing."""
    resolved_thresholds = tuple(float(value) for value in thresholds)
    if resolved_thresholds != BENEFIT_THRESHOLDS:
        raise ValueError("Prompt 4B2 Benefit thresholds are exactly 0, 5, and 10.")
    arrays = _aligned_vectors(
        y_true=y_true,
        global_prediction=global_prediction,
        proposed_correction=proposed_correction,
        predicted_benefit=predicted_benefit,
    )
    y = arrays["y_true"]
    global_pred = arrays["global_prediction"]
    proposal = arrays["proposed_correction"]
    expected = arrays["predicted_benefit"]
    if np.any(proposal < 0.0):
        raise ValueError("Benefit routing requires a positive-only proposal.")
    q90 = float(q90_train)
    if not np.isfinite(q90):
        raise ValueError("q90_train must be finite.")
    actual_benefit = realized_benefit(y, global_pred, proposal)
    beneficial = actual_benefit > 0.0
    operational_tail = y > q90
    rows: list[dict[str, Any]] = []
    for threshold in BENEFIT_THRESHOLDS:
        selected = expected > threshold
        harmful = selected & (actual_benefit < 0.0)
        prediction = global_pred + np.where(selected, proposal, 0.0)
        metrics = compute_regression_metrics(y, prediction)
        selected_count = int(np.count_nonzero(selected))
        beneficial_count = int(np.count_nonzero(beneficial))
        rows.append(
            {
                "candidate_id": f"benefit_router_b{int(threshold)}",
                "benefit_threshold": threshold,
                "selected_row_count": selected_count,
                "selected_row_percentage": float(100.0 * np.mean(selected)),
                "selected_positive_benefit_proportion": (
                    float(np.mean(beneficial[selected])) if selected_count else None
                ),
                "mean_realized_benefit_selected": (
                    float(np.mean(actual_benefit[selected])) if selected_count else None
                ),
                "mean_realized_damage_harmful_selected": (
                    float(np.mean(-actual_benefit[harmful])) if np.any(harmful) else None
                ),
                "tail_proportion_selected": (
                    float(np.mean(operational_tail[selected])) if selected_count else None
                ),
                "precision_realized_benefit_positive": (
                    float(np.count_nonzero(selected & beneficial) / selected_count)
                    if selected_count else None
                ),
                "recall_realized_benefit_positive": (
                    float(np.count_nonzero(selected & beneficial) / beneficial_count)
                    if beneficial_count else None
                ),
                "overall_mae": metrics["mae"],
                "body_mae": (
                    float(np.mean(np.abs(prediction[~operational_tail] - y[~operational_tail])))
                    if np.any(~operational_tail) else None
                ),
                "top_decile_mae": metrics["top_decile_mae"],
                "top_five_percent_mae": metrics["top_five_percent_mae"],
                "rmse": metrics["rmse"],
            }
        )
    return pd.DataFrame(rows)


# Compatibility aliases for orchestration code that uses shorter names.
BENEFIT_PARAMETERS = BENEFIT_ROUTER_PARAMETERS
BenefitRouterPreprocessor = BenefitPreprocessor
save_bundle = save_benefit_router_bundle
load_bundle = load_benefit_router_bundle
