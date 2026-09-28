"""Frozen CatBoost components for Prompt 4A Tail experiments.

This module contains model primitives only.  It does not discover files, load
Development, inspect IID data, or choose a Prompt 4A Candidate.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


SEED = 42
THREAD_COUNT = 4
TARGET = "loan_amount_000s"
PRIMARY_FEATURE_CONTRACT = "main_without_sensitive_without_lender"
Q90_QUANTILE = 0.90
TAIL_FIT_COUNT = 360_000
TAIL_STOP_COUNT = 40_000
TAIL_WEIGHTS = (2, 4)
MISSING_SENTINEL = "__MISSING__"
BUNDLE_FORMAT_VERSION = 1

PRIMARY_FEATURE_NAMES = (
    "agency_name", "loan_type_name", "property_type_name",
    "loan_purpose_name", "owner_occupancy_name", "preapproval_name",
    "msamd_name", "state_name", "state_abbr", "state_code",
    "county_name", "county_code", "census_tract_number",
    "lien_status_name", "applicant_income_000s", "population",
    "hud_median_family_income", "tract_to_msamd_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
    "log1p_applicant_income", "log1p_population",
    "log1p_hud_median_family_income", "log1p_owner_occupied_units",
    "log1p_1_to_4_family_units", "applicant_income_to_area_income",
    "tract_income_ratio", "owner_occupied_unit_ratio",
    "family_units_per_1000_people", "owner_occupied_units_per_1000_people",
    "has_co_applicant", "loan_program_group", "applicant_income_area_group",
    "tract_income_level", "us_region",
)

NUMERIC_FEATURE_NAMES = (
    "applicant_income_000s", "population", "hud_median_family_income",
    "tract_to_msamd_income", "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units", "log1p_applicant_income",
    "log1p_population", "log1p_hud_median_family_income",
    "log1p_owner_occupied_units", "log1p_1_to_4_family_units",
    "applicant_income_to_area_income", "tract_income_ratio",
    "owner_occupied_unit_ratio", "family_units_per_1000_people",
    "owner_occupied_units_per_1000_people", "has_co_applicant",
)

CATEGORICAL_FEATURE_NAMES = tuple(
    name for name in PRIMARY_FEATURE_NAMES if name not in NUMERIC_FEATURE_NAMES
)
CAT_FEATURE_INDICES = tuple(
    PRIMARY_FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURE_NAMES
)

GATE_PARAMETERS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 1500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10,
    "auto_class_weights": "Balanced",
    "random_seed": SEED,
    "thread_count": THREAD_COUNT,
    "early_stopping_rounds": 100,
    "verbose": False,
}

SPECIALIST_PARAMETERS = {
    "loss_function": "MAE",
    "iterations": 2000,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 20,
    "random_strength": 1,
    "random_seed": SEED,
    "thread_count": THREAD_COUNT,
    "early_stopping_rounds": 100,
    "verbose": False,
}

_WEIGHTED_BASE_PARAMETERS = {
    key: value
    for key, value in SPECIALIST_PARAMETERS.items()
    if key != "early_stopping_rounds"
}
WEIGHTED_PARAMETERS = dict(_WEIGHTED_BASE_PARAMETERS)
TAIL_WEIGHTED_CONFIGS = {
    weight: dict(WEIGHTED_PARAMETERS) for weight in TAIL_WEIGHTS
}
GATE_SELECTION_CONFIG = GATE_PARAMETERS
SPECIALIST_SELECTION_CONFIG = SPECIALIST_PARAMETERS


def _one_dimensional(values: Sequence[Any], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    return array


def ordered_digest(values: Sequence[Any]) -> str:
    array = _one_dimensional(values, "digest values").astype(str)
    return hashlib.sha256("\n".join(array.tolist()).encode("utf-8")).hexdigest()


def membership_digest(values: Sequence[Any]) -> str:
    array = _one_dimensional(values, "membership values").astype(str)
    if len(set(array.tolist())) != array.size:
        raise ValueError("Membership values must be unique.")
    return ordered_digest(np.sort(array))


def operational_tail_labels(y: Sequence[float], q90_train: float) -> np.ndarray:
    values = _one_dimensional(y, "target").astype(np.float64)
    if not np.isfinite(values).all() or not np.isfinite(q90_train):
        raise ValueError("Tail labels require finite targets and threshold.")
    return (values > float(q90_train)).astype(np.int8)


def make_tail_weights(
    y: Sequence[float], q90_train: float, tail_weight: int | float
) -> np.ndarray:
    if tail_weight not in TAIL_WEIGHTS:
        raise ValueError(f"tail_weight must be one of {TAIL_WEIGHTS}.")
    labels = operational_tail_labels(y, q90_train)
    return np.where(labels == 1, float(tail_weight), 1.0).astype(np.float64)


def tail_sample_weights(
    y: Sequence[float], q90_train: float, tail_weight: int | float
) -> np.ndarray:
    return make_tail_weights(y, q90_train, tail_weight)


def duplicate_safe_target_bins(y: Sequence[float], bins: int = 10) -> np.ndarray:
    """Create deterministic quantile bins without splitting equal targets."""
    values = _one_dimensional(y, "target").astype(np.float64)
    if not np.isfinite(values).all() or bins < 2:
        raise ValueError("Target values must be finite and bins must be at least two.")
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size <= 2:
        return np.zeros(values.size, dtype=np.int16)
    return np.searchsorted(edges[1:-1], values, side="left").astype(np.int16)


def make_internal_tail_split(
    y_train: Sequence[float],
    train_row_hashes: Sequence[Any],
    q90_train: float,
    validation_row_hashes: Sequence[Any] = (),
    *,
    random_state: int = SEED,
    fit_rows: int = TAIL_FIT_COUNT,
    stop_rows: int = TAIL_STOP_COUNT,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the frozen Train-only 360k/40k positional split."""
    target = _one_dimensional(y_train, "Train target").astype(np.float64)
    hashes = _one_dimensional(train_row_hashes, "Train row hashes").astype(str)
    if target.size != hashes.size or target.size != fit_rows + stop_rows:
        raise ValueError("Internal Tail split row counts are not exact.")
    if len(set(hashes.tolist())) != hashes.size:
        raise ValueError("Train row hashes must be unique.")
    validation = np.asarray(validation_row_hashes, dtype=str).reshape(-1)
    if set(hashes.tolist()) & set(validation.tolist()):
        raise ValueError("Internal Tail split overlaps external Validation.")

    tail = operational_tail_labels(target, q90_train)
    decile = duplicate_safe_target_bins(target)
    composite = np.asarray(
        [f"{int(t)}:{int(d)}" for t, d in zip(tail, decile)], dtype=object
    )
    counts = pd.Series(composite).value_counts()
    stratify = composite
    method = "operational_tail_plus_duplicate_safe_target_bin"
    if counts.min() < 2:
        stratify = tail
        method = "operational_tail_only_rare_composite_fallback"

    positions = np.arange(target.size, dtype=np.int64)
    fit_index, stop_index = train_test_split(
        positions,
        train_size=fit_rows,
        test_size=stop_rows,
        random_state=int(random_state),
        shuffle=True,
        stratify=stratify,
    )
    fit_index = np.sort(fit_index)
    stop_index = np.sort(stop_index)
    if fit_index.size != fit_rows or stop_index.size != stop_rows:
        raise RuntimeError("Internal Tail split produced incorrect counts.")
    if np.intersect1d(fit_index, stop_index).size:
        raise RuntimeError("Internal Tail fit and stop positions overlap.")
    evidence = {
        "status": "PASS",
        "random_state": int(random_state),
        "stratification": method,
        "fit_rows": int(fit_index.size),
        "stop_rows": int(stop_index.size),
        "overlap_rows": 0,
        "validation_overlap_rows": 0,
        "external_validation_overlap": 0,
        "fit_tail_rows": int(tail[fit_index].sum()),
        "stop_tail_rows": int(tail[stop_index].sum()),
        "fit_row_hash_digest": ordered_digest(hashes[fit_index]),
        "stop_row_hash_digest": ordered_digest(hashes[stop_index]),
        "train_membership_digest": membership_digest(hashes),
    }
    return fit_index, stop_index, evidence


deterministic_tail_system_split = make_internal_tail_split


def validate_fit_isolation(
    fit_row_hashes: Sequence[Any],
    authorized_train_row_hashes: Sequence[Any],
    *,
    stop_row_hashes: Sequence[Any] = (),
    validation_row_hashes: Sequence[Any] = (),
    require_full_authorized_fit: bool = False,
    require_partition_authorized: bool = False,
) -> dict[str, Any]:
    fit = _one_dimensional(fit_row_hashes, "fit row hashes").astype(str)
    authorized = _one_dimensional(
        authorized_train_row_hashes, "authorized Train row hashes"
    ).astype(str)
    stop = np.asarray(stop_row_hashes, dtype=str).reshape(-1)
    validation = np.asarray(validation_row_hashes, dtype=str).reshape(-1)
    for name, values in (("fit", fit), ("authorized", authorized), ("stop", stop)):
        if len(set(values.tolist())) != values.size:
            raise ValueError(f"{name} row hashes must be unique.")
    fit_set, authorized_set = set(fit.tolist()), set(authorized.tolist())
    stop_set, validation_set = set(stop.tolist()), set(validation.tolist())
    if not fit_set.issubset(authorized_set) or not stop_set.issubset(authorized_set):
        raise ValueError("A fit or stop row is outside authorized Train membership.")
    if fit_set & stop_set:
        raise ValueError("Fit and stop memberships overlap.")
    if (fit_set | stop_set) & validation_set:
        raise ValueError("Model selection or fitting overlaps external Validation.")
    if require_full_authorized_fit and fit_set != authorized_set:
        raise ValueError("Full refit does not equal authorized Train membership.")
    if require_partition_authorized and fit_set | stop_set != authorized_set:
        raise ValueError("Selection fit and stop rows do not partition authorized Train membership.")
    return {
        "status": "PASS",
        "fit_rows": int(fit.size),
        "stop_rows": int(stop.size),
        "fit_row_hash_digest": ordered_digest(fit),
        "fit_membership_digest": membership_digest(fit),
        "authorized_train_membership_digest": membership_digest(authorized),
        "fit_stop_overlap": 0,
        "validation_overlap": 0,
        "full_authorized_fit": bool(fit_set == authorized_set),
        "partitions_authorized_train": bool(fit_set | stop_set == authorized_set),
    }


class ExplicitCatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """Preserve the exact Prompt 2 numeric and categorical feature roles."""

    def __init__(
        self,
        feature_names: Sequence[str] = PRIMARY_FEATURE_NAMES,
        categorical_features: Sequence[str] = CATEGORICAL_FEATURE_NAMES,
        missing_sentinel: str = MISSING_SENTINEL,
    ) -> None:
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.missing_sentinel = missing_sentinel

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatBoost preprocessing requires a pandas DataFrame.")
        names = list(self.feature_names)
        categorical = list(self.categorical_features)
        if names != list(PRIMARY_FEATURE_NAMES):
            raise ValueError("Prompt 4A Tail models require the exact primary feature order.")
        if categorical != list(CATEGORICAL_FEATURE_NAMES):
            raise ValueError("Prompt 4A CatBoost categorical roles are frozen.")
        missing = sorted(set(names) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required Tail-model features: {missing}")
        if TARGET in names or "respondent_id" in names:
            raise ValueError("Target or lender field entered the Tail-model contract.")
        self.feature_names_in_ = names
        self.categorical_features_ = categorical
        self.numeric_features_ = [name for name in names if name not in categorical]
        self.cat_feature_indices_ = [names.index(name) for name in categorical]
        if tuple(self.numeric_features_) != NUMERIC_FEATURE_NAMES:
            raise RuntimeError("Frozen numeric feature roles changed.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_in_"):
            raise RuntimeError("CatBoost preprocessor has not been fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CatBoost preprocessing requires a pandas DataFrame.")
        missing = sorted(set(self.feature_names_in_) - set(X.columns))
        if missing:
            raise ValueError(f"Missing required Tail-model features: {missing}")
        frame = X.loc[:, self.feature_names_in_].copy()
        for name in self.categorical_features_:
            frame[name] = frame[name].fillna(self.missing_sentinel).astype(str)
        for name in self.numeric_features_:
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype(np.float64)
        return frame

    def evidence(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "feature_names": list(self.feature_names_in_),
            "numeric_features": list(self.numeric_features_),
            "categorical_features": list(self.categorical_features_),
            "cat_feature_indices": list(self.cat_feature_indices_),
            "missing_sentinel": self.missing_sentinel,
        }


@dataclass
class TailModelMetadata:
    model_id: str
    model_role: str
    feature_names: list[str]
    feature_contract: str
    model_configuration: dict[str, Any]
    seed: int
    selected_iteration: int
    training_row_count: int
    train_membership_digest: str
    q90_train: float
    target_definition: str
    package_versions: dict[str, str]
    authorized_train_membership_digest: str = ""
    development_source_sha256: str = ""
    validation_row_hash_digest: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feature_names != list(PRIMARY_FEATURE_NAMES):
            raise ValueError("Bundle feature contract is not the frozen 35-feature contract.")
        if self.feature_contract != PRIMARY_FEATURE_CONTRACT:
            raise ValueError("Bundle feature-contract name is invalid.")
        if self.seed != SEED or self.selected_iteration < 1 or self.training_row_count < 1:
            raise ValueError("Bundle seed, iteration, or row-count metadata is invalid.")
        if not np.isfinite(self.q90_train):
            raise ValueError("Bundle q90_train must be finite.")


@dataclass
class RegressionBundle:
    metadata: TailModelMetadata
    preprocessor: ExplicitCatBoostPreprocessor
    model: Any

    def predict(self, raw_frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(raw_frame)
        prediction = np.asarray(self.model.predict(transformed), dtype=np.float64).reshape(-1)
        if prediction.size != len(raw_frame) or not np.isfinite(prediction).all():
            raise RuntimeError("Tail regression bundle produced invalid predictions.")
        return prediction


@dataclass
class GateBundle:
    metadata: TailModelMetadata
    preprocessor: ExplicitCatBoostPreprocessor
    model: Any

    def predict_tail_probability(self, raw_frame: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(raw_frame)
        probabilities = np.asarray(self.model.predict_proba(transformed), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape != (len(raw_frame), 2):
            raise RuntimeError("Tail Gate did not produce a two-class probability matrix.")
        tail_probability = probabilities[:, 1]
        if not np.isfinite(tail_probability).all() or np.any(
            (tail_probability < 0.0) | (tail_probability > 1.0)
        ):
            raise RuntimeError("Tail Gate produced invalid probabilities.")
        return tail_probability

    def predict_proba(self, raw_frame: pd.DataFrame) -> np.ndarray:
        tail = self.predict_tail_probability(raw_frame)
        return np.column_stack([1.0 - tail, tail])


TailRegressionBundle = RegressionBundle
TailGateBundle = GateBundle


def fixed_refit_config(
    selection_parameters: Mapping[str, Any], selected_iteration: int
) -> dict[str, Any]:
    if int(selected_iteration) < 1:
        raise ValueError("selected_iteration must be positive.")
    result = dict(selection_parameters)
    result.pop("early_stopping_rounds", None)
    result["iterations"] = int(selected_iteration)
    result["verbose"] = False
    return result


def _catboost_constructor_parameters(parameters: Mapping[str, Any]) -> tuple[dict, int | None]:
    resolved = dict(parameters)
    early_stopping_rounds = resolved.pop("early_stopping_rounds", None)
    resolved.update({"allow_writing_files": False, "task_type": "CPU", "verbose": False})
    return resolved, early_stopping_rounds


def _selected_iteration(model: Any, requested_iterations: int, early_stopping: bool) -> int:
    if early_stopping:
        best = int(model.get_best_iteration())
        if best >= 0:
            return best + 1
    tree_count = int(getattr(model, "tree_count_", requested_iterations))
    return tree_count if tree_count > 0 else int(requested_iterations)


def fit_gate(
    X_fit: pd.DataFrame,
    y_fit: Sequence[int],
    *,
    fit_row_hashes: Sequence[Any],
    authorized_train_row_hashes: Sequence[Any],
    X_stop: pd.DataFrame | None = None,
    y_stop: Sequence[int] | None = None,
    stop_row_hashes: Sequence[Any] = (),
    validation_row_hashes: Sequence[Any] = (),
    parameters: Mapping[str, Any] | None = None,
) -> tuple[Any, ExplicitCatBoostPreprocessor, int, dict[str, Any]]:
    """Fit one Gate selection or fixed-refit model on authorized Train rows."""
    from catboost import CatBoostClassifier

    if len(X_fit) != len(y_fit) or len(X_fit) != len(fit_row_hashes):
        raise ValueError("Gate fit inputs are not row-aligned.")
    use_stop = X_stop is not None or y_stop is not None or len(stop_row_hashes) > 0
    if use_stop and (X_stop is None or y_stop is None or len(X_stop) != len(y_stop)):
        raise ValueError("Gate stop inputs must be complete and row-aligned.")
    isolation = validate_fit_isolation(
        fit_row_hashes, authorized_train_row_hashes,
        stop_row_hashes=stop_row_hashes,
        validation_row_hashes=validation_row_hashes,
        require_full_authorized_fit=not use_stop,
        require_partition_authorized=use_stop,
    )
    labels = _one_dimensional(y_fit, "Gate labels").astype(np.int8)
    if not set(np.unique(labels)).issubset({0, 1}) or np.unique(labels).size != 2:
        raise ValueError("Gate fit labels must contain both binary classes.")
    preprocessor = ExplicitCatBoostPreprocessor().fit(X_fit)
    transformed_fit = preprocessor.transform(X_fit)
    config = dict(GATE_PARAMETERS if parameters is None else parameters)
    constructor, early_stopping_rounds = _catboost_constructor_parameters(config)
    model = CatBoostClassifier(**constructor)
    fit_kwargs: dict[str, Any] = {
        "cat_features": preprocessor.cat_feature_indices_, "verbose": False,
    }
    if use_stop:
        stop_labels = _one_dimensional(y_stop, "Gate stop labels").astype(np.int8)
        if len(X_stop) != len(stop_row_hashes):
            raise ValueError("Gate stop hashes are not row-aligned.")
        fit_kwargs.update({
            "eval_set": (preprocessor.transform(X_stop), stop_labels),
            "use_best_model": True,
        })
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    elif early_stopping_rounds is not None:
        raise ValueError("A full Gate refit configuration must not contain early stopping.")
    model.fit(transformed_fit, labels, **fit_kwargs)
    selected = _selected_iteration(model, int(constructor["iterations"]), use_stop)
    return model, preprocessor, selected, isolation


def fit_regressor(
    X_fit: pd.DataFrame,
    y_fit: Sequence[float],
    *,
    fit_row_hashes: Sequence[Any],
    authorized_train_row_hashes: Sequence[Any],
    X_stop: pd.DataFrame | None = None,
    y_stop: Sequence[float] | None = None,
    stop_row_hashes: Sequence[Any] = (),
    validation_row_hashes: Sequence[Any] = (),
    sample_weight: Sequence[float] | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> tuple[Any, ExplicitCatBoostPreprocessor, int, dict[str, Any]]:
    """Fit one Specialist or weighted CatBoost regressor with isolation checks."""
    from catboost import CatBoostRegressor

    target = _one_dimensional(y_fit, "regression target").astype(np.float64)
    if len(X_fit) != target.size or len(X_fit) != len(fit_row_hashes):
        raise ValueError("Regressor fit inputs are not row-aligned.")
    if not np.isfinite(target).all():
        raise ValueError("Regressor targets must be finite.")
    use_stop = X_stop is not None or y_stop is not None or len(stop_row_hashes) > 0
    if use_stop and (X_stop is None or y_stop is None or len(X_stop) != len(y_stop)):
        raise ValueError("Regressor stop inputs must be complete and row-aligned.")
    isolation = validate_fit_isolation(
        fit_row_hashes, authorized_train_row_hashes,
        stop_row_hashes=stop_row_hashes,
        validation_row_hashes=validation_row_hashes,
        require_full_authorized_fit=not use_stop,
        require_partition_authorized=use_stop,
    )
    preprocessor = ExplicitCatBoostPreprocessor().fit(X_fit)
    transformed_fit = preprocessor.transform(X_fit)
    config = dict(SPECIALIST_PARAMETERS if parameters is None else parameters)
    constructor, early_stopping_rounds = _catboost_constructor_parameters(config)
    model = CatBoostRegressor(**constructor)
    fit_kwargs: dict[str, Any] = {
        "cat_features": preprocessor.cat_feature_indices_, "verbose": False,
    }
    if sample_weight is not None:
        weights = _one_dimensional(sample_weight, "sample weights").astype(np.float64)
        if weights.size != target.size or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Sample weights must be finite, positive, and row-aligned.")
        fit_kwargs["sample_weight"] = weights
    if use_stop:
        stop_target = _one_dimensional(y_stop, "regression stop target").astype(np.float64)
        if len(X_stop) != len(stop_row_hashes) or not np.isfinite(stop_target).all():
            raise ValueError("Regressor stop inputs are invalid.")
        fit_kwargs.update({
            "eval_set": (preprocessor.transform(X_stop), stop_target),
            "use_best_model": True,
        })
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    elif early_stopping_rounds is not None:
        raise ValueError("A full regressor refit configuration must not contain early stopping.")
    model.fit(transformed_fit, target, **fit_kwargs)
    selected = _selected_iteration(model, int(constructor["iterations"]), use_stop)
    return model, preprocessor, selected, isolation


def _bundle_destination(path: str | Path) -> Path:
    destination = Path(path).resolve()
    lowered = [part.lower() for part in destination.parts]
    required = ["regression_v2", "outputs", "models", "prompt4a"]
    if not any(lowered[i:i + len(required)] == required for i in range(len(lowered) - len(required) + 1)):
        raise ValueError("Prompt 4A bundles must stay under regression_v2/outputs/models/prompt4a.")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_bundle(bundle: GateBundle | RegressionBundle, destination: str | Path) -> dict[str, Any]:
    directory = _bundle_destination(destination)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "bundle.joblib"
    temporary = directory / "bundle.joblib.tmp"
    joblib.dump(bundle, temporary, compress=3)
    reloaded = joblib.load(temporary)
    if type(reloaded) is not type(bundle):
        raise TypeError("Tail bundle temporary reload has the wrong type.")
    os.replace(temporary, artifact)
    manifest = {
        "status": "COMPLETE",
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "bundle_type": "gate" if isinstance(bundle, GateBundle) else "regression",
        "artifact": artifact.name,
        "artifact_sha256": _sha256(artifact),
        "metadata": asdict(bundle.metadata),
        "preprocessing": bundle.preprocessor.evidence(),
    }
    manifest_path = directory / "manifest.json"
    manifest_temporary = directory / "manifest.json.tmp"
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    json.loads(manifest_temporary.read_text(encoding="utf-8"))
    os.replace(manifest_temporary, manifest_path)
    return manifest


def save_gate_bundle(bundle: GateBundle, destination: str | Path) -> dict[str, Any]:
    if not isinstance(bundle, GateBundle):
        raise TypeError("save_gate_bundle requires a GateBundle.")
    return _save_bundle(bundle, destination)


def save_regression_bundle(bundle: RegressionBundle, destination: str | Path) -> dict[str, Any]:
    if not isinstance(bundle, RegressionBundle):
        raise TypeError("save_regression_bundle requires a RegressionBundle.")
    return _save_bundle(bundle, destination)


def _load_bundle(destination: str | Path, expected_type: type) -> Any:
    directory = _bundle_destination(destination)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise RuntimeError("Tail bundle manifest is incomplete or incompatible.")
    artifact = directory / manifest["artifact"]
    if _sha256(artifact) != manifest.get("artifact_sha256"):
        raise RuntimeError("Tail bundle artifact hash does not match its manifest.")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, expected_type):
        raise TypeError("Tail bundle payload has the wrong type.")
    if asdict(bundle.metadata) != manifest.get("metadata"):
        raise RuntimeError("Tail bundle metadata does not match its manifest.")
    return bundle


def load_gate_bundle(destination: str | Path) -> GateBundle:
    return _load_bundle(destination, GateBundle)


def load_regression_bundle(destination: str | Path) -> RegressionBundle:
    return _load_bundle(destination, RegressionBundle)


save_tail_bundle = _save_bundle


def load_tail_bundle(destination: str | Path) -> GateBundle | RegressionBundle:
    directory = _bundle_destination(destination)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    kind = manifest.get("bundle_type")
    if kind == "gate":
        return load_gate_bundle(directory)
    if kind == "regression":
        return load_regression_bundle(directory)
    raise ValueError(f"Unknown Tail bundle type: {kind}")


# Explicit role aliases used by orchestration and focused tests.
fit_gate_selection = fit_gate
fit_gate_full_refit = fit_gate
fit_specialist_selection = fit_regressor
fit_specialist_full_refit = fit_regressor
fit_tail_weighted_regressor = fit_regressor
