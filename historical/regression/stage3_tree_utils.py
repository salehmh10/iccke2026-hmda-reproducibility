"""Serializable utilities for Stage 3 tree-based regression models.

The notebook is the main Stage 3 deliverable. This module keeps custom
transformers importable so saved pipelines work in a clean Python process.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor


RANDOM_SEED = 42
TARGET_COLUMN = "loan_amount_000s"

BASE_NUMERIC_FEATURES = [
    "applicant_income_000s",
    "population",
    "hud_median_family_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
    "applicant_income_to_area_income",
    "tract_income_ratio",
    "owner_occupied_unit_ratio",
    "family_units_per_1000_people",
    "owner_occupied_units_per_1000_people",
    "has_co_applicant",
]

BASE_CATEGORICAL_FEATURES = [
    "agency_name",
    "loan_type_name",
    "property_type_name",
    "loan_purpose_name",
    "owner_occupancy_name",
    "preapproval_name",
    "state_name",
    "lien_status_name",
    "loan_program_group",
    "applicant_income_area_group",
    "tract_income_level",
    "us_region",
]

SENSITIVE_NUMERIC_FEATURES = ["minority_population"]
SENSITIVE_CATEGORICAL_FEATURES = [
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "majority_minority_tract",
]
SENSITIVE_FEATURES = SENSITIVE_CATEGORICAL_FEATURES[:-1] + SENSITIVE_NUMERIC_FEATURES + [
    "majority_minority_tract"
]

ENGINEERED_NUMERIC_FEATURES = [
    "applicant_income_to_tract_income",
    "applicant_vs_tract_income_gap_000s",
    "family_owner_unit_count_difference",
]
ENGINEERED_CATEGORICAL_FEATURES = [
    "loan_type_property_group",
    "purpose_occupancy_group",
    "purpose_preapproval_group",
    "agency_loan_program_group",
    "purpose_income_area_group",
]

EXTENDED_SOURCE_FEATURES = [
    "respondent_id",
    "msamd_name",
    "state_county_group",
    "state_county_tract_group",
]
EXTENDED_FREQUENCY_FEATURES = [f"{name}__frequency" for name in EXTENDED_SOURCE_FEATURES]

VALID_FEATURE_PACKS = ("tree_base_v1", "tree_engineered_v1", "tree_extended_v1")
VALID_SENSITIVE_MODES = ("without_sensitive", "with_sensitive")


def canonical_json(value: Any) -> str:
    """Return compact stable JSON for hashes and experiment IDs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def configuration_digest(value: Any, length: int = 12) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def deterministic_experiment_id(
    model_name: str,
    sensitive_mode: str,
    target_mode: str,
    evaluation_stage: str,
    fold_number: int | None,
    configuration: dict[str, Any],
    feature_pack: str,
) -> str:
    """Build an idempotent Stage 3 experiment ID."""
    fold_text = "na" if fold_number is None else str(int(fold_number))
    digest = configuration_digest({"feature_pack": feature_pack, "configuration": configuration})
    return (
        f"stage3__{model_name}__{sensitive_mode}__{target_mode}__"
        f"{evaluation_stage}__fold-{fold_text}__cfg-{digest}"
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Write JSON through a temporary file, then replace the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, destination)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def category_to_object(values: Any) -> Any:
    """Convert pandas categories to objects and keep missing values."""
    if isinstance(values, pd.DataFrame):
        converted = values.astype(object)
        return converted.where(pd.notna(converted), np.nan)
    array = np.asarray(values, dtype=object)
    array[pd.isna(array)] = np.nan
    return array


def _safe_text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].astype(object)
    values = values.where(pd.notna(values), "__MISSING__")
    return values.astype(str)


class SafeTreeFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add fixed target-independent features without changing the input."""

    def __init__(self, feature_pack: str = "tree_base_v1") -> None:
        self.feature_pack = feature_pack

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SafeTreeFeatureEngineer":
        self._validate_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_frame(X)
        result = X.copy(deep=False)
        if self.feature_pack in {"tree_engineered_v1", "tree_extended_v1"}:
            result = result.copy()
            result["loan_type_property_group"] = (
                _safe_text_series(result, "loan_type_name") + " | " + _safe_text_series(result, "property_type_name")
            )
            result["purpose_occupancy_group"] = (
                _safe_text_series(result, "loan_purpose_name") + " | " + _safe_text_series(result, "owner_occupancy_name")
            )
            result["purpose_preapproval_group"] = (
                _safe_text_series(result, "loan_purpose_name") + " | " + _safe_text_series(result, "preapproval_name")
            )
            result["agency_loan_program_group"] = (
                _safe_text_series(result, "agency_name") + " | " + _safe_text_series(result, "loan_program_group")
            )
            result["purpose_income_area_group"] = (
                _safe_text_series(result, "loan_purpose_name") + " | " + _safe_text_series(result, "applicant_income_area_group")
            )
            income_area_ratio = pd.to_numeric(result["applicant_income_to_area_income"], errors="coerce")
            tract_ratio = pd.to_numeric(result["tract_income_ratio"], errors="coerce")
            safe_tract_ratio = tract_ratio.where(tract_ratio > 0)
            result["applicant_income_to_tract_income"] = (
                income_area_ratio.div(safe_tract_ratio).replace([np.inf, -np.inf], np.nan).astype("float32")
            )
            applicant_income = pd.to_numeric(result["applicant_income_000s"], errors="coerce")
            area_income_000s = pd.to_numeric(result["hud_median_family_income"], errors="coerce") / 1000.0
            result["applicant_vs_tract_income_gap_000s"] = (
                applicant_income - area_income_000s * tract_ratio
            ).replace([np.inf, -np.inf], np.nan).astype("float32")
            family_units = pd.to_numeric(result["number_of_1_to_4_family_units"], errors="coerce")
            owner_units = pd.to_numeric(result["number_of_owner_occupied_units"], errors="coerce")
            result["family_owner_unit_count_difference"] = (family_units - owner_units).astype("float32")
        if self.feature_pack == "tree_extended_v1":
            result["state_county_group"] = (
                _safe_text_series(result, "state_name") + " | " + _safe_text_series(result, "county_name")
            )
            result["state_county_tract_group"] = (
                result["state_county_group"].astype(str) + " | " + _safe_text_series(result, "census_tract_number")
            )
        return result

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        names = list(self.feature_names_in_ if input_features is None else input_features)
        if self.feature_pack in {"tree_engineered_v1", "tree_extended_v1"}:
            names.extend(ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES)
        if self.feature_pack == "tree_extended_v1":
            names.extend(["state_county_group", "state_county_tract_group"])
        return np.asarray(list(dict.fromkeys(names)), dtype=object)

    def _validate_frame(self, X: Any) -> None:
        if self.feature_pack not in VALID_FEATURE_PACKS:
            raise ValueError(f"Unknown feature pack: {self.feature_pack}")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SafeTreeFeatureEngineer requires a pandas DataFrame.")
        required = set(BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES)
        if self.feature_pack == "tree_extended_v1":
            required.update(["respondent_id", "msamd_name", "county_name", "census_tract_number"])
        missing = sorted(required.difference(X.columns))
        if missing:
            raise ValueError(f"Missing feature-engineering columns: {missing}")


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Learn category frequencies inside each fitted pipeline."""

    def __init__(self, columns: Iterable[str] = ()) -> None:
        self.columns = tuple(columns)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrequencyEncoder requires a pandas DataFrame.")
        missing = sorted(set(self.columns).difference(X.columns))
        if missing:
            raise ValueError(f"Missing frequency columns: {missing}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.frequency_maps_ = {}
        row_count = max(len(X), 1)
        for column in self.columns:
            values = _safe_text_series(X, column)
            counts = values.value_counts(dropna=False)
            self.frequency_maps_[column] = (counts / row_count).astype(float).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrequencyEncoder requires a pandas DataFrame.")
        result = X.copy()
        for column in self.columns:
            values = _safe_text_series(result, column)
            result[f"{column}__frequency"] = values.map(self.frequency_maps_[column]).fillna(0.0).astype("float32")
        return result

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        names = list(self.feature_names_in_ if input_features is None else input_features)
        names.extend(f"{column}__frequency" for column in self.columns)
        return np.asarray(list(dict.fromkeys(names)), dtype=object)


def feature_lists(feature_pack: str, sensitive_mode: str) -> dict[str, list[str]]:
    """Return stable feature lists for a pack and sensitive mode."""
    if feature_pack not in VALID_FEATURE_PACKS:
        raise ValueError(f"Unknown feature pack: {feature_pack}")
    if sensitive_mode not in VALID_SENSITIVE_MODES:
        raise ValueError(f"Unknown sensitive mode: {sensitive_mode}")
    numeric = list(BASE_NUMERIC_FEATURES)
    categorical = list(BASE_CATEGORICAL_FEATURES)
    frequency_sources: list[str] = []
    raw = list(BASE_NUMERIC_FEATURES + BASE_CATEGORICAL_FEATURES)
    if feature_pack in {"tree_engineered_v1", "tree_extended_v1"}:
        numeric.extend(ENGINEERED_NUMERIC_FEATURES)
        categorical.extend(ENGINEERED_CATEGORICAL_FEATURES)
    if feature_pack == "tree_extended_v1":
        frequency_sources = list(EXTENDED_SOURCE_FEATURES)
        numeric.extend(EXTENDED_FREQUENCY_FEATURES)
        raw.extend(["respondent_id", "msamd_name", "county_name", "census_tract_number"])
    if sensitive_mode == "with_sensitive":
        numeric.extend(SENSITIVE_NUMERIC_FEATURES)
        categorical.extend(SENSITIVE_CATEGORICAL_FEATURES)
        raw.extend(SENSITIVE_FEATURES)
    return {
        "numeric": list(dict.fromkeys(numeric)),
        "categorical": list(dict.fromkeys(categorical)),
        "frequency_sources": frequency_sources,
        "raw": list(dict.fromkeys(raw)),
    }


def make_one_hot_encoder() -> OneHotEncoder:
    """Build a sparse, bounded, version-compatible encoder."""
    kwargs: dict[str, Any] = {
        "handle_unknown": "infrequent_if_exist",
        "min_frequency": 20,
        "max_categories": 64,
        "dtype": np.float32,
    }
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        kwargs["sparse_output"] = True
    else:  # pragma: no cover
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)


def make_ordinal_encoder() -> OrdinalEncoder:
    """Build a bounded ordinal encoder for native HGB categories."""
    kwargs: dict[str, Any] = {
        "handle_unknown": "use_encoded_value",
        "unknown_value": np.nan,
        "encoded_missing_value": np.nan,
        "dtype": np.float32,
    }
    signature = inspect.signature(OrdinalEncoder).parameters
    if "min_frequency" in signature:
        kwargs["min_frequency"] = 20
    if "max_categories" in signature:
        kwargs["max_categories"] = 254
    return OrdinalEncoder(**kwargs)


def make_tree_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("to_object", FunctionTransformer(category_to_object, validate=False, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )


def make_hist_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("to_object", FunctionTransformer(category_to_object, validate=False, feature_names_out="one-to-one")),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_ordinal_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def make_estimator(configuration: dict[str, Any], numeric_count: int, categorical_count: int) -> BaseEstimator:
    """Return a new unfitted estimator from a frozen configuration."""
    config = deepcopy(configuration)
    model_name = config.pop("model_name")
    config.pop("feature_pack", None)
    config.pop("target_mode", None)
    if model_name == "decision_tree":
        return DecisionTreeRegressor(random_state=RANDOM_SEED, **config)
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=int(config.pop("n_jobs", 2)), **config)
    if model_name == "extra_trees":
        return ExtraTreesRegressor(random_state=RANDOM_SEED, n_jobs=int(config.pop("n_jobs", 2)), **config)
    if model_name == "hist_gradient_boosting":
        mask = [False] * numeric_count + [True] * categorical_count
        return HistGradientBoostingRegressor(
            random_state=RANDOM_SEED,
            categorical_features=mask,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            **config,
        )
    if model_name == "ebm":
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("EBM is unavailable because interpret could not be imported.") from exc
        return ExplainableBoostingRegressor(random_state=RANDOM_SEED, **config)
    raise ValueError(f"Unknown model name: {model_name}")


def make_complete_pipeline(configuration: dict[str, Any], sensitive_mode: str) -> BaseEstimator:
    """Build a fresh raw-DataFrame-to-prediction pipeline."""
    feature_pack = configuration["feature_pack"]
    lists = feature_lists(feature_pack, sensitive_mode)
    model_name = configuration["model_name"]
    if model_name == "hist_gradient_boosting":
        preprocessor = make_hist_preprocessor(lists["numeric"], lists["categorical"])
    else:
        preprocessor = make_tree_preprocessor(lists["numeric"], lists["categorical"])
    inner = Pipeline([
        ("feature_engineering", SafeTreeFeatureEngineer(feature_pack=feature_pack)),
        ("frequency_encoding", FrequencyEncoder(columns=lists["frequency_sources"])),
        ("preprocessor", preprocessor),
        ("regressor", make_estimator(configuration, len(lists["numeric"]), len(lists["categorical"]))),
    ])
    if configuration["target_mode"] == "log1p":
        return TransformedTargetRegressor(
            regressor=inner,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=True,
        )
    if configuration["target_mode"] != "raw":
        raise ValueError(f"Unknown target mode: {configuration['target_mode']}")
    return inner


def fitted_pipeline(model: BaseEstimator) -> Pipeline:
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def fitted_estimator(model: BaseEstimator) -> BaseEstimator:
    return fitted_pipeline(model).named_steps["regressor"]


def transformed_feature_names(model: BaseEstimator) -> np.ndarray:
    return fitted_pipeline(model).named_steps["preprocessor"].get_feature_names_out()


def _validated_numeric_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector.")
    try:
        array = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def evaluate_regression_predictions(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Calculate the fixed Stage 1 metrics on the original target scale."""
    true = _validated_numeric_vector(y_true, "y_true")
    pred = _validated_numeric_vector(y_pred, "y_pred")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length.")
    errors = pred - true
    absolute_errors = np.abs(errors)
    mse = float(np.mean(errors**2))
    negative_rate = float(np.mean(pred < 0))
    warnings: list[str] = []
    if np.any(true == 0):
        mape = None
        warnings.append("MAPE is unavailable because y_true contains zero.")
    else:
        mape = float(np.mean(np.abs(errors / true)) * 100)
    denominator = float(np.sum(np.abs(true)))
    wape = None if denominator == 0 else float(np.sum(absolute_errors) / denominator * 100)
    r_squared = None if len(true) < 2 or np.all(true == true[0]) else float(r2_score(true, pred))
    rmsle = None
    rmsle_clipped_zero = None
    if np.any(true < 0):
        warnings.append("RMSLE is unavailable because y_true contains negative values.")
    elif negative_rate > 0:
        rmsle_clipped_zero = float(
            np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(true)) ** 2))
        )
        warnings.append("Negative predictions were clipped only for rmsle_clipped_zero.")
    else:
        rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
    mae = float(np.mean(absolute_errors))
    rmse = float(np.sqrt(mse))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape_percent": mape,
        "r_squared": r_squared,
        "rmsle": rmsle,
        "rmsle_clipped_zero": rmsle_clipped_zero,
        "median_absolute_error": float(np.median(absolute_errors)),
        "wape_percent": wape,
        "mean_signed_error": float(np.mean(errors)),
        "p90_absolute_error": float(np.quantile(absolute_errors, 0.90)),
        "negative_prediction_rate": negative_rate,
        "mae_usd": mae * 1000,
        "rmse_usd": rmse * 1000,
        "metric_warnings": warnings,
    }


def read_training_rows(
    csv_path: str | Path,
    train_row_ids: Iterable[int],
    categorical_columns: Iterable[str],
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read only saved training rows and preserve their source row IDs."""
    train_ids = np.sort(np.asarray(list(train_row_ids), dtype=np.int64))
    train_set = set(train_ids.tolist())
    columns = None if usecols is None else list(usecols)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    selected = header if columns is None else columns
    dtype_map = {name: "category" for name in categorical_columns if name in selected}
    frame = pd.read_csv(
        csv_path,
        usecols=columns,
        dtype=dtype_map,
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in train_set,
        low_memory=False,
    )
    if len(frame) != len(train_ids):
        raise AssertionError(f"Training-only load returned {len(frame)} rows, expected {len(train_ids)}.")
    frame.index = pd.Index(train_ids, name="row_id")
    return frame


def upsert_registry(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Upsert Stage 3 rows without changing prior rows or schema."""
    if "experiment_id" not in existing.columns or "experiment_id" not in new_rows.columns:
        raise ValueError("Registry frames require experiment_id.")
    aligned = new_rows.copy()
    for column in existing.columns:
        if column not in aligned.columns:
            aligned[column] = np.nan
    extra = [column for column in aligned.columns if column not in existing.columns]
    if extra:
        raise ValueError(f"Stage 3 Registry rows contain unknown columns: {extra}")
    aligned = aligned[existing.columns]
    prior = existing.loc[~existing["experiment_id"].isin(aligned["experiment_id"])].copy()
    combined = pd.concat([prior, aligned], ignore_index=True)
    if combined["experiment_id"].duplicated().any():
        raise AssertionError("Registry experiment IDs are not unique after upsert.")
    return combined


def feature_source_name(transformed_name: str, possible_features: Iterable[str]) -> str:
    """Map a transformed feature name back to its source when possible."""
    name = str(transformed_name).split("__", 1)[-1]
    for feature in sorted(set(possible_features), key=len, reverse=True):
        if name == feature or name.startswith(feature + "_"):
            return feature
    return name


def save_model(path: str | Path, model: BaseEstimator, compress: int = 3) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(model, temporary, compress=compress)
    os.replace(temporary, destination)


def finite_prediction_check(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    prediction = np.asarray(model.predict(X), dtype=float)
    if prediction.ndim != 1 or len(prediction) != len(X) or not np.isfinite(prediction).all():
        raise AssertionError("Model predictions are not a complete finite vector.")
    return prediction


def model_size_bytes(path: str | Path) -> int:
    return int(Path(path).stat().st_size)


def stable_frame_digest(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> str:
    selected = frame if columns is None else frame[list(columns)]
    values = pd.util.hash_pandas_object(selected, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def package_versions() -> dict[str, str | None]:
    import sklearn

    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import interpret

        versions["interpret"] = getattr(interpret, "__version__", "unknown")
    except Exception:
        versions["interpret"] = None
    return versions


def metric_columns() -> list[str]:
    return [
        "mae",
        "mse",
        "rmse",
        "mape_percent",
        "r_squared",
        "rmsle",
        "rmsle_clipped_zero",
        "median_absolute_error",
        "wape_percent",
        "mean_signed_error",
        "p90_absolute_error",
        "negative_prediction_rate",
        "mae_usd",
        "rmse_usd",
    ]


def relative_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline == 0:
        return math.nan
    return float((baseline - candidate) / baseline * 100.0)
