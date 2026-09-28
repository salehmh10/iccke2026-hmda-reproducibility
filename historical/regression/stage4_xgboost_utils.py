"""Shared, recovery-safe helpers for the Stage 4I-K XGBoost track.

Only saved Train row IDs are read. The locked Test target is never loaded and
no Test prediction is created. Learned transforms stay inside each bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import stage4_boosting_utils as s4


STAGE_NAMES = {
    "stage4i": "Stage 4I — Initial XGBoost Model and Importance Analysis",
    "stage4j": "Stage 4J — XGBoost Feature Confirmation and Final Tuning",
    "stage4k": "Stage 4K — Final XGBoost Sensitive Comparison and Full-Train Models",
}
VERSION = "stage4i_stage4k_xgboost_v1_20260714"
FEATURE_ENGINEER_VERSION = "xgboost_feature_engineer_v2_20260714"
TARGET = s4.TARGET_COLUMN
SEED = 42
THREAD_COUNT = 2
EXECUTION_MODE = "cpu"
MAX_SHAP_ROWS = 300

SENSITIVE_NUMERIC = ("minority_population",)
SENSITIVE_CATEGORICAL = (
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "majority_minority_tract",
)
SENSITIVE_COLUMNS = SENSITIVE_CATEGORICAL + SENSITIVE_NUMERIC

PROPOSAL_RATIO = "applicant_to_estimated_tract_income_ratio"
PROPOSAL_RESPONDENT_PURPOSE = "respondent_purpose_group"
PROPOSAL_INCOME_LIEN = "income_band_lien_status_group"
SAFE_PROPOSALS = (PROPOSAL_RATIO, PROPOSAL_RESPONDENT_PURPOSE, PROPOSAL_INCOME_LIEN)
RATIO_ZERO_FLAG = "applicant_to_estimated_tract_income_denominator_zero"

STARTING_PARAMETERS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
    "gamma": 0.0,
    "max_bin": 256,
    "random_state": SEED,
    "n_jobs": THREAD_COUNT,
    "eval_metric": "mae",
}


def paths(root: str | Path) -> dict[str, Path]:
    project = Path(root).resolve()
    return {
        "root": project,
        "initial": project / "artifacts/results/stage4/xgboost/initial",
        "confirmation": project / "artifacts/results/stage4/xgboost/feature_confirmation",
        "final": project / "artifacts/results/stage4/xgboost/final",
        "predictions": project / "artifacts/predictions/xgboost",
        "preliminary_models": project / "artifacts/models/xgboost/preliminary",
        "candidate_models": project / "artifacts/models/xgboost/candidates",
        "final_models": project / "artifacts/models/xgboost/final",
        "features": project / "artifacts/features/stage4/xgboost",
        "figures": project / "artifacts/figures/stage4/xgboost",
        "checkpoints": project / "artifacts/checkpoints/stage4/xgboost",
        "manifests": project / "artifacts/manifests/stage4/xgboost",
        "reports": project / "artifacts/reports",
        "registry": project / "artifacts/results/experiment_results.csv",
    }


def ensure_directories(root: str | Path) -> None:
    for name, path in paths(root).items():
        if name not in {"root", "registry"}:
            path.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def environment_metadata(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    s4.activate_local_packages(project)
    import sklearn
    import xgboost

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "scipy_sparse": True,
    }


def source_path(root: str | Path, sensitive_mode: str) -> Path:
    project = Path(root).resolve()
    if sensitive_mode == "without_sensitive":
        return project / "data/regression_without_sensitive_features.csv"
    if sensitive_mode == "with_sensitive":
        return project / "data/regression_with_sensitive_features.csv"
    raise ValueError(f"Unknown sensitive mode: {sensitive_mode}")


def sample_ids(root: str | Path, sample_name: str) -> tuple[np.ndarray, np.ndarray]:
    filenames = {
        "discovery": "stage4_discovery_sample.csv",
        "feature_confirmation": "stage4_feature_confirmation_sample.csv",
        "final_selection": "stage4_final_selection_sample.csv",
    }
    if sample_name not in filenames:
        raise ValueError(f"Unknown Stage 4 sample: {sample_name}")
    frame = pd.read_csv(
        Path(root) / "artifacts/splits/stage4" / filenames[sample_name],
        dtype={"row_id": "int64", "sample_role": "string", "target_bin": "int64"},
    )
    train = frame.loc[frame["sample_role"].eq("train"), "row_id"].to_numpy(dtype=np.int64)
    validation = frame.loc[frame["sample_role"].eq("validation"), "row_id"].to_numpy(dtype=np.int64)
    return train, validation


def all_train_ids(root: str | Path) -> np.ndarray:
    frame = pd.read_csv(Path(root) / "artifacts/splits/train_row_ids.csv", usecols=["row_id"], dtype={"row_id": "int64"})
    return frame["row_id"].to_numpy(dtype=np.int64)


def sample_digest(root: str | Path, sample_name: str) -> str:
    if sample_name == "full_train":
        return s4.sha256_file(Path(root) / "artifacts/splits/train_row_ids.csv")
    filenames = {
        "discovery": "stage4_discovery_sample.csv",
        "feature_confirmation": "stage4_feature_confirmation_sample.csv",
        "final_selection": "stage4_final_selection_sample.csv",
    }
    return s4.sha256_file(Path(root) / "artifacts/splits/stage4" / filenames[sample_name])


def load_feature_pack(root: str | Path, pack_id: str) -> dict[str, Any]:
    data = read_json(Path(root) / "artifacts/features/stage4/boosting_feature_packs.json")
    if pack_id not in data["packs"]:
        raise KeyError(f"Unknown Feature Pack: {pack_id}")
    result = json.loads(json.dumps(data["packs"][pack_id]))
    if "one_hot_categorical" not in result:
        result["one_hot_categorical"] = list(result.get("categorical", ()))
    result.update({"pack_id": pack_id, "version": data["version"], "selected_proposals": []})
    return result


class XGBoostFeatureEngineerV2(BaseEstimator, TransformerMixin):
    """Preserve Stage 4B fixed Features and add only reviewed Stage 4J Features."""

    def __init__(
        self,
        selected_proposals: Sequence[str] = SAFE_PROPOSALS,
        income_quantiles: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        epsilon: float = 1e-8,
        missing_token: str = "<MISSING>",
    ) -> None:
        self.selected_proposals = selected_proposals
        self.income_quantiles = income_quantiles
        self.epsilon = epsilon
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "XGBoostFeatureEngineerV2":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        selected = tuple(self.selected_proposals)
        unknown = sorted(set(selected).difference(SAFE_PROPOSALS))
        if unknown:
            raise ValueError(f"Unknown Stage 4J proposals: {unknown}")
        required = {
            PROPOSAL_RATIO: {"applicant_income_000s", "hud_median_family_income", "tract_income_ratio"},
            PROPOSAL_RESPONDENT_PURPOSE: {"respondent_id", "loan_purpose_name"},
            PROPOSAL_INCOME_LIEN: {"applicant_income_000s", "lien_status_name"},
        }
        missing = sorted({column for proposal in selected for column in required[proposal]}.difference(X.columns))
        if missing:
            raise ValueError(f"Feature Engineer v2 sources are missing: {missing}")
        self.base_engineer_ = s4.Stage4FixedFeatureEngineer().fit(X, y)
        self.selected_proposals_ = selected
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        if PROPOSAL_INCOME_LIEN in selected:
            income = pd.to_numeric(X["applicant_income_000s"], errors="coerce").to_numpy(dtype=float)
            finite = income[np.isfinite(income)]
            if not len(finite):
                raise ValueError("Income-band Feature needs finite training values.")
            quantiles = np.asarray(tuple(self.income_quantiles), dtype=float)
            if quantiles[0] != 0 or quantiles[-1] != 1 or np.any(np.diff(quantiles) <= 0):
                raise ValueError("income_quantiles must increase from 0 to 1.")
            edges = np.unique(np.quantile(finite, quantiles))
            if len(edges) < 2:
                edges = np.asarray([finite.min(), finite.max() + 1.0], dtype=float)
            edges[0], edges[-1] = -np.inf, np.inf
            self.income_band_edges_ = edges
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not hasattr(self, "selected_proposals_"):
            raise RuntimeError("XGBoostFeatureEngineerV2 is not fitted.")
        missing = sorted(set(self.feature_names_in_).difference(X.columns))
        if missing:
            raise ValueError(f"Transform input is missing fitted columns: {missing}")
        result = self.base_engineer_.transform(X.copy())
        if PROPOSAL_RATIO in self.selected_proposals_:
            applicant = pd.to_numeric(result["applicant_income_000s"], errors="coerce")
            denominator = pd.to_numeric(result["estimated_tract_family_income_000s"], errors="coerce")
            finite = np.isfinite(applicant) & np.isfinite(denominator)
            zero = finite & (denominator.abs() <= float(self.epsilon))
            safe = denominator.copy()
            safe.loc[zero] = float(self.epsilon)
            values = pd.Series(np.nan, index=result.index, dtype=float)
            values.loc[finite] = applicant.loc[finite] / safe.loc[finite]
            result[PROPOSAL_RATIO] = values.where(np.isfinite(values))
            result[RATIO_ZERO_FLAG] = zero.astype(float)
        if PROPOSAL_RESPONDENT_PURPOSE in self.selected_proposals_:
            left = result["respondent_id"].astype("string").fillna(self.missing_token)
            right = result["loan_purpose_name"].astype("string").fillna(self.missing_token)
            result[PROPOSAL_RESPONDENT_PURPOSE] = (left + " | " + right).astype(object)
        if PROPOSAL_INCOME_LIEN in self.selected_proposals_:
            income = pd.to_numeric(result["applicant_income_000s"], errors="coerce")
            labels = [f"income_band_{i}" for i in range(len(self.income_band_edges_) - 1)]
            band = pd.cut(income, self.income_band_edges_, labels=labels, include_lowest=True).astype("string").fillna(self.missing_token)
            lien = result["lien_status_name"].astype("string").fillna(self.missing_token)
            result[PROPOSAL_INCOME_LIEN] = (band + " | " + lien).astype(object)
        if not result.index.equals(X.index):
            raise AssertionError("Feature Engineer v2 changed row order.")
        return result

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        base = list(input_features if input_features is not None else self.feature_names_in_)
        for name in s4.STAGE4B_FIXED_FEATURES:
            if name not in base:
                base.append(name)
        if PROPOSAL_RATIO in self.selected_proposals_:
            base.extend(name for name in (PROPOSAL_RATIO, RATIO_ZERO_FLAG) if name not in base)
        for name in (PROPOSAL_RESPONDENT_PURPOSE, PROPOSAL_INCOME_LIEN):
            if name in self.selected_proposals_ and name not in base:
                base.append(name)
        return np.asarray(base, dtype=object)


def feature_pack_spec(root: str | Path, pack_id: str, selected_proposals: Sequence[str] = ()) -> dict[str, Any]:
    base_id = "xgboost_sparse_v1" if pack_id.startswith("xgboost_sparse_v2") else pack_id
    spec = load_feature_pack(root, base_id)
    proposals = tuple(selected_proposals)
    if proposals:
        unknown = sorted(set(proposals).difference(SAFE_PROPOSALS))
        if unknown:
            raise ValueError(f"Unsafe or unknown proposals: {unknown}")
        spec["pack_id"] = pack_id
        spec["version"] = FEATURE_ENGINEER_VERSION
        spec["selected_proposals"] = list(proposals)
        numeric = list(spec.get("numeric", ()))
        one_hot = list(spec.get("one_hot_categorical", ()))
        frequency_sources = list(spec.get("frequency_sources", ()))
        if PROPOSAL_RATIO in proposals:
            numeric.extend([PROPOSAL_RATIO, RATIO_ZERO_FLAG])
        if PROPOSAL_RESPONDENT_PURPOSE in proposals:
            frequency_sources.append(PROPOSAL_RESPONDENT_PURPOSE)
        if PROPOSAL_INCOME_LIEN in proposals:
            one_hot.append(PROPOSAL_INCOME_LIEN)
        spec["numeric"] = list(dict.fromkeys(numeric))
        spec["one_hot_categorical"] = list(dict.fromkeys(one_hot))
        spec["frequency_sources"] = list(dict.fromkeys(frequency_sources))
        spec["frequency_features"] = [f"{name}__frequency" for name in spec["frequency_sources"]]
    return spec


def feature_pack_digest(spec: dict[str, Any]) -> str:
    return sha256_text(s4.canonical_json(spec))


class XGBoostModelBundle:
    """Complete serializable bundle that accepts raw named DataFrames."""

    def __init__(self, preprocessor: Pipeline, model: Any, target_mode: str, metadata: dict[str, Any]) -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.target_mode = target_mode
        self.metadata = metadata

    def transform(self, X: pd.DataFrame) -> Any:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        raw = list(self.metadata["raw_columns"])
        missing = sorted(set(raw).difference(X.columns))
        if missing:
            raise ValueError(f"Bundle input is missing columns: {missing}")
        ready = self.preprocessor.transform(X.loc[:, raw].copy())
        if not sparse.issparse(ready):
            raise ValueError("XGBoost preprocessing unexpectedly produced a dense matrix.")
        return ready

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prediction = s4.inverse_target(self.model.predict(self.transform(X)), self.target_mode)
        if not np.isfinite(prediction).all():
            raise ValueError("Bundle predictions are not finite.")
        return prediction


def prepare_preprocessor(
    root: str | Path,
    spec: dict[str, Any],
    sensitive_mode: str,
) -> tuple[Pipeline, list[str], list[str], list[str]]:
    del root
    numeric = list(spec.get("numeric", ()))
    one_hot = list(spec.get("one_hot_categorical", ()))
    frequency_sources = list(spec.get("frequency_sources", ()))
    raw = list(spec.get("raw", ()))
    if sensitive_mode == "with_sensitive":
        numeric.extend(name for name in SENSITIVE_NUMERIC if name not in numeric)
        one_hot.extend(name for name in SENSITIVE_CATEGORICAL if name not in one_hot)
        raw.extend(name for name in SENSITIVE_COLUMNS if name not in raw)
    elif sensitive_mode != "without_sensitive":
        raise ValueError(f"Unknown sensitive mode: {sensitive_mode}")
    frequency_features = [f"{name}__frequency" for name in frequency_sources]
    model_numeric = list(dict.fromkeys(numeric + frequency_features))
    one_hot = list(dict.fromkeys(one_hot))
    selected = model_numeric + one_hot
    proposals = tuple(spec.get("selected_proposals", ()))
    engineer: Any = XGBoostFeatureEngineerV2(proposals) if proposals else s4.Stage4FixedFeatureEngineer()
    steps: list[tuple[str, Any]] = [("features", engineer)]
    sanitize_columns = list(dict.fromkeys(one_hot + frequency_sources))
    if sanitize_columns:
        steps.append(("sanitize", s4.Stage4CategoricalSanitizer(tuple(sanitize_columns))))
    if frequency_sources:
        steps.append(("frequency", s4.Stage4FrequencyEncoder(tuple(frequency_sources), drop_original=True)))
    steps.append(("select", s4.Stage4ColumnSelector(tuple(selected))))
    transform = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), model_numeric),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
                ]),
                one_hot,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    steps.append(("preprocess", transform))
    return Pipeline(steps), raw, selected, one_hot


def make_model(root: str | Path, parameters: dict[str, Any], early_stopping_rounds: int | None) -> Any:
    s4.activate_local_packages(Path(root).resolve())
    from xgboost import XGBRegressor

    values = dict(parameters)
    values.setdefault("objective", "reg:squarederror")
    values.setdefault("tree_method", "hist")
    values.setdefault("device", EXECUTION_MODE)
    values.setdefault("random_state", SEED)
    values.setdefault("n_jobs", THREAD_COUNT)
    values.setdefault("eval_metric", "mae")
    if early_stopping_rounds is not None:
        values["early_stopping_rounds"] = int(early_stopping_rounds)
    else:
        values.pop("early_stopping_rounds", None)
    return XGBRegressor(**values)


def extended_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, Any]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    metrics = s4.evaluate_regression_predictions(true, pred)
    absolute = np.abs(pred - true)
    top_decile = true >= np.quantile(true, 0.90)
    top_five = true >= np.quantile(true, 0.95)
    metrics.update({
        "top_decile_mae": float(absolute[top_decile].mean()),
        "top_five_percent_mae": float(absolute[top_five].mean()),
        "underestimation_rate": float(np.mean(pred < true)),
        "overestimation_rate": float(np.mean(pred > true)),
    })
    return metrics


def fit_validation(
    root: str | Path,
    spec: dict[str, Any],
    sensitive_mode: str,
    target_mode: str,
    parameters: dict[str, Any],
    train_ids: Iterable[int],
    validation_ids: Iterable[int],
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    project = Path(root).resolve()
    preprocessor, raw, selected, categorical = prepare_preprocessor(project, spec, sensitive_mode)
    train_array = np.asarray(list(train_ids), dtype=np.int64)
    validation_array = np.asarray(list(validation_ids), dtype=np.int64)
    all_ids = np.concatenate([train_array, validation_array])
    frame = s4.read_training_rows(source_path(project, sensitive_mode), all_ids, raw + [TARGET])
    X_train = frame.loc[train_array, raw].copy()
    X_validation = frame.loc[validation_array, raw].copy()
    y_train = frame.loc[train_array, TARGET].to_numpy(dtype=float)
    y_validation = frame.loc[validation_array, TARGET].to_numpy(dtype=float)
    y_fit = s4.transform_target(y_train, target_mode)
    y_eval = s4.transform_target(y_validation, target_mode)
    train_ready = preprocessor.fit_transform(X_train, y_fit)
    validation_ready = preprocessor.transform(X_validation)
    if not sparse.issparse(train_ready) or not sparse.issparse(validation_ready):
        raise ValueError("XGBoost preprocessing unexpectedly became dense.")
    model = make_model(project, parameters, early_stopping_rounds)
    started = time.perf_counter()
    model.fit(train_ready, y_fit, eval_set=[(validation_ready, y_eval)], verbose=False)
    fit_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    prediction = s4.inverse_target(model.predict(validation_ready), target_mode)
    prediction_seconds = time.perf_counter() - prediction_started
    train_prediction = s4.inverse_target(model.predict(train_ready), target_mode)
    best_zero = getattr(model, "best_iteration", None)
    effective = int(best_zero) + 1 if best_zero is not None else int(parameters["n_estimators"])
    return {
        "preprocessor": preprocessor,
        "model": model,
        "raw_columns": raw,
        "selected_features": selected,
        "categorical_features": categorical,
        "train_ids": train_array,
        "validation_ids": validation_array,
        "y_validation": y_validation,
        "validation_prediction": prediction,
        "metrics": extended_metrics(y_validation, prediction),
        "training_metrics": extended_metrics(y_train, train_prediction),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "best_iteration": effective,
        "feature_names": preprocessor.named_steps["preprocess"].get_feature_names_out().tolist(),
        "sparse_train_shape": list(train_ready.shape),
        "sparse_validation_shape": list(validation_ready.shape),
    }


def fit_full_train(
    root: str | Path,
    spec: dict[str, Any],
    sensitive_mode: str,
    target_mode: str,
    parameters: dict[str, Any],
    train_ids: Iterable[int],
) -> dict[str, Any]:
    project = Path(root).resolve()
    preprocessor, raw, selected, categorical = prepare_preprocessor(project, spec, sensitive_mode)
    ids = np.asarray(list(train_ids), dtype=np.int64)
    frame = s4.read_training_rows(source_path(project, sensitive_mode), ids, raw + [TARGET])
    X = frame.loc[ids, raw].copy()
    y = frame.loc[ids, TARGET].to_numpy(dtype=float)
    ready = preprocessor.fit_transform(X, s4.transform_target(y, target_mode))
    if not sparse.issparse(ready):
        raise ValueError("Full-Train XGBoost preprocessing unexpectedly became dense.")
    model = make_model(project, parameters, None)
    started = time.perf_counter()
    model.fit(ready, s4.transform_target(y, target_mode), verbose=False)
    fit_seconds = time.perf_counter() - started
    reference_ids = ids[: min(500, len(ids))]
    reference_ready = preprocessor.transform(frame.loc[reference_ids, raw].copy())
    reference_prediction = s4.inverse_target(model.predict(reference_ready), target_mode)
    return {
        "preprocessor": preprocessor,
        "model": model,
        "raw_columns": raw,
        "selected_features": selected,
        "categorical_features": categorical,
        "train_ids": ids,
        "fit_seconds": fit_seconds,
        "best_iteration": int(parameters["n_estimators"]),
        "feature_names": preprocessor.named_steps["preprocess"].get_feature_names_out().tolist(),
        "reference_ids": reference_ids,
        "reference_prediction": reference_prediction,
        "sparse_train_shape": list(ready.shape),
    }


def prediction_frame(result: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": result["validation_ids"],
        "y_true": result["y_validation"],
        "y_pred": result["validation_prediction"],
        "residual": result["validation_prediction"] - result["y_validation"],
        "absolute_error": np.abs(result["validation_prediction"] - result["y_validation"]),
        "stage_id": config["stage_id"],
        "experiment_id": config["experiment_id"],
        "sensitive_mode": config["sensitive_mode"],
        "target_mode": config["target_mode"],
    })


def bundle_metadata(root: str | Path, config: dict[str, Any], spec: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    project = Path(root).resolve()
    split_dir = project / "artifacts/splits"
    return {
        "bundle_version": VERSION,
        "stage_id": config["stage_id"],
        "experiment_id": config["experiment_id"],
        "feature_engineer_version": spec["version"],
        "feature_pack": spec["pack_id"],
        "feature_pack_definition": spec,
        "sensitive_mode": config["sensitive_mode"],
        "target_mode": config["target_mode"],
        "inverse_target_transformation": "identity" if config["target_mode"] == "raw" else "numpy.expm1",
        "parameters": config["parameters"],
        "fixed_or_best_iteration": int(result["best_iteration"]),
        "random_seed": SEED,
        "execution_mode": EXECUTION_MODE,
        "thread_count": THREAD_COUNT,
        "raw_columns": list(result["raw_columns"]),
        "feature_names": list(result["feature_names"]),
        "training_row_count": int(len(result["train_ids"])),
        "source_path": str(source_path(project, config["sensitive_mode"])),
        "source_sha256": s4.sha256_file(source_path(project, config["sensitive_mode"])),
        "sample_digest": sample_digest(project, config["sample_name"]),
        "feature_pack_digest": feature_pack_digest(spec),
        "split_hashes": {
            "train_row_ids": s4.sha256_file(split_dir / "train_row_ids.csv"),
            "test_row_ids": s4.sha256_file(split_dir / "test_row_ids.csv"),
            "cv_fold_assignments": s4.sha256_file(split_dir / "cv_fold_assignments.csv"),
            "split_config": s4.sha256_file(split_dir / "split_config.json"),
        },
        "environment": environment_metadata(project),
        "sparse_preprocessing": True,
        "test_row_count": 0,
    }


def atomic_native_save(model: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp" + destination.suffix)
    model.get_booster().save_model(str(temporary))
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise IOError("Native XGBoost temporary file is invalid.")
    os.replace(temporary, destination)


def validate_checkpoint(root: str | Path, checkpoint_path: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    path = Path(checkpoint_path)
    path = path if path.is_absolute() else project / path
    if not path.is_file():
        return {"valid": False, "reason": "missing_checkpoint", "path": str(path)}
    try:
        data = read_json(path)
    except Exception as exc:
        return {"valid": False, "reason": f"invalid_json: {exc}", "path": str(path)}
    if data.get("status") != "PASS":
        return {"valid": False, "reason": "status_not_pass", "path": str(path)}
    required = ("model_path", "native_model_path", "prediction_path")
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for key in required:
        value = data.get(key)
        target = Path(value) if value and Path(value).is_absolute() else project / str(value or "")
        if not value or not target.is_file() or target.stat().st_size <= 0:
            missing.append(key)
        else:
            resolved[key] = target
    mismatches: list[str] = []
    for key, hash_key in {"model_path": "model_sha256", "native_model_path": "native_model_sha256", "prediction_path": "prediction_sha256"}.items():
        if key in resolved and data.get(hash_key) != s4.sha256_file(resolved[key]):
            mismatches.append(hash_key)
    try:
        if data.get("sample_digest") != sample_digest(project, data["sample_name"]):
            mismatches.append("sample_digest")
        spec = feature_pack_spec(project, data["feature_pack"], data.get("selected_proposals", ()))
        if data.get("feature_pack_digest") != feature_pack_digest(spec):
            mismatches.append("feature_pack_digest")
        if data.get("source_hash_digest") != s4.sha256_file(source_path(project, data["sensitive_mode"])):
            mismatches.append("source_hash_digest")
    except Exception as exc:
        mismatches.append(f"provenance_validation_error: {exc}")
    valid = not missing and not mismatches
    return {"valid": valid, "reason": "ok" if valid else f"missing={missing}; mismatches={mismatches}", "path": str(path), "checkpoint": data}


def registry_row(checkpoint: dict[str, Any], evaluation_stage: str, notes: str = "") -> dict[str, Any]:
    metrics = checkpoint.get("metrics", {})
    return {
        "experiment_id": checkpoint["experiment_id"],
        "timestamp_utc": checkpoint.get("created_at_utc", s4.utc_now()),
        "model_family": "boosting",
        "model_name": "XGBoost",
        "sensitive_mode": checkpoint["sensitive_mode"],
        "feature_set": checkpoint["feature_pack"],
        "target_mode": checkpoint["target_mode"],
        "evaluation_stage": evaluation_stage,
        "fold_number": "",
        "training_row_count": checkpoint["training_row_count"],
        "validation_row_count": checkpoint["validation_row_count"],
        "test_row_count": 0,
        "parameter_json": s4.canonical_json(checkpoint["parameters"]),
        "mae": metrics.get("mae", ""),
        "mse": metrics.get("mse", ""),
        "rmse": metrics.get("rmse", ""),
        "mape_percent": metrics.get("mape_percent", ""),
        "r_squared": metrics.get("r_squared", ""),
        "rmsle": metrics.get("rmsle", ""),
        "rmsle_clipped_zero": metrics.get("rmsle_clipped_zero", ""),
        "median_absolute_error": metrics.get("median_absolute_error", ""),
        "wape_percent": metrics.get("wape_percent", ""),
        "mean_signed_error": metrics.get("mean_signed_error", ""),
        "p90_absolute_error": metrics.get("p90_absolute_error", ""),
        "negative_prediction_rate": metrics.get("negative_prediction_rate", ""),
        "fit_time_seconds": checkpoint.get("fit_time_seconds", ""),
        "prediction_time_seconds": checkpoint.get("prediction_time_seconds", ""),
        "status": checkpoint["status"],
        "notes": notes,
        "model_artifact_path": checkpoint["model_path"],
        "prediction_artifact_path": checkpoint["prediction_path"],
    }


def upsert_registry_preserve_prior(root: str | Path, rows: Sequence[dict[str, Any]], allowed_stage_ids: Sequence[str]) -> pd.DataFrame:
    project = Path(root).resolve()
    path = paths(project)["registry"]
    raw_bytes = path.read_bytes()
    newline = "\r\n" if b"\r\n" in raw_bytes else "\n"
    records = raw_bytes.decode("utf-8").splitlines()
    if not records:
        raise ValueError("Registry is empty.")
    header = next(csv.reader([records[0]]))
    if header != s4.REGISTRY_COLUMNS:
        raise ValueError("Registry schema does not match the fixed schema.")
    prefixes = tuple(f"{stage_id}__" for stage_id in allowed_stage_ids)
    owned_ids = {str(row["experiment_id"]) for row in rows}
    if any(not experiment_id.startswith(prefixes) for experiment_id in owned_ids):
        raise ValueError(f"Registry IDs must use one of these prefixes: {prefixes}")
    kept = [records[0]]
    for record in records[1:]:
        if record and next(csv.reader([record]))[0] not in owned_ids:
            kept.append(record)
    generated: list[str] = []
    for row in sorted(rows, key=lambda value: str(value["experiment_id"])):
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="").writerow([row.get(column, "") for column in header])
        generated.append(buffer.getvalue())
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((newline.join(kept + generated) + newline).encode("utf-8"))
    os.replace(temporary, path)
    frame = pd.read_csv(path)
    if frame["experiment_id"].duplicated().any():
        raise AssertionError("Registry Experiment IDs are not unique after upsert.")
    return frame


def native_predict(bundle: XGBoostModelBundle, native_path: str | Path, ready: Any) -> np.ndarray:
    s4.activate_local_packages(Path(bundle.metadata["source_path"]).parents[1])
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(native_path))
    matrix = xgb.DMatrix(ready)
    effective = int(bundle.metadata["fixed_or_best_iteration"])
    output = booster.predict(matrix, iteration_range=(0, effective))
    return s4.inverse_target(output, bundle.target_mode)


def clean_reload_check(root: str | Path, model_path: str | Path, native_path: str | Path, prediction_path: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    s4.activate_local_packages(project)
    bundle: XGBoostModelBundle = joblib.load(model_path)
    reference = pd.read_csv(prediction_path, dtype={"row_id": "int64"}).head(500)
    raw = list(bundle.metadata["raw_columns"])
    frame = s4.read_training_rows(bundle.metadata["source_path"], reference["row_id"].to_numpy(dtype=np.int64), raw)
    X = frame.loc[reference["row_id"].to_numpy(dtype=np.int64), raw].copy()
    before = frame_digest(X)
    prediction = bundle.predict(X)
    after = frame_digest(X)
    expected = reference["y_pred"].to_numpy(dtype=float)
    ready = bundle.transform(X)
    native_prediction = native_predict(bundle, native_path, ready)
    result = {
        "model_path": str(Path(model_path)),
        "native_model_path": str(Path(native_path)),
        "prediction_path": str(Path(prediction_path)),
        "rows": int(len(X)),
        "finite_predictions": bool(np.isfinite(prediction).all()),
        "output_length_matches": len(prediction) == len(X),
        "source_frame_unchanged": before == after,
        "reference_prediction_match": bool(np.allclose(prediction, expected, rtol=1e-7, atol=1e-7)),
        "native_prediction_match": bool(np.allclose(native_prediction, expected, rtol=1e-6, atol=1e-6)),
        "inverse_target_verified": bundle.target_mode in {"raw", "log1p"},
        "native_model_loaded": True,
        "sparse_transformation": bool(sparse.issparse(ready)),
        "sparse_shape": list(ready.shape),
    }
    excluded = {"model_path", "native_model_path", "prediction_path", "rows", "sparse_shape", "status"}
    result["status"] = "PASS" if all(value for key, value in result.items() if key not in excluded) else "FAIL"
    return result


def source_feature_name(encoded_name: str, spec: dict[str, Any], sensitive_mode: str) -> str:
    name = encoded_name.split("__", 1)[-1]
    candidates = list(spec.get("one_hot_categorical", ())) + list(spec.get("frequency_sources", ())) + list(spec.get("numeric", ()))
    if sensitive_mode == "with_sensitive":
        candidates += list(SENSITIVE_COLUMNS)
    for candidate in sorted(set(candidates), key=len, reverse=True):
        if name == candidate or name.startswith(candidate + "_") or name.startswith(candidate + "__"):
            return candidate
    return name


def importance_tables(bundle: XGBoostModelBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    booster = bundle.model.get_booster()
    feature_names = list(bundle.metadata["feature_names"])
    spec = bundle.metadata["feature_pack_definition"]
    rows = []
    score_maps = {kind: booster.get_score(importance_type=kind) for kind in ("gain", "weight", "total_gain")}
    for index, encoded in enumerate(feature_names):
        key = f"f{index}"
        rows.append({
            "encoded_feature": encoded,
            "source_feature": source_feature_name(encoded, spec, bundle.metadata["sensitive_mode"]),
            "gain": float(score_maps["gain"].get(key, 0.0)),
            "weight": float(score_maps["weight"].get(key, 0.0)),
            "total_gain": float(score_maps["total_gain"].get(key, 0.0)),
        })
    complete = pd.DataFrame(rows).sort_values(["gain", "weight"], ascending=False, kind="mergesort").reset_index(drop=True)
    aggregate = complete.groupby("source_feature", as_index=False)[["gain", "weight", "total_gain"]].sum().sort_values(["total_gain", "gain"], ascending=False, kind="mergesort").reset_index(drop=True)
    return complete, aggregate


def error_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy()
    frame["target_decile"] = pd.qcut(frame["y_true"], 10, labels=False, duplicates="drop")
    deciles = frame.groupby("target_decile", as_index=False).agg(
        rows=("row_id", "size"),
        target_min=("y_true", "min"),
        target_max=("y_true", "max"),
        mae=("absolute_error", "mean"),
        mean_signed_error=("residual", "mean"),
        underestimation_rate=("residual", lambda value: float((value < 0).mean())),
        overestimation_rate=("residual", lambda value: float((value > 0).mean())),
    )
    worst = frame.nlargest(20, "absolute_error").reset_index(drop=True)
    return deciles, worst


def recheck_protected_baseline(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    baseline = read_json(project / "artifacts/manifests/stage4/xgboost/stage4i_protected_hashes_before.json")
    mismatches: dict[str, dict[str, str]] = {}
    for key, expected in baseline["hashes"].items():
        path = Path(key) if Path(key).is_absolute() else project / key
        actual = s4.sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    return {"file_count": len(baseline["hashes"]), "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}


def run_config_parent(root: str | Path, config_path: str | Path, timeout_seconds: float, output_path: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    config = Path(config_path)
    config = config if config.is_absolute() else project / config
    command = [sys.executable, str(project / "stage4_xgboost_worker.py"), "--root", str(project), "--config", str(config)]
    result = s4.run_worker_process(command, timeout_seconds=float(timeout_seconds), cwd=project)
    output = Path(output_path)
    output = output if output.is_absolute() else project / output
    s4.atomic_write_json(output, result)
    return result


def _main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    reload_parser = sub.add_parser("reload")
    reload_parser.add_argument("--root", default=".")
    reload_parser.add_argument("--model", required=True)
    reload_parser.add_argument("--native", required=True)
    reload_parser.add_argument("--predictions", required=True)
    reload_parser.add_argument("--output", required=True)
    run_parser = sub.add_parser("run-config")
    run_parser.add_argument("--root", default=".")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--timeout", required=True, type=float)
    run_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "reload":
        result = clean_reload_check(args.root, args.model, args.native, args.predictions)
        s4.atomic_write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 1
    if args.command == "run-config":
        result = run_config_parent(args.root, args.config, args.timeout, args.output)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "success" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
