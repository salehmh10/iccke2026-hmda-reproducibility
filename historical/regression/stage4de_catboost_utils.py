"""Stage 4D–E CatBoost Feature confirmation and final-model helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import stage4_boosting_utils as s4
import stage4_catboost_utils as c4


STAGE_ID = "stage4de"
STAGE_NAME = "Stage 4D–E — CatBoost Feature Confirmation and Final Model"
VERSION = "stage4de_catboost_v1_20260714"
SEED = 42
THREAD_COUNT = 4
EXECUTION_MODE = "CPU"
TARGET = s4.TARGET_COLUMN

PROPOSAL_RATIO = "applicant_to_estimated_tract_income_ratio"
PROPOSAL_RESPONDENT_PURPOSE = "respondent_purpose_group"
PROPOSAL_INCOME_LIEN = "income_band_lien_status_group"
APPROVED_PROPOSALS = (PROPOSAL_RATIO, PROPOSAL_RESPONDENT_PURPOSE, PROPOSAL_INCOME_LIEN)
RATIO_ZERO_FLAG = "applicant_to_estimated_tract_income_denominator_zero"


def paths(root: str | Path) -> dict[str, Path]:
    project = Path(root).resolve()
    return {
        "root": project,
        "results": project / "artifacts/results/stage4/catboost/final",
        "confirmation": project / "artifacts/results/stage4/catboost/feature_confirmation",
        "predictions": project / "artifacts/predictions/catboost/final",
        "models": project / "artifacts/models/catboost/final",
        "candidate_models": project / "artifacts/models/catboost/stage4de_candidates",
        "features": project / "artifacts/features/stage4/catboost",
        "figures": project / "artifacts/figures/stage4/catboost/final",
        "checkpoints": project / "artifacts/checkpoints/stage4/catboost/stage4de",
        "manifests": project / "artifacts/manifests/stage4/catboost",
        "reports": project / "artifacts/reports",
        "registry": project / "artifacts/results/experiment_results.csv",
    }


def ensure_directories(root: str | Path) -> None:
    for name, path in paths(root).items():
        if name not in {"root", "registry"}:
            path.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_digest(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def recheck_protected(root: str | Path, save: bool = False) -> dict[str, Any]:
    project = Path(root).resolve()
    p = paths(project)
    before = read_json(p["manifests"] / "stage4de_protected_hashes_before.json")
    mismatches: dict[str, Any] = {}
    for name, expected in before["hashes"].items():
        path = Path(name)
        path = path if path.is_absolute() else project / path
        if not path.is_file():
            mismatches[name] = {"status": "missing"}
            continue
        actual = s4.sha256_file(path)
        size = path.stat().st_size
        if actual != expected or size != before["sizes"][name]:
            mismatches[name] = {"status": "changed", "expected_sha256": expected, "actual_sha256": actual, "expected_bytes": before["sizes"][name], "actual_bytes": size}
    result = {
        "stage": STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "file_count": before["file_count"],
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    if save:
        s4.atomic_write_json(p["manifests"] / "stage4de_protected_hashes_after.json", result)
    return result


class CatBoostFeatureEngineerV2(BaseEstimator, TransformerMixin):
    """Preserve Stage 4B fixed Features and add approved Stage 4D Features."""

    def __init__(
        self,
        selected_proposals: Sequence[str] = APPROVED_PROPOSALS,
        income_quantiles: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        epsilon: float = 1e-8,
        missing_token: str = "<MISSING>",
    ) -> None:
        self.selected_proposals = selected_proposals
        self.income_quantiles = income_quantiles
        self.epsilon = epsilon
        self.missing_token = missing_token

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CatBoostFeatureEngineerV2":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        selected = tuple(self.selected_proposals)
        unknown = sorted(set(selected).difference(APPROVED_PROPOSALS))
        if unknown:
            raise ValueError(f"Unknown Stage 4D proposals: {unknown}")
        required = {
            PROPOSAL_RATIO: {"applicant_income_000s", "hud_median_family_income", "tract_income_ratio"},
            PROPOSAL_RESPONDENT_PURPOSE: {"respondent_id", "loan_purpose_name"},
            PROPOSAL_INCOME_LIEN: {"applicant_income_000s", "lien_status_name"},
        }
        missing = sorted({name for proposal in selected for name in required[proposal]}.difference(X.columns))
        if missing:
            raise ValueError(f"Feature Engineer v2 sources are missing: {missing}")
        self.base_engineer_ = s4.Stage4FixedFeatureEngineer()
        self.base_engineer_.fit(X, y)
        self.selected_proposals_ = selected
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        if PROPOSAL_INCOME_LIEN in selected:
            income = pd.to_numeric(X["applicant_income_000s"], errors="coerce").to_numpy(dtype=float)
            finite = income[np.isfinite(income)]
            if not len(finite):
                raise ValueError("Income-band Feature requires finite training income values.")
            quantiles = np.asarray(tuple(self.income_quantiles), dtype=float)
            if quantiles[0] != 0.0 or quantiles[-1] != 1.0 or np.any(np.diff(quantiles) <= 0):
                raise ValueError("income_quantiles must increase from 0 to 1.")
            edges = np.unique(np.quantile(finite, quantiles))
            if len(edges) < 2:
                edges = np.asarray([finite.min(), finite.max() + 1.0], dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.income_band_edges_ = edges
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if not hasattr(self, "selected_proposals_"):
            raise RuntimeError("CatBoostFeatureEngineerV2 is not fitted.")
        missing = sorted(set(self.feature_names_in_).difference(X.columns))
        if missing:
            raise ValueError(f"Transform input is missing fitted columns: {missing}")
        result = self.base_engineer_.transform(X)
        if PROPOSAL_RATIO in self.selected_proposals_:
            applicant = pd.to_numeric(result["applicant_income_000s"], errors="coerce")
            denominator = pd.to_numeric(result["estimated_tract_family_income_000s"], errors="coerce")
            finite = np.isfinite(applicant) & np.isfinite(denominator)
            zero = finite & (denominator.abs() <= float(self.epsilon))
            safe_denominator = denominator.copy()
            safe_denominator.loc[zero] = float(self.epsilon)
            values = pd.Series(np.nan, index=result.index, dtype=float)
            values.loc[finite] = applicant.loc[finite] / safe_denominator.loc[finite]
            result[PROPOSAL_RATIO] = values.where(np.isfinite(values))
            result[RATIO_ZERO_FLAG] = zero.astype(float)
        if PROPOSAL_RESPONDENT_PURPOSE in self.selected_proposals_:
            left = result["respondent_id"].astype("string").fillna(self.missing_token)
            right = result["loan_purpose_name"].astype("string").fillna(self.missing_token)
            result[PROPOSAL_RESPONDENT_PURPOSE] = (left + " | " + right).astype(object)
        if PROPOSAL_INCOME_LIEN in self.selected_proposals_:
            income = pd.to_numeric(result["applicant_income_000s"], errors="coerce")
            labels = [f"income_band_{index}" for index in range(len(self.income_band_edges_) - 1)]
            bands = pd.cut(income, bins=self.income_band_edges_, labels=labels, include_lowest=True).astype("string").fillna(self.missing_token)
            lien = result["lien_status_name"].astype("string").fillna(self.missing_token)
            result[PROPOSAL_INCOME_LIEN] = (bands + " | " + lien).astype(object)
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
        if PROPOSAL_RESPONDENT_PURPOSE in self.selected_proposals_ and PROPOSAL_RESPONDENT_PURPOSE not in base:
            base.append(PROPOSAL_RESPONDENT_PURPOSE)
        if PROPOSAL_INCOME_LIEN in self.selected_proposals_ and PROPOSAL_INCOME_LIEN not in base:
            base.append(PROPOSAL_INCOME_LIEN)
        return np.asarray(base, dtype=object)


class CatBoostModelBundle:
    """Serializable final bundle that accepts raw DataFrames and returns original-scale predictions."""

    def __init__(self, pipeline: Pipeline, target_mode: str, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.target_mode = target_mode
        self.metadata = metadata

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        output = self.pipeline.predict(X)
        prediction = s4.inverse_target(output, self.target_mode)
        if not np.isfinite(prediction).all():
            raise ValueError("Bundle predictions are not finite.")
        return prediction


def proposal_review(root: str | Path) -> pd.DataFrame:
    project = Path(root).resolve()
    source = pd.read_csv(paths(project)["features"] / "catboost_round2_feature_candidates.csv")
    if len(source) > 3 or set(source["feature_name"]) != set(APPROVED_PROPOSALS):
        raise AssertionError("Stage 4D–E must review exactly the saved Stage 4C proposals.")
    source_columns = set(pd.read_csv(project / "data/regression_without_sensitive_features.csv", nrows=0).columns)
    rows = []
    for _, row in source.iterrows():
        sources = str(row["source_columns"]).split("|")
        target_derived = bool(row["target_derived"])
        sensitive_derived = bool(row["sensitive_derived"])
        stable = row["feature_name"] != PROPOSAL_RATIO or "epsilon" in str(row["zero_denominator_handling"]).lower()
        available = set(sources).issubset(source_columns)
        approved = not target_derived and not sensitive_derived and stable and available and str(row["leakage_review"]).startswith("PASS")
        result = row.to_dict()
        result.update({
            "source_columns_available": available,
            "formula_stable": stable,
            "approved_for_combined_confirmation": approved,
            "review_reason": "Approved for independent combined confirmation." if approved else "Rejected by target, sensitive, stability, availability, or leakage rule.",
        })
        rows.append(result)
    review = pd.DataFrame(rows)
    s4.atomic_write_csv(review, paths(project)["features"] / "catboost_round2_proposal_review.csv")
    return review


def pack_spec(root: str | Path, pack_id: str, proposals: Sequence[str] | None = None) -> dict[str, Any]:
    project = Path(root).resolve()
    original = c4.load_feature_pack(project, "catboost_native_v1")
    selected = tuple(proposals or ())
    if pack_id == "catboost_native_v1":
        selected = ()
    elif not selected:
        raise ValueError("A Stage 4D v2 pack requires selected proposals.")
    numeric = list(original["numeric"])
    categorical = list(original["categorical"])
    if PROPOSAL_RATIO in selected:
        numeric.extend([PROPOSAL_RATIO, RATIO_ZERO_FLAG])
    if PROPOSAL_RESPONDENT_PURPOSE in selected:
        categorical.append(PROPOSAL_RESPONDENT_PURPOSE)
    if PROPOSAL_INCOME_LIEN in selected:
        categorical.append(PROPOSAL_INCOME_LIEN)
    return {
        "pack_id": pack_id,
        "version": VERSION,
        "base_pack": "catboost_native_v1",
        "raw": list(original["raw"]),
        "numeric": numeric,
        "categorical": categorical,
        "selected_proposals": list(selected),
        "model_features": numeric + categorical,
        "target_derived": False,
        "sensitive_derived": False,
    }


def prepare_pipeline(root: str | Path, spec: dict[str, Any], sensitive_mode: str, parameters: dict[str, Any]) -> tuple[Pipeline, list[str], list[str], list[str]]:
    project = Path(root).resolve()
    numeric = list(spec["numeric"])
    categorical = list(spec["categorical"])
    raw = list(spec["raw"])
    if sensitive_mode == "with_sensitive":
        numeric.extend(name for name in c4.SENSITIVE_NUMERIC if name not in numeric)
        categorical.extend(name for name in c4.SENSITIVE_CATEGORICAL if name not in categorical)
        raw.extend(name for name in c4.SENSITIVE_COLUMNS if name not in raw)
    elif sensitive_mode != "without_sensitive":
        raise ValueError(f"Unknown sensitive mode: {sensitive_mode}")
    selected = numeric + categorical
    proposals = tuple(spec.get("selected_proposals", ()))
    engineer: Any = CatBoostFeatureEngineerV2(proposals) if proposals else s4.Stage4FixedFeatureEngineer()
    steps: list[tuple[str, Any]] = [
        ("features", engineer),
        ("select", s4.Stage4ColumnSelector(tuple(selected))),
        ("sanitize", s4.Stage4CategoricalSanitizer(tuple(categorical))),
    ]
    rare = [name for name in ("respondent_id", "msamd_name", "county_name", "census_tract_number", PROPOSAL_RESPONDENT_PURPOSE) if name in categorical]
    if rare:
        steps.append(("rare", s4.Stage4RareCategoryGrouper(tuple(rare), min_count=2)))
    s4.activate_local_packages(project)
    from catboost import CatBoostRegressor
    model_parameters = dict(parameters)
    model_parameters.pop("early_stopping_rounds", None)
    model = CatBoostRegressor(cat_features=categorical, **model_parameters)
    return Pipeline(steps + [("model", model)]), raw, selected, categorical


def sample_ids(root: str | Path, sample_name: str) -> tuple[np.ndarray, np.ndarray]:
    project = Path(root).resolve()
    filenames = {
        "feature_confirmation": "stage4_feature_confirmation_sample.csv",
        "final_selection": "stage4_final_selection_sample.csv",
    }
    if sample_name not in filenames:
        raise ValueError(f"Unknown sample: {sample_name}")
    manifest = pd.read_csv(project / "artifacts/splits/stage4" / filenames[sample_name], dtype={"row_id": "int64"})
    train = manifest.loc[manifest["sample_role"].eq("train"), "row_id"].to_numpy(dtype=np.int64)
    validation = manifest.loc[manifest["sample_role"].eq("validation"), "row_id"].to_numpy(dtype=np.int64)
    return train, validation


def source_path(root: str | Path, sensitive_mode: str) -> Path:
    return c4.source_path(root, sensitive_mode)


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
    pipeline, raw, selected, categorical = prepare_pipeline(project, spec, sensitive_mode, parameters)
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
    preprocess = Pipeline(pipeline.steps[:-1])
    train_ready = preprocess.fit_transform(X_train, y_fit)
    validation_ready = preprocess.transform(X_validation)
    model = pipeline.named_steps["model"]
    fit_kwargs: dict[str, Any] = {"eval_set": (validation_ready, y_eval), "use_best_model": early_stopping_rounds is not None}
    if early_stopping_rounds is not None:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    started = time.perf_counter()
    model.fit(train_ready, y_fit, **fit_kwargs)
    fit_seconds = time.perf_counter() - started
    fitted = Pipeline(preprocess.steps + [("model", model)])
    prediction_started = time.perf_counter()
    prediction = s4.inverse_target(fitted.predict(X_validation), target_mode)
    prediction_seconds = time.perf_counter() - prediction_started
    metrics = c4.extended_metrics(y_validation, prediction)
    return {
        "pipeline": fitted, "raw_columns": raw, "selected_features": selected, "categorical_features": categorical,
        "train_ids": train_array, "validation_ids": validation_array, "y_validation": y_validation,
        "validation_prediction": prediction, "metrics": metrics,
        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
        "best_iteration_zero_based": int(model.get_best_iteration()) if early_stopping_rounds is not None else int(parameters["iterations"]) - 1,
        "tree_count": int(model.tree_count_),
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
    pipeline, raw, selected, categorical = prepare_pipeline(project, spec, sensitive_mode, parameters)
    ids = np.asarray(list(train_ids), dtype=np.int64)
    frame = s4.read_training_rows(source_path(project, sensitive_mode), ids, raw + [TARGET])
    X = frame.loc[ids, raw].copy()
    y = frame.loc[ids, TARGET].to_numpy(dtype=float)
    y_fit = s4.transform_target(y, target_mode)
    started = time.perf_counter()
    pipeline.fit(X, y_fit)
    fit_seconds = time.perf_counter() - started
    return {
        "pipeline": pipeline, "raw_columns": raw, "selected_features": selected, "categorical_features": categorical,
        "train_ids": ids, "fit_seconds": fit_seconds, "tree_count": int(pipeline.named_steps["model"].tree_count_),
    }


def prediction_frame(result: dict[str, Any], experiment_id: str, sensitive_mode: str, target_mode: str) -> pd.DataFrame:
    pred = result["validation_prediction"]
    true = result["y_validation"]
    return pd.DataFrame({
        "row_id": result["validation_ids"], "y_true": true, "y_pred": pred,
        "residual": pred - true, "absolute_error": np.abs(pred - true),
        "sensitive_mode": sensitive_mode, "target_mode": target_mode, "experiment_id": experiment_id,
    })


def environment_metadata(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    s4.activate_local_packages(project)
    import catboost
    import joblib
    import sklearn
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "catboost": catboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def bundle_provenance(root: str | Path, sensitive_mode: str, target_mode: str, raw_columns: Sequence[str]) -> dict[str, Any]:
    """Return explicit, serializable provenance stored inside each final bundle."""
    project = Path(root).resolve()
    split_dir = project / "artifacts/splits"
    return {
        "environment": environment_metadata(project),
        "source_file_sha256": s4.sha256_file(source_path(project, sensitive_mode)),
        "split_hashes": {
            "train_row_ids_sha256": s4.sha256_file(split_dir / "train_row_ids.csv"),
            "test_row_ids_sha256": s4.sha256_file(split_dir / "test_row_ids.csv"),
            "cv_fold_assignments_sha256": s4.sha256_file(split_dir / "cv_fold_assignments.csv"),
            "split_config_sha256": s4.sha256_file(split_dir / "split_config.json"),
        },
        "target_transform_contract": {
            "fit_space": target_mode,
            "prediction_inverse": "numpy.expm1" if target_mode == "log1p" else "identity",
            "bundle_prediction_space": "original_target",
        },
        "input_column_contract": {
            "required_raw_columns": list(raw_columns),
            "selection_is_name_based": True,
            "reordered_columns_supported": True,
            "extra_columns_ignored": True,
        },
    }
