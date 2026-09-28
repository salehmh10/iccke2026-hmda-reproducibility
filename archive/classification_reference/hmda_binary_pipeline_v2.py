"""Leakage-safe utilities for HMDA binary mortgage classification.

The module intentionally keeps the locked test set out of model-selection APIs.
All data-dependent transformations are fitted on training data only, while
calibration and threshold selection use dedicated holdouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.neural_network import MLPClassifier

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier


RANDOM_STATE = 42
TARGET_COL = "loan_approved"
DATA_FILE = Path("hmda_classification_stratified_500k.csv")

REQUIRED_COLUMNS = {
    "respondent_id",
    "agency_name",
    "loan_type_name",
    "property_type_name",
    "loan_purpose_name",
    "owner_occupancy_name",
    "loan_amount_000s",
    "preapproval_name",
    "msamd_name",
    "state_name",
    "state_code",
    "county_name",
    "county_code",
    "census_tract_number",
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "applicant_income_000s",
    "lien_status_name",
    "population",
    "minority_population",
    "hud_median_family_income",
    "tract_to_msamd_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
    TARGET_COL,
}

# These fields either disclose the decision or only exist after the decision.
OUTCOME_DERIVED_COLUMNS = {
    "action_taken",
    "action_taken_name",
    "denial_reason_1",
    "denial_reason_2",
    "denial_reason_3",
    "denial_reason_name_1",
    "denial_reason_name_2",
    "denial_reason_name_3",
    "purchaser_type",
    "purchaser_type_name",
}

SENSITIVE_COLUMNS = {
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
}

# Lender identity and tract/county identifiers are high-cardinality shortcuts.
# State and MSA labels plus tract socioeconomic measurements remain available.
IDENTIFIER_COLUMNS = {"respondent_id"}
REDUNDANT_GEO_COLUMNS = {
    "state_code",
    "county_name",
    "county_code",
    "census_tract_number",
}


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = RANDOM_STATE
    split_mode: str = "stratified_random"
    group_col: str = "respondent_id"
    quick_mode: bool = True
    quick_train_rows: int = 100_000
    quick_validation_rows: int = 25_000
    quick_calibration_rows: int = 25_000
    quick_threshold_rows: int = 15_000
    quick_test_rows: int = 30_000
    min_category_frequency: int = 100
    selection_repeats: int = 4
    selection_min_features: int = 10
    selection_max_features: int = 24
    selection_tolerance: float = 0.002


@dataclass(frozen=True)
class FeatureSchema:
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]
    dropped_cols: list[str]


class LockedHoldout:
    """Prevent accidental test access before the model contract is frozen."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()
        self._unlocked = False
        self._reason: str | None = None

    @property
    def is_unlocked(self) -> bool:
        return self._unlocked

    def unlock(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("A non-empty model-lock reason is required.")
        self._reason = reason.strip()
        self._unlocked = True

    def get(self) -> pd.DataFrame:
        if not self._unlocked:
            raise RuntimeError(
                "The test set is locked. Select features, model, calibration, "
                "and threshold before calling unlock()."
            )
        return self._frame.copy()


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns using quantiles learned from training only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> "QuantileClipper":
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


class PlattCalibrator:
    """Independent sigmoid calibration on model probabilities."""

    def __init__(self, random_state: int = RANDOM_STATE) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", random_state=random_state)

    @staticmethod
    def _logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    def fit(self, probability: np.ndarray, y: Sequence[int]) -> "PlattCalibrator":
        self.model.fit(self._logit(probability), np.asarray(y, dtype=int))
        return self

    def predict(self, probability: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._logit(probability))[:, 1]


class NativeCategoryEncoder:
    """Keep LightGBM category codes identical across every data split."""

    def fit(self, X: pd.DataFrame, categorical_cols: Sequence[str]) -> "NativeCategoryEncoder":
        self.categorical_cols_ = list(categorical_cols)
        self.categories_ = {
            col: pd.Index(X[col].astype("string").fillna("Missing").unique()).sort_values().tolist()
            for col in self.categorical_cols_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = X.copy()
        for col in self.categorical_cols_:
            values = output[col].astype("string").fillna("Missing")
            output[col] = pd.Categorical(values, categories=self.categories_[col])
        return output


class LightGBMBundle:
    """Prediction wrapper that applies the train-fitted category vocabulary."""

    def __init__(self, model: LGBMClassifier, encoder: NativeCategoryEncoder) -> None:
        self.model = model
        self.encoder = encoder
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.encoder.transform(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.encoder.transform(X))


def fit_outlier_bounds(
    X_train: pd.DataFrame,
    numeric_cols: Sequence[str],
    *,
    lower_multiplier: float = 3.0,
    upper_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Fit robust lower/upper fences on train only.

    The asymmetric policy is deliberate: low-tail observations are candidates
    for removal, while high-tail observations are retained for a tail-aware
    model. Bounds are never re-estimated on validation or test.
    """
    rows: list[dict[str, Any]] = []
    for col in numeric_cols:
        values = pd.to_numeric(X_train[col], errors="coerce")
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = max(q3 - q1, 1e-12)
        rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": q1 - lower_multiplier * iqr,
                "upper_bound": q3 + upper_multiplier * iqr,
                "lower_multiplier": lower_multiplier,
                "upper_multiplier": upper_multiplier,
            }
        )
    return pd.DataFrame(rows)


def apply_outlier_features(
    X: pd.DataFrame,
    bounds: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Add train-fitted outlier counts and return row-level lower/upper masks."""
    output = X.copy()
    lower_mask = np.zeros(len(output), dtype=bool)
    upper_mask = np.zeros(len(output), dtype=bool)
    lower_count = np.zeros(len(output), dtype=np.int16)
    upper_count = np.zeros(len(output), dtype=np.int16)
    for row in bounds.itertuples(index=False):
        values = pd.to_numeric(output[row.feature], errors="coerce").to_numpy(dtype=float)
        lower = np.nan_to_num(values < row.lower_bound, nan=False)
        upper = np.nan_to_num(values > row.upper_bound, nan=False)
        lower_mask |= lower
        upper_mask |= upper
        lower_count += lower.astype(np.int16)
        upper_count += upper.astype(np.int16)
    output["lower_outlier_count"] = lower_count
    output["upper_outlier_count"] = upper_count
    output["has_upper_outlier"] = upper_mask.astype("int8")
    return output, lower_mask, upper_mask


def outlier_report(
    X: pd.DataFrame,
    bounds: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize outlier counts without fitting on the inspected frame."""
    rows: list[dict[str, Any]] = []
    for row in bounds.itertuples(index=False):
        values = pd.to_numeric(X[row.feature], errors="coerce")
        rows.append(
            {
                "feature": row.feature,
                "lower_bound": row.lower_bound,
                "upper_bound": row.upper_bound,
                "lower_outlier_n": int((values < row.lower_bound).sum()),
                "upper_outlier_n": int((values > row.upper_bound).sum()),
                "lower_outlier_pct": float((values < row.lower_bound).mean()),
                "upper_outlier_pct": float((values > row.upper_bound).mean()),
            }
        )
    return pd.DataFrame(rows)


def add_outlier_columns_to_schema(schema: FeatureSchema) -> FeatureSchema:
    added = ["lower_outlier_count", "upper_outlier_count", "has_upper_outlier"]
    return FeatureSchema(
        feature_cols=schema.feature_cols + added,
        categorical_cols=list(schema.categorical_cols),
        numeric_cols=schema.numeric_cols + added,
        dropped_cols=list(schema.dropped_cols),
    )


def add_random_shadow_features(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
    schema: FeatureSchema,
    *,
    random_state: int = RANDOM_STATE,
    n_numeric_shadows: int = 4,
    n_categorical_shadows: int = 2,
    n_random_projections: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, FeatureSchema, list[str]]:
    """Add train-fitted random negative controls and random projections.

    Shadow columns are deliberately marked and are not production features by
    default. They help reveal whether a loop is overfitting noise. Numeric and
    categorical distributions for the other split are sampled only from train.
    Random projections use train-fitted location/scale and fixed coefficients.
    """
    rng = np.random.default_rng(random_state)
    train_out = X_train.copy()
    other_out = X_other.copy()
    random_columns: list[str] = []

    for index in range(n_numeric_shadows):
        source = schema.numeric_cols[index % len(schema.numeric_cols)]
        train_values = pd.to_numeric(train_out[source], errors="coerce").fillna(
            pd.to_numeric(train_out[source], errors="coerce").median()
        )
        name = f"shadow_random_numeric_{index + 1}"
        train_out[name] = rng.permutation(train_values.to_numpy())
        other_out[name] = rng.choice(train_values.to_numpy(), size=len(other_out), replace=True)
        random_columns.append(name)

    categorical_values = [col for col in schema.categorical_cols if col in X_train.columns]
    for index in range(n_categorical_shadows):
        source = categorical_values[index % len(categorical_values)]
        values = X_train[source].astype("string").fillna("Missing")
        name = f"shadow_random_category_{index + 1}"
        train_out[name] = rng.permutation(values.to_numpy())
        other_out[name] = rng.choice(values.to_numpy(), size=len(other_out), replace=True)
        random_columns.append(name)

    if n_random_projections:
        projection_sources = list(schema.numeric_cols)
        train_numeric = train_out[projection_sources].apply(pd.to_numeric, errors="coerce")
        other_numeric = other_out[projection_sources].apply(pd.to_numeric, errors="coerce")
        medians = train_numeric.median()
        scales = train_numeric.std(ddof=0).replace(0, 1.0).fillna(1.0)
        train_z = ((train_numeric.fillna(medians) - medians) / scales).to_numpy()
        other_z = ((other_numeric.fillna(medians) - medians) / scales).to_numpy()
        for index in range(n_random_projections):
            coefficients = rng.normal(0.0, 1.0, size=len(projection_sources))
            coefficients /= max(np.linalg.norm(coefficients), 1e-12)
            name = f"random_projection_{index + 1}"
            train_out[name] = train_z @ coefficients
            other_out[name] = other_z @ coefficients
            random_columns.append(name)

    extended_schema = FeatureSchema(
        feature_cols=schema.feature_cols + random_columns,
        categorical_cols=schema.categorical_cols + [
            col for col in random_columns if col.startswith("shadow_random_category_")
        ],
        numeric_cols=schema.numeric_cols + [
            col for col in random_columns if not col.startswith("shadow_random_category_")
        ],
        dropped_cols=list(schema.dropped_cols),
    )
    return train_out, other_out, extended_schema, random_columns


def build_mlp_classifier(
    schema: FeatureSchema,
    *,
    architecture: str = "deep",
    min_frequency: int = 100,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build sparse-safe multi-layer perceptrons for tabular data."""
    architectures = {
        "compact": (64, 32),
        "deep": (128, 64, 32),
        "wide_deep": (256, 128, 64, 32),
    }
    if architecture not in architectures:
        raise ValueError(f"Unknown MLP architecture: {architecture}")
    return Pipeline(
        steps=[
            ("preprocess", build_linear_preprocessor(schema, min_frequency)),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=architectures[architecture],
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=512,
                    learning_rate_init=1e-3,
                    max_iter=150,
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=15,
                    random_state=random_state,
                    verbose=False,
                ),
            ),
        ]
    )


def run_outlier_deep_experiment(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    schema: FeatureSchema,
    config: ExperimentConfig,
    *,
    loop_seeds: Sequence[int] = (101, 202, 303, 404, 505),
    lower_multiplier: float = 3.0,
    upper_multiplier: float = 1.5,
    upper_weight: float = 2.0,
) -> dict[str, Any]:
    """Run five outlier-policy repeats and compare robust/deep model families.

    The function only receives train and validation frames. It has no test-set
    argument by design, so the advanced search cannot accidentally consume the
    locked test holdout.
    """
    X_train = prepare_model_frame(train_frame, schema)
    X_validation = prepare_model_frame(validation_frame, schema)
    _, _, random_schema, random_columns = add_random_shadow_features(
        X_train,
        X_validation,
        schema,
        random_state=config.random_state,
    )
    y_train = train_frame[TARGET_COL].astype(int).to_numpy()
    y_validation = validation_frame[TARGET_COL].astype(int).to_numpy()
    bounds = fit_outlier_bounds(
        X_train,
        schema.numeric_cols,
        lower_multiplier=lower_multiplier,
        upper_multiplier=upper_multiplier,
    )
    X_train_outlier, lower_train, upper_train = apply_outlier_features(X_train, bounds)
    X_validation_outlier, _, _ = apply_outlier_features(X_validation, bounds)
    outlier_schema = add_outlier_columns_to_schema(schema)
    random_outlier_schema = add_outlier_columns_to_schema(random_schema)
    iterations = 250 if config.quick_mode else 900

    def make_model(seed: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=iterations,
            depth=7,
            learning_rate=0.04,
            l2_leaf_reg=8.0,
            random_seed=seed,
            random_strength=1.0,
            od_type="Iter",
            od_wait=40,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )

    loop_source = (
        stratified_cap(train_frame, 60_000, config.random_state)
        if config.quick_mode
        else train_frame
    )
    loop_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for loop_id, seed in enumerate(loop_seeds, start=1):
        loop_train, loop_eval = train_test_split(
            loop_source,
            test_size=0.20,
            stratify=loop_source[TARGET_COL],
            random_state=seed,
        )
        loop_train_base = prepare_model_frame(loop_train, schema)
        loop_eval_base = prepare_model_frame(loop_eval, schema)
        loop_train_base, loop_eval_base, loop_random_schema, _ = add_random_shadow_features(
            loop_train_base,
            loop_eval_base,
            schema,
            random_state=seed,
        )
        loop_bounds = fit_outlier_bounds(
            loop_train_base,
            schema.numeric_cols,
            lower_multiplier=lower_multiplier,
            upper_multiplier=upper_multiplier,
        )
        loop_train_X, loop_lower, loop_upper = apply_outlier_features(
            loop_train_base, loop_bounds
        )
        loop_eval_X, _, _ = apply_outlier_features(loop_eval_base, loop_bounds)
        loop_outlier_schema = add_outlier_columns_to_schema(loop_random_schema)
        loop_y = loop_train[TARGET_COL].astype(int).to_numpy()
        loop_eval_y = loop_eval[TARGET_COL].astype(int).to_numpy()
        cat_indices = [loop_train_X.columns.get_loc(col) for col in loop_outlier_schema.categorical_cols]

        for variant in ("core_lower_removed", "tail_aware_upper_weighted"):
            if variant == "core_lower_removed":
                fit_X = loop_train_X.loc[~loop_lower].reset_index(drop=True)
                fit_y = loop_y[~loop_lower]
                fit_weight = None
            else:
                fit_X = loop_train_X
                fit_y = loop_y
                fit_weight = np.where(loop_upper, upper_weight, 1.0)

            model = make_model(seed + loop_id)
            fit_pool = Pool(fit_X, fit_y, cat_features=cat_indices, weight=fit_weight)
            eval_pool = Pool(loop_eval_X, loop_eval_y, cat_features=cat_indices)
            model.fit(fit_pool, eval_set=eval_pool, use_best_model=True)
            probability = model.predict_proba(loop_eval_X)[:, 1]
            metrics = classification_metrics(loop_eval_y, probability, 0.5)
            loop_rows.append(
                {
                    "loop": loop_id,
                    "seed": seed,
                    "variant": variant,
                    "lower_removed_n": int(loop_lower.sum()),
                    "upper_weighted_n": int(loop_upper.sum()),
                    **metrics,
                }
            )
            for feature, importance in zip(loop_outlier_schema.feature_cols, model.get_feature_importance()):
                importance_rows.append(
                    {
                        "loop": loop_id,
                        "variant": variant,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

    loop_results = pd.DataFrame(loop_rows)
    loop_summary = (
        loop_results.groupby("variant", as_index=False)
        .agg(
            mean_denial_pr_auc=("denial_pr_auc", "mean"),
            std_denial_pr_auc=("denial_pr_auc", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_denial_recall=("denial_recall", "mean"),
        )
        .sort_values(["mean_denial_pr_auc", "mean_roc_auc"], ascending=False)
    )

    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    final_cat_indices = [
        X_train_outlier.columns.get_loc(col) for col in outlier_schema.categorical_cols
    ]
    for variant in ("core_lower_removed", "tail_aware_upper_weighted"):
        if variant == "core_lower_removed":
            fit_X = X_train_outlier.loc[~lower_train].reset_index(drop=True)
            fit_y = y_train[~lower_train]
            fit_weight = None
        else:
            fit_X = X_train_outlier
            fit_y = y_train
            fit_weight = np.where(upper_train, upper_weight, 1.0)
        model = make_model(config.random_state + (1 if variant.startswith("core") else 2))
        model.fit(
            Pool(fit_X, fit_y, cat_features=final_cat_indices, weight=fit_weight),
            eval_set=Pool(X_validation_outlier, y_validation, cat_features=final_cat_indices),
            use_best_model=True,
        )
        probability = model.predict_proba(X_validation_outlier)[:, 1]
        model_name = f"CatBoost {variant}"
        models[model_name] = model
        rows.append({"model": model_name, "family": "outlier-aware CatBoost", **classification_metrics(y_validation, probability, 0.5)})
        for feature, importance in zip(outlier_schema.feature_cols, model.get_feature_importance()):
            importance_rows.append(
                {"loop": "final", "variant": variant, "feature": feature, "importance": float(importance)}
            )

    architectures = ("compact", "deep") if config.quick_mode else ("compact", "deep", "wide_deep")
    for architecture in architectures:
        model = build_mlp_classifier(
            outlier_schema,
            architecture=architecture,
            min_frequency=config.min_category_frequency,
            random_state=config.random_state + len(models),
        )
        model.fit(X_train_outlier, y_train)
        probability = model.predict_proba(X_validation_outlier)[:, 1]
        model_name = f"MLP {architecture}"
        models[model_name] = model
        rows.append({"model": model_name, "family": "deep MLP", **classification_metrics(y_validation, probability, 0.5)})

    model_results = pd.DataFrame(rows).sort_values(
        ["denial_pr_auc", "roc_auc", "brier_score"], ascending=[False, False, True]
    ).reset_index(drop=True)
    importance_frame = pd.DataFrame(importance_rows)
    stable_features = (
        importance_frame.groupby("feature", as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            appearances=("importance", "count"),
        )
    )
    stable_features["stability_lower_bound"] = (
        stable_features["mean_importance"] - stable_features["std_importance"].fillna(0)
    )
    stable_features = stable_features.sort_values(
        ["stability_lower_bound", "mean_importance"], ascending=False
    ).reset_index(drop=True)
    stable_features["is_random_control"] = stable_features["feature"].str.startswith(
        ("shadow_random_", "random_projection_")
    )
    production_features = stable_features.loc[
        ~stable_features["is_random_control"], "feature"
    ].tolist()

    return {
        "bounds": bounds,
        "outlier_report": outlier_report(X_train, bounds),
        "lower_train_mask": lower_train,
        "upper_train_mask": upper_train,
        "outlier_schema": outlier_schema,
        "random_outlier_schema": random_outlier_schema,
        "random_columns": random_columns,
        "production_features": production_features,
        "loop_results": loop_results,
        "loop_summary": loop_summary,
        "models": models,
        "model_results": model_results,
        "importance": importance_frame,
        "stable_features": stable_features,
    }


def load_primary_data(path: Path = DATA_FILE, nrows: int | None = None) -> pd.DataFrame:
    dtype = {
        "respondent_id": "string",
        "state_code": "string",
        "county_code": "string",
        "census_tract_number": "string",
    }
    frame = pd.read_csv(path, dtype=dtype, low_memory=False, nrows=nrows)
    validate_schema(frame)
    return frame


def validate_schema(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    leaked = sorted((OUTCOME_DERIVED_COLUMNS & set(frame.columns)) - {TARGET_COL})
    if leaked:
        raise ValueError(f"Outcome-derived leakage columns are present: {leaked}")

    target_values = set(frame[TARGET_COL].dropna().astype(int).unique())
    if target_values != {0, 1}:
        raise ValueError(f"{TARGET_COL} must contain exactly 0 and 1; got {target_values}")


def clean_before_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    validate_schema(frame)
    duplicate_rows = int(frame.duplicated().sum())
    cleaned = frame.drop_duplicates().reset_index(drop=True)
    report = {
        "input_rows": int(len(frame)),
        "duplicate_rows_removed": duplicate_rows,
        "output_rows": int(len(cleaned)),
    }
    return cleaned, report


def profile_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in frame.columns:
        series = frame[col]
        item: dict[str, Any] = {
            "column": col,
            "dtype": str(series.dtype),
            "missing_n": int(series.isna().sum()),
            "missing_pct": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=False)),
        }
        if pd.api.types.is_numeric_dtype(series):
            item.update(
                min=float(series.min()),
                median=float(series.median()),
                max=float(series.max()),
            )
        else:
            item.update(min=np.nan, median=np.nan, max=np.nan)
        rows.append(item)
    return pd.DataFrame(rows)


def _split_random(frame: pd.DataFrame, random_state: int) -> dict[str, pd.DataFrame]:
    remaining, test = train_test_split(
        frame,
        test_size=0.10,
        stratify=frame[TARGET_COL],
        random_state=random_state,
    )
    remaining, threshold = train_test_split(
        remaining,
        test_size=0.05 / 0.90,
        stratify=remaining[TARGET_COL],
        random_state=random_state + 1,
    )
    remaining, calibration = train_test_split(
        remaining,
        test_size=0.10 / 0.85,
        stratify=remaining[TARGET_COL],
        random_state=random_state + 2,
    )
    train, validation = train_test_split(
        remaining,
        test_size=0.15 / 0.75,
        stratify=remaining[TARGET_COL],
        random_state=random_state + 3,
    )
    return {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "calibration": calibration.reset_index(drop=True),
        "threshold": threshold.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def _split_grouped(
    frame: pd.DataFrame,
    group_col: str,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    if group_col not in frame.columns:
        raise ValueError(f"Group column {group_col!r} is absent.")
    if frame[group_col].nunique(dropna=False) < 20:
        raise ValueError("At least 20 groups are required for the five-way grouped split.")

    groups = frame[group_col].fillna("Missing").astype(str)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=random_state)
    fold = np.full(len(frame), -1, dtype=np.int8)
    for fold_id, (_, heldout_idx) in enumerate(splitter.split(frame, frame[TARGET_COL], groups)):
        fold[heldout_idx] = fold_id
    if (fold < 0).any():
        raise AssertionError("Some rows were not assigned to a group fold.")

    fold_sets = {
        "train": set(range(0, 12)),
        "validation": set(range(12, 15)),
        "calibration": {15, 16},
        "threshold": {17},
        "test": {18, 19},
    }
    return {
        name: frame[np.isin(fold, list(ids))].reset_index(drop=True)
        for name, ids in fold_sets.items()
    }


def make_five_way_splits(frame: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    if config.split_mode == "stratified_random":
        splits = _split_random(frame, config.random_state)
    elif config.split_mode == "stratified_group":
        splits = _split_grouped(frame, config.group_col, config.random_state)
    else:
        raise ValueError("split_mode must be 'stratified_random' or 'stratified_group'.")
    assert_split_integrity(splits, config.group_col if config.split_mode == "stratified_group" else None)
    return splits


def assert_split_integrity(
    splits: Mapping[str, pd.DataFrame],
    group_col: str | None = None,
) -> None:
    expected = {"train", "validation", "calibration", "threshold", "test"}
    if set(splits) != expected:
        raise AssertionError(f"Expected split names {sorted(expected)}, got {sorted(splits)}")

    hashes: dict[str, set[int]] = {}
    for name, frame in splits.items():
        if frame.empty or frame[TARGET_COL].nunique() != 2:
            raise AssertionError(f"Split {name!r} is empty or lacks one target class.")
        hashes[name] = set(pd.util.hash_pandas_object(frame, index=False).astype("uint64").tolist())

    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = hashes[left].intersection(hashes[right])
            if overlap:
                raise AssertionError(f"Row overlap between {left} and {right}: {len(overlap)}")
            if group_col is not None:
                left_groups = set(splits[left][group_col].fillna("Missing").astype(str))
                right_groups = set(splits[right][group_col].fillna("Missing").astype(str))
                group_overlap = left_groups.intersection(right_groups)
                if group_overlap:
                    raise AssertionError(
                        f"Group overlap between {left} and {right}: {len(group_overlap)}"
                    )


def split_summary(splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(frame),
                "approved_n": int(frame[TARGET_COL].sum()),
                "approval_rate": float(frame[TARGET_COL].mean()),
                "denial_rate": float(1 - frame[TARGET_COL].mean()),
            }
            for name, frame in splits.items()
        ]
    )


def stratified_cap(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    sampled, _ = train_test_split(
        frame,
        train_size=max_rows,
        stratify=frame[TARGET_COL],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def apply_quick_caps(
    splits: Mapping[str, pd.DataFrame],
    config: ExperimentConfig,
) -> dict[str, pd.DataFrame]:
    if not config.quick_mode:
        return {name: frame.copy() for name, frame in splits.items()}
    caps = {
        "train": config.quick_train_rows,
        "validation": config.quick_validation_rows,
        "calibration": config.quick_calibration_rows,
        "threshold": config.quick_threshold_rows,
        "test": config.quick_test_rows,
    }
    return {
        name: stratified_cap(frame, caps[name], config.random_state + index)
        for index, (name, frame) in enumerate(splits.items())
    }


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def add_domain_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, pre-decision HMDA features with correct units."""
    output = frame.copy()
    loan_thousands = output["loan_amount_000s"].astype(float)
    income_thousands = output["applicant_income_000s"].astype(float)
    loan_dollars = loan_thousands * 1_000.0
    income_dollars = income_thousands * 1_000.0

    output["loan_to_income"] = _safe_ratio(loan_thousands, income_thousands)
    output["loan_to_area_median_income"] = _safe_ratio(
        loan_dollars, output["hud_median_family_income"]
    )
    output["applicant_to_area_median_income"] = _safe_ratio(
        income_dollars, output["hud_median_family_income"]
    )
    output["tract_to_msamd_income_ratio"] = output["tract_to_msamd_income"].astype(float) / 100.0
    output["owner_occupied_share"] = _safe_ratio(
        output["number_of_owner_occupied_units"],
        output["number_of_1_to_4_family_units"],
    )
    output["owner_units_per_1000_people"] = 1_000.0 * _safe_ratio(
        output["number_of_owner_occupied_units"], output["population"]
    )
    output["family_units_per_1000_people"] = 1_000.0 * _safe_ratio(
        output["number_of_1_to_4_family_units"], output["population"]
    )
    output["loan_per_family_unit_000s"] = _safe_ratio(
        loan_thousands, output["number_of_1_to_4_family_units"]
    )
    output["log_loan_amount"] = np.log1p(loan_thousands.clip(lower=0))
    output["log_applicant_income"] = np.log1p(income_thousands.clip(lower=0))
    output["log_population"] = np.log1p(output["population"].astype(float).clip(lower=0))
    output["log_area_median_income"] = np.log1p(
        output["hud_median_family_income"].astype(float).clip(lower=0)
    )
    output["has_co_applicant"] = (
        output["co_applicant_sex_name"].astype("string").fillna("No co-applicant")
        != "No co-applicant"
    ).astype("int8")
    return output


def build_feature_schema(
    training_frame: pd.DataFrame,
    *,
    include_sensitive: bool = False,
) -> FeatureSchema:
    engineered = add_domain_features(training_frame)
    dropped = set(IDENTIFIER_COLUMNS) | set(REDUNDANT_GEO_COLUMNS) | {TARGET_COL}
    dropped |= OUTCOME_DERIVED_COLUMNS.intersection(engineered.columns)
    if not include_sensitive:
        dropped |= SENSITIVE_COLUMNS

    feature_cols = [col for col in engineered.columns if col not in dropped]
    categorical_cols = engineered[feature_cols].select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return FeatureSchema(
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        dropped_cols=sorted(dropped.intersection(engineered.columns)),
    )


def prepare_model_frame(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    engineered = add_domain_features(frame)
    missing = sorted(set(schema.feature_cols).difference(engineered.columns))
    if missing:
        raise ValueError(f"Model frame is missing features: {missing}")
    output = engineered[schema.feature_cols].copy()
    for col in schema.categorical_cols:
        output[col] = output[col].astype("string").fillna("Missing")
    output[schema.numeric_cols] = output[schema.numeric_cols].replace([np.inf, -np.inf], np.nan)
    return output


def remove_low_information_features(
    X_train: pd.DataFrame,
    schema: FeatureSchema,
    *,
    dominant_fraction: float = 0.9995,
    max_categorical_levels: int = 1_000,
) -> tuple[FeatureSchema, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    dropped: list[str] = []
    for col in schema.feature_cols:
        counts = X_train[col].value_counts(dropna=False, normalize=True)
        dominant = float(counts.iloc[0]) if not counts.empty else 1.0
        n_unique = int(X_train[col].nunique(dropna=False))
        reason = None
        if n_unique <= 1:
            reason = "constant"
        elif dominant >= dominant_fraction:
            reason = "near_constant"
        elif col in schema.categorical_cols and n_unique > max_categorical_levels:
            reason = "high_cardinality_shortcut"
        if reason:
            dropped.append(col)
            rows.append(
                {
                    "feature": col,
                    "reason": reason,
                    "n_unique": n_unique,
                    "dominant_fraction": dominant,
                }
            )

    kept = [col for col in schema.feature_cols if col not in dropped]
    categorical = [col for col in schema.categorical_cols if col in kept]
    numeric = [col for col in schema.numeric_cols if col in kept]
    reduced = FeatureSchema(
        feature_cols=kept,
        categorical_cols=categorical,
        numeric_cols=numeric,
        dropped_cols=sorted(set(schema.dropped_cols + dropped)),
    )
    return reduced, pd.DataFrame(rows)


def build_linear_preprocessor(schema: FeatureSchema, min_frequency: int = 100) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("clip", QuantileClipper()),
            ("scale", RobustScaler(with_centering=False)),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, schema.numeric_cols), ("categorical", categorical, schema.categorical_cols)],
        remainder="drop",
    )


def build_elastic_net_logistic(
    schema: FeatureSchema,
    min_frequency: int = 100,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_linear_preprocessor(schema, min_frequency)),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=0.20,
                    C=0.5,
                    solver="saga",
                    class_weight="balanced",
                    max_iter=3_000,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_catboost_candidates(
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    X_validation: pd.DataFrame,
    y_validation: Sequence[int],
    schema: FeatureSchema,
    *,
    quick_mode: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[CatBoostClassifier, pd.DataFrame, list[CatBoostClassifier]]:
    """Tune a small, explicit CatBoost search on validation only."""
    iterations = 350 if quick_mode else 1_500
    candidates = [
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5.0, "auto_class_weights": None},
        {"depth": 7, "learning_rate": 0.04, "l2_leaf_reg": 8.0, "auto_class_weights": "Balanced"},
    ]
    if not quick_mode:
        candidates.extend(
            [
                {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 10.0, "auto_class_weights": None},
                {"depth": 7, "learning_rate": 0.035, "l2_leaf_reg": 12.0, "auto_class_weights": "SqrtBalanced"},
            ]
        )

    cat_indices = [X_train.columns.get_loc(col) for col in schema.categorical_cols]
    train_pool = Pool(X_train, np.asarray(y_train, dtype=int), cat_features=cat_indices)
    validation_pool = Pool(
        X_validation,
        np.asarray(y_validation, dtype=int),
        cat_features=cat_indices,
    )
    rows: list[dict[str, Any]] = []
    models: list[CatBoostClassifier] = []
    for candidate_id, params in enumerate(candidates):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=iterations,
            random_seed=random_state + candidate_id,
            random_strength=1.0,
            bootstrap_type="Bayesian",
            bagging_temperature=1.0,
            od_type="Iter",
            od_wait=60,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            **params,
        )
        model.fit(train_pool, eval_set=validation_pool, use_best_model=True)
        probability = model.predict_proba(X_validation)[:, 1]
        metrics = classification_metrics(y_validation, probability, threshold=0.5)
        rows.append(
            {
                "candidate": candidate_id,
                **params,
                "best_iteration": int(model.get_best_iteration()),
                "val_roc_auc": metrics["roc_auc"],
                "val_denial_pr_auc": metrics["denial_pr_auc"],
                "val_brier_score": metrics["brier_score"],
            }
        )
        models.append(model)

    results = pd.DataFrame(rows).sort_values(
        ["val_denial_pr_auc", "val_roc_auc", "val_brier_score"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_id = int(results.iloc[0]["candidate"])
    return models[best_id], results, models


def fit_lightgbm_candidates(
    X_train: pd.DataFrame,
    y_train: Sequence[int],
    X_validation: pd.DataFrame,
    y_validation: Sequence[int],
    schema: FeatureSchema,
    *,
    quick_mode: bool = True,
    random_state: int = RANDOM_STATE,
) -> tuple[LightGBMBundle, pd.DataFrame, list[LightGBMBundle]]:
    """Tune LightGBM with train-fitted category vocabularies and early stopping."""
    encoder = NativeCategoryEncoder().fit(X_train, schema.categorical_cols)
    train_native = encoder.transform(X_train)
    validation_native = encoder.transform(X_validation)
    estimators = 450 if quick_mode else 1_800
    candidates = [
        {
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 80,
            "learning_rate": 0.04,
            "class_weight": None,
        },
        {
            "num_leaves": 48,
            "max_depth": 9,
            "min_child_samples": 120,
            "learning_rate": 0.035,
            "class_weight": "balanced",
        },
    ]
    if not quick_mode:
        candidates.extend(
            [
                {
                    "num_leaves": 64,
                    "max_depth": 10,
                    "min_child_samples": 160,
                    "learning_rate": 0.025,
                    "class_weight": None,
                },
                {
                    "num_leaves": 40,
                    "max_depth": 8,
                    "min_child_samples": 100,
                    "learning_rate": 0.03,
                    "class_weight": "balanced",
                },
            ]
        )

    rows: list[dict[str, Any]] = []
    bundles: list[LightGBMBundle] = []
    for candidate_id, params in enumerate(candidates):
        model = LGBMClassifier(
            objective="binary",
            n_estimators=estimators,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=random_state + candidate_id,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(
            train_native,
            np.asarray(y_train, dtype=int),
            categorical_feature=schema.categorical_cols,
            eval_set=[(validation_native, np.asarray(y_validation, dtype=int))],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )
        bundle = LightGBMBundle(model, encoder)
        probability = bundle.predict_proba(X_validation)[:, 1]
        metrics = classification_metrics(y_validation, probability, threshold=0.5)
        rows.append(
            {
                "candidate": candidate_id,
                **params,
                "best_iteration": int(model.best_iteration_),
                "val_roc_auc": metrics["roc_auc"],
                "val_denial_pr_auc": metrics["denial_pr_auc"],
                "val_brier_score": metrics["brier_score"],
            }
        )
        bundles.append(bundle)

    results = pd.DataFrame(rows).sort_values(
        ["val_denial_pr_auc", "val_roc_auc", "val_brier_score"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best_id = int(results.iloc[0]["candidate"])
    return bundles[best_id], results, bundles


def class_weight_ratio(y: Sequence[int]) -> float:
    values = np.asarray(y, dtype=int)
    negatives = int((values == 0).sum())
    positives = int((values == 1).sum())
    # LightGBM's scale_pos_weight applies to label 1. Here approvals are label 1
    # and the minority denial class is label 0, so explicit sample weights are safer.
    return positives / max(negatives, 1)


def balanced_sample_weights(y: Sequence[int]) -> np.ndarray:
    values = np.asarray(y, dtype=int)
    counts = np.bincount(values, minlength=2)
    total = len(values)
    weights = {label: total / (2.0 * max(int(count), 1)) for label, count in enumerate(counts)}
    return np.asarray([weights[int(label)] for label in values], dtype=float)


def select_threshold(
    y_true: Sequence[int],
    approval_probability: Sequence[float],
    *,
    objective: str = "balanced_accuracy",
) -> tuple[float, float]:
    y_array = np.asarray(y_true, dtype=int)
    probability = np.asarray(approval_probability, dtype=float)
    candidates = np.unique(np.r_[np.linspace(0.02, 0.98, 193), np.quantile(probability, np.linspace(0.02, 0.98, 97))])
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        prediction = (probability >= threshold).astype(int)
        if objective == "balanced_accuracy":
            score = balanced_accuracy_score(y_array, prediction)
        elif objective == "macro_f1":
            score = f1_score(y_array, prediction, average="macro", zero_division=0)
        else:
            raise ValueError("objective must be 'balanced_accuracy' or 'macro_f1'.")
        if score > best_score:
            best_threshold = float(threshold)
            best_score = float(score)
    return best_threshold, best_score


def classification_metrics(
    y_true: Sequence[int],
    approval_probability: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    y_array = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(approval_probability, dtype=float), 1e-7, 1 - 1e-7)
    prediction = (probability >= threshold).astype(int)
    denial_true = 1 - y_array
    denial_pred = 1 - prediction
    matrix = confusion_matrix(y_array, prediction, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, prediction)),
        "macro_f1": float(f1_score(y_array, prediction, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_array, prediction)),
        "approval_precision": float(precision_score(y_array, prediction, zero_division=0)),
        "approval_recall": float(recall_score(y_array, prediction, zero_division=0)),
        "denial_precision": float(precision_score(denial_true, denial_pred, zero_division=0)),
        "denial_recall": float(recall_score(denial_true, denial_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_array, probability)),
        "approval_pr_auc": float(average_precision_score(y_array, probability)),
        "denial_pr_auc": float(average_precision_score(denial_true, 1 - probability)),
        "brier_score": float(brier_score_loss(y_array, probability)),
        "log_loss": float(log_loss(y_array, probability, labels=[0, 1])),
        "confusion_matrix_labels_0_1": matrix.tolist(),
    }


def stable_permutation_selection(
    fitted_estimators: Sequence[Any],
    X_validation: pd.DataFrame,
    y_validation: Sequence[int],
    *,
    min_features: int = 10,
    max_features: int = 24,
    n_repeats: int = 4,
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], pd.DataFrame]:
    """Select raw features by repeated held-out permutation importance.

    Multiple independently fitted estimators provide a stability signal. The
    validation set is never used to fit preprocessing or base estimators.
    """
    if not fitted_estimators:
        raise ValueError("At least one fitted estimator is required.")
    records: list[pd.DataFrame] = []
    for index, estimator in enumerate(fitted_estimators):
        result = permutation_importance(
            estimator,
            X_validation,
            np.asarray(y_validation, dtype=int),
            scoring="roc_auc",
            n_repeats=n_repeats,
            random_state=random_state + index,
            # CatBoost already uses native threads. Process-level joblib here
            # only duplicates memory and can leave noisy resource trackers on Windows.
            n_jobs=1,
        )
        records.append(
            pd.DataFrame(
                {
                    "feature": X_validation.columns,
                    "estimator": index,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
        )

    detail = pd.concat(records, ignore_index=True)
    summary = (
        detail.groupby("feature", as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            repeat_noise=("importance_std", "mean"),
            positive_fraction=("importance_mean", lambda values: float((values > 0).mean())),
        )
        .fillna({"importance_std": 0.0})
    )
    summary["lower_stability_bound"] = (
        summary["importance_mean"] - summary["importance_std"] - summary["repeat_noise"]
    )
    summary = summary.sort_values(
        ["lower_stability_bound", "importance_mean"], ascending=False
    ).reset_index(drop=True)

    stable = summary[
        (summary["lower_stability_bound"] > 0) & (summary["positive_fraction"] >= 0.5)
    ]["feature"].tolist()
    ranked = summary["feature"].tolist()
    selected = list(dict.fromkeys(stable + ranked[:min_features]))[:max_features]
    summary["selected"] = summary["feature"].isin(selected)
    return selected, summary


def choose_reduced_feature_set(
    full_score: float,
    reduced_score: float,
    full_features: Sequence[str],
    reduced_features: Sequence[str],
    tolerance: float = 0.002,
) -> tuple[list[str], str]:
    if reduced_score >= full_score - tolerance:
        return list(reduced_features), "reduced_set_within_tolerance"
    return list(full_features), "full_set_retained_due_to_validation_loss"


def make_subschema(schema: FeatureSchema, selected_features: Iterable[str]) -> FeatureSchema:
    selected_set = set(selected_features)
    unknown = selected_set.difference(schema.feature_cols)
    if unknown:
        raise ValueError(f"Selected features are not in the schema: {sorted(unknown)}")
    ordered = [col for col in schema.feature_cols if col in selected_set]
    return FeatureSchema(
        feature_cols=ordered,
        categorical_cols=[col for col in schema.categorical_cols if col in selected_set],
        numeric_cols=[col for col in schema.numeric_cols if col in selected_set],
        dropped_cols=sorted(set(schema.dropped_cols) | (set(schema.feature_cols) - selected_set)),
    )


def subgroup_diagnostics(
    raw_frame: pd.DataFrame,
    approval_probability: Sequence[float],
    threshold: float,
    columns: Sequence[str] = (
        "applicant_sex_name",
        "applicant_race_name_1",
        "applicant_ethnicity_name",
    ),
    min_group_size: int = 250,
) -> pd.DataFrame:
    probability = np.asarray(approval_probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    y = raw_frame[TARGET_COL].astype(int).to_numpy()
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in raw_frame.columns:
            continue
        values = raw_frame[column].astype("string").fillna("Missing")
        for group, index in values.groupby(values).groups.items():
            idx = np.asarray(list(index), dtype=int)
            if len(idx) < min_group_size:
                continue
            group_y = y[idx]
            group_pred = prediction[idx]
            rows.append(
                {
                    "attribute": column,
                    "group": str(group),
                    "n": len(idx),
                    "observed_approval_rate": float(group_y.mean()),
                    "predicted_approval_rate": float(group_pred.mean()),
                    "approval_recall": float(recall_score(group_y, group_pred, zero_division=0)),
                    "denial_recall": float(
                        recall_score(1 - group_y, 1 - group_pred, zero_division=0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def config_fingerprint(config: ExperimentConfig, schema: FeatureSchema) -> str:
    payload = repr((asdict(config), asdict(schema))).encode("utf-8")
    return sha256(payload).hexdigest()[:12]


def probability_predictor(model: Any) -> Callable[[pd.DataFrame], np.ndarray]:
    return lambda X: np.asarray(model.predict_proba(X)[:, 1], dtype=float)
