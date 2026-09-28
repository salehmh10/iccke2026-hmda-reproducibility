"""Clean binary mortgage approval classification pipeline.

Fixes applied:
- no test resampling
- no test leakage for early stopping
- sensitive features excluded from the main model contract
- deduplication before splitting
- shared untouched test set for every model
- validation-driven threshold tuning
- weighted soft-voting ensemble instead of majority vote
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve

RANDOM_STATE = 42
TARGET_COL = "loan_approved"
DATA_FILE = Path("hmda_classification_stratified_500k.csv")
OUTPUT_DIR = Path("artifacts_fixed")
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_CSV = OUTPUT_DIR / "classification_results.csv"
BEST_MODELS_CSV = OUTPUT_DIR / "best_family_models.csv"
THRESHOLDS_CSV = OUTPUT_DIR / "thresholds.csv"
CALIBRATION_CSV = OUTPUT_DIR / "calibration_curves.csv"

DROP_COLS = ["respondent_id"]
SENSITIVE_COLS = [
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
]
CODELIKE_CAT_COLS = ["state_code", "county_code", "census_tract_number"]

ENGINEERED_NUMERIC_COLS = [
    "loan_to_income",
    "loan_to_median_income",
    "income_to_median_income",
    "owner_occupied_share",
    "owner_units_per_pop",
    "family_units_per_pop",
    "loan_per_population",
    "loan_per_family_unit",
    "log_loan_amount",
    "log_applicant_income",
    "log_population",
    "log_minority_population",
    "log_owner_occupied_units",
    "log_family_units",
    "log_hud_median_family_income",
    "log_tract_to_msamd_income",
]

RF_DEPTH_GRID = [4, 6, 8, 10, 12, 14, 16, 18, 20]


@dataclass(frozen=True)
class FeatureConfig:
    feature_cols: list[str]
    categorical_cols: list[str]
    numeric_cols: list[str]


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df.drop_duplicates().reset_index(drop=True)


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    model_df = data.copy()
    model_df["loan_to_income"] = model_df["loan_amount_000s"] / model_df["applicant_income_000s"].replace(0, np.nan)
    model_df["loan_to_median_income"] = model_df["loan_amount_000s"] / model_df["hud_median_family_income"].replace(0, np.nan)
    model_df["income_to_median_income"] = model_df["applicant_income_000s"] / model_df["hud_median_family_income"].replace(0, np.nan)
    model_df["owner_occupied_share"] = model_df["number_of_owner_occupied_units"] / model_df["number_of_1_to_4_family_units"].replace(0, np.nan)
    model_df["owner_units_per_pop"] = model_df["number_of_owner_occupied_units"] / model_df["population"].replace(0, np.nan)
    model_df["family_units_per_pop"] = model_df["number_of_1_to_4_family_units"] / model_df["population"].replace(0, np.nan)
    model_df["loan_per_population"] = model_df["loan_amount_000s"] / model_df["population"].replace(0, np.nan)
    model_df["loan_per_family_unit"] = model_df["loan_amount_000s"] / model_df["number_of_1_to_4_family_units"].replace(0, np.nan)
    model_df["log_loan_amount"] = np.log1p(model_df["loan_amount_000s"])
    model_df["log_applicant_income"] = np.log1p(model_df["applicant_income_000s"])
    model_df["log_population"] = np.log1p(model_df["population"])
    model_df["log_minority_population"] = np.log1p(model_df["minority_population"])
    model_df["log_owner_occupied_units"] = np.log1p(model_df["number_of_owner_occupied_units"])
    model_df["log_family_units"] = np.log1p(model_df["number_of_1_to_4_family_units"])
    model_df["log_hud_median_family_income"] = np.log1p(model_df["hud_median_family_income"])
    model_df["log_tract_to_msamd_income"] = np.log1p(model_df["tract_to_msamd_income"])
    return model_df


def build_feature_config(frame: pd.DataFrame, include_sensitive: bool = False) -> FeatureConfig:
    excluded = set(DROP_COLS)
    if not include_sensitive:
        excluded.update(SENSITIVE_COLS)

    feature_cols = [col for col in frame.columns if col != TARGET_COL and col not in excluded]
    categorical_cols = frame[feature_cols].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in CODELIKE_CAT_COLS:
        if col in feature_cols and col not in categorical_cols:
            categorical_cols.append(col)
    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    return FeatureConfig(feature_cols=feature_cols, categorical_cols=categorical_cols, numeric_cols=numeric_cols)


def prepare_features(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    X = add_engineered_features(data)
    X = X[config.feature_cols].copy()
    for col in config.categorical_cols:
        X[col] = X[col].astype("string").fillna("Missing")
    return X


def split_train_val_test(
    df: pd.DataFrame,
    group_col: str = "msamd_name",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a held-out test split and a validation split.

    If the requested group column exists, we prefer group-held-out splits to
    improve generalization checks. Otherwise we fall back to stratified random
    splits.
    """
    if group_col in df.columns and df[group_col].nunique(dropna=False) > 1:
        groups = df[group_col].fillna("Missing").astype(str)
        first_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_temp_idx, test_idx = next(first_split.split(df, groups=groups))
        train_temp_df = df.iloc[train_temp_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        train_temp_groups = train_temp_df[group_col].fillna("Missing").astype(str)
        second_split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, val_idx = next(second_split.split(train_temp_df, groups=train_temp_groups))
        train_df = train_temp_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_temp_df.iloc[val_idx].reset_index(drop=True)
        return train_df, val_df, test_df

    train_temp_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    train_df, val_df = train_test_split(
        train_temp_df,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_temp_df[TARGET_COL],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_under_sampled_train(data: pd.DataFrame) -> pd.DataFrame:
    counts = data[TARGET_COL].value_counts()
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    target_size = int(counts.min())

    minority_part = data[data[TARGET_COL] == minority_label]
    majority_part = data[data[TARGET_COL] == majority_label].sample(
        n=target_size,
        random_state=RANDOM_STATE,
        replace=False,
    )

    return (
        pd.concat([minority_part, majority_part], axis=0)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def make_200k_each_train(data: pd.DataFrame, target_size: int = 200_000) -> pd.DataFrame:
    counts = data[TARGET_COL].value_counts()
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()

    minority_source = data[data[TARGET_COL] == minority_label]
    majority_source = data[data[TARGET_COL] == majority_label]

    minority_part = minority_source.sample(
        n=target_size,
        random_state=RANDOM_STATE,
        replace=len(minority_source) < target_size,
    )
    majority_part = majority_source.sample(
        n=target_size,
        random_state=RANDOM_STATE,
        replace=len(majority_source) < target_size,
    )

    return (
        pd.concat([minority_part, majority_part], axis=0)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=50,
            sparse_output=True,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=True,
        )


def build_preprocessor(config: FeatureConfig, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler(with_mean=False)))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), config.numeric_cols),
            ("categorical", build_one_hot_encoder(), config.categorical_cols),
        ],
        remainder="drop",
    )


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "balanced_accuracy") -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, float(best_score)


def build_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "denial_precision": precision_score(1 - y_true, 1 - y_pred, zero_division=0),
        "denial_recall": recall_score(1 - y_true, 1 - y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }


def build_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    model: str,
    train_dataset: str,
    feature_set: str,
    split_name: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame(
        {
            "model": model,
            "train_dataset": train_dataset,
            "feature_set": feature_set,
            "split": split_name,
            "bin": np.arange(1, len(frac_pos) + 1),
            "mean_predicted_prob": mean_pred,
            "fraction_of_positives": frac_pos,
        }
    )


def upsert_result(results: list[dict[str, Any]], result: dict[str, Any]) -> None:
    results[:] = [
        item
        for item in results
        if not (
            item["model"] == result["model"]
            and item["train_dataset"] == result["train_dataset"]
            and item["feature_set"] == result["feature_set"]
        )
    ]
    results.append(result)


def save_results_csv(results: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(results).sort_values(["feature_set", "model", "train_dataset"]).to_csv(path, index=False)


def save_thresholds_csv(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows).sort_values(["feature_set", "model", "train_dataset"]).to_csv(path, index=False)


def save_calibration_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    else:
        pd.DataFrame().to_csv(path, index=False)


def save_model_artifact(filename: str, obj: Any) -> Path:
    path = MODEL_DIR / filename
    joblib.dump(obj, path)
    return path


def calibrate_prefit_model(estimator: Any, X_cal: Any, y_cal: np.ndarray) -> CalibratedClassifierCV:
    """Calibrate a fitted estimator without refitting the base model."""
    try:
        calibrator = CalibratedClassifierCV(estimator=estimator, cv="prefit", method="sigmoid")
    except TypeError:
        calibrator = CalibratedClassifierCV(base_estimator=estimator, cv="prefit", method="sigmoid")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def load_model_artifact(filename: str) -> Any | None:
    path = MODEL_DIR / filename
    if path.exists():
        return joblib.load(path)
    return None


def train_logistic(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FeatureConfig,
    results: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model_filename = f"calibrated_logreg_{dataset_name}.joblib"
    model_path = MODEL_DIR / model_filename
    if model_path.exists():
        pipeline = joblib.load(model_path)
    else:
        class_weight = "balanced" if dataset_name == "unbalanced_train_df" else None
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(config, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        solver="saga",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
        X_train = prepare_features(train_df, config)
        y_train = train_df[TARGET_COL].astype(int)
        pipeline.fit(X_train, y_train)
        X_val = prepare_features(val_df, config)
        y_val = val_df[TARGET_COL].astype(int).to_numpy()
        pipeline = calibrate_prefit_model(pipeline, X_val, y_val)
        save_model_artifact(model_filename, pipeline)

    X_val = prepare_features(val_df, config)
    y_val = val_df[TARGET_COL].astype(int).to_numpy()
    X_test = prepare_features(test_df, config)
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    threshold, val_threshold_score = select_threshold(y_val, val_prob, metric="balanced_accuracy")

    thresholds.append(
        {
            "feature_set": "main",
            "model": "Logistic Regression",
            "train_dataset": dataset_name,
            "threshold": threshold,
            "val_threshold_score": val_threshold_score,
        }
    )

    val_metrics = build_metrics(y_val, val_prob, threshold)
    test_metrics = build_metrics(y_test, test_prob, threshold)
    calibration_rows.append(
        build_calibration_bins(
            y_test,
            test_prob,
            model="Logistic Regression",
            train_dataset=dataset_name,
            feature_set="main",
            split_name="test",
        )
    )

    result = {
        "feature_set": "main",
        "model": "Logistic Regression",
        "train_dataset": dataset_name,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_denial_precision": val_metrics["denial_precision"],
        "val_denial_recall": val_metrics["denial_recall"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_brier_score": val_metrics["brier_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_denial_precision": test_metrics["denial_precision"],
        "test_denial_recall": test_metrics["denial_recall"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier_score": test_metrics["brier_score"],
        "threshold": threshold,
    }
    upsert_result(results, result)

    return result, {
        "val_prob": val_prob,
        "test_prob": test_prob,
        "y_val": y_val,
        "y_test": y_test,
    }


def train_random_forest(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FeatureConfig,
    results: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_name = dataset_name
    model_filename = f"calibrated_rf_{cache_name}.joblib"
    model_path = MODEL_DIR / model_filename
    best_depth = None

    X_train_raw = prepare_features(train_df, config)
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    X_val_raw = prepare_features(val_df, config)
    y_val = val_df[TARGET_COL].astype(int).to_numpy()
    X_test_raw = prepare_features(test_df, config)
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    preprocessor = build_preprocessor(config, scale_numeric=False)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    if model_path.exists():
        bundle = joblib.load(model_path)
        preprocessor = bundle["preprocessor"]
        rf_model = bundle["model"]
        X_val = preprocessor.transform(X_val_raw)
        X_test = preprocessor.transform(X_test_raw)
    else:
        class_weight = "balanced" if dataset_name == "unbalanced_train_df" else None
        best_key = None
        best_model = None

        for depth in RF_DEPTH_GRID:
            candidate = RandomForestClassifier(
                n_estimators=150,
                max_depth=depth,
                min_samples_leaf=20,
                class_weight=class_weight,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            candidate.fit(X_train, y_train)
            val_prob = candidate.predict_proba(X_val)[:, 1]
            val_pred = (val_prob >= 0.5).astype(int)
            key = (
                roc_auc_score(y_val, val_prob),
                balanced_accuracy_score(y_val, val_pred),
                f1_score(y_val, val_pred, zero_division=0),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_depth = depth
                best_model = candidate

        assert best_model is not None
        rf_model = best_model
        rf_model = calibrate_prefit_model(rf_model, X_val, y_val)
        save_model_artifact(model_filename, {"preprocessor": preprocessor, "model": rf_model})

    val_prob = rf_model.predict_proba(X_val)[:, 1]
    test_prob = rf_model.predict_proba(X_test)[:, 1]
    threshold, val_threshold_score = select_threshold(y_val, val_prob, metric="balanced_accuracy")

    thresholds.append(
        {
            "feature_set": "main",
            "model": "Random Forest",
            "train_dataset": dataset_name,
            "threshold": threshold,
            "val_threshold_score": val_threshold_score,
        }
    )

    val_metrics = build_metrics(y_val, val_prob, threshold)
    test_metrics = build_metrics(y_test, test_prob, threshold)
    calibration_rows.append(
        build_calibration_bins(
            y_test,
            test_prob,
            model="Random Forest",
            train_dataset=dataset_name,
            feature_set="main",
            split_name="test",
        )
    )
    result = {
        "feature_set": "main",
        "model": "Random Forest",
        "train_dataset": dataset_name,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_denial_precision": val_metrics["denial_precision"],
        "val_denial_recall": val_metrics["denial_recall"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_brier_score": val_metrics["brier_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_denial_precision": test_metrics["denial_precision"],
        "test_denial_recall": test_metrics["denial_recall"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier_score": test_metrics["brier_score"],
        "threshold": threshold,
        "best_max_depth": best_depth,
    }
    upsert_result(results, result)

    return result, {
        "val_prob": val_prob,
        "test_prob": test_prob,
        "y_val": y_val,
        "y_test": y_test,
    }


def prepare_lgbm_X(data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    X = prepare_features(data, config)
    for col in config.categorical_cols:
        X[col] = X[col].astype("category")
    return X


def train_catboost(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FeatureConfig,
    results: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_name = dataset_name
    model_filename = f"calibrated_catboost_{cache_name}.joblib"
    model_path = MODEL_DIR / model_filename

    X_train = prepare_features(train_df, config)
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    X_val = prepare_features(val_df, config)
    y_val = val_df[TARGET_COL].astype(int).to_numpy()
    X_test = prepare_features(test_df, config)
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    cat_feature_indices = [X_train.columns.get_loc(col) for col in config.categorical_cols]

    if model_path.exists():
        cat_model = CatBoostClassifier()
        cat_model.load_model(str(model_path))
    else:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        class_weights = [1.0, float(neg / max(pos, 1))] if dataset_name == "unbalanced_train_df" else None

        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_feature_indices)

        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=1200,
            learning_rate=0.05,
            depth=10,
            class_weights=class_weights,
            random_seed=RANDOM_STATE,
            verbose=100,
            od_type="Iter",
            od_wait=50,
        )
        cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        cat_model = calibrate_prefit_model(cat_model, X_val, y_val)
        save_model_artifact(model_filename, cat_model)

    val_prob = cat_model.predict_proba(X_val)[:, 1]
    test_prob = cat_model.predict_proba(X_test)[:, 1]
    threshold, val_threshold_score = select_threshold(y_val, val_prob, metric="balanced_accuracy")

    thresholds.append(
        {
            "feature_set": "main",
            "model": "CatBoost",
            "train_dataset": dataset_name,
            "threshold": threshold,
            "val_threshold_score": val_threshold_score,
        }
    )

    val_metrics = build_metrics(y_val, val_prob, threshold)
    test_metrics = build_metrics(y_test, test_prob, threshold)
    calibration_rows.append(
        build_calibration_bins(
            y_test,
            test_prob,
            model="CatBoost",
            train_dataset=dataset_name,
            feature_set="main",
            split_name="test",
        )
    )
    result = {
        "feature_set": "main",
        "model": "CatBoost",
        "train_dataset": dataset_name,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_denial_precision": val_metrics["denial_precision"],
        "val_denial_recall": val_metrics["denial_recall"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_brier_score": val_metrics["brier_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_denial_precision": test_metrics["denial_precision"],
        "test_denial_recall": test_metrics["denial_recall"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier_score": test_metrics["brier_score"],
        "threshold": threshold,
    }
    upsert_result(results, result)

    return result, {
        "val_prob": val_prob,
        "test_prob": test_prob,
        "y_val": y_val,
        "y_test": y_test,
    }


def train_lgbm(
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: FeatureConfig,
    results: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_name = dataset_name
    model_filename = f"calibrated_lgbm_{cache_name}.joblib"
    model_path = MODEL_DIR / model_filename

    X_train = prepare_lgbm_X(train_df, config)
    y_train = train_df[TARGET_COL].astype(int).to_numpy()
    X_val = prepare_lgbm_X(val_df, config)
    y_val = val_df[TARGET_COL].astype(int).to_numpy()
    X_test = prepare_lgbm_X(test_df, config)
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    if model_path.exists():
        lgbm_model = joblib.load(model_path)
    else:
        lgbm_class_weight = "balanced" if dataset_name == "unbalanced_train_df" else None
        lgbm_model = LGBMClassifier(
            objective="binary",
            n_estimators=700,
            learning_rate=0.04,
            num_leaves=64,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight=lgbm_class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        try:
            lgbm_model.fit(
                X_train,
                y_train,
                categorical_feature=config.categorical_cols,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except TypeError:
            lgbm_model.fit(
                X_train,
                y_train,
                categorical_feature=config.categorical_cols,
            )
        lgbm_model = calibrate_prefit_model(lgbm_model, X_val, y_val)
        save_model_artifact(model_filename, lgbm_model)

    val_prob = lgbm_model.predict_proba(X_val)[:, 1]
    test_prob = lgbm_model.predict_proba(X_test)[:, 1]
    threshold, val_threshold_score = select_threshold(y_val, val_prob, metric="balanced_accuracy")

    thresholds.append(
        {
            "feature_set": "main",
            "model": "LightGBM",
            "train_dataset": dataset_name,
            "threshold": threshold,
            "val_threshold_score": val_threshold_score,
        }
    )

    val_metrics = build_metrics(y_val, val_prob, threshold)
    test_metrics = build_metrics(y_test, test_prob, threshold)
    calibration_rows.append(
        build_calibration_bins(
            y_test,
            test_prob,
            model="LightGBM",
            train_dataset=dataset_name,
            feature_set="main",
            split_name="test",
        )
    )
    result = {
        "feature_set": "main",
        "model": "LightGBM",
        "train_dataset": dataset_name,
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_denial_precision": val_metrics["denial_precision"],
        "val_denial_recall": val_metrics["denial_recall"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_brier_score": val_metrics["brier_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_denial_precision": test_metrics["denial_precision"],
        "test_denial_recall": test_metrics["denial_recall"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier_score": test_metrics["brier_score"],
        "threshold": threshold,
    }
    upsert_result(results, result)

    return result, {
        "val_prob": val_prob,
        "test_prob": test_prob,
        "y_val": y_val,
        "y_test": y_test,
    }


def train_sensitive_diagnostic(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
) -> None:
    diagnostic_config = build_feature_config(add_engineered_features(train_df), include_sensitive=True)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(diagnostic_config, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    X_train = prepare_features(train_df, diagnostic_config)
    y_train = train_df[TARGET_COL].astype(int)
    X_val = prepare_features(val_df, diagnostic_config)
    y_val = val_df[TARGET_COL].astype(int).to_numpy()
    X_test = prepare_features(test_df, diagnostic_config)
    y_test = test_df[TARGET_COL].astype(int).to_numpy()

    pipeline.fit(X_train, y_train)
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]
    threshold, val_threshold_score = select_threshold(y_val, val_prob, metric="balanced_accuracy")

    thresholds.append(
        {
            "feature_set": "diagnostic_sensitive",
            "model": "Logistic Regression",
            "train_dataset": "unbalanced_train_df",
            "threshold": threshold,
            "val_threshold_score": val_threshold_score,
        }
    )

    val_metrics = build_metrics(y_val, val_prob, threshold)
    test_metrics = build_metrics(y_test, test_prob, threshold)
    calibration_rows.append(
        build_calibration_bins(
            y_test,
            test_prob,
            model="Logistic Regression",
            train_dataset="unbalanced_train_df",
            feature_set="diagnostic_sensitive",
            split_name="test",
        )
    )
    result = {
        "feature_set": "diagnostic_sensitive",
        "model": "Logistic Regression",
        "train_dataset": "unbalanced_train_df",
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_denial_precision": val_metrics["denial_precision"],
        "val_denial_recall": val_metrics["denial_recall"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_brier_score": val_metrics["brier_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_denial_precision": test_metrics["denial_precision"],
        "test_denial_recall": test_metrics["denial_recall"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier_score": test_metrics["brier_score"],
        "threshold": threshold,
    }
    upsert_result(results, result)


def select_best_family_models(
    results: list[dict[str, Any]],
    family_names: Iterable[str],
) -> pd.DataFrame:
    results_df = pd.DataFrame(results)
    best_rows: list[pd.Series] = []
    for family in family_names:
        family_df = results_df[results_df["model"] == family].copy()
        family_df = family_df.sort_values(
            ["val_roc_auc", "val_balanced_accuracy", "val_f1"],
            ascending=False,
        )
        best_rows.append(family_df.iloc[0])
    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    best_df.to_csv(BEST_MODELS_CSV, index=False)
    return best_df


def build_soft_voting_ensemble(
    best_models: pd.DataFrame,
    family_predictions: dict[tuple[str, str], dict[str, np.ndarray]],
    results: list[dict[str, Any]],
    thresholds: list[dict[str, Any]],
    calibration_rows: list[pd.DataFrame],
) -> None:
    weights = best_models["val_roc_auc"].to_numpy(dtype=float)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()

    y_val = next(iter(family_predictions.values()))["y_val"]
    y_test = next(iter(family_predictions.values()))["y_test"]

    val_prob = np.zeros_like(y_val, dtype=float)
    test_prob = np.zeros_like(y_test, dtype=float)

    for weight, row in zip(weights, best_models.itertuples(index=False)):
        key = (row.model, row.train_dataset)
        pred_bundle = family_predictions[key]
        val_prob += weight * pred_bundle["val_prob"]
        test_prob += weight * pred_bundle["test_prob"]

    threshold, val_threshold_score = select_threshold(y_val, val_prob, metric="balanced_accuracy")
    thresholds.append(
        {
            "feature_set": "main",
            "model": "Soft Voting Ensemble",
            "train_dataset": "best_family_models",
            "threshold": threshold,
            "val_threshold_score": val_threshold_score,
        }
    )

    val_metrics = build_metrics(y_val, val_prob, threshold)
    test_metrics = build_metrics(y_test, test_prob, threshold)
    calibration_rows.append(
        build_calibration_bins(
            y_test,
            test_prob,
            model="Soft Voting Ensemble",
            train_dataset="best_family_models",
            feature_set="main",
            split_name="test",
        )
    )
    result = {
        "feature_set": "main",
        "model": "Soft Voting Ensemble",
        "train_dataset": "best_family_models",
        "val_accuracy": val_metrics["accuracy"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_denial_precision": val_metrics["denial_precision"],
        "val_denial_recall": val_metrics["denial_recall"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_brier_score": val_metrics["brier_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_denial_precision": test_metrics["denial_precision"],
        "test_denial_recall": test_metrics["denial_recall"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_brier_score": test_metrics["brier_score"],
        "threshold": threshold,
        "ensemble_weights": weights.tolist(),
    }
    upsert_result(results, result)


def print_summary(results: list[dict[str, Any]]) -> None:
    results_df = pd.DataFrame(results)
    ordered = results_df.sort_values(["val_roc_auc", "val_balanced_accuracy", "test_roc_auc"], ascending=False)
    cols = [
        "feature_set",
        "model",
        "train_dataset",
        "val_roc_auc",
        "val_balanced_accuracy",
        "test_roc_auc",
        "test_balanced_accuracy",
        "test_f1",
    ]
    print(ordered[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean binary mortgage approval classification pipeline")
    parser.add_argument("--data-file", type=Path, default=DATA_FILE)
    parser.add_argument("--compare-sensitive", action="store_true", help="Train a diagnostic logistic model with sensitive columns")
    args = parser.parse_args()

    ensure_output_dirs()
    df = load_data(args.data_file)
    df_model = add_engineered_features(df)

    main_config = build_feature_config(df_model, include_sensitive=False)

    train_base_df, val_df, test_df = split_train_val_test(df_model)
    under_sampled_train_df = make_under_sampled_train(train_base_df)
    balanced_200k_train_df = make_200k_each_train(train_base_df)
    unbalanced_train_df = train_base_df.copy()

    train_variants = {
        "under_sampled_train_df": under_sampled_train_df,
        "balanced_200k_train_df": balanced_200k_train_df,
        "unbalanced_train_df": unbalanced_train_df,
    }

    results: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    family_predictions: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for dataset_name, train_df in train_variants.items():
        logistic_result, logistic_bundle = train_logistic(
            dataset_name, train_df, val_df, test_df, main_config, results, thresholds, calibration_rows
        )
        family_predictions[(logistic_result["model"], dataset_name)] = logistic_bundle

        rf_result, rf_bundle = train_random_forest(
            dataset_name, train_df, val_df, test_df, main_config, results, thresholds, calibration_rows
        )
        family_predictions[(rf_result["model"], dataset_name)] = rf_bundle

        cat_result, cat_bundle = train_catboost(
            dataset_name, train_df, val_df, test_df, main_config, results, thresholds, calibration_rows
        )
        family_predictions[(cat_result["model"], dataset_name)] = cat_bundle

        lgbm_result, lgbm_bundle = train_lgbm(
            dataset_name, train_df, val_df, test_df, main_config, results, thresholds, calibration_rows
        )
        family_predictions[(lgbm_result["model"], dataset_name)] = lgbm_bundle

    if args.compare_sensitive:
        train_sensitive_diagnostic(train_base_df, val_df, test_df, results, thresholds, calibration_rows)

    best_models = select_best_family_models(
        results,
        family_names=["Logistic Regression", "Random Forest", "CatBoost", "LightGBM"],
    )
    build_soft_voting_ensemble(best_models, family_predictions, results, thresholds, calibration_rows)

    save_results_csv(results, RESULTS_CSV)
    save_thresholds_csv(thresholds, THRESHOLDS_CSV)
    save_calibration_csv(calibration_rows, CALIBRATION_CSV)
    print_summary(results)


if __name__ == "__main__":
    main()
