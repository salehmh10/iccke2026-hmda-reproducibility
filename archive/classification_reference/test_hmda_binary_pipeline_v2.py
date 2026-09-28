from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hmda_binary_pipeline_v2 import (
    DATA_FILE,
    SENSITIVE_COLUMNS,
    TARGET_COL,
    ExperimentConfig,
    LockedHoldout,
    add_domain_features,
    add_random_shadow_features,
    apply_outlier_features,
    assert_split_integrity,
    build_elastic_net_logistic,
    build_feature_schema,
    classification_metrics,
    clean_before_split,
    fit_outlier_bounds,
    load_primary_data,
    make_five_way_splits,
    prepare_model_frame,
    select_threshold,
)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    frame = load_primary_data(Path(DATA_FILE), nrows=8_000)
    # The first rows can be dominated by one class in arbitrary source order.
    if frame[TARGET_COL].nunique() != 2:
        pytest.skip("The source prefix does not contain both target classes.")
    return frame


def test_deduplication_happens_before_split(sample: pd.DataFrame) -> None:
    duplicated = pd.concat([sample, sample.iloc[[0]]], ignore_index=True)
    cleaned, report = clean_before_split(duplicated)
    assert report["duplicate_rows_removed"] >= 1
    assert not cleaned.duplicated().any()


def test_five_way_split_has_no_overlap(sample: pd.DataFrame) -> None:
    cleaned, _ = clean_before_split(sample)
    splits = make_five_way_splits(cleaned, ExperimentConfig(quick_mode=False))
    assert_split_integrity(splits)
    assert sum(map(len, splits.values())) == len(cleaned)
    for frame in splits.values():
        assert frame[TARGET_COL].nunique() == 2


def test_feature_contract_excludes_ids_sensitive_and_target(sample: pd.DataFrame) -> None:
    schema = build_feature_schema(sample)
    assert "respondent_id" not in schema.feature_cols
    assert TARGET_COL not in schema.feature_cols
    assert not SENSITIVE_COLUMNS.intersection(schema.feature_cols)


def test_engineered_ratios_use_consistent_units(sample: pd.DataFrame) -> None:
    engineered = add_domain_features(sample.iloc[[0]])
    row = sample.iloc[0]
    expected = row["loan_amount_000s"] * 1_000 / row["hud_median_family_income"]
    assert np.isclose(engineered.iloc[0]["loan_to_area_median_income"], expected)
    assert np.isclose(
        engineered.iloc[0]["loan_to_income"],
        row["loan_amount_000s"] / row["applicant_income_000s"],
    )


def test_locked_holdout_rejects_early_access(sample: pd.DataFrame) -> None:
    locked = LockedHoldout(sample.iloc[:10])
    with pytest.raises(RuntimeError):
        locked.get()
    locked.unlock("Feature set, model, calibration, and threshold frozen")
    assert len(locked.get()) == 10


def test_linear_pipeline_and_threshold_smoke(sample: pd.DataFrame) -> None:
    cleaned, _ = clean_before_split(sample)
    splits = make_five_way_splits(cleaned, ExperimentConfig(quick_mode=False))
    schema = build_feature_schema(splits["train"])
    X_train = prepare_model_frame(splits["train"], schema)
    X_validation = prepare_model_frame(splits["validation"], schema)
    model = build_elastic_net_logistic(schema, min_frequency=5)
    model.set_params(model__max_iter=1_000, model__tol=1e-3)
    model.fit(X_train, splits["train"][TARGET_COL].astype(int))
    probability = model.predict_proba(X_validation)[:, 1]
    threshold, _ = select_threshold(splits["validation"][TARGET_COL], probability)
    metrics = classification_metrics(splits["validation"][TARGET_COL], probability, threshold)
    assert 0 <= threshold <= 1
    assert 0 <= metrics["denial_pr_auc"] <= 1
    assert np.isfinite(metrics["brier_score"])


def test_random_controls_and_outlier_bounds_are_train_fitted(sample: pd.DataFrame) -> None:
    cleaned, _ = clean_before_split(sample)
    splits = make_five_way_splits(cleaned, ExperimentConfig(quick_mode=False))
    schema = build_feature_schema(splits["train"])
    X_train = prepare_model_frame(splits["train"], schema)
    X_validation = prepare_model_frame(splits["validation"], schema)
    train_shadow, validation_shadow, shadow_schema, random_columns = add_random_shadow_features(
        X_train,
        X_validation,
        schema,
        random_state=17,
        n_numeric_shadows=2,
        n_categorical_shadows=1,
        n_random_projections=1,
    )
    assert len(random_columns) == 4
    assert all(column.startswith(("shadow_random_", "random_projection_")) for column in random_columns)
    assert TARGET_COL not in shadow_schema.feature_cols
    assert len(train_shadow) == len(X_train)
    assert len(validation_shadow) == len(X_validation)

    bounds = fit_outlier_bounds(X_train, schema.numeric_cols)
    train_with_flags, train_lower, train_upper = apply_outlier_features(X_train, bounds)
    validation_with_flags, validation_lower, validation_upper = apply_outlier_features(
        X_validation, bounds
    )
    assert {"lower_outlier_count", "upper_outlier_count", "has_upper_outlier"}.issubset(
        train_with_flags.columns
    )
    assert len(train_lower) == len(X_train)
    assert len(train_upper) == len(X_train)
    assert len(validation_lower) == len(X_validation)
    assert len(validation_upper) == len(X_validation)
