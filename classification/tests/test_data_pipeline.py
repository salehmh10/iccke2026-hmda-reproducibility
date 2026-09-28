from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data import (
    ANALYTICAL_TARGET,
    AUDIT_ONLY_COLUMNS,
    DataContractError,
    EXPECTED_COLUMNS,
    RAW_TARGET,
    build_dataset_variants,
    build_modeling_view,
    load_raw_dataset,
    load_data_config,
    load_split_manifest,
    load_variant,
    make_stratified_split,
    persist_split_manifest,
    persist_variant_manifests,
    primary_feature_columns,
)
from src.data.loader import summarize_quality
from src.data.schema import FLOAT_COLUMNS, INTEGER_COLUMNS, STRING_COLUMNS


def make_frame(rows: int = 100, *, duplicate_first: bool = False) -> pd.DataFrame:
    data: dict[str, object] = {}
    for column in STRING_COLUMNS:
        data[column] = pd.Series(
            [f"{column}_{row % 7}" for row in range(rows)], dtype="string"
        )
    # Verify code-like fields retain their textual representation.
    data["respondent_id"] = pd.Series(
        [f"{row % 13:010d}" for row in range(rows)], dtype="string"
    )
    data["state_code"] = pd.Series(
        [f"{row % 5:02d}" for row in range(rows)], dtype="string"
    )
    data["county_code"] = pd.Series(
        [f"{row % 11:03d}" for row in range(rows)], dtype="string"
    )
    data["census_tract_number"] = pd.Series(
        [f"{row % 17:04d}.00" for row in range(rows)], dtype="string"
    )
    for offset, column in enumerate(INTEGER_COLUMNS):
        data[column] = pd.Series(
            np.arange(rows, dtype=np.int64) + offset + 1, dtype="Int64"
        )
    for offset, column in enumerate(FLOAT_COLUMNS):
        data[column] = pd.Series(
            np.arange(rows, dtype=np.float64) + offset + 0.5, dtype="Float64"
        )
    data["minority_population"] = pd.Series(
        (np.arange(rows, dtype=np.float64) % 100) + 0.5, dtype="Float64"
    )
    data[RAW_TARGET] = pd.Series(
        [1 if row % 5 else 0 for row in range(rows)], dtype="Int8"
    )
    frame = pd.DataFrame(data, columns=EXPECTED_COLUMNS)
    if duplicate_first:
        frame.iloc[-1] = frame.iloc[0]
    return frame


def test_explicit_loader_preserves_schema_codes_and_source(tmp_path) -> None:
    source = tmp_path / "input.csv"
    expected = make_frame(30)
    expected.to_csv(source, index=False)
    bytes_before = source.read_bytes()

    actual = load_raw_dataset(source)

    assert source.read_bytes() == bytes_before
    assert tuple(actual.columns) == EXPECTED_COLUMNS
    assert actual.loc[0, "respondent_id"] == "0000000000"
    assert actual.loc[0, "state_code"] == "00"
    assert str(actual[RAW_TARGET].dtype) == "Int8"
    assert all(str(actual[column].dtype) == "string" for column in STRING_COLUMNS)


def test_modeling_view_derives_denial_and_only_deduplicates() -> None:
    raw = make_frame(100, duplicate_first=True)
    raw_copy = raw.copy(deep=True)

    modeling = build_modeling_view(raw)
    quality = summarize_quality(raw, modeling)

    pd.testing.assert_frame_equal(raw, raw_copy)
    assert len(modeling) == 99
    assert quality.duplicate_rows_removed == 1
    assert quality.raw_rows == quality.modeling_rows + quality.duplicate_rows_removed
    assert modeling.index.tolist() == list(range(99))
    assert np.array_equal(
        modeling[ANALYTICAL_TARGET].to_numpy(dtype=np.int8),
        1 - modeling[RAW_TARGET].to_numpy(dtype=np.int8),
    )


def test_loader_rejects_nonbinary_target(tmp_path) -> None:
    source = tmp_path / "invalid.csv"
    frame = make_frame(20)
    frame.loc[0, RAW_TARGET] = 2
    frame.to_csv(source, index=False)
    with pytest.raises(DataContractError, match="binary values"):
        load_raw_dataset(source)


def test_loader_rejects_invalid_negative_financial_value(tmp_path) -> None:
    source = tmp_path / "invalid_negative.csv"
    frame = make_frame(20)
    frame.loc[0, "loan_amount_000s"] = -1
    frame.to_csv(source, index=False)
    with pytest.raises(DataContractError, match="expected non-negative"):
        load_raw_dataset(source)


def test_primary_features_exclude_target_protected_and_identifiers() -> None:
    columns = list(EXPECTED_COLUMNS) + [ANALYTICAL_TARGET]
    features = primary_feature_columns(columns)
    assert RAW_TARGET not in features
    assert ANALYTICAL_TARGET not in features
    assert not set(AUDIT_ONLY_COLUMNS) & set(features)
    assert "loan_amount_000s" in features


def test_repository_data_configuration_matches_contract() -> None:
    config = load_data_config("config/data.yaml")
    assert config.raw_path.as_posix() == "hmda_classification_stratified_500k.csv"
    assert config.seed == 20260809
    assert (
        config.train_fraction + config.validation_fraction + config.test_fraction
    ) == pytest.approx(1.0)


def test_split_is_deterministic_disjoint_stratified_and_persisted(tmp_path) -> None:
    modeling = build_modeling_view(make_frame(200))
    first = make_stratified_split(modeling, seed=20260809)
    second = make_stratified_split(modeling, seed=20260809)

    assert np.array_equal(first.train, second.train)
    assert np.array_equal(first.validation, second.validation)
    assert np.array_equal(first.test, second.test)
    assert (len(first.train), len(first.validation), len(first.test)) == (120, 40, 40)
    first.validate(modeling)
    for indices in (first.train, first.validation, first.test):
        prevalence = modeling.loc[indices, ANALYTICAL_TARGET].mean()
        assert prevalence == pytest.approx(modeling[ANALYTICAL_TARGET].mean(), abs=0.025)

    manifest_path = persist_split_manifest(
        first, modeling, tmp_path, source_sha256="a" * 64
    )
    manifest_before = manifest_path.read_bytes()
    persisted = load_split_manifest(tmp_path)
    assert np.array_equal(persisted.train, first.train)
    assert np.array_equal(persisted.validation, first.validation)
    assert np.array_equal(persisted.test, first.test)
    persist_split_manifest(first, modeling, tmp_path, source_sha256="a" * 64)
    assert manifest_path.read_bytes() == manifest_before


def test_all_imbalance_strategies_are_train_only_and_reloadable(tmp_path) -> None:
    modeling = build_modeling_view(make_frame(200))
    splits = make_stratified_split(modeling)
    variants = build_dataset_variants(modeling, splits)

    weighted = variants["original_weighted"]
    assert np.array_equal(weighted.train_indices, splits.train)
    assert weighted.sample_weights is not None
    weighted_totals = {
        label: weighted.sample_weights[weighted.train_target(modeling) == label].sum()
        for label in (0, 1)
    }
    assert weighted_totals[0] == pytest.approx(weighted_totals[1])

    for name in ("oversampled", "undersampled"):
        variant = variants[name]
        assert variant.class_counts(modeling)[0] == variant.class_counts(modeling)[1]
        assert set(variant.train_indices).issubset(set(splits.train))
    for variant in variants.values():
        assert np.array_equal(variant.validation_indices, splits.validation)
        assert np.array_equal(variant.test_indices, splits.test)

    manifest_paths = persist_variant_manifests(variants, modeling, splits, tmp_path)
    common_hashes = splits.hashes()
    assert len(manifest_paths) == 3
    for name in variants:
        manifest = json.loads(
            (tmp_path / f"variant_{name}_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["scope"] == "train_only"
        assert manifest["validation_index_sha256"] == common_hashes["validation"]
        assert manifest["test_index_sha256"] == common_hashes["test"]
        loaded = load_variant(name, tmp_path, splits)
        assert np.array_equal(loaded.train_indices, variants[name].train_indices)
        assert np.array_equal(loaded.validation_indices, splits.validation)


def test_tampered_split_indices_are_detected(tmp_path) -> None:
    modeling = build_modeling_view(make_frame(100))
    splits = make_stratified_split(modeling)
    persist_split_manifest(splits, modeling, tmp_path, source_sha256="b" * 64)
    np.save(tmp_path / "split_test.npy", np.array([999], dtype=np.int64), allow_pickle=False)
    with pytest.raises(RuntimeError, match="manifest validation"):
        load_split_manifest(tmp_path)
