"""Deterministic loading, validation, and de-duplication for HMDA data."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    ANALYTICAL_TARGET,
    EXPECTED_COLUMNS,
    FLOAT_COLUMNS,
    INTEGER_COLUMNS,
    RAW_TARGET,
    READ_DTYPES,
    STRING_COLUMNS,
)

NUMERIC_COLUMNS = INTEGER_COLUMNS + FLOAT_COLUMNS
SEMANTIC_MISSING_LABELS = frozenset(
    {
        "not applicable",
        "no co-applicant",
        "information not provided by applicant in mail, internet, or telephone application",
        "not available",
        "not provided",
        "unknown",
    }
)


class DataContractError(ValueError):
    """Raised when input data violates the declared data contract."""


@dataclass(frozen=True)
class DataQualitySummary:
    raw_rows: int
    raw_columns: int
    modeling_rows: int
    duplicate_rows_removed: int
    machine_null_cells: int
    blank_string_cells: int
    semantic_missing_cells: int
    target_approval_count: int
    target_denial_count: int
    constant_columns: tuple[str, ...]
    near_constant_columns: tuple[str, ...]
    negative_values: dict[str, int]
    zero_values: dict[str, int]
    cardinalities: dict[str, int]
    dtypes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedData:
    """Raw data plus a separate, de-duplicated analytical view."""

    raw: pd.DataFrame
    modeling: pd.DataFrame
    quality: DataQualitySummary
    source_sha256: str


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _validate_columns(frame: pd.DataFrame) -> None:
    actual = tuple(frame.columns)
    if actual != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(actual))
        extra = sorted(set(actual) - set(EXPECTED_COLUMNS))
        raise DataContractError(
            "CSV schema/order mismatch; "
            f"missing={missing}, extra={extra}, order_matches={not missing and not extra}"
        )


def validate_raw_frame(frame: pd.DataFrame) -> None:
    """Fail loudly on schema, missing-cell, index, or target violations."""

    _validate_columns(frame)
    if not isinstance(frame.index, pd.RangeIndex) or frame.index.start != 0 or frame.index.step != 1:
        raise DataContractError("Raw row index must be a zero-based contiguous RangeIndex")
    if not frame.index.is_unique:
        raise DataContractError("Raw row index must be unique")

    actual_dtypes = {column: str(frame[column].dtype) for column in EXPECTED_COLUMNS}
    wrong_dtypes = {
        column: {"expected": expected, "actual": actual_dtypes[column]}
        for column, expected in READ_DTYPES.items()
        if actual_dtypes[column] != expected
    }
    if wrong_dtypes:
        raise DataContractError(f"Explicit dtype contract mismatch: {wrong_dtypes}")

    null_count = int(frame.isna().sum().sum())
    if null_count:
        raise DataContractError(f"Unexpected machine-null cells: {null_count}")

    blank_count = int(
        sum(frame[column].str.strip().eq("").sum() for column in STRING_COLUMNS)
    )
    if blank_count:
        raise DataContractError(f"Unexpected blank string cells: {blank_count}")

    negative_counts = {
        column: int((frame[column] < 0).sum()) for column in NUMERIC_COLUMNS
    }
    invalid_negative_counts = {
        column: count for column, count in negative_counts.items() if count
    }
    if invalid_negative_counts:
        raise DataContractError(
            f"Financial/context fields expected non-negative: {invalid_negative_counts}"
        )
    outside_percentage = int(
        ((frame["minority_population"] < 0) | (frame["minority_population"] > 100)).sum()
    )
    if outside_percentage:
        raise DataContractError(
            f"minority_population has {outside_percentage} values outside [0, 100]"
        )

    target_values = set(frame[RAW_TARGET].astype("int8").unique().tolist())
    if target_values != {0, 1}:
        raise DataContractError(
            f"{RAW_TARGET} must contain both and only binary values {{0, 1}}; got {target_values}"
        )


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the source without inference-based type drift or source mutation."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    try:
        frame = pd.read_csv(
            source_path,
            dtype=READ_DTYPES,
            keep_default_na=False,
            na_filter=False,
        )
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"Could not parse CSV with the explicit schema: {exc}") from exc
    validate_raw_frame(frame)
    return frame


def build_modeling_view(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates and derive denial without changing ``raw``.

    The retained index is the zero-based data-row offset in the source CSV.
    This provides a stable, auditable key without introducing an identifier
    into the predictive feature set.
    """

    validate_raw_frame(raw)
    raw_columns_before = tuple(raw.columns)
    raw_shape_before = raw.shape

    modeling = raw.loc[~raw.duplicated(subset=list(EXPECTED_COLUMNS), keep="first")].copy()
    modeling[ANALYTICAL_TARGET] = (1 - modeling[RAW_TARGET]).astype("Int8")

    if raw.shape != raw_shape_before or tuple(raw.columns) != raw_columns_before:
        raise AssertionError("Raw frame was mutated while constructing modeling view")
    if modeling.duplicated(subset=list(EXPECTED_COLUMNS)).any():
        raise AssertionError("Exact duplicates remain in modeling view")
    if not modeling.index.is_unique:
        raise AssertionError("Modeling source-row indices are not unique")
    expected_target = (1 - modeling[RAW_TARGET]).astype("Int8")
    pd.testing.assert_series_equal(
        modeling[ANALYTICAL_TARGET], expected_target, check_names=False
    )
    return modeling


def summarize_quality(raw: pd.DataFrame, modeling: pd.DataFrame) -> DataQualitySummary:
    validate_raw_frame(raw)
    removed = len(raw) - len(modeling)
    if removed != int(raw.duplicated(subset=list(EXPECTED_COLUMNS), keep="first").sum()):
        raise AssertionError("Rows were lost for a reason other than exact de-duplication")
    semantic_missing_counts = {
        column: int(raw[column].str.strip().str.casefold().isin(SEMANTIC_MISSING_LABELS).sum())
        for column in STRING_COLUMNS
    }
    cardinalities = {
        column: int(raw[column].nunique(dropna=False)) for column in EXPECTED_COLUMNS
    }
    near_constant_columns = tuple(
        column
        for column in EXPECTED_COLUMNS
        if cardinalities[column] > 1
        and float(raw[column].value_counts(dropna=False, normalize=True).iloc[0]) >= 0.99
    )
    return DataQualitySummary(
        raw_rows=len(raw),
        raw_columns=len(raw.columns),
        modeling_rows=len(modeling),
        duplicate_rows_removed=removed,
        machine_null_cells=int(raw.isna().sum().sum()),
        blank_string_cells=int(
            sum(raw[column].str.strip().eq("").sum() for column in STRING_COLUMNS)
        ),
        semantic_missing_cells=sum(semantic_missing_counts.values()),
        target_approval_count=int((raw[RAW_TARGET] == 1).sum()),
        target_denial_count=int((raw[RAW_TARGET] == 0).sum()),
        constant_columns=tuple(
            column for column in EXPECTED_COLUMNS if raw[column].nunique(dropna=False) <= 1
        ),
        near_constant_columns=near_constant_columns,
        negative_values={
            column: int((raw[column] < 0).sum()) for column in NUMERIC_COLUMNS
        },
        zero_values={
            column: int((raw[column] == 0).sum()) for column in NUMERIC_COLUMNS
        },
        cardinalities=cardinalities,
        dtypes={column: str(raw[column].dtype) for column in EXPECTED_COLUMNS},
    )


def load_and_prepare(path: str | Path) -> PreparedData:
    source_path = Path(path)
    source_sha256 = sha256_file(source_path)
    raw = load_raw_dataset(source_path)
    modeling = build_modeling_view(raw)
    quality = summarize_quality(raw, modeling)
    return PreparedData(
        raw=raw,
        modeling=modeling,
        quality=quality,
        source_sha256=source_sha256,
    )
