"""Data loading, validation, splitting, and train-only sampling."""

from .config import DataPipelineConfig, load_data_config
from .loader import (
    DataContractError,
    DataQualitySummary,
    PreparedData,
    build_modeling_view,
    load_and_prepare,
    load_raw_dataset,
)
from .schema import (
    ANALYTICAL_TARGET,
    AUDIT_ONLY_COLUMNS,
    EXPECTED_COLUMNS,
    PRIMARY_EXCLUDED_COLUMNS,
    PROTECTED_COLUMNS,
    RAW_TARGET,
    primary_feature_columns,
)
from .splitting import (
    DEFAULT_SEED,
    SplitIndices,
    load_split_manifest,
    make_stratified_split,
    persist_split_manifest,
)
from .variants import (
    DatasetVariant,
    build_dataset_variants,
    load_variant,
    persist_variant_manifests,
)

__all__ = [
    "ANALYTICAL_TARGET",
    "AUDIT_ONLY_COLUMNS",
    "DEFAULT_SEED",
    "DataContractError",
    "DataPipelineConfig",
    "DataQualitySummary",
    "DatasetVariant",
    "EXPECTED_COLUMNS",
    "PRIMARY_EXCLUDED_COLUMNS",
    "PROTECTED_COLUMNS",
    "PreparedData",
    "RAW_TARGET",
    "SplitIndices",
    "build_dataset_variants",
    "build_modeling_view",
    "load_and_prepare",
    "load_data_config",
    "load_raw_dataset",
    "load_split_manifest",
    "load_variant",
    "make_stratified_split",
    "persist_split_manifest",
    "persist_variant_manifests",
    "primary_feature_columns",
]
