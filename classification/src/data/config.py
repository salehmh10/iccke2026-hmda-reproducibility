"""Validated repository configuration for the HMDA data pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .loader import DataContractError
from .schema import ANALYTICAL_TARGET, PROTECTED_COLUMNS, RAW_TARGET


@dataclass(frozen=True)
class DataPipelineConfig:
    raw_path: Path
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float


def load_data_config(path: str | Path) -> DataPipelineConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataContractError("Data configuration must be a mapping")
    if payload.get("raw_target") != RAW_TARGET:
        raise DataContractError("Configured raw target does not match the schema contract")
    if payload.get("analytical_target") != ANALYTICAL_TARGET:
        raise DataContractError("Configured analytical target does not match the schema contract")
    if payload.get("provisional_target_mapping") != {0: 1, 1: 0}:
        raise DataContractError("Configured target mapping must derive denial as 1 - approval")
    if tuple(payload.get("protected_columns", ())) != PROTECTED_COLUMNS:
        raise DataContractError("Configured protected columns do not match the audit contract")
    split = payload.get("split", {})
    if split.get("strategy") != "stratified_random":
        raise DataContractError("This extract supports only the documented stratified fallback")
    return DataPipelineConfig(
        raw_path=Path(payload["raw_path"]),
        seed=int(payload["seed"]),
        train_fraction=float(split["train_fraction"]),
        validation_fraction=float(split["validation_fraction"]),
        test_fraction=float(split["test_fraction"]),
    )
