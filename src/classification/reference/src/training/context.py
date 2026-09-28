"""Shared, leakage-safe experiment data preparation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import PreparedData, load_and_prepare
from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.data.splitting import SplitIndices, hash_indices, load_split_manifest
from src.data.variants import DatasetVariant, VARIANT_NAMES, load_variant
from src.features.preprocessing import build_feature_pipeline


@dataclass
class ExperimentContext:
    prepared: PreparedData
    splits: SplitIndices
    variants: dict[str, DatasetVariant]
    preprocessor: object
    validation_features: object
    validation_target: np.ndarray
    validation_indices: np.ndarray
    contract_metadata: dict[str, object]


def _feature_frame(modeling: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    return modeling.loc[indices].drop(columns=[RAW_TARGET, ANALYTICAL_TARGET])


def feature_contract_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in ("src/features/financial.py", "src/features/preprocessing.py"):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stratified_variant_sample(
    modeling: pd.DataFrame,
    variant: DatasetVariant,
    *,
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return source indices and aligned weights after a reproducible cap."""

    positions = np.arange(len(variant.train_indices), dtype=np.int64)
    target = modeling.loc[variant.train_indices, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)
    if max_rows is not None and len(positions) > max_rows:
        positions, _ = train_test_split(
            positions,
            train_size=max_rows,
            stratify=target,
            random_state=seed,
            shuffle=True,
        )
        positions = np.sort(positions)
    selected_indices = variant.train_indices[positions]
    weights = variant.sample_weights[positions] if variant.sample_weights is not None else None
    return selected_indices, weights


def build_experiment_context(
    *,
    root: Path,
    max_fit_rows: int | None,
    seed: int,
    reuse_preprocessor: bool = True,
) -> ExperimentContext:
    """Load manifests and fit preprocessing strictly on original training rows."""

    prepared = load_and_prepare(root / "hmda_classification_stratified_500k.csv")
    manifest_dir = root / "data" / "manifests"
    splits = load_split_manifest(manifest_dir)
    splits.validate(prepared.modeling)
    variants = {name: load_variant(name, manifest_dir, splits) for name in VARIANT_NAMES}

    fit_indices, _ = stratified_variant_sample(
        prepared.modeling,
        variants["original_weighted"],
        max_rows=max_fit_rows,
        seed=seed,
    )
    contract_hash = feature_contract_sha256(root)
    preprocessor_name = (
        f"primary_s{seed}_n{max_fit_rows}_{contract_hash[:8]}_"
        f"{prepared.source_sha256[:8]}_{hash_indices(splits.train)[:8]}.joblib"
    )
    preprocessor_path = root / "artifacts" / "preprocessors" / preprocessor_name
    metadata_path = preprocessor_path.with_suffix(".metadata.json")
    expected_metadata = {
        "source_sha256": prepared.source_sha256,
        "train_split_sha256": hash_indices(splits.train),
        "fit_indices_sha256": hash_indices(fit_indices),
        "fit_rows": int(len(fit_indices)),
        "seed": int(seed),
        "feature_contract_sha256": contract_hash,
        "artifact_file": preprocessor_path.name,
    }
    if reuse_preprocessor and preprocessor_path.exists() and metadata_path.exists():
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual_metadata != expected_metadata:
            raise RuntimeError(
                f"preprocessor cache metadata mismatch: {metadata_path}; refusing stale reuse"
            )
        preprocessor = joblib.load(preprocessor_path)
    else:
        preprocessor = build_feature_pipeline(strict_schema=True)
        preprocessor.fit(_feature_frame(prepared.modeling, fit_indices))
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    validation_frame = _feature_frame(prepared.modeling, splits.validation)
    validation_features = preprocessor.transform(validation_frame)
    validation_target = prepared.modeling.loc[
        splits.validation, ANALYTICAL_TARGET
    ].to_numpy(dtype=np.int8)
    return ExperimentContext(
        prepared=prepared,
        splits=splits,
        variants=variants,
        preprocessor=preprocessor,
        validation_features=validation_features,
        validation_target=validation_target,
        validation_indices=splits.validation,
        contract_metadata={
            "source_sha256": prepared.source_sha256,
            "split_index_sha256": splits.hashes(),
            "feature_contract_sha256": contract_hash,
            "preprocessor_fit_indices_sha256": hash_indices(fit_indices),
            "preprocessor_fit_rows": int(len(fit_indices)),
            "seed": int(seed),
        },
    )


def transform_variant_training(
    context: ExperimentContext,
    variant: DatasetVariant,
    *,
    max_rows: int | None,
    seed: int,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray | None]:
    """Transform one sampled training variant with the shared frozen pipeline."""

    indices, weights = stratified_variant_sample(
        context.prepared.modeling, variant, max_rows=max_rows, seed=seed
    )
    frame = _feature_frame(context.prepared.modeling, indices)
    features = context.preprocessor.transform(frame)
    target = context.prepared.modeling.loc[indices, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)
    return features, target, indices, weights
