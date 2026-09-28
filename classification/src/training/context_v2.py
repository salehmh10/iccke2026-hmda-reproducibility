"""Isolated experiment context for the selected feature-engineering V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from src.data.loader import load_and_prepare
from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.data.splitting import hash_indices, load_split_manifest
from src.data.variants import VARIANT_NAMES, DatasetVariant, load_variant
from src.features.advanced import build_advanced_feature_pipeline
from src.training.context import ExperimentContext, stratified_variant_sample


FEATURE_GENERATION = "feature_v2_onehot_numeric"
SELECTED_FAMILIES: tuple[str, ...] = ("onehot_numeric",)


def _feature_frame(modeling: object, indices: np.ndarray):
    return modeling.loc[indices].drop(columns=[RAW_TARGET, ANALYTICAL_TARGET])


def feature_contract_sha256_v2(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/features/financial.py",
        "src/features/preprocessing.py",
        "src/features/advanced.py",
        "src/training/context_v2.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    digest.update(json.dumps(SELECTED_FAMILIES).encode("utf-8"))
    return digest.hexdigest()


def build_experiment_context_v2(
    *,
    root: Path,
    max_fit_rows: int | None,
    seed: int,
    reuse_preprocessor: bool = True,
) -> ExperimentContext:
    """Fit/reload V2 preprocessing on original-weighted training rows only."""

    selection_path = root / "reports" / "generations" / "feature_v2" / "CONFIRMED_FEATURE_CONFIG.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if tuple(selection["selected_families"]) != SELECTED_FAMILIES:
        raise RuntimeError("confirmed V2 selection does not match the executable feature contract")
    if selection.get("test_rows_used") is not False:
        raise RuntimeError("V2 feature selection must be train-only")

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
    contract_hash = feature_contract_sha256_v2(root)
    preprocessor_dir = root / "artifacts" / "generations" / "feature_v2" / "preprocessors"
    preprocessor_name = (
        f"v2_s{seed}_n{max_fit_rows}_{contract_hash[:8]}_"
        f"{prepared.source_sha256[:8]}_{hash_indices(splits.train)[:8]}.joblib"
    )
    preprocessor_path = preprocessor_dir / preprocessor_name
    metadata_path = preprocessor_path.with_suffix(".metadata.json")
    expected_metadata = {
        "feature_generation": FEATURE_GENERATION,
        "selected_families": list(SELECTED_FAMILIES),
        "source_sha256": prepared.source_sha256,
        "train_split_sha256": hash_indices(splits.train),
        "fit_indices_sha256": hash_indices(fit_indices),
        "fit_rows": int(len(fit_indices)),
        "seed": int(seed),
        "feature_contract_sha256": contract_hash,
        "artifact_file": preprocessor_path.name,
        "test_rows_used_for_fit": False,
    }
    if reuse_preprocessor and preprocessor_path.exists() and metadata_path.exists():
        actual_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if actual_metadata != expected_metadata:
            raise RuntimeError("V2 preprocessor metadata mismatch; refusing stale reuse")
        preprocessor = joblib.load(preprocessor_path)
    else:
        preprocessor = build_advanced_feature_pipeline(
            families=SELECTED_FAMILIES, strict_schema=True
        )
        preprocessor.fit(_feature_frame(prepared.modeling, fit_indices))
        preprocessor_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        metadata_path.write_text(
            json.dumps(expected_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    validation_indices = splits.validation
    validation_features = preprocessor.transform(
        _feature_frame(prepared.modeling, validation_indices)
    )
    validation_target = prepared.modeling.loc[
        validation_indices, ANALYTICAL_TARGET
    ].to_numpy(dtype=np.int8)
    return ExperimentContext(
        prepared=prepared,
        splits=splits,
        variants=variants,
        preprocessor=preprocessor,
        validation_features=validation_features,
        validation_target=validation_target,
        validation_indices=validation_indices,
        contract_metadata={
            "feature_generation": FEATURE_GENERATION,
            "selected_families": list(SELECTED_FAMILIES),
            "source_sha256": prepared.source_sha256,
            "split_index_sha256": splits.hashes(),
            "feature_contract_sha256": contract_hash,
            "preprocessor_fit_indices_sha256": hash_indices(fit_indices),
            "preprocessor_fit_rows": int(len(fit_indices)),
            "seed": int(seed),
            "test_rows_used_for_feature_selection_or_fit": False,
        },
    )


def transform_variant_training_v2(
    context: ExperimentContext,
    variant: DatasetVariant,
    *,
    max_rows: int | None,
    seed: int,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray | None]:
    indices, weights = stratified_variant_sample(
        context.prepared.modeling, variant, max_rows=max_rows, seed=seed
    )
    features = context.preprocessor.transform(
        _feature_frame(context.prepared.modeling, indices)
    )
    target = context.prepared.modeling.loc[indices, ANALYTICAL_TARGET].to_numpy(
        dtype=np.int8
    )
    return features, target, indices, weights
