"""Reloadable inference bundles for Prompt 3 Deep models."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from .deep_metrics import inverse_target
    from .deep_preprocessing import FTPreprocessor, RealMLPPreprocessor, select_named_frame
except ImportError:  # Direct use with regression_v2/src on sys.path.
    from deep_metrics import inverse_target
    from deep_preprocessing import FTPreprocessor, RealMLPPreprocessor, select_named_frame


BUNDLE_FORMAT_VERSION = 1


def _prompt3_model_destination(path: str | Path) -> Path:
    """Reject every bundle write outside regression_v2/outputs/models/prompt3."""
    destination = Path(path).resolve()
    parts = [part.lower() for part in destination.parts]
    required = ["regression_v2", "outputs", "models", "prompt3"]
    for start in range(len(parts) - len(required) + 1):
        if parts[start : start + len(required)] == required:
            return destination
    raise ValueError("Prompt 3 bundle writes must stay under regression_v2/outputs/models/prompt3.")


def _atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary, compress=3)
    joblib.load(temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, path)


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    return int(sum(item.stat().st_size for item in root.rglob("*") if item.is_file()))


@dataclass
class BundleMetadata:
    """Scientific identity stored by both bundle families."""

    model_id: str
    family: str
    feature_names: list[str]
    feature_contract_name: str
    target_mode: str
    model_configuration: dict[str, Any]
    package_versions: dict[str, str]
    development_source_sha256: str
    train_row_hash_digest: str
    validation_row_hash_digest: str
    training_seed: int
    selected_epoch: int | None
    device: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RealMLPBundle:
    """A complete official RealMLP model with Train-fitted preprocessing."""

    metadata: BundleMetadata
    preprocessor: RealMLPPreprocessor
    model: Any

    def predict(self, raw_frame: pd.DataFrame) -> np.ndarray:
        selected = select_named_frame(raw_frame, self.metadata.feature_names)
        transformed = self.preprocessor.transform(selected)
        prediction = np.asarray(self.model.predict(transformed), dtype=np.float64).reshape(-1)
        if prediction.size != len(selected) or not np.isfinite(prediction).all():
            raise RuntimeError("RealMLP bundle produced invalid predictions.")
        return inverse_target(prediction, self.metadata.target_mode)


@dataclass
class FTTransformerBundle:
    """FT architecture, state dictionary, and full Train-fitted preprocessing."""

    metadata: BundleMetadata
    preprocessor: FTPreprocessor
    architecture: dict[str, Any]
    state_dict: dict[str, Any]
    prediction_batch_size: int = 2048

    def _build_model(self):
        import torch
        from rtdl_revisiting_models import FTTransformer

        model = FTTransformer(**self.architecture)
        model.load_state_dict(self.state_dict, strict=True)
        model.to(torch.device("cpu"))
        model.eval()
        return model

    def predict(self, raw_frame: pd.DataFrame) -> np.ndarray:
        import torch

        selected = select_named_frame(raw_frame, self.metadata.feature_names)
        numeric, categorical = self.preprocessor.transform(selected)
        model = self._build_model()
        output: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(selected), self.prediction_batch_size):
                end = min(start + self.prediction_batch_size, len(selected))
                x_cont = torch.from_numpy(numeric[start:end])
                x_cat = torch.from_numpy(categorical[start:end])
                batch = model(x_cont, x_cat).reshape(-1).detach().cpu().numpy()
                output.append(batch.astype(np.float64, copy=False))
        prediction = np.concatenate(output) if output else np.empty(0, dtype=np.float64)
        prediction = inverse_target(prediction, self.metadata.target_mode)
        if prediction.size != len(selected) or not np.isfinite(prediction).all():
            raise RuntimeError("FT-Transformer bundle produced invalid predictions.")
        return prediction


def save_realmlp_bundle(bundle: RealMLPBundle, destination: str | Path) -> dict[str, Any]:
    """Save a RealMLP directory bundle and write its manifest last."""
    directory = _prompt3_model_destination(destination)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "bundle.joblib"
    _atomic_joblib(artifact, bundle)
    reloaded = joblib.load(artifact)
    if not isinstance(reloaded, RealMLPBundle):
        raise TypeError("Reloaded RealMLP artifact has the wrong type.")
    manifest = {
        "status": "COMPLETE",
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "family": "realmlp",
        "artifact": "bundle.joblib",
        "metadata": asdict(bundle.metadata),
        "preprocessing": bundle.preprocessor.evidence(),
        "model_size_bytes": artifact.stat().st_size,
    }
    _atomic_json(directory / "manifest.json", manifest)
    manifest["bundle_size_bytes"] = directory_size_bytes(directory)
    _atomic_json(directory / "manifest.json", manifest)
    return manifest


def save_fttransformer_bundle(
    bundle: FTTransformerBundle, destination: str | Path
) -> dict[str, Any]:
    """Save FT state, preprocessing, metadata, and a final manifest."""
    import torch

    directory = _prompt3_model_destination(destination)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "model_state.pt"
    state_temporary = state_path.with_suffix(".pt.tmp")
    cpu_state = {name: tensor.detach().cpu() for name, tensor in bundle.state_dict.items()}
    torch.save(cpu_state, state_temporary)
    loaded_state = torch.load(state_temporary, map_location="cpu", weights_only=True)
    if set(loaded_state) != set(cpu_state):
        raise RuntimeError("FT state-dict reload validation failed.")
    os.replace(state_temporary, state_path)
    payload = {
        "metadata": bundle.metadata,
        "preprocessor": bundle.preprocessor,
        "architecture": bundle.architecture,
        "prediction_batch_size": bundle.prediction_batch_size,
    }
    _atomic_joblib(directory / "bundle.joblib", payload)
    manifest = {
        "status": "COMPLETE",
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "family": "fttransformer",
        "state_dict": "model_state.pt",
        "artifact": "bundle.joblib",
        "metadata": asdict(bundle.metadata),
        "architecture": bundle.architecture,
        "preprocessing": bundle.preprocessor.evidence(),
        "model_size_bytes": state_path.stat().st_size,
    }
    _atomic_json(directory / "manifest.json", manifest)
    manifest["bundle_size_bytes"] = directory_size_bytes(directory)
    _atomic_json(directory / "manifest.json", manifest)
    return manifest


def load_bundle(path: str | Path) -> RealMLPBundle | FTTransformerBundle:
    """Load either Prompt 3 bundle through one public interface."""
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError(f"Bundle manifest is not complete: {manifest_path}")
    family = manifest.get("family")
    payload = joblib.load(directory / manifest["artifact"])
    if family == "realmlp":
        if not isinstance(payload, RealMLPBundle):
            raise TypeError("RealMLP bundle payload has the wrong type.")
        return payload
    if family == "fttransformer":
        import torch

        state = torch.load(directory / manifest["state_dict"], map_location="cpu", weights_only=True)
        return FTTransformerBundle(
            metadata=payload["metadata"],
            preprocessor=payload["preprocessor"],
            architecture=payload["architecture"],
            state_dict=state,
            prediction_batch_size=int(payload["prediction_batch_size"]),
        )
    raise ValueError(f"Unknown bundle family: {family}")
