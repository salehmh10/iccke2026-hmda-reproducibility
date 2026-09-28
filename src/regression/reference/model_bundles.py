"""Reloadable model bundle contract for Regression V2 Prompt 2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from .metrics import inverse_target
except ImportError:  # Direct script use with regression_v2/src on sys.path.
    from metrics import inverse_target


@dataclass
class ModelBundle:
    model_id: str
    family: str
    feature_names: list[str]
    feature_contract_name: str
    target_mode: str
    preprocessor: Any
    model: Any
    package_versions: dict[str, str]
    model_parameters: dict[str, Any]
    selected_best_iteration: int | None
    development_source_sha256: str
    train_row_hash_digest: str
    validation_row_hash_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def _select(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(raw_frame, pd.DataFrame):
            raise TypeError("ModelBundle.predict requires a pandas DataFrame.")
        missing = sorted(set(self.feature_names) - set(raw_frame.columns))
        if missing:
            raise ValueError(f"Missing required model features: {missing}")
        return raw_frame.loc[:, self.feature_names]

    def predict_model_scale(self, raw_frame: pd.DataFrame) -> np.ndarray:
        selected = self._select(raw_frame)
        transformed = self.preprocessor.transform(selected)
        prediction = np.asarray(self.model.predict(transformed), dtype=np.float64).reshape(-1)
        if prediction.shape[0] != len(raw_frame):
            raise RuntimeError("Model prediction did not preserve row count.")
        return prediction

    def predict(self, raw_frame: pd.DataFrame) -> np.ndarray:
        prediction = inverse_target(self.predict_model_scale(raw_frame), self.target_mode)
        if not np.isfinite(prediction).all():
            raise RuntimeError("Model bundle produced non-finite predictions.")
        return prediction


def atomic_joblib_dump(bundle: ModelBundle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(bundle, temporary, compress=3)
    reloaded = joblib.load(temporary)
    if not isinstance(reloaded, ModelBundle):
        raise TypeError("Reloaded artifact is not a ModelBundle.")
    os.replace(temporary, destination)
    return destination


def load_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"Artifact is not a ModelBundle: {path}")
    return bundle
