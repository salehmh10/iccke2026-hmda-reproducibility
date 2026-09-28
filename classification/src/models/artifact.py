"""Serializable final inference artifacts with explicit denial semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.classical import denial_probability
from src.models.neural import predict_neural_probability


def _calibrate(calibrator: Any | None, probability: np.ndarray) -> np.ndarray:
    raw = np.asarray(probability, dtype=np.float64).reshape(-1)
    if calibrator is None:
        return raw
    if hasattr(calibrator, "predict_proba"):
        return np.asarray(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=np.float64)
    return np.asarray(calibrator.predict(raw), dtype=np.float64)


@dataclass
class FinalSklearnArtifact:
    preprocessor: Any
    model: Any
    calibrator: Any | None
    threshold: float
    metadata: dict[str, Any]

    def predict_denial_probability(self, raw_features: pd.DataFrame) -> np.ndarray:
        encoded = self.preprocessor.transform(raw_features)
        return _calibrate(self.calibrator, denial_probability(self.model, encoded))

    def predict_proba(self, raw_features: pd.DataFrame) -> np.ndarray:
        denial = self.predict_denial_probability(raw_features)
        return np.column_stack((1.0 - denial, denial))

    def predict(self, raw_features: pd.DataFrame) -> np.ndarray:
        return (self.predict_denial_probability(raw_features) >= self.threshold).astype(np.int8)


@dataclass
class FinalHybridArtifact:
    preprocessor: Any
    ml_model: Any
    dl_model: Any
    ml_weight: float
    calibrator: Any | None
    threshold: float
    metadata: dict[str, Any]

    def predict_denial_probability(self, raw_features: pd.DataFrame) -> np.ndarray:
        encoded = self.preprocessor.transform(raw_features)
        ml_probability = denial_probability(self.ml_model, encoded)
        dl_probability = predict_neural_probability(
            self.dl_model, encoded, device="cpu", batch_size=2048
        )
        blended = self.ml_weight * ml_probability + (1.0 - self.ml_weight) * dl_probability
        return _calibrate(self.calibrator, blended)

    def predict_proba(self, raw_features: pd.DataFrame) -> np.ndarray:
        denial = self.predict_denial_probability(raw_features)
        return np.column_stack((1.0 - denial, denial))

    def predict(self, raw_features: pd.DataFrame) -> np.ndarray:
        return (self.predict_denial_probability(raw_features) >= self.threshold).astype(np.int8)

