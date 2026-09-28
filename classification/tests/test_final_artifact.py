from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_best_artifacts_reload_and_predict_valid_probabilities() -> None:
    raw = pd.read_csv(ROOT / "hmda_classification_stratified_500k.csv", nrows=16)
    features = raw.drop(columns=["loan_approved"])
    for name in ("best_predictive_model.joblib", "best_practical_model.joblib"):
        artifact = joblib.load(ROOT / "artifacts" / "models" / name)
        probability = artifact.predict_proba(features)
        prediction = artifact.predict(features)
        assert probability.shape == (16, 2)
        assert prediction.shape == (16,)
        assert np.isfinite(probability).all()
        assert np.allclose(probability.sum(axis=1), 1.0)
        assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_screening_predictions_are_not_one_class() -> None:
    ledger = pd.read_csv(ROOT / "reports" / "EXPERIMENT_RESULTS.csv")
    screening = ledger[
        (ledger["status"] == "completed")
        & (ledger["sample_size"] == 60000)
        & (ledger["experiment_id"].str.contains("_v", regex=False))
    ]
    assert len(screening) == 42
    assert (screening["unique_prediction_classes"] >= 2).all()
    non_dummy = screening[screening["model"] != "dummy"]
    dummy = screening[screening["model"] == "dummy"]
    assert not non_dummy["degenerate"].astype(bool).any()
    # The detector must be able to flag chance-level baselines even when their
    # randomized predictions happen to contain both labels.
    assert dummy["degenerate"].astype(bool).any()
