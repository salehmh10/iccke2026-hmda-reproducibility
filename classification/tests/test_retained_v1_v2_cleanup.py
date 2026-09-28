from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.classical import denial_probability


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rejected_generation_and_obsolete_archives_are_absent() -> None:
    for relative in (
        "artifacts/generations/hpo_v3",
        "reports/generations/hpo_v3",
        "experiments/hpo_v3",
        "artifacts/archive_30k",
        "artifacts/archive_prefingerprint",
        "src/models/tuned.py",
    ):
        assert not (ROOT / relative).exists()


def test_only_retained_v1_model_files_remain() -> None:
    expected = {
        ".gitkeep",
        "best_practical_model.joblib",
        "best_predictive_model.joblib",
        "finalist_selection.json",
        "FINAL_ARTIFACT_MANIFEST.json",
        "FINAL_GENERATION.lock.json",
        "ml_catboost_undersampled_s20260809_n60000_ve7e17f0120.joblib",
        "ml_catboost_undersampled_s20260809_n60000_ve7e17f0120.provenance.json",
    }
    assert {path.name for path in (ROOT / "artifacts/models").iterdir() if path.is_file()} == expected
    assert {path.name for path in (ROOT / "artifacts/predictions").iterdir() if path.is_file()} == {
        "best_predictive_hybrid_test.npz",
        "best_single_catboost_test.npz",
        "best_practical_lightgbm_test.npz",
    }


def test_retained_v1_best_mcc_catboost_is_runnable() -> None:
    bundle = joblib.load(
        ROOT / "artifacts/models/ml_catboost_undersampled_s20260809_n60000_ve7e17f0120.joblib"
    )
    raw = pd.read_csv(ROOT / "hmda_classification_stratified_500k.csv", nrows=16)
    encoded = bundle["preprocessor"].transform(raw.drop(columns=["loan_approved"]))
    probability = denial_probability(bundle["model"], encoded)
    assert probability.shape == (16,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_complete_v2_generation_is_still_present() -> None:
    artifact = ROOT / "artifacts/generations/feature_v2"
    report = ROOT / "reports/generations/feature_v2"
    assert len([path for path in artifact.rglob("*") if path.is_file()]) == 138
    assert len([path for path in report.rglob("*") if path.is_file()]) == 29
    for name in (
        "best_predictive_model.joblib",
        "best_single_model.joblib",
        "best_practical_model.joblib",
        "FINAL_GENERATION.lock.json",
    ):
        assert (artifact / name).is_file()


def test_retained_model_manifest_hashes_match() -> None:
    manifest = json.loads((ROOT / "artifacts/RETAINED_MODELS_MANIFEST.json").read_text(encoding="utf-8"))
    for item in manifest["files"] + manifest["generation_locks"]:
        path = ROOT / item["path"]
        assert path.is_file()
        assert _sha256(path) == item["sha256"]
