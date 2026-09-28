from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts import evaluate_feature_v2
from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.training.context_v2 import build_experiment_context_v2


ROOT = Path(__file__).resolve().parents[1]
V2_ARTIFACT = ROOT / "artifacts" / "generations" / "feature_v2"
V2_REPORT = ROOT / "reports" / "generations" / "feature_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_feature_v2_experiment_matrix_is_complete() -> None:
    with (V2_REPORT / "EXPERIMENT_RESULTS.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 42
    assert all(row["status"] == "completed" for row in rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["model"]] = counts.get(row["model"], 0) + 1
    assert len(counts) == 14
    assert set(counts.values()) == {3}


def test_feature_v2_lock_hashes_match() -> None:
    lock = json.loads((V2_ARTIFACT / "FINAL_GENERATION.lock.json").read_text(encoding="utf-8"))
    assert lock["files_frozen"] == len(lock["sha256"])
    assert lock["files_frozen"] >= 25
    for relative, expected in lock["sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_feature_v2_evaluator_refuses_reaccess_before_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        evaluate_feature_v2,
        "build_experiment_context_v2",
        lambda **_: (_ for _ in ()).throw(AssertionError("context/test must not be reached")),
    )
    with pytest.raises(RuntimeError, match="already accessed/frozen"):
        evaluate_feature_v2.main()


def test_feature_v2_packages_reload_on_validation_rows() -> None:
    context = build_experiment_context_v2(
        root=ROOT, max_fit_rows=60_000, seed=20260809, reuse_preprocessor=True
    )
    indices = context.splits.validation[:32]
    frame = context.prepared.modeling.loc[indices].drop(columns=[RAW_TARGET, ANALYTICAL_TARGET])
    for name in (
        "best_predictive_model.joblib",
        "best_single_model.joblib",
        "best_practical_model.joblib",
    ):
        artifact = joblib.load(V2_ARTIFACT / name)
        probability = artifact.predict_denial_probability(frame)
        assert probability.shape == (32,)
        assert np.isfinite(probability).all()
        assert ((probability >= 0.0) & (probability <= 1.0)).all()
