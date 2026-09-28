from pathlib import Path

import pytest

from src.training.tracking import append_record, build_record, cumulative_training_seconds, experiment_exists


def test_append_only_tracking_rejects_duplicate_id(tmp_path: Path) -> None:
    ledger = tmp_path / "results.csv"
    row = build_record(
        root=tmp_path,
        experiment_id="exp-1",
        dataset_variant="original_weighted",
        sample_size=10,
        feature_set="primary",
        model="dummy",
        hyperparameters={},
        seed=7,
        class_weighting="sample_weight",
        sampler="none",
        train_runtime_seconds=1.25,
        threshold=0.5,
        status="completed",
        metrics={"mcc": 0.0},
    )
    append_record(ledger, row)
    assert cumulative_training_seconds(ledger) == 1.25
    assert experiment_exists(ledger, "exp-1")
    assert not experiment_exists(ledger, "exp")
    with pytest.raises(ValueError, match="duplicate experiment_id"):
        append_record(ledger, row)
