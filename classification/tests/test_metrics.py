import numpy as np

from src.evaluation.metrics import calibration_table, evaluate_binary, optimize_threshold


def test_metrics_label_denial_as_positive() -> None:
    y = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    result = evaluate_binary(y, probability, 0.5)
    assert result["tn"] == 2 and result["tp"] == 2
    assert result["recall_approval"] == 1.0
    assert result["recall_denial"] == 1.0
    assert result["pr_auc"] == 1.0
    assert not result["degenerate"]


def test_degenerate_one_class_is_flagged() -> None:
    y = np.array([0, 0, 1, 1])
    result = evaluate_binary(y, np.repeat(0.1, 4), 0.5)
    assert result["unique_prediction_classes"] == 1
    assert result["degenerate"]


def test_threshold_optimization_and_calibration_are_valid() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([0.05, 0.15, 0.45, 0.40, 0.70, 0.95])
    selected = optimize_threshold(y, probability, grid_size=19)
    assert 0.0 < selected.threshold < 1.0
    assert selected.metrics["unique_prediction_classes"] == 2
    table = calibration_table(y, probability, n_bins=3)
    assert table and all(0.0 <= row["observed_denial_rate"] <= 1.0 for row in table)

