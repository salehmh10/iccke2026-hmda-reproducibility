import numpy as np

from src.models.artifact import _calibrate


def test_no_calibrator_preserves_probability() -> None:
    probability = np.array([0.1, 0.8])
    assert np.array_equal(_calibrate(None, probability), probability)

