import numpy as np
import pytest

from src.models.classical import SUPPORTED_CLASSICAL_MODELS, denial_probability, make_classical_model


@pytest.mark.parametrize("name", ["dummy", "logistic_regression", "random_forest", "extra_trees", "bagging"])
def test_sklearn_model_smoke(name: str) -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(80, 6))
    y = ((x[:, 0] + 0.5 * x[:, 1]) > 0).astype(int)
    model = make_classical_model(name, seed=9, n_jobs=1)
    model.fit(x, y)
    probability = denial_probability(model, x)
    assert probability.shape == (80,)
    assert np.isfinite(probability).all()
    assert ((0 <= probability) & (probability <= 1)).all()


def test_supported_models_are_unique() -> None:
    assert len(SUPPORTED_CLASSICAL_MODELS) == len(set(SUPPORTED_CLASSICAL_MODELS))

