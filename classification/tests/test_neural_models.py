import numpy as np
import torch

from src.models.neural import (
    ModernHopfieldClassifier,
    ModernHopfieldLayer,
    TabularMLP,
    TabularTransformer,
    set_neural_seed,
)


def test_neural_architecture_shapes_and_finite_gradients() -> None:
    set_neural_seed(11)
    features = torch.randn(8, 12)
    for model in (
        TabularMLP(12, hidden_dim=32),
        TabularTransformer(12, token_dim=8, n_heads=2, n_layers=1),
        ModernHopfieldClassifier(12, model_dim=16, memory_patterns=8),
    ):
        output = model(features)
        assert output.shape == (8,)
        output.mean().backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_hopfield_association_is_batch_preserving() -> None:
    layer = ModernHopfieldLayer(model_dim=10, memory_patterns=7)
    state = torch.randn(5, 10)
    retrieved = layer(state)
    assert retrieved.shape == state.shape


def test_seed_is_deterministic_for_initialization() -> None:
    set_neural_seed(33)
    first = TabularMLP(4, hidden_dim=16).network[0].weight.detach().cpu().numpy().copy()
    set_neural_seed(33)
    second = TabularMLP(4, hidden_dim=16).network[0].weight.detach().cpu().numpy().copy()
    assert np.array_equal(first, second)

