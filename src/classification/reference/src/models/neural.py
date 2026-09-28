"""Compact neural architectures for encoded tabular HMDA features.

NumPy must be imported before PyTorch in this environment to avoid loading two
Intel OpenMP runtimes in the unsafe order.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import sparse
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_neural_seed(seed: int) -> None:
    """Set Python, NumPy, CPU, and CUDA seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TabularMLP(nn.Module):
    """Batch-normalized MLP for a preprocessed tabular matrix."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class NumericalFeatureTokenizer(nn.Module):
    """Represent each scalar encoded feature as an independent token."""

    def __init__(self, n_features: int, token_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, token_dim))
        self.bias = nn.Parameter(torch.empty(n_features, token_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TabularTransformer(nn.Module):
    """Small FT-Transformer-style classifier over encoded feature tokens."""

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 16,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(input_dim, token_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(features)
        cls = self.cls.expand(features.shape[0], -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1))
        return self.head(self.norm(encoded[:, 0])).squeeze(-1)


class ModernHopfieldLayer(nn.Module):
    """One continuous modern-Hopfield retrieval update over learned memories.

    The softmax association is the attention-equivalent update from modern
    Hopfield networks; this project makes no claim of a third-party reference
    implementation or superiority.
    """

    def __init__(self, model_dim: int, memory_patterns: int = 32, beta: float = 1.0) -> None:
        super().__init__()
        self.keys = nn.Parameter(torch.empty(memory_patterns, model_dim))
        self.values = nn.Parameter(torch.empty(memory_patterns, model_dim))
        self.beta = beta / np.sqrt(model_dim)
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        association = torch.softmax(self.beta * state @ self.keys.T, dim=-1)
        return association @ self.values


class ModernHopfieldClassifier(nn.Module):
    """Tabular classifier with a learned modern-Hopfield associative memory."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 64,
        memory_patterns: int = 32,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_dim, model_dim), nn.GELU())
        self.memory = ModernHopfieldLayer(model_dim, memory_patterns)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        state = self.project(features)
        retrieved = self.memory(state)
        return self.head(self.norm(state + retrieved)).squeeze(-1)


@dataclass
class NeuralTrainingResult:
    """Best-state and measured training outcome."""

    model: nn.Module
    best_epoch: int
    best_validation_pr_auc: float
    runtime_seconds: float
    history: list[dict[str, float]]
    device: str


def as_dense_float32(features: Any) -> np.ndarray:
    """Convert sparse/dense encoded features to a C-contiguous float32 array."""

    dense = features.toarray() if sparse.issparse(features) else np.asarray(features)
    return np.ascontiguousarray(dense, dtype=np.float32)


def predict_neural_probability(
    model: nn.Module,
    features: Any,
    *,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    """Return denial probabilities in deterministic evaluation mode."""

    matrix = as_dense_float32(features)
    loader = DataLoader(torch.from_numpy(matrix), batch_size=batch_size, shuffle=False)
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device, non_blocking=True))
            output.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(output).astype(np.float64, copy=False)


def train_neural_model(
    model: nn.Module,
    x_train: Any,
    y_train: np.ndarray,
    x_validation: Any,
    y_validation: np.ndarray,
    *,
    seed: int,
    positive_weight: float = 1.0,
    max_epochs: int = 20,
    patience: int = 4,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    max_grad_norm: float = 5.0,
    scheduler_name: str = "none",
    device: str | None = None,
) -> NeuralTrainingResult:
    """Train with weighted BCE, validation PR-AUC early stopping, and clipping."""

    set_neural_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_x = as_dense_float32(x_train)
    validation_x = as_dense_float32(x_validation)
    train_y = np.asarray(y_train, dtype=np.float32).reshape(-1)
    validation_y = np.asarray(y_validation, dtype=np.int8).reshape(-1)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=selected_device == "cuda",
        drop_last=False,
    )

    model = model.to(selected_device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(positive_weight), device=selected_device)
    )
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if scheduler_name not in {"none", "cosine"}:
        raise ValueError("scheduler_name must be none or cosine")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        if scheduler_name == "cosine"
        else None
    )
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for features, target in loader:
            features = features.to(selected_device, non_blocking=True)
            target = target.to(selected_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            smoothed_target = target * (1.0 - label_smoothing) + 0.5 * label_smoothing
            loss = criterion(logits, smoothed_target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite neural training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * target.shape[0]
            seen += int(target.shape[0])

        if scheduler is not None:
            scheduler.step()
        validation_probability = predict_neural_probability(
            model, validation_x, device=selected_device, batch_size=batch_size * 2
        )
        validation_score = float(average_precision_score(validation_y, validation_probability))
        history.append(
            {"epoch": float(epoch), "train_loss": total_loss / seen, "validation_pr_auc": validation_score}
        )
        if validation_score > best_score + 1e-5:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("neural training produced no valid checkpoint")
    model.load_state_dict(best_state)
    return NeuralTrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_pr_auc=best_score,
        runtime_seconds=time.perf_counter() - started,
        history=history,
        device=selected_device,
    )
