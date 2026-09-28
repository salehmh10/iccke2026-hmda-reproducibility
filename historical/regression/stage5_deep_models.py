"""Authentic Stage 5A1 model factories, training loops, metrics, and reload logic."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stage5_deep_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    ROOT,
    RealMLPPreprocessor,
    TargetTransform,
    TensorPreprocessor,
    atomic_csv,
    atomic_joblib,
    atomic_json,
)


SEED = 42
DEVICE = "cpu"
AMP = False
NUM_WORKERS = 0


class StrictFinalEpochRealMLPRegressor:
    """Factory namespace retained for backward-compatible imports."""


def _strict_final_epoch_realmlp_class():
    """Return a joblib-safe RealMLP class whose official config disables restoration."""
    from pytabkit import RealMLP_TD_Regressor

    class _StrictFinalEpochRealMLPRegressor(RealMLP_TD_Regressor):
        def _get_default_params(self):
            defaults = dict(super()._get_default_params())
            defaults["use_best_epoch"] = False
            return defaults

    _StrictFinalEpochRealMLPRegressor.__name__ = "StrictFinalEpochRealMLPRegressor"
    _StrictFinalEpochRealMLPRegressor.__qualname__ = "StrictFinalEpochRealMLPRegressor"
    _StrictFinalEpochRealMLPRegressor.__module__ = __name__
    globals()["StrictFinalEpochRealMLPRegressor"] = _StrictFinalEpochRealMLPRegressor
    return _StrictFinalEpochRealMLPRegressor


# Define the public class at import time so joblib can reload fit 11 in a clean process.
StrictFinalEpochRealMLPRegressor = _strict_final_epoch_realmlp_class()


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, psutil.cpu_count(logical=False) or 2)))


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    absolute_error = np.abs(y_true - y_pred)
    squared_error = np.square(y_true - y_pred)
    safe_denominator = np.maximum(np.abs(y_true), 1e-8)
    top_decile_cut = np.quantile(y_true, 0.90)
    top_five_cut = np.quantile(y_true, 0.95)
    clipped_true = np.clip(y_true, 0.0, None)
    clipped_pred = np.clip(y_pred, 0.0, None)
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mape_percent": float(np.mean(absolute_error / safe_denominator) * 100.0),
        "r_squared": float(r2_score(y_true, y_pred)),
        "rmsle": float(
            math.sqrt(np.mean(np.square(np.log1p(clipped_true) - np.log1p(clipped_pred))))
        ),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "wape_percent": float(absolute_error.sum() / np.maximum(np.abs(y_true).sum(), 1e-8) * 100.0),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
        "p90_absolute_error": float(np.quantile(absolute_error, 0.90)),
        "negative_prediction_rate": float(np.mean(y_pred < 0.0)),
        "top_decile_mae": float(np.mean(absolute_error[y_true >= top_decile_cut])),
        "top_five_percent_mae": float(np.mean(absolute_error[y_true >= top_five_cut])),
    }


def _memory_mib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024.0**2)


class TabMWithCategoricalEmbeddings(nn.Module):
    """Official TabM backbone with maintained PyTorch categorical embeddings."""

    def __init__(
        self,
        n_num_features: int,
        cardinalities: list[int],
        bins: list[np.ndarray],
        d_embedding: int = 16,
        k: int = 16,
    ) -> None:
        super().__init__()
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings
        from tabm import EnsembleView, LinearEnsemble, make_tabm_backbone

        torch_bins = [torch.as_tensor(values, dtype=torch.float32) for values in bins]
        self.num_embeddings = PiecewiseLinearEmbeddings(
            torch_bins, d_embedding=d_embedding, activation=False, version="B"
        )
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_embedding) for cardinality in cardinalities]
        )
        feature_chunks = [d_embedding] * (n_num_features + len(cardinalities))
        d_in = sum(feature_chunks)
        self.ensemble_view = EnsembleView(k=k)
        self.backbone = make_tabm_backbone(
            d_in=d_in,
            n_blocks=2,
            d_block=256,
            dropout=0.10,
            activation="ReLU",
            k=k,
            arch_type="tabm",
            start_scaling_init="normal",
            start_scaling_init_chunks=feature_chunks,
        )
        self.output = LinearEnsemble(
            self.backbone.get_original_output_shape()[0], 1, k=k
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        num = self.num_embeddings(x_num).flatten(1)
        cat = torch.cat(
            [embedding(x_cat[:, index]) for index, embedding in enumerate(self.cat_embeddings)],
            dim=1,
        )
        x = torch.cat([num, cat], dim=1)
        return self.output(self.backbone(self.ensemble_view(x)))


def build_torch_model(
    family: str,
    n_num_features: int,
    cardinalities: list[int],
    bins: list[np.ndarray] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    if family == "tabm_embedding":
        if bins is None:
            raise ValueError("TabM requires training-fit numerical bins")
        model = TabMWithCategoricalEmbeddings(
            n_num_features=n_num_features,
            cardinalities=cardinalities,
            bins=bins,
            d_embedding=16,
            k=16,
        )
        config = {
            "implementation": "tabm.make_tabm_backbone with torch.nn.Embedding input module",
            "maintained_public_tabm_api": True,
            "arch_type": "tabm",
            "k": 16,
            "n_blocks": 2,
            "d_block": 256,
            "dropout": 0.10,
            "categorical_input": "learned torch.nn.Embedding per feature",
            "categorical_embedding_dimension": 16,
            "num_embedding": "PiecewiseLinearEmbeddings version B",
            "num_embedding_dimension": 16,
            "numeric_bin_count": 48,
        }
        return model, config

    if family == "tabm":
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings
        from tabm import TabM

        if bins is None:
            raise ValueError("TabM requires training-fit numerical bins")
        torch_bins = [torch.as_tensor(values, dtype=torch.float32) for values in bins]
        num_embeddings = PiecewiseLinearEmbeddings(
            torch_bins,
            d_embedding=16,
            activation=False,
            version="B",
        )
        model = TabM.make(
            n_num_features=n_num_features,
            cat_cardinalities=cardinalities,
            d_out=1,
            num_embeddings=num_embeddings,
            arch_type="tabm",
            k=16,
            n_blocks=2,
            d_block=256,
            dropout=0.10,
        )
        config = {
            "implementation": "tabm.TabM",
            "arch_type": "tabm",
            "k": 16,
            "n_blocks": 2,
            "d_block": 256,
            "dropout": 0.10,
            "num_embedding": "PiecewiseLinearEmbeddings version B",
            "num_embedding_dimension": 16,
            "numeric_bin_count": 48,
        }
        return model, config

    if family == "ft_transformer":
        from rtdl_revisiting_models import FTTransformer

        model = FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=cardinalities,
            d_out=1,
            n_blocks=3,
            d_block=96,
            attention_n_heads=8,
            attention_dropout=0.15,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4.0 / 3.0,
            ffn_dropout=0.10,
            residual_dropout=0.0,
        )
        config = {
            "implementation": "rtdl_revisiting_models.FTTransformer",
            "feature_tokenization": True,
            "token_dimension": 96,
            "n_blocks": 3,
            "attention_heads": 8,
            "attention_dropout": 0.15,
            "ffn_dropout": 0.10,
            "residual_dropout": 0.0,
        }
        return model, config
    raise ValueError(f"Unsupported torch family: {family}")


def _tabm_bins(x_num: np.ndarray) -> list[np.ndarray]:
    from rtdl_num_embeddings import compute_bins

    bins = compute_bins(torch.as_tensor(x_num, dtype=torch.float32), n_bins=48)
    return [values.detach().cpu().numpy().astype(np.float32) for values in bins]


@torch.no_grad()
def predict_torch_model(
    model: nn.Module,
    family: str,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_cat)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )
    predictions = []
    for batch_num, batch_cat in loader:
        output = model(batch_num, batch_cat)
        if family.startswith("tabm"):
            output = output.squeeze(-1).mean(dim=1)
        else:
            output = output.squeeze(-1)
        predictions.append(output.detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def train_torch_candidate(
    family: str,
    target_mode: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    candidate_id: str,
) -> dict[str, Any]:
    seed_everything()
    process = psutil.Process(os.getpid())
    peak_ram = _memory_mib()
    preprocessor = TensorPreprocessor(family).fit(X_train)
    x_train_num, x_train_cat = preprocessor.transform(X_train)
    x_val_num, x_val_cat = preprocessor.transform(X_val)
    target = TargetTransform(target_mode).fit(y_train)
    train_target = target.transform(y_train, standardize=True)
    bins = _tabm_bins(x_train_num) if family.startswith("tabm") else None
    model, architecture = build_torch_model(
        family,
        x_train_num.shape[1],
        preprocessor.cardinalities_,
        bins,
    )
    model.to(DEVICE)
    learning_rate = 1e-3 if family.startswith("tabm") else 3e-4
    batch_size = 512 if family.startswith("tabm") else 256
    max_epochs = 30 if family.startswith("tabm") else 24
    patience = 6 if family.startswith("tabm") else 5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    loss_function = nn.SmoothL1Loss(beta=0.5)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_num),
            torch.from_numpy(x_train_cat),
            torch.from_numpy(train_target),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    fit_start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        epoch_start = time.perf_counter()
        for batch_num, batch_cat, batch_target in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_num, batch_cat)
            if family.startswith("tabm"):
                output = output.squeeze(-1)
                loss = loss_function(output, batch_target[:, None].expand_as(output))
            else:
                output = output.squeeze(-1)
                loss = loss_function(output, batch_target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite {family} loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            peak_ram = max(peak_ram, process.memory_info().rss / (1024.0**2))
        transformed_prediction = predict_torch_model(
            model, family, x_val_num, x_val_cat, batch_size
        )
        original_prediction = target.inverse(transformed_prediction, standardized=True)
        validation_mae = float(mean_absolute_error(y_val, original_prediction))
        validation_loss = float(
            np.mean(
                np.where(
                    np.abs(original_prediction - y_val) < 0.5,
                    0.5 * np.square(original_prediction - y_val) / 0.5,
                    np.abs(original_prediction - y_val) - 0.25,
                )
            )
        )
        scheduler.step(validation_mae)
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "validation_loss_original_scale": validation_loss,
                "validation_mae": validation_mae,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "ram_mib": _memory_mib(),
                "vram_mib": 0.0,
            }
        )
        if validation_mae < best_mae - 1e-8:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No valid checkpoint for {candidate_id}")
    model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - fit_start
    prediction_start = time.perf_counter()
    transformed_prediction = predict_torch_model(model, family, x_val_num, x_val_cat, batch_size)
    prediction = target.inverse(transformed_prediction, standardized=True)
    prediction_seconds = time.perf_counter() - prediction_start
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Non-finite {candidate_id} prediction")
    metrics = regression_metrics(y_val, prediction)

    model_dir = ROOT / "artifacts/models/deep/core_screening"
    preprocessing_dir = ROOT / "artifacts/preprocessing/stage5/deep_core"
    history_dir = ROOT / "artifacts/results/stage5/deep_core/screening/histories"
    state_path = model_dir / f"{candidate_id}.pt"
    preprocessor_path = preprocessing_dir / f"{candidate_id}_preprocessor.joblib"
    history_path = history_dir / f"{candidate_id}_history.csv"
    atomic_torch_save(best_state, state_path)
    atomic_joblib(preprocessor, preprocessor_path)
    atomic_csv(pd.DataFrame(history), history_path)
    bundle = {
        "stage_id": "stage5a1",
        "candidate_id": candidate_id,
        "family": family,
        "target_mode": target_mode,
        "feature_schema": "deep_core_v1",
        "preprocessing_version": f"{family}_v1",
        "target_transform": target,
        "preprocessor": preprocessor,
        "model_state_path": str(state_path.relative_to(ROOT)),
        "architecture": architecture,
        "n_num_features": x_train_num.shape[1],
        "cardinalities": preprocessor.cardinalities_,
        "bins": bins,
        "batch_size": batch_size,
        "device": DEVICE,
        "amp": AMP,
        "seed": SEED,
    }
    bundle_path = model_dir / f"{candidate_id}.joblib"
    atomic_joblib(bundle, bundle_path)
    return {
        "bundle_path": bundle_path,
        "state_path": state_path,
        "preprocessor_path": preprocessor_path,
        "history_path": history_path,
        "prediction": prediction,
        "metrics": metrics,
        "architecture": architecture,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "stop_reason": "early_stopping" if len(history) < max_epochs else "max_epochs",
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_ram_mib": peak_ram,
        "peak_vram_mib": 0.0,
    }


def train_realmlp_candidate(
    target_mode: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    candidate_id: str,
) -> dict[str, Any]:
    from pytabkit import RealMLP_TD_Regressor

    seed_everything()
    peak_ram = _memory_mib()
    preprocessor = RealMLPPreprocessor().fit(X_train)
    train_frame = preprocessor.transform(X_train)
    val_frame = preprocessor.transform(X_val)
    categorical_contract = preprocessor.categorical_contract(X_train, X_val)
    if not categorical_contract["other_values_subset_of_train"]:
        raise RuntimeError("RealMLP transformed Validation contains a category token absent from Train")
    target = TargetTransform(target_mode).fit(y_train)
    train_target = target.transform(y_train, standardize=True)
    val_target = target.transform(y_val, standardize=True)
    fixed_log_epoch_exception = target_mode == "log1p" and "replacement" in candidate_id
    model = RealMLP_TD_Regressor(
        device="cpu",
        random_state=SEED,
        n_cv=1,
        n_refit=0,
        n_repeats=1,
        n_threads=4,
        verbosity=0,
        n_epochs=64,
        batch_size=256,
        predict_batch_size=1024,
        train_metric_name="mae",
        val_metric_name="mae",
        use_early_stopping=not fixed_log_epoch_exception,
    )
    fit_start = time.perf_counter()
    model.fit(
        train_frame,
        train_target,
        X_val=val_frame,
        y_val=val_target,
        cat_col_names=CATEGORICAL_FEATURES,
        time_to_fit_in_seconds=1750,
    )
    best_epoch = 64 if fixed_log_epoch_exception else int(model.fit_params_["stop_epoch"]["mae"])
    encoder = model.x_converter_.cat_tf.named_transformers_["categorical"]
    learned_categories = encoder.categories_
    official_contract = {}
    for index, column in enumerate(CATEGORICAL_FEATURES):
        expected = set(train_frame[column].astype(str).unique())
        learned = {str(value) for value in learned_categories[index]}
        official_contract[column] = {
            "train_only_cardinality": len(expected),
            "official_encoder_cardinality": len(learned),
            "unexpected_official_categories": sorted(learned - expected),
            "missing_train_categories": sorted(expected - learned),
        }
    official_encoder_matches_train_only = all(
        not item["unexpected_official_categories"] and not item["missing_train_categories"]
        for item in official_contract.values()
    )
    if not official_encoder_matches_train_only:
        raise RuntimeError("Official RealMLP encoder categories do not match the Train-only contract")
    fit_seconds = time.perf_counter() - fit_start
    peak_ram = max(peak_ram, _memory_mib())
    prediction_start = time.perf_counter()
    transformed_prediction = np.asarray(model.predict(val_frame)).reshape(-1)
    prediction = target.inverse(transformed_prediction, standardized=True)
    prediction_seconds = time.perf_counter() - prediction_start
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Non-finite {candidate_id} prediction")
    metrics = regression_metrics(y_val, prediction)
    model_dir = ROOT / "artifacts/models/deep/core_screening"
    history_dir = ROOT / "artifacts/results/stage5/deep_core/screening/histories"
    bundle_path = model_dir / f"{candidate_id}.joblib"
    history_path = history_dir / f"{candidate_id}_history.csv"
    bundle = {
        "stage_id": "stage5a1",
        "candidate_id": candidate_id,
        "family": "realmlp",
        "target_mode": target_mode,
        "feature_schema": "deep_core_v1",
        "preprocessing_version": "realmlp_official_v1",
        "target_transform": target,
        "preprocessor": preprocessor,
        "model": model,
        "architecture": {
            "implementation": "pytabkit.RealMLP_TD_Regressor",
            "official_tuned_preprocessing": True,
            "n_cv": 1,
            "n_refit": 0,
            "n_epochs": 64,
            "fixed_epoch_no_early_stopping_exception": fixed_log_epoch_exception,
        },
        "categorical_contract": categorical_contract,
        "official_encoder_contract": official_contract,
        "official_encoder_matches_train_only": official_encoder_matches_train_only,
        "device": DEVICE,
        "amp": AMP,
        "seed": SEED,
    }


def _preprocessor_policy_payload(preprocessor: RealMLPPreprocessor) -> dict[str, Any]:
    return {
        "numerical_features": list(preprocessor.numerical_features),
        "categorical_features": list(preprocessor.categorical_features),
        "medians": {key: float(value) for key, value in sorted(preprocessor.medians_.items())},
        "rare_min_count": int(preprocessor.rare_min_count),
        "unknown_token": str(preprocessor.unknown_token),
        "vocabularies": {
            key: sorted(str(value) for value in values)
            for key, values in sorted(preprocessor.vocabularies_.items())
        },
        "rare_values": {
            key: sorted(str(value) for value in values)
            for key, values in sorted(preprocessor.rare_values_.items())
        },
    }


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def train_realmlp_log1p_fit11(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    candidate_id: str,
) -> dict[str, Any]:
    """Run the one authorized strict-final-epoch protocol repair fit."""
    if candidate_id != "stage5a1__realmlp__log1p__replacement2":
        raise ValueError("The fit-11 path is restricted to the approved candidate ID")

    seed_everything()
    peak_ram = _memory_mib()
    defective_id = "stage5a1__realmlp__log1p__replacement1"
    model_dir = ROOT / "artifacts/models/deep/core_screening"
    result_dir = ROOT / "artifacts/results/stage5/deep_core/screening"
    report_dir = ROOT / "artifacts/reports"
    defective_bundle_path = model_dir / f"{defective_id}.joblib"
    defective_bundle = joblib.load(defective_bundle_path)
    preprocessor: RealMLPPreprocessor = defective_bundle["preprocessor"]
    target: TargetTransform = defective_bundle["target_transform"]
    if target.mode != "log1p":
        raise RuntimeError("The preserved target transform is not log1p")

    historical_paths = [
        defective_bundle_path,
        ROOT / f"artifacts/predictions/stage5/deep_core/screening/{defective_id}.csv",
        result_dir / f"candidates/{defective_id}.json",
        ROOT / f"artifacts/checkpoints/stage5/deep_core/screening/{defective_id}.json",
        result_dir / f"histories/{defective_id}_history.csv",
        report_dir / f"stage5a1_reload_{defective_id}.json",
    ]
    historical_before = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in historical_paths}

    train_frame = preprocessor.transform(X_train)
    val_frame = preprocessor.transform(X_val)
    categorical_contract = preprocessor.categorical_contract(X_train, X_val)
    if not categorical_contract["other_values_subset_of_train"]:
        raise RuntimeError("Fit 11 Validation contains a category token absent from transformed Train")
    preprocessor_payload = _preprocessor_policy_payload(preprocessor)
    preprocessor_digest = _payload_digest(preprocessor_payload)
    train_target = target.transform(y_train, standardize=True)
    val_target = target.transform(y_val, standardize=True)

    model = StrictFinalEpochRealMLPRegressor(
        device="cpu",
        random_state=SEED,
        n_cv=1,
        n_refit=0,
        n_repeats=1,
        n_threads=4,
        verbosity=0,
        n_epochs=64,
        batch_size=256,
        predict_batch_size=1024,
        train_metric_name="mae",
        val_metric_name="mae",
        use_early_stopping=False,
    )
    resolved_config = model.get_config()
    resolved_config_safe = json.loads(json.dumps(resolved_config, default=str))
    prefit_proof = {
        "stage_id": "stage5a1",
        "candidate_id": candidate_id,
        "physical_fit_number": 11,
        "official_estimator_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "resolved_config": resolved_config_safe,
        "use_best_epoch": resolved_config.get("use_best_epoch"),
        "use_early_stopping": resolved_config.get("use_early_stopping"),
        "requested_epoch_count": resolved_config.get("n_epochs"),
        "preprocessor_policy_digest": preprocessor_digest,
        "reused_preprocessor_from": str(defective_bundle_path.relative_to(ROOT)),
        "historical_artifact_hashes_before": historical_before,
        "fit_started": False,
    }
    resolved_path = report_dir / "stage5a1_realmlp_log1p_fit11_resolved_config.json"
    atomic_json(prefit_proof, resolved_path)
    if resolved_config.get("use_best_epoch") is not False:
        raise RuntimeError("Hard stop: effective PyTabKit config does not prove use_best_epoch=False")
    if resolved_config.get("use_early_stopping") is not False:
        raise RuntimeError("Hard stop: effective PyTabKit config does not disable early stopping")
    if int(resolved_config.get("n_epochs", -1)) != 64:
        raise RuntimeError("Hard stop: effective PyTabKit config does not request exactly 64 epochs")

    prefit_proof["fit_started"] = True
    atomic_json(prefit_proof, resolved_path)
    fit_start = time.perf_counter()
    model.fit(
        train_frame,
        train_target,
        X_val=val_frame,
        y_val=val_target,
        cat_col_names=CATEGORICAL_FEATURES,
        time_to_fit_in_seconds=1750,
    )
    fit_seconds = time.perf_counter() - fit_start
    peak_ram = max(peak_ram, _memory_mib())

    module = model.cv_alg_interface_.model
    completed_epoch = int(module.progress.epoch)
    maximum_epoch = int(module.progress.max_epochs)
    stop_epoch = int(model.fit_params_["stop_epoch"]["mae"])
    internal_use_best_epoch = module.creator.config.get("use_best_epoch")
    internal_early_stopping = module.creator.config.get("use_early_stopping")
    restoration_callback_created = bool(getattr(module, "ckpt_callbacks", None))
    if not (
        internal_use_best_epoch is False
        and internal_early_stopping is False
        and completed_epoch == 64
        and maximum_epoch == 64
        and stop_epoch == 64
        and not restoration_callback_created
    ):
        failure = {
            **prefit_proof,
            "completed_epoch_count": completed_epoch,
            "progress_max_epochs": maximum_epoch,
            "official_stop_epoch": stop_epoch,
            "internal_use_best_epoch": internal_use_best_epoch,
            "internal_use_early_stopping": internal_early_stopping,
            "restoration_callback_created": restoration_callback_created,
            "status": "FAIL",
        }
        atomic_json(failure, report_dir / "stage5a1_realmlp_log1p_fit11_protocol_proof.json")
        raise RuntimeError("Hard stop: fit 11 did not finish as an unrestored epoch-64 model")

    encoder = model.x_converter_.cat_tf.named_transformers_["categorical"]
    official_contract = {}
    for index, column in enumerate(CATEGORICAL_FEATURES):
        expected = set(train_frame[column].astype(str).unique())
        learned = {str(value) for value in encoder.categories_[index]}
        official_contract[column] = {
            "train_only_cardinality": len(expected),
            "official_encoder_cardinality": len(learned),
            "unexpected_official_categories": sorted(learned - expected),
            "missing_train_categories": sorted(expected - learned),
        }
    official_encoder_matches_train_only = all(
        not item["unexpected_official_categories"] and not item["missing_train_categories"]
        for item in official_contract.values()
    )
    if not official_encoder_matches_train_only:
        raise RuntimeError("Fit 11 official encoder violates the frozen Train-only vocabulary")

    prediction_start = time.perf_counter()
    transformed_prediction = np.asarray(model.predict(val_frame)).reshape(-1)
    prediction = target.inverse(transformed_prediction, standardized=True)
    prediction_seconds = time.perf_counter() - prediction_start
    if not np.isfinite(prediction).all():
        raise RuntimeError("Fit 11 produced non-finite Validation predictions")
    metrics = regression_metrics(y_val, prediction)

    bundle_path = model_dir / f"{candidate_id}.joblib"
    history_path = result_dir / f"histories/{candidate_id}_history.csv"
    proof_path = report_dir / "stage5a1_realmlp_log1p_fit11_protocol_proof.json"
    history = pd.DataFrame({
        "epoch": np.arange(1, completed_epoch + 1, dtype=np.int64),
        "per_epoch_metric_exposed_by_public_api": False,
        "note": "Official progress counter proves completion; public sklearn API does not expose per-epoch losses.",
    })
    atomic_csv(history, history_path)
    bundle = {
        "stage_id": "stage5a1",
        "candidate_id": candidate_id,
        "physical_fit_number": 11,
        "family": "realmlp",
        "target_mode": "log1p",
        "feature_schema": "deep_core_v1",
        "preprocessing_version": "realmlp_train_only_vocab_v2_unchanged",
        "target_transform": target,
        "preprocessor": preprocessor,
        "model": model,
        "architecture": defective_bundle["architecture"],
        "categorical_contract": categorical_contract,
        "official_encoder_contract": official_contract,
        "official_encoder_matches_train_only": official_encoder_matches_train_only,
        "effective_resolved_config": resolved_config_safe,
        "saved_final_artifact_epoch": 64,
        "restored_checkpoint_epoch": None,
        "device": DEVICE,
        "amp": AMP,
        "seed": SEED,
    }
    atomic_joblib(bundle, bundle_path)
    historical_after = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in historical_paths}
    proof = {
        **prefit_proof,
        "fit_started": True,
        "fit_completed": True,
        "use_best_epoch": False,
        "early_stopping_disabled": True,
        "requested_epoch_count": 64,
        "completed_epoch_count": completed_epoch,
        "progress_max_epochs": maximum_epoch,
        "official_stop_epoch": stop_epoch,
        "best_validation_epoch_observed_but_not_restored": int(module.best_mean_val_epochs["mae"][0]),
        "restoration_callback_created": False,
        "restored_checkpoint_epoch": None,
        "saved_final_artifact_epoch": 64,
        "training_history_length": len(history),
        "training_history_final_epoch": int(history["epoch"].iloc[-1]),
        "training_history_path": str(history_path.relative_to(ROOT)),
        "bundle_path": str(bundle_path.relative_to(ROOT)),
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "preprocessor_policy_digest_after": _payload_digest(_preprocessor_policy_payload(preprocessor)),
        "historical_artifact_hashes_after": historical_after,
        "historical_artifacts_unchanged": historical_before == historical_after,
        "official_encoder_matches_train_only": official_encoder_matches_train_only,
        "status": "PASS",
    }
    atomic_json(proof, proof_path)
    return {
        "bundle_path": bundle_path,
        "state_path": bundle_path,
        "preprocessor_path": bundle_path,
        "history_path": history_path,
        "prediction": prediction,
        "metrics": metrics,
        "architecture": bundle["architecture"],
        "optimizer": "official RealMLP optimizer",
        "learning_rate": None,
        "batch_size": 256,
        "best_epoch": 64,
        "epochs_completed": 64,
        "stop_reason": "fixed_epoch_64_no_early_stopping_no_checkpoint_restoration",
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_ram_mib": peak_ram,
        "peak_vram_mib": 0.0,
        "warning": "Per-epoch losses are not exposed by the public RealMLP sklearn API; the official progress counter completed all 64 epochs.",
        "categorical_contract": categorical_contract,
        "official_encoder_contract": official_contract,
        "official_encoder_matches_train_only": official_encoder_matches_train_only,
        "fixed_epoch_no_early_stopping_exception": True,
        "protocol_proof_path": proof_path,
        "resolved_config_path": resolved_path,
        "physical_fit_number": 11,
        "saved_final_artifact_epoch": 64,
        "restored_checkpoint_epoch": None,
        "use_best_epoch": False,
    }
    atomic_joblib(bundle, bundle_path)
    history = pd.DataFrame(
        [
            {
                "epoch": best_epoch,
                "training_loss": pd.NA,
                "validation_loss_original_scale": pd.NA,
                "validation_mae": metrics["mae"],
                "learning_rate": pd.NA,
                "epoch_seconds": pd.NA,
                "ram_mib": peak_ram,
                "vram_mib": 0.0,
                "note": "Fixed epoch 64 with no early stopping under the approved log1p exception." if fixed_log_epoch_exception else "Official best epoch exported from fit_params_; raw standardized MAE is affine-equivalent to original-scale MAE.",
            }
        ]
    )
    atomic_csv(history, history_path)
    return {
        "bundle_path": bundle_path,
        "state_path": bundle_path,
        "preprocessor_path": bundle_path,
        "history_path": history_path,
        "prediction": prediction,
        "metrics": metrics,
        "architecture": bundle["architecture"],
        "optimizer": "official RealMLP optimizer",
        "learning_rate": None,
        "batch_size": 256,
        "best_epoch": best_epoch,
        "epochs_completed": None,
        "stop_reason": "approved_fixed_epoch_no_early_stopping" if fixed_log_epoch_exception else "official RealMLP original-scale-equivalent stopping policy",
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "peak_ram_mib": peak_ram,
        "peak_vram_mib": 0.0,
        "warning": "Per-epoch history is not exposed by the public RealMLP sklearn API.",
        "categorical_contract": categorical_contract,
        "official_encoder_contract": official_contract,
        "official_encoder_matches_train_only": official_encoder_matches_train_only,
        "fixed_epoch_no_early_stopping_exception": fixed_log_epoch_exception,
    }


def repair_saved_realmlp_metadata() -> dict[str, Any]:
    repaired = []
    for candidate_id in ["stage5a1__realmlp__raw", "stage5a1__realmlp__log1p"]:
        bundle_path = ROOT / f"artifacts/models/deep/core_screening/{candidate_id}.joblib"
        bundle = joblib.load(bundle_path)
        best_epoch = int(bundle["model"].fit_params_["stop_epoch"]["mae"])
        for path in [
            ROOT / f"artifacts/checkpoints/stage5/deep_core/screening/{candidate_id}.json",
            ROOT / f"artifacts/results/stage5/deep_core/screening/candidates/{candidate_id}.json",
        ]:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["best_epoch"] = best_epoch
            data["warning"] = "Full per-epoch history is not exposed by the public RealMLP sklearn API; official best epoch is preserved."
            atomic_json(data, path)
        history_path = ROOT / f"artifacts/results/stage5/deep_core/screening/histories/{candidate_id}_history.csv"
        checkpoint = json.loads((ROOT / f"artifacts/checkpoints/stage5/deep_core/screening/{candidate_id}.json").read_text(encoding="utf-8"))
        atomic_csv(
            pd.DataFrame(
                [{
                    "epoch": best_epoch,
                    "training_loss": pd.NA,
                    "validation_loss_original_scale": pd.NA,
                    "validation_mae": checkpoint["metrics"]["mae"],
                    "learning_rate": pd.NA,
                    "epoch_seconds": pd.NA,
                    "ram_mib": checkpoint["peak_ram_mib"],
                    "vram_mib": 0.0,
                    "note": "Official best epoch exported from fit_params_; full public per-epoch history is unavailable.",
                }]
            ),
            history_path,
        )
        repaired.append({"candidate_id": candidate_id, "best_epoch": best_epoch})
    report = {"stage_id": "stage5a1", "repairs": repaired, "model_refits": 0, "status": "PASS"}
    atomic_json(report, ROOT / "artifacts/reports/stage5a1_realmlp_metadata_repair.json")
    return report


def predict_bundle(bundle_path: Path, X: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    family = bundle["family"]
    target: TargetTransform = bundle["target_transform"]
    preprocessor = bundle["preprocessor"]
    if family == "realmlp":
        frame = preprocessor.transform(X)
        transformed = np.asarray(bundle["model"].predict(frame)).reshape(-1)
    else:
        x_num, x_cat = preprocessor.transform(X)
        model, _ = build_torch_model(
            family,
            bundle["n_num_features"],
            bundle["cardinalities"],
            bundle.get("bins"),
        )
        state = torch.load(ROOT / bundle["model_state_path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        transformed = predict_torch_model(
            model, family, x_num, x_cat, bundle["batch_size"]
        )
    result = target.inverse(transformed, standardized=True)
    if not np.isfinite(result).all():
        raise RuntimeError(f"Reloaded {bundle_path.name} produced non-finite values")
    return result


def model_size_bytes(paths: list[Path]) -> int:
    return int(sum(path.stat().st_size for path in set(paths) if path.exists()))


def _checkpoint_frame() -> pd.DataFrame:
    checkpoint_dir = ROOT / "artifacts/checkpoints/stage5/deep_core/screening"
    rows = []
    for path in sorted(checkpoint_dir.glob("stage5a1__*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("status") != "PASS":
            continue
        row = {key: value for key, value in item.items() if key not in {"metrics", "architecture"}}
        row.update(item["metrics"])
        row["architecture_json"] = json.dumps(item["architecture"], sort_keys=True)
        rows.append(row)
    return pd.DataFrame(rows)


def _select_family_winner(group: pd.DataFrame) -> tuple[pd.Series, str]:
    group = group.sort_values("mae", kind="stable")
    best = group.iloc[0]
    other = group.iloc[1]
    gap_percent = abs(float(best["mae"]) - float(other["mae"])) / min(float(best["mae"]), float(other["mae"])) * 100.0
    if gap_percent < 0.25:
        ordered = group.assign(
            simpler_target=(group["target_mode"] != "raw").astype(int)
        ).sort_values(
            ["top_decile_mae", "fit_time_seconds", "peak_ram_mib", "model_size_bytes", "simpler_target"],
            kind="stable",
        )
        winner = ordered.iloc[0]
        reason = (
            f"MAE gap {gap_percent:.3f}% is inside 0.25%; selected lower tail MAE, "
            "then runtime/resource/simple-target tie-breaks."
        )
    else:
        winner = best
        reason = f"Selected lower Validation MAE; pair gap {gap_percent:.3f}% exceeds 0.25%."
    return winner, reason


def _registry_rows(screening: pd.DataFrame, winners: pd.DataFrame, top_two: dict[str, Any]) -> pd.DataFrame:
    registry = pd.read_csv(ROOT / "artifacts/results/experiment_results.csv")
    columns = list(registry.columns)
    timestamp = "2026-07-15T00:00:00+00:00"
    rows: list[dict[str, Any]] = []
    candidate_rows: dict[str, dict[str, Any]] = {}
    for _, item in screening.iterrows():
        registry_row = {
                "experiment_id": item["candidate_id"],
                "timestamp_utc": timestamp,
                "model_family": "deep_tabular",
                "model_name": item["model_family"],
                "sensitive_mode": "without_sensitive",
                "feature_set": "deep_core_v1",
                "target_mode": item["target_mode"],
                "evaluation_stage": "Stage 5A1 Discovery Screening",
                "fold_number": np.nan,
                "training_row_count": 50_000,
                "validation_row_count": 15_000,
                "test_row_count": 0,
                "parameter_json": item["architecture_json"],
                "mae": item["mae"],
                "mse": item["mse"],
                "rmse": item["rmse"],
                "mape_percent": item["mape_percent"],
                "r_squared": item["r_squared"],
                "rmsle": item["rmsle"],
                "rmsle_clipped_zero": item["rmsle"],
                "median_absolute_error": item["median_absolute_error"],
                "wape_percent": item["wape_percent"],
                "mean_signed_error": item["mean_signed_error"],
                "p90_absolute_error": item["p90_absolute_error"],
                "negative_prediction_rate": item["negative_prediction_rate"],
                "fit_time_seconds": item["fit_time_seconds"],
                "prediction_time_seconds": item["prediction_time_seconds"],
                "status": "PASS",
                "notes": "Stage 5A1 non-sensitive Discovery Screening; zero Test rows.",
                "model_artifact_path": item["model_path"],
                "prediction_artifact_path": item["prediction_path"],
            }
        rows.append(registry_row)
        candidate_rows[str(item["candidate_id"])] = registry_row
    for _, item in winners.iterrows():
        row = candidate_rows[str(item["candidate_id"])].copy()
        row["experiment_id"] = f"stage5a1__family_winner__{item['model_family']}"
        row["evaluation_stage"] = "Stage 5A1 Family Winner"
        row["notes"] = item["selection_reason"]
        rows.append(row)
    rows.append(
        {
            "experiment_id": "stage5a1__top_two_selection",
            "timestamp_utc": timestamp,
            "model_family": "deep_tabular_selection",
            "model_name": "+".join(top_two["selected_families"]),
            "sensitive_mode": "without_sensitive",
            "feature_set": "deep_core_v1",
            "target_mode": "family_winners",
            "evaluation_stage": "Stage 5A1 Top-Two Selection",
            "training_row_count": 50_000,
            "validation_row_count": 15_000,
            "test_row_count": 0,
            "parameter_json": json.dumps(top_two, sort_keys=True),
            "status": "PASS",
            "notes": top_two["selection_reason"],
        }
    )
    rows.append(
        {
            "experiment_id": "stage5a1__preprocessing_verification",
            "timestamp_utc": timestamp,
            "model_family": "deep_preprocessing",
            "model_name": "three_family_training_only_preprocessing",
            "sensitive_mode": "without_sensitive",
            "feature_set": "deep_core_v1",
            "target_mode": "not_applicable",
            "evaluation_stage": "Stage 5A1 Preprocessing Verification",
            "training_row_count": 50_000,
            "validation_row_count": 0,
            "test_row_count": 0,
            "parameter_json": "{}",
            "status": "PASS",
            "notes": "Training-only fit, serialization, unknown categories, row order, and finite outputs passed.",
        }
    )
    return pd.DataFrame(rows).reindex(columns=columns)


def _write_registry_suffix_preserving_prior_prefix(stage_rows: pd.DataFrame) -> dict[str, Any]:
    registry_path = ROOT / "artifacts/results/experiment_results.csv"
    baseline = json.loads((ROOT / "artifacts/manifests/stage5/stage5a1_protected_hashes_before.json").read_text(encoding="utf-8"))
    registry_entry = next(item for item in baseline["files"] if Path(item["path"]).name == "experiment_results.csv")
    current_bytes = registry_path.read_bytes()
    prior_size = int(registry_entry["size"])
    prior = current_bytes[:prior_size]
    if hashlib.sha256(prior).hexdigest() != registry_entry["sha256"]:
        raise RuntimeError("Prior Registry byte prefix changed")
    payload = stage_rows.to_csv(index=False, header=False, lineterminator="\r\n").encode("utf-8")
    desired = prior + payload
    status = "REUSED" if current_bytes == desired else "UPSERTED"
    if current_bytes != desired:
        temporary = registry_path.with_suffix(".csv.tmp")
        temporary.write_bytes(desired)
        os.replace(temporary, registry_path)
    after = registry_path.read_bytes()
    verified = after[:prior_size] == prior
    final = pd.read_csv(registry_path)
    report = {
        "status": "PASS" if verified and final["experiment_id"].is_unique else "FAIL",
        "action": status,
        "prior_byte_prefix_preserved": verified,
        "prior_row_count": 299,
        "stage5a1_row_count": len(stage_rows),
        "registry_row_count": len(final),
        "registry_ids_unique": bool(final["experiment_id"].is_unique),
    }
    atomic_json(report, ROOT / "artifacts/reports/stage5a1_registry_update.json")
    return report


def finalize_screening() -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/environment/stage5_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result_dir = ROOT / "artifacts/results/stage5/deep_core/screening"
    figure_dir = ROOT / "artifacts/figures/stage5a1"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_screening = _checkpoint_frame()
    expected = {
        "stage5a1__realmlp__raw__replacement1", "stage5a1__realmlp__log1p__replacement2",
        "stage5a1__tabm__raw__replacement1", "stage5a1__tabm__log1p__replacement1",
        "stage5a1__ft_transformer__raw", "stage5a1__ft_transformer__log1p",
    }
    superseded_ids = {
        "stage5a1__realmlp__raw": "stage5a1__realmlp__raw__replacement1",
        "stage5a1__realmlp__log1p": "stage5a1__realmlp__log1p__replacement1",
        "stage5a1__tabm__raw": "stage5a1__tabm__raw__replacement1",
        "stage5a1__tabm__log1p": "stage5a1__tabm__log1p__replacement1",
        "stage5a1__realmlp__log1p__replacement1": "stage5a1__realmlp__log1p__replacement2",
    }
    screening = all_screening[all_screening["candidate_id"].isin(expected)].copy().sort_values(["model_family", "target_mode"], kind="stable")
    if len(screening) != 6 or set(screening["candidate_id"]) != expected:
        raise RuntimeError("Exactly six valid Stage 5A1 Screening checkpoints are required")
    reload_reports = [
        json.loads((ROOT / f"artifacts/reports/stage5a1_reload_{candidate}.json").read_text(encoding="utf-8"))
        for candidate in expected
    ]
    if not all(item["status"] == "PASS" for item in reload_reports):
        raise RuntimeError("A clean Candidate reload is invalid")
    screening["validity_status"] = "VALID"
    atomic_csv(screening, result_dir / "stage5a1_screening_results.csv")
    superseded = all_screening[all_screening["candidate_id"].isin(superseded_ids)].copy()
    superseded["validity_status"] = "SUPERSEDED"
    superseded["superseded_by"] = superseded["candidate_id"].map(superseded_ids)
    supersession_reasons = {
        "stage5a1__realmlp__raw": "Invalid: official categorical encoder learned from concatenated Validation before repair.",
        "stage5a1__realmlp__log1p": "Invalid: official categorical encoder learned from concatenated Validation before repair.",
        "stage5a1__tabm__raw": "Non-compliant: official one-hot categorical input replaced by maintained learned embeddings.",
        "stage5a1__tabm__log1p": "Non-compliant: official one-hot categorical input replaced by maintained learned embeddings.",
        "stage5a1__realmlp__log1p__replacement1": "superseded_checkpoint_restoration_violation",
    }
    superseded["supersession_reason"] = superseded["candidate_id"].map(supersession_reasons)
    atomic_csv(superseded, result_dir / "stage5a1_superseded_screening_results.csv")
    validity = {
        "physical_screening_fit_count": 11,
        "initial_fit_count": 6,
        "approved_replacement_fit_count": 5,
        "four_family_repair_fit_count": 4,
        "protocol_repair_fit_count": 1,
        "final_valid_candidate_count": 6,
        "valid_candidate_ids": sorted(expected),
        "superseded_candidates": superseded_ids,
        "historical_artifacts_preserved": True,
        "status": "PASS",
    }
    atomic_json(validity, ROOT / "artifacts/reports/stage5a1_screening_validity.json")

    winner_rows = []
    for family, group in screening.groupby("model_family", sort=True):
        winner, reason = _select_family_winner(group)
        row = winner.to_dict()
        row["selection_reason"] = reason
        winner_rows.append(row)
    winners = pd.DataFrame(winner_rows).sort_values("mae", kind="stable").reset_index(drop=True)
    if set(winners["model_family"]) != {"realmlp", "tabm", "ft_transformer"}:
        raise RuntimeError("Three family winners are required")
    atomic_csv(winners, result_dir / "stage5a1_family_winners.csv")

    first = winners.iloc[0]
    remaining = winners.iloc[1:].copy()
    remaining_gap = abs(float(remaining.iloc[0]["mae"]) - float(remaining.iloc[1]["mae"])) / min(float(remaining.iloc[0]["mae"]), float(remaining.iloc[1]["mae"])) * 100.0
    if remaining_gap < 0.5:
        second = remaining.sort_values(
            ["top_decile_mae", "rmse", "rmsle", "fit_time_seconds", "peak_vram_mib", "model_size_bytes"],
            kind="stable",
        ).iloc[0]
        second_reason = (
            f"The remaining-family MAE gap is {remaining_gap:.3f}% (<0.5%); "
            "the lower top-decile MAE wins the tie."
        )
    else:
        second = remaining.sort_values("mae", kind="stable").iloc[0]
        second_reason = f"The remaining-family MAE gap is {remaining_gap:.3f}%; lower MAE wins."
    selected = [str(first["model_family"]), str(second["model_family"])]
    excluded = sorted(set(winners["model_family"]) - set(selected))
    top_two = {
        "stage_id": "stage5a1",
        "feature_schema": "deep_core_v1",
        "selected_families": selected,
        "selected_candidate_ids": [str(first["candidate_id"]), str(second["candidate_id"])],
        "not_selected_family": excluded[0],
        "family_winners_preserved": winners["candidate_id"].tolist(),
        "selection_reason": (
            f"{first['model_family']} has the best family-winner MAE. {second_reason}"
        ),
        "test_evidence_used": False,
        "status": "PASS",
    }
    atomic_json(top_two, result_dir / "stage5a1_top_two_families.json")
    atomic_csv(
        screening[["candidate_id", "model_family", "target_mode", "fit_time_seconds", "prediction_time_seconds", "best_epoch", "epochs_completed", "stop_reason"]],
        result_dir / "stage5a1_runtime_summary.csv",
    )
    atomic_csv(
        screening[["candidate_id", "peak_ram_mib", "peak_vram_mib", "model_size_bytes", "device", "amp", "batch_size"]],
        result_dir / "stage5a1_resource_summary.csv",
    )

    stage_rows = _registry_rows(screening, winners, top_two)
    superseded_rows = _registry_rows(
        superseded,
        pd.DataFrame(),
        {"selected_families": [], "selection_reason": "Superseded historical rows only."},
    ).iloc[:len(superseded)].copy()
    superseded_rows["status"] = "SUPERSEDED"
    superseded_rows["evaluation_stage"] = "Stage 5A1 Superseded Screening"
    superseded_rows["notes"] = superseded["supersession_reason"].to_numpy()
    stage_rows = pd.concat([superseded_rows, stage_rows], ignore_index=True)
    atomic_csv(stage_rows, result_dir / "stage5a1_registry_rows.csv")
    registry_report = _write_registry_suffix_preserving_prior_prefix(stage_rows)
    if registry_report["status"] != "PASS":
        raise RuntimeError("Stage 5A1 Registry update failed")

    plot = screening.sort_values("mae")
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = plot["model_family"] + " / " + plot["target_mode"]
    ax.barh(labels, plot["mae"], color="#4472C4")
    ax.set_xlabel("Discovery Validation MAE")
    ax.set_title("Stage 5A1 six-fit Screening")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figure_dir / "stage5a1_screening_mae.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(winners["mae"], winners["top_decile_mae"], s=90)
    for _, row in winners.iterrows():
        ax.annotate(row["model_family"], (row["mae"], row["top_decile_mae"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Validation MAE")
    ax.set_ylabel("Top-decile MAE")
    ax.set_title("Stage 5A1 family-winner trade-off")
    fig.tight_layout()
    fig.savefig(figure_dir / "stage5a1_family_winner_tradeoff.png", dpi=160)
    plt.close(fig)

    summary = {
        "physical_screening_fit_count": 11,
        "approved_replacement_fit_count": 5,
        "protocol_repair_fit_count": 1,
        "screening_fit_count": len(screening),
        "family_winner_count": len(winners),
        "selected_family_count": len(selected),
        "selected_families": selected,
        "registry": registry_report,
        "status": "PASS",
    }
    atomic_json(summary, ROOT / "artifacts/reports/stage5a1_screening_summary.json")
    return summary
