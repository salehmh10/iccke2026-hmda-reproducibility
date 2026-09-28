"""Train compact neural tabular families on all three dataset variants."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np  # Must precede torch imports in this environment.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import optimize_threshold
from src.models.neural import (
    ModernHopfieldClassifier,
    TabularMLP,
    TabularTransformer,
    as_dense_float32,
    predict_neural_probability,
    train_neural_model,
)
from src.training.context import build_experiment_context, transform_variant_training
from src.training.tracking import (
    append_record,
    build_record,
    cumulative_training_seconds,
    experiment_contract_fingerprint,
    experiment_exists,
    write_experiment_provenance,
)


DL_MODELS = ("mlp", "tabnet", "tabular_transformer", "modern_hopfield")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--models", nargs="+", choices=DL_MODELS, default=list(DL_MODELS))
    parser.add_argument("--variants", nargs="+", default=["original_weighted", "oversampled", "undersampled"])
    parser.add_argument("--max-train-rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def _positive_weight(y: np.ndarray, weighted: bool) -> float:
    if not weighted:
        return 1.0
    negative = int((y == 0).sum())
    positive = int((y == 1).sum())
    return float(negative / positive)


def _make_torch_model(name: str, input_dim: int) -> tuple[torch.nn.Module, dict[str, object]]:
    if name == "mlp":
        params = {"input_dim": input_dim, "hidden_dim": 128, "dropout": 0.2}
        return TabularMLP(**params), params
    if name == "tabular_transformer":
        params = {"input_dim": input_dim, "token_dim": 16, "n_heads": 4, "n_layers": 2, "dropout": 0.1}
        return TabularTransformer(**params), params
    if name == "modern_hopfield":
        params = {"input_dim": input_dim, "model_dim": 64, "memory_patterns": 32, "dropout": 0.15}
        return ModernHopfieldClassifier(**params), params
    raise KeyError(name)


def _train_tabnet(
    x_train: object,
    y_train: np.ndarray,
    x_validation: object,
    y_validation: np.ndarray,
    *,
    seed: int,
    weighted: bool,
) -> tuple[object, float, dict[str, object]]:
    from pytorch_tabnet.tab_model import TabNetClassifier

    train_dense = as_dense_float32(x_train)
    validation_dense = as_dense_float32(x_validation)
    params: dict[str, object] = {
        "n_d": 16,
        "n_a": 16,
        "n_steps": 3,
        "gamma": 1.3,
        "lambda_sparse": 1e-4,
        "seed": seed,
        "verbose": 0,
        "device_name": "cuda" if torch.cuda.is_available() else "cpu",
    }
    model = TabNetClassifier(**params)
    weights: int | dict[int, float]
    if weighted:
        pos_weight = _positive_weight(y_train, True)
        weights = {0: 1.0, 1: pos_weight}
    else:
        weights = 0
    started = time.perf_counter()
    model.fit(
        train_dense,
        y_train,
        eval_set=[(validation_dense, y_validation)],
        eval_name=["validation"],
        eval_metric=["auc"],
        max_epochs=25,
        patience=4,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=weights,
    )
    runtime = time.perf_counter() - started
    params.update({"max_epochs": 25, "patience": 4, "batch_size": 1024, "weights": weights})
    return model, runtime, params


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    ledger = root / "reports" / "EXPERIMENT_RESULTS.csv"
    context = build_experiment_context(
        root=root, max_fit_rows=60000, seed=args.seed, reuse_preprocessor=True
    )
    code_files = [
        "src/models/neural.py",
        "src/evaluation/metrics.py",
        "src/training/context.py",
        "src/training/tracking.py",
        "scripts/train_dl.py",
    ]
    contract_fingerprint = experiment_contract_fingerprint(
        root, context.contract_metadata, code_files
    )
    version = contract_fingerprint[:10]
    validation_positions = np.arange(len(context.validation_target), dtype=np.int64)
    if len(validation_positions) > 20000:
        validation_positions, _ = train_test_split(
            validation_positions,
            train_size=20000,
            stratify=context.validation_target,
            random_state=args.seed,
            shuffle=True,
        )
        validation_positions = np.sort(validation_positions)
    early_validation_features = context.validation_features[validation_positions]
    early_validation_target = context.validation_target[validation_positions]
    completed = 0
    failed = 0
    for variant_name in args.variants:
        variant = context.variants[variant_name]
        x_train, y_train, _, _ = transform_variant_training(
            context, variant, max_rows=args.max_train_rows, seed=args.seed
        )
        input_dim = int(x_train.shape[1])
        for model_name in args.models:
            experiment_id = f"dl_{model_name}_{variant_name}_s{args.seed}_n{len(y_train)}_v{version}"
            if experiment_exists(ledger, experiment_id):
                print(json.dumps({"experiment_id": experiment_id, "status": "already_recorded"}))
                continue
            if cumulative_training_seconds(ledger) >= 43200.0:
                raise RuntimeError("hard cumulative training budget reached")
            started = time.perf_counter()
            params: dict[str, object] = {}
            try:
                weighted = variant_name == "original_weighted"
                if model_name == "tabnet":
                    model, runtime, params = _train_tabnet(
                        x_train,
                        y_train,
                        early_validation_features,
                        early_validation_target,
                        seed=args.seed,
                        weighted=weighted,
                    )
                    validation_probability = model.predict_proba(
                        as_dense_float32(context.validation_features)
                    )[:, 1]
                    artifact_path = root / "artifacts" / "models" / f"{experiment_id}.joblib"
                    joblib.dump(
                        {"model": model, "preprocessor": context.preprocessor, "target": "target_denied"},
                        artifact_path,
                    )
                else:
                    model, params = _make_torch_model(model_name, input_dim)
                    epoch_config = {
                        "mlp": (18, 4, 1024),
                        "tabular_transformer": (10, 3, 512),
                        "modern_hopfield": (16, 4, 1024),
                    }[model_name]
                    result = train_neural_model(
                        model,
                        x_train,
                        y_train,
                        early_validation_features,
                        early_validation_target,
                        seed=args.seed,
                        positive_weight=_positive_weight(y_train, weighted),
                        max_epochs=epoch_config[0],
                        patience=epoch_config[1],
                        batch_size=epoch_config[2],
                        learning_rate=1e-3,
                    )
                    runtime = result.runtime_seconds
                    validation_probability = predict_neural_probability(
                        result.model,
                        context.validation_features,
                        device=result.device,
                        batch_size=2048,
                    )
                    params.update(
                        {
                            "max_epochs": epoch_config[0],
                            "patience": epoch_config[1],
                            "batch_size": epoch_config[2],
                            "best_epoch": result.best_epoch,
                            "device": result.device,
                            "positive_weight": _positive_weight(y_train, weighted),
                        }
                    )
                    artifact_path = root / "artifacts" / "models" / f"{experiment_id}.pt"
                    cpu_state = {key: value.detach().cpu() for key, value in result.model.state_dict().items()}
                    torch.save(
                        {
                            "model_name": model_name,
                            "model_parameters": params,
                            "state_dict": cpu_state,
                            "threshold_pending": True,
                            "experiment_id": experiment_id,
                        },
                        artifact_path,
                    )
                threshold_result = optimize_threshold(context.validation_target, validation_probability)
                prediction_path = root / "artifacts" / "predictions" / f"{experiment_id}_validation.npz"
                np.savez_compressed(
                    prediction_path,
                    source_indices=context.validation_indices,
                    y_true=context.validation_target,
                    denial_probability=validation_probability,
                    threshold=np.array([threshold_result.threshold]),
                )
                write_experiment_provenance(
                    root,
                    experiment_id,
                    contract_fingerprint,
                    context.contract_metadata,
                    code_files,
                )
                record = build_record(
                    root=root,
                    experiment_id=experiment_id,
                    dataset_variant=variant_name,
                    sample_size=len(y_train),
                    feature_set="primary_financial_v1",
                    model=model_name,
                    hyperparameters=params,
                    seed=args.seed,
                    class_weighting="weighted_bce_or_tabnet_weights" if weighted else "none",
                    sampler=variant.sampler,
                    train_runtime_seconds=runtime,
                    threshold=threshold_result.threshold,
                    status="completed",
                    metrics=threshold_result.metrics,
                )
                append_record(ledger, record)
                completed += 1
                print(json.dumps({"experiment_id": experiment_id, "status": "completed", "runtime": runtime, "pr_auc": threshold_result.metrics["pr_auc"], "mcc": threshold_result.metrics["mcc"]}))
            except Exception as exc:
                runtime = time.perf_counter() - started
                record = build_record(
                    root=root,
                    experiment_id=experiment_id,
                    dataset_variant=variant_name,
                    sample_size=len(y_train),
                    feature_set="primary_financial_v1",
                    model=model_name,
                    hyperparameters=params,
                    seed=args.seed,
                    class_weighting="weighted_loss" if variant_name == "original_weighted" else "none",
                    sampler=variant.sampler,
                    train_runtime_seconds=runtime,
                    threshold=0.5,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                append_record(ledger, record)
                failed += 1
                print(json.dumps({"experiment_id": experiment_id, "status": "failed", "error": str(exc)}))
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(json.dumps({"completed": completed, "failed": failed, "cumulative_training_seconds": cumulative_training_seconds(ledger)}))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
