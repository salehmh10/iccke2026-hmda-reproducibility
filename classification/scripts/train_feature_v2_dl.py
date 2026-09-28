"""Train every existing tabular DL family with the selected V2 features."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.model_selection import train_test_split

from scripts.train_dl import DL_MODELS, _make_torch_model, _positive_weight, _train_tabnet
from scripts.train_feature_v2_ml import _array_sha256, _total_measured_fit_seconds
from src.data.splitting import hash_indices
from src.evaluation.metrics import optimize_threshold
from src.models.neural import predict_neural_probability, train_neural_model, as_dense_float32
from src.training.context_v2 import (
    FEATURE_GENERATION,
    build_experiment_context_v2,
    transform_variant_training_v2,
)
from src.training.tracking import (
    append_record,
    build_record,
    cumulative_training_seconds,
    experiment_contract_fingerprint,
    experiment_exists,
    write_experiment_provenance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--models", nargs="+", choices=DL_MODELS, default=list(DL_MODELS))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original_weighted", "oversampled", "undersampled"],
    )
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    report_dir = root / "reports" / "generations" / "feature_v2"
    model_dir = root / "artifacts" / "generations" / "feature_v2" / "models"
    prediction_dir = root / "artifacts" / "generations" / "feature_v2" / "predictions"
    ledger = report_dir / "EXPERIMENT_RESULTS.csv"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    context = build_experiment_context_v2(
        root=root, max_fit_rows=60_000, seed=args.seed, reuse_preprocessor=True
    )
    validation_positions = np.arange(len(context.validation_target), dtype=np.int64)
    if len(validation_positions) > 20_000:
        validation_positions, _ = train_test_split(
            validation_positions,
            train_size=20_000,
            stratify=context.validation_target,
            random_state=args.seed,
            shuffle=True,
        )
        validation_positions = np.sort(validation_positions)
    early_features = context.validation_features[validation_positions]
    early_target = context.validation_target[validation_positions]
    code_files = [
        "src/features/advanced.py",
        "src/models/neural.py",
        "src/evaluation/metrics.py",
        "src/training/context_v2.py",
        "src/training/tracking.py",
        "scripts/train_dl.py",
        "scripts/train_feature_v2_dl.py",
        "scripts/train_feature_v2_ml.py",
    ]
    completed = 0
    failed = 0
    for variant_name in args.variants:
        variant = context.variants[variant_name]
        x_train, y_train, train_indices, sample_weights = transform_variant_training_v2(
            context, variant, max_rows=args.max_train_rows, seed=args.seed
        )
        run_contract = {
            **context.contract_metadata,
            "variant": variant_name,
            "sampler": variant.sampler,
            "sample_size": int(len(y_train)),
            "sampled_source_indices_sha256": hash_indices(train_indices),
            "sample_weights_sha256": _array_sha256(sample_weights),
        }
        fingerprint = experiment_contract_fingerprint(root, run_contract, code_files)
        version = fingerprint[:10]
        input_dim = int(x_train.shape[1])
        for model_name in args.models:
            experiment_id = (
                f"v2_dl_{model_name}_{variant_name}_s{args.seed}_n{len(y_train)}_v{version}"
            )
            extension = ".joblib" if model_name == "tabnet" else ".pt"
            artifact_path = model_dir / f"{experiment_id}{extension}"
            prediction_path = prediction_dir / f"{experiment_id}_validation.npz"
            provenance_path = model_dir / f"{experiment_id}.provenance.json"
            if experiment_exists(ledger, experiment_id):
                missing = [
                    str(path)
                    for path in (artifact_path, prediction_path, provenance_path)
                    if not path.is_file()
                ]
                if missing:
                    raise RuntimeError(f"recorded V2 DL experiment is incomplete: {missing}")
                print(json.dumps({"experiment_id": experiment_id, "status": "already_recorded"}))
                continue
            if _total_measured_fit_seconds(root, ledger) >= 43_200.0:
                raise RuntimeError("hard cumulative training budget reached")
            started = time.perf_counter()
            params: dict[str, object] = {}
            try:
                weighted = variant_name == "original_weighted"
                if model_name == "tabnet":
                    model, runtime, params = _train_tabnet(
                        x_train,
                        y_train,
                        early_features,
                        early_target,
                        seed=args.seed,
                        weighted=weighted,
                    )
                    validation_probability = model.predict_proba(
                        as_dense_float32(context.validation_features)
                    )[:, 1]
                    joblib.dump(
                        {
                            "model": model,
                            "preprocessor": context.preprocessor,
                            "target": "target_denied",
                            "feature_generation": FEATURE_GENERATION,
                            "experiment_id": experiment_id,
                        },
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
                        early_features,
                        early_target,
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
                    cpu_state = {
                        key: value.detach().cpu()
                        for key, value in result.model.state_dict().items()
                    }
                    torch.save(
                        {
                            "model_name": model_name,
                            "model_parameters": params,
                            "state_dict": cpu_state,
                            "threshold_pending": True,
                            "experiment_id": experiment_id,
                            "feature_generation": FEATURE_GENERATION,
                        },
                        artifact_path,
                    )
                threshold = optimize_threshold(
                    context.validation_target, validation_probability
                )
                np.savez_compressed(
                    prediction_path,
                    source_indices=context.validation_indices,
                    y_true=context.validation_target,
                    denial_probability=validation_probability,
                    threshold=np.array([threshold.threshold]),
                )
                write_experiment_provenance(
                    root,
                    experiment_id,
                    fingerprint,
                    run_contract,
                    code_files,
                    artifact_dir=model_dir,
                )
                append_record(
                    ledger,
                    build_record(
                        root=root,
                        experiment_id=experiment_id,
                        dataset_variant=variant_name,
                        sample_size=len(y_train),
                        feature_set=FEATURE_GENERATION,
                        model=model_name,
                        hyperparameters=params,
                        seed=args.seed,
                        class_weighting=(
                            "weighted_bce_or_tabnet_weights" if weighted else "none"
                        ),
                        sampler=variant.sampler,
                        train_runtime_seconds=runtime,
                        threshold=threshold.threshold,
                        status="completed",
                        metrics=threshold.metrics,
                    ),
                )
                completed += 1
                print(
                    json.dumps(
                        {
                            "experiment_id": experiment_id,
                            "status": "completed",
                            "runtime": runtime,
                            "pr_auc": threshold.metrics["pr_auc"],
                            "mcc": threshold.metrics["mcc"],
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                runtime = time.perf_counter() - started
                append_record(
                    ledger,
                    build_record(
                        root=root,
                        experiment_id=experiment_id,
                        dataset_variant=variant_name,
                        sample_size=len(y_train),
                        feature_set=FEATURE_GENERATION,
                        model=model_name,
                        hyperparameters=params,
                        seed=args.seed,
                        class_weighting=(
                            "weighted_loss" if variant_name == "original_weighted" else "none"
                        ),
                        sampler=variant.sampler,
                        train_runtime_seconds=runtime,
                        threshold=0.5,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                )
                failed += 1
                print(
                    json.dumps(
                        {"experiment_id": experiment_id, "status": "failed", "error": str(exc)}
                    ),
                    flush=True,
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "completed": completed,
                "failed": failed,
                "v2_ledger_fit_seconds": cumulative_training_seconds(ledger),
                "total_measured_fit_seconds": _total_measured_fit_seconds(root, ledger),
            }
        )
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
