"""Train and track classical models on all three train-only variants."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import optimize_threshold
from src.models.classical import (
    SUPPORTED_CLASSICAL_MODELS,
    denial_probability,
    estimator_parameters,
    make_classical_model,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--models", nargs="+", choices=SUPPORTED_CLASSICAL_MODELS, default=list(SUPPORTED_CLASSICAL_MODELS))
    parser.add_argument("--variants", nargs="+", default=["original_weighted", "oversampled", "undersampled"])
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    ledger = root / "reports" / "EXPERIMENT_RESULTS.csv"
    budget_limit = 43200.0
    context = build_experiment_context(
        root=root, max_fit_rows=args.max_train_rows, seed=args.seed, reuse_preprocessor=True
    )
    code_files = [
        "src/models/classical.py",
        "src/evaluation/metrics.py",
        "src/training/context.py",
        "src/training/tracking.py",
        "scripts/train_ml.py",
    ]
    contract_fingerprint = experiment_contract_fingerprint(
        root, context.contract_metadata, code_files
    )
    version = contract_fingerprint[:10]
    completed = 0
    failed = 0
    for variant_name in args.variants:
        variant = context.variants[variant_name]
        x_train, y_train, train_indices, sample_weight = transform_variant_training(
            context, variant, max_rows=args.max_train_rows, seed=args.seed
        )
        for model_name in args.models:
            group_safe_stacking = model_name == "stacking" and variant_name == "oversampled"
            suffix = "_groupcv" if group_safe_stacking else ""
            experiment_id = f"ml_{model_name}_{variant_name}{suffix}_s{args.seed}_n{len(y_train)}_v{version}"
            if experiment_exists(ledger, experiment_id):
                print(json.dumps({"experiment_id": experiment_id, "status": "already_recorded"}))
                continue
            if cumulative_training_seconds(ledger) >= budget_limit:
                raise RuntimeError("hard cumulative training budget reached")
            estimator = make_classical_model(model_name, seed=args.seed, n_jobs=args.n_jobs)
            if group_safe_stacking:
                splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
                cv_splits = list(
                    splitter.split(
                        np.zeros(len(y_train), dtype=np.int8),
                        y_train,
                        groups=train_indices,
                    )
                )
                for fit_positions, oof_positions in cv_splits:
                    if set(train_indices[fit_positions]).intersection(train_indices[oof_positions]):
                        raise RuntimeError("group-safe stacking CV leaked a source row across folds")
                estimator.set_params(cv=cv_splits)
            parameters = estimator_parameters(estimator)
            if group_safe_stacking:
                parameters["cv"] = "StratifiedGroupKFold(n_splits=3, groups=source_row_index)"
            started = time.perf_counter()
            try:
                fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
                estimator.fit(x_train, y_train, **fit_kwargs)
                runtime = time.perf_counter() - started
                validation_probability = denial_probability(estimator, context.validation_features)
                threshold_result = optimize_threshold(
                    context.validation_target, validation_probability
                )
                artifact_path = root / "artifacts" / "models" / f"{experiment_id}.joblib"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(
                    {
                        "model": estimator,
                        "preprocessor": context.preprocessor,
                        "target": "target_denied",
                        "threshold": threshold_result.threshold,
                        "experiment_id": experiment_id,
                    },
                    artifact_path,
                )
                prediction_path = root / "artifacts" / "predictions" / f"{experiment_id}_validation.npz"
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
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
                    hyperparameters=parameters,
                    seed=args.seed,
                    class_weighting="balanced_sample_weight" if sample_weight is not None else "none",
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
                    hyperparameters=parameters,
                    seed=args.seed,
                    class_weighting="balanced_sample_weight" if sample_weight is not None else "none",
                    sampler=variant.sampler,
                    train_runtime_seconds=runtime,
                    threshold=0.5,
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                append_record(ledger, record)
                failed += 1
                print(json.dumps({"experiment_id": experiment_id, "status": "failed", "error": str(exc)}))
    print(json.dumps({"completed": completed, "failed": failed, "cumulative_training_seconds": cumulative_training_seconds(ledger)}))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
