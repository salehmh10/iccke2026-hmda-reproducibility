"""Train all classical families with the isolated selected V2 feature set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitting import hash_indices
from src.evaluation.metrics import optimize_threshold
from src.models.classical import (
    SUPPORTED_CLASSICAL_MODELS,
    denial_probability,
    estimator_parameters,
    make_classical_model,
)
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
    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_CLASSICAL_MODELS,
        default=list(SUPPORTED_CLASSICAL_MODELS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original_weighted", "oversampled", "undersampled"],
    )
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def _array_sha256(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _total_measured_fit_seconds(root: Path, v2_ledger: Path) -> float:
    total = cumulative_training_seconds(root / "reports" / "EXPERIMENT_RESULTS.csv")
    reproduction = json.loads((root / "reports" / "REPRODUCIBILITY.json").read_text())
    total += sum(float(value) for value in reproduction["training_runtime_seconds"])
    screening = json.loads(
        (root / "reports" / "generations" / "feature_v2" / "SCREENING_BUDGET.json").read_text()
    )
    total += float(screening["model_fit_seconds"])
    confirmation = json.loads(
        (root / "reports" / "generations" / "feature_v2" / "CONFIRMED_FEATURE_CONFIG.json").read_text()
    )
    total += float(confirmation["model_fit_seconds"])
    total += cumulative_training_seconds(v2_ledger)
    return total


def _resume_complete(
    ledger: Path,
    experiment_id: str,
    model_path: Path,
    prediction_path: Path,
    provenance_path: Path,
) -> bool:
    if not experiment_exists(ledger, experiment_id):
        return False
    missing = [
        str(path) for path in (model_path, prediction_path, provenance_path) if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"recorded V2 experiment has missing outputs: {missing}")
    return True


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
        root=root,
        max_fit_rows=args.max_train_rows,
        seed=args.seed,
        reuse_preprocessor=True,
    )
    code_files = [
        "src/features/advanced.py",
        "src/models/classical.py",
        "src/evaluation/metrics.py",
        "src/training/context_v2.py",
        "src/training/tracking.py",
        "scripts/train_feature_v2_ml.py",
    ]
    completed = 0
    failed = 0
    for variant_name in args.variants:
        variant = context.variants[variant_name]
        x_train, y_train, train_indices, sample_weight = transform_variant_training_v2(
            context, variant, max_rows=args.max_train_rows, seed=args.seed
        )
        run_contract = {
            **context.contract_metadata,
            "variant": variant_name,
            "sampler": variant.sampler,
            "sample_size": int(len(y_train)),
            "sampled_source_indices_sha256": hash_indices(train_indices),
            "sample_weights_sha256": _array_sha256(sample_weight),
        }
        fingerprint = experiment_contract_fingerprint(root, run_contract, code_files)
        version = fingerprint[:10]
        for model_name in args.models:
            group_safe_stacking = model_name == "stacking" and variant_name == "oversampled"
            suffix = "_groupcv" if group_safe_stacking else ""
            experiment_id = (
                f"v2_ml_{model_name}_{variant_name}{suffix}_s{args.seed}_"
                f"n{len(y_train)}_v{version}"
            )
            artifact_path = model_dir / f"{experiment_id}.joblib"
            prediction_path = prediction_dir / f"{experiment_id}_validation.npz"
            provenance_path = model_dir / f"{experiment_id}.provenance.json"
            if _resume_complete(
                ledger, experiment_id, artifact_path, prediction_path, provenance_path
            ):
                print(json.dumps({"experiment_id": experiment_id, "status": "already_recorded"}))
                continue
            if _total_measured_fit_seconds(root, ledger) >= 43_200.0:
                raise RuntimeError("hard cumulative training budget reached")
            estimator = make_classical_model(
                model_name, seed=args.seed, n_jobs=args.n_jobs
            )
            if group_safe_stacking:
                splitter = StratifiedGroupKFold(
                    n_splits=3, shuffle=True, random_state=args.seed
                )
                cv_splits = list(
                    splitter.split(
                        np.zeros(len(y_train), dtype=np.int8),
                        y_train,
                        groups=train_indices,
                    )
                )
                for fit_positions, oof_positions in cv_splits:
                    if set(train_indices[fit_positions]).intersection(
                        train_indices[oof_positions]
                    ):
                        raise RuntimeError("V2 stacking CV leaked a source row")
                estimator.set_params(cv=cv_splits)
            parameters = estimator_parameters(estimator)
            if group_safe_stacking:
                parameters["cv"] = "StratifiedGroupKFold(groups=source_row_index)"
            started = time.perf_counter()
            try:
                fit_kwargs = (
                    {"sample_weight": sample_weight} if sample_weight is not None else {}
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    estimator.fit(x_train, y_train, **fit_kwargs)
                runtime = time.perf_counter() - started
                convergence_warnings = sum(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                )
                parameters["convergence_warnings"] = convergence_warnings
                validation_probability = denial_probability(
                    estimator, context.validation_features
                )
                threshold_result = optimize_threshold(
                    context.validation_target, validation_probability
                )
                joblib.dump(
                    {
                        "model": estimator,
                        "preprocessor": context.preprocessor,
                        "target": "target_denied",
                        "threshold": threshold_result.threshold,
                        "experiment_id": experiment_id,
                        "feature_generation": FEATURE_GENERATION,
                    },
                    artifact_path,
                )
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
                        hyperparameters=parameters,
                        seed=args.seed,
                        class_weighting=(
                            "balanced_sample_weight" if sample_weight is not None else "none"
                        ),
                        sampler=variant.sampler,
                        train_runtime_seconds=runtime,
                        threshold=threshold_result.threshold,
                        status="completed",
                        metrics=threshold_result.metrics,
                    ),
                )
                completed += 1
                print(
                    json.dumps(
                        {
                            "experiment_id": experiment_id,
                            "status": "completed",
                            "runtime": runtime,
                            "pr_auc": threshold_result.metrics["pr_auc"],
                            "mcc": threshold_result.metrics["mcc"],
                            "convergence_warnings": convergence_warnings,
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
                        hyperparameters=parameters,
                        seed=args.seed,
                        class_weighting=(
                            "balanced_sample_weight" if sample_weight is not None else "none"
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
