"""Independent train-only OOF confirmation for the V2 compact-feature choice."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_and_prepare
from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.data.splitting import load_split_manifest
from src.data.variants import load_variant
from src.features.advanced import FEATURE_FAMILIES, build_advanced_feature_pipeline
from src.models.classical import denial_probability, make_classical_model
from src.training.context import stratified_variant_sample


CONFIGS = {
    "baseline_v1_equivalent": (),
    "onehot_numeric": ("onehot_numeric",),
    "all_candidates": FEATURE_FAMILIES,
}
MODELS = ("lightgbm", "logistic_regression")
SEED = 20260811


def main() -> int:
    root = PROJECT_ROOT
    output_dir = root / "reports" / "generations" / "feature_v2"
    result_path = output_dir / "CONFIRMATORY_OOF_RESULTS.csv"
    decision_path = output_dir / "CONFIRMED_FEATURE_CONFIG.json"
    if result_path.exists() or decision_path.exists():
        raise FileExistsError("confirmatory V2 outputs already exist; refusing overwrite")

    prepared = load_and_prepare(root / "hmda_classification_stratified_500k.csv")
    splits = load_split_manifest(root / "data" / "manifests")
    variant = load_variant("original_weighted", root / "data" / "manifests", splits)
    indices, _ = stratified_variant_sample(
        prepared.modeling, variant, max_rows=60_000, seed=20260809
    )
    if not set(indices).issubset(set(splits.train)):
        raise RuntimeError("confirmatory analysis attempted to use non-training rows")
    frame = prepared.modeling.loc[indices].drop(columns=[RAW_TARGET, ANALYTICAL_TARGET])
    target = prepared.modeling.loc[indices, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    records: list[dict[str, object]] = []
    fit_seconds = 0.0
    for config_name, families in CONFIGS.items():
        predictions: dict[str, list[np.ndarray]] = {model: [] for model in MODELS}
        targets: list[np.ndarray] = []
        convergence_failures: dict[str, int] = {model: 0 for model in MODELS}
        dimensions: list[int] = []
        for fold, (fit_positions, held_positions) in enumerate(
            splitter.split(frame, target), start=1
        ):
            pipeline = build_advanced_feature_pipeline(families=families)
            x_fit = pipeline.fit_transform(frame.iloc[fit_positions])
            x_held = pipeline.transform(frame.iloc[held_positions])
            dimensions.append(int(x_fit.shape[1]))
            y_fit = target[fit_positions]
            y_held = target[held_positions]
            targets.append(y_held)
            weights = compute_sample_weight(class_weight="balanced", y=y_fit)
            for model_name in MODELS:
                model = make_classical_model(model_name, seed=SEED + fold, n_jobs=8)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(x_fit, y_fit, sample_weight=weights)
                fit_seconds += time.perf_counter() - started
                convergence_failures[model_name] += sum(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                )
                probability = denial_probability(model, x_held)
                predictions[model_name].append(probability)
                print(
                    json.dumps(
                        {
                            "configuration": config_name,
                            "fold": fold,
                            "model": model_name,
                            "pr_auc": average_precision_score(y_held, probability),
                            "convergence_warnings": convergence_failures[model_name],
                        }
                    ),
                    flush=True,
                )
        y_oof = np.concatenate(targets)
        for model_name in MODELS:
            probability = np.concatenate(predictions[model_name])
            records.append(
                {
                    "configuration": config_name,
                    "families": json.dumps(families),
                    "seed": SEED,
                    "folds": 3,
                    "model": model_name,
                    "encoded_features": float(np.mean(dimensions)),
                    "oof_pr_auc": float(average_precision_score(y_oof, probability)),
                    "oof_roc_auc": float(roc_auc_score(y_oof, probability)),
                    "convergence_warnings": convergence_failures[model_name],
                }
            )

    results = pd.DataFrame(records)
    pivot = results.pivot(index="configuration", columns="model", values="oof_pr_auc")
    warnings_pivot = results.pivot(
        index="configuration", columns="model", values="convergence_warnings"
    )
    dimensions = results.groupby("configuration")["encoded_features"].mean()
    baseline = pivot.loc["baseline_v1_equivalent"]
    candidates = pivot.drop(index="baseline_v1_equivalent")
    eligible: list[str] = []
    for name in candidates.index:
        if (
            candidates.loc[name, "lightgbm"] > baseline["lightgbm"]
            and candidates.loc[name, "logistic_regression"] > baseline["logistic_regression"]
            and warnings_pivot.loc[name, "logistic_regression"]
            <= warnings_pivot.loc["baseline_v1_equivalent", "logistic_regression"]
        ):
            eligible.append(str(name))
    if eligible:
        selected = min(
            eligible,
            key=lambda name: (
                -float(
                    (candidates.loc[name, "lightgbm"] - baseline["lightgbm"])
                    + 0.25
                    * (
                        candidates.loc[name, "logistic_regression"]
                        - baseline["logistic_regression"]
                    )
                ),
                dimensions.loc[name],
                name,
            ),
        )
    else:
        selected = "onehot_numeric"
    decision = {
        "selection_stage": "independent third-seed train-only OOF confirmation",
        "seed": SEED,
        "selection_rule": (
            "candidate must improve PR-AUC for both LightGBM and logistic and have no more "
            "logistic convergence warnings than baseline; rank LightGBM delta primary plus "
            "0.25 * logistic delta, then fewer features"
        ),
        "eligible_configurations": eligible,
        "selected_configuration": selected,
        "selected_families": list(CONFIGS[selected]),
        "baseline_pr_auc": {model: float(baseline[model]) for model in MODELS},
        "selected_pr_auc": {model: float(pivot.loc[selected, model]) for model in MODELS},
        "selected_delta": {
            model: float(pivot.loc[selected, model] - baseline[model]) for model in MODELS
        },
        "encoded_features": int(dimensions.loc[selected]),
        "convergence_warnings": {
            model: int(warnings_pivot.loc[selected, model]) for model in MODELS
        },
        "model_fit_seconds": fit_seconds,
        "test_rows_used": False,
    }
    results.to_csv(result_path, index=False)
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
