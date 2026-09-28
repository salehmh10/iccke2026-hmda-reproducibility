"""Train-only MI, redundancy, and repeated-OOF ablation for feature V2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_and_prepare
from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.data.splitting import load_split_manifest
from src.data.variants import load_variant
from src.evaluation.metrics import optimize_threshold
from src.features.advanced import (
    CATEGORY_CONTEXT_PAIRS,
    FEATURE_FAMILIES,
    AdvancedFinancialFeatureEngineer,
    build_advanced_feature_pipeline,
)
from src.features.financial import CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES
from src.models.classical import denial_probability, make_classical_model
from src.training.context import stratified_variant_sample


CONFIGS: dict[str, tuple[str, ...]] = {
    "baseline_v1_equivalent": (),
    "numeric_relative": ("numeric_relative",),
    "nonlinear": ("nonlinear",),
    "category_context": ("category_context",),
    "onehot_numeric": ("onehot_numeric",),
    "categorical_numeric_combined": ("category_context", "onehot_numeric"),
    "all_candidates": FEATURE_FAMILIES,
}

SCREENING_MODELS = ("lightgbm", "logistic_regression")

NUMERIC_RELATIVE_METADATA: dict[str, tuple[str, tuple[str, ...]]] = {
    "loan_to_area_median": (
        "loan_amount_000s / (hud_median_family_income / 1000)",
        ("loan_amount_000s", "hud_median_family_income"),
    ),
    "loan_to_tract_income": (
        "loan_amount_000s / tract_implied_income_000s",
        ("loan_amount_000s", "hud_median_family_income", "tract_to_msamd_income_ratio"),
    ),
    "applicant_income_to_tract_income": (
        "applicant_income_000s / tract_implied_income_000s",
        ("applicant_income_000s", "hud_median_family_income", "tract_to_msamd_income_ratio"),
    ),
    "loan_minus_income_000s": (
        "loan_amount_000s - applicant_income_000s",
        ("loan_amount_000s", "applicant_income_000s"),
    ),
    "abs_loan_minus_income_000s": (
        "abs(loan_amount_000s - applicant_income_000s)",
        ("loan_amount_000s", "applicant_income_000s"),
    ),
    "loan_income_normalized_gap": (
        "(loan - income) / (abs(loan) + abs(income) + eps)",
        ("loan_amount_000s", "applicant_income_000s"),
    ),
    "loan_share_of_loan_plus_income": (
        "loan / (loan + income)",
        ("loan_amount_000s", "applicant_income_000s"),
    ),
    "applicant_minus_area_income_000s": (
        "applicant_income_000s - area_median_income_000s",
        ("applicant_income_000s", "hud_median_family_income"),
    ),
    "applicant_minus_tract_income_000s": (
        "applicant_income_000s - tract_implied_income_000s",
        ("applicant_income_000s", "hud_median_family_income", "tract_to_msamd_income_ratio"),
    ),
    "tract_minus_area_income_000s": (
        "tract_implied_income_000s - area_median_income_000s",
        ("hud_median_family_income", "tract_to_msamd_income_ratio"),
    ),
}

NONLINEAR_METADATA: dict[str, tuple[str, tuple[str, ...]]] = {
    "sqrt_loan_amount_000s": ("sqrt(loan_amount_000s)", ("loan_amount_000s",)),
    "sqrt_applicant_income_000s": (
        "sqrt(applicant_income_000s)",
        ("applicant_income_000s",),
    ),
    "log1p_loan_to_income": ("log1p(loan_to_income)", ("loan_to_income",)),
    "log1p_applicant_income_to_area_median": (
        "log1p(applicant_income_to_area_median)",
        ("applicant_income_to_area_median",),
    ),
    "tract_income_ratio_abs_deviation": (
        "abs(tract_to_msamd_income_ratio - 1)",
        ("tract_to_msamd_income_ratio",),
    ),
    "loan_to_income_gt_2": ("1[loan_to_income > 2]", ("loan_to_income",)),
    "loan_to_income_gt_3": ("1[loan_to_income > 3]", ("loan_to_income",)),
    "applicant_income_below_area": (
        "1[applicant_income_000s < area_median_income_000s]",
        ("applicant_income_000s", "hud_median_family_income"),
    ),
    "tract_income_below_msa": (
        "1[tract_to_msamd_income_ratio < 1]",
        ("tract_to_msamd_income_ratio",),
    ),
    "high_lti_low_tract_income": (
        "1[loan_to_income > 2 and tract_to_msamd_income_ratio < 1]",
        ("loan_to_income", "tract_to_msamd_income_ratio"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-rows", type=int, default=60_000)
    parser.add_argument("--mi-rows", type=int, default=40_000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260809, 20260810])
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, frame: pd.DataFrame, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _family_and_metadata(name: str) -> tuple[str, str, tuple[str, ...]]:
    if name in NUMERIC_RELATIVE_METADATA:
        formula, sources = NUMERIC_RELATIVE_METADATA[name]
        return "numeric_relative", formula, sources
    if name in NONLINEAR_METADATA:
        formula, sources = NONLINEAR_METADATA[name]
        return "nonlinear", formula, sources
    if name.endswith("__robust_z") and "__by__" not in name:
        source = name.removesuffix("__robust_z")
        return "nonlinear", f"({source} - train_median) / train_IQR", (source,)
    if name.endswith("__frequency"):
        source = name.removesuffix("__frequency")
        return "category_context", f"train_count({source}=level) / train_rows", (source,)
    if name.endswith("__rare_or_unseen"):
        source = name.removesuffix("__rare_or_unseen")
        return "category_context", f"1[train_count({source}=level) < 500 or unseen]", (source,)
    if "__by__" in name:
        numeric, remainder = name.split("__by__", 1)
        category, statistic = remainder.rsplit("__", 1)
        formula = (
            f"({numeric} - train_median({numeric}|{category})) / "
            + (f"abs(train_median({numeric}|{category}))" if statistic == "relative_median" else f"train_IQR({numeric}|{category})")
        )
        return "category_context", formula, (numeric, category)
    if "__x__" in name:
        numeric, remainder = name.split("__x__", 1)
        category = next(
            candidate for candidate in CATEGORICAL_FEATURES if remainder.startswith(category_prefix(candidate))
        )
        return "onehot_numeric", f"{numeric} * 1[{category}=fitted_level]", (numeric, category)
    raise KeyError(f"unclassified V2 candidate: {name}")


def category_prefix(category: str) -> str:
    return f"{category}__"


def _mutual_information_table(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    seed: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").astype("float64")
            fill = float(values.median())
            if not np.isfinite(fill):
                fill = 0.0
            array = values.fillna(fill).to_numpy().reshape(-1, 1)
            unique_count = int(np.unique(array).size)
            discrete = unique_count <= 20
            if discrete:
                discrete_codes, _ = pd.factorize(array.reshape(-1), sort=True)
                array = discrete_codes.reshape(-1, 1)
            mi = float(
                mutual_info_classif(
                    array,
                    target,
                    discrete_features=discrete,
                    random_state=seed,
                )[0]
            )
            kind = "numeric_discrete" if discrete else "numeric_continuous"
        else:
            codes, _ = pd.factorize(series.astype("string").fillna("__MISSING__"), sort=True)
            unique_count = int(np.unique(codes).size)
            mi = float(mutual_info_score(target, codes))
            kind = "categorical"
        records.append(
            {
                "feature": name,
                "kind": kind,
                "unique_values": unique_count,
                "target_mutual_information": mi,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["target_mutual_information", "feature"], ascending=[False, True]
    )


def _candidate_validation_table(
    engineered: pd.DataFrame,
    mi_table: pd.DataFrame,
    selected_families: tuple[str, ...],
) -> pd.DataFrame:
    baseline = set(MODEL_NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    candidates = [column for column in engineered if column not in baseline]
    numeric = engineered[candidates].select_dtypes(include=[np.number]).copy()
    for column in numeric:
        values = numeric[column].replace([np.inf, -np.inf], np.nan)
        numeric[column] = values.fillna(values.median()).fillna(0.0)
    redundancy_sample = numeric.iloc[: min(20_000, len(numeric))]
    correlation = redundancy_sample.corr(method="spearman").abs()
    mi_map = dict(zip(mi_table["feature"], mi_table["target_mutual_information"], strict=True))
    records: list[dict[str, Any]] = []
    for name in candidates:
        family, formula, sources = _family_and_metadata(name)
        if name in correlation:
            values = correlation[name].drop(index=name).dropna()
            redundant_with = str(values.idxmax()) if not values.empty else ""
            max_redundancy = float(values.max()) if not values.empty else 0.0
        else:
            redundant_with = ""
            max_redundancy = 0.0
        kept = family in selected_families
        reason = (
            f"kept: family selected by repeated train-only OOF; max_abs_spearman={max_redundancy:.4f}"
            if kept
            else "rejected: family did not win the predeclared repeated-OOF selection rule"
        )
        records.append(
            {
                "feature": name,
                "family": family,
                "formula": formula,
                "source_features": json.dumps(sources),
                "target_mutual_information": float(mi_map.get(name, np.nan)),
                "source_mutual_information": json.dumps(
                    {source: float(mi_map.get(source, np.nan)) for source in sources},
                    sort_keys=True,
                ),
                "max_abs_spearman_redundancy": max_redundancy,
                "most_redundant_with": redundant_with,
                "selected": kept,
                "keep_reject_reason": reason,
                "leakage_contract": "row-local or X-only train-fitted; no target",
                "inference_fallback": (
                    "global train median/IQR or zero support for unseen category"
                    if family == "category_context"
                    else "unseen category activates no fitted interaction"
                    if family == "onehot_numeric"
                    else "safe NaN followed by train-median imputation"
                ),
            }
        )
    return pd.DataFrame(records).sort_values(["selected", "family", "feature"], ascending=[False, True, True])


def _select_configuration(summary: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    pivot = summary.pivot(index="configuration", columns="model", values="mean_oof_pr_auc")
    baseline = pivot.loc["baseline_v1_equivalent"]
    nonbaseline = pivot.drop(index="baseline_v1_equivalent").copy()
    deltas = nonbaseline.subtract(baseline, axis=1)
    deltas["mean_delta"] = deltas[list(SCREENING_MODELS)].mean(axis=1)
    eligible = deltas[
        (deltas["lightgbm"] >= -0.00025)
        & (deltas["logistic_regression"] >= -0.00050)
        & (deltas["mean_delta"] > 0.0)
    ]
    pool = eligible if not eligible.empty else deltas
    best_score = float(pool["mean_delta"].max())
    near = pool[pool["mean_delta"] >= best_score - 0.00020].copy()
    dimensions = summary.groupby("configuration")["mean_encoded_features"].mean()
    selected = str(min(near.index, key=lambda name: (dimensions.loc[name], name)))
    decision = {
        "selection_rule": (
            "positive mean PR-AUC delta across LightGBM/logistic, with LightGBM delta >= -0.00025 "
            "and logistic delta >= -0.00050; within 0.00020 of best choose the fewest encoded features"
        ),
        "eligible_rule_satisfied": bool(not eligible.empty),
        "selected_configuration": selected,
        "selected_families": list(CONFIGS[selected]),
        "baseline_pr_auc": {model: float(baseline[model]) for model in SCREENING_MODELS},
        "selected_pr_auc": {
            model: float(pivot.loc[selected, model]) for model in SCREENING_MODELS
        },
        "selected_delta": {
            model: float(deltas.loc[selected, model]) for model in SCREENING_MODELS
        },
        "mean_delta": float(deltas.loc[selected, "mean_delta"]),
        "test_rows_used": False,
    }
    return selected, decision


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "reports" / "generations" / "feature_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_and_prepare(root / "hmda_classification_stratified_500k.csv")
    splits = load_split_manifest(root / "data" / "manifests")
    variant = load_variant("original_weighted", root / "data" / "manifests", splits)
    indices, _ = stratified_variant_sample(
        prepared.modeling, variant, max_rows=args.max_rows, seed=20260809
    )
    if not set(indices).issubset(set(splits.train)):
        raise RuntimeError("feature V2 analysis attempted to use non-training rows")
    frame = prepared.modeling.loc[indices].drop(columns=[RAW_TARGET, ANALYTICAL_TARGET])
    target = prepared.modeling.loc[indices, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)

    fold_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    total_fit_seconds = 0.0
    total_preprocess_seconds = 0.0
    for config_name, families in CONFIGS.items():
        for seed in args.seeds:
            splitter = StratifiedKFold(
                n_splits=args.folds, shuffle=True, random_state=seed
            )
            per_model_y: dict[str, list[np.ndarray]] = {name: [] for name in SCREENING_MODELS}
            per_model_p: dict[str, list[np.ndarray]] = {name: [] for name in SCREENING_MODELS}
            fold_pr: dict[str, list[float]] = {name: [] for name in SCREENING_MODELS}
            encoded_dimensions: list[int] = []
            for fold, (fit_positions, held_positions) in enumerate(
                splitter.split(frame, target), start=1
            ):
                pipeline = build_advanced_feature_pipeline(families=families)
                preprocess_started = time.perf_counter()
                x_fit = pipeline.fit_transform(frame.iloc[fit_positions])
                x_held = pipeline.transform(frame.iloc[held_positions])
                preprocess_runtime = time.perf_counter() - preprocess_started
                total_preprocess_seconds += preprocess_runtime
                encoded_dimensions.append(int(x_fit.shape[1]))
                y_fit = target[fit_positions]
                y_held = target[held_positions]
                sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
                for model_name in SCREENING_MODELS:
                    model = make_classical_model(
                        model_name, seed=seed + fold, n_jobs=args.n_jobs
                    )
                    fit_started = time.perf_counter()
                    model.fit(x_fit, y_fit, sample_weight=sample_weight)
                    fit_runtime = time.perf_counter() - fit_started
                    total_fit_seconds += fit_runtime
                    probability = denial_probability(model, x_held)
                    pr_auc = float(average_precision_score(y_held, probability))
                    fold_pr[model_name].append(pr_auc)
                    per_model_y[model_name].append(y_held)
                    per_model_p[model_name].append(probability)
                    fold_records.append(
                        {
                            "configuration": config_name,
                            "families": json.dumps(families),
                            "seed": seed,
                            "fold": fold,
                            "model": model_name,
                            "fit_rows": int(len(fit_positions)),
                            "held_rows": int(len(held_positions)),
                            "encoded_features": int(x_fit.shape[1]),
                            "preprocess_seconds": preprocess_runtime,
                            "model_fit_seconds": fit_runtime,
                            "pr_auc": pr_auc,
                            "roc_auc": float(roc_auc_score(y_held, probability)),
                            "brier_score": float(brier_score_loss(y_held, probability)),
                        }
                    )
                    print(
                        json.dumps(
                            {
                                "config": config_name,
                                "seed": seed,
                                "fold": fold,
                                "model": model_name,
                                "pr_auc": round(pr_auc, 6),
                                "features": int(x_fit.shape[1]),
                            }
                        ),
                        flush=True,
                    )
            for model_name in SCREENING_MODELS:
                y_oof = np.concatenate(per_model_y[model_name])
                p_oof = np.concatenate(per_model_p[model_name])
                threshold = optimize_threshold(y_oof, p_oof)
                seed_records.append(
                    {
                        "configuration": config_name,
                        "families": json.dumps(families),
                        "seed": seed,
                        "model": model_name,
                        "oof_rows": int(len(y_oof)),
                        "encoded_features": float(np.mean(encoded_dimensions)),
                        "oof_pr_auc": float(average_precision_score(y_oof, p_oof)),
                        "oof_roc_auc": float(roc_auc_score(y_oof, p_oof)),
                        "oof_brier_score": float(brier_score_loss(y_oof, p_oof)),
                        "oof_mcc": float(threshold.metrics["mcc"]),
                        "oof_balanced_accuracy": float(
                            threshold.metrics["balanced_accuracy"]
                        ),
                        "oof_threshold": float(threshold.threshold),
                        "fold_pr_auc_std": float(np.std(fold_pr[model_name], ddof=1)),
                    }
                )

    folds = pd.DataFrame(fold_records)
    seeds = pd.DataFrame(seed_records)
    summary = (
        seeds.groupby(["configuration", "model"], as_index=False)
        .agg(
            mean_oof_pr_auc=("oof_pr_auc", "mean"),
            seed_pr_auc_std=("oof_pr_auc", "std"),
            mean_fold_pr_auc_std=("fold_pr_auc_std", "mean"),
            mean_oof_roc_auc=("oof_roc_auc", "mean"),
            mean_oof_mcc=("oof_mcc", "mean"),
            mean_oof_balanced_accuracy=("oof_balanced_accuracy", "mean"),
            mean_oof_brier=("oof_brier_score", "mean"),
            mean_encoded_features=("encoded_features", "mean"),
        )
        .sort_values(["model", "mean_oof_pr_auc"], ascending=[True, False])
    )
    baseline_values = summary[summary["configuration"] == "baseline_v1_equivalent"].set_index(
        "model"
    )["mean_oof_pr_auc"]
    summary["absolute_pr_auc_delta_vs_baseline"] = summary.apply(
        lambda row: row["mean_oof_pr_auc"] - baseline_values.loc[row["model"]], axis=1
    )
    summary["relative_pr_auc_delta_percent"] = summary.apply(
        lambda row: 100.0
        * row["absolute_pr_auc_delta_vs_baseline"]
        / baseline_values.loc[row["model"]],
        axis=1,
    )
    selected_name, decision = _select_configuration(summary)

    mi_positions = np.arange(len(frame))
    if len(mi_positions) > args.mi_rows:
        mi_splitter = StratifiedKFold(n_splits=max(2, len(frame) // args.mi_rows), shuffle=True, random_state=20260809)
        _, mi_positions = next(mi_splitter.split(frame, target))
        if len(mi_positions) > args.mi_rows:
            mi_positions = mi_positions[: args.mi_rows]
    mi_frame = frame.iloc[mi_positions]
    mi_target = target[mi_positions]
    all_engineer = AdvancedFinancialFeatureEngineer(families=FEATURE_FAMILIES).fit(mi_frame)
    engineered = all_engineer.transform(mi_frame)
    mi_inputs = engineered.copy()
    # Raw HUD income is a source for several candidates but is intentionally not
    # emitted by either model pipeline. Include it only in this train-only source-MI audit.
    mi_inputs["hud_median_family_income"] = pd.to_numeric(
        mi_frame["hud_median_family_income"], errors="coerce"
    ).astype("float64")
    mi_table = _mutual_information_table(mi_inputs, mi_target, seed=20260809)
    validation_table = _candidate_validation_table(
        engineered, mi_table, tuple(decision["selected_families"])
    )

    _write_csv(output_dir / "OOF_FOLD_RESULTS.csv", folds, args.overwrite)
    _write_csv(output_dir / "OOF_SEED_RESULTS.csv", seeds, args.overwrite)
    _write_csv(output_dir / "ABLATION_RESULTS.csv", summary, args.overwrite)
    _write_csv(output_dir / "FEATURE_MI.csv", mi_table, args.overwrite)
    _write_csv(output_dir / "FEATURE_VALIDATION.csv", validation_table, args.overwrite)
    _write_json(output_dir / "SELECTED_FEATURE_CONFIG.json", decision, args.overwrite)
    _write_json(
        output_dir / "SCREENING_BUDGET.json",
        {
            "model_fit_seconds": total_fit_seconds,
            "preprocessing_seconds": total_preprocess_seconds,
            "total_screening_seconds": total_fit_seconds + total_preprocess_seconds,
            "folds": args.folds,
            "seeds": args.seeds,
            "rows": len(frame),
            "test_rows_used": False,
        },
        args.overwrite,
    )
    print(json.dumps({"selected": selected_name, **decision}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
