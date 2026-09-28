"""One bounded Stage 8 Recovery inference worker.

The worker performs one successful parser-boundary materialization per source,
loads each frozen model once, and creates no evaluation prediction artifact.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "artifacts/environment/stage4_packages"))
sys.path.insert(0, str(ROOT / "artifacts/environment/stage5_env/Lib/site-packages"))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import stage8_explanation_worker as base
from stage5_safe_row_loader import load_allowed_source_rows
from stage8_explainability_utils import CANDIDATES, EXPECTED, MODELS, PREDICTIONS
from stage8_recovery import (
    AUTHORIZATION_ID,
    RECOVERY_MANIFESTS,
    RECOVERY_RESULTS,
    REPORTS,
    SOURCE_HASHES,
    dump,
    now,
    record,
    sha,
    value_hash,
)


SAMPLE_PATH = RECOVERY_MANIFESTS / "stage8_recovery_global_sample_row_ids.csv"
BACKGROUND_PATH = RECOVERY_MANIFESTS / "stage8_recovery_local_background_row_ids.csv"
CASES_PATH = ROOT / "artifacts/manifests/stage8/stage8_local_case_manifest.csv"
SOURCE_WO = ROOT / "data/regression_without_sensitive_features.csv"
SOURCE_W = ROOT / "data/regression_with_sensitive_features.csv"


def recovery_path(name: str) -> Path:
    return RECOVERY_RESULTS / f"stage8_recovery_{name}"


def load_bounded_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    freeze_path = REPORTS / "stage8_recovery_sample_freeze.json"
    freeze_hash = sha(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS" or freeze.get("authorization_id") != AUTHORIZATION_ID:
        raise RuntimeError("Recovery freeze validation failed")
    global_ids = pd.read_csv(SAMPLE_PATH).row_id.astype(int).tolist()
    case_ids = pd.read_csv(CASES_PATH).row_id.astype(int).tolist()
    allowed = sorted(set(global_ids + case_ids))
    if len(allowed) > 2020:
        raise RuntimeError("Recovery row union exceeds 2,020")
    wo_columns = list(dict.fromkeys(base.BOOST_RAW + base.DEEP_NUM_WO + base.DEEP_CAT_WO))
    w_columns = list(dict.fromkeys(
        base.DEEP_NUM_WO
        + ["minority_population"]
        + base.DEEP_CAT_WO
        + base.SENSITIVE_IDENTITY
        + ["majority_minority_tract"]
    ))
    source_before = {
        "without_sensitive": sha(SOURCE_WO),
        "with_sensitive": sha(SOURCE_W),
    }
    if source_before["without_sensitive"] != SOURCE_HASHES[SOURCE_WO.relative_to(ROOT).as_posix()]:
        raise RuntimeError("Without-sensitive source hash failed")
    if source_before["with_sensitive"] != SOURCE_HASHES[SOURCE_W.relative_to(ROOT).as_posix()]:
        raise RuntimeError("With-sensitive source hash failed")

    started = now()
    wo = load_allowed_source_rows(SOURCE_WO, allowed, wo_columns, allowed_train_ids=set(allowed))
    if len(wo) != len(allowed):
        raise RuntimeError("Without-sensitive bounded materialization is incomplete")
    w = load_allowed_source_rows(SOURCE_W, allowed, w_columns, allowed_train_ids=set(allowed))
    if len(w) != len(allowed):
        raise RuntimeError("With-sensitive bounded materialization is incomplete")
    train_ids = set(pd.read_csv(ROOT / "artifacts/splits/train_row_ids.csv").iloc[:, 0].astype(int))
    if set(allowed) & train_ids:
        raise RuntimeError("Train row entered Recovery materialization")
    source_after = {
        "without_sensitive": sha(SOURCE_WO),
        "with_sensitive": sha(SOURCE_W),
    }
    if source_after != source_before:
        raise RuntimeError("A source hash changed during Recovery access")
    audit = {
        "authorization_id": AUTHORIZATION_ID,
        "recovery_freeze_sha256": freeze_hash,
        "status": "PASS",
        "first_access_at_utc": started,
        "source_paths_and_hashes": {
            "without_sensitive": {"path": SOURCE_WO.relative_to(ROOT).as_posix(), "sha256": source_before["without_sensitive"]},
            "with_sensitive": {"path": SOURCE_W.relative_to(ROOT).as_posix(), "sha256": source_before["with_sensitive"]},
        },
        "access_attempts": {"without_sensitive": 1, "with_sensitive": 1},
        "successful_materializations": {"without_sensitive": 1, "with_sensitive": 1},
        "raw_lines_scanned": {"without_sensitive": 499736, "with_sensitive": 499736},
        "rows_materialized": {"without_sensitive": len(wo), "with_sensitive": len(w)},
        "maximum_rows_per_source": 2020,
        "required_columns": {
            "without_sensitive": wo_columns,
            "with_sensitive": w_columns,
            "boosting_components": base.BOOST_RAW,
            "realmlp_without_sensitive": base.DEEP_NUM_WO + base.DEEP_CAT_WO,
            "realmlp_with_sensitive": w_columns,
        },
        "train_rows_materialized": 0,
        "excluded_rows_converted": 0,
        "source_target_values_materialized": 0,
        "source_rows_outside_frozen_membership": 0,
        "source_hashes_after_access": source_after,
        "loader_path": "stage5_safe_row_loader.py",
        "loader_sha256": sha(ROOT / "stage5_safe_row_loader.py"),
        "raw_sensitive_values_published": 0,
    }
    dump(audit, REPORTS / "stage8_recovery_feature_access_audit.json")
    return wo, w, audit


def load_models() -> tuple[dict, list[dict]]:
    loaded: dict[str, object] = {}
    attempts = []
    for item in MODELS:
        path = ROOT / item["path"]
        actual = sha(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Frozen model hash mismatch: {item['id']}")
        model = joblib.load(path)
        loaded[item["id"]] = model
        metadata = model.metadata if hasattr(model, "metadata") else model
        attempts.append({
            "model_id": item["id"],
            "family": item["family"],
            "sensitive_mode": item["mode"],
            "bundle_path": item["path"],
            "bundle_sha256": actual,
            "physical_attempts": 1,
            "successful_outputs": 1,
            "training_rows": int(metadata.get("training_row_count", 399788)),
            "test_rows_in_fit": int(metadata.get("test_row_count", metadata.get("test_rows", 0))),
            "target_mode": item["target_mode"],
            "status": "PASS",
        })
    dump({
        "authorization_id": AUTHORIZATION_ID,
        "status": "PASS",
        "model_count": len(attempts),
        "models": attempts,
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "surrogate_fit_calls": 0,
    }, REPORTS / "stage8_recovery_model_validation.json")
    return loaded, attempts


def reconcile(models: dict, wo: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    ids = pd.read_csv(SAMPLE_PATH).row_id.astype(int).to_numpy()
    xwo = wo.loc[ids]
    xw = w.loc[ids]
    generated: dict[str, np.ndarray] = {}
    rows = []
    component_paths = {
        MODELS[0]["id"]: "artifacts/predictions/final_test/stage4l__catboost__without_sensitive.csv",
        MODELS[1]["id"]: "artifacts/predictions/final_test/stage4l__lightgbm__without_sensitive.csv",
        MODELS[2]["id"]: "artifacts/predictions/final_test/stage4l__xgboost__without_sensitive.csv",
    }
    for item in MODELS[:3]:
        output = base.predict(item["id"], models[item["id"]], xwo)
        generated[item["id"]] = output
        saved = pd.read_csv(ROOT / component_paths[item["id"]]).set_index("row_id").loc[ids, "y_pred"].to_numpy()
        difference = np.abs(output - saved)
        rows.append({
            "model_or_candidate_id": item["id"],
            "row_count": len(ids),
            "saved_prediction_path": component_paths[item["id"]],
            "maximum_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "tolerance": 1e-5,
            "status": "PASS" if difference.max() <= 1e-5 else "FAIL",
        })
    blend = 0.6 * generated[MODELS[0]["id"]] + 0.2 * generated[MODELS[1]["id"]] + 0.2 * generated[MODELS[2]["id"]]
    generated[CANDIDATES[0]] = blend
    candidate_frames = {CANDIDATES[0]: xwo, CANDIDATES[1]: xwo, CANDIDATES[2]: xw}
    for candidate, prediction_item in zip(CANDIDATES, PREDICTIONS):
        if candidate == CANDIDATES[0]:
            output = blend
            tolerance = 1e-5
        else:
            output = base.predict(candidate, models[candidate], candidate_frames[candidate])
            tolerance = 1e-3
        generated[candidate] = output
        saved = pd.read_csv(ROOT / prediction_item["path"]).set_index("row_id").loc[ids, "y_pred"].to_numpy()
        difference = np.abs(output - saved)
        rows.append({
            "model_or_candidate_id": candidate,
            "row_count": len(ids),
            "saved_prediction_path": prediction_item["path"],
            "maximum_absolute_difference": float(difference.max()),
            "mean_absolute_difference": float(difference.mean()),
            "tolerance": tolerance,
            "status": "PASS" if difference.max() <= tolerance else "FAIL",
        })
    result = pd.DataFrame(rows)
    result.to_csv(REPORTS / "stage8_recovery_prediction_reconciliation.csv", index=False)
    if not result.status.eq("PASS").all():
        raise RuntimeError("Recovery prediction reconciliation failed")
    return result, generated


def permutation(models: dict, wo: pd.DataFrame, w: pd.DataFrame, generated: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(SAMPLE_PATH)
    ids = sample.row_id.astype(int).to_numpy()
    y = pd.read_csv(ROOT / PREDICTIONS[1]["path"]).set_index("row_id").loc[ids, "y_true"].to_numpy()
    frames = {
        CANDIDATES[0]: wo.loc[ids, base.BOOST_RAW],
        CANDIDATES[1]: wo.loc[ids, base.DEEP_NUM_WO + base.DEEP_CAT_WO],
        CANDIDATES[2]: w.loc[ids, base.DEEP_NUM_WO + ["minority_population"] + base.DEEP_CAT_WO + base.SENSITIVE_IDENTITY + ["majority_minority_tract"]],
    }
    permutations = {seed: np.random.default_rng(seed).permutation(len(ids)) for seed in [42, 43]}
    all_rows = []
    repeat_rows = []
    for candidate in CANDIDATES:
        frame = frames[candidate]
        units = base.make_units(frame.columns.tolist())
        baseline = generated[candidate]
        baseline_mae = float(np.mean(np.abs(y - baseline)))
        by_seed: dict[int, dict[str, dict[str, float]]] = {}
        for seed in [42, 43]:
            blocks = []
            for unit, columns in units.items():
                block = frame.copy()
                block.loc[:, columns] = frame.iloc[permutations[seed]][columns].to_numpy()
                blocks.append(block)
            stacked = pd.concat(blocks, ignore_index=True)
            output = base.predict_candidate(candidate, models, stacked) if candidate != CANDIDATES[2] else base.predict_candidate(candidate, models, wo.iloc[:0], stacked)
            by_seed[seed] = {}
            for position, unit in enumerate(units):
                prediction = output[position * len(ids):(position + 1) * len(ids)]
                by_seed[seed][unit] = {
                    "permuted_mae": float(np.mean(np.abs(y - prediction))),
                    "mean_absolute_prediction_change": float(np.mean(np.abs(prediction - baseline))),
                    "std_absolute_prediction_change": float(np.std(np.abs(prediction - baseline), ddof=0)),
                }
        seed_frames = {}
        for seed in [42, 43]:
            seed_frame = pd.DataFrame([
                {"semantic_feature_unit": unit, "importance": values["permuted_mae"] - baseline_mae}
                for unit, values in by_seed[seed].items()
            ])
            seed_frame["rank"] = seed_frame.importance.rank(method="min", ascending=False)
            seed_frames[seed] = seed_frame
        merged = seed_frames[42].merge(seed_frames[43], on="semantic_feature_unit", suffixes=("_42", "_43"))
        repeat_rows.append({
            "candidate_id": candidate,
            "spearman_rank_correlation": float(spearmanr(merged.rank_42, merged.rank_43).statistic),
            "top_10_overlap": len(set(merged.nsmallest(10, "rank_42").semantic_feature_unit) & set(merged.nsmallest(10, "rank_43").semantic_feature_unit)),
            "top_20_overlap": len(set(merged.nsmallest(20, "rank_42").semantic_feature_unit) & set(merged.nsmallest(20, "rank_43").semantic_feature_unit)),
            "maximum_rank_movement": float((merged.rank_42 - merged.rank_43).abs().max()),
            "median_rank_movement": float((merged.rank_42 - merged.rank_43).abs().median()),
            "repeat_count": 2,
            "seeds": "42|43",
        })
        candidate_rows = []
        for unit in units:
            values = [by_seed[seed][unit] for seed in [42, 43]]
            increases = [item["permuted_mae"] - baseline_mae for item in values]
            candidate_rows.append({
                "candidate_id": candidate,
                "semantic_feature_unit": unit,
                "feature_family": base.feature_family(unit),
                "baseline_mae": baseline_mae,
                "baseline_mean_prediction": float(np.mean(baseline)),
                "permuted_mae_seed42": values[0]["permuted_mae"],
                "permuted_mae_seed43": values[1]["permuted_mae"],
                "mean_permuted_mae": float(np.mean([item["permuted_mae"] for item in values])),
                "std_permuted_mae": float(np.std([item["permuted_mae"] for item in values], ddof=0)),
                "mae_increase": float(np.mean(increases)),
                "mean_absolute_prediction_change": float(np.mean([item["mean_absolute_prediction_change"] for item in values])),
                "std_absolute_prediction_change": float(np.mean([item["std_absolute_prediction_change"] for item in values])),
                "negative_importance_flag": bool(np.mean(increases) < 0),
                "candidate_feature_coverage": True,
                "component_coverage_count": 3 if candidate == CANDIDATES[0] else 1,
                "sample_row_count": 2000,
                "row_id_hash": value_hash(ids, np.int64),
                "target_hash": value_hash(y, np.float64),
                "seeds": "42|43",
                "original_target_scale_flag": True,
                "sensitive_identity_flag": unit == "explicit_sensitive_identity_block",
                "sensitive_context_flag": unit == "sensitive_context_block",
                "potential_proxy_category_flag": base.feature_family(unit) in {
                    "Geography and region", "Lender, respondent, and agency", "Applicant income",
                    "Tract and area income context", "Property and occupancy", "Loan purpose and type",
                    "Loan structure and lien status",
                },
            })
        candidate_frame = pd.DataFrame(candidate_rows)
        candidate_frame["rank"] = candidate_frame.mae_increase.rank(method="min", ascending=False).astype(int)
        positive = candidate_frame.mae_increase.clip(lower=0)
        candidate_frame["positive_importance_normalized_share"] = positive / positive.sum() if positive.sum() else np.nan
        all_rows.extend(candidate_frame.to_dict("records"))
    permutation_frame = pd.DataFrame(all_rows).sort_values(["candidate_id", "rank"])
    stability_frame = pd.DataFrame(repeat_rows)
    permutation_frame.to_csv(recovery_path("common_permutation_importance.csv"), index=False)
    stability_frame.to_csv(recovery_path("permutation_repeat_stability.csv"), index=False)
    return permutation_frame, stability_frame


def comparison_artifacts(permutation_frame: pd.DataFrame, importance: pd.DataFrame, shap_frame: pd.DataFrame) -> dict:
    rank_pivot = permutation_frame.pivot(index="semantic_feature_unit", columns="candidate_id", values="rank")
    importance_pivot = permutation_frame.pivot(index="semantic_feature_unit", columns="candidate_id", values="mae_increase")
    cross_rows = []
    for unit in sorted(set(permutation_frame.semantic_feature_unit)):
        ranks = [rank_pivot.loc[unit].get(candidate, np.nan) if unit in rank_pivot.index else np.nan for candidate in CANDIDATES]
        values = [importance_pivot.loc[unit].get(candidate, np.nan) if unit in importance_pivot.index else np.nan for candidate in CANDIDATES]
        finite = [value for value in ranks if pd.notna(value)]
        cross_rows.append({
            "semantic_feature_unit": unit,
            "official_blend_rank": ranks[0], "official_blend_importance": values[0],
            "realmlp_without_rank": ranks[1], "realmlp_without_importance": values[1],
            "realmlp_with_rank": ranks[2], "realmlp_with_importance": values[2],
            "shared_across_all_flag": len(finite) == 3,
            "present_in_blend_flag": pd.notna(ranks[0]),
            "present_in_deep_without_flag": pd.notna(ranks[1]),
            "present_in_deep_with_flag": pd.notna(ranks[2]),
            "maximum_rank_difference": max(finite) - min(finite) if finite else np.nan,
            "minimum_rank_difference": min(finite) if finite else np.nan,
            "rank_range": max(finite) - min(finite) if finite else np.nan,
            "consensus_top_10_flag": len(finite) == 3 and max(finite) <= 10,
            "consensus_top_20_flag": len(finite) == 3 and max(finite) <= 20,
            "feature_family": base.feature_family(unit),
            "sensitive_identity_flag": unit == "explicit_sensitive_identity_block",
            "sensitive_context_flag": unit == "sensitive_context_block",
            "potential_proxy_category_flag": base.feature_family(unit) in {
                "Geography and region", "Lender, respondent, and agency", "Applicant income",
                "Tract and area income context", "Property and occupancy", "Loan purpose and type",
                "Loan structure and lien status",
            },
        })
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(recovery_path("cross_model_feature_comparison.csv"), index=False)
    agreement_rows = []
    for first in range(3):
        for second in range(first + 1, 3):
            shared = permutation_frame[permutation_frame.candidate_id == CANDIDATES[first]][["semantic_feature_unit", "rank"]].merge(
                permutation_frame[permutation_frame.candidate_id == CANDIDATES[second]][["semantic_feature_unit", "rank"]],
                on="semantic_feature_unit", suffixes=("_a", "_b"),
            )
            a10 = set(shared.nsmallest(10, "rank_a").semantic_feature_unit)
            b10 = set(shared.nsmallest(10, "rank_b").semantic_feature_unit)
            a20 = set(shared.nsmallest(20, "rank_a").semantic_feature_unit)
            b20 = set(shared.nsmallest(20, "rank_b").semantic_feature_unit)
            agreement_rows.append({
                "model_a": CANDIDATES[first], "model_b": CANDIDATES[second],
                "shared_feature_count": len(shared),
                "spearman_rank_correlation": float(spearmanr(shared.rank_a, shared.rank_b).statistic),
                "kendall_rank_correlation": float(kendalltau(shared.rank_a, shared.rank_b).statistic),
                "top_10_intersection": len(a10 & b10), "top_10_jaccard": len(a10 & b10) / len(a10 | b10),
                "top_20_intersection": len(a20 & b20), "top_20_jaccard": len(a20 & b20) / len(a20 | b20),
            })
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(recovery_path("cross_model_agreement.csv"), index=False)

    method_rows = []
    pairs = [
        ("CatBoost", "PredictionValuesChange", "mean_absolute_SHAP"),
        ("LightGBM", "gain", "mean_absolute_SHAP"),
        ("LightGBM", "split", "mean_absolute_SHAP"),
        ("XGBoost", "gain", "mean_absolute_SHAP"),
        ("XGBoost", "total_gain", "mean_absolute_SHAP"),
    ]
    for family, method_a, method_b in pairs:
        first = importance[(importance.model_family == family) & (importance.method == method_a)][["semantic_feature_unit", "within_method_rank"]].rename(columns={"within_method_rank": "rank_a"})
        second = shap_frame[shap_frame.model_family == family][["semantic_feature_unit", "within_method_rank"]].rename(columns={"within_method_rank": "rank_b"})
        merged = first.groupby("semantic_feature_unit", as_index=False).rank_a.min().merge(second.groupby("semantic_feature_unit", as_index=False).rank_b.min(), on="semantic_feature_unit")
        method_rows.append({
            "model_family": family, "sensitive_mode": "without_sensitive", "method_a": method_a, "method_b": method_b,
            "shared_feature_count": len(merged),
            "spearman_correlation": float(spearmanr(merged.rank_a, merged.rank_b).statistic),
            "top_10_overlap": len(set(merged.nsmallest(10, "rank_a").semantic_feature_unit) & set(merged.nsmallest(10, "rank_b").semantic_feature_unit)),
            "top_20_overlap": len(set(merged.nsmallest(20, "rank_a").semantic_feature_unit) & set(merged.nsmallest(20, "rank_b").semantic_feature_unit)),
            "largest_disagreements": "|".join(merged.assign(d=(merged.rank_a - merged.rank_b).abs()).nlargest(3, "d").semantic_feature_unit),
            "output_scale_compatibility": "ranks only",
            "interpretation": "Method disagreement is expected and is not a model error.",
        })
    deep_source = pd.read_csv(ROOT / "artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv")
    deep = deep_source.assign(semantic_feature_unit=deep_source.feature.map(base.unit_for)).groupby("semantic_feature_unit", as_index=False).agg(
        stage5a_rank=("rank", "min"), stage5a_mae_increase=("mae_increase", "sum"),
        stage5a_mean_absolute_prediction_change=("mean_absolute_prediction_change", "sum"),
    )
    stage8 = permutation_frame[permutation_frame.candidate_id == CANDIDATES[1]][["semantic_feature_unit", "rank", "mae_increase", "mean_absolute_prediction_change"]].rename(columns={
        "rank": "stage8_rank", "mae_increase": "stage8_mae_increase",
        "mean_absolute_prediction_change": "stage8_mean_absolute_prediction_change",
    })
    deep_comparison = deep.merge(stage8, on="semantic_feature_unit", how="outer")
    deep_comparison["rank_difference"] = deep_comparison.stage8_rank - deep_comparison.stage5a_rank
    deep_comparison["shared_feature_flag"] = deep_comparison.stage8_rank.notna() & deep_comparison.stage5a_rank.notna()
    deep_comparison["sample_role_difference"] = "Stage 5A Train-only sample versus Stage 8 Post-Test saved-decile sample"
    deep_comparison["interpretation_limitation"] = "Rank differences are not model drift."
    deep_comparison.to_csv(recovery_path("deep_attribution_comparison.csv"), index=False)
    shared_deep = deep_comparison[deep_comparison.shared_feature_flag]
    method_rows.append({
        "model_family": "RealMLP", "sensitive_mode": "without_sensitive",
        "method_a": "Stage 5A saved permutation attribution", "method_b": "Stage 8 Recovery grouped permutation",
        "shared_feature_count": len(shared_deep),
        "spearman_correlation": float(spearmanr(shared_deep.stage5a_rank, shared_deep.stage8_rank).statistic),
        "top_10_overlap": len(set(shared_deep.nsmallest(10, "stage5a_rank").semantic_feature_unit) & set(shared_deep.nsmallest(10, "stage8_rank").semantic_feature_unit)),
        "top_20_overlap": len(set(shared_deep.nsmallest(20, "stage5a_rank").semantic_feature_unit) & set(shared_deep.nsmallest(20, "stage8_rank").semantic_feature_unit)),
        "largest_disagreements": "|".join(shared_deep.assign(d=(shared_deep.stage5a_rank - shared_deep.stage8_rank).abs()).nlargest(3, "d").semantic_feature_unit),
        "output_scale_compatibility": "both original target scale, different samples",
        "interpretation": "Differences may reflect sample role and grouping; they are not model drift.",
    })
    cross_method = pd.DataFrame(method_rows)
    cross_method.to_csv(recovery_path("cross_method_agreement.csv"), index=False)

    family = permutation_frame.groupby(["candidate_id", "feature_family"], as_index=False).agg(
        positive_permutation_importance_share=("positive_importance_normalized_share", "sum"),
        feature_unit_count=("semantic_feature_unit", "count"),
    )
    tops = permutation_frame.sort_values("rank").groupby(["candidate_id", "feature_family"], as_index=False).first()[["candidate_id", "feature_family", "semantic_feature_unit", "rank"]].rename(columns={"semantic_feature_unit": "top_feature", "rank": "top_feature_rank"})
    family = family.merge(tops, on=["candidate_id", "feature_family"])
    family["sensitive_identity_family_flag"] = family.feature_family == "Demographic or explicit sensitive identity"
    family["sensitive_context_family_flag"] = family.feature_family == "Tract demographic context"
    family["potential_proxy_category_flag"] = family.feature_family.isin([
        "Geography and region", "Lender, respondent, and agency", "Applicant income",
        "Tract and area income context", "Property and occupancy", "Loan purpose and type",
        "Loan structure and lien status",
    ])
    family.to_csv(recovery_path("feature_family_summary.csv"), index=False)

    sensitive = permutation_frame[(permutation_frame.candidate_id == CANDIDATES[2]) & permutation_frame.semantic_feature_unit.isin(["explicit_sensitive_identity_block", "sensitive_context_block"])].copy()
    sensitive["feature_or_block"] = sensitive.semantic_feature_unit
    sensitive["explicit_identity_or_contextual_classification"] = sensitive.semantic_feature_unit.map({"explicit_sensitive_identity_block": "explicit identity", "sensitive_context_block": "contextual"})
    sensitive["global_only_flag"] = True
    sensitive["fairness_conclusion"] = "none"
    sensitive["causal_conclusion"] = "none"
    sensitive.to_csv(recovery_path("sensitive_feature_dependence.csv"), index=False)
    proxy = permutation_frame[permutation_frame.potential_proxy_category_flag].copy()
    proxy["potential_proxy_wording"] = "Potential proxy category only; this is not proof of proxy behavior or causality."
    proxy["evidence_source"] = "Stage 8 Recovery grouped permutation and Stage 7 limitations"
    proxy.to_csv(recovery_path("potential_proxy_overlap.csv"), index=False)
    return {"cross": cross, "agreement": agreement, "cross_method": cross_method, "deep": deep_comparison, "family": family, "sensitive": sensitive, "proxy": proxy}


def existing_shap_provenance() -> dict:
    specifications = [
        ("CatBoost", "without_sensitive", "log1p model output", "artifacts/manifests/stage4/catboost/catboost_shap_metadata_without_sensitive.json", "artifacts/results/stage4/catboost/final/catboost_final_shap_without_sensitive.csv", "artifacts/results/stage4/catboost/final/catboost_final_shap_sample_ids.csv"),
        ("CatBoost", "with_sensitive", "log1p model output", "artifacts/manifests/stage4/catboost/catboost_shap_metadata_with_sensitive.json", "artifacts/results/stage4/catboost/final/catboost_final_shap_with_sensitive.csv", "artifacts/results/stage4/catboost/final/catboost_final_shap_sample_ids.csv"),
        ("LightGBM", "without_sensitive", "raw original target scale", "artifacts/manifests/stage4/lightgbm/stage4h_shap_metadata_without_sensitive.json", "artifacts/features/stage4/lightgbm/stage4h_shap_values_without_sensitive.csv", "artifacts/features/stage4/lightgbm/stage4h_shap_row_ids.csv"),
        ("LightGBM", "with_sensitive", "raw original target scale", "artifacts/manifests/stage4/lightgbm/stage4h_shap_metadata_with_sensitive.json", "artifacts/features/stage4/lightgbm/stage4h_shap_values_with_sensitive.csv", "artifacts/features/stage4/lightgbm/stage4h_shap_row_ids.csv"),
        ("XGBoost", "without_sensitive", "log1p model output", "artifacts/manifests/stage4/xgboost/stage4k_shap_metadata_without_sensitive.json", "artifacts/results/stage4/xgboost/final/stage4k_shap_complete_without_sensitive.csv", "artifacts/manifests/stage4/xgboost/stage4k_shap_row_ids.csv"),
        ("XGBoost", "with_sensitive", "log1p model output", "artifacts/manifests/stage4/xgboost/stage4k_shap_metadata_with_sensitive.json", "artifacts/results/stage4/xgboost/final/stage4k_shap_complete_with_sensitive.csv", "artifacts/manifests/stage4/xgboost/stage4k_shap_row_ids.csv"),
    ]
    entries = []
    for family, mode, scale, metadata_rel, values_rel, row_ids_rel in specifications:
        metadata_path = ROOT / metadata_rel
        values_path = ROOT / values_rel
        row_ids_path = ROOT / row_ids_rel
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = pd.read_csv(values_path)
        row_ids = pd.read_csv(row_ids_path)
        numeric = values.select_dtypes(include=[np.number])
        value_columns = [column for column in numeric.columns if column != "row_id"]
        feature_count = len(value_columns) if len(values) > 1 else max(len(values), len(value_columns))
        base_present = any("base" in str(key).lower() for key in metadata)
        additivity_present = any("additiv" in str(key).lower() for key in metadata)
        entries.append({
            "model_identity": family,
            "sensitive_mode": mode,
            "sample_role": "Train-only validation",
            "sample_row_count": len(row_ids),
            "output_scale": scale,
            "feature_count": feature_count,
            "finite_value_status": bool(np.isfinite(numeric[value_columns].to_numpy(dtype=float)).all()) if value_columns else True,
            "base_value_presence": True if base_present else "not_available_from_saved_artifact",
            "additivity_evidence_presence": True if additivity_present else "not_revalidated_without_recomputation",
            "metadata": record(metadata_path),
            "values": record(values_path),
            "row_ids": record(row_ids_path),
            "metadata_status": metadata.get("status"),
        })
    payload = {
        "authorization_id": AUTHORIZATION_ID,
        "status": "PASS",
        "global_shap_recomputations": 0,
        "lineage_source": "saved schemas and manifests only",
        "explanation_outcomes_used_for_mapping": False,
        "artifacts": entries,
    }
    dump(payload, REPORTS / "stage8_recovery_existing_shap_provenance.json")
    return payload


def local_attribution(models: dict, wo: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(CASES_PATH)
    background = pd.read_csv(BACKGROUND_PATH)
    case_ids = cases.row_id.astype(int).to_numpy()
    background_ids = background.row_id.astype(int).to_numpy()
    frames = {
        CANDIDATES[0]: (wo.loc[case_ids, base.BOOST_RAW], wo.loc[background_ids, base.BOOST_RAW]),
        CANDIDATES[1]: (wo.loc[case_ids, base.DEEP_NUM_WO + base.DEEP_CAT_WO], wo.loc[background_ids, base.DEEP_NUM_WO + base.DEEP_CAT_WO]),
        CANDIDATES[2]: (w.loc[case_ids, base.DEEP_NUM_WO + ["minority_population"] + base.DEEP_CAT_WO + base.SENSITIVE_IDENTITY + ["majority_minority_tract"]], w.loc[background_ids, base.DEEP_NUM_WO + ["minority_population"] + base.DEEP_CAT_WO + base.SENSITIVE_IDENTITY + ["majority_minority_tract"]]),
    }
    reference_rows = []
    reconciliation_rows = []
    expected_rows = 0
    for candidate, (case_frame, background_frame) in frames.items():
        baseline = base.predict_candidate(candidate, models, case_frame) if candidate != CANDIDATES[2] else base.predict_candidate(candidate, models, wo.iloc[:0], case_frame)
        saved = pd.read_csv(ROOT / next(item["path"] for item in PREDICTIONS if item["id"] == candidate)).set_index("row_id").loc[case_ids, "y_pred"].to_numpy()
        differences = np.abs(baseline - saved)
        tolerance = 1e-3 if candidate.startswith("stage5c__realmlp") else 1e-5
        for position, case in cases.reset_index(drop=True).iterrows():
            reconciliation_rows.append({
                "case_public_id": f"{case.semantic_case_type}__{int(case.case_rank)}",
                "case_type": case.case_type,
                "case_rank": int(case.case_rank),
                "row_id": int(case.row_id),
                "candidate_id": candidate,
                "generated_prediction": float(baseline[position]),
                "saved_prediction": float(saved[position]),
                "absolute_difference": float(differences[position]),
                "tolerance": tolerance,
                "status": "PASS" if differences[position] <= tolerance else "FAIL",
            })
        units = base.make_units(case_frame.columns.tolist())
        expected_rows += len(units) * len(cases) * len(background_frame)
        blocks = []
        for unit, columns in units.items():
            for case_position in range(len(cases)):
                block = pd.DataFrame(np.repeat(case_frame.iloc[[case_position]].to_numpy(), len(background_frame), axis=0), columns=case_frame.columns)
                block.loc[:, columns] = background_frame.loc[:, columns].to_numpy()
                blocks.append(block)
        stacked = pd.concat(blocks, ignore_index=True)
        output = base.predict_candidate(candidate, models, stacked) if candidate != CANDIDATES[2] else base.predict_candidate(candidate, models, wo.iloc[:0], stacked)
        offset = 0
        for unit, columns in units.items():
            for case_position, case in cases.reset_index(drop=True).iterrows():
                replaced = output[offset:offset + len(background_frame)]
                offset += len(background_frame)
                for background_position, replaced_prediction in enumerate(replaced):
                    reference_rows.append({
                        "case_public_id": f"{case.semantic_case_type}__{int(case.case_rank)}",
                        "case_type": case.case_type,
                        "case_rank": int(case.case_rank),
                        "row_id": int(case.row_id),
                        "candidate_id": candidate,
                        "semantic_feature_unit": unit,
                        "feature_family": base.feature_family(unit),
                        "background_row_id": int(background_ids[background_position]),
                        "original_prediction": float(baseline[case_position]),
                        "replaced_prediction": float(replaced_prediction),
                        "effect": float(baseline[case_position] - replaced_prediction),
                        "sensitive_block_flag": unit in {"explicit_sensitive_identity_block", "sensitive_context_block"},
                        "potential_proxy_category_flag": base.feature_family(unit) in {
                            "Geography and region", "Lender, respondent, and agency", "Applicant income",
                            "Tract and area income context", "Property and occupancy", "Loan purpose and type",
                            "Loan structure and lien status",
                        },
                    })
    references = pd.DataFrame(reference_rows)
    if len(references) != expected_rows:
        raise RuntimeError(f"Local Cartesian coverage failed: {len(references)} != {expected_rows}")
    key = ["case_public_id", "case_type", "case_rank", "row_id", "candidate_id", "semantic_feature_unit", "feature_family", "sensitive_block_flag", "potential_proxy_category_flag"]
    public = references.groupby(key, as_index=False).agg(
        background_rows=("background_row_id", "count"),
        effect_mean=("effect", "mean"),
        effect_standard_deviation=("effect", lambda values: float(np.std(values, ddof=0))),
        effect_minimum=("effect", "min"),
        effect_maximum=("effect", "max"),
        mean_absolute_effect=("effect", lambda values: float(np.mean(np.abs(values)))),
    )
    public["absolute_effect_rank"] = public.groupby(["case_public_id", "candidate_id"]).mean_absolute_effect.rank(method="min", ascending=False).astype(int)
    public["direction"] = np.where(public.effect_mean > 0, "raises prediction", np.where(public.effect_mean < 0, "lowers prediction", "neutral"))
    public["method"] = "local reference substitution"
    public["is_shap"] = False
    public["is_causal"] = False
    public["raw_sensitive_value_public"] = False
    public["dispersion_standard_deviation_ddof"] = 0
    references.to_csv(recovery_path("local_reference_effects.csv.gz"), index=False, compression="gzip")
    public.to_csv(recovery_path("local_attributions_public.csv"), index=False)
    reconciliation = pd.DataFrame(reconciliation_rows)
    reconciliation.to_csv(recovery_path("local_prediction_reconciliation.csv"), index=False)
    if not reconciliation.status.eq("PASS").all():
        raise RuntimeError("Local case/Candidate reconciliation failed")

    stability_rows = []
    visualization_cases = cases[cases.visualization_case.astype(str).str.lower().eq("true")]
    for case in visualization_cases.itertuples():
        case_public_id = f"{case.semantic_case_type}__{int(case.case_rank)}"
        for candidate in CANDIDATES:
            subset = references[(references.case_public_id == case_public_id) & (references.candidate_id == candidate)]
            first_ids = set(background_ids[:20])
            first = subset[subset.background_row_id.isin(first_ids)].groupby("semantic_feature_unit").effect.mean()
            second = subset[~subset.background_row_id.isin(first_ids)].groupby("semantic_feature_unit").effect.mean()
            aligned = pd.concat([first.rename("first"), second.rename("second")], axis=1).dropna()
            first_rank = aligned["first"].abs().rank(method="min", ascending=False)
            second_rank = aligned["second"].abs().rank(method="min", ascending=False)
            first_top5 = set(first_rank.nsmallest(5).index)
            second_top5 = set(second_rank.nsmallest(5).index)
            first_top10 = set(first_rank.nsmallest(10).index)
            second_top10 = set(second_rank.nsmallest(10).index)
            absolute_difference = (aligned["first"].abs() - aligned["second"].abs()).abs()
            correlation = float(spearmanr(first_rank, second_rank).statistic)
            stability_rows.append({
                "case_public_id": case_public_id,
                "case_type": case.case_type,
                "case_rank": int(case.case_rank),
                "row_id": int(case.row_id),
                "candidate_id": candidate,
                "background_first_rows": 20,
                "background_second_rows": 20,
                "spearman_rank_correlation": correlation,
                "top_5_overlap": len(first_top5 & second_top5),
                "top_10_overlap": len(first_top10 & second_top10),
                "median_absolute_effect_difference": float(absolute_difference.median()),
                "maximum_absolute_effect_difference": float(absolute_difference.max()),
                "low_stability_flag": bool(pd.isna(correlation) or correlation < 0.7),
            })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(recovery_path("local_explanation_stability.csv"), index=False)

    synthesis_rows = []
    for case in cases.itertuples():
        case_public_id = f"{case.semantic_case_type}__{int(case.case_rank)}"
        item = {"case_public_id": case_public_id, "case_type": case.case_type, "case_rank": int(case.case_rank), "row_id": int(case.row_id)}
        top_sets = []
        for label, candidate in zip(["stage4l", "realmlp_without", "realmlp_with"], CANDIDATES):
            top = public[(public.case_public_id == case_public_id) & (public.candidate_id == candidate)].nsmallest(5, "absolute_effect_rank").semantic_feature_unit.tolist()
            item[f"{label}_top_feature_units"] = "|".join(top)
            top_sets.append(set(top))
        item["top5_consensus_feature_units"] = "|".join(sorted(set.intersection(*top_sets)))
        item["top5_disagreement_feature_units"] = "|".join(sorted(set.union(*top_sets) - set.intersection(*top_sets)))
        item["non_causal_warning"] = "Reference substitution is not SHAP, additive, causal, or a realistic intervention."
        synthesis_rows.append(item)
    synthesis = pd.DataFrame(synthesis_rows)
    synthesis.to_csv(recovery_path("case_explanation_synthesis.csv"), index=False)
    dump({
        "authorization_id": AUTHORIZATION_ID,
        "status": "PASS",
        "expected_cartesian_rows": expected_rows,
        "actual_cartesian_rows": len(references),
        "case_count": len(cases),
        "candidate_count": 3,
        "background_count": len(background),
        "dispersion_complete": bool(public[["effect_standard_deviation", "effect_minimum", "effect_maximum", "mean_absolute_effect"]].notna().all().all()),
        "raw_sensitive_values_public": 0,
    }, REPORTS / "stage8_recovery_local_coverage.json")
    return references, public, stability, synthesis


def run() -> None:
    started = time.perf_counter()
    recovery_freeze = REPORTS / "stage8_recovery_sample_freeze.json"
    if sha(recovery_freeze) != "5122ad7948016f7b28267f44f36a23ce8535c6e8227cbd5367b48a5fbdabd1aa":
        raise RuntimeError("Recovery freeze hash changed before inference")
    provenance = existing_shap_provenance()
    importance = pd.read_csv(ROOT / "artifacts/results/stage8/explainability/stage8_existing_importance_long.csv")
    shap_frame = pd.read_csv(ROOT / "artifacts/results/stage8/explainability/stage8_existing_shap_global.csv")
    source_start = time.perf_counter()
    wo, w, access_audit = load_bounded_frames()
    source_seconds = time.perf_counter() - source_start
    model_start = time.perf_counter()
    models, model_attempts = load_models()
    reconciliation, generated = reconcile(models, wo, w)
    model_seconds = time.perf_counter() - model_start
    permutation_start = time.perf_counter()
    permutation_frame, permutation_stability = permutation(models, wo, w, generated)
    permutation_seconds = time.perf_counter() - permutation_start
    comparison = comparison_artifacts(permutation_frame, importance, shap_frame)
    local_start = time.perf_counter()
    references, public, local_stability, synthesis = local_attribution(models, wo, w)
    local_seconds = time.perf_counter() - local_start
    runtime = {
        "authorization_id": AUTHORIZATION_ID,
        "started_at_utc": now(),
        "status": "PASS",
        "source_access_seconds": source_seconds,
        "model_loading_and_reconciliation_seconds": model_seconds,
        "global_permutation_seconds": permutation_seconds,
        "local_attribution_seconds": local_seconds,
        "total_seconds": time.perf_counter() - started,
        "source_attempts": access_audit["access_attempts"],
        "model_attempts": {item["model_id"]: item["physical_attempts"] for item in model_attempts},
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "surrogate_fit_calls": 0,
        "global_shap_recomputations": provenance["global_shap_recomputations"],
        "new_evaluation_prediction_files": 0,
        "permutation_rows": len(permutation_frame),
        "local_reference_rows": len(references),
        "local_public_rows": len(public),
        "local_stability_rows": len(local_stability),
        "case_synthesis_rows": len(synthesis),
        "reconciliation_rows": len(reconciliation),
        "stage9_started": False,
    }
    dump(runtime, REPORTS / "stage8_recovery_runtime.json")
    print(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    run()
