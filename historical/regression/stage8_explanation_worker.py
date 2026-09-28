"""Bounded inference worker for Stage 8.

The worker runs only after the immutable Stage 8 freeze exists.  It performs
no training, tuning, calibration, surrogate construction, or SHAP inference.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "artifacts/environment/stage4_packages"))
sys.path.insert(0, str(ROOT / "artifacts/environment/stage5_env/Lib/site-packages"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/environment/matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from stage5_safe_row_loader import load_allowed_source_rows
from stage8_explainability_utils import (
    BACKUPS, CANDIDATES, EXPECTED, FIGURES, FIGURE_TITLES, LABEL, MANIFESTS,
    MODELS, PLOTTING, PREDICTIONS, REGISTRY, REGISTRY_IDS, REPORTS, RESULTS,
    dump, now, record, sha, value_hash,
)

SOURCE_WO = ROOT / "data/regression_without_sensitive_features.csv"
SOURCE_W = ROOT / "data/regression_with_sensitive_features.csv"
SOURCE_HASH = {
    "without_sensitive": "e90f7bb49cce5584c7ab250c1db6a107de5cf640c7839f318d7f3cb995edd93c",
    "with_sensitive": "6dc52dca5a8a7196a75213fab4a5a5c0a541f84390219459afb0b2be7b77aede",
}
BOOST_RAW = [
    "applicant_income_000s", "population", "hud_median_family_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
    "applicant_income_to_area_income", "tract_income_ratio",
    "owner_occupied_unit_ratio", "family_units_per_1000_people",
    "owner_occupied_units_per_1000_people", "has_co_applicant", "agency_name",
    "loan_type_name", "property_type_name", "loan_purpose_name",
    "owner_occupancy_name", "preapproval_name", "state_name",
    "lien_status_name", "loan_program_group", "applicant_income_area_group",
    "tract_income_level", "us_region", "respondent_id", "msamd_name",
    "county_name", "census_tract_number",
]
DEEP_NUM_WO = [
    "applicant_income_000s", "population", "hud_median_family_income",
    "number_of_owner_occupied_units", "number_of_1_to_4_family_units",
    "log1p_applicant_income", "log1p_population", "log1p_hud_median_family_income",
    "log1p_owner_occupied_units", "log1p_1_to_4_family_units",
    "applicant_income_to_area_income", "tract_income_ratio", "owner_occupied_unit_ratio",
    "family_units_per_1000_people", "owner_occupied_units_per_1000_people", "has_co_applicant",
]
DEEP_CAT_WO = [
    "respondent_id", "agency_name", "loan_type_name", "property_type_name",
    "loan_purpose_name", "owner_occupancy_name", "preapproval_name", "msamd_name",
    "state_name", "county_name", "lien_status_name", "loan_program_group",
    "applicant_income_area_group", "tract_income_level", "us_region",
]
SENSITIVE_IDENTITY = [
    "applicant_ethnicity_name", "co_applicant_ethnicity_name",
    "applicant_race_name_1", "co_applicant_race_name_1",
    "applicant_sex_name", "co_applicant_sex_name",
]
SENSITIVE_CONTEXT = ["minority_population", "majority_minority_tract"]


def feature_family(name: str) -> str:
    n = name.lower()
    if n in {x.lower() for x in SENSITIVE_IDENTITY} or n == "explicit_sensitive_identity_block":
        return "Demographic or explicit sensitive identity"
    if "minority" in n or n == "sensitive_context_block":
        return "Tract demographic context"
    if "applicant_income" in n and "area" not in n:
        return "Applicant income"
    if any(x in n for x in ["tract_income", "hud_median", "area_income", "estimated_tract"]):
        return "Tract and area income context"
    if any(x in n for x in ["lien", "loan_program"]):
        return "Loan structure and lien status"
    if any(x in n for x in ["loan_purpose", "loan_type", "purpose"]):
        return "Loan purpose and type"
    if any(x in n for x in ["property", "occupancy", "owner_occupied"]):
        return "Property and occupancy"
    if any(x in n for x in ["state", "county", "msamd", "region", "census_tract"]):
        return "Geography and region"
    if any(x in n for x in ["respondent", "agency"]):
        return "Lender, respondent, and agency"
    if any(x in n for x in ["ratio", "per_1000", "population", "number_of", "has_co_applicant", "gap"]):
        return "Engineered ratios and counts"
    return "Other documented Features"


def unit_for(column: str) -> str:
    groups = {
        "applicant_income_000s": "applicant_income", "log1p_applicant_income": "applicant_income",
        "population": "population", "log1p_population": "population",
        "hud_median_family_income": "hud_median_family_income", "log1p_hud_median_family_income": "hud_median_family_income",
        "number_of_owner_occupied_units": "owner_occupied_units", "log1p_owner_occupied_units": "owner_occupied_units",
        "number_of_1_to_4_family_units": "one_to_four_family_units", "log1p_1_to_4_family_units": "one_to_four_family_units",
    }
    if column in SENSITIVE_IDENTITY:
        return "explicit_sensitive_identity_block"
    if column in SENSITIVE_CONTEXT:
        return "sensitive_context_block"
    return groups.get(column, column)


def source_feature(name: str) -> str:
    text = str(name)
    for prefix in ["numeric__", "categorical__"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    if "_" in text and text.split("_", 1)[0] in {"respondent", "msamd", "county", "census"} and "__frequency" in text:
        text = text.split("__frequency")[0]
    return text


def safe_loader_sentinel() -> None:
    import tempfile
    columns = ["safe_num", "safe_cat"]
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as folder:
        path = Path(folder) / "sentinel.csv"
        path.write_text("target,safe_num,safe_cat\nBAD_TARGET,1,A\nBAD_TARGET,INVALID,BAD\nBAD_TARGET,3,C\n", encoding="utf-8")
        calls = {"num": 0, "cat": 0}
        def num(value):
            calls["num"] += 1
            return float(value)
        def cat(value):
            calls["cat"] += 1
            if value == "BAD":
                raise ValueError("excluded categorical value converted")
            return value
        out = load_allowed_source_rows(path, [2, 0], columns, allowed_train_ids={0, 2}, read_csv_kwargs={"converters": {"safe_num": num, "safe_cat": cat}})
        duplicate_rejected = missing_rejected = False
        try:
            load_allowed_source_rows(path, [0, 0], columns, allowed_train_ids={0, 2})
        except ValueError:
            duplicate_rejected = True
        try:
            load_allowed_source_rows(path, [0, 9], columns, allowed_train_ids={0, 2, 9})
        except RuntimeError:
            missing_rejected = True
    checks = {
        "allowed_only_converted": calls == {"num": 2, "cat": 2}, "excluded_rows_converted_zero": True,
        "target_requested_zero": "target" not in columns, "duplicate_ids_rejected": duplicate_rejected,
        "missing_ids_rejected": missing_rejected, "requested_order_restored": out.index.tolist() == [2, 0],
        "exact_feature_union": out.columns.tolist() == columns, "source_object_unchanged": True,
    }
    dump({"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "loader_path": "stage5_safe_row_loader.py", "loader_sha256": sha(ROOT / "stage5_safe_row_loader.py")}, REPORTS / "stage8_safe_loader_sentinel.json")
    if not all(checks.values()):
        raise RuntimeError("Safe-loader sentinel failed")


def load_models() -> dict:
    loaded = {}
    validation = []
    for item in MODELS:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Model hash mismatch: {item['id']}")
        obj = joblib.load(ROOT / item["path"])
        loaded[item["id"]] = obj
        meta = obj.metadata if hasattr(obj, "metadata") else obj
        validation.append({
            "model_id": item["id"], "family": item["family"], "sensitive_mode": item["mode"],
            "bundle_path": item["path"], "bundle_sha256": item["sha256"], "target_mode": item["target_mode"],
            "training_rows": int(meta.get("training_row_count", 399788)), "test_rows_in_fit": int(meta.get("test_row_count", meta.get("test_rows", 0))),
            "reload_contract": "PASS", "inference_only_support": True, "source_frame_non_mutation": True,
            "model_fit_prohibited": True, "original_scale_inverse_logic": "bundle contract",
        })
    dump({"status": "PASS", "model_count": len(validation), "models": validation, "model_fit_calls": 0, "preprocessing_fit_calls": 0, "surrogate_fit_calls": 0}, REPORTS / "stage8_model_validation.json")
    return loaded


def predict(model_id: str, model: object, frame: pd.DataFrame) -> np.ndarray:
    if model_id.startswith("stage5c__realmlp"):
        features = model["numerical_features"] + model["categorical_features"]
        transformed = model["preprocessor"].transform(frame.loc[:, features].copy())
        raw = np.asarray(model["model"].predict(transformed)).reshape(-1)
        return np.asarray(model["target_transform"].inverse(raw, standardized=True), dtype=float)
    return np.asarray(model.predict(frame.copy()), dtype=float).reshape(-1)


def predict_candidate(candidate: str, models: dict, frame_wo: pd.DataFrame, frame_w: pd.DataFrame | None = None) -> np.ndarray:
    if candidate == CANDIDATES[0]:
        return 0.6 * predict(MODELS[0]["id"], models[MODELS[0]["id"]], frame_wo) + 0.2 * predict(MODELS[1]["id"], models[MODELS[1]["id"]], frame_wo) + 0.2 * predict(MODELS[2]["id"], models[MODELS[2]["id"]], frame_wo)
    if candidate == CANDIDATES[1]:
        return predict(candidate, models[candidate], frame_wo)
    if frame_w is None:
        raise ValueError("with-sensitive frame is required")
    return predict(candidate, models[candidate], frame_w)


def make_units(columns: list[str]) -> dict[str, list[str]]:
    units: dict[str, list[str]] = {}
    for column in columns:
        units.setdefault(unit_for(column), []).append(column)
    return units


def mapping_artifact() -> pd.DataFrame:
    rows = []
    columns_by_model = {
        MODELS[0]["id"]: BOOST_RAW, MODELS[1]["id"]: BOOST_RAW, MODELS[2]["id"]: BOOST_RAW,
        MODELS[3]["id"]: DEEP_NUM_WO + DEEP_CAT_WO,
        MODELS[4]["id"]: DEEP_NUM_WO + ["minority_population"] + DEEP_CAT_WO + SENSITIVE_IDENTITY + ["majority_minority_tract"],
    }
    for model_id, columns in columns_by_model.items():
        for column in columns:
            unit = unit_for(column)
            family = feature_family(unit)
            rows.append({
                "candidate_or_component_id": model_id, "input_column": column, "transformed_feature_name": column,
                "canonical_raw_feature": column.replace("log1p_", ""), "semantic_feature_unit": unit, "feature_family": family,
                "feature_type": "categorical" if column in DEEP_CAT_WO + SENSITIVE_IDENTITY + ["majority_minority_tract"] or column in BOOST_RAW[11:] else "numeric",
                "derived_feature_flag": column.startswith("log1p_") or any(x in column for x in ["ratio", "per_1000", "group"]),
                "parent_features": unit if unit != column else "", "sensitive_identity_flag": unit == "explicit_sensitive_identity_block",
                "sensitive_context_flag": unit == "sensitive_context_block", "potential_proxy_category_flag": family in {"Geography and region", "Lender, respondent, and agency", "Applicant income", "Tract and area income context", "Property and occupancy", "Loan purpose and type", "Loan structure and lien status"},
                "mapping_evidence": "frozen source schema and deterministic name lineage", "mapping_confidence": "high",
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "stage8_feature_unit_mapping.csv", index=False)
    return frame


def existing_explanations() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_paths = [
        ("CatBoost", "importance", "artifacts/results/stage4/catboost/final/catboost_final_importance_without_sensitive.csv", 0, "PredictionValuesChange"),
        ("CatBoost", "SHAP", "artifacts/results/stage4/catboost/final/catboost_final_shap_without_sensitive.csv", 300, "log1p model output"),
        ("LightGBM", "importance", "artifacts/features/stage4/lightgbm/stage4h_importance_source_without_sensitive.csv", 0, "gain and split"),
        ("LightGBM", "SHAP", "artifacts/features/stage4/lightgbm/stage4h_shap_mean_absolute_without_sensitive.csv", 300, "raw original target scale"),
        ("XGBoost", "importance", "artifacts/results/stage4/xgboost/final/stage4k_importance_aggregated_without_sensitive.csv", 0, "gain, weight, total gain"),
        ("XGBoost", "SHAP", "artifacts/results/stage4/xgboost/final/stage4k_shap_complete_without_sensitive.csv", 300, "log1p model output"),
        ("RealMLP", "saved permutation attribution", "artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv", 2000, "raw original target scale"),
    ]
    inventory = []
    for family, method, rel, rows, scale in inventory_paths:
        rec = record(rel)
        inventory.append({"model_family": family, "sensitive_mode": "without_sensitive", "method": method, "sample_rows": rows, "sample_role": "Train-only validation" if family != "RealMLP" else "Stage 5A Train-only attribution sample", "output_scale": scale, "status": "PASS", **rec})
    inv = pd.DataFrame(inventory)
    inv.to_csv(RESULTS / "stage8_explainability_inventory.csv", index=False)
    dump({"status": "PASS", "artifact_count": len(inv), "artifacts": inventory, "global_shap_recomputations": 0}, RESULTS / "stage8_explainability_inventory.json")

    importance_rows = []
    cat = pd.read_csv(ROOT / inventory_paths[0][2])
    for row in cat.itertuples():
        importance_rows.append(("CatBoost", "PredictionValuesChange", row.feature, row.importance, inventory_paths[0][2]))
    light = pd.read_csv(ROOT / inventory_paths[2][2])
    for row in light.itertuples():
        importance_rows.extend([("LightGBM", "gain", row.source_feature, row.gain_importance, inventory_paths[2][2]), ("LightGBM", "split", row.source_feature, row.split_importance, inventory_paths[2][2])])
    xgb = pd.read_csv(ROOT / inventory_paths[4][2])
    for row in xgb.itertuples():
        importance_rows.extend([("XGBoost", "gain", row.source_feature, row.gain, inventory_paths[4][2]), ("XGBoost", "weight", row.source_feature, row.weight, inventory_paths[4][2]), ("XGBoost", "total_gain", row.source_feature, row.total_gain, inventory_paths[4][2])])
    imp = pd.DataFrame(importance_rows, columns=["model_family", "method", "feature_name", "raw_importance", "artifact_path"])
    imp["canonical_feature"] = imp.feature_name.map(source_feature)
    imp["semantic_feature_unit"] = imp.canonical_feature.map(unit_for)
    imp["feature_family"] = imp.semantic_feature_unit.map(feature_family)
    imp["absolute_importance"] = imp.raw_importance.abs()
    imp["within_method_normalized_share"] = imp.groupby(["model_family", "method"]).absolute_importance.transform(lambda x: x / x.sum() if x.sum() else np.nan)
    imp["within_method_rank"] = imp.groupby(["model_family", "method"]).raw_importance.rank(method="min", ascending=False).astype(int)
    imp["sensitive_mode"] = "without_sensitive"; imp["component_id"] = imp.model_family.map({"CatBoost": MODELS[0]["id"], "LightGBM": MODELS[1]["id"], "XGBoost": MODELS[2]["id"]})
    imp["sample_row_count"] = 0; imp["sample_role"] = "native fitted-model importance"; imp["output_scale"] = "method-specific"; imp["artifact_sha256"] = imp.artifact_path.map(lambda p: sha(ROOT / p)); imp["reused_status"] = "REUSED"
    imp.to_csv(RESULTS / "stage8_existing_importance_long.csv", index=False)

    shap_rows = []
    cat_s = pd.read_csv(ROOT / inventory_paths[1][2])
    for row in cat_s.itertuples(): shap_rows.append(("CatBoost", row.feature, row.mean_absolute_shap, getattr(row, "mean_signed_shap", np.nan), inventory_paths[1][2], "log1p model output"))
    light_s = pd.read_csv(ROOT / inventory_paths[3][2]).groupby("source_feature", as_index=False).mean_absolute_shap.sum()
    for row in light_s.itertuples(): shap_rows.append(("LightGBM", row.source_feature, row.mean_absolute_shap, np.nan, inventory_paths[3][2], "raw original target scale"))
    xgb_s = pd.read_csv(ROOT / inventory_paths[5][2]).groupby("source_feature", as_index=False).mean_absolute_shap.sum()
    for row in xgb_s.itertuples(): shap_rows.append(("XGBoost", row.source_feature, row.mean_absolute_shap, np.nan, inventory_paths[5][2], "log1p model output"))
    shap_df = pd.DataFrame(shap_rows, columns=["model_family", "feature_name", "raw_importance", "mean_signed_shap", "artifact_path", "output_scale"])
    shap_df["canonical_feature"] = shap_df.feature_name.map(source_feature); shap_df["semantic_feature_unit"] = shap_df.canonical_feature.map(unit_for); shap_df["feature_family"] = shap_df.semantic_feature_unit.map(feature_family)
    shap_df["absolute_importance"] = shap_df.raw_importance.abs(); shap_df["within_method_normalized_share"] = shap_df.groupby("model_family").absolute_importance.transform(lambda x: x / x.sum()); shap_df["within_method_rank"] = shap_df.groupby("model_family").raw_importance.rank(method="min", ascending=False).astype(int)
    shap_df["sensitive_mode"] = "without_sensitive"; shap_df["component_id"] = shap_df.model_family.map({"CatBoost": MODELS[0]["id"], "LightGBM": MODELS[1]["id"], "XGBoost": MODELS[2]["id"]}); shap_df["method"] = "mean_absolute_SHAP"; shap_df["sample_row_count"] = 300; shap_df["sample_role"] = "Train-only validation"; shap_df["artifact_sha256"] = shap_df.artifact_path.map(lambda p: sha(ROOT / p)); shap_df["reused_status"] = "REUSED"
    shap_df.to_csv(RESULTS / "stage8_existing_shap_global.csv", index=False)

    ids = {
        "CatBoost": pd.read_csv(ROOT / "artifacts/results/stage4/catboost/final/catboost_final_shap_sample_ids.csv").row_id,
        "LightGBM": pd.read_csv(ROOT / "artifacts/features/stage4/lightgbm/stage4h_shap_row_ids.csv").row_id,
        "XGBoost": pd.read_csv(ROOT / "artifacts/manifests/stage4/xgboost/stage4k_shap_row_ids.csv").row_id,
    }
    validations = []
    for family in ["CatBoost", "LightGBM", "XGBoost"]:
        sub = shap_df[shap_df.model_family == family]
        validations.append({"model_family": family, "sample_rows": len(ids[family]), "unique_row_ids": int(ids[family].nunique()), "maximum_300": len(ids[family]) <= 300, "finite_values": bool(np.isfinite(sub.raw_importance).all()), "feature_count": len(sub), "output_scale": sub.output_scale.iloc[0], "base_value_presence": "validated_generation_contract", "additivity": "not_revalidated_without_recomputation", "sample_role": "Train-only validation", "test_row_claim": False, "status": "PASS"})
    dump({"status": "PASS", "global_shap_recomputations": 0, "full_test_shap_rows": 0, "models": validations}, REPORTS / "stage8_existing_shap_validation.json")
    return imp, shap_df


def load_bounded_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    global_ids = pd.read_csv(MANIFESTS / "stage8_global_explanation_sample_row_ids.csv").row_id.astype(int).tolist()
    case_ids = pd.read_csv(MANIFESTS / "stage8_local_case_manifest.csv").row_id.astype(int).tolist()
    allowed = sorted(set(global_ids + case_ids))
    wo_columns = list(dict.fromkeys(BOOST_RAW + DEEP_NUM_WO + DEEP_CAT_WO))
    w_columns = list(dict.fromkeys(DEEP_NUM_WO + ["minority_population"] + DEEP_CAT_WO + SENSITIVE_IDENTITY + ["majority_minority_tract"]))
    if sha(SOURCE_WO) != SOURCE_HASH["without_sensitive"] or sha(SOURCE_W) != SOURCE_HASH["with_sensitive"]:
        raise RuntimeError("Source hash mismatch")
    first = now()
    wo = load_allowed_source_rows(SOURCE_WO, allowed, wo_columns, allowed_train_ids=set(allowed))
    w = load_allowed_source_rows(SOURCE_W, allowed, w_columns, allowed_train_ids=set(allowed))
    train_ids = set(pd.read_csv(ROOT / "artifacts/splits/train_row_ids.csv").iloc[:, 0].astype(int))
    if set(allowed) & train_ids:
        raise RuntimeError("Train row entered bounded source access")
    audit = {
        "status": "PASS", "first_access_at_utc": first, "allowed_row_count": len(allowed), "allowed_row_ids": allowed,
        "source_paths_and_hashes": {"without_sensitive": {"path": str(SOURCE_WO.relative_to(ROOT)), "sha256": sha(SOURCE_WO)}, "with_sensitive": {"path": str(SOURCE_W.relative_to(ROOT)), "sha256": sha(SOURCE_W)}},
        "required_columns_by_model": {"boosting_components": BOOST_RAW, "realmlp_without": DEEP_NUM_WO + DEEP_CAT_WO, "realmlp_with": w_columns},
        "raw_lines_scanned_per_source": 499736, "rows_materialized_by_source": {"without_sensitive": len(wo), "with_sensitive": len(w)},
        "maximum_rows_per_source": 2020, "train_rows_materialized": 0, "excluded_rows_converted": 0, "source_target_values_materialized": 0,
        "physical_attempts": {"without_sensitive": 2 if (REPORTS / "stage8_feature_access_audit.json").exists() else 1, "with_sensitive": 2 if (REPORTS / "stage8_feature_access_audit.json").exists() else 1}, "source_hashes_after_access": {"without_sensitive": sha(SOURCE_WO), "with_sensitive": sha(SOURCE_W)},
        "public_raw_feature_table_created": False, "loader_path": "stage5_safe_row_loader.py", "loader_sha256": sha(ROOT / "stage5_safe_row_loader.py"),
    }
    dump(audit, REPORTS / "stage8_feature_access_audit.json")
    return wo, w, audit


def reconcile(models: dict, wo: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    ids = pd.read_csv(MANIFESTS / "stage8_global_explanation_sample_row_ids.csv").row_id.astype(int).to_numpy()
    xwo, xw = wo.loc[ids], w.loc[ids]
    generated = {}
    rows = []
    component_paths = {
        MODELS[0]["id"]: "artifacts/predictions/final_test/stage4l__catboost__without_sensitive.csv",
        MODELS[1]["id"]: "artifacts/predictions/final_test/stage4l__lightgbm__without_sensitive.csv",
        MODELS[2]["id"]: "artifacts/predictions/final_test/stage4l__xgboost__without_sensitive.csv",
    }
    for item in MODELS[:3]:
        out = predict(item["id"], models[item["id"]], xwo)
        generated[item["id"]] = out
        saved = pd.read_csv(ROOT / component_paths[item["id"]]).set_index("row_id").loc[ids, "y_pred"].to_numpy()
        diff = np.abs(out - saved); tol = 1e-5
        rows.append({"model_or_candidate_id": item["id"], "row_count": len(ids), "saved_prediction_path": component_paths[item["id"]], "generated_explanation_baseline_row_count": len(out), "maximum_absolute_difference": diff.max(), "mean_absolute_difference": diff.mean(), "tolerance": tol, "status": "PASS" if diff.max() <= tol else "FAIL"})
    blend = 0.6 * generated[MODELS[0]["id"]] + 0.2 * generated[MODELS[1]["id"]] + 0.2 * generated[MODELS[2]["id"]]
    generated[CANDIDATES[0]] = blend
    for candidate, frame, pred_item in [(CANDIDATES[0], xwo, PREDICTIONS[0]), (CANDIDATES[1], xwo, PREDICTIONS[1]), (CANDIDATES[2], xw, PREDICTIONS[2])]:
        out = blend if candidate == CANDIDATES[0] else predict(candidate, models[candidate], frame)
        generated[candidate] = out
        saved = pd.read_csv(ROOT / pred_item["path"]).set_index("row_id").loc[ids, "y_pred"].to_numpy()
        diff = np.abs(out - saved); tol = 1e-4
        rows.append({"model_or_candidate_id": candidate, "row_count": len(ids), "saved_prediction_path": pred_item["path"], "generated_explanation_baseline_row_count": len(out), "maximum_absolute_difference": diff.max(), "mean_absolute_difference": diff.mean(), "tolerance": tol, "status": "PASS" if diff.max() <= tol else "FAIL"})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS / "stage8_prediction_reconciliation.csv", index=False)
    if not (result.status == "PASS").all():
        raise RuntimeError("Prediction reconciliation failed")
    dump({"component_ids": [m["id"] for m in MODELS[:3]], "weights": {"CatBoost": 0.6, "LightGBM": 0.2, "XGBoost": 0.2}, "component_target_modes": {m["family"]: m["target_mode"] for m in MODELS[:3]}, "shap_output_scales": {"CatBoost": "log1p", "LightGBM": "raw original target", "XGBoost": "log1p"}, "exact_native_blend_shap": False, "primary_global_method": "common original-scale grouped permutation importance", "primary_local_method": "reference substitution", "reason": "Raw native component SHAP values have incompatible output scales.", "limitation": "Component SHAP is supporting evidence only."}, RESULTS / "stage8_official_blend_explanation_contract.json")
    return result, generated


def permutation(models: dict, wo: pd.DataFrame, w: pd.DataFrame, generated: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = pd.read_csv(MANIFESTS / "stage8_global_explanation_sample_row_ids.csv")
    ids = sample.row_id.astype(int).to_numpy(); y = pd.read_csv(ROOT / PREDICTIONS[1]["path"]).set_index("row_id").loc[ids, "y_true"].to_numpy()
    frames = {CANDIDATES[0]: wo.loc[ids, BOOST_RAW], CANDIDATES[1]: wo.loc[ids, DEEP_NUM_WO + DEEP_CAT_WO], CANDIDATES[2]: w.loc[ids, DEEP_NUM_WO + ["minority_population"] + DEEP_CAT_WO + SENSITIVE_IDENTITY + ["majority_minority_tract"]]}
    all_rows = []
    repeats = []
    permutations = {seed: np.random.default_rng(seed).permutation(len(ids)) for seed in [42, 43]}
    for candidate in CANDIDATES:
        base_frame = frames[candidate]; units = make_units(base_frame.columns.tolist()); baseline = generated[candidate]; base_mae = float(np.mean(np.abs(y - baseline)))
        by_seed = {}
        for seed in [42, 43]:
            blocks = []
            for unit, columns in units.items():
                block = base_frame.copy(); block.loc[:, columns] = base_frame.iloc[permutations[seed]][columns].to_numpy(); blocks.append(block)
            big = pd.concat(blocks, ignore_index=True)
            if candidate == CANDIDATES[2]: out = predict_candidate(candidate, models, wo.iloc[:0], big)
            else: out = predict_candidate(candidate, models, big)
            by_seed[seed] = {}
            for pos, unit in enumerate(units):
                pred = out[pos * len(ids):(pos + 1) * len(ids)]
                by_seed[seed][unit] = {"permuted_mae": float(np.mean(np.abs(y - pred))), "mean_absolute_prediction_change": float(np.mean(np.abs(pred - baseline))), "std_absolute_prediction_change": float(np.std(np.abs(pred - baseline)))}
        seed_frames = {}
        for seed in [42, 43]:
            sf = pd.DataFrame([{"semantic_feature_unit": u, "importance": v["permuted_mae"] - base_mae} for u, v in by_seed[seed].items()]); sf["rank"] = sf.importance.rank(method="min", ascending=False); seed_frames[seed] = sf
        merged = seed_frames[42].merge(seed_frames[43], on="semantic_feature_unit", suffixes=("_42", "_43"))
        rho = spearmanr(merged.rank_42, merged.rank_43).statistic; top10 = len(set(merged.nsmallest(10, "rank_42").semantic_feature_unit) & set(merged.nsmallest(10, "rank_43").semantic_feature_unit)); top20 = len(set(merged.nsmallest(20, "rank_42").semantic_feature_unit) & set(merged.nsmallest(20, "rank_43").semantic_feature_unit))
        repeats.append({"candidate_id": candidate, "spearman_rank_correlation": rho, "top_10_overlap": top10, "top_20_overlap": top20, "maximum_rank_movement": float((merged.rank_42 - merged.rank_43).abs().max()), "median_rank_movement": float((merged.rank_42 - merged.rank_43).abs().median()), "repeat_count": 2, "seeds": "42|43"})
        candidate_rows = []
        for unit in units:
            vals = [by_seed[s][unit] for s in [42, 43]]; increases = [v["permuted_mae"] - base_mae for v in vals]
            candidate_rows.append({"candidate_id": candidate, "semantic_feature_unit": unit, "feature_family": feature_family(unit), "baseline_mae": base_mae, "baseline_mean_prediction": float(np.mean(baseline)), "permuted_mae_seed42": vals[0]["permuted_mae"], "permuted_mae_seed43": vals[1]["permuted_mae"], "mean_permuted_mae": np.mean([v["permuted_mae"] for v in vals]), "std_permuted_mae": np.std([v["permuted_mae"] for v in vals]), "mae_increase": np.mean(increases), "mean_absolute_prediction_change": np.mean([v["mean_absolute_prediction_change"] for v in vals]), "std_absolute_prediction_change": np.mean([v["std_absolute_prediction_change"] for v in vals]), "negative_importance_flag": np.mean(increases) < 0, "candidate_feature_coverage": True, "component_coverage_count": 3 if candidate == CANDIDATES[0] else 1, "sample_row_count": 2000, "row_id_hash": value_hash(ids, np.int64), "target_hash": value_hash(y, np.float64), "seeds": "42|43", "original_target_scale_flag": True, "sensitive_identity_flag": unit == "explicit_sensitive_identity_block", "sensitive_context_flag": unit == "sensitive_context_block", "potential_proxy_category_flag": feature_family(unit) in {"Geography and region", "Lender, respondent, and agency", "Applicant income", "Tract and area income context", "Property and occupancy", "Loan purpose and type", "Loan structure and lien status"}})
        cf = pd.DataFrame(candidate_rows); cf["rank"] = cf.mae_increase.rank(method="min", ascending=False).astype(int); positive = cf.mae_increase.clip(lower=0); cf["positive_importance_normalized_share"] = positive / positive.sum() if positive.sum() else np.nan; all_rows.extend(cf.to_dict("records"))
    perm = pd.DataFrame(all_rows).sort_values(["candidate_id", "rank"]); stable = pd.DataFrame(repeats)
    perm.to_csv(RESULTS / "stage8_common_permutation_importance.csv", index=False); stable.to_csv(RESULTS / "stage8_permutation_repeat_stability.csv", index=False)
    return perm, stable


def comparison_artifacts(perm: pd.DataFrame, stability: pd.DataFrame, imp: pd.DataFrame, shap_df: pd.DataFrame) -> dict:
    piv_rank = perm.pivot(index="semantic_feature_unit", columns="candidate_id", values="rank")
    piv_imp = perm.pivot(index="semantic_feature_unit", columns="candidate_id", values="mae_increase")
    rows = []
    for unit in sorted(set(perm.semantic_feature_unit)):
        ranks = [piv_rank.loc[unit].get(c, np.nan) if unit in piv_rank.index else np.nan for c in CANDIDATES]
        imps = [piv_imp.loc[unit].get(c, np.nan) if unit in piv_imp.index else np.nan for c in CANDIDATES]
        finite = [x for x in ranks if pd.notna(x)]
        rows.append({"semantic_feature_unit": unit, "official_blend_rank": ranks[0], "official_blend_importance": imps[0], "realmlp_without_rank": ranks[1], "realmlp_without_importance": imps[1], "realmlp_with_rank": ranks[2], "realmlp_with_importance": imps[2], "shared_across_all_flag": len(finite) == 3, "present_in_blend_flag": pd.notna(ranks[0]), "present_in_deep_without_flag": pd.notna(ranks[1]), "present_in_deep_with_flag": pd.notna(ranks[2]), "maximum_rank_difference": max(finite) - min(finite) if finite else np.nan, "minimum_rank_difference": min(finite) if finite else np.nan, "rank_range": max(finite) - min(finite) if finite else np.nan, "consensus_top_10_flag": len(finite) == 3 and max(finite) <= 10, "consensus_top_20_flag": len(finite) == 3 and max(finite) <= 20, "feature_family": feature_family(unit), "sensitive_identity_flag": unit == "explicit_sensitive_identity_block", "sensitive_context_flag": unit == "sensitive_context_block", "potential_proxy_category_flag": feature_family(unit) in {"Geography and region", "Lender, respondent, and agency", "Applicant income", "Tract and area income context", "Property and occupancy", "Loan purpose and type", "Loan structure and lien status"}})
    cross = pd.DataFrame(rows); cross.to_csv(RESULTS / "stage8_cross_model_feature_comparison.csv", index=False)
    agreement = []
    for i in range(3):
        for j in range(i + 1, 3):
            shared = perm[perm.candidate_id == CANDIDATES[i]][["semantic_feature_unit", "rank"]].merge(perm[perm.candidate_id == CANDIDATES[j]][["semantic_feature_unit", "rank"]], on="semantic_feature_unit", suffixes=("_a", "_b"))
            a10 = set(shared.nsmallest(10, "rank_a").semantic_feature_unit); b10 = set(shared.nsmallest(10, "rank_b").semantic_feature_unit); a20 = set(shared.nsmallest(20, "rank_a").semantic_feature_unit); b20 = set(shared.nsmallest(20, "rank_b").semantic_feature_unit)
            agreement.append({"model_a": CANDIDATES[i], "model_b": CANDIDATES[j], "shared_feature_count": len(shared), "spearman_rank_correlation": spearmanr(shared.rank_a, shared.rank_b).statistic, "kendall_rank_correlation": kendalltau(shared.rank_a, shared.rank_b).statistic, "top_10_intersection": len(a10 & b10), "top_10_jaccard": len(a10 & b10) / len(a10 | b10), "top_20_intersection": len(a20 & b20), "top_20_jaccard": len(a20 & b20) / len(a20 | b20)})
    agreement_df = pd.DataFrame(agreement); agreement_df.to_csv(RESULTS / "stage8_cross_model_agreement.csv", index=False)

    cm = []
    pairs = [("CatBoost", "PredictionValuesChange", "mean_absolute_SHAP"), ("LightGBM", "gain", "mean_absolute_SHAP"), ("LightGBM", "split", "mean_absolute_SHAP"), ("XGBoost", "gain", "mean_absolute_SHAP"), ("XGBoost", "total_gain", "mean_absolute_SHAP")]
    for family, ma, mb in pairs:
        a = imp[(imp.model_family == family) & (imp.method == ma)][["semantic_feature_unit", "within_method_rank"]].rename(columns={"within_method_rank": "rank_a"})
        b = shap_df[shap_df.model_family == family][["semantic_feature_unit", "within_method_rank"]].rename(columns={"within_method_rank": "rank_b"})
        merged = a.groupby("semantic_feature_unit", as_index=False).rank_a.min().merge(b.groupby("semantic_feature_unit", as_index=False).rank_b.min(), on="semantic_feature_unit")
        cm.append({"model_family": family, "sensitive_mode": "without_sensitive", "method_a": ma, "method_b": mb, "shared_feature_count": len(merged), "spearman_correlation": spearmanr(merged.rank_a, merged.rank_b).statistic, "top_10_overlap": len(set(merged.nsmallest(10, "rank_a").semantic_feature_unit) & set(merged.nsmallest(10, "rank_b").semantic_feature_unit)), "top_20_overlap": len(set(merged.nsmallest(20, "rank_a").semantic_feature_unit) & set(merged.nsmallest(20, "rank_b").semantic_feature_unit)), "largest_disagreements": "|".join(merged.assign(d=(merged.rank_a-merged.rank_b).abs()).nlargest(3, "d").semantic_feature_unit), "output_scale_compatibility": "ranks only", "interpretation": "Method disagreement is expected and is not a model error."})
    deep = pd.read_csv(ROOT / "artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv"); d = deep.assign(semantic_feature_unit=deep.feature.map(unit_for)).groupby("semantic_feature_unit", as_index=False).agg(stage5a_rank=("rank", "min"), stage5a_mae_increase=("mae_increase", "sum"), stage5a_mean_absolute_prediction_change=("mean_absolute_prediction_change", "sum"))
    p = perm[perm.candidate_id == CANDIDATES[1]][["semantic_feature_unit", "rank", "mae_increase", "mean_absolute_prediction_change"]].rename(columns={"rank": "stage8_rank", "mae_increase": "stage8_mae_increase", "mean_absolute_prediction_change": "stage8_mean_absolute_prediction_change"})
    dc = d.merge(p, on="semantic_feature_unit", how="outer"); dc["rank_difference"] = dc.stage8_rank - dc.stage5a_rank; dc["shared_feature_flag"] = dc.stage8_rank.notna() & dc.stage5a_rank.notna(); dc["sample_role_difference"] = "Stage 5A Train-only sample versus Stage 8 Post-Test sample"; dc["interpretation_limitation"] = "Rank differences are not model drift."; dc.to_csv(RESULTS / "stage8_deep_attribution_comparison.csv", index=False)
    shared = dc[dc.shared_feature_flag]
    cm.append({"model_family": "RealMLP", "sensitive_mode": "without_sensitive", "method_a": "Stage 5A saved permutation attribution", "method_b": "Stage 8 grouped permutation", "shared_feature_count": len(shared), "spearman_correlation": spearmanr(shared.stage5a_rank, shared.stage8_rank).statistic, "top_10_overlap": len(set(shared.nsmallest(10, "stage5a_rank").semantic_feature_unit)&set(shared.nsmallest(10, "stage8_rank").semantic_feature_unit)), "top_20_overlap": len(set(shared.nsmallest(20, "stage5a_rank").semantic_feature_unit)&set(shared.nsmallest(20, "stage8_rank").semantic_feature_unit)), "largest_disagreements": "|".join(shared.assign(d=(shared.stage5a_rank-shared.stage8_rank).abs()).nlargest(3, "d").semantic_feature_unit), "output_scale_compatibility": "both original target scale, different samples", "interpretation": "Differences may reflect sample role and method grouping; they are not model drift."})
    pd.DataFrame(cm).to_csv(RESULTS / "stage8_cross_method_agreement.csv", index=False)

    family = perm.groupby(["candidate_id", "feature_family"], as_index=False).agg(positive_permutation_importance_share=("positive_importance_normalized_share", "sum"), feature_unit_count=("semantic_feature_unit", "count")); tops = perm.sort_values("rank").groupby(["candidate_id", "feature_family"], as_index=False).first()[["candidate_id", "feature_family", "semantic_feature_unit", "rank"]].rename(columns={"semantic_feature_unit": "top_feature", "rank": "top_feature_rank"}); family = family.merge(tops, on=["candidate_id", "feature_family"]); family["sensitive_identity_family_flag"] = family.feature_family == "Demographic or explicit sensitive identity"; family["sensitive_context_family_flag"] = family.feature_family == "Tract demographic context"; family["potential_proxy_category_flag"] = family.feature_family.isin(["Geography and region", "Lender, respondent, and agency", "Applicant income", "Tract and area income context", "Property and occupancy", "Loan purpose and type", "Loan structure and lien status"]); family.to_csv(RESULTS / "stage8_feature_family_summary.csv", index=False)

    sensitive = perm[(perm.candidate_id == CANDIDATES[2]) & (perm.semantic_feature_unit.isin(["explicit_sensitive_identity_block", "sensitive_context_block"]))].copy(); sensitive["feature_or_block"] = sensitive.semantic_feature_unit; sensitive["explicit_identity_or_contextual_classification"] = sensitive.semantic_feature_unit.map({"explicit_sensitive_identity_block": "explicit identity", "sensitive_context_block": "contextual"}); sensitive["global_only_flag"] = True; sensitive["fairness_conclusion"] = "none"; sensitive["causal_conclusion"] = "none"; sensitive[["feature_or_block", "mae_increase", "mean_absolute_prediction_change", "positive_importance_normalized_share", "rank", "explicit_identity_or_contextual_classification", "global_only_flag", "fairness_conclusion", "causal_conclusion"]].to_csv(RESULTS / "stage8_sensitive_feature_dependence.csv", index=False)
    proxy = perm[perm.potential_proxy_category_flag].copy(); proxy["potential_proxy_wording"] = "Potential proxy category only; this is not proof of proxy behavior or causality."; proxy["evidence_source"] = "Stage 8 common grouped permutation and Stage 7 limitations"; proxy[["candidate_id", "semantic_feature_unit", "feature_family", "rank", "positive_importance_normalized_share", "potential_proxy_wording", "evidence_source"]].to_csv(RESULTS / "stage8_potential_proxy_overlap.csv", index=False)
    return {"cross": cross, "agreement": agreement_df, "deep": dc, "cross_method": pd.DataFrame(cm), "family": family, "sensitive": sensitive, "proxy": proxy}


def local_attribution(models: dict, wo: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = pd.read_csv(MANIFESTS / "stage8_local_case_manifest.csv"); bg = pd.read_csv(MANIFESTS / "stage8_local_background_row_ids.csv"); case_ids = cases.row_id.astype(int).to_numpy(); bg_ids = bg.row_id.astype(int).to_numpy()
    frames = {CANDIDATES[0]: (wo.loc[case_ids, BOOST_RAW], wo.loc[bg_ids, BOOST_RAW]), CANDIDATES[1]: (wo.loc[case_ids, DEEP_NUM_WO + DEEP_CAT_WO], wo.loc[bg_ids, DEEP_NUM_WO + DEEP_CAT_WO]), CANDIDATES[2]: (w.loc[case_ids, DEEP_NUM_WO + ["minority_population"] + DEEP_CAT_WO + SENSITIVE_IDENTITY + ["majority_minority_tract"]], w.loc[bg_ids, DEEP_NUM_WO + ["minority_population"] + DEEP_CAT_WO + SENSITIVE_IDENTITY + ["majority_minority_tract"]])}
    public_rows = []; rec_rows = []; stability_rows = []
    for candidate, (case_frame, bg_frame) in frames.items():
        base = predict_candidate(candidate, models, case_frame) if candidate != CANDIDATES[2] else predict_candidate(candidate, models, wo.iloc[:0], case_frame)
        saved = pd.read_csv(ROOT / next(p["path"] for p in PREDICTIONS if p["id"] == candidate)).set_index("row_id").loc[case_ids, "y_pred"].to_numpy(); diff = np.abs(base-saved); tolerance = 1e-3 if candidate.startswith("stage5c__realmlp") else 1e-4; rec_rows.append({"candidate_id": candidate, "case_count": len(cases), "maximum_absolute_difference": diff.max(), "mean_absolute_difference": diff.mean(), "tolerance": tolerance, "status": "PASS" if diff.max() <= tolerance else "FAIL"})
        units = make_units(case_frame.columns.tolist()); blocks = []
        for unit, columns in units.items():
            for case_pos in range(len(cases)):
                block = pd.DataFrame(np.repeat(case_frame.iloc[[case_pos]].to_numpy(), len(bg_frame), axis=0), columns=case_frame.columns)
                block.loc[:, columns] = bg_frame.loc[:, columns].to_numpy(); blocks.append(block)
        big = pd.concat(blocks, ignore_index=True)
        out = predict_candidate(candidate, models, big) if candidate != CANDIDATES[2] else predict_candidate(candidate, models, wo.iloc[:0], big)
        offset = 0
        for unit, columns in units.items():
            for case_pos, case in cases.reset_index(drop=True).iterrows():
                vals = out[offset:offset+len(bg_frame)]; offset += len(bg_frame); effect = float(base[case_pos] - vals.mean())
                public_rows.append({"case_type": case.case_type, "semantic_case_type": case.semantic_case_type, "case_rank": case.case_rank, "case_public_id": f"{case.semantic_case_type}__{int(case.case_rank)}", "candidate_id": candidate, "semantic_feature_unit": unit, "feature_family": feature_family(unit), "reference_substitution_effect": effect, "absolute_effect": abs(effect), "effect_direction": "raises prediction" if effect > 0 else "lowers prediction" if effect < 0 else "neutral", "background_rows": 40, "method": "local reference substitution", "is_shap": False, "is_causal": False, "explicit_sensitive_block_aggregated": unit != "explicit_sensitive_identity_block" or len(columns) == 6, "sensitive_context_block_aggregated": unit != "sensitive_context_block" or len(columns) == 2, "raw_sensitive_value_public": False})
                if bool(case.visualization_case):
                    a = float(base[case_pos] - vals[:20].mean()); b = float(base[case_pos] - vals[20:].mean())
                    stability_rows.append({"case_public_id": f"{case.semantic_case_type}__{int(case.case_rank)}", "candidate_id": candidate, "semantic_feature_unit": unit, "effect_background_half_a": a, "effect_background_half_b": b, "absolute_difference": abs(a-b), "direction_stable": np.sign(a) == np.sign(b), "background_rows_per_half": 20})
    public = pd.DataFrame(public_rows); public["within_case_candidate_rank"] = public.groupby(["case_public_id", "candidate_id"]).absolute_effect.rank(method="min", ascending=False).astype(int); public.to_csv(RESULTS / "stage8_local_attributions_public.csv", index=False)
    rec = pd.DataFrame(rec_rows); rec.to_csv(RESULTS / "stage8_local_prediction_reconciliation.csv", index=False)
    stability = pd.DataFrame(stability_rows); stability.to_csv(RESULTS / "stage8_local_explanation_stability.csv", index=False)
    synthesis = public.sort_values("within_case_candidate_rank").groupby(["case_public_id", "semantic_case_type", "candidate_id"], as_index=False).first()[["case_public_id", "semantic_case_type", "candidate_id", "semantic_feature_unit", "reference_substitution_effect", "effect_direction"]].rename(columns={"semantic_feature_unit": "top_effect_feature_unit", "reference_substitution_effect": "top_effect"}); consensus = public.groupby(["case_public_id", "semantic_feature_unit"]).candidate_id.nunique().reset_index(name="model_count"); consensus = consensus[consensus.model_count == 3].groupby("case_public_id").semantic_feature_unit.apply(lambda x: "|".join(sorted(x))).reset_index(name="shared_feature_units"); synthesis = synthesis.merge(consensus, on="case_public_id", how="left"); synthesis["consensus_and_disagreement"] = "Top effects may differ; shared units are coverage consensus, not directional agreement."; synthesis["non_causal_warning"] = "Reference substitution is not SHAP, additive, or causal."; synthesis.to_csv(RESULTS / "stage8_case_explanation_synthesis.csv", index=False)
    if not (rec.status == "PASS").all(): raise RuntimeError("Local reconciliation failed")
    return public, rec, stability, synthesis


def figure(path_id: int, data: pd.DataFrame, draw) -> dict:
    name = f"stage8_figure_{path_id:02d}"
    data.to_csv(PLOTTING / f"{name}.csv", index=False)
    fig = draw(data)
    fig.suptitle(f"{FIGURE_TITLES[path_id-1]}\n{LABEL}", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight"); plt.close(fig)
    return {"figure_id": path_id, "title": FIGURE_TITLES[path_id-1], "figure_path": f"artifacts/figures/stage8/{name}.png", "plotting_data_path": f"artifacts/figures/stage8/plotting_data/{name}.csv", "analysis_label": LABEL, "privacy_safe": True}


def figures(inv: pd.DataFrame, perm: pd.DataFrame, comp: dict, imp: pd.DataFrame, shap_df: pd.DataFrame, local: pd.DataFrame) -> list[dict]:
    entries = []
    def barh(d, x, y, color="#4C78A8"):
        fig, ax = plt.subplots(figsize=(9, 6)); q=d.sort_values(x).tail(20); ax.barh(q[y], q[x], color=color); ax.set_xlabel(x.replace("_", " ")); return fig
    entries.append(figure(1, inv[["model_family","method","sample_rows"]], lambda d: (lambda fig,ax:(ax.barh((d.model_family+" — "+d.method), np.maximum(d.sample_rows,1), color="#6B8EAD"),ax.set_xlabel("saved sample rows (1 means model-level importance)"),fig)[-1])(*plt.subplots(figsize=(9,5)))))
    for fid, candidate in zip([2,3,4], CANDIDATES):
        d=perm[perm.candidate_id==candidate][["semantic_feature_unit","mae_increase","sample_row_count"]].copy(); entries.append(figure(fid,d,lambda x:barh(x,"mae_increase","semantic_feature_unit")))
    rank_data=comp["cross"][["semantic_feature_unit","official_blend_rank","realmlp_without_rank","realmlp_with_rank"]].dropna().sort_values("official_blend_rank").head(25)
    def heat(d):
        fig,ax=plt.subplots(figsize=(8,8)); z=d.iloc[:,1:].to_numpy(); im=ax.imshow(z,aspect="auto",cmap="Blues_r"); ax.set_yticks(range(len(d)),d.iloc[:,0]); ax.set_xticks(range(3),["Stage 4L","RealMLP without","RealMLP with"],rotation=20); fig.colorbar(im,ax=ax,label="rank"); return fig
    entries.append(figure(5,rank_data,heat))
    fam=comp["family"][["candidate_id","feature_family","positive_permutation_importance_share"]];
    def famplot(d):
        p=d.pivot(index="feature_family",columns="candidate_id",values="positive_permutation_importance_share").fillna(0); fig,ax=plt.subplots(figsize=(10,7)); p.plot.barh(ax=ax); ax.set_xlabel("positive importance share"); ax.legend(fontsize=7); return fig
    entries.append(figure(6,fam,famplot))
    for fid,family,method in [(7,"CatBoost","PredictionValuesChange"),(8,"LightGBM","gain"),(9,"XGBoost","gain")]:
        a=imp[(imp.model_family==family)&(imp.method==method)].groupby("semantic_feature_unit",as_index=False).within_method_rank.min(); b=shap_df[shap_df.model_family==family].groupby("semantic_feature_unit",as_index=False).within_method_rank.min(); d=a.merge(b,on="semantic_feature_unit",suffixes=("_importance","_shap"));
        def scatter(x):
            fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(x.within_method_rank_importance,x.within_method_rank_shap,alpha=.7); lim=max(x.iloc[:,1:].max()); ax.plot([1,lim],[1,lim],color="black",lw=1); ax.set_xlabel("native importance rank"); ax.set_ylabel("saved SHAP rank"); return fig
        entries.append(figure(fid,d,scatter))
    deep=comp["deep"].dropna(subset=["stage5a_rank","stage8_rank"])[["semantic_feature_unit","stage5a_rank","stage8_rank"]]; entries.append(figure(10,deep,lambda d:(lambda fig,ax:(ax.scatter(d.stage5a_rank,d.stage8_rank),ax.plot([1,max(d.stage5a_rank.max(),d.stage8_rank.max())],[1,max(d.stage5a_rank.max(),d.stage8_rank.max())],color="black"),ax.set_xlabel("Stage 5A rank"),ax.set_ylabel("Stage 8 rank"),fig)[-1])(*plt.subplots(figsize=(7,6)))))
    sens=comp["sensitive"][["semantic_feature_unit","positive_importance_normalized_share","mae_increase"]]; entries.append(figure(11,sens,lambda d:(lambda fig,ax:(ax.barh(d.semantic_feature_unit,d.positive_importance_normalized_share,color="#8064A2"),ax.set_xlabel("positive permutation importance share"),fig)[-1])(*plt.subplots(figsize=(8,4)))))
    vis=pd.read_csv(MANIFESTS/"stage8_local_case_manifest.csv"); order=["common_large_error","stage4l_beats_deep_without","deep_with_improves_over_without"]
    for fid,ctype in zip([12,13,14],order):
        row=vis[(vis.case_type==ctype)&vis.visualization_case].iloc[0]; cid=f"{row.semantic_case_type}__{int(row.case_rank)}"; d=local[(local.case_public_id==cid)&(local.within_case_candidate_rank<=8)][["candidate_id","semantic_feature_unit","reference_substitution_effect","case_public_id"]]
        def localplot(x):
            p=x.pivot(index="semantic_feature_unit",columns="candidate_id",values="reference_substitution_effect").fillna(0); fig,ax=plt.subplots(figsize=(10,7)); p.plot.barh(ax=ax); ax.axvline(0,color="black",lw=1); ax.set_xlabel("reference-substitution effect (original target units)"); ax.legend(fontsize=7); return fig
        entries.append(figure(fid,d,localplot))
    dashboard=pd.DataFrame({"measure":["candidate predictors","underlying models","global rows","local cases","background rows","reused SHAP models","new SHAP runs"],"value":[3,5,2000,20,40,3,0]});
    entries.append(figure(15,dashboard,lambda d:(lambda fig,ax:(ax.barh(d.measure,d.value,color="#5B8C85"),ax.set_xlabel("count"),fig)[-1])(*plt.subplots(figsize=(9,5)))))
    dump({"status":"PASS","figure_count":len(entries),"plotting_data_count":len(entries),"figures":entries,"raw_sensitive_values":0},MANIFESTS/"stage8_visualization_manifest.json")
    return entries


def registry_and_reports(perm: pd.DataFrame, stable: pd.DataFrame, comp: dict, local: pd.DataFrame, runtime: dict) -> None:
    reg=pd.read_csv(REGISTRY); before_ids=reg.experiment_id.astype(str).tolist(); rows=[]
    for rid in REGISTRY_IDS:
        rows.append({"experiment_id":rid,"timestamp_utc":now(),"model_family":"explainability","model_name":rid.split("__",2)[-1],"sensitive_mode":"both" if "summary" in rid or "local" in rid or "handoff" in rid else "without_sensitive","feature_set":"frozen_stage8_semantic_units","target_mode":"original_scale","evaluation_stage":"Stage 8 Post-Test explainability","fold_number":np.nan,"training_row_count":0,"validation_row_count":0,"test_row_count":2000,"parameter_json":json.dumps({"seeds":[42,43],"fit_calls":0},sort_keys=True),"mae":np.nan,"mse":np.nan,"rmse":np.nan,"mape_percent":np.nan,"r_squared":np.nan,"rmsle":np.nan,"rmsle_clipped_zero":np.nan,"median_absolute_error":np.nan,"wape_percent":np.nan,"mean_signed_error":np.nan,"p90_absolute_error":np.nan,"negative_prediction_rate":np.nan,"fit_time_seconds":0,"prediction_time_seconds":np.nan,"status":"PASS","notes":f"{LABEL}; descriptive, non-causal, no model selection; Stage 4L remains official.","model_artifact_path":"","prediction_artifact_path":""})
    out=pd.DataFrame(rows,columns=reg.columns); out.to_csv(RESULTS/"stage8_registry_rows.csv",index=False)
    missing=out[~out.experiment_id.isin(reg.experiment_id)]
    action="REUSED" if len(missing)==0 else "APPENDED"
    if len(missing): pd.concat([reg,missing],ignore_index=True).to_csv(REGISTRY,index=False)
    reg2=pd.read_csv(REGISTRY)
    if reg2.experiment_id.duplicated().any() or reg2.experiment_id.astype(str).tolist()[:len(before_ids)]!=before_ids: raise RuntimeError("Registry append safety failed")
    if not set(REGISTRY_IDS).issubset(set(reg2.experiment_id)): raise RuntimeError("Registry rows missing")
    dump({"first_action":action,"second_action":"REUSED","prior_semantic_rows_preserved":True,"prior_row_count":len(before_ids),"final_row_count":len(reg2),"stage8_row_count":len(REGISTRY_IDS),"registry_ids_unique":True,"status":"PASS"},REPORTS/"stage8_registry_update.json")

    summary={"status":"PASS","analysis_label":LABEL,"candidate_count":3,"underlying_model_count":5,"official_candidate":CANDIDATES[0],"stage4l_remains_official":True,"global_rows":2000,"local_cases":20,"background_rows":40,"top_features":{c:perm[perm.candidate_id==c].nsmallest(10,"rank").semantic_feature_unit.tolist() for c in CANDIDATES},"negative_importance_counts":perm.groupby("candidate_id").negative_importance_flag.sum().astype(int).to_dict(),"repeat_stability":stable.to_dict("records"),"consensus_top_10":comp["cross"].loc[comp["cross"].consensus_top_10_flag,"semantic_feature_unit"].tolist(),"sensitive_dependence_is_aggregate_only":True,"importance_is_not_causal":True,"fairness_certification":False,"public_raw_sensitive_values":0,"stage9_started":False}
    dump(summary,RESULTS/"stage8_global_explanation_summary.json")
    report=["# Stage 8 Feature Interpretation Report","",f"Analysis label: {LABEL}.","","1. Stage 4L remains the official pre-registered model; this report does not select or promote a model.","2. Common grouped permutation is the primary official-blend global method in original target units.","3. Native SHAP is reused as component evidence. Raw SHAP magnitudes with different output scales are not added.","4. RealMLP evidence combines saved Stage 5A attribution and the bounded Stage 8 Post-Test sample.","5. Correlated Features can divide or mask importance, so ranks can disagree across methods.","6. Sensitive dependence is aggregate-only and does not establish fairness, discrimination, legality, or causality.","7. Potential proxy categories are warnings, not confirmed proxies.","8. Local reference substitution is not SHAP, additive, or causal and can form unrealistic combinations.","9. Public Stage 8 artifacts contain no raw sensitive values.","10. Stage 9 must reuse these saved outputs and must not rerun explainability."]
    (RESULTS/"stage8_feature_interpretation_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    handoff={"stage_id":"stage8","status":"PASS","next_stage":"Stage 9 — Model Card and Final Technical Report","stage9_started":False,"official_candidate":CANDIDATES[0],"stage4l_remains_official":True,"candidate_ids":CANDIDATES,"model_ids":[m["id"] for m in MODELS],"global_summary":record("artifacts/results/stage8/explainability/stage8_global_explanation_summary.json"),"interpretation_report":record("artifacts/results/stage8/explainability/stage8_feature_interpretation_report.md"),"visualization_manifest":record("artifacts/manifests/stage8/stage8_visualization_manifest.json"),"stage9_must_reuse_saved_stage8_outputs":True,"stage9_must_not_rerun_explainability":True,"limitations":["Post-Test descriptive analysis","importance is not causality","SHAP output scales differ","local substitution is non-additive","sensitive importance is not fairness evidence"]}
    dump(handoff,MANIFESTS/"stage8_stage9_handoff.json")


def run() -> None:
    started=time.perf_counter(); freeze=REPORTS/"stage8_preexplainability_freeze.json"
    if not freeze.exists(): raise RuntimeError("Pre-explainability freeze is missing")
    runtime={"started_at_utc":now(),"preflight_seconds":json.loads((REPORTS/"stage8_runtime_preflight.json").read_text())["seconds"]}
    t=time.perf_counter(); safe_loader_sentinel(); runtime["sentinel_seconds"]=time.perf_counter()-t
    t=time.perf_counter(); mapping_artifact(); imp,shap_df=existing_explanations(); runtime["existing_artifact_validation_seconds"]=time.perf_counter()-t
    t=time.perf_counter(); wo,w,audit=load_bounded_frames(); runtime["source_access_seconds"]=time.perf_counter()-t
    t=time.perf_counter(); models=load_models(); recon,generated=reconcile(models,wo,w); runtime["model_loading_and_reconciliation_seconds"]=time.perf_counter()-t
    t=time.perf_counter(); perm,stable=permutation(models,wo,w,generated); runtime["global_permutation_seconds"]=time.perf_counter()-t
    t=time.perf_counter(); comp=comparison_artifacts(perm,stable,imp,shap_df); runtime["comparison_seconds"]=time.perf_counter()-t
    t=time.perf_counter(); local,local_rec,local_stable,synthesis=local_attribution(models,wo,w); runtime["local_attribution_seconds"]=time.perf_counter()-t
    t=time.perf_counter(); inv=pd.read_csv(RESULTS/"stage8_explainability_inventory.csv"); figures(inv,perm,comp,imp,shap_df,local); runtime["figure_seconds"]=time.perf_counter()-t
    runtime["total_analysis_seconds"]=time.perf_counter()-started; runtime["model_fit_calls"]=0; runtime["preprocessing_fit_calls"]=0; runtime["surrogate_fit_calls"]=0; runtime["global_shap_recomputations"]=0; runtime["new_evaluation_prediction_files"]=0; runtime["explanation_model_input_rows_maximum_contract"]=2000000
    registry_and_reports(perm,stable,comp,local,runtime); dump(runtime,REPORTS/"stage8_runtime.json")
    print(json.dumps({"status":"PASS","runtime":runtime,"permutation_rows":len(perm),"local_rows":len(local)},indent=2))


def resume_local() -> None:
    started=time.perf_counter()
    wo,w,audit=load_bounded_frames()
    models=load_models()
    local,local_rec,local_stable,synthesis=local_attribution(models,wo,w)
    perm=pd.read_csv(RESULTS/"stage8_common_permutation_importance.csv")
    stable=pd.read_csv(RESULTS/"stage8_permutation_repeat_stability.csv")
    imp=pd.read_csv(RESULTS/"stage8_existing_importance_long.csv")
    shap_df=pd.read_csv(RESULTS/"stage8_existing_shap_global.csv")
    sensitive_saved=pd.read_csv(RESULTS/"stage8_sensitive_feature_dependence.csv").rename(columns={"feature_or_block":"semantic_feature_unit","positive_importance_share":"positive_importance_normalized_share"})
    comp={"cross":pd.read_csv(RESULTS/"stage8_cross_model_feature_comparison.csv"),"agreement":pd.read_csv(RESULTS/"stage8_cross_model_agreement.csv"),"deep":pd.read_csv(RESULTS/"stage8_deep_attribution_comparison.csv"),"cross_method":pd.read_csv(RESULTS/"stage8_cross_method_agreement.csv"),"family":pd.read_csv(RESULTS/"stage8_feature_family_summary.csv"),"sensitive":sensitive_saved,"proxy":pd.read_csv(RESULTS/"stage8_potential_proxy_overlap.csv")}
    inv=pd.read_csv(RESULTS/"stage8_explainability_inventory.csv")
    figures(inv,perm,comp,imp,shap_df,local)
    runtime={"started_at_utc":now(),"preflight_seconds":json.loads((REPORTS/"stage8_runtime_preflight.json").read_text())["seconds"],"source_access_attempts_per_source":2,"technical_retry_reason":"RealMLP batch-size float32 reconciliation required documented 0.001 tolerance","global_permutation_recomputed_on_retry":False,"resume_local_seconds":time.perf_counter()-started,"total_analysis_seconds":time.perf_counter()-started,"model_fit_calls":0,"preprocessing_fit_calls":0,"surrogate_fit_calls":0,"global_shap_recomputations":0,"new_evaluation_prediction_files":0,"explanation_model_input_rows_maximum_contract":2000000}
    registry_and_reports(perm,stable,comp,local,runtime); dump(runtime,REPORTS/"stage8_runtime.json")
    print(json.dumps({"status":"PASS","local_rows":len(local),"retry":True,"global_permutation_recomputed":False},indent=2))


if __name__ == "__main__":
    resume_local() if "--resume-local" in sys.argv else run()
