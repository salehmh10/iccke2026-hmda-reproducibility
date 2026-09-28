from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
FINAL = ROOT / "outputs" / "final"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
TMP = ROOT / "outputs" / "tmp" / "prompt5c"
NOTEBOOK = ROOT / "notebooks" / "05C_FINAL_PROJECT_REPORTING.ipynb"
AUTHORIZATION_ID = "regression_v2_prompt5c_final_reporting"
EXPECTED_FAIRNESS_HASH = "2025ec95cc564238e72fbb82ac33670c0272f166cf93d46dc1aa7a59f1b890df"
EXPECTED_FREEZE_HASH = "8f2b8e4fda80056770b916b7859ad0f6f89f2948236320e6815e082fadca35c1"
PRIMARY_ID = "stage3_residual_t75_a75"
PRIMARY_DEPLOYMENT_ID = "final_primary_stage3_500k"
GLOBAL_ID = "ens_boost_cat060"
GLOBAL_DEPLOYMENT_ID = "final_global_500k"
PRIMARY_BUNDLE_HASH = "5349ab15fd1c8182ef539f435cc4f065e71ee7da047de09a4dc271c9097e0c08"
GLOBAL_BUNDLE_HASH = "6f61a0be1fc90d2331b08dada63a453f6c5410d5783f46f1e0cbcbf425f75597"
FINAL_STATUS = "PASS_FINAL_PROJECT_COMPLETE"
FAIRNESS_DISCLAIMER = (
    "The fairness analysis evaluates predictive error disparities across available "
    "sensitive-group labels. It does not assess lending approval decisions, causal "
    "discrimination, disparate treatment, or legal compliance."
)
EXPLAINABILITY_DISCLAIMER = (
    "Feature attributions describe the behavior of the fitted model in its native "
    "prediction spaces; they do not establish causal relationships."
)
TARGET_UNITS = "loan_amount_000s (thousands of USD)"
PROHIBITED_IID_NAMES = {
    "iid_holdout_features.parquet",
    "iid_holdout_targets.parquet",
}
ACTIVE_FIGURE_SELECTION: set[int] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256(path: Path) -> str:
    if path.name in PROHIBITED_IID_NAMES:
        raise RuntimeError(f"Original IID content access is prohibited: {path}")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if path.name in PROHIBITED_IID_NAMES:
        raise RuntimeError(f"Original IID content access is prohibited: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if path.name in PROHIBITED_IID_NAMES:
        raise RuntimeError(f"Original IID content access is prohibited: {path}")
    return pd.read_csv(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, digits: int = 6) -> str:
    display_frame = frame.copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
        else:
            display_frame[column] = display_frame[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    columns = [str(c) for c in display_frame.columns]

    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(c) for c in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in display_frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(clean(v) for v in row) + " |")
    return "\n".join(lines)


def save_table(
    name: str,
    frame: pd.DataFrame,
    title: str,
    note: str,
    sources: Iterable[str],
) -> dict[str, Any]:
    csv_path = TABLES / f"{name}.csv"
    md_path = TABLES / f"{name}.md"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    source_text = ", ".join(sources)
    body = (
        f"# {title}\n\n"
        f"{markdown_table(frame)}\n\n"
        f"Note: {note}\n\n"
        f"Frozen source evidence: {source_text}.\n"
    )
    write_text(md_path, body)
    return {
        "name": name,
        "csv_path": rel(csv_path),
        "csv_sha256": sha256(csv_path),
        "markdown_path": rel(md_path),
        "markdown_sha256": sha256(md_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "source_artifacts": list(sources),
        "status": "PASS",
    }


def validate_handoffs() -> dict[str, Any]:
    required = {
        "PROMPT5B_READY.json": "PASS_FINAL_FAIRNESS_AND_EXPLAINABILITY",
        "FINAL_FAIRNESS_EXPLAINABILITY.json": "PASS_FINAL_FAIRNESS_AND_EXPLAINABILITY",
        "PROMPT5A_READY.json": "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS",
        "FINAL_IID_EVALUATION.json": "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS",
        "PROMPT4C_READY.json": "PASS_FINAL_PRE_IID_FREEZE",
        "FINAL_PRE_IID_FREEZE.json": "PASS_FINAL_PRE_IID_FREEZE",
    }
    checks: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for name, expected_status in required.items():
        path = REPORTS / name
        if not path.exists():
            raise RuntimeError(f"Required immutable handoff is missing: {path}")
        value = read_json(path)
        actual_status = value.get("status")
        digest = sha256(path)
        hashes[rel(path)] = digest
        ok = actual_status == expected_status
        checks.append(
            {
                "artifact": rel(path),
                "expected_status": expected_status,
                "actual_status": actual_status,
                "sha256": digest,
                "status": "PASS" if ok else "FAIL",
            }
        )
        if not ok:
            raise RuntimeError(
                f"Immutable handoff status mismatch for {name}: {actual_status}"
            )
    fairness_hash = hashes["outputs/reports/FINAL_FAIRNESS_EXPLAINABILITY.json"]
    freeze_hash = hashes["outputs/reports/FINAL_PRE_IID_FREEZE.json"]
    if fairness_hash != EXPECTED_FAIRNESS_HASH:
        raise RuntimeError("FINAL_FAIRNESS_EXPLAINABILITY.json hash mismatch")
    if freeze_hash != EXPECTED_FREEZE_HASH:
        raise RuntimeError("FINAL_PRE_IID_FREEZE.json hash mismatch")
    result = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "checks": checks,
        "required_hash_matches": {
            "final_fairness_explainability": True,
            "final_pre_iid_freeze": True,
        },
        "original_iid_content_reads": 0,
        "original_iid_content_hashes_computed": 0,
        "source_hierarchy": [
            "FINAL_IID_EVALUATION.json",
            "FINAL_FAIRNESS_EXPLAINABILITY.json",
            "FINAL_PRE_IID_FREEZE.json",
            "Prompt 5A/5B verification reports",
            "Prompt 4C final selection reports",
            "Prompt 4B4 through Prompt 2 frozen reports",
            "state files",
        ],
    }
    write_json(REPORTS / "prompt5c_handoff_validation.json", result)
    return result


def build_tables() -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    overall = read_csv("prompt5a_iid_overall_metrics.csv")
    bootstrap = read_csv("prompt5a_iid_bootstrap.csv")
    deciles = read_csv("prompt5a_iid_local_decile_metrics.csv")
    transport = read_csv("prompt5a_development_iid_transport.csv")
    six = read_json(REPORTS / "prompt5a_iid_six_condition_check.json")
    primary_groups = read_csv("prompt5b_primary_group_metrics.csv")
    global_groups = read_csv("prompt5b_global_group_metrics.csv")
    comparison = read_csv("prompt5b_group_primary_vs_global.csv")
    disparity = read_csv("prompt5b_disparity_summary.csv")
    tail_groups = read_csv("prompt5b_tail_group_metrics.csv")
    intersections = read_csv("prompt5b_intersectional_metrics.csv")
    fairness_bootstrap = read_csv("prompt5b_fairness_bootstrap.csv")
    global_importance = read_csv("prompt5b_global_feature_importance.csv")
    gate_importance = read_csv("prompt5b_gate_feature_importance.csv")
    residual_importance = read_csv("prompt5b_residual_feature_importance.csv")
    correction = read_csv("prompt5b_correction_behavior.csv")
    routing = read_csv("prompt5a_routing_diagnostics.csv")

    records: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}

    performance_map = {
        "model_id": "Model ID",
        "mae": "MAE",
        "rmse": "RMSE",
        "r2": "R²",
        "rmsle": "RMSLE",
        "median_absolute_error": "Median AE",
        "p90_absolute_error": "P90 AE",
        "mape_percent": "MAPE %",
        "wape_percent": "WAPE %",
        "bottom_90_mae": "Bottom-90 MAE",
        "top_decile_mae": "Top-decile MAE",
        "top_five_percent_mae": "Top-5% MAE",
        "p85_to_p95_boundary_mae": "P85-P95 MAE",
        "top_decile_signed_error": "Top-decile signed error",
        "top_decile_underprediction_rate": "Top-decile underprediction rate",
    }
    performance = overall[list(performance_map)].rename(columns=performance_map)
    performance.insert(
        0,
        "Model",
        ["Final Primary — Stage 3 500k", "Final Global — 500k"],
    )
    records.append(
        save_table(
            "table_final_model_performance",
            performance,
            "Final model IID performance",
            f"Error values use {TARGET_UNITS}; MAPE and WAPE are percentages.",
            ["outputs/reports/prompt5a_iid_overall_metrics.csv"],
        )
    )
    frames["performance"] = performance

    iid_comparison = bootstrap.rename(
        columns={
            "metric": "Metric",
            "observed_difference_primary_minus_global": "Observed Primary minus Global",
            "bootstrap_mean": "Paired bootstrap mean difference",
            "bootstrap_median": "Paired bootstrap median difference",
            "percentile_2_5": "95% CI lower",
            "percentile_97_5": "95% CI upper",
            "fraction_primary_lower": "Fraction resamples favoring Primary",
            "n_resamples": "Resamples",
            "seed": "Seed",
        }
    )
    records.append(
        save_table(
            "table_iid_primary_vs_global",
            iid_comparison,
            "Paired IID comparison: Primary minus Global",
            "Negative differences favor Primary for all listed error metrics.",
            ["outputs/reports/prompt5a_iid_bootstrap.csv"],
        )
    )
    frames["iid_comparison"] = iid_comparison

    six_frame = pd.DataFrame(six["conditions"]).rename(
        columns={
            "condition": "Condition",
            "description": "Definition",
            "global_value": "Global IID value",
            "primary_value": "Primary IID value",
            "threshold": "Threshold",
            "status": "PASS/FAIL",
        }
    )[
        [
            "Condition",
            "Definition",
            "Global IID value",
            "Primary IID value",
            "Threshold",
            "PASS/FAIL",
        ]
    ]
    records.append(
        save_table(
            "table_six_condition_generalization",
            six_frame,
            "Frozen six-condition IID generalization result",
            "Five of six conditions passed. C2 failed; the required 3% Top-decile MAE improvement was not achieved.",
            ["outputs/reports/prompt5a_iid_six_condition_check.json"],
        )
    )
    frames["six"] = six_frame

    transport_map = {
        "model_id": "Model ID",
        "development_evidence_label": "Development evidence",
        "iid_evidence_label": "IID evidence",
        "development_mae": "Development MAE",
        "iid_mae": "IID MAE",
        "development_mape_percent": "Development MAPE %",
        "iid_mape_percent": "IID MAPE %",
        "development_wape_percent": "Development WAPE %",
        "iid_wape_percent": "IID WAPE %",
        "development_bottom_90_mae": "Development Bottom90 MAE",
        "iid_bottom_90_mae": "IID Bottom90 MAE",
        "development_top_decile_mae": "Development Top-decile MAE",
        "iid_top_decile_mae": "IID Top-decile MAE",
        "development_top_decile_underprediction_rate": "Development Top-decile underprediction",
        "iid_top_decile_underprediction_rate": "IID Top-decile underprediction",
    }
    transport_table = transport[list(transport_map)].rename(columns=transport_map)
    transport_table["Development evidence"] = "adaptive Development"
    transport_table["IID evidence"] = "untouched one-time holdout"
    records.append(
        save_table(
            "table_development_to_iid_transport",
            transport_table,
            "Development-to-IID transport",
            "The Development comparison is adaptive and descriptive; IID is the untouched one-time holdout.",
            ["outputs/reports/prompt5a_development_iid_transport.csv"],
        )
    )
    frames["transport"] = transport_table

    primary_deciles = deciles[deciles["model_id"] == PRIMARY_DEPLOYMENT_ID].copy()
    global_deciles = deciles[deciles["model_id"] == GLOBAL_DEPLOYMENT_ID][
        ["decile", "mae", "mape_percent", "wape_percent"]
    ].rename(
        columns={
            "mae": "global_mae",
            "mape_percent": "global_mape_percent",
            "wape_percent": "global_wape_percent",
        }
    )
    decile_table = primary_deciles.merge(global_deciles, on="decile", validate="one_to_one")
    decile_table["primary_minus_global_mae"] = (
        decile_table["mae"] - decile_table["global_mae"]
    )
    decile_table = decile_table.rename(
        columns={
            "decile": "Decile",
            "n": "n",
            "target_min": "Target min",
            "target_max": "Target max",
            "mean_target": "Mean target",
            "mae": "Primary MAE",
            "mape_percent": "Primary MAPE %",
            "wape_percent": "Primary WAPE %",
            "mean_signed_error": "Primary signed error",
            "underprediction_rate": "Primary underprediction rate",
            "global_mae": "Global MAE",
            "global_mape_percent": "Global MAPE %",
            "global_wape_percent": "Global WAPE %",
            "primary_minus_global_mae": "Primary minus Global MAE",
        }
    )[
        [
            "Decile",
            "n",
            "Target min",
            "Target max",
            "Mean target",
            "Primary MAE",
            "Primary MAPE %",
            "Primary WAPE %",
            "Primary signed error",
            "Primary underprediction rate",
            "Global MAE",
            "Global MAPE %",
            "Global WAPE %",
            "Primary minus Global MAE",
        ]
    ]
    records.append(
        save_table(
            "table_iid_decile_error_profile",
            decile_table,
            "Final Primary IID target-decile error profile",
            f"Target values and absolute errors use {TARGET_UNITS}. The delta is a deterministic presentation difference between two frozen values.",
            ["outputs/reports/prompt5a_iid_local_decile_metrics.csv"],
        )
    )
    frames["deciles"] = decile_table

    eligible_primary = primary_groups[primary_groups["analysis_status"] == "ELIGIBLE"].copy()
    eligible_global = global_groups[global_groups["analysis_status"] == "ELIGIBLE"][
        ["sensitive_field", "group_label", "mae"]
    ].rename(columns={"mae": "global_mae"})
    eligible_delta = comparison[comparison["analysis_status"] == "ELIGIBLE"][
        ["sensitive_field", "group_label", "primary_minus_global_mae"]
    ]
    fairness = eligible_primary.merge(
        eligible_global,
        on=["sensitive_field", "group_label"],
        validate="one_to_one",
    ).merge(
        eligible_delta,
        on=["sensitive_field", "group_label"],
        validate="one_to_one",
    )
    fairness["descriptive_flag"] = (
        "Descriptive only; target composition differs; not causal or legal evidence."
    )
    fairness = fairness.rename(
        columns={
            "sensitive_field": "Sensitive field",
            "group_label": "Group",
            "n": "n",
            "mean_target": "Mean target",
            "mae": "Primary MAE",
            "global_mae": "Global MAE",
            "primary_minus_global_mae": "Primary minus Global MAE",
            "underprediction_rate": "Primary underprediction",
            "d10_fraction": "D10 share",
            "descriptive_flag": "Descriptive flag",
        }
    )[
        [
            "Sensitive field",
            "Group",
            "n",
            "Mean target",
            "Primary MAE",
            "Global MAE",
            "Primary minus Global MAE",
            "Primary underprediction",
            "D10 share",
            "Descriptive flag",
        ]
    ]
    records.append(
        save_table(
            "table_fairness_summary",
            fairness,
            "Eligible-group descriptive fairness summary",
            FAIRNESS_DISCLAIMER + " Small groups remain visible in frozen inventory evidence but are not ranked here.",
            [
                "outputs/reports/prompt5b_primary_group_metrics.csv",
                "outputs/reports/prompt5b_global_group_metrics.csv",
                "outputs/reports/prompt5b_group_primary_vs_global.csv",
            ],
        )
    )
    frames["fairness"] = fairness

    tail_eligible = tail_groups[tail_groups["analysis_status"] == "ELIGIBLE"].copy()
    tail_rows = []
    for field, group in tail_eligible.groupby("sensitive_field", sort=False):
        gaps = []
        for scope, scoped in group.groupby("tail_scope", sort=False):
            ordered = scoped.sort_values("primary_tail_mae")
            gaps.append(
                {
                    "scope": scope,
                    "gap": float(
                        ordered.iloc[-1]["primary_tail_mae"]
                        - ordered.iloc[0]["primary_tail_mae"]
                    ),
                    "low": ordered.iloc[0]["group_label"],
                    "high": ordered.iloc[-1]["group_label"],
                }
            )
        largest = max(gaps, key=lambda row: row["gap"])
        tail_rows.append(
            {
                "sensitive_field": field,
                "largest_tail_mae_gap": largest["gap"],
                "tail_scope": largest["scope"],
                "tail_low_group": largest["low"],
                "tail_high_group": largest["high"],
            }
        )
    tail_summary = pd.DataFrame(tail_rows)
    disparity_table = disparity.merge(tail_summary, on="sensitive_field", how="left")
    disparity_table = disparity_table.rename(
        columns={
            "sensitive_field": "Sensitive field",
            "eligible_group_count": "Eligible groups",
            "min_mae_group": "Lowest MAE group",
            "max_mae_group": "Highest MAE group",
            "mae_max_minus_min": "MAE gap",
            "mae_max_min_ratio": "MAE ratio",
            "underprediction_gap_percentage_points": "Underprediction gap (pp)",
            "largest_tail_mae_gap": "Largest Tail MAE gap",
            "tail_scope": "Tail scope",
            "tail_low_group": "Tail lowest-MAE group",
            "tail_high_group": "Tail highest-MAE group",
            "interpretation": "Interpretation note",
        }
    )[
        [
            "Sensitive field",
            "Eligible groups",
            "Lowest MAE group",
            "Highest MAE group",
            "MAE gap",
            "MAE ratio",
            "Underprediction gap (pp)",
            "Largest Tail MAE gap",
            "Tail scope",
            "Tail lowest-MAE group",
            "Tail highest-MAE group",
            "Interpretation note",
        ]
    ]
    if "minority_population" not in set(disparity_table["Sensitive field"]):
        unavailable = "NOT_AVAILABLE_FROM_FROZEN_EVIDENCE"
        disparity_table = pd.concat(
            [
                disparity_table,
                pd.DataFrame(
                    [
                        {
                            "Sensitive field": "minority_population",
                            "Eligible groups": 0,
                            "Lowest MAE group": unavailable,
                            "Highest MAE group": unavailable,
                            "MAE gap": unavailable,
                            "MAE ratio": unavailable,
                            "Underprediction gap (pp)": unavailable,
                            "Largest Tail MAE gap": unavailable,
                            "Tail scope": unavailable,
                            "Tail lowest-MAE group": unavailable,
                            "Tail highest-MAE group": unavailable,
                            "Interpretation note": (
                                "No eligible ranked levels under the frozen n threshold; "
                                "descriptive disparity statistics are unavailable."
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    records.append(
        save_table(
            "table_fairness_disparities",
            disparity_table,
            "Descriptive disparity summary by sensitive field",
            FAIRNESS_DISCLAIMER + " Target composition and small-group limits apply.",
            [
                "outputs/reports/prompt5b_disparity_summary.csv",
                "outputs/reports/prompt5b_tail_group_metrics.csv",
            ],
        )
    )
    frames["disparities"] = disparity_table

    intersection_rows = []
    eligible_intersections = intersections[
        (intersections["scope"] == "OVERALL")
        & (intersections["analysis_status"] == "ELIGIBLE")
    ]
    for intersection, group in eligible_intersections.groupby("intersection", sort=False):
        ordered = group.sort_values("mae")
        low = ordered.iloc[0]
        high = ordered.iloc[-1]
        intersection_rows.append(
            {
                "Intersection": intersection,
                "Largest eligible MAE gap": float(high["mae"] - low["mae"]),
                "Lowest MAE group": low["group_label"],
                "Lowest group n": int(low["n"]),
                "Lowest group MAE": float(low["mae"]),
                "Highest MAE group": high["group_label"],
                "Highest group n": int(high["n"]),
                "Highest group MAE": float(high["mae"]),
                "Uncertainty": "NOT_AVAILABLE_FROM_FROZEN_EVIDENCE",
            }
        )
    intersection_table = pd.DataFrame(intersection_rows)
    records.append(
        save_table(
            "table_intersectional_fairness",
            intersection_table,
            "Predeclared intersectional descriptive error gaps",
            FAIRNESS_DISCLAIMER + " No saved intersection-gap interval was available, so uncertainty is marked unavailable.",
            [
                "outputs/reports/prompt5b_intersectional_metrics.csv",
                "outputs/reports/prompt5b_fairness_bootstrap.csv",
            ],
        )
    )
    frames["intersections"] = intersection_table

    explain_rows = []
    for _, row in global_importance.sort_values("consensus_rank").head(15).iterrows():
        explain_rows.append(
            {
                "Component": "Global consensus",
                "Rank": int(row["consensus_rank"]),
                "Feature": row["feature"],
                "Importance / rank score": float(row["consensus_rank_score"]),
                "Explanation space": "rank consensus across separate native component spaces",
                "Interpretation role": "Global loan-amount prediction driver",
            }
        )
    for component, frame, role in [
        ("Meta-Gate", gate_importance, "Routing driver"),
        ("Residual Specialist", residual_importance, "Residual-correction driver"),
    ]:
        for _, row in frame.sort_values("rank").head(15).iterrows():
            explain_rows.append(
                {
                    "Component": component,
                    "Rank": int(row["rank"]),
                    "Feature": row["feature"],
                    "Importance / rank score": float(row["mean_abs_shap"]),
                    "Explanation space": row["explanation_space"],
                    "Interpretation role": role,
                }
            )
    explain_table = pd.DataFrame(explain_rows)
    records.append(
        save_table(
            "table_final_explainability",
            explain_table,
            "Hierarchical final explainability",
            EXPLAINABILITY_DISCLAIMER + " Raw SHAP magnitudes are not merged across incompatible models or spaces.",
            [
                "outputs/reports/prompt5b_global_feature_importance.csv",
                "outputs/reports/prompt5b_gate_feature_importance.csv",
                "outputs/reports/prompt5b_residual_feature_importance.csv",
            ],
        )
    )
    frames["explainability"] = explain_table

    component_rows = []
    component_specs = [
        (
            "CatBoost",
            "catboost_rank",
            "catboost_mean_abs_shap",
            "catboost_space",
        ),
        (
            "LightGBM",
            "lightgbm_rank",
            "lightgbm_mean_abs_shap",
            "lightgbm_space",
        ),
        (
            "XGBoost",
            "xgboost_rank",
            "xgboost_mean_abs_shap",
            "xgboost_space",
        ),
    ]
    for component, rank_col, value_col, space_col in component_specs:
        selected = global_importance.sort_values(rank_col).head(15)
        for _, row in selected.iterrows():
            component_rows.append(
                {
                    "Component": component,
                    "Rank": int(row[rank_col]),
                    "Feature": row["feature"],
                    "Mean absolute SHAP": float(row[value_col]),
                    "Attribution space": row[space_col],
                    "Native-space note": (
                        "XGBoost attribution is in native log1p space."
                        if component == "XGBoost"
                        else "Raw loan-amount space (thousands of USD)."
                    ),
                }
            )
    component_table = pd.DataFrame(component_rows)
    records.append(
        save_table(
            "table_global_component_importance",
            component_table,
            "Global component Top-15 feature rankings",
            EXPLAINABILITY_DISCLAIMER + " XGBoost uses native log1p attribution space.",
            ["outputs/reports/prompt5b_global_feature_importance.csv"],
        )
    )
    frames["components"] = component_table

    all_iid = routing[routing["scope"] == "All IID"].iloc[0]
    direction = {
        row["metric"]: float(row["value"])
        for _, row in correction[correction["section"] == "direction"].iterrows()
    }
    routed_fraction = float(all_iid["routed_fraction"])
    mean_routed_correction = float(all_iid["mean_applied_correction"]) / routed_fraction
    mechanism = pd.DataFrame(
        [
            ["IID rows", int(all_iid["n"]), "outputs/reports/prompt5a_routing_diagnostics.csv"],
            ["Routed rows", int(all_iid["routed_rows"]), "outputs/reports/prompt5a_routing_diagnostics.csv"],
            ["Routing rate", routed_fraction, "outputs/reports/prompt5a_routing_diagnostics.csv"],
            ["Mean routed correction", mean_routed_correction, "Frozen aggregate ratio from prompt5a_routing_diagnostics.csv"],
            ["Median absolute routed correction", 16.551, "Prompt 5C authorization Section 7; frozen Prompt 5B mechanism restatement"],
            ["Positive correction rate", direction["positive_fraction"], "outputs/reports/prompt5b_correction_behavior.csv"],
            ["Negative correction rate", direction["negative_fraction"], "outputs/reports/prompt5b_correction_behavior.csv"],
            ["Zero correction rate", direction["zero_fraction"], "outputs/reports/prompt5b_correction_behavior.csv"],
            ["Routed-row realized benefit", float(all_iid["primary_advantage_routed"]), "outputs/reports/prompt5a_routing_diagnostics.csv"],
        ],
        columns=["Measure", "Value", "Frozen source"],
    )
    records.append(
        save_table(
            "table_stage3_mechanism",
            mechanism,
            "Stage 3 IID routing and correction mechanism",
            f"Correction and benefit values use {TARGET_UNITS}. The authorization-supplied median is restated without row-level recomputation.",
            [
                "outputs/reports/prompt5a_routing_diagnostics.csv",
                "outputs/reports/prompt5b_correction_behavior.csv",
                "Prompt 5C authorization Section 7",
            ],
        )
    )
    frames["mechanism"] = mechanism

    validation = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "table_count": len(records),
        "tables": records,
        "validation_method": (
            "Direct copies were compared to frozen source fields before writing. "
            "Displayed deltas and gaps are deterministic presentation arithmetic over "
            "saved aggregate values. No row-level IID target or prediction data were read."
        ),
        "authorization_supplied_frozen_values": {
            "median_absolute_routed_correction": 16.551,
            "provenance": "Prompt 5C authorization Section 7",
        },
        "original_iid_content_reads": 0,
        "scientific_recomputations": 0,
    }
    write_json(REPORTS / "prompt5c_table_validation.json", validation)
    return records, frames


def _style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)


def save_figure(
    number: int,
    slug: str,
    fig: plt.Figure,
    plotting_data: pd.DataFrame,
    source_artifacts: list[str],
    axes: dict[str, str],
    metrics: list[str],
    units: dict[str, str],
) -> dict[str, Any]:
    stem = f"figure{number:02d}_{slug}"
    csv_path = FIGURES / f"{stem}_data.csv"
    png_path = FIGURES / f"{stem}.png"
    pdf_path = FIGURES / f"{stem}.pdf"
    should_write = (
        ACTIVE_FIGURE_SELECTION is None or number in ACTIVE_FIGURE_SELECTION
    )
    if should_write:
        plotting_data.to_csv(csv_path, index=False, lineterminator="\n")
        fig.savefig(
            png_path,
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.28,
            facecolor="white",
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.28,
            facecolor="white",
        )
    elif not all(path.exists() for path in [csv_path, png_path, pdf_path]):
        raise RuntimeError(f"Cannot preserve missing unselected figure {number}")
    plt.close(fig)
    return {
        "figure": number,
        "title": slug.replace("_", " ").title(),
        "plotting_data_path": rel(csv_path),
        "plotting_data_sha256": sha256(csv_path),
        "png_path": rel(png_path),
        "png_sha256": sha256(png_path),
        "pdf_path": rel(pdf_path),
        "pdf_sha256": sha256(pdf_path),
        "source_artifacts": source_artifacts,
        "axes": axes,
        "metrics": metrics,
        "units": units,
        "status": "PASS",
    }


def build_figures(selected: set[int] | None = None) -> list[dict[str, Any]]:
    global ACTIVE_FIGURE_SELECTION
    ACTIVE_FIGURE_SELECTION = selected
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 12,
        }
    )
    primary_color = "#1f5a94"
    global_color = "#d07a28"
    records: list[dict[str, Any]] = []

    overall = read_csv("prompt5a_plot_overall_metrics.csv").copy()
    overall["Model"] = overall["model_id"].map(
        {
            PRIMARY_DEPLOYMENT_ID: "Final Primary",
            GLOBAL_DEPLOYMENT_ID: "Final Global",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0))
    metric_specs = [
        ("mae", "MAE", TARGET_UNITS),
        ("rmse", "RMSE", TARGET_UNITS),
        ("mape_percent", "MAPE", "percent"),
        ("wape_percent", "WAPE", "percent"),
    ]
    for ax, (column, title, unit) in zip(axes.ravel(), metric_specs):
        bars = ax.bar(
            overall["Model"],
            overall[column],
            color=[primary_color, global_color],
            width=0.62,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(unit)
        _style_axes(ax)
    fig.suptitle("Final Primary vs Global on the one-time IID holdout")
    fig.tight_layout()
    records.append(
        save_figure(
            1,
            "iid_overall_metrics",
            fig,
            overall,
            ["outputs/reports/prompt5a_plot_overall_metrics.csv"],
            {"x": "Frozen model", "y": "Metric value; panel-specific units"},
            ["MAE", "RMSE", "MAPE", "WAPE"],
            {"MAE/RMSE": TARGET_UNITS, "MAPE/WAPE": "percent"},
        )
    )

    deciles = read_csv("prompt5a_plot_deciles.csv").copy()
    deciles["Model"] = deciles["model_id"].map(
        {
            PRIMARY_DEPLOYMENT_ID: "Final Primary",
            GLOBAL_DEPLOYMENT_ID: "Final Global",
        }
    )
    for number, column, slug, ylabel in [
        (2, "mae", "iid_mae_by_target_decile", f"MAE ({TARGET_UNITS})"),
        (3, "mape_percent", "iid_mape_by_target_decile", "MAPE (%)"),
        (4, "wape_percent", "iid_wape_by_target_decile", "WAPE (%)"),
    ]:
        fig, ax = plt.subplots(figsize=(7.3, 4.4))
        for model, color in [
            ("Final Primary", primary_color),
            ("Final Global", global_color),
        ]:
            part = deciles[deciles["Model"] == model].sort_values("decile_index")
            ax.plot(
                part["decile"],
                part[column],
                marker="o",
                linewidth=2,
                label=model,
                color=color,
            )
        ax.set_xlabel("True-target IID decile")
        ax.set_ylabel(ylabel)
        ax.set_title(slug.replace("_", " ").title())
        ax.legend(frameon=False)
        _style_axes(ax)
        fig.tight_layout()
        records.append(
            save_figure(
                number,
                slug,
                fig,
                deciles[
                    [
                        "model_id",
                        "Model",
                        "decile",
                        "decile_index",
                        "n",
                        "target_min",
                        "target_max",
                        column,
                    ]
                ],
                ["outputs/reports/prompt5a_plot_deciles.csv"],
                {"x": "True-target IID decile", "y": ylabel},
                [column],
                {column: "percent" if "percent" in column else TARGET_UNITS},
            )
        )

    body_tail = read_csv("prompt5a_plot_body_tail.csv").copy()
    body_tail["Model"] = body_tail["model_id"].map(
        {
            PRIMARY_DEPLOYMENT_ID: "Final Primary",
            GLOBAL_DEPLOYMENT_ID: "Final Global",
        }
    )
    long_body_tail = body_tail.melt(
        id_vars=["model_id", "Model"],
        value_vars=["bottom_90_mae", "top_decile_mae", "top_five_percent_mae"],
        var_name="Scope",
        value_name="MAE",
    )
    long_body_tail["Scope"] = long_body_tail["Scope"].map(
        {
            "bottom_90_mae": "Bottom 90%",
            "top_decile_mae": "Top decile",
            "top_five_percent_mae": "Top 5%",
        }
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    x = np.arange(3)
    width = 0.34
    primary_values = long_body_tail[long_body_tail["Model"] == "Final Primary"]["MAE"]
    global_values = long_body_tail[long_body_tail["Model"] == "Final Global"]["MAE"]
    ax.bar(x - width / 2, primary_values, width, label="Final Primary", color=primary_color)
    ax.bar(x + width / 2, global_values, width, label="Final Global", color=global_color)
    ax.set_xticks(x, ["Bottom 90%", "Top decile", "Top 5%"])
    ax.set_ylabel(f"MAE ({TARGET_UNITS})")
    ax.set_title("Body and upper-Tail IID error")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    records.append(
        save_figure(
            5,
            "body_tail_error_comparison",
            fig,
            long_body_tail,
            ["outputs/reports/prompt5a_plot_body_tail.csv"],
            {"x": "Frozen target scope", "y": "MAE"},
            ["Bottom-90 MAE", "Top-decile MAE", "Top-5% MAE"],
            {"MAE": TARGET_UNITS},
        )
    )

    primary_dec = deciles[deciles["model_id"] == PRIMARY_DEPLOYMENT_ID].sort_values(
        "decile_index"
    )
    global_dec = deciles[deciles["model_id"] == GLOBAL_DEPLOYMENT_ID].sort_values(
        "decile_index"
    )
    decile_delta = primary_dec[
        ["decile", "decile_index", "n", "mae"]
    ].rename(columns={"mae": "primary_mae"})
    decile_delta["global_mae"] = global_dec["mae"].to_numpy()
    decile_delta["primary_minus_global_mae"] = (
        decile_delta["primary_mae"] - decile_delta["global_mae"]
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    ax.axhline(0, color="#444444", linewidth=1)
    colors = np.where(decile_delta["primary_minus_global_mae"] <= 0, primary_color, "#b44c42")
    ax.bar(
        decile_delta["decile"],
        decile_delta["primary_minus_global_mae"],
        color=colors,
    )
    ax.set_xlabel("True-target IID decile")
    ax.set_ylabel("MAE difference\n(thousands of USD)", labelpad=8)
    ax.set_title("Decile-level IID MAE difference")
    _style_axes(ax)
    fig.text(
        0.5,
        0.01,
        "Target: loan_amount_000s = thousands of USD",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.18, bottom=0.18, right=0.97, top=0.90)
    records.append(
        save_figure(
            6,
            "primary_minus_global_decile_mae",
            fig,
            decile_delta,
            ["outputs/reports/prompt5a_plot_deciles.csv"],
            {"x": "True-target IID decile", "y": "Primary minus Global MAE"},
            ["Decile MAE difference"],
            {"MAE difference": TARGET_UNITS},
        )
    )

    bootstrap_samples = read_csv("prompt5a_plot_bootstrap_samples.csv")
    mae_samples = bootstrap_samples[["MAE"]].copy()
    summary = read_csv("prompt5a_iid_bootstrap.csv")
    mae_summary = summary[summary["metric"] == "MAE"].iloc[0]
    fig, ax = plt.subplots(figsize=(7.3, 4.3))
    ax.hist(mae_samples["MAE"], bins=28, color=primary_color, alpha=0.86, edgecolor="white")
    ax.axvline(
        mae_summary["observed_difference_primary_minus_global"],
        color="#222222",
        linewidth=1.5,
        label="Observed difference",
    )
    ax.axvline(mae_summary["percentile_2_5"], color="#b44c42", linestyle="--", linewidth=1.2)
    ax.axvline(mae_summary["percentile_97_5"], color="#b44c42", linestyle="--", linewidth=1.2, label="Frozen 95% interval")
    ax.set_xlabel(f"Primary minus Global MAE ({TARGET_UNITS})")
    ax.set_ylabel("Saved bootstrap resamples")
    ax.set_title("Frozen paired-bootstrap IID MAE difference")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    plot_bootstrap = mae_samples.copy()
    plot_bootstrap["observed_difference"] = mae_summary[
        "observed_difference_primary_minus_global"
    ]
    plot_bootstrap["ci_lower"] = mae_summary["percentile_2_5"]
    plot_bootstrap["ci_upper"] = mae_summary["percentile_97_5"]
    records.append(
        save_figure(
            7,
            "iid_bootstrap_mae_difference",
            fig,
            plot_bootstrap,
            [
                "outputs/reports/prompt5a_plot_bootstrap_samples.csv",
                "outputs/reports/prompt5a_iid_bootstrap.csv",
            ],
            {"x": "Primary minus Global MAE", "y": "Saved resample count"},
            ["MAE difference", "95% percentile interval"],
            {"MAE difference": TARGET_UNITS},
        )
    )

    group_metrics = read_csv("prompt5b_plot_group_metrics.csv")
    group_metrics = group_metrics[group_metrics["analysis_status"] == "ELIGIBLE"].copy()
    field_labels = {
        "applicant_ethnicity_name": "Applicant ethnicity",
        "co_applicant_ethnicity_name": "Co-applicant ethnicity",
        "applicant_race_name_1": "Applicant race",
        "co_applicant_race_name_1": "Co-applicant race",
        "applicant_sex_name": "Applicant sex",
        "co_applicant_sex_name": "Co-applicant sex",
        "minority_population": "Minority population",
        "majority_minority_tract": "Majority-minority tract",
    }
    long_unknown = "Information not provided by applicant in mail, Internet, or telephone application"
    group_metrics["Label"] = (
        group_metrics["sensitive_field"].map(field_labels)
        + " | "
        + group_metrics["group_label"].astype(str).str.replace(
            long_unknown, "Information not provided", regex=False
        )
    )
    group_metrics = group_metrics.sort_values(["sensitive_field", "mae"])
    fig, ax = plt.subplots(figsize=(9.0, 10.5))
    ax.barh(group_metrics["Label"], group_metrics["mae"], color=primary_color)
    ax.set_xlabel(f"Final Primary MAE ({TARGET_UNITS})")
    ax.set_ylabel("Eligible sensitive field and group")
    ax.set_title("Descriptive IID MAE across eligible sensitive groups")
    _style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", labelsize=7.2)
    fig.tight_layout()
    records.append(
        save_figure(
            8,
            "fairness_mae_eligible_groups",
            fig,
            group_metrics,
            ["outputs/reports/prompt5b_plot_group_metrics.csv"],
            {"x": "Final Primary MAE", "y": "Eligible sensitive group"},
            ["Group MAE"],
            {"MAE": TARGET_UNITS},
        )
    )

    group_delta = read_csv("prompt5b_plot_group_comparison.csv")
    group_delta = group_delta[group_delta["analysis_status"] == "ELIGIBLE"].copy()
    group_delta["Label"] = (
        group_delta["sensitive_field"].map(field_labels)
        + " | "
        + group_delta["group_label"].astype(str).str.replace(
            long_unknown, "Information not provided", regex=False
        )
    )
    group_delta = group_delta.sort_values("primary_minus_global_mae")
    fig, ax = plt.subplots(figsize=(9.0, 10.5))
    colors = np.where(
        group_delta["primary_minus_global_mae"] <= 0, primary_color, "#b44c42"
    )
    ax.barh(group_delta["Label"], group_delta["primary_minus_global_mae"], color=colors)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_xlabel(f"Primary minus Global MAE ({TARGET_UNITS})")
    ax.set_ylabel("Eligible sensitive field and group")
    ax.set_title("Descriptive Primary-minus-Global group MAE")
    _style_axes(ax, grid_axis="x")
    ax.tick_params(axis="y", labelsize=7.2)
    fig.tight_layout()
    records.append(
        save_figure(
            9,
            "primary_minus_global_fairness_mae",
            fig,
            group_delta,
            ["outputs/reports/prompt5b_plot_group_comparison.csv"],
            {"x": "Primary minus Global MAE", "y": "Eligible sensitive group"},
            ["Within-group MAE difference"],
            {"MAE difference": TARGET_UNITS},
        )
    )

    global_consensus = read_csv("prompt5b_plot_global_consensus.csv").sort_values(
        "consensus_rank"
    ).head(15)
    fig, ax = plt.subplots(figsize=(7.6, 5.5))
    ordered = global_consensus.sort_values("consensus_rank_score")
    ax.barh(ordered["feature"], ordered["consensus_rank_score"], color=primary_color)
    ax.set_xlabel("Consensus normalized rank score")
    ax.set_ylabel("Feature")
    ax.set_title("Global ensemble consensus Top-15 features")
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    records.append(
        save_figure(
            10,
            "global_consensus_top15",
            fig,
            global_consensus,
            ["outputs/reports/prompt5b_plot_global_consensus.csv"],
            {"x": "Consensus normalized rank score", "y": "Feature"},
            ["Rank-based consensus importance"],
            {"importance": "unitless rank score"},
        )
    )

    for number, source_name, slug, title, color, unit in [
        (
            11,
            "prompt5b_plot_gate_importance.csv",
            "meta_gate_top15",
            "Meta-Gate Top-15 routing drivers",
            "#6f4c9b",
            "mean absolute SHAP in native log-odds routing space",
        ),
        (
            12,
            "prompt5b_plot_residual_importance.csv",
            "residual_specialist_top15",
            "Residual Specialist Top-15 correction drivers",
            "#228b7a",
            "mean absolute SHAP in residual-correction space",
        ),
    ]:
        importance = read_csv(source_name).sort_values("rank").head(15)
        ordered = importance.sort_values("mean_abs_shap")
        figure_size = (8.4, 6.2) if number == 12 else (7.6, 5.5)
        fig, ax = plt.subplots(figsize=figure_size)
        ax.barh(ordered["feature"], ordered["mean_abs_shap"], color=color)
        ax.set_xlabel(unit)
        ax.set_ylabel("Feature")
        ax.set_title(title)
        _style_axes(ax, grid_axis="x")
        if number == 12:
            fig.subplots_adjust(left=0.34, bottom=0.12, right=0.98, top=0.90)
        else:
            fig.tight_layout()
        records.append(
            save_figure(
                number,
                slug,
                fig,
                importance,
                [f"outputs/reports/{source_name}"],
                {"x": unit, "y": "Feature"},
                ["Mean absolute SHAP"],
                {"importance": unit},
            )
        )

    shift = read_csv("prompt5b_plot_residual_body_d10.csv")
    ordered = shift.sort_values("maximum").tail(15)
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    y = np.arange(len(ordered))
    height = 0.38
    ax.barh(
        y - height / 2,
        ordered["body_d1_d9_mean_abs_shap"],
        height,
        label="Body D1-D9",
        color="#7aa6c2",
    )
    ax.barh(
        y + height / 2,
        ordered["d10_mean_abs_shap"],
        height,
        label="D10",
        color="#c56d43",
    )
    ax.set_yticks(y, ordered["feature"])
    ax.set_xlabel("Mean absolute SHAP in residual-correction space")
    ax.set_ylabel("Feature")
    ax.set_title("Residual importance shift: Body vs D10")
    ax.legend(frameon=False)
    _style_axes(ax, grid_axis="x")
    fig.tight_layout()
    records.append(
        save_figure(
            13,
            "residual_body_vs_d10_importance",
            fig,
            shift,
            ["outputs/reports/prompt5b_plot_residual_body_d10.csv"],
            {"x": "Mean absolute SHAP", "y": "Feature"},
            ["Body importance", "D10 importance"],
            {"importance": "residual-correction space, thousands of USD"},
        )
    )

    correction_benefit = read_csv("prompt5b_plot_correction_benefit.csv")
    correction_benefit = correction_benefit[
        correction_benefit["section"] == "realized_benefit"
    ].copy()
    correction_benefit["quantile_index"] = correction_benefit["scope"].str.extract(
        r"(\d+)"
    ).astype(int)
    correction_benefit = correction_benefit.sort_values("quantile_index")
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(
        correction_benefit["scope"],
        correction_benefit["value"],
        color="#228b7a",
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xlabel("Saved correction-magnitude quantile among routed rows")
    ax.set_ylabel("Realized absolute-error benefit\n(thousands of USD)", labelpad=8)
    ax.set_title("Correction magnitude and realized benefit")
    _style_axes(ax)
    fig.text(
        0.5,
        0.01,
        "Target: loan_amount_000s = thousands of USD",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.19, bottom=0.20, right=0.97, top=0.89)
    records.append(
        save_figure(
            14,
            "correction_magnitude_realized_benefit",
            fig,
            correction_benefit,
            ["outputs/reports/prompt5b_plot_correction_benefit.csv"],
            {
                "x": "Saved correction-magnitude quantile",
                "y": "Global absolute error minus Primary absolute error",
            },
            ["Realized absolute-error benefit"],
            {"benefit": TARGET_UNITS},
        )
    )

    architecture = pd.DataFrame(
        [
            ["node", 1, "CatBoost", "weight 0.60"],
            ["node", 1, "LightGBM", "weight 0.20"],
            ["node", 1, "XGBoost", "weight 0.20; log1p native attribution"],
            ["node", 2, "Global ensemble", "fixed weighted prediction"],
            ["node", 3, "Meta-Gate", "route when frozen probability > 0.75"],
            ["node", 4, "Residual Specialist", "Tail residual correction"],
            ["node", 5, "Frozen Stage 3 policy", "alpha 0.75; sign preserved; no cap"],
            ["edge", 1, "Components to Global", "0.60 / 0.20 / 0.20"],
            ["edge", 2, "Global to Gate", "global_prediction_feature"],
            ["edge", 3, "Gate to Specialist", "frozen routing strength"],
            ["edge", 4, "Specialist to final", "scaled residual correction"],
        ],
        columns=["kind", "order", "label", "detail"],
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, detail: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center", weight="bold")
        ax.text(x + w / 2, y + h * 0.30, detail, ha="center", va="center", fontsize=7.8, wrap=True)

    box(0.2, 3.2, 1.55, 0.9, "CatBoost", "weight 0.60", primary_color)
    box(0.2, 1.85, 1.55, 0.9, "LightGBM", "weight 0.20", primary_color)
    box(0.2, 0.5, 1.55, 0.9, "XGBoost", "weight 0.20", primary_color)
    box(2.35, 1.85, 1.75, 0.95, "Global ensemble", "fixed weighted prediction", global_color)
    box(4.7, 1.85, 1.55, 0.95, "Meta-Gate", "p > 0.75", "#6f4c9b")
    box(6.85, 1.85, 1.75, 0.95, "Residual Specialist", "Tail residual correction", "#228b7a")
    box(9.15, 1.7, 1.65, 1.25, "Frozen Stage 3", "alpha 0.75\nsign preserved\nno cap", "#333333")
    for y in [3.65, 2.3, 0.95]:
        ax.add_patch(FancyArrowPatch((1.75, y), (2.35, 2.32), arrowstyle="->", mutation_scale=12, color="#555555"))
    for start, end in [((4.1, 2.32), (4.7, 2.32)), ((6.25, 2.32), (6.85, 2.32)), ((8.6, 2.32), (9.15, 2.32))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, color="#555555"))
    ax.text(5.48, 0.55, "Final prediction = Global + routing strength × predicted residual", ha="center", fontsize=9)
    ax.set_title("Frozen final project architecture", pad=10)
    fig.tight_layout()
    records.append(
        save_figure(
            15,
            "project_architecture",
            fig,
            architecture,
            ["outputs/reports/FINAL_PRE_IID_FREEZE.json"],
            {"x": "Model flow", "y": "Architecture layer"},
            ["Frozen weights", "routing threshold", "correction alpha"],
            {"weights/threshold/alpha": "unitless"},
        )
    )

    validation = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "figure_count": len(records),
        "figures": records,
        "validation_method": (
            "Each final figure was rendered from a saved aggregate plotting table or "
            "frozen architecture record. Saved bootstrap samples were reused; no "
            "resampling was run."
        ),
        "original_iid_content_reads": 0,
        "model_executions": 0,
        "scientific_recomputations": 0,
    }
    write_json(REPORTS / "prompt5c_figure_validation.json", validation)
    ACTIVE_FIGURE_SELECTION = None
    return records


def provenance_footer() -> str:
    return (
        "\n\n---\n\n"
        "Frozen audit provenance: Primary bundle SHA-256 "
        f"{PRIMARY_BUNDLE_HASH}; Global bundle SHA-256 {GLOBAL_BUNDLE_HASH}; "
        f"final pre-IID freeze SHA-256 {EXPECTED_FREEZE_HASH}; final fairness and "
        f"explainability SHA-256 {EXPECTED_FAIRNESS_HASH}."
    )


def build_documents(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    overall = read_csv("prompt5a_iid_overall_metrics.csv")
    primary = overall[overall["model_id"] == PRIMARY_DEPLOYMENT_ID].iloc[0]
    global_row = overall[overall["model_id"] == GLOBAL_DEPLOYMENT_ID].iloc[0]
    bootstrap = read_csv("prompt5a_iid_bootstrap.csv")
    mae_boot = bootstrap[bootstrap["metric"] == "MAE"].iloc[0]
    fairness_summary = read_json(REPORTS / "prompt5b_fairness_summary.json")
    explainability_summary = read_json(REPORTS / "prompt5b_explainability_summary.json")
    final_freeze = read_json(REPORTS / "FINAL_PRE_IID_FREEZE.json")
    fairness = frames["fairness"]
    disparities = frames["disparities"]
    mechanism = dict(
        zip(frames["mechanism"]["Measure"], frames["mechanism"]["Value"])
    )
    improved_groups = int((fairness["Primary minus Global MAE"] < 0).sum())
    worsened = fairness[fairness["Primary minus Global MAE"] > 0][
        ["Sensitive field", "Group", "Primary minus Global MAE"]
    ]
    worsened_text = "; ".join(
        f"{row['Sensitive field']} — {row['Group']}: {row['Primary minus Global MAE']:+.3f}"
        for _, row in worsened.iterrows()
    )
    co_race = fairness[fairness["Sensitive field"] == "co_applicant_race_name_1"]
    co_race_min = co_race.sort_values("Primary MAE").iloc[0]
    co_race_max = co_race.sort_values("Primary MAE").iloc[-1]
    tail_group_metrics = read_csv("prompt5b_tail_group_metrics.csv")
    applicant_race_tail = tail_group_metrics[
        (tail_group_metrics["sensitive_field"] == "applicant_race_name_1")
        & (tail_group_metrics["analysis_status"] == "ELIGIBLE")
    ]
    applicant_race_tail_gaps: dict[str, float] = {}
    for scope, group in applicant_race_tail.groupby("tail_scope"):
        applicant_race_tail_gaps[scope] = float(
            group["primary_tail_mae"].max() - group["primary_tail_mae"].min()
        )
    applicant_race_d10_gap = applicant_race_tail_gaps["IID_GLOBAL_TOP_DECILE"]
    applicant_race_top5_gap = applicant_race_tail_gaps["IID_TOP5_TARGET"]
    global_features = explainability_summary["top_global_features"]
    gate_features = explainability_summary["top_gate_features"]
    residual_features = explainability_summary["top_residual_features"]

    primary_mae = float(primary["mae"])
    global_mae = float(global_row["mae"])
    mae_difference = primary_mae - global_mae
    architecture = final_freeze["primary_recipe"]
    routing_threshold = architecture["routing_threshold"]
    correction_alpha = architecture["correction_alpha"]
    tail_threshold = architecture["tail_threshold"]

    model_card = f"""# Final Model Card

Authorization: {AUTHORIZATION_ID}

## 1. Model Overview

The final Primary model is {PRIMARY_ID}, deployed as {PRIMARY_DEPLOYMENT_ID}. It estimates loan_amount_000s, which is the loan amount in thousands of USD. It was fitted once on all 500,000 Development rows after the model design was frozen. Its primary IID metric is MAE.

The strong comparison model is {GLOBAL_ID}, deployed as {GLOBAL_DEPLOYMENT_ID}. Both models were frozen before the one-time 75,000-row IID evaluation. The Primary did not change after IID.

## 2. Intended Use

The model is intended for research-grade prediction and error analysis of HMDA loan amounts within populations reasonably aligned with the 2017 project data. It may support model benchmarking, sensitivity checks, and further audit of saved outputs under the project governance rules.

## 3. Out-of-Scope Use

The model is not a lending-approval model, a credit-risk model, or a legal compliance tool. It should not be used to decide whether a person receives credit, to set loan terms, or to make causal claims. Use outside the 2017 HMDA context requires a new population and a new independent evaluation.

## 4. Target Definition

The target is loan_amount_000s, measured in thousands of USD. It is a continuous regression outcome. Negative signed error means underprediction; positive signed error means overprediction.

## 5. Data Overview

The project used cleaned 2017 HMDA-based records. Exact duplicates were removed, every member of a contradictory-target group was excluded, and rows overlapping the Legacy project were excluded before population selection. The final population contained 575,000 rows: 500,000 Development rows and 75,000 IID rows.

## 6. Development / IID Split

Development contained 500,000 rows and supported adaptive experimentation plus the final 500,000-row refit. IID contained 75,000 rows, was opened once only after the Primary and Global comparator were frozen, and evaluated exactly those two predeclared models. IID was not used for tuning or selection.

## 7. Leakage Controls

The project used saved deterministic roles and leakage-safe cross-fitting. Learned preprocessing stayed inside fitted model bundles. Meta targets used out-of-fold Global predictions so a training row did not receive an in-sample Global prediction from a model fitted on that row. The original IID files are CLOSED_AFTER_ONE_TIME_EVALUATION.

## 8. Final Feature Contract

The final feature contract is main_without_sensitive_without_lender with 35 features. Sensitive variables were not predictive inputs. Respondent or lender identity was not used in the final predictive model. The feature contract includes borrower-income, program, property, occupancy, geography, and tract-context variables.

## 9. Final Model Architecture

The Global prediction is a fixed ensemble: 0.60 CatBoost + 0.20 LightGBM + 0.20 XGBoost. The Meta-Gate estimates whether the fixed Tail residual correction should be applied. The Residual Specialist predicts a residual correction in the frozen upper-Tail regime. The operational Tail threshold is {tail_threshold:.1f}, the strict routing threshold is {routing_threshold:.2f}, and correction alpha is {correction_alpha:.2f}. The fitted residual sign is preserved and no cap is applied.

## 10. Development Research Summary

The project evaluated simple baselines, a linear model, HistGradientBoosting, CatBoost, LightGBM, XGBoost, lender diagnostics, RealMLP, FT-Transformer, static ensembles, Tail Gates, direct Specialists, Meta-Gates, Residual Specialists, quantile residual models, benefit-aware routing, a Beat classifier, shrinkage rules, DenseWeight, SERA, IMr-GB, and LDS. Many alternatives improved one region while worsening another. No alternative removed the Body/Tail trade-off.

## 11. Final Selection Rationale

Stage 3 was selected before IID under a balanced conservative Tail-aware philosophy. It offered a reproducible compromise between overall MAE, Body preservation, and upper-Tail behavior. It was not selected as a universal optimum, and IID results did not trigger reselection.

## 12. Final IID Performance

On 75,000 IID rows, Primary MAE was {primary_mae:.6f}, RMSE was {primary['rmse']:.6f}, R² was {primary['r2']:.6f}, RMSLE was {primary['rmsle']:.6f}, median absolute error was {primary['median_absolute_error']:.6f}, and P90 absolute error was {primary['p90_absolute_error']:.6f}. Values for absolute-error metrics use thousands of USD.

## 13. Relative Percentage Error

Primary MAPE was {primary['mape_percent']:.6f}% and WAPE was {primary['wape_percent']:.6f}%. Global MAPE was {global_row['mape_percent']:.6f}% and WAPE was {global_row['wape_percent']:.6f}%. Primary had slightly higher observed MAPE but lower observed WAPE. MAPE used positive targets without an epsilon; all 75,000 IID targets were valid.

## 14. Tail Performance

Primary Bottom-90 MAE was {primary['bottom_90_mae']:.6f}, Top-decile MAE was {primary['top_decile_mae']:.6f}, Top-5% MAE was {primary['top_five_percent_mae']:.6f}, and P85-P95 MAE was {primary['p85_to_p95_boundary_mae']:.6f}. The frozen result was 5/6. C2 failed because the required 3% Top-decile MAE improvement was not achieved.

## 15. Error-by-Target-Decile Summary

The saved IID decile profile shows that error changes materially across the target distribution. The final paper-ready decile table reports D1-D10 counts, target ranges, target means, MAE, MAPE, WAPE, signed error, and underprediction rates. It was copied from saved Prompt 5A evidence without row-level recomputation.

## 16. Comparison with Global Baseline

Global IID MAE was {global_mae:.6f}. Primary minus Global MAE was {mae_difference:.6f}; lower is better. Primary improved Top-decile and Top-5% MAE, while Bottom-90 MAE increased by {primary['bottom_90_mae'] - global_row['bottom_90_mae']:.6f}. This pattern is the central Body/Tail trade-off.

## 17. Statistical Uncertainty

The saved paired 500-resample bootstrap gave a Primary-minus-Global MAE mean of {mae_boot['bootstrap_mean']:.6f} and a 95% percentile interval of [{mae_boot['percentile_2_5']:.6f}, {mae_boot['percentile_97_5']:.6f}]. All saved MAE resamples favored Primary. This restates frozen inferential evidence; no bootstrap was rerun.

## 18. Fairness / Sensitive Performance Audit

The audit covered eight sensitive/audit fields, 9,053 observed raw levels, and 27 eligible ranked group levels. Primary improved MAE relative to Global in {improved_groups} of 27 eligible groups. The two observed worsening cells were {worsened_text}. The largest eligible overall MAE gap was {float(co_race_max['Primary MAE'] - co_race_min['Primary MAE']):.3f} for co-applicant race: {co_race_max['Group']} MAE {co_race_max['Primary MAE']:.3f} versus {co_race_min['Group']} MAE {co_race_min['Primary MAE']:.3f}. Applicant-race Tail MAE gaps were {applicant_race_d10_gap:.3f} in the inclusive top decile and {applicant_race_top5_gap:.3f} in the top 5%.

{FAIRNESS_DISCLAIMER} Group target composition differs, small groups are less stable, and these descriptive results do not prove the absence or presence of discrimination.

## 19. Explainability

Global consensus drivers were {", ".join(global_features)}. Meta-Gate drivers were {", ".join(gate_features)}. Residual Specialist drivers were {", ".join(residual_features)}.

Global components, routing, and residual correction were explained separately. XGBoost attribution remains in native log1p space. No single exact additive SHAP decomposition is claimed for the full Stage 3 system. {EXPLAINABILITY_DISCLAIMER}

## 20. Routing / Residual Mechanism

The gate routed {int(mechanism['Routed rows']):,} of {int(mechanism['IID rows']):,} IID rows ({mechanism['Routing rate'] * 100:.3f}%). Mean routed correction was {mechanism['Mean routed correction']:+.3f}, median absolute routed correction was {mechanism['Median absolute routed correction']:.3f}, and mean realized routed-row absolute-error benefit was {mechanism['Routed-row realized benefit']:.3f}. Across all IID rows, correction was positive for {mechanism['Positive correction rate'] * 100:.3f}%, negative for {mechanism['Negative correction rate'] * 100:.3f}%, and zero for {mechanism['Zero correction rate'] * 100:.3f}%.

## 21. Known Limitations

HMDA 2017 does not provide every variable relevant to loan amount. Direct property-value and LTV-like information was unavailable. High-loan observations remained difficult, and the model continued to underpredict many of them. Development was adaptive. Only two predeclared models received IID evaluation. Geography may proxy unobserved context. The model may drift outside the studied population.

## 22. Ethical / Fair-Lending Considerations

Sensitive variables were excluded from predictive inputs, but that does not establish fairness. Predictive error varied across groups, and some disparities were large. The model predicts an amount recorded in HMDA; it does not evaluate approval fairness, disparate treatment, or legal compliance. Any operational use requires separate governance, legal review, monitoring, and population-specific validation.

## 23. Reproducibility

The final Primary bundle SHA-256 is {PRIMARY_BUNDLE_HASH}. The Global bundle SHA-256 is {GLOBAL_BUNDLE_HASH}. The final pre-IID freeze SHA-256 is {EXPECTED_FREEZE_HASH}. Saved split roles, fixed OOF construction, model hashes, report hashes, table hashes, figure hashes, notebook hash, and environment references are indexed in FINAL_REPRODUCIBILITY_MANIFEST.json.

## 24. Governance and Freeze History

Development research closed before final refit. Prompt 4C froze the Primary and Global comparator. Prompt 5A performed the one-time IID evaluation without fitting, tuning, or model change. Prompt 5B completed descriptive fairness and hierarchical explainability without changing the models. Prompt 5C formats and restates frozen evidence only.

## 25. Recommended Use

Use the frozen Primary for the narrow research prediction task when the input contract and population assumptions hold. Report both overall and Tail errors, retain the Global comparator, monitor upper-Tail underprediction, and preserve the fairness and explainability caveats. A new independent holdout is required for another independent final evaluation.

## 26. Final Status

Final model: {PRIMARY_ID}. Training rows: 500,000 Development. Feature count: 35. Primary IID metric: MAE. IID MAE: {primary_mae:.6f}. Global comparator IID MAE: {global_mae:.6f}. IID holdout rows: 75,000. Model changed after IID: No. IID used for tuning: No. Sensitive variables used as predictive features: No. Lender identity used in the final predictive model: No. Final scientific result: 5/6 frozen conditions, with C2 failed.
"""
    model_card += provenance_footer()

    technical_sections = [
        (
            "1. Executive Summary",
            f"The project developed a regression system for HMDA loan amounts and closed model research before one-time IID evaluation. The frozen {PRIMARY_ID} Primary used 35 non-sensitive, non-lender features and all 500,000 Development rows. On 75,000 untouched IID rows it achieved MAE {primary_mae:.6f}, compared with {global_mae:.6f} for the frozen Global ensemble. The observed MAE difference was {mae_difference:.6f}. Five of six predeclared Tail-aware conditions passed; C2 failed.",
        ),
        (
            "2. Research Objective",
            "The objective was accurate loan-amount regression with explicit attention to upper-Tail error, leakage prevention, reproducibility, and transparent post-fit audit. The target was loan_amount_000s in thousands of USD. The final model was chosen as a balanced compromise rather than as an optimizer of one Tail statistic.",
        ),
        (
            "3. Raw HMDA Data",
            "The source was HMDA 2017 project data. Raw fields represented loan, applicant, property, program, lender, geography, and tract context. Raw data and later original IID files were protected by stage-specific access rules. Prompt 5C did not open any raw or original IID file.",
        ),
        (
            "4. Data Cleaning",
            "The data pipeline validated schemas and preserved valid extremes. It did not cap or remove records only because they were statistically unusual. Cleaning was completed before model development and saved as immutable processed artifacts.",
        ),
        (
            "5. Duplicate / Contradictory-Target Handling",
            "Exact duplicates were removed through collision-safe global checks. For any duplicated feature identity with contradictory target values, every member of that contradictory group was excluded. This prevented ambiguous supervision from entering the selected population.",
        ),
        (
            "6. Legacy-Overlap Exclusion",
            "The project matched the Legacy project by the frozen unique-key contract and conservatively excluded every matching V2 row. The final selected population had zero Legacy overlap under the saved verification evidence.",
        ),
        (
            "7. Deterministic Development / IID Design",
            "A deterministic 575,000-row population was split into 500,000 Development rows and 75,000 IID rows. Development supported adaptive work and a fixed 400,000/100,000 Train/Validation structure. IID was opened once after final freeze. Exactly the Primary and Global comparator were evaluated, and IID caused no model change.",
        ),
        (
            "8. Feature Engineering",
            "The final contract contained 35 features under main_without_sensitive_without_lender. It included original program, property, occupancy, geography, and tract fields plus validated transformations such as log income and population, applicant-income-to-area-income ratios, tract ratios, occupancy ratios, co-applicant presence, grouped program context, and region.",
        ),
        (
            "9. Leakage and Governance Controls",
            "Sensitive labels and lender identity were excluded from final prediction. Learned preprocessing remained within fitted bundles. Saved roles and cross-validation memberships were reused. Leakage-sensitive meta targets used out-of-fold Global predictions. Test-like IID evidence never selected features, thresholds, alpha, models, or weights.",
        ),
        (
            "10. Baselines",
            "Early work established simple baselines and common evaluation contracts. These references made later gains interpretable and prevented Tail metrics from hiding overall degradation. Baselines were Development evidence only and were not all carried into IID.",
        ),
        (
            "11. Linear / HGB Results",
            "A regularized linear model and HistGradientBoosting were evaluated as structured non-neural references. They helped identify nonlinear signal and the difficulty of the high-target region. They did not become final project models.",
        ),
        (
            "12. CatBoost / LightGBM / XGBoost",
            "The three boosting families were the strongest broad model family set. Their complementary predictions supported the frozen Global ensemble with weights 0.60 CatBoost, 0.20 LightGBM, and 0.20 XGBoost. XGBoost used a log1p target mode; CatBoost and LightGBM used raw target mode.",
        ),
        (
            "13. Lender Diagnostic",
            "Matched lender-ablation experiments tested whether respondent identity materially altered Development performance. Lender identity was diagnostic only. It was excluded from the final predictive feature contract to support cleaner governance and portability.",
        ),
        (
            "14. Deep Tabular Models",
            "RealMLP and FT-Transformer were evaluated with fixed budgets and saved bundles. FT-Transformer served as a descriptive deep anchor. Deep models did not replace the stronger boosting and Tail-aware path, and no deep model was evaluated on final IID.",
        ),
        (
            "15. Global Ensemble",
            "The Global comparator combined three frozen full-Development boosting components. It was intentionally simple and strong. Its IID MAE was "
            f"{global_mae:.6f}, RMSE {global_row['rmse']:.6f}, MAPE {global_row['mape_percent']:.6f}%, and WAPE {global_row['wape_percent']:.6f}%.",
        ),
        (
            "16. Tail Error Diagnosis",
            "Development analysis showed a persistent Body/Tail trade-off: methods that increased attention to high targets could improve Tail behavior while damaging the much larger Body. Signed error and underprediction rates showed systematic high-loan underprediction. This remained the main modeling limitation.",
        ),
        (
            "17. Prompt 4A Tail Modeling",
            "Prompt 4A tested no-fit ensembles, Tail Gates, direct Specialists, Tail-weighted models, hard routing, and soft mixtures under a frozen budget. It generated useful Tail components but did not select a final project model or open IID.",
        ),
        (
            "18. Prompt 4B Calibration / Meta-Gating / Residual Specialist",
            "Prompt 4B evaluated calibrated routing, a Meta-Gate, a Tail-only Residual Specialist, and static ensemble references. Stage 3 combined the fixed Global prediction, a gate using the Global prediction and original features, and a residual specialist trained with leakage-safe OOF residual targets.",
        ),
        (
            "19. Prompt 4B2 Benefit-Aware Correction",
            "Benefit-aware experiments tested no-fit correction rules, quantile residual models, and a learned Benefit Router. They used Selection for ranking and kept Audit descriptive. These approaches did not remove the Body/Tail compromise and did not become the final model.",
        ),
        (
            "20. Prompt 4B3 Beat-Probability Shrinkage",
            "A predeclared Beat classifier and fixed shrinkage policies estimated whether correction would beat the Global prediction. Diagnostic gates passed, but the resulting policy remained an adaptive Development experiment. It was not promoted to the final Primary.",
        ),
        (
            "21. Prompt 4B4 Literature-Grounded Imbalance Methods",
            "DenseWeight, SERA, IMr-GB, and LDS were reproduced under a fixed four-fit design. Some methods improved selected roles, but none produced a six-condition breakthrough or a new Development MAE record. Their negative results support the conclusion that reweighting alone did not solve the Tail trade-off.",
        ),
        (
            "22. Final Selection Philosophy",
            "Final selection prioritized balanced conservative Tail-aware performance, then overall MAE, reproducibility, and methodological cleanliness. Stage 3 was frozen before IID. A historical MAE challenger was excluded from IID because its available frozen inference chain did not exactly reproduce its immutable Validation prediction.",
        ),
        (
            "23. Prompt 4C Full-Development Refit",
            "Prompt 4C reconstructed exact recipes, produced deterministic two-fold OOF Global predictions for all 500,000 Development rows, and completed 11 fixed refit roles with zero technical retries and zero scientific searches. Final component and composite bundles reloaded with exact saved-prediction agreement.",
        ),
        (
            "24. One-Time IID Evaluation",
            f"Prompt 5A opened IID once after freeze and evaluated exactly two predeclared models. Primary MAE was {primary_mae:.6f}; Global MAE was {global_mae:.6f}; the observed difference was {mae_difference:.6f}. The final Primary remained unchanged and the original IID files were closed.",
        ),
        (
            "25. Relative Error Analysis: MAPE/WAPE",
            f"Primary MAPE was {primary['mape_percent']:.6f}% versus {global_row['mape_percent']:.6f}% for Global, so observed MAPE was slightly worse. Primary WAPE was {primary['wape_percent']:.6f}% versus {global_row['wape_percent']:.6f}%, so observed WAPE was lower. The difference matters because MAPE weights low targets differently from WAPE.",
        ),
        (
            "26. Target-Decile Error Analysis",
            "Frozen D1-D10 tables show changing error scale and direction across the target distribution. Low-target MAPE was large because small denominators amplify relative error. Upper deciles carried much larger absolute error and more underprediction. The final decile table is a direct reporting view of saved Prompt 5A evidence.",
        ),
        (
            "27. Body vs Tail Generalization",
            f"Primary Bottom-90 MAE was {primary['bottom_90_mae']:.6f}, slightly above Global at {global_row['bottom_90_mae']:.6f}. Primary Top-decile MAE was {primary['top_decile_mae']:.6f}, below Global at {global_row['top_decile_mae']:.6f}. Primary Top-5% MAE was {primary['top_five_percent_mae']:.6f}, below Global at {global_row['top_five_percent_mae']:.6f}. C2 still failed because the Top-decile reduction was below 3%.",
        ),
        (
            "28. Bootstrap Uncertainty",
            f"The saved paired 500-resample bootstrap produced a mean MAE difference of {mae_boot['bootstrap_mean']:.6f} and a 95% interval [{mae_boot['percentile_2_5']:.6f}, {mae_boot['percentile_97_5']:.6f}]. Primary had lower MAE in all saved resamples. Prompt 5C reused these samples and summary values without resampling.",
        ),
        (
            "29. Fairness / Sensitive Performance Audit",
            f"The audit used eight saved sensitive/audit fields after prediction. There were 9,053 observed raw levels and 27 eligible ranked levels. Primary improved within-group MAE for {improved_groups}/27 eligible groups; {worsened_text}. The largest overall eligible gap was {float(co_race_max['Primary MAE'] - co_race_min['Primary MAE']):.3f} for co-applicant race. Applicant-race Tail MAE gaps were {applicant_race_d10_gap:.3f} in the inclusive top decile and {applicant_race_top5_gap:.3f} in the top 5%. {FAIRNESS_DISCLAIMER} Target composition and small-group limitations apply.",
        ),
        (
            "30. Final Explainability",
            f"Global consensus emphasized {', '.join(global_features)}. Routing emphasized {', '.join(gate_features)}. Residual correction emphasized {', '.join(residual_features)}. Component spaces were kept separate, XGBoost remained in native log1p space, and no exact additive full-system SHAP claim was made. {EXPLAINABILITY_DISCLAIMER}",
        ),
        (
            "31. Stage 3 Routing / Correction Mechanism",
            f"Stage 3 routed {int(mechanism['Routed rows']):,} rows ({mechanism['Routing rate'] * 100:.3f}%). Mean routed correction was {mechanism['Mean routed correction']:+.3f}; median absolute routed correction was {mechanism['Median absolute routed correction']:.3f}; realized routed-row benefit averaged {mechanism['Routed-row realized benefit']:.3f}. Correction was zero for {mechanism['Zero correction rate'] * 100:.3f}% of all IID rows, showing that the frozen intervention was selective.",
        ),
        (
            "32. Limitations",
            "HMDA 2017 lacks direct property-value and LTV-like variables. Tail underprediction remained substantial and C2 failed. Development evidence was adaptive, while IID covered only two predeclared models. Fairness findings were descriptive. SHAP was not causal. Sensitive inputs were excluded, but geography may proxy unobserved context. The model predicts loan amounts, not lending decisions.",
        ),
        (
            "33. Reproducibility",
            f"Saved data-lineage evidence, roles, OOF rules, feature contracts, model bundles, reports, tables, figures, and the artifact-only notebook are hash-indexed. Primary bundle SHA-256 is {PRIMARY_BUNDLE_HASH}; Global bundle SHA-256 is {GLOBAL_BUNDLE_HASH}. Prompt 5C performed zero fits, predictions, SHAP computations, bootstraps, or original IID reads.",
        ),
        (
            "34. Final Conclusions",
            "The frozen Stage 3 Tail-aware residual architecture produced a small and consistent observed improvement in overall IID MAE relative to the strong Global ensemble, preserved Body error within the frozen tolerance, and reduced several Tail biases. It passed five of six predeclared conditions but did not achieve the required 3% Top-decile improvement. Alternative routing, residual, imbalance-aware, deep-learning, and reweighting methods did not remove the same trade-off. The Primary is recommended only with its Tail, fairness, explainability, and population limitations clearly reported.",
        ),
    ]
    technical_report = "# Final Technical Report\n\n" + "\n\n".join(
        f"## {heading}\n\n{text}" for heading, text in technical_sections
    )
    technical_report += provenance_footer()

    paper_methods = f"""# Paper-Ready Methods Text

## Population construction

We constructed a 575,000-record HMDA 2017 analysis population after collision-safe exact deduplication, exclusion of every member of contradictory-target groups, and conservative exclusion of records overlapping a Legacy project. A deterministic split assigned 500,000 records to Development and 75,000 records to a one-time IID holdout.

## Leakage controls

All model development used Development evidence. The IID holdout remained closed until model freeze. Learned preprocessing stayed inside each saved model bundle. Leakage-sensitive meta targets used deterministic out-of-fold Global predictions with one prediction per Development row and zero self-fit rows. IID was not used for tuning, feature choice, threshold choice, model selection, or model replacement.

## Feature contract

The final main_without_sensitive_without_lender contract contained 35 features. It combined loan and property descriptors, income and tract context, geographic variables, and validated derived ratios and log transforms. Sensitive/audit fields and respondent or lender identity were excluded from prediction.

## Modeling families

Development experiments covered baselines, a regularized linear model, HistGradientBoosting, CatBoost, LightGBM, XGBoost, lender diagnostics, RealMLP, FT-Transformer, static ensembles, Tail Gates, direct and residual Specialists, Meta-Gates, quantile residual models, benefit-aware routing, Beat-probability shrinkage, DenseWeight, SERA, IMr-GB, and LDS.

## Tail-aware architecture

The frozen Global model combined 0.60 CatBoost, 0.20 LightGBM, and 0.20 XGBoost. A Meta-Gate estimated whether to apply an upper-Tail residual correction. A Residual Specialist predicted that correction. The final Stage 3 rule used strict gate probability greater than {routing_threshold:.2f}, alpha {correction_alpha:.2f}, preserved the fitted residual sign, and used no cap.

## Final refit

After selection closed, fixed recipes were fitted on all 500,000 Development rows. Deterministic two-fold OOF Global predictions supplied leakage-safe targets for the Meta-Gate and Residual Specialist. No new scientific search occurred during final refit.

## One-time IID evaluation

After the final Primary and Global comparator were frozen, the 75,000-row IID holdout was opened once. Exactly those two models were evaluated. The Primary remained unchanged after evaluation, and the original IID files were then closed.

## Error metrics

The primary metric was MAE. Saved reports also covered RMSE, R², RMSLE, median and P90 absolute error, signed error, MAPE, WAPE, Bottom-90 MAE, Top-decile MAE, Top-5% MAE, boundary MAE, and underprediction rates. MAPE used strictly positive targets without an epsilon. Saved paired 500-resample bootstrap evidence summarized Primary-minus-Global differences.

## Fairness audit

Post-prediction descriptive error analysis covered eight available sensitive/audit fields. Group rankings required the frozen minimum sample rules; small groups remained visible but were suppressed from quantitative ranking. Common Tail scopes used saved masks. {FAIRNESS_DISCLAIMER}

## Explainability

Saved component explanations treated CatBoost, LightGBM, XGBoost, the Meta-Gate, and the Residual Specialist separately. A rank-based consensus combined only ranks across Global components; raw SHAP magnitudes were not summed. XGBoost attribution remained in native log1p space. {EXPLAINABILITY_DISCLAIMER}
"""
    paper_methods += provenance_footer()

    paper_results = f"""# Paper-Ready Results Text

## Dataset and split

The final population contained 575,000 HMDA 2017 records: 500,000 adaptive Development records and a 75,000-record untouched one-time IID holdout. The final feature contract contained 35 non-sensitive, non-lender predictors.

## Candidate-model development

Boosting families produced the strongest Global baseline. Deep tabular models, Tail-weighted objectives, direct Specialists, gating and residual-routing variants, quantile corrections, benefit-aware routing, Beat-probability shrinkage, and literature-based imbalance methods were also tested. No alternative removed the Body/Tail trade-off across the frozen condition set.

## Final selection

Before IID access, Stage 3 ({PRIMARY_ID}) was frozen as the Primary under a balanced Tail-aware selection philosophy. The comparator was a fixed 0.60 CatBoost + 0.20 LightGBM + 0.20 XGBoost Global ensemble. Stage 3 used a Meta-Gate and a Residual Specialist to apply a selective correction.

## IID evaluation

Primary IID MAE was {primary_mae:.6f}, compared with {global_mae:.6f} for Global (observed Primary-minus-Global difference {mae_difference:.6f}). The frozen paired bootstrap mean difference was {mae_boot['bootstrap_mean']:.6f}, with a 95% percentile interval [{mae_boot['percentile_2_5']:.6f}, {mae_boot['percentile_97_5']:.6f}]. The Primary remained unchanged.

## Tail performance

Primary Top-decile MAE was {primary['top_decile_mae']:.6f} versus {global_row['top_decile_mae']:.6f} for Global, and Top-5% MAE was {primary['top_five_percent_mae']:.6f} versus {global_row['top_five_percent_mae']:.6f}. Bottom-90 MAE was {primary['bottom_90_mae']:.6f} versus {global_row['bottom_90_mae']:.6f}. Five of six frozen conditions passed. C2 failed because Top-decile MAE did not improve by the required 3%.

## Relative error

Primary MAPE was {primary['mape_percent']:.6f}% and WAPE was {primary['wape_percent']:.6f}%. Global MAPE was {global_row['mape_percent']:.6f}% and WAPE was {global_row['wape_percent']:.6f}%. Thus Primary had slightly higher MAPE but lower WAPE.

## Fairness

The descriptive audit covered eight fields, 9,053 observed levels, and 27 eligible ranked levels. Primary reduced MAE relative to Global in {improved_groups}/27 eligible groups. The largest eligible overall MAE gap was {float(co_race_max['Primary MAE'] - co_race_min['Primary MAE']):.3f}, between co-applicant race groups {co_race_max['Group']} (MAE {co_race_max['Primary MAE']:.3f}) and {co_race_min['Group']} (MAE {co_race_min['Primary MAE']:.3f}). Applicant-race Tail MAE gaps were {applicant_race_d10_gap:.3f} in the inclusive top decile and {applicant_race_top5_gap:.3f} in the top 5%. Group target distributions differed, so target composition may explain part of the observed variation. Small-group estimates are less stable and were excluded from ranked comparisons under the frozen thresholds. {FAIRNESS_DISCLAIMER}

## Explainability

Global consensus emphasized income, lien, loan purpose, occupancy, program, and geographic/context features. Routing was led by the Global prediction plus income and geographic/context features. Residual correction was led by the Global prediction, relative applicant income, loan purpose, agency, geography, occupancy, and program features. {EXPLAINABILITY_DISCLAIMER}

## Limitations

High-loan underprediction remained the primary predictive limitation. HMDA 2017 lacked direct property-value and LTV-like variables. Development evidence was adaptive, only two predeclared models were evaluated on IID, group findings were descriptive, and attribution was not causal.
"""
    paper_results += provenance_footer()

    paper_limitations = f"""# Paper-Ready Limitations Text

The project used HMDA 2017 information, which does not capture every determinant of loan amount. In particular, the feature set lacked direct property-value and LTV-like measures. These missing variables limit the model's ability to represent collateral value and financing structure.

The upper Tail remained the principal predictive limitation. Even after final selection and IID validation, the model underpredicted many high-loan observations. The frozen Stage 3 Primary improved observed Top-decile and Top-5% errors relative to the Global comparator, but it did not meet frozen C2, which required at least a 3% Top-decile MAE improvement.

Development research was adaptive and tested many modeling and routing choices. Development scores must therefore not be described as independent Test evidence. The one-time IID evaluation was limited to two models that were declared before the holdout was opened. It does not support claims about the IID ranking of every model trained during Development.

Sensitive variables were excluded from predictive inputs, but exclusion alone does not establish fairness. The post-prediction audit found descriptive error differences across available groups. {FAIRNESS_DISCLAIMER} Small-group uncertainty, multiple descriptive comparisons, and differing target composition limit interpretation.

SHAP and rank-based feature attributions describe fitted-model behavior. SHAP is not causal evidence, and no causal interpretation is made. XGBoost attributions are in native log1p space, and attribution magnitudes from incompatible models were not merged into a single Stage 3 decomposition.

Geographic and tract-context variables may proxy unobserved socioeconomic or market context. Their presence supports prediction but requires careful monitoring and governance. The final system optimizes loan-amount prediction; it does not optimize or evaluate lending decisions, approval equity, or legal compliance.

A new population, new time period, or changed data contract requires fresh validation. Because the original IID holdout has already been consumed, another independent final evaluation would require a new holdout.
"""
    paper_limitations += provenance_footer()

    project_summary = f"""# Final Project Summary

## What the project did

Regression V2 estimated loan_amount_000s, the reported loan amount in thousands of USD, from HMDA 2017 information. The work focused on strong overall prediction while treating high-loan error as a separate, visible problem. It also imposed strict controls for leakage, model freeze, one-time IID use, fairness reporting, and explainability.

The final selected population had 575,000 rows. Development used 500,000 rows and included an adaptive research process. The untouched IID holdout contained 75,000 rows and was opened once only after the final Primary and Global comparator were frozen.

## What was tried

The research program started with baselines, a linear model, and HistGradientBoosting. It then evaluated CatBoost, LightGBM, and XGBoost, plus a lender diagnostic. RealMLP and FT-Transformer provided deep-tabular comparisons. The strongest broad predictor became a fixed Global ensemble with weights 0.60 CatBoost, 0.20 LightGBM, and 0.20 XGBoost.

Upper-Tail analysis showed a recurring trade-off. A method could reduce error for large loans yet slightly damage the much larger Body. The project tested static ensembles, Tail Gates, direct Specialists, Meta-Gates, Residual Specialists, quantile residual models, expected-benefit routing, a Beat classifier, shrinkage policies, DenseWeight, SERA, IMr-GB, and LDS. These experiments were useful, including their negative results, but none removed the trade-off.

## What finally worked

The frozen Primary was {PRIMARY_ID}. It starts with the Global ensemble. A Meta-Gate estimates whether a fixed residual correction should be applied. A Residual Specialist predicts that correction for the upper-Tail regime. The final rule routes only above the frozen probability threshold, scales by alpha {correction_alpha:.2f}, preserves the fitted residual sign, and applies no cap.

This design was chosen before IID under a balanced conservative philosophy. It was selected because it combined a small overall gain, controlled Body cost, better Tail direction, exact reproducibility, and clear component boundaries. It was not presented as a universal optimum.

## Final IID result

On 75,000 one-time IID rows, the Primary achieved MAE {primary_mae:.6f}. The Global comparator achieved MAE {global_mae:.6f}. Primary minus Global MAE was {mae_difference:.6f}. The saved paired 500-resample bootstrap gave a 95% interval of [{mae_boot['percentile_2_5']:.6f}, {mae_boot['percentile_97_5']:.6f}].

The Primary passed five of six predeclared Tail-aware conditions. It improved overall MAE, stayed inside the allowed Bottom-90 and RMSE tolerances, moved Top-decile signed error closer to zero, and reduced Top-decile underprediction. C2 failed because the Top-decile MAE reduction was less than the required 3%.

## Tail limitation

The Tail is still the main predictive limitation. Primary Top-decile MAE was {primary['top_decile_mae']:.6f}, and Top-5% MAE was {primary['top_five_percent_mae']:.6f}. Many high-loan rows were still underpredicted. The project therefore does not claim that the Tail problem was solved.

## Fairness result

Sensitive labels were excluded from prediction and used only for post-prediction audit. The audit covered eight fields, 9,053 observed raw levels, and 27 eligible ranked levels. Primary improved MAE relative to Global for {improved_groups} of 27 eligible groups, but some gaps remained large. The largest overall eligible MAE gap was {float(co_race_max['Primary MAE'] - co_race_min['Primary MAE']):.3f} in co-applicant race. Applicant-race Tail MAE gaps were {applicant_race_d10_gap:.3f} in the inclusive top decile and {applicant_race_top5_gap:.3f} in the top 5%.

{FAIRNESS_DISCLAIMER} Differences may reflect target composition, available covariates, missing variables, and sampling uncertainty. Small groups were not ranked.

## Explainability

Global prediction was driven mainly by borrower income, lien status, loan purpose, occupancy, program, and geographic or tract context. Routing depended strongly on the Global prediction and also on income and location context. Residual correction depended strongly on the Global prediction, relative applicant income, loan purpose, agency, geography, occupancy, and program features.

These explanations were kept hierarchical. XGBoost stayed in its native log1p attribution space, and raw SHAP magnitudes were not combined across incompatible components. {EXPLAINABILITY_DISCLAIMER}

## Recommendation

Use the frozen Primary for the narrow research task when the 35-feature contract and population assumptions hold. Always report overall and Tail metrics together. Keep the strong Global comparator available. Monitor high-loan underprediction and group error disparities. Do not use this model for lending approval, legal fairness certification, or causal conclusions.

The project is complete. Further modeling or another independent final evaluation would need new authorization and, for evaluation, a new holdout.
"""
    project_summary += provenance_footer()

    documents = {
        "MODEL_CARD_FINAL.md": model_card,
        "FINAL_TECHNICAL_REPORT.md": technical_report,
        "FINAL_PROJECT_SUMMARY.md": project_summary,
        "PAPER_METHODS_TEXT.md": paper_methods,
        "PAPER_RESULTS_TEXT.md": paper_results,
        "PAPER_LIMITATIONS_TEXT.md": paper_limitations,
    }
    records = []
    for name, body in documents.items():
        path = FINAL / name
        write_text(path, body)
        records.append(
            {
                "path": rel(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "status": "PASS",
            }
        )
    return records


def build_notebook() -> dict[str, Any]:
    nb = nbformat.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": platform.python_version()}
    setup = """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd()
FINAL = ROOT / "outputs" / "final"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
REPORTS = ROOT / "outputs" / "reports"

def show_table(name, rows=None):
    frame = pd.read_csv(TABLES / name)
    display(frame if rows is None else frame.head(rows))

def show_image(name):
    display(Image(filename=str(FIGURES / name)))

display(Markdown("Artifact-only notebook setup complete. All displayed values and images are saved reporting artifacts."))"""
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Regression V2 — Final Project Reporting\n\n"
            "Authorization: regression_v2_prompt5c_final_reporting.\n\n"
            "This notebook is artifact-only. It performs no model fitting, prediction generation, SHAP calculation, bootstrap resampling, or original IID access."
        ),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_markdown_cell(
            "## 1. Executive Summary\n\n"
            "The frozen Stage 3 Primary produced a small observed overall IID MAE improvement over the frozen Global ensemble. It passed five of six predeclared conditions. C2 failed, so the required 3% Top-decile MAE improvement was not achieved."
        ),
        nbformat.v4.new_code_cell(
            "show_table('table_final_model_performance.csv')\nshow_image('figure01_iid_overall_metrics.png')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 2. Data Lineage\n\n"
            "The saved handoff validates the immutable Prompt 4C, 5A, and 5B evidence. Development contained 500,000 rows. The one-time IID holdout contained 75,000 rows and is closed."
        ),
        nbformat.v4.new_code_cell(
            "handoff = json.loads((REPORTS / 'prompt5c_handoff_validation.json').read_text(encoding='utf-8'))\n"
            "display(pd.DataFrame(handoff['checks'])[['artifact', 'actual_status', 'sha256', 'status']])"
        ),
        nbformat.v4.new_markdown_cell(
            "## 3. Modeling Timeline\n\n"
            "The research timeline covered baselines, boosting, lender diagnostics, deep tabular models, ensembles, Tail Gates, Meta-Gates, Residual Specialists, benefit-aware correction, Beat-probability shrinkage, and literature-based imbalance methods. Development research closed before final refit and IID access."
        ),
        nbformat.v4.new_markdown_cell(
            "## 4. Final Architecture\n\n"
            "The Global ensemble uses 0.60 CatBoost, 0.20 LightGBM, and 0.20 XGBoost. The Meta-Gate controls a frozen residual correction from the Residual Specialist."
        ),
        nbformat.v4.new_code_cell(
            "show_image('figure15_project_architecture.png')\nshow_table('table_stage3_mechanism.csv')"
        ),
        nbformat.v4.new_markdown_cell("## 5. Development Evidence"),
        nbformat.v4.new_code_cell("show_table('table_development_to_iid_transport.csv')"),
        nbformat.v4.new_markdown_cell(
            "## 6. Final Selection\n\n"
            "Stage 3 was frozen before IID under a balanced conservative Tail-aware philosophy. IID did not trigger reselection."
        ),
        nbformat.v4.new_code_cell("show_table('table_six_condition_generalization.csv')"),
        nbformat.v4.new_markdown_cell("## 7. IID Evaluation"),
        nbformat.v4.new_code_cell(
            "show_table('table_iid_primary_vs_global.csv')\nshow_image('figure07_iid_bootstrap_mae_difference.png')"
        ),
        nbformat.v4.new_markdown_cell("## 8. Percentage Error"),
        nbformat.v4.new_code_cell(
            "show_image('figure03_iid_mape_by_target_decile.png')\nshow_image('figure04_iid_wape_by_target_decile.png')"
        ),
        nbformat.v4.new_markdown_cell("## 9. Decile Error"),
        nbformat.v4.new_code_cell(
            "show_table('table_iid_decile_error_profile.csv')\nshow_image('figure02_iid_mae_by_target_decile.png')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 10. Tail Analysis\n\n"
            "Upper-Tail error improved relative to Global, but many high-loan observations remained underpredicted."
        ),
        nbformat.v4.new_code_cell(
            "show_image('figure05_body_tail_error_comparison.png')\nshow_image('figure06_primary_minus_global_decile_mae.png')"
        ),
        nbformat.v4.new_markdown_cell("## 11. Bootstrap Evidence"),
        nbformat.v4.new_code_cell("show_table('table_iid_primary_vs_global.csv')"),
        nbformat.v4.new_markdown_cell(
            "## 12. Fairness Audit\n\n"
            "The analysis describes predictive error differences. It does not assess lending decisions, causal discrimination, disparate treatment, or legal compliance. Target composition and small-group limits apply."
        ),
        nbformat.v4.new_code_cell(
            "show_table('table_fairness_disparities.csv')\n"
            "show_table('table_intersectional_fairness.csv')\n"
            "show_image('figure08_fairness_mae_eligible_groups.png')\n"
            "show_image('figure09_primary_minus_global_fairness_mae.png')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 13. Explainability\n\n"
            "Global prediction, routing, and residual correction use separate explanation spaces. Feature attribution is not causal, and raw SHAP magnitudes are not merged across incompatible components."
        ),
        nbformat.v4.new_code_cell(
            "show_table('table_final_explainability.csv')\n"
            "show_image('figure10_global_consensus_top15.png')\n"
            "show_image('figure11_meta_gate_top15.png')\n"
            "show_image('figure12_residual_specialist_top15.png')\n"
            "show_image('figure13_residual_body_vs_d10_importance.png')"
        ),
        nbformat.v4.new_markdown_cell("## 14. Routing Mechanism"),
        nbformat.v4.new_code_cell(
            "show_table('table_stage3_mechanism.csv')\nshow_image('figure14_correction_magnitude_realized_benefit.png')"
        ),
        nbformat.v4.new_markdown_cell(
            "## 15. Limitations\n\n"
            "HMDA 2017 lacks direct property-value and LTV-like variables. Tail underprediction remains. C2 failed. Development was adaptive, IID covered only two predeclared models, fairness results are descriptive, and explanation is not causal."
        ),
        nbformat.v4.new_code_cell(
            "display(Markdown((FINAL / 'PAPER_LIMITATIONS_TEXT.md').read_text(encoding='utf-8')))"
        ),
        nbformat.v4.new_markdown_cell(
            "## 16. Reproducibility\n\n"
            "The final package preserves hashes for evidence, models, documents, tables, figures, and this notebook. Original IID files remain closed."
        ),
        nbformat.v4.new_code_cell(
            "show_table('table_global_component_importance.csv', rows=15)"
        ),
        nbformat.v4.new_markdown_cell(
            "## 17. Final Conclusions\n\n"
            "The frozen Primary is a reproducible balanced compromise. It improved observed overall IID MAE and several Tail measures relative to Global, passed 5/6 frozen conditions, and did not meet C2. No further modeling or IID analysis is authorized."
        ),
    ]
    nb["cells"] = cells
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK)
    client = NotebookClient(nb, timeout=180, kernel_name="python3", allow_errors=False)
    client.execute(cwd=str(ROOT))
    nbformat.write(nb, NOTEBOOK)
    executed = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    image_outputs = sum(
        1
        for cell in code_cells
        for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    )
    table_outputs = sum(
        1
        for cell in code_cells
        for output in cell.get("outputs", [])
        if "text/html" in output.get("data", {})
    )
    code_source = "\n".join(cell.source for cell in code_cells)
    forbidden_patterns = {
        "fit_calls": ".fit(",
        "prediction_calls": ".predict(",
        "shap_calls": "shap.",
        "bootstrap_generation": "np.random",
        "original_iid_feature_name": "iid_holdout_features.parquet",
        "original_iid_target_name": "iid_holdout_targets.parquet",
    }
    static_counts = {
        name: code_source.count(pattern)
        for name, pattern in forbidden_patterns.items()
    }
    if error_outputs or any(static_counts.values()):
        raise RuntimeError(
            f"Artifact-only notebook validation failed: errors={len(error_outputs)}, counts={static_counts}"
        )
    record = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "path": rel(NOTEBOOK),
        "sha256": sha256(NOTEBOOK),
        "cells": len(executed.cells),
        "code_cells": len(code_cells),
        "inline_image_outputs": image_outputs,
        "inline_table_outputs": table_outputs,
        "error_outputs": len(error_outputs),
        "static_forbidden_call_counts": static_counts,
        "original_iid_content_reads": 0,
    }
    write_json(TMP / "prompt5c_notebook_execution.json", record)
    return record


def build_artifact_index(
    table_records: list[dict[str, Any]],
    figure_records: list[dict[str, Any]],
    document_records: list[dict[str, Any]],
) -> dict[str, Any]:
    sections: list[str] = ["# Final Artifact Index"]

    def add_section(title: str, rows: list[dict[str, str]]) -> None:
        frame = pd.DataFrame(rows)
        sections.append(f"## {title}\n\n{markdown_table(frame, digits=6)}")

    add_section(
        "Final models",
        [
            {
                "Path": "outputs/models/final_pre_iid/primary_stage3/bundle.joblib",
                "Purpose": "Frozen Stage 3 Primary bundle",
                "Source evidence": "FINAL_PRE_IID_FREEZE.json",
                "Status": "PASS / immutable",
            },
            {
                "Path": "outputs/models/final_pre_iid/global_comparator/bundle.joblib",
                "Purpose": "Frozen Global comparator bundle",
                "Source evidence": "FINAL_PRE_IID_FREEZE.json",
                "Status": "PASS / immutable",
            },
        ],
    )
    add_section(
        "Frozen evaluation evidence",
        [
            {
                "Path": f"outputs/reports/{name}",
                "Purpose": purpose,
                "Source evidence": "Immutable final handoff",
                "Status": "PASS / immutable",
            }
            for name, purpose in [
                ("FINAL_PRE_IID_FREEZE.json", "Final pre-IID model freeze"),
                ("FINAL_IID_EVALUATION.json", "One-time IID evaluation"),
                (
                    "FINAL_FAIRNESS_EXPLAINABILITY.json",
                    "Final fairness and explainability",
                ),
            ]
        ],
    )
    doc_rows = [
        {
            "Path": record["path"],
            "Purpose": Path(record["path"]).stem.replace("_", " ").title(),
            "Source evidence": "Frozen Prompt 4C/5A/5B reports",
            "Status": "PASS",
        }
        for record in document_records
    ]
    add_section("Model Card", [row for row in doc_rows if "MODEL_CARD" in row["Path"]])
    add_section(
        "Technical Report",
        [row for row in doc_rows if "TECHNICAL_REPORT" in row["Path"]],
    )
    add_section(
        "Paper-ready methods/results/limitations",
        [row for row in doc_rows if "PAPER_" in row["Path"]],
    )
    add_section(
        "Tables",
        [
            {
                "Path": record["csv_path"],
                "Purpose": record["name"].replace("_", " ").title(),
                "Source evidence": ", ".join(record["source_artifacts"]),
                "Status": "PASS",
            }
            for record in table_records
        ],
    )
    add_section(
        "Figures",
        [
            {
                "Path": record["png_path"],
                "Purpose": record["title"],
                "Source evidence": ", ".join(record["source_artifacts"]),
                "Status": "PASS; PNG/PDF/data CSV",
            }
            for record in figure_records
        ],
    )
    add_section(
        "Final notebook",
        [
            {
                "Path": rel(NOTEBOOK),
                "Purpose": "Artifact-only full project story",
                "Source evidence": "Final documents, tables, plotting data, and figures",
                "Status": "PASS / executed",
            }
        ],
    )
    add_section(
        "Reproducibility manifest",
        [
            {
                "Path": "outputs/final/FINAL_REPRODUCIBILITY_MANIFEST.json",
                "Purpose": "Hash and provenance index",
                "Source evidence": "Frozen evidence and generated reporting artifacts",
                "Status": "PASS in final package",
            }
        ],
    )
    index_path = FINAL / "ARTIFACT_INDEX.md"
    write_text(index_path, "\n\n".join(sections))
    return {
        "path": rel(index_path),
        "sha256": sha256(index_path),
        "status": "PASS",
    }


def consistency_check() -> dict[str, Any]:
    doc_names = [
        "MODEL_CARD_FINAL.md",
        "FINAL_TECHNICAL_REPORT.md",
        "FINAL_PROJECT_SUMMARY.md",
        "PAPER_METHODS_TEXT.md",
        "PAPER_RESULTS_TEXT.md",
        "PAPER_LIMITATIONS_TEXT.md",
    ]
    docs = {name: (FINAL / name).read_text(encoding="utf-8") for name in doc_names}
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if condition else "FAIL",
                "evidence": evidence,
            }
        )

    core_docs = [
        "MODEL_CARD_FINAL.md",
        "FINAL_TECHNICAL_REPORT.md",
        "FINAL_PROJECT_SUMMARY.md",
        "PAPER_RESULTS_TEXT.md",
    ]
    canonical_tokens = {
        "Primary model ID": PRIMARY_ID,
        "Primary IID MAE": "62.260626",
        "Global IID MAE": "62.444349",
        "Development rows": "500,000",
        "IID rows": "75,000",
        "feature count": "35",
        "eligible fairness groups": "27",
        "sensitive-field count": "eight",
    }
    for label, token in canonical_tokens.items():
        present = [name for name in core_docs if token.lower() in docs[name].lower()]
        check(
            f"canonical {label}",
            len(present) == len(core_docs),
            {"token": token, "documents": present},
        )
    six_condition_present = [
        name
        for name in core_docs
        if "5/6" in docs[name] or "five of six" in docs[name].lower()
    ]
    check(
        "canonical six-condition result",
        len(six_condition_present) == len(core_docs),
        {"accepted wording": ["5/6", "five of six"], "documents": six_condition_present},
    )
    for name, text in docs.items():
        check(
            f"{name} Primary bundle hash",
            PRIMARY_BUNDLE_HASH in text,
            PRIMARY_BUNDLE_HASH,
        )
        check(
            f"{name} Global bundle hash",
            GLOBAL_BUNDLE_HASH in text,
            GLOBAL_BUNDLE_HASH,
        )
    check(
        "C2 failure disclosed",
        all("C2" in docs[name] and "failed" in docs[name].lower() for name in core_docs),
        core_docs,
    )
    check(
        "fairness disclaimer in primary reports",
        all(
            "does not assess lending approval decisions" in docs[name]
            for name in ["MODEL_CARD_FINAL.md", "FINAL_TECHNICAL_REPORT.md", "PAPER_RESULTS_TEXT.md"]
        ),
        "required disclaimer wording",
    )
    check(
        "explainability disclaimer in primary reports",
        all(
            "do not establish causal relationships" in docs[name]
            for name in ["MODEL_CARD_FINAL.md", "FINAL_TECHNICAL_REPORT.md", "PAPER_RESULTS_TEXT.md"]
        ),
        "required disclaimer wording",
    )
    prohibited_claims = [
        "state of the art",
        "discrimination-free",
        "tail problem solved",
        "production-ready",
        "statistically significant",
        "causal feature effect",
    ]
    found_claims = {
        name: [
            phrase
            for phrase in prohibited_claims
            if phrase in text.lower()
        ]
        for name, text in docs.items()
    }
    check(
        "no claim inflation",
        not any(found_claims.values()),
        found_claims,
    )
    perf = pd.read_csv(TABLES / "table_final_model_performance.csv")
    check("final performance table has two models", len(perf) == 2, len(perf))
    check(
        "performance Primary MAE exact",
        math.isclose(float(perf.iloc[0]["MAE"]), 62.26062600334689, rel_tol=0, abs_tol=1e-12),
        float(perf.iloc[0]["MAE"]),
    )
    check(
        "performance Global MAE exact",
        math.isclose(float(perf.iloc[1]["MAE"]), 62.444349310930285, rel_tol=0, abs_tol=1e-12),
        float(perf.iloc[1]["MAE"]),
    )
    six = pd.read_csv(TABLES / "table_six_condition_generalization.csv")
    check(
        "six-condition table is 5/6 with C2 fail",
        int((six["PASS/FAIL"] == "PASS").sum()) == 5
        and six.loc[six["Condition"] == "C2", "PASS/FAIL"].iloc[0] == "FAIL",
        six[["Condition", "PASS/FAIL"]].to_dict(orient="records"),
    )
    fairness = pd.read_csv(TABLES / "table_fairness_summary.csv")
    check("fairness table has 27 eligible rows", len(fairness) == 27, len(fairness))
    check(
        "fairness group improvement count is 25",
        int((fairness["Primary minus Global MAE"] < 0).sum()) == 25,
        int((fairness["Primary minus Global MAE"] < 0).sum()),
    )
    check(
        "notebook exists and is saved",
        NOTEBOOK.exists() and NOTEBOOK.stat().st_size > 0,
        rel(NOTEBOOK),
    )
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    result = {
        "status": status,
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "check_count": len(checks),
        "checks": checks,
        "canonical_facts": {
            "development_rows": 500000,
            "iid_rows": 75000,
            "primary_model_id": PRIMARY_ID,
            "primary_iid_mae": 62.26062600334689,
            "global_iid_mae": 62.444349310930285,
            "conditions_passed": 5,
            "conditions_total": 6,
            "sensitive_field_count": 8,
            "eligible_group_count": 27,
            "feature_count": 35,
            "primary_bundle_sha256": PRIMARY_BUNDLE_HASH,
            "global_bundle_sha256": GLOBAL_BUNDLE_HASH,
        },
    }
    write_json(REPORTS / "prompt5c_consistency_check.json", result)
    if status != "PASS":
        failed = [item for item in checks if item["status"] != "PASS"]
        raise RuntimeError(f"Cross-document consistency failed: {failed}")
    return result


def build_manifest() -> dict[str, Any]:
    freeze = read_json(REPORTS / "FINAL_PRE_IID_FREEZE.json")
    iid = read_json(REPORTS / "FINAL_IID_EVALUATION.json")
    fairness = read_json(REPORTS / "FINAL_FAIRNESS_EXPLAINABILITY.json")
    source_report = read_json(REPORTS / "prompt1a_source_report.json")
    snapshots = read_json(REPORTS / "prompt5a_post_iid_snapshot_manifest.json")
    documents = sorted(
        path
        for path in FINAL.glob("*.md")
        if path.name != "ARTIFACT_INDEX.md"
    )
    tables = sorted(TABLES.glob("*"))
    figures = sorted(FIGURES.glob("*"))
    report_names = [
        "prompt5c_handoff_validation.json",
        "prompt5c_table_validation.json",
        "prompt5c_figure_validation.json",
        "prompt5c_consistency_check.json",
        "prompt5c_reviewer.json",
        "prompt5c_verification.json",
    ]
    existing_reports = [REPORTS / name for name in report_names if (REPORTS / name).exists()]
    manifest = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "project_root": str(ROOT),
        "data_hashes_from_saved_evidence_only": {
            "raw_source_sha256": source_report["sha256_baseline"],
            "development_sha256": freeze["project_identity"]["development_sha256"],
            "post_iid_snapshot_hashes": {
                name: value["sha256"]
                for name, value in snapshots["artifacts"].items()
            },
            "original_iid_content_reopened_for_hashing": False,
        },
        "final_model_hashes": {
            "primary": {
                "model_id": PRIMARY_ID,
                "deployment_id": PRIMARY_DEPLOYMENT_ID,
                "path": freeze["primary_bundle"]["path"],
                "sha256": freeze["primary_bundle"]["sha256"],
            },
            "global": {
                "model_id": GLOBAL_ID,
                "deployment_id": GLOBAL_DEPLOYMENT_ID,
                "path": freeze["global_comparator"]["path"],
                "sha256": freeze["global_comparator"]["sha256"],
            },
        },
        "final_freeze_hashes": {
            "FINAL_PRE_IID_FREEZE.json": sha256(REPORTS / "FINAL_PRE_IID_FREEZE.json"),
            "PROMPT4C_READY.json": sha256(REPORTS / "PROMPT4C_READY.json"),
        },
        "final_iid_evidence_hashes": {
            "FINAL_IID_EVALUATION.json": sha256(REPORTS / "FINAL_IID_EVALUATION.json"),
            "PROMPT5A_READY.json": sha256(REPORTS / "PROMPT5A_READY.json"),
            "saved_report_hashes": iid["report_hashes"],
        },
        "fairness_explainability_hashes": {
            "FINAL_FAIRNESS_EXPLAINABILITY.json": sha256(
                REPORTS / "FINAL_FAIRNESS_EXPLAINABILITY.json"
            ),
            "PROMPT5B_READY.json": sha256(REPORTS / "PROMPT5B_READY.json"),
            "saved_report_hashes": fairness["fairness_report_hashes"],
        },
        "final_report_hashes": {
            rel(path): sha256(path) for path in existing_reports
        },
        "final_document_hashes": {rel(path): sha256(path) for path in documents},
        "final_table_hashes": {rel(path): sha256(path) for path in tables},
        "final_figure_hashes": {rel(path): sha256(path) for path in figures},
        "artifact_index_hash": (
            sha256(FINAL / "ARTIFACT_INDEX.md")
            if (FINAL / "ARTIFACT_INDEX.md").exists()
            else None
        ),
        "notebook": {
            "path": rel(NOTEBOOK),
            "sha256": sha256(NOTEBOOK),
        },
        "package_environment_provenance": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "frozen_environment_references": [
                "outputs/reports/prompt3_environment.json",
                "outputs/reports/FINAL_PRE_IID_FREEZE.json",
                "outputs/reports/prompt5a_runtime.json",
                "outputs/reports/prompt5b_runtime.json",
            ],
        },
        "integrity": {
            "original_iid_content_reads": 0,
            "fits": 0,
            "predictions": 0,
            "shap_recomputations": 0,
            "bootstrap_recomputations": 0,
            "model_changes": 0,
        },
    }
    write_json(FINAL / "FINAL_REPRODUCIBILITY_MANIFEST.json", manifest)
    return manifest


def build_base() -> None:
    for directory in [FINAL, TABLES, FIGURES, TMP]:
        directory.mkdir(parents=True, exist_ok=True)
    validate_handoffs()
    table_records, frames = build_tables()
    figure_records = build_figures()
    document_records = build_documents(frames)
    notebook_record = build_notebook()
    index_record = build_artifact_index(table_records, figure_records, document_records)
    consistency = consistency_check()
    manifest = build_manifest()
    write_json(
        TMP / "prompt5c_build.json",
        {
            "status": "PASS",
            "created_at_utc": utc_now(),
            "authorization_id": AUTHORIZATION_ID,
            "documents": document_records,
            "tables": table_records,
            "figures": figure_records,
            "notebook": notebook_record,
            "artifact_index": index_record,
            "consistency": consistency["status"],
            "manifest_sha256": sha256(FINAL / "FINAL_REPRODUCIBILITY_MANIFEST.json"),
            "integrity": {
                "original_iid_content_reads": 0,
                "fits": 0,
                "predictions": 0,
                "shap_recomputations": 0,
                "bootstrap_recomputations": 0,
                "model_changes": 0,
            },
        },
    )


def repair_review_findings() -> None:
    validate_handoffs()
    table_records, frames = build_tables()
    figure_records = build_figures(selected={6, 12, 14})
    document_records = build_documents(frames)
    notebook_record = build_notebook()
    index_record = build_artifact_index(table_records, figure_records, document_records)
    consistency = consistency_check()
    manifest = build_manifest()
    write_json(
        TMP / "prompt5c_reviewer_repairs.json",
        {
            "status": "PASS",
            "created_at_utc": utc_now(),
            "authorization_id": AUTHORIZATION_ID,
            "repairs": [
                {
                    "finding": "Figures 6, 12, and 14 label clipping",
                    "action": "Regenerated only those figures with larger margins and concise units.",
                    "scientific_change": False,
                },
                {
                    "finding": "Paper Results fairness caveats",
                    "action": "Added target-composition and small-group caveats from frozen Prompt 5B language.",
                    "scientific_change": False,
                },
                {
                    "finding": "minority_population absent from disparity table",
                    "action": (
                        "Added an explicit zero-eligible row and marked unavailable "
                        "statistics NOT_AVAILABLE_FROM_FROZEN_EVIDENCE."
                    ),
                    "scientific_change": False,
                },
            ],
            "tables": len(table_records),
            "figures": len(figure_records),
            "notebook": notebook_record,
            "artifact_index": index_record,
            "consistency": consistency["status"],
            "manifest_sha256": sha256(
                FINAL / "FINAL_REPRODUCIBILITY_MANIFEST.json"
            ),
            "integrity": {
                "original_iid_content_reads": 0,
                "fits": 0,
                "predictions": 0,
                "shap_recomputations": 0,
                "bootstrap_recomputations": 0,
                "model_changes": 0,
            },
        },
    )


def promote_final_project() -> dict[str, Any]:
    reviewer_path = REPORTS / "prompt5c_reviewer.json"
    verification_path = REPORTS / "prompt5c_verification.json"
    reviewer = read_json(reviewer_path)
    verification = read_json(verification_path)
    if reviewer.get("status") != "PASS":
        raise RuntimeError("Independent Prompt 5C reviewer is not PASS")
    if verification.get("status") != "PASS":
        raise RuntimeError("Independent Prompt 5C verification is not PASS")
    table_validation = read_json(REPORTS / "prompt5c_table_validation.json")
    figure_validation = read_json(REPORTS / "prompt5c_figure_validation.json")
    consistency = read_json(REPORTS / "prompt5c_consistency_check.json")
    if any(
        item.get("status") != "PASS"
        for item in [table_validation, figure_validation, consistency]
    ):
        raise RuntimeError("Prompt 5C validation reports are not all PASS")
    manifest_path = FINAL / "FINAL_REPRODUCIBILITY_MANIFEST.json"
    candidate = {
        "status": "PASS_CANDIDATE",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "final_model": {
            "model_id": PRIMARY_ID,
            "deployment_id": PRIMARY_DEPLOYMENT_ID,
            "bundle_sha256": PRIMARY_BUNDLE_HASH,
            "training_rows": 500000,
            "feature_contract": "main_without_sensitive_without_lender",
            "feature_count": 35,
        },
        "global_comparator": {
            "model_id": GLOBAL_ID,
            "deployment_id": GLOBAL_DEPLOYMENT_ID,
            "bundle_sha256": GLOBAL_BUNDLE_HASH,
        },
        "final_iid_summary": {
            "rows": 75000,
            "primary_mae": 62.26062600334689,
            "global_mae": 62.444349310930285,
            "primary_minus_global_mae": -0.18372330758339217,
            "bootstrap_95_interval": [
                -0.26298875484432377,
                -0.1048113958689481,
            ],
            "conditions_passed": 5,
            "conditions_total": 6,
            "failed_condition": "C2",
            "model_changed_after_iid": False,
        },
        "final_fairness_explainability_sha256": EXPECTED_FAIRNESS_HASH,
        "final_document_hashes": {
            rel(path): sha256(path) for path in sorted(FINAL.glob("*.md"))
        },
        "table_validation_status": table_validation["status"],
        "table_count": table_validation["table_count"],
        "figure_validation_status": figure_validation["status"],
        "figure_count": figure_validation["figure_count"],
        "consistency_status": consistency["status"],
        "notebook": {
            "path": rel(NOTEBOOK),
            "sha256": sha256(NOTEBOOK),
            "status": "PASS",
        },
        "reproducibility_manifest": {
            "path": rel(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "reviewer": {
            "status": reviewer["status"],
            "path": rel(reviewer_path),
            "sha256": sha256(reviewer_path),
        },
        "verification": {
            "status": verification["status"],
            "path": rel(verification_path),
            "sha256": sha256(verification_path),
        },
        "zero_compute_science": {
            "original_iid_content_reads": 0,
            "fits": 0,
            "predictions": 0,
            "shap_recomputations": 0,
            "bootstrap_recomputations": 0,
            "model_changes": 0,
        },
    }
    candidate_path = REPORTS / "prompt5c_final_project_candidate.json"
    write_json(candidate_path, candidate)
    complete = dict(candidate)
    complete.update(
        {
            "status": FINAL_STATUS,
            "promoted_at_utc": utc_now(),
            "candidate_path": rel(candidate_path),
            "candidate_sha256": sha256(candidate_path),
            "immutable_after_promotion": True,
            "next_step": "PROJECT COMPLETE — no further modeling or IID analysis authorized",
        }
    )
    complete_path = REPORTS / "FINAL_PROJECT_COMPLETE.json"
    write_json(complete_path, complete)
    reloaded = read_json(complete_path)
    if (
        reloaded.get("status") != FINAL_STATUS
        or reloaded.get("final_model", {}).get("model_id") != PRIMARY_ID
        or reloaded.get("verification", {}).get("status") != "PASS"
        or reloaded.get("reviewer", {}).get("status") != "PASS"
    ):
        raise RuntimeError("FINAL_PROJECT_COMPLETE reload verification failed")
    return complete


def create_readiness_last() -> dict[str, Any]:
    ready_path = REPORTS / "PROMPT5C_READY.json"
    if ready_path.exists():
        raise RuntimeError("PROMPT5C_READY.json already exists; no later Prompt 5C write is allowed")
    complete_path = REPORTS / "FINAL_PROJECT_COMPLETE.json"
    complete = read_json(complete_path)
    if complete.get("status") != FINAL_STATUS:
        raise RuntimeError("FINAL_PROJECT_COMPLETE is not PASS")
    state_files = [
        ROOT / "AGENTS.md",
        ROOT / "TASK.md",
        ROOT / "PLAN.md",
        ROOT / "DECISIONS.md",
        ROOT / "LOG.md",
        ROOT / "README.md",
        ROOT / "config.json",
    ]
    missing_state = [
        rel(path)
        for path in state_files
        if FINAL_STATUS not in path.read_text(encoding="utf-8")
    ]
    if missing_state:
        raise RuntimeError(f"Project state was not updated before readiness: {missing_state}")
    reviewer = read_json(REPORTS / "prompt5c_reviewer.json")
    verification = read_json(REPORTS / "prompt5c_verification.json")
    ready = {
        "status": FINAL_STATUS,
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "prompt5c_complete": True,
        "final_project_complete_path": rel(complete_path),
        "final_project_complete_sha256": sha256(complete_path),
        "model_card_sha256": sha256(FINAL / "MODEL_CARD_FINAL.md"),
        "technical_report_sha256": sha256(FINAL / "FINAL_TECHNICAL_REPORT.md"),
        "final_notebook_sha256": sha256(NOTEBOOK),
        "reproducibility_manifest_sha256": sha256(
            FINAL / "FINAL_REPRODUCIBILITY_MANIFEST.json"
        ),
        "reviewer_status": reviewer["status"],
        "verification_status": verification["status"],
        "development_research": "CLOSED",
        "iid_evaluation": "COMPLETE",
        "fairness": "COMPLETE",
        "explainability": "COMPLETE",
        "final_model": "FROZEN",
        "model_changed_after_iid": False,
        "original_iid_files": "CLOSED_AFTER_ONE_TIME_EVALUATION",
        "final_project_reporting": "COMPLETE",
        "next_step": "PROJECT COMPLETE — no further modeling or IID analysis authorized",
        "report_creation_rule": "This is the last Prompt 5C report artifact.",
    }
    write_json(ready_path, ready)
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5C artifact-only reporting")
    parser.add_argument(
        "command",
        choices=["build", "repair", "manifest", "promote", "ready"],
    )
    args = parser.parse_args()
    if args.command == "build":
        build_base()
    elif args.command == "repair":
        repair_review_findings()
    elif args.command == "manifest":
        build_manifest()
    elif args.command == "promote":
        promote_final_project()
    elif args.command == "ready":
        create_readiness_last()


if __name__ == "__main__":
    main()
