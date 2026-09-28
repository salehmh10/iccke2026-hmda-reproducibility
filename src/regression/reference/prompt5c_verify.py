from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs" / "reports"
FINAL = ROOT / "outputs" / "final"
TABLES = FINAL / "tables"
FIGURES = FINAL / "figures"
NOTEBOOK = ROOT / "notebooks" / "05C_FINAL_PROJECT_REPORTING.ipynb"
OUTPUT = REPORTS / "prompt5c_verification.json"
AUTHORIZATION = "regression_v2_prompt5c_final_reporting"
EXPECTED_FAIRNESS = "2025ec95cc564238e72fbb82ac33670c0272f166cf93d46dc1aa7a59f1b890df"
EXPECTED_FREEZE = "8f2b8e4fda80056770b916b7859ad0f6f89f2948236320e6815e082fadca35c1"
PRIMARY_HASH = "5349ab15fd1c8182ef539f435cc4f065e71ee7da047de09a4dc271c9097e0c08"
GLOBAL_HASH = "6f61a0be1fc90d2331b08dada63a453f6c5410d5783f46f1e0cbcbf425f75597"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


checks: list[dict[str, Any]] = []


def check(name: str, condition: bool, evidence: Any) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if condition else "FAIL",
            "evidence": evidence,
        }
    )


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)


def main() -> None:
    immutable_statuses = {
        "PROMPT5B_READY.json": "PASS_FINAL_FAIRNESS_AND_EXPLAINABILITY",
        "FINAL_FAIRNESS_EXPLAINABILITY.json": "PASS_FINAL_FAIRNESS_AND_EXPLAINABILITY",
        "PROMPT5A_READY.json": "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS",
        "FINAL_IID_EVALUATION.json": "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS",
        "PROMPT4C_READY.json": "PASS_FINAL_PRE_IID_FREEZE",
        "FINAL_PRE_IID_FREEZE.json": "PASS_FINAL_PRE_IID_FREEZE",
    }
    for name, expected in immutable_statuses.items():
        path = REPORTS / name
        value = read_json(path) if path.exists() else {}
        check(
            f"immutable status {name}",
            path.exists() and value.get("status") == expected,
            value.get("status"),
        )
    check(
        "exact final fairness/explainability hash",
        digest(REPORTS / "FINAL_FAIRNESS_EXPLAINABILITY.json") == EXPECTED_FAIRNESS,
        digest(REPORTS / "FINAL_FAIRNESS_EXPLAINABILITY.json"),
    )
    check(
        "exact final pre-IID freeze hash",
        digest(REPORTS / "FINAL_PRE_IID_FREEZE.json") == EXPECTED_FREEZE,
        digest(REPORTS / "FINAL_PRE_IID_FREEZE.json"),
    )

    freeze = read_json(REPORTS / "FINAL_PRE_IID_FREEZE.json")
    selection_freeze = read_json(REPORTS / "prompt4c_final_selection_freeze.json")
    iid = read_json(REPORTS / "FINAL_IID_EVALUATION.json")
    fairness_final = read_json(REPORTS / "FINAL_FAIRNESS_EXPLAINABILITY.json")
    check(
        "final Primary ID unchanged",
        freeze["final_selection"]["historical_recipe_id"]
        == "stage3_residual_t75_a75"
        and selection_freeze["final_primary_recipe"] == "stage3_residual_t75_a75",
        freeze["final_selection"]["historical_recipe_id"],
    )
    check(
        "final Global comparator ID unchanged",
        freeze["final_selection"]["simple_comparator"] == "final_global_500k"
        and selection_freeze["simple_comparator"] == "ens_boost_cat060",
        {
            "deployment": freeze["final_selection"]["simple_comparator"],
            "historical_recipe": selection_freeze["simple_comparator"],
        },
    )
    check(
        "Primary bundle hash unchanged",
        freeze["primary_bundle"]["sha256"] == PRIMARY_HASH,
        freeze["primary_bundle"]["sha256"],
    )
    check(
        "Global bundle hash unchanged",
        freeze["global_comparator"]["sha256"] == GLOBAL_HASH,
        freeze["global_comparator"]["sha256"],
    )
    check(
        "final architecture weights",
        freeze["primary_recipe"]["global_weights"]
        == {"catboost": 0.6, "lightgbm": 0.2, "xgboost": 0.2},
        freeze["primary_recipe"]["global_weights"],
    )
    check(
        "final routing threshold and alpha",
        close(freeze["primary_recipe"]["routing_threshold"], 0.75)
        and close(freeze["primary_recipe"]["correction_alpha"], 0.75),
        {
            "threshold": freeze["primary_recipe"]["routing_threshold"],
            "alpha": freeze["primary_recipe"]["correction_alpha"],
        },
    )

    required_documents = [
        "MODEL_CARD_FINAL.md",
        "FINAL_TECHNICAL_REPORT.md",
        "FINAL_PROJECT_SUMMARY.md",
        "PAPER_METHODS_TEXT.md",
        "PAPER_RESULTS_TEXT.md",
        "PAPER_LIMITATIONS_TEXT.md",
        "ARTIFACT_INDEX.md",
    ]
    check(
        "all required documents exist",
        all((FINAL / name).exists() for name in required_documents),
        required_documents,
    )
    required_tables = [
        "table_final_model_performance",
        "table_iid_primary_vs_global",
        "table_six_condition_generalization",
        "table_development_to_iid_transport",
        "table_iid_decile_error_profile",
        "table_fairness_summary",
        "table_fairness_disparities",
        "table_intersectional_fairness",
        "table_final_explainability",
        "table_global_component_importance",
        "table_stage3_mechanism",
    ]
    check(
        "all required table CSV files exist",
        all((TABLES / f"{name}.csv").exists() for name in required_tables),
        required_tables,
    )
    check(
        "all required table Markdown files exist",
        all((TABLES / f"{name}.md").exists() for name in required_tables),
        required_tables,
    )
    figure_stems = [
        "figure01_iid_overall_metrics",
        "figure02_iid_mae_by_target_decile",
        "figure03_iid_mape_by_target_decile",
        "figure04_iid_wape_by_target_decile",
        "figure05_body_tail_error_comparison",
        "figure06_primary_minus_global_decile_mae",
        "figure07_iid_bootstrap_mae_difference",
        "figure08_fairness_mae_eligible_groups",
        "figure09_primary_minus_global_fairness_mae",
        "figure10_global_consensus_top15",
        "figure11_meta_gate_top15",
        "figure12_residual_specialist_top15",
        "figure13_residual_body_vs_d10_importance",
        "figure14_correction_magnitude_realized_benefit",
        "figure15_project_architecture",
    ]
    for suffix in [".png", ".pdf", "_data.csv"]:
        check(
            f"all 15 final figure {suffix} files exist",
            all((FIGURES / f"{stem}{suffix}").exists() for stem in figure_stems),
            len([stem for stem in figure_stems if (FIGURES / f"{stem}{suffix}").exists()]),
        )

    handoff = read_json(REPORTS / "prompt5c_handoff_validation.json")
    table_validation = read_json(REPORTS / "prompt5c_table_validation.json")
    figure_validation = read_json(REPORTS / "prompt5c_figure_validation.json")
    consistency = read_json(REPORTS / "prompt5c_consistency_check.json")
    reviewer = read_json(REPORTS / "prompt5c_reviewer.json")
    check("Prompt 5C handoff validation PASS", handoff.get("status") == "PASS", handoff.get("status"))
    check("Prompt 5C table validation PASS", table_validation.get("status") == "PASS", table_validation.get("status"))
    check("Prompt 5C figure validation PASS", figure_validation.get("status") == "PASS", figure_validation.get("status"))
    check("Prompt 5C consistency PASS", consistency.get("status") == "PASS", consistency.get("status"))
    check(
        "exactly one independent reviewer PASS",
        reviewer.get("status") == "PASS"
        and reviewer.get("independent_read_only") is True
        and reviewer.get("reviewer_cycles_completed") == 1,
        {
            "status": reviewer.get("status"),
            "independent_read_only": reviewer.get("independent_read_only"),
            "reviewer_cycles_completed": reviewer.get("reviewer_cycles_completed"),
        },
    )

    source_overall = pd.read_csv(REPORTS / "prompt5a_iid_overall_metrics.csv")
    final_performance = pd.read_csv(TABLES / "table_final_model_performance.csv")
    primary_source = source_overall[
        source_overall["model_id"] == "final_primary_stage3_500k"
    ].iloc[0]
    global_source = source_overall[
        source_overall["model_id"] == "final_global_500k"
    ].iloc[0]
    check(
        "Primary IID MAE exact",
        close(final_performance.iloc[0]["MAE"], primary_source["mae"]),
        final_performance.iloc[0]["MAE"],
    )
    check(
        "Global IID MAE exact",
        close(final_performance.iloc[1]["MAE"], global_source["mae"]),
        final_performance.iloc[1]["MAE"],
    )
    metric_pairs = {
        "RMSE": "rmse",
        "R²": "r2",
        "RMSLE": "rmsle",
        "Median AE": "median_absolute_error",
        "P90 AE": "p90_absolute_error",
        "MAPE %": "mape_percent",
        "WAPE %": "wape_percent",
        "Bottom-90 MAE": "bottom_90_mae",
        "Top-decile MAE": "top_decile_mae",
        "Top-5% MAE": "top_five_percent_mae",
        "P85-P95 MAE": "p85_to_p95_boundary_mae",
        "Top-decile signed error": "top_decile_signed_error",
        "Top-decile underprediction rate": "top_decile_underprediction_rate",
    }
    exact_primary_metrics = all(
        close(final_performance.iloc[0][target], primary_source[source])
        for target, source in metric_pairs.items()
    )
    exact_global_metrics = all(
        close(final_performance.iloc[1][target], global_source[source])
        for target, source in metric_pairs.items()
    )
    check("all Primary main table metrics exact", exact_primary_metrics, list(metric_pairs))
    check("all Global main table metrics exact", exact_global_metrics, list(metric_pairs))

    source_bootstrap = pd.read_csv(REPORTS / "prompt5a_iid_bootstrap.csv")
    final_bootstrap = pd.read_csv(TABLES / "table_iid_primary_vs_global.csv")
    check(
        "IID comparison has exact seven metrics",
        final_bootstrap["Metric"].tolist() == source_bootstrap["metric"].tolist(),
        final_bootstrap["Metric"].tolist(),
    )
    check(
        "IID comparison observed differences exact",
        all(
            close(a, b)
            for a, b in zip(
                final_bootstrap["Observed Primary minus Global"],
                source_bootstrap["observed_difference_primary_minus_global"],
            )
        ),
        "seven metrics",
    )
    check(
        "IID comparison intervals exact",
        all(
            close(a, b)
            for a, b in zip(
                final_bootstrap["95% CI lower"], source_bootstrap["percentile_2_5"]
            )
        )
        and all(
            close(a, b)
            for a, b in zip(
                final_bootstrap["95% CI upper"], source_bootstrap["percentile_97_5"]
            )
        ),
        "seven intervals",
    )

    six = pd.read_csv(TABLES / "table_six_condition_generalization.csv")
    check(
        "six-condition result is 5/6",
        int((six["PASS/FAIL"] == "PASS").sum()) == 5 and len(six) == 6,
        six[["Condition", "PASS/FAIL"]].to_dict(orient="records"),
    )
    check(
        "C2 alone failed",
        six.loc[six["PASS/FAIL"] == "FAIL", "Condition"].tolist() == ["C2"],
        six.loc[six["PASS/FAIL"] == "FAIL", "Condition"].tolist(),
    )

    fairness_table = pd.read_csv(TABLES / "table_fairness_summary.csv")
    check("fairness table has 27 eligible groups", len(fairness_table) == 27, len(fairness_table))
    check(
        "Primary improves 25 of 27 eligible groups",
        int((fairness_table["Primary minus Global MAE"] < 0).sum()) == 25,
        int((fairness_table["Primary minus Global MAE"] < 0).sum()),
    )
    worsened = fairness_table[fairness_table["Primary minus Global MAE"] > 0]
    check(
        "only frozen two fairness cells worsen",
        set(zip(worsened["Sensitive field"], worsened["Group"]))
        == {
            ("applicant_sex_name", "Female"),
            ("co_applicant_sex_name", "Male"),
        },
        worsened[
            ["Sensitive field", "Group", "Primary minus Global MAE"]
        ].to_dict(orient="records"),
    )
    disparities = pd.read_csv(TABLES / "table_fairness_disparities.csv")
    numeric_mae_gaps = pd.to_numeric(disparities["MAE gap"], errors="coerce")
    check(
        "largest eligible MAE gap exact",
        close(float(numeric_mae_gaps.max()), 43.25496404918562),
        float(numeric_mae_gaps.max()),
    )
    minority_row = disparities[
        disparities["Sensitive field"] == "minority_population"
    ]
    check(
        "minority_population zero-eligible disparity row present",
        len(minority_row) == 1
        and int(minority_row.iloc[0]["Eligible groups"]) == 0
        and minority_row.iloc[0]["MAE gap"]
        == "NOT_AVAILABLE_FROM_FROZEN_EVIDENCE",
        minority_row.to_dict(orient="records"),
    )
    check(
        "frozen fairness group count in final handoff",
        fairness_final["main_fairness_findings"]["eligible_group_levels"] == 27,
        fairness_final["main_fairness_findings"]["eligible_group_levels"],
    )

    explainability = pd.read_csv(TABLES / "table_final_explainability.csv")
    check(
        "explainability has separate three components",
        set(explainability["Component"])
        == {"Global consensus", "Meta-Gate", "Residual Specialist"},
        sorted(explainability["Component"].unique()),
    )
    check(
        "each explainability component has Top-15",
        explainability.groupby("Component").size().to_dict()
        == {"Global consensus": 15, "Meta-Gate": 15, "Residual Specialist": 15},
        explainability.groupby("Component").size().to_dict(),
    )
    component_table = pd.read_csv(TABLES / "table_global_component_importance.csv")
    xgb_spaces = component_table[
        component_table["Component"] == "XGBoost"
    ]["Attribution space"].unique()
    check(
        "XGBoost native log1p attribution caveat",
        len(xgb_spaces) == 1 and "log1p" in xgb_spaces[0],
        xgb_spaces.tolist(),
    )

    figure_hash_checks = []
    for record in figure_validation["figures"]:
        figure_hash_checks.append(
            digest(ROOT / record["plotting_data_path"]) == record["plotting_data_sha256"]
            and digest(ROOT / record["png_path"]) == record["png_sha256"]
            and digest(ROOT / record["pdf_path"]) == record["pdf_sha256"]
        )
    check(
        "all figure hashes match validation report",
        all(figure_hash_checks) and len(figure_hash_checks) == 15,
        {"matched": sum(figure_hash_checks), "total": len(figure_hash_checks)},
    )
    check(
        "every figure records axes metrics and units",
        all(
            record.get("axes") and record.get("metrics") and record.get("units")
            for record in figure_validation["figures"]
        ),
        15,
    )

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    source = "\n".join(cell.source for cell in code_cells)
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
    check("final notebook exists and executes without errors", not error_outputs, len(error_outputs))
    check("final notebook renders all 15 figures inline", image_outputs == 15, image_outputs)
    check("final notebook renders quantitative tables inline", table_outputs >= 10, table_outputs)
    static_forbidden = {
        ".fit(": source.count(".fit("),
        ".predict(": source.count(".predict("),
        "shap.": source.lower().count("shap."),
        "np.random": source.count("np.random"),
        "iid_holdout_features.parquet": source.count("iid_holdout_features.parquet"),
        "iid_holdout_targets.parquet": source.count("iid_holdout_targets.parquet"),
    }
    check(
        "notebook has zero forbidden scientific calls or original IID paths",
        not any(static_forbidden.values()),
        static_forbidden,
    )

    manifest_path = FINAL / "FINAL_REPRODUCIBILITY_MANIFEST.json"
    manifest = read_json(manifest_path)
    check("reproducibility manifest exists and PASS", manifest.get("status") == "PASS", manifest.get("status"))
    check(
        "manifest reports zero-compute integrity",
        all(value == 0 for value in manifest["integrity"].values()),
        manifest["integrity"],
    )
    check(
        "manifest did not reopen original IID for hashing",
        manifest["data_hashes_from_saved_evidence_only"][
            "original_iid_content_reopened_for_hashing"
        ]
        is False,
        False,
    )
    check(
        "handoff records zero original IID reads",
        handoff.get("original_iid_content_reads") == 0
        and handoff.get("original_iid_content_hashes_computed") == 0,
        {
            "reads": handoff.get("original_iid_content_reads"),
            "hashes": handoff.get("original_iid_content_hashes_computed"),
        },
    )
    check(
        "final IID evaluation records no fit/refit/tuning/model change",
        iid["no_fit"] is True
        and iid["no_refit"] is True
        and iid["no_tuning"] is True
        and iid["no_model_change"] is True,
        {
            "no_fit": iid["no_fit"],
            "no_refit": iid["no_refit"],
            "no_tuning": iid["no_tuning"],
            "no_model_change": iid["no_model_change"],
        },
    )

    model_card = (FINAL / "MODEL_CARD_FINAL.md").read_text(encoding="utf-8")
    technical = (FINAL / "FINAL_TECHNICAL_REPORT.md").read_text(encoding="utf-8")
    results = (FINAL / "PAPER_RESULTS_TEXT.md").read_text(encoding="utf-8")
    limitations = (FINAL / "PAPER_LIMITATIONS_TEXT.md").read_text(encoding="utf-8")
    required_disclaimer = "does not assess lending approval decisions"
    check(
        "fairness disclaimer is consistent",
        all(required_disclaimer in text for text in [model_card, technical, results]),
        required_disclaimer,
    )
    check(
        "explainability non-causal disclaimer is consistent",
        all(
            "do not establish causal relationships" in text
            for text in [model_card, technical, results]
        ),
        "required explainability wording",
    )
    limitation_terms = [
        "HMDA 2017",
        "property-value",
        "LTV-like",
        "C2",
        "adaptive",
        "two models",
        "descriptive",
        "not causal",
        "Sensitive variables were excluded",
        "Geographic",
        "lending decisions",
    ]
    check(
        "paper limitations covers required topics",
        all(term.lower() in limitations.lower() for term in limitation_terms),
        limitation_terms,
    )
    inflated = [
        phrase
        for phrase in [
            "state of the art",
            "discrimination-free",
            "tail problem solved",
            "production-ready",
            "statistically significant",
            "causal feature effect",
        ]
        if any(
            phrase in text.lower()
            for text in [model_card, technical, results, limitations]
        )
    ]
    check("no inflated claims", not inflated, inflated)

    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    result = {
        "status": status,
        "created_at_utc": now(),
        "authorization_id": AUTHORIZATION,
        "independent_of_reporting_generator": True,
        "check_count": len(checks),
        "checks_passed": sum(item["status"] == "PASS" for item in checks),
        "checks": checks,
        "integrity": {
            "original_iid_content_reads": 0,
            "fits": 0,
            "predictions": 0,
            "shap_recomputations": 0,
            "bootstrap_recomputations": 0,
            "model_changes": 0,
        },
        "final_primary_unchanged": True,
        "all_final_claims_supported_by_frozen_evidence": status == "PASS",
    }
    write_json(OUTPUT, result)
    if status != "PASS":
        failures = [item for item in checks if item["status"] == "FAIL"]
        raise RuntimeError(f"Independent Prompt 5C verification failed: {failures}")


if __name__ == "__main__":
    main()
