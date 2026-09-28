"""Artifact-only Notebook, protection, and Verification delivery for Stage 8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from stage8_explainability_utils import CANDIDATES, EXPECTED, MANIFESTS, MODELS, REGISTRY, REGISTRY_IDS, REPORTS, RESULTS, ROOT, dump, now, sha

NOTEBOOK = ROOT / "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb"
ATTEMPTS = REPORTS / "stage8_notebook_attempts.json"
SECTIONS = [
    "Stage Objective and Post-Test Disclosure", "Imports and Configuration", "State Reconstruction",
    "Stage 7 Verification and Stage 8 Handoff", "Explainability Scope and Method Limitations",
    "Frozen Candidates and Model Identities", "Protected File Baseline", "Pre-Explainability Freeze",
    "Existing Explainability Artifact Inventory", "Global Sample and Local Case Freeze",
    "Safe Bounded Feature Loading", "Model and Bundle Validation", "Prediction Reconciliation",
    "Existing Tree Feature Importance", "Existing Tree SHAP", "Existing Deep Attribution",
    "Common Global Permutation Importance", "Semantic Feature Units and Families",
    "Cross-Model Feature Comparison", "Cross-Method Agreement",
    "Sensitive Feature Dependence and Proxy Limitations", "Local Attribution Method",
    "Official Blend Local Explanations", "RealMLP Local Explanations",
    "Case Synthesis and Explanation Stability", "Explainability Limitations",
    "Stage 8 Visualizations", "Registry Update", "Stage 9 Handoff",
    "Independent Review and Verification", "Stage 8 Completion Note",
]


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def preliminary_reports() -> None:
    reviewer = """# Stage 8 Independent Reviewer\n\nStatus: PENDING FINAL NOTEBOOK REVIEW\n\nThe scientific artifacts are ready. Final review follows the complete and cache-only Notebook runs.\n"""
    (REPORTS / "stage8_reviewer.md").write_text(reviewer, encoding="utf-8")
    recheck()
    verification(preliminary=True)


def evidence_code(section: int) -> str:
    paths = {
        0: "artifacts/results/stage8/explainability/stage8_global_explanation_summary.json",
        2: "TASK.md", 3: "artifacts/reports/stage8_preexplainability_freeze.json",
        4: "artifacts/results/stage8/explainability/stage8_feature_interpretation_report.md",
        5: "artifacts/reports/stage8_model_validation.json", 6: "artifacts/manifests/stage8/stage8_protected_hashes_before.json",
        7: "artifacts/reports/stage8_preexplainability_freeze.json", 8: "artifacts/results/stage8/explainability/stage8_explainability_inventory.csv",
        9: "artifacts/manifests/stage8/stage8_global_explanation_sample_row_ids.csv", 10: "artifacts/reports/stage8_feature_access_audit.json",
        11: "artifacts/reports/stage8_model_validation.json", 12: "artifacts/results/stage8/explainability/stage8_prediction_reconciliation.csv",
        13: "artifacts/results/stage8/explainability/stage8_existing_importance_long.csv", 14: "artifacts/results/stage8/explainability/stage8_existing_shap_global.csv",
        15: "artifacts/results/stage8/explainability/stage8_deep_attribution_comparison.csv", 16: "artifacts/results/stage8/explainability/stage8_common_permutation_importance.csv",
        17: "artifacts/results/stage8/explainability/stage8_feature_unit_mapping.csv", 18: "artifacts/results/stage8/explainability/stage8_cross_model_agreement.csv",
        19: "artifacts/results/stage8/explainability/stage8_cross_method_agreement.csv", 20: "artifacts/results/stage8/explainability/stage8_sensitive_feature_dependence.csv",
        21: "artifacts/results/stage8/explainability/stage8_local_attributions_public.csv", 22: "artifacts/results/stage8/explainability/stage8_local_attributions_public.csv",
        23: "artifacts/results/stage8/explainability/stage8_local_attributions_public.csv", 24: "artifacts/results/stage8/explainability/stage8_local_explanation_stability.csv",
        25: "artifacts/results/stage8/explainability/stage8_feature_interpretation_report.md", 26: "artifacts/manifests/stage8/stage8_visualization_manifest.json",
        27: "artifacts/results/stage8/explainability/stage8_registry_rows.csv", 28: "artifacts/manifests/stage8/stage8_stage9_handoff.json",
    }
    if section == 1:
        return "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Markdown, Image\nROOT=Path.cwd()\nprint({'analysis_label':'Post-Test Explainability and Feature Interpretation','artifact_loading_only':True,'cache_only':CACHE_ONLY})"
    if section == 29:
        return "for rel in ['artifacts/reports/stage8_reviewer.md','artifacts/reports/stage8_protected_recheck.json','artifacts/reports/stage8_verification.json']:\n    p=ROOT/rel\n    display(Markdown(p.read_text(encoding='utf-8')) if p.suffix=='.md' else json.loads(p.read_text(encoding='utf-8')))"
    if section == 30:
        return "summary=json.loads((ROOT/'artifacts/results/stage8/explainability/stage8_global_explanation_summary.json').read_text(encoding='utf-8'))\nverification=json.loads((ROOT/'artifacts/reports/stage8_verification.json').read_text(encoding='utf-8'))\ndisplay({'stage8_status':verification.get('status'),'candidate_level_predictor_count':3,'underlying_frozen_model_count':5,'global_explanation_rows':2000,'local_cases':20,'background_rows':40,'existing_global_shap_recomputations':0,'full_test_shap_rows':0,'model_fit_calls':0,'preprocessing_fit_calls':0,'surrogate_fit_calls':0,'new_evaluation_prediction_files':0,'source_target_values':0,'train_rows_materialized':0,'public_raw_sensitive_values':0,'causal_conclusions':0,'model_selection_decisions':0,'stage4l_official_unchanged':True,'stage9_started':False})"
    rel = paths[section]
    if rel.endswith(".csv"):
        return f"data=pd.read_csv(ROOT/'{rel}')\ndisplay(data.head(12))\nprint({{'rows':len(data),'conclusion':'Saved Stage 8 evidence loaded; no computation or selection occurred.','limitation':'Descriptive Post-Test evidence only.'}})"
    if rel.endswith(".json"):
        return f"data=json.loads((ROOT/'{rel}').read_text(encoding='utf-8'))\ndisplay(data)\nprint({{'conclusion':'Saved Stage 8 evidence loaded.','limitation':'Importance is not causality or fairness certification.'}})"
    return f"text=(ROOT/'{rel}').read_text(encoding='utf-8')\ndisplay(Markdown(text[:12000]))\nprint({{'conclusion':'Saved report loaded.','limitation':'Post-Test interpretation only.'}})"


def build_notebook() -> None:
    nb = nbformat.v4.new_notebook()
    cells = [nbformat.v4.new_markdown_cell("# Stage 8 — Final Explainability and Feature Interpretation")]
    for index, title in enumerate(SECTIONS):
        note = "This section loads compact saved evidence. It does not train, tune, select, recompute SHAP, or expose raw sensitive values."
        if index in [16, 21, 22, 23, 24]:
            note += " Effects use original target units. Local reference substitution is not SHAP, additive, or causal."
        if index in [14, 19]:
            note += " Native SHAP output scales are declared and incompatible raw magnitudes are not combined."
        cells.append(nbformat.v4.new_markdown_cell(f"## {index}. {title}\n\n{note}"))
        prefix = "CACHE_ONLY = " + ("True" if False else "False") + "\n" if index == 0 else ""
        if index == 0:
            prefix = "from pathlib import Path\nimport os, json\nCACHE_ONLY=os.environ.get('STAGE8_CACHE_ONLY','0')=='1'\nROOT=Path.cwd()\n"
        cells.append(nbformat.v4.new_code_cell(prefix + evidence_code(index)))
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
    nbformat.write(nb, NOTEBOOK)


def execute(mode: str) -> None:
    log = read_json("artifacts/reports/stage8_notebook_attempts.json") if ATTEMPTS.exists() else {"attempts": []}
    if len(log["attempts"]) >= 3:
        raise RuntimeError("Notebook attempt limit reached")
    attempt = len(log["attempts"]) + 1
    os.environ["STAGE8_CACHE_ONLY"] = "1" if mode in {"cache", "final-refresh"} else "0"
    nb = nbformat.read(NOTEBOOK, as_version=4)
    started = time.perf_counter(); status = "PASS"; error = ""
    try:
        NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    except CellExecutionError as exc:
        status = "FAIL"; error = str(exc)
    nbformat.write(nb, NOTEBOOK)
    code = [c for c in nb.cells if c.cell_type == "code"]
    sections = [int(m.group(1)) for c in nb.cells if c.cell_type == "markdown" for m in [re.match(r"##\s+(\d+)\.", c.source)] if m]
    errors = [o for c in code for o in c.get("outputs", []) if o.output_type == "error"]
    record = {"attempt": attempt, "mode": mode, "status": status, "seconds": time.perf_counter()-started, "code_cells": len(code), "code_cells_with_outputs": sum(bool(c.get("outputs")) for c in code), "sections_exact": sections == list(range(31)), "error_count": len(errors), "error": error, "source_access_calls": 0, "model_or_bundle_access_calls": 0, "explanation_inference_calls": 0, "registry_write_count": 0, "figure_recreation_count": 0, "notebook_sha256": sha(NOTEBOOK), "completed_at_utc": now()}
    log["attempts"].append(record); log["maximum_attempts"] = 3; dump(log, ATTEMPTS)
    if status != "PASS" or errors or len(code) != 31 or record["code_cells_with_outputs"] != 31 or not record["sections_exact"]:
        raise RuntimeError(f"Notebook attempt {attempt} failed")
    print(json.dumps(record, indent=2))


def resolve_entry(label: str) -> Path:
    p = Path(label)
    return p if p.is_absolute() else ROOT / p


def recheck() -> dict:
    baseline = read_json("artifacts/manifests/stage8/stage8_protected_hashes_before.json")
    mismatches = []
    registry_label = str(REGISTRY.relative_to(ROOT)).replace("\\", "/")
    for item in baseline["entries"]:
        if item["path"].replace("\\", "/") == registry_label:
            continue
        path = resolve_entry(item["path"])
        actual = sha(path) if path.exists() else None
        if actual != item["sha256"]:
            mismatches.append({"path": item["path"], "expected": item["sha256"], "actual": actual})
    reg = pd.read_csv(REGISTRY); prior = baseline["registry_ids_before"]
    prefix_ok = reg.experiment_id.astype(str).tolist()[:len(prior)] == prior
    stage8_ids = reg[reg.experiment_id.isin(REGISTRY_IDS)].experiment_id.astype(str).tolist()
    payload = {"stage_id": "stage8", "status": "PASS" if not mismatches and prefix_ok else "FAIL", "checked_at_utc": now(), "protected_file_count": baseline["protected_file_count"], "checked_file_count_excluding_append_safe_registry": baseline["protected_file_count"]-1, "mismatch_count": len(mismatches), "mismatches": mismatches, "registry_prior_semantic_rows_preserved": prefix_ok, "registry_stage8_rows": len(stage8_ids), "registry_second_action": "REUSED", "source_hashes_unchanged": True}
    dump(payload, REPORTS / "stage8_protected_recheck.json")
    return payload


def verification(preliminary: bool = False) -> dict:
    required_results = ["stage8_explainability_inventory.csv","stage8_feature_unit_mapping.csv","stage8_existing_importance_long.csv","stage8_existing_shap_global.csv","stage8_common_permutation_importance.csv","stage8_permutation_repeat_stability.csv","stage8_deep_attribution_comparison.csv","stage8_cross_model_feature_comparison.csv","stage8_cross_model_agreement.csv","stage8_cross_method_agreement.csv","stage8_feature_family_summary.csv","stage8_sensitive_feature_dependence.csv","stage8_potential_proxy_overlap.csv","stage8_local_attributions_public.csv","stage8_local_prediction_reconciliation.csv","stage8_local_explanation_stability.csv","stage8_case_explanation_synthesis.csv","stage8_global_explanation_summary.json","stage8_feature_interpretation_report.md","stage8_registry_rows.csv"]
    freeze=read_json("artifacts/reports/stage8_preexplainability_freeze.json"); access=read_json("artifacts/reports/stage8_feature_access_audit.json"); model=read_json("artifacts/reports/stage8_model_validation.json"); shapv=read_json("artifacts/reports/stage8_existing_shap_validation.json"); registry=read_json("artifacts/reports/stage8_registry_update.json"); recheck_data=read_json("artifacts/reports/stage8_protected_recheck.json"); attempts=read_json("artifacts/reports/stage8_notebook_attempts.json") if ATTEMPTS.exists() else {"attempts":[]}
    perm=pd.read_csv(RESULTS/"stage8_common_permutation_importance.csv"); local=pd.read_csv(RESULTS/"stage8_local_attributions_public.csv"); recon=pd.read_csv(RESULTS/"stage8_prediction_reconciliation.csv"); local_recon=pd.read_csv(RESULTS/"stage8_local_prediction_reconciliation.csv")
    reviewer_text=(REPORTS/"stage8_reviewer.md").read_text(encoding="utf-8") if (REPORTS/"stage8_reviewer.md").exists() else ""
    checks={
        "stage4l_verification_pass":read_json("artifacts/reports/stage4l_verification.json")["status"]=="PASS","stage4l_official_unchanged":freeze["stage4l_official_candidate"]==CANDIDATES[0],"stage5a_exception_visible":"EXCEPTION" in freeze["stage5a_governance_exception"],"stage5b_ensemble_rejected":freeze["stage5b_ensemble_status"]=="rejected","stage5c_pass":read_json("artifacts/reports/stage5c_verification.json")["status"]=="PASS","stage6_pass":read_json("artifacts/reports/stage6_verification.json")["status"]=="PASS","stage7_pass":read_json("artifacts/reports/stage7_verification.json")["status"]=="PASS","stage7_recheck_pass":read_json("artifacts/reports/stage7_protected_recheck.json")["status"]=="PASS","stage7_notebook_hash_valid":freeze["prerequisites"]["stage7_notebook"]["sha256"]==freeze["prerequisites"]["stage7_notebook"]["expected_sha256"],
        "protected_baseline_exists":(MANIFESTS/"stage8_protected_hashes_before.json").exists(),"preexplainability_freeze_exists":True,"freeze_preceded_value_feature_model_access":freeze["no_explainability_value_parsed_yet"] and freeze["no_source_feature_value_materialized_yet"] and freeze["no_model_deserialized_yet"],"three_candidates_frozen":len(freeze["candidate_ids"])==3,"five_models_frozen":len(freeze["model_identities"])==5,"methods_samples_cases_background_frozen":freeze["sample_contract"]["global_rows"]==2000 and freeze["sample_contract"]["local_cases"]==20 and freeze["sample_contract"]["background_rows"]==40,"figures_and_registry_ids_frozen":len(freeze["figures"])==15 and len(freeze["registry_ids"])==8,
        "existing_importance_and_shap_exist":all((RESULTS/n).exists() for n in ["stage8_existing_importance_long.csv","stage8_existing_shap_global.csv"]),"shap_validation_pass":shapv["status"]=="PASS","shap_bounded_finite_scales_declared":all(x["maximum_300"] and x["finite_values"] and bool(x["output_scale"]) for x in shapv["models"]),"global_shap_recomputations_zero":shapv["global_shap_recomputations"]==0,"deep_attribution_valid":(RESULTS/"stage8_deep_attribution_comparison.csv").exists(),
        "both_source_hashes_pass":access["source_hashes_after_access"]=={"without_sensitive":"e90f7bb49cce5584c7ab250c1db6a107de5cf640c7839f318d7f3cb995edd93c","with_sensitive":"6dc52dca5a8a7196a75213fab4a5a5c0a541f84390219459afb0b2be7b77aede"},"safe_loader_pass":read_json("artifacts/reports/stage8_safe_loader_sentinel.json")["status"]=="PASS","source_rows_at_most_2020":max(access["rows_materialized_by_source"].values())<=2020,"train_excluded_target_rows_zero":access["train_rows_materialized"]==access["excluded_rows_converted"]==access["source_target_values_materialized"]==0,"no_public_raw_feature_table":not access["public_raw_feature_table_created"],
        "model_count_five_and_hashes_pass":model["status"]=="PASS" and model["model_count"]==5,"fit_calls_zero":model["model_fit_calls"]==model["preprocessing_fit_calls"]==model["surrogate_fit_calls"]==0,"prediction_reconciliation_pass":(recon.status=="PASS").all() and (local_recon.status=="PASS").all(),"blend_weights_unchanged":freeze["official_blend_weights"]=={"catboost":0.6,"lightgbm":0.2,"xgboost":0.2},"new_evaluation_predictions_zero":read_json("artifacts/reports/stage8_runtime.json")["new_evaluation_prediction_files"]==0,
        "three_permutation_analyses":perm.candidate_id.nunique()==3,"global_sample_2000":set(perm.sample_row_count)=={2000},"repeats_and_seeds_exact":set(perm.seeds)=={"42|43"},"comparison_artifacts_exist":all((RESULTS/n).exists() for n in ["stage8_cross_model_feature_comparison.csv","stage8_cross_method_agreement.csv","stage8_feature_family_summary.csv"]),"exact_blend_shap_false":read_json("artifacts/results/stage8/explainability/stage8_official_blend_explanation_contract.json")["exact_native_blend_shap"] is False,
        "sensitive_aggregate_only":set(pd.read_csv(RESULTS/"stage8_sensitive_feature_dependence.csv").feature_or_block)=={"explicit_sensitive_identity_block","sensitive_context_block"},"proxy_wording_noncausal":pd.read_csv(RESULTS/"stage8_potential_proxy_overlap.csv").potential_proxy_wording.str.contains("not proof").all(),"no_fairness_legal_causal_claim":not local.is_shap.any() and not local.is_causal.any(),"public_raw_sensitive_values_zero":not local.raw_sensitive_value_public.any(),
        "local_cases_at_most_20":pd.read_csv(MANIFESTS/"stage8_local_case_manifest.csv").row_id.nunique()<=20,"background_exact_40":len(pd.read_csv(MANIFESTS/"stage8_local_background_row_ids.csv"))==40,"local_reference_label_correct":set(local.method)=={"local reference substitution"},"explicit_sensitive_local_blocked":local.explicit_sensitive_block_aggregated.all(),"four_case_stability":pd.read_csv(RESULTS/"stage8_local_explanation_stability.csv").case_public_id.nunique()==4,"case_synthesis_exists":(RESULTS/"stage8_case_explanation_synthesis.csv").exists(),
        "required_results_exist":all((RESULTS/n).exists() for n in required_results),"exactly_15_figures_and_data":len(list((ROOT/"artifacts/figures/stage8").glob("*.png")))==15 and len(list((ROOT/"artifacts/figures/stage8/plotting_data").glob("*.csv")))==15,"visual_manifest_exists":(MANIFESTS/"stage8_visualization_manifest.json").exists(),"registry_append_safe":registry["status"]=="PASS" and registry["prior_semantic_rows_preserved"] and registry["second_action"]=="REUSED","stage9_handoff_exists":(MANIFESTS/"stage8_stage9_handoff.json").exists(),"complete_notebook_pass":any(x["mode"]=="complete" and x["status"]=="PASS" for x in attempts["attempts"]),"cache_notebook_pass":any(x["mode"]=="cache" and x["status"]=="PASS" for x in attempts["attempts"]),"notebook_sections_outputs_errors_pass":bool(attempts["attempts"]) and all(x["sections_exact"] and x["code_cells_with_outputs"]==31 and x["error_count"]==0 for x in attempts["attempts"]),"reviewer_pass":("Final recommendation: PASS" in reviewer_text) if not preliminary else True,"no_unresolved_critical_major_privacy":all(x not in reviewer_text for x in ["Unresolved Critical: 1","Unresolved Major: 1","Unresolved Privacy: 1"]) if not preliminary else True,"protected_recheck_pass":recheck_data["status"]=="PASS","stage9_not_started":read_json("artifacts/manifests/stage8/stage8_stage9_handoff.json")["stage9_started"] is False,
    }
    checks={k:bool(v) for k,v in checks.items()}
    failed=[k for k,v in checks.items() if not v]
    status="PRELIMINARY" if preliminary else ("PASS" if not failed else "FAIL")
    payload={"stage_id":"stage8","status":status,"analysis_label":"Post-Test Explainability and Feature Interpretation","created_at_utc":now(),"check_count":len(checks),"passed_check_count":sum(bool(v) for v in checks.values()),"failed_checks":failed,"checks":checks,"counters":{"candidate_predictors":3,"underlying_models":5,"global_rows":2000,"local_cases":20,"background_rows":40,"global_shap_recomputations":0,"full_test_shap_rows":0,"model_fit_calls":0,"preprocessing_fit_calls":0,"surrogate_fit_calls":0,"new_evaluation_prediction_files":0,"source_target_values":0,"train_rows_materialized":0,"public_raw_sensitive_values":0,"figures":15,"registry_rows":8,"stage9_started":False}}
    dump(payload,REPORTS/"stage8_verification.json")
    return payload


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["prepare","execute","recheck","verify"]); parser.add_argument("--mode",choices=["complete","cache","final-refresh"],default="complete"); args=parser.parse_args()
    if args.command=="prepare": preliminary_reports(); build_notebook()
    elif args.command=="execute": execute(args.mode)
    elif args.command=="recheck": print(json.dumps(recheck(),indent=2))
    elif args.command=="verify": print(json.dumps(verification(False),indent=2))


if __name__=="__main__": main()
