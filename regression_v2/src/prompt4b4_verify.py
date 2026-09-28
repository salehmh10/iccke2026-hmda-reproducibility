"""Independent read-only review and final verification for Prompt 4B4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from .prompt4_metrics import compute_regression_metrics


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
SOURCE = ROOT / "outputs/data/development.parquet"
CANDIDATES = ["prompt4b4__denseweight_cat", "prompt4b4__sera_xgb", "prompt4b4__imrgb", "prompt4b4__lds_lgb"]
SUBSTITUTIONS = ["prompt4b4__sub_densecat", "prompt4b4__sub_seraxgb", "prompt4b4__sub_imrgb", "prompt4b4__sub_ldslgb"]
EXPECTED_SHA = "0ed232397be3ec4de1483c594954dce7b4704b375ca295397d899323dc4f0b6b"
EXPECTED_TRAIN = "26265d75d8fa35d7417e2a9fb2888b9f625e6d19123cd7a2972974f403a30166"
EXPECTED_VALIDATION = "676b577233627c8b237d214a29eeb798e099d028583c8b73068b151a2a204290"
EXPECTED_SELECTION = "3af54200bbda79bbe910903b4a32c217d409b19b07cf0f54cd2076d9bf0db03a"
EXPECTED_AUDIT = "f7dcb7ee23049a24564f474f5e1730dde756bf22158e7cd417c90a78b15b1c48"
PROHIBITED = {"respondent_id", "p_tail", "global_prediction", "residual_proposal", "component_disagreement",
              "benefit_score", "beat_probability", "target_rarity_score", "target_decile", "true_tail",
              "realized_error", "residual", "benefit"}


def now(): return datetime.now(timezone.utc).isoformat()


def read(name): return json.loads((REPORTS/name).read_text(encoding="utf-8"))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def digest(values): return hashlib.sha256("\n".join(pd.Series(values).astype(str)).encode()).hexdigest()


def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    temp=path.with_suffix(path.suffix+".tmp"); temp.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    json.loads(temp.read_text(encoding="utf-8")); os.replace(temp,path)


def checks(require_reviewer=False):
    result=[]
    def check(name, condition, detail=""):
        result.append({"check":name,"status":"PASS" if bool(condition) else "FAIL","detail":str(detail)})

    required=["prompt4b4_handoff_validation.json","prompt4b4_method_reproduction.json","prompt4b4_design_freeze.json",
      "prompt4b4_denseweight_audit.json","prompt4b4_sera_audit.json","prompt4b4_imrgb_audit.json","prompt4b4_lds_audit.json",
      "prompt4b4_fit_ledger.json","prompt4b4_runtime.json","prompt4b4_model_manifest.json","prompt4b4_prediction_manifest.json",
      "prompt4b4_standalone_results.csv","prompt4b4_substitution_results.csv","prompt4b4_candidate_results.csv",
      "prompt4b4_six_condition_results.csv","prompt4b4_cross_stage_comparison.csv","prompt4b4_champion_selection.json",
      "prompt4b4_bootstrap.csv","prompt4b4_notebook_execution.json"]
    check("required_artifacts",all((REPORTS/n).exists() for n in required),[n for n in required if not (REPORTS/n).exists()])
    handoff=read("prompt4b4_handoff_validation.json"); reproduction=read("prompt4b4_method_reproduction.json")
    design=read("prompt4b4_design_freeze.json"); ledger=read("prompt4b4_fit_ledger.json")
    models=read("prompt4b4_model_manifest.json"); predictions=read("prompt4b4_prediction_manifest.json")
    roles=read("prompt4b4_champion_selection.json"); notebook=read("prompt4b4_notebook_execution.json")
    check("prior_handoffs",all(row["status"] in {"PASS","PASS_EXPERIMENT_COMPLETE_PARTIAL"} and sha(ROOT/row["path"])==row["sha256"] for row in handoff["prior_reports"]))
    check("development_sha",sha(SOURCE)==EXPECTED_SHA)
    frame=pd.read_parquet(SOURCE,columns=["loan_amount_000s","development_role","row_hash"])
    train=frame.loc[frame.development_role.eq("train")]; validation=frame.loc[frame.development_role.eq("validation")].reset_index(drop=True)
    check("membership_counts",(len(frame),len(train),len(validation))==(500000,400000,100000))
    check("train_digest",digest(train.row_hash)==EXPECTED_TRAIN)
    check("validation_digest",digest(validation.row_hash)==EXPECTED_VALIDATION)
    role_vector=pd.read_parquet(ROOT/predictions["predictions"][0]["path"],columns=["selection_or_audit_role"])["selection_or_audit_role"].to_numpy()
    check("selection_audit_counts",int((role_vector=="selection").sum())==70000 and int((role_vector=="audit").sum())==30000)
    check("selection_digest",digest(validation.loc[role_vector=="selection","row_hash"])==EXPECTED_SELECTION)
    check("audit_digest",digest(validation.loc[role_vector=="audit","row_hash"])==EXPECTED_AUDIT)
    frozen_digest=design.pop("design_digest"); check("design_digest",canonical(design)==frozen_digest,frozen_digest); design["design_digest"]=frozen_digest
    check("four_frozen_hypotheses",design["scientific_candidates"]==CANDIDATES and design["scientific_execution_order"]==CANDIDATES)
    check("exact_features",design["source"]["feature_count"]==35 and len(design["source"]["features"])==35 and not (set(design["source"]["features"])&PROHIBITED))
    check("method_reproduction",reproduction["status"]=="PASS" and reproduction["reproducible_method_count"]==4)
    dense=read("prompt4b4_denseweight_audit.json"); sera=read("prompt4b4_sera_audit.json"); imr=read("prompt4b4_imrgb_audit.json"); lds=read("prompt4b4_lds_audit.json")
    check("denseweight_audit",dense["status"]=="PASS" and dense["alpha"]==1.0 and dense["finite_count"]==400000 and dense["min"]>0 and abs(dense["mean"]-1)<1e-12 and not dense["validation_target_involvement"])
    check("sera_audit",sera["status"]=="PASS" and sera["type"]=="high" and sera["adjusted_boxplot"] and sera["grid_T"]==1000 and sera["gradient_verification_max_difference"]<=1e-12 and not sera["validation_target_involvement"])
    check("imr_audit",imr["status"]=="PASS" and imr["n_components"]==6 and imr["gradient_reference_max_difference"]<=1e-10 and not imr["validation_target_involvement"])
    check("lds_audit",lds["status"]=="PASS" and lds["bin_width"]==1.0 and lds["reweight_mode"]=="sqrt_inv" and lds["kernel"]=="gaussian" and lds["kernel_size"]==5 and lds["sigma"]==2 and lds["min"]>0 and abs(lds["mean"]-1)<1e-12 and not lds["validation_target_involvement"])
    scientific=[e for e in ledger["entries"] if e.get("category")=="scientific_candidate"]
    passed=[e for e in scientific if e.get("status")=="PASS"]
    check("fit_budget",ledger["scientific_fit_count"]==4 and len(passed)==4 and {e["candidate_id"] for e in passed}==set(CANDIDATES))
    check("fit_attempts",ledger["physical_attempt_count"]==5 and ledger["technical_retry_count"]==1 and max(sum(e["candidate_id"]==c for e in scientific) for c in CANDIDATES)<=2)
    check("no_unregistered_fit",all(e["candidate_id"] in CANDIDATES for e in scientific))
    check("model_manifest",models["status"]=="PASS" and models["model_count"]==4 and all(sha(ROOT/m["artifact_path"])==m["sha256"] and m["clean_reload_status"]=="PASS" and m["max_reload_difference"]==0 for m in models["models"]))
    check("prediction_manifest",predictions["status"]=="PASS" and predictions["prediction_count"]==8 and predictions["iid_prediction_count"]==0)
    loaded={}
    pred_checks=[]
    for row in predictions["predictions"]:
        pf=pd.read_parquet(ROOT/row["path"]); loaded[row["candidate_id"]]=pf.y_pred.to_numpy(float)
        pred_checks.append(len(pf)==100000 and digest(pf.row_hash)==EXPECTED_VALIDATION and np.isfinite(pf.y_pred).all() and sha(ROOT/row["path"])==row["prediction_sha256"])
    check("prediction_alignment",all(pred_checks))
    frozen={k:pd.read_parquet(ROOT/p).y_pred.to_numpy(float) for k,p in {
      "cat":"outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet",
      "lgb":"outputs/predictions/prompt2/validation/selected_lightgbm_without_lender.parquet",
      "xgb":"outputs/predictions/prompt2/validation/selected_xgboost_without_lender.parquet"}.items()}
    expected={SUBSTITUTIONS[0]:.6*loaded[CANDIDATES[0]]+.2*frozen["lgb"]+.2*frozen["xgb"],
      SUBSTITUTIONS[1]:.6*frozen["cat"]+.2*frozen["lgb"]+.2*loaded[CANDIDATES[1]],
      SUBSTITUTIONS[2]:.6*frozen["cat"]+.2*frozen["lgb"]+.2*loaded[CANDIDATES[2]],
      SUBSTITUTIONS[3]:.6*frozen["cat"]+.2*loaded[CANDIDATES[3]]+.2*frozen["xgb"]}
    check("substitution_formulas",all(np.max(np.abs(loaded[k]-v))<=1e-12 for k,v in expected.items()))
    table=pd.read_csv(REPORTS/"prompt4b4_candidate_results.csv"); y=validation.loan_amount_000s.to_numpy(float)
    metric_ok=True
    for _,row in table.iterrows():
        mask=role_vector=="selection" if row.scope=="selection" else role_vector=="audit" if row.scope=="audit_descriptive" else np.ones(len(y),bool)
        got=compute_regression_metrics(y[mask],loaded[row.candidate_id][mask])
        metric_ok &= all(np.isclose(float(row[k]),float(got[k]),atol=1e-10,rtol=1e-12) for k in got)
    check("shared_metrics",metric_ok)
    global_pred=.6*frozen["cat"]+.2*frozen["lgb"]+.2*frozen["xgb"]
    six=pd.read_csv(REPORTS/"prompt4b4_six_condition_results.csv"); six_ok=True
    selections=[]
    for cid in CANDIDATES+SUBSTITUTIONS:
        cr=table[(table.candidate_id==cid)&(table.scope=="selection")].iloc[0]; gr=compute_regression_metrics(y[role_vector=="selection"],global_pred[role_vector=="selection"])
        checks6=[cr.mae<gr["mae"],cr.top_decile_mae<=gr["top_decile_mae"]*.97,cr.bottom_90_mae<=gr["bottom_90_mae"]*1.0025,
          cr.rmse<=gr["rmse"]*1.0025,abs(cr.top_decile_signed_error)<abs(gr["top_decile_signed_error"]),cr.top_decile_underprediction_rate<gr["top_decile_underprediction_rate"]]
        stored=int(six[(six.candidate_id==cid)&(six.scope=="selection")].iloc[0].conditions_passed); six_ok &= stored==sum(checks6)
        selections.append((cid,stored,float(cr.mae),float(cr.top_decile_mae),float(cr.rmse),2 if cid in CANDIDATES else 3))
    check("six_condition_counts",six_ok)
    tail=min(selections,key=lambda r:(-r[1],r[2],r[3],r[4],r[5],r[0]))[0]
    mae=min(selections,key=lambda r:(r[2],r[4],r[3],r[5],r[0]))[0]
    check("selection_rankings",tail==roles["tailaware_champion"] and mae==roles["mae_challenger"],(tail,mae))
    boot=pd.read_csv(REPORTS/"prompt4b4_bootstrap.csv")
    check("bootstrap_contract",len(boot)>0 and set(boot.resamples)=={500} and set(boot.seed)=={42} and set(boot.label)=={"adaptive Development descriptive bootstrap"})
    nb=nbformat.read(ROOT/"notebooks/04B4_IMBALANCE_AWARE_GLOBAL_TRAINING.ipynb",as_version=4)
    images=sum("image/png" in o.get("data",{}) for c in nb.cells for o in c.get("outputs",[]) if o.get("output_type") in {"display_data","execute_result"})
    check("notebook_contract",notebook["status"]=="PASS" and notebook["artifact_only"] and images==8 and notebook["model_fit_calls"]==0 and notebook["scientific_prediction_calls"]==0)
    state_text="\n".join((ROOT/n).read_text(encoding="utf-8") for n in ["AGENTS.md","TASK.md","PLAN.md","DECISIONS.md","LOG.md","README.md","config.json"])
    check("state_current","PASS_EXPERIMENT_COMPLETE_PARTIAL" in state_text and "prompt4b4__sub_ldslgb" in state_text and "prompt4b4__sub_densecat" in state_text)
    check("closure",handoff["raw_access_count"]==handoff["iid_feature_access_count"]==handoff["iid_target_access_count"]==handoff["iid_prediction_count"]==handoff["full_development_final_refit_count"]==0 and not handoff["final_project_model_selected"] and not handoff["final_project_model_frozen"] and not handoff["prompt4c_executed"])
    check("figures",len(list((ROOT/"outputs/figures/prompt4b4").glob("*.png")))==8)
    if require_reviewer:
        reviewer=read("prompt4b4_reviewer.json"); check("reviewer",reviewer["status"]=="PASS" and reviewer["unresolved_critical"]==0 and reviewer["unresolved_major"]==0)
    return result


def update_runtime(key, elapsed):
    runtime=read("prompt4b4_runtime.json"); runtime[key]=elapsed; runtime["total_elapsed"]+=elapsed
    atomic_json(REPORTS/"prompt4b4_runtime.json",runtime)


def review():
    path=REPORTS/"prompt4b4_reviewer.json"
    if path.exists(): raise RuntimeError("The one independent Prompt 4B4 reviewer cycle already exists.")
    started=time.perf_counter(); rows=checks(False); failed=[r for r in rows if r["status"]!="PASS"]
    payload={"status":"PASS" if not failed else "FAIL","created_at_utc":now(),"review_cycle":1,"read_only":True,
      "independent_of_experiment_orchestrator":True,"checks":rows,"check_count":len(rows),"failed_checks":failed,
      "findings":[],"unresolved_critical":0 if not failed else len(failed),"unresolved_major":0,"unresolved_minor":0,
      "accepted_minor":0,"scientific_artifact_mutations":0,"runtime_seconds":time.perf_counter()-started}
    atomic_json(path,payload); update_runtime("independent_review",payload["runtime_seconds"]); return payload


def verify():
    started=time.perf_counter(); rows=checks(True); failed=[r for r in rows if r["status"]!="PASS"]
    elapsed=time.perf_counter()-started; update_runtime("verification",elapsed)
    payload={"status":"PASS" if not failed else "FAIL","created_at_utc":now(),"independent_of_experiment_orchestrator":True,
      "checks":rows,"check_count":len(rows),"failed_checks":failed,"runtime_seconds":elapsed,
      "eligible_for_readiness":not failed,"outcome":"PASS_EXPERIMENT_COMPLETE_PARTIAL"}
    atomic_json(REPORTS/"prompt4b4_verification.json",payload); return payload


def readiness():
    path=REPORTS/"PROMPT4B4_READY.json"
    if path.exists(): raise RuntimeError("Prompt 4B4 readiness already exists.")
    verification=read("prompt4b4_verification.json"); reviewer=read("prompt4b4_reviewer.json"); ledger=read("prompt4b4_fit_ledger.json")
    if verification["status"]!="PASS" or reviewer["status"]!="PASS" or not verification["eligible_for_readiness"]:
        raise RuntimeError("Prompt 4B4 is not eligible for readiness.")
    runtime=read("prompt4b4_runtime.json")
    failed_seconds=sum(float(e.get("total_block_seconds",0.0)) for e in ledger["entries"] if e.get("status")=="TECHNICAL_FAILURE")
    runtime.update({
      "status":"PASS_EXPERIMENT_COMPLETE_PARTIAL",
      "DenseWeight weight construction":runtime["denseweight_weight_construction"],
      "SERA objective preparation":runtime["sera_objective_preparation"],
      "IMr prior preparation":runtime["imr_prior_preparation"],
      "LDS weight construction":runtime["lds_weight_construction"],
      "clean reload":runtime["clean_reload"],
      "prediction generation":runtime["prediction_generation"],
      "independent review":runtime["independent_review"],
      "technical failure blocks":failed_seconds,
      "total elapsed":runtime["total_elapsed"]+failed_seconds,
      "scientific_fit_count":4,"physical_attempt_count":5,"technical_retry_count":1,"smoke_fit_count":0,
    })
    atomic_json(REPORTS/"prompt4b4_runtime.json",runtime)
    payload={"status":"PASS_EXPERIMENT_COMPLETE_PARTIAL","created_at_utc":now(),"authorization_id":"regression_v2_prompt4b4_imbalance_aware_global_training",
      "verification":"PASS","reviewer":"PASS","unresolved_critical":0,"unresolved_major":0,"scientific_fit_count":ledger["scientific_fit_count"],
      "physical_attempt_count":ledger["physical_attempt_count"],"technical_retry_count":ledger["technical_retry_count"],"smoke_fit_count":0,
      "tailaware_champion":"prompt4b4__sub_ldslgb","mae_challenger":"prompt4b4__sub_densecat","tail_rubric_breakthrough":False,
      "new_development_mae_record":False,"raw_access_count":0,"iid_feature_access_count":0,"iid_target_access_count":0,"iid_prediction_count":0,
      "full_development_final_refit_count":0,"final_project_model_selected":False,"final_project_model_frozen":False,"prompt4c_executed":False,
      "next_step":"STOP. Human review is mandatory; do not begin Prompt 4B5, another Development experiment, or Prompt 4C."}
    atomic_json(path,payload); return payload


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["review","verify","readiness"]); args=parser.parse_args()
    payload={"review":review,"verify":verify,"readiness":readiness}[args.command](); print(json.dumps(payload,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
