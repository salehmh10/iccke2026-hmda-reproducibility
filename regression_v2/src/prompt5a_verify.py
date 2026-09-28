"""Independent saved-artifact review and verification for Prompt 5A.

This module never opens an original IID file, loads a model, or generates a
prediction. It uses only sealed Prompt 5A artifacts and the immutable freeze.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


AUTHORIZATION_ID = "regression_v2_prompt5a_one_time_iid_evaluation"
STATUS_PASS = "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS"
FREEZE_SHA = "8f2b8e4fda80056770b916b7859ad0f6f89f2948236320e6815e082fadca35c1"
PRIMARY_SHA = "5349ab15fd1c8182ef539f435cc4f065e71ee7da047de09a4dc271c9097e0c08"
GLOBAL_SHA = "6f61a0be1fc90d2331b08dada63a453f6c5410d5783f46f1e0cbcbf425f75597"
PRIMARY = "final_primary_stage3_500k"
GLOBAL = "final_global_500k"
REPORTS = Path("outputs/reports")
POST_IID = Path("outputs/data/post_iid")
PREDICTIONS = Path("outputs/predictions/prompt5a/iid")
FIGURES = Path("outputs/figures/prompt5a")
NOTEBOOK = Path("notebooks/05A_ONE_TIME_IID_EVALUATION_AND_ERROR_ANALYSIS.ipynb")
EVALUATION = POST_IID / "iid_evaluation_frame.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_path(value: str | Path | None = None) -> Path:
    return Path(value or Path(__file__).resolve().parents[1]).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(root: Path, relative: Path) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def atomic_json(root: Path, relative: Path, payload: dict[str, Any]) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def row_digest(values: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values.astype(str):
        encoded = value.encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii")); digest.update(b":"); digest.update(encoded); digest.update(b"\n")
    return digest.hexdigest()


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    error = p - y; absolute = np.abs(error)
    top10 = y >= float(np.quantile(y, .90)); top05 = y >= float(np.quantile(y, .95))
    q85 = float(np.quantile(y, .85)); q95 = float(np.quantile(y, .95)); boundary = (y >= q85) & (y <= q95)
    valid = y > 0.0
    return {
        "n_rows": int(y.size), "mae": float(np.mean(absolute)), "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - np.sum(error**2) / np.sum((y - np.mean(y))**2)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(np.clip(p, 0.0, None)) - np.log1p(y))**2))),
        "median_absolute_error": float(np.median(absolute)), "p90_absolute_error": float(np.quantile(absolute, .90)),
        "mean_signed_error": float(np.mean(error)), "negative_prediction_count": int(np.count_nonzero(p < 0.0)),
        "negative_prediction_rate": float(np.mean(p < 0.0)), "bottom_90_mae": float(np.mean(absolute[~top10])),
        "top_decile_mae": float(np.mean(absolute[top10])), "top_five_percent_mae": float(np.mean(absolute[top05])),
        "top_decile_signed_error": float(np.mean(error[top10])), "top_five_percent_signed_error": float(np.mean(error[top05])),
        "top_decile_underprediction_rate": float(np.mean(error[top10] < 0.0)), "top_five_percent_underprediction_rate": float(np.mean(error[top05] < 0.0)),
        "p85_to_p95_boundary_mae": float(np.mean(absolute[boundary])), "top_decile_rows": int(top10.sum()),
        "top_five_percent_rows": int(top05.sum()), "boundary_rows": int(boundary.sum()),
        "mape_percent": float(100.0 * np.mean(absolute[valid] / y[valid])) if valid.any() else np.nan,
        "mape_invalid_nonpositive_rows": int((~valid).sum()), "mape_valid_rows": int(valid.sum()), "mape_valid_coverage": float(np.mean(valid)),
        "wape_percent": float(100.0 * np.sum(absolute) / np.sum(y)),
    }


def close(a: Any, b: Any, tolerance: float = 1e-10) -> bool:
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=tolerance, equal_nan=True))


def independent_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["y_true"].to_numpy(float); p = frame["primary_prediction"].to_numpy(float); g = frame["global_prediction"].to_numpy(float)
    top10 = y >= float(np.quantile(y, .90)); top05 = y >= float(np.quantile(y, .95))
    specs = {"MAE": None, "RMSE": None, "MAPE": None, "WAPE": None, "Bottom-90 MAE": ~top10, "Top-decile MAE": top10, "Top-5% MAE": top05}
    def value(name: str, yy: np.ndarray, pp: np.ndarray, mask: np.ndarray | None) -> float:
        if mask is not None: yy, pp = yy[mask], pp[mask]
        error = pp - yy
        if name.endswith("MAE"): return float(np.mean(np.abs(error)))
        if name == "RMSE": return float(np.sqrt(np.mean(error**2)))
        if name == "MAPE": return float(100.0 * np.mean(np.abs(error[yy > 0.0]) / yy[yy > 0.0]))
        return float(100.0 * np.sum(np.abs(error)) / np.sum(yy))
    observed = {name: value(name,y,p,mask)-value(name,y,g,mask) for name,mask in specs.items()}
    samples = {name: np.empty(500) for name in specs}; rng=np.random.default_rng(42)
    for index in range(500):
        chosen=rng.integers(0,len(y),size=len(y)); yy=y[chosen]; pp=p[chosen]; gg=g[chosen]
        for name,base_mask in specs.items():
            mask=None if base_mask is None else base_mask[chosen]
            samples[name][index]=value(name,yy,pp,mask)-value(name,yy,gg,mask)
    return pd.DataFrame([{"metric":name,"observed_difference_primary_minus_global":observed[name],"bootstrap_mean":float(v.mean()),"bootstrap_median":float(np.median(v)),"percentile_2_5":float(np.quantile(v,.025)),"percentile_97_5":float(np.quantile(v,.975)),"fraction_primary_lower":float(np.mean(v<0)),"n_resamples":500,"seed":42} for name,v in samples.items()])


def audit(root: Path, *, reproduce_bootstrap: bool) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []; failures: list[str] = []
    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence})
        if not condition: failures.append(name)

    freeze = read_json(root, REPORTS / "FINAL_PRE_IID_FREEZE.json")
    ledger = read_json(root, REPORTS / "prompt5a_iid_access_ledger.json")
    lock = read_json(root, REPORTS / "prompt5a_prediction_lock.json")
    manifest = read_json(root, REPORTS / "prompt5a_post_iid_snapshot_manifest.json")
    check("exact pre-IID freeze hash", sha256(root / REPORTS / "FINAL_PRE_IID_FREEZE.json") == FREEZE_SHA, FREEZE_SHA)
    check("exact Primary bundle hash", sha256(root / "outputs/models/final_pre_iid/primary_stage3/bundle.joblib") == PRIMARY_SHA, PRIMARY_SHA)
    check("exact Global bundle hash", sha256(root / "outputs/models/final_pre_iid/global_comparator/bundle.joblib") == GLOBAL_SHA, GLOBAL_SHA)
    check("Block A excluded before IID", freeze["historical_blocka"]["iid_eligible"] is False and set(lock["predictions"]) == {PRIMARY, GLOBAL}, {"eligible": freeze["historical_blocka"]["iid_eligible"], "models": list(lock["predictions"])})

    events = [item["event"] for item in ledger["events"]]
    required_order = ["feature_read_success", "prediction_generation_complete", "prediction_lock_created_and_reloaded", "target_read_success", "target_snapshot_persisted_and_reloaded", "evaluation_complete"]
    order_ok = all(name in events for name in required_order) and [events.index(name) for name in required_order] == sorted(events.index(name) for name in required_order)
    check("access order", order_ok, required_order)
    check("access counts", ledger["feature_successful_content_reads"] == 1 and ledger["target_successful_content_reads"] == 1, {"features":ledger["feature_successful_content_reads"],"targets":ledger["target_successful_content_reads"]})
    check("zero post-target predictions", ledger.get("model_prediction_calls_after_target_access") == 0 and lock.get("model_prediction_calls_after_target_access") == 0, 0)
    check("zero fits calibration and tuning", all(ledger.get(key)==0 for key in ("fit_count","refit_count","tuning_operation_count")) and ledger.get("post_iid_model_changes")==0, {key:ledger.get(key) for key in ("fit_count","refit_count","tuning_operation_count","post_iid_model_changes")})

    predictions = {}
    prediction_ok = True
    for model_id in (PRIMARY, GLOBAL):
        item=lock["predictions"][model_id]; path=root/item["path"]; data=pd.read_parquet(path); predictions[model_id]=data
        prediction_ok &= sha256(path)==item["sha256"] and len(data)==75_000 and data["row_hash"].is_unique and row_digest(data["row_hash"])==lock["iid_feature_snapshot_row_digest"] and np.isfinite(data["prediction"]).all() and "loan_amount_000s" not in data
    check("prediction integrity", prediction_ok, {key:lock["predictions"][key]["sha256"] for key in (PRIMARY,GLOBAL)})
    target=pd.read_parquet(root/POST_IID/"iid_target_snapshot.parquet")
    target_ok=len(target)==75_000 and list(target.columns)==["row_hash","loan_amount_000s"] and target["row_hash"].is_unique and np.isfinite(target["loan_amount_000s"]).all() and set(target["row_hash"].astype(str))==set(predictions[PRIMARY]["row_hash"].astype(str))
    check("target snapshot integrity", target_ok, {"rows":len(target),"unique":int(target.row_hash.nunique()),"positive":int((target.loan_amount_000s>0).sum())})

    frame=pd.read_parquet(root/EVALUATION); y=frame["y_true"].to_numpy(float)
    check("evaluation alignment", len(frame)==75_000 and frame["row_hash"].is_unique and row_digest(frame["row_hash"])==lock["iid_feature_snapshot_row_digest"], {"rows":len(frame),"row_digest":row_digest(frame["row_hash"])})
    formula_ok=np.allclose(frame["primary_signed_error"],frame["primary_prediction"]-y,rtol=0,atol=0) and np.allclose(frame["global_signed_error"],frame["global_prediction"]-y,rtol=0,atol=0) and np.allclose(frame["delta_abs_error"],frame["primary_abs_error"]-frame["global_abs_error"],rtol=0,atol=0) and np.array_equal(frame["primary_better_flag"].to_numpy(bool),frame["delta_abs_error"].to_numpy()<0)
    check("row-level error formulas", formula_ok, "signed, absolute, delta, and win flag reproduce")

    reported=pd.read_csv(root/REPORTS/"prompt5a_iid_overall_metrics.csv").set_index("model_id")
    metric_failures=[]
    for model_id,column in ((PRIMARY,"primary_prediction"),(GLOBAL,"global_prediction")):
        calculated=metric_values(y,frame[column].to_numpy(float))
        for key,value in calculated.items():
            if not close(value,reported.loc[model_id,key]): metric_failures.append(f"{model_id}:{key}")
    check("overall metrics including MAPE WAPE and Tail", not metric_failures, metric_failures)

    labels=pd.qcut(pd.Series(y),10,labels=False,duplicates="drop").to_numpy()+1
    decile_ok=np.array_equal(np.array([int(value[1:]) for value in frame["iid_local_decile"]]),labels)
    decile_report=pd.read_csv(root/REPORTS/"prompt5a_iid_local_decile_metrics.csv")
    decile_ok &= len(decile_report)==20 and set(decile_report["decile"])=={f"D{i}" for i in range(1,11)}
    check("IID-local decile reproduction", decile_ok, {"rows":len(decile_report),"deciles":int(np.unique(labels).size)})
    cutpoints=np.array([freeze["iid_protocol"]["development_frozen_target_cutpoints"][f"q{i:02d}"] for i in range(10,100,10)],float)
    bands=np.searchsorted(cutpoints,y,side="left")+1
    band_report=pd.read_csv(root/REPORTS/"prompt5a_development_frozen_band_metrics.csv")
    band_ok=np.array_equal(bands,frame["development_frozen_band_index"].to_numpy(int)) and len(band_report)==20
    check("Development-frozen band reproduction", band_ok, {"cutpoints":cutpoints.tolist(),"rows":len(band_report)})

    pm=reported.loc[PRIMARY]; gm=reported.loc[GLOBAL]
    six_expected=[pm.mae<gm.mae,pm.top_decile_mae<=gm.top_decile_mae*.97,pm.bottom_90_mae<=gm.bottom_90_mae*1.0025,pm.rmse<=gm.rmse*1.0025,abs(pm.top_decile_signed_error)<abs(gm.top_decile_signed_error),pm.top_decile_underprediction_rate<gm.top_decile_underprediction_rate]
    six=read_json(root,REPORTS/"prompt5a_iid_six_condition_check.json")
    check("six-condition reproduction", six["conditions_passed"]==sum(six_expected) and [item["status"]=="PASS" for item in six["conditions"]]==six_expected, {"expected":six_expected,"reported":six["conditions_passed"]})

    if reproduce_bootstrap:
        calculated=independent_bootstrap(frame); saved=pd.read_csv(root/REPORTS/"prompt5a_iid_bootstrap.csv")
        boot_ok=list(calculated.metric)==list(saved.metric) and all(close(calculated.loc[i,key],saved.loc[i,key],1e-11) for i in range(7) for key in ("observed_difference_primary_minus_global","bootstrap_mean","bootstrap_median","percentile_2_5","percentile_97_5","fraction_primary_lower")) and (saved.n_resamples==500).all() and (saved.seed==42).all()
        check("paired bootstrap reproduction", boot_ok, {"resamples":500,"seed":42,"metrics":saved.metric.tolist()})

    rowwise=pd.read_csv(root/REPORTS/"prompt5a_primary_vs_global_rowwise.csv"); all_row=rowwise[rowwise.scope.eq("All IID")].iloc[0]; delta=frame.delta_abs_error.to_numpy(float)
    rowwise_ok=close(all_row.fraction_primary_better,np.mean(delta<0)) and close(all_row.fraction_global_better,np.mean(delta>0)) and int(all_row.exact_ties)==int(np.sum(delta==0)) and len(rowwise)==11
    check("Primary versus Global rowwise analysis", rowwise_ok, {"rows":len(rowwise),"primary_win_rate":float(np.mean(delta<0)),"global_win_rate":float(np.mean(delta>0))})
    required_error_reports=("prompt5a_error_distribution.csv","prompt5a_large_error_analysis.csv","prompt5a_underprediction_analysis.csv","prompt5a_routing_diagnostics.csv","prompt5a_development_iid_transport.csv")
    check("detailed error reports", all((root/REPORTS/name).is_file() and (root/REPORTS/name).stat().st_size>100 for name in required_error_reports), list(required_error_reports))
    routed=frame.meta_gate_probability.to_numpy(float)>.75; strength=np.zeros(len(frame)); strength[routed]=.75*(frame.loc[routed,"meta_gate_probability"].to_numpy(float)-.75)/.25
    routing_ok=np.array_equal(routed,frame.routing_condition_activated.to_numpy(bool)) and np.allclose(strength,frame.routing_strength.to_numpy(float),rtol=0,atol=1e-15) and np.allclose(frame.primary_prediction,frame.global_base_prediction+frame.applied_residual_correction,rtol=0,atol=0)
    check("frozen routing diagnostics", routing_ok, {"routed_rows":int(routed.sum()),"routed_fraction":float(routed.mean())})

    manifest_failures=[]
    for name,item in manifest["artifacts"].items():
        path=root/item["path"]
        if not path.is_file() or sha256(path)!=item["sha256"] or pq.ParquetFile(path).metadata.num_rows!=item["rows"]: manifest_failures.append(name)
    check("post-IID snapshot manifest", not manifest_failures and len(manifest["artifacts"])==6, {"artifacts":len(manifest["artifacts"]),"failures":manifest_failures})
    figure_files=sorted((root/FIGURES).glob("*.png")); plot_files=sorted((root/REPORTS).glob("prompt5a_plot_*.csv"))
    check("statistical figures and plotting data", len(figure_files)==11 and all(path.stat().st_size>10_000 for path in figure_files) and len(plot_files)>=6, {"figures":len(figure_files),"plot_data":len(plot_files)})

    notebook=nbformat.read(root/NOTEBOOK,as_version=4); code=[cell for cell in notebook.cells if cell.cell_type=="code"]; outputs=[out for cell in code for out in cell.get("outputs",[])]; sources="\n".join(cell.source for cell in code)
    notebook_ok=not any(out.get("output_type")=="error" for out in outputs) and sum("image/png" in out.get("data",{}) for out in outputs)>=11 and sum("text/html" in out.get("data",{}) for out in outputs)>=10 and not any(token in sources for token in ("iid_holdout_features","iid_holdout_targets",".fit(",".predict(","joblib.load"))
    check("artifact-only inline notebook", notebook_ok, {"code_cells":len(code),"images":sum("image/png" in out.get("data",{}) for out in outputs),"tables":sum("text/html" in out.get("data",{}) for out in outputs)})
    tree=ast.parse((root/"src/prompt5a_evaluation.py").read_text(encoding="utf-8")); training_calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in {"fit","fit_transform","partial_fit"}]
    check("static zero-training implementation", not training_calls, training_calls)
    check("closure and unchanged Primary", ledger["original_feature_status"]=="ORIGINAL_IID_CONTENT_CLOSED_AFTER_PROMPT5A" and ledger["original_target_status"]=="ORIGINAL_IID_CONTENT_CLOSED_AFTER_PROMPT5A" and ledger["post_iid_model_changes"]==0, {"feature":ledger["original_feature_status"],"target":ledger["original_target_status"],"changes":ledger["post_iid_model_changes"]})
    return checks, failures


def review(root: Path) -> dict[str, Any]:
    path=root/REPORTS/"prompt5a_reviewer.json"
    if path.exists(): raise RuntimeError("The single Prompt 5A reviewer cycle has already been used.")
    started=time.perf_counter(); checks,failures=audit(root,reproduce_bootstrap=True)
    payload={"status":"PASS" if not failures else "FAIL","created_at_utc":utc_now(),"elapsed_seconds":time.perf_counter()-started,"review_type":"independent_saved_artifact_read_only","reviewer_count":1,"check_count":len(checks),"checks":checks,"findings":failures,"required_repairs":failures,"scope_guard":{"scientific_artifacts_written":0,"review_report_written":1,"models_loaded":0,"predictions_generated":0,"models_fitted":0,"original_iid_accesses":0}}
    atomic_json(root,REPORTS/"prompt5a_reviewer.json",payload)
    return payload


def verify(root: Path) -> dict[str, Any]:
    started=time.perf_counter(); reviewer=read_json(root,REPORTS/"prompt5a_reviewer.json"); candidate=read_json(root,REPORTS/"prompt5a_final_iid_evaluation_candidate.json")
    checks,failures=audit(root,reproduce_bootstrap=True)
    extra=[]
    extra.append({"check":"single reviewer PASS","status":"PASS" if reviewer.get("status")=="PASS" and reviewer.get("reviewer_count")==1 else "FAIL","evidence":{"status":reviewer.get("status"),"count":reviewer.get("reviewer_count")}})
    if extra[-1]["status"]=="FAIL": failures.append("single reviewer PASS")
    candidate_ok=candidate.get("no_fit") and candidate.get("no_tuning") and candidate.get("no_model_change") and candidate.get("evaluation_frame_sha256")==sha256(root/EVALUATION) and candidate.get("prediction_lock_sha256")==sha256(root/REPORTS/"prompt5a_prediction_lock.json")
    extra.append({"check":"candidate handoff integrity","status":"PASS" if candidate_ok else "FAIL","evidence":{"evaluation_frame_sha256":candidate.get("evaluation_frame_sha256"),"prediction_lock_sha256":candidate.get("prediction_lock_sha256")}})
    if not candidate_ok: failures.append("candidate handoff integrity")
    payload={"status":"PASS" if not failures else "FAIL","created_at_utc":utc_now(),"authorization_id":AUTHORIZATION_ID,"independent_of_prediction_and_evaluation_orchestration":True,"elapsed_seconds":time.perf_counter()-started,"check_count":len(checks)+len(extra),"checks":checks+extra,"failures":failures,"read_counts":{"features":1,"targets":1},"model_count":2,"prediction_rows_total":150_000,"fits":0,"refits":0,"tuning":0,"model_changes":0}
    atomic_json(root,REPORTS/"prompt5a_verification.json",payload)
    return payload


def promote(root: Path) -> dict[str, Any]:
    candidate=read_json(root,REPORTS/"prompt5a_final_iid_evaluation_candidate.json"); reviewer=read_json(root,REPORTS/"prompt5a_reviewer.json"); verification=read_json(root,REPORTS/"prompt5a_verification.json")
    if reviewer["status"]!="PASS" or verification["status"]!="PASS": raise RuntimeError("Review and verification must pass before promotion.")
    payload=dict(candidate); payload["status"]=STATUS_PASS; payload["promotion"]={"promoted_at_utc":utc_now(),"candidate_sha256":sha256(root/REPORTS/"prompt5a_final_iid_evaluation_candidate.json"),"reviewer_sha256":sha256(root/REPORTS/"prompt5a_reviewer.json"),"verification_sha256":sha256(root/REPORTS/"prompt5a_verification.json"),"reviewer_status":"PASS","verification_status":"PASS"}
    path=atomic_json(root,REPORTS/"FINAL_IID_EVALUATION.json",payload); loaded=json.loads(path.read_text(encoding="utf-8"))
    if loaded["status"]!=STATUS_PASS or loaded["evaluation_frame_sha256"]!=sha256(root/EVALUATION): raise RuntimeError("Promoted IID handoff reload failed.")
    return {"status":loaded["status"],"sha256":sha256(path)}


def ready(root: Path) -> dict[str, Any]:
    final_path=root/REPORTS/"FINAL_IID_EVALUATION.json"; final=json.loads(final_path.read_text(encoding="utf-8")); reviewer=read_json(root,REPORTS/"prompt5a_reviewer.json"); verification=read_json(root,REPORTS/"prompt5a_verification.json"); ledger=read_json(root,REPORTS/"prompt5a_iid_access_ledger.json"); lock=read_json(root,REPORTS/"prompt5a_prediction_lock.json"); overall=pd.read_csv(root/REPORTS/"prompt5a_iid_overall_metrics.csv").set_index("model_id"); six=read_json(root,REPORTS/"prompt5a_iid_six_condition_check.json")
    if final["status"]!=STATUS_PASS or reviewer["status"]!="PASS" or verification["status"]!="PASS": raise RuntimeError("Prompt 5A is not ready.")
    payload={"status":STATUS_PASS,"created_at_utc":utc_now(),"authorization_id":AUTHORIZATION_ID,"prompt5a_complete":True,"final_pre_iid_freeze_sha256":FREEZE_SHA,"final_iid_evaluation_path":"outputs/reports/FINAL_IID_EVALUATION.json","final_iid_evaluation_sha256":sha256(final_path),"prediction_lock_sha256":sha256(root/REPORTS/"prompt5a_prediction_lock.json"),"evaluation_frame_sha256":sha256(root/EVALUATION),"primary_mae":float(overall.loc[PRIMARY,"mae"]),"global_mae":float(overall.loc[GLOBAL,"mae"]),"six_conditions_passed":int(six["conditions_passed"]),"reviewer_status":"PASS","verification_status":"PASS","access_counts":{"feature_successful_content_reads":ledger["feature_successful_content_reads"],"target_successful_content_reads":ledger["target_successful_content_reads"],"evaluated_models":ledger["prediction_models"],"prediction_rows_total":ledger["prediction_rows"]},"no_fit":True,"no_refit":True,"no_tuning":True,"no_model_change":True,"original_iid_files_status":"CLOSED_AFTER_ONE_TIME_EVALUATION","next_step":"Prompt 5B - Fairness/Sensitive Analysis and Final Explainability","report_creation_rule":"This is the last Prompt 5A report artifact."}
    atomic_json(root,REPORTS/"PROMPT5A_READY.json",payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("review","verify","promote","ready")); parser.add_argument("--root",default=None); return parser.parse_args()


def main() -> None:
    args=parse_args(); root=root_path(args.root)
    result={"review":review,"verify":verify,"promote":promote,"ready":ready}[args.command](root)
    print(json.dumps(result,indent=2,default=str))


if __name__=="__main__":
    main()
