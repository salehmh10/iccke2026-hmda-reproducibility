"""Bounded orchestration for Prompt 4B4 imbalance-aware Global training.

Only the frozen Development Parquet and saved Prompt 2-4B3 artifacts are read.
Raw and IID paths are rejected by the read guard. Scientific fits are sequential
and resumable; valid saved fits are never repeated.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import psutil
from nbclient import NotebookClient
from sklearn.mixture import GaussianMixture

from .ensemble_utils import make_validation_selection_audit_split
from .model_bundles import ModelBundle, atomic_joblib_dump, load_bundle
from .preprocessing import CatBoostFramePreprocessor, make_dense_tree_preprocessor, make_xgb_preprocessor
from .prompt4_metrics import compute_regression_metrics
from .prompt4b4_methods import (
    IMrObjective, SERAObjective, denseweight_direct_from_density, denseweight_official,
    fit_imr_gmm, fixed_substitution_predictions, imr_gradient, lds_weights, select_mae_challenger,
    select_tailaware, sera_bruteforce_sigma, sera_sigma, six_condition_evaluator,
    scientific_resume_action, validate_feature_contract,
)


SEED = 42
THREADS = 4
TARGET = "loan_amount_000s"
CONTRACT = "main_without_sensitive_without_lender"
EXPECTED_SOURCE_SHA = "0ed232397be3ec4de1483c594954dce7b4704b375ca295397d899323dc4f0b6b"
EXPECTED_TRAIN_DIGEST = "26265d75d8fa35d7417e2a9fb2888b9f625e6d19123cd7a2972974f403a30166"
EXPECTED_VALIDATION_DIGEST = "676b577233627c8b237d214a29eeb798e099d028583c8b73068b151a2a204290"
EXPECTED_SELECTION_DIGEST = "3af54200bbda79bbe910903b4a32c217d409b19b07cf0f54cd2076d9bf0db03a"
EXPECTED_AUDIT_DIGEST = "f7dcb7ee23049a24564f474f5e1730dde756bf22158e7cd417c90a78b15b1c48"
AUTHORIZATION = "regression_v2_prompt4b4_imbalance_aware_global_training"

REPORTS = Path("outputs/reports")
DATA = Path("outputs/data/development.parquet")
TMP = Path("outputs/tmp/prompt4b4")
MODELS = Path("outputs/models/prompt4b4")
PREDICTIONS = Path("outputs/predictions/prompt4b4/validation")
FIGURES = Path("outputs/figures/prompt4b4")
NOTEBOOK = Path("notebooks/04B4_IMBALANCE_AWARE_GLOBAL_TRAINING.ipynb")

CANDIDATES = [
    "prompt4b4__denseweight_cat", "prompt4b4__sera_xgb",
    "prompt4b4__imrgb", "prompt4b4__lds_lgb",
]
SUBSTITUTIONS = [
    "prompt4b4__sub_densecat", "prompt4b4__sub_seraxgb",
    "prompt4b4__sub_imrgb", "prompt4b4__sub_ldslgb",
]
METHOD_OF = dict(zip(CANDIDATES, ("DenseWeight", "SERA", "IMr-GB", "LDS")))
FAMILY_OF = dict(zip(CANDIDATES, ("catboost", "xgboost", "xgboost", "lightgbm")))

PRIOR_REPORTS = [
    "PROMPT4B3_READY.json", "prompt4b3_verification.json", "prompt4b3_reviewer.json",
    "PROMPT4B2_READY.json", "PROMPT4B_READY.json", "PROMPT4A_READY.json",
    "PROMPT3_READY.json", "PROMPT2_READY.json", "DATA_READY.json",
    "final_verification.json", "feature_roles.json",
]


def root_path(root: str | Path | None = None) -> Path:
    result = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    if not (result / "AGENTS.md").exists() or not (result / DATA).exists():
        raise RuntimeError("Prompt 4B4 project root is invalid.")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, Path): return value.as_posix()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def ordered_digest(values: Any) -> str:
    return hashlib.sha256("\n".join(pd.Series(values).astype(str).tolist()).encode()).hexdigest()


def atomic_json(root: Path, relative: Path, payload: dict[str, Any]) -> Path:
    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    json.loads(temp.read_text(encoding="utf-8")); os.replace(temp, path); return path


def atomic_csv(root: Path, relative: Path, frame: pd.DataFrame) -> Path:
    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp"); frame.to_csv(temp, index=False); os.replace(temp, path); return path


def atomic_parquet(root: Path, relative: Path, frame: pd.DataFrame) -> Path:
    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp"); frame.to_parquet(temp, index=False, compression="zstd")
    check = pd.read_parquet(temp)
    if len(check) != len(frame) or list(check.columns) != list(frame.columns): raise RuntimeError("Parquet reload failed.")
    os.replace(temp, path); return path


def read_json(root: Path, relative: Path) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def guarded_read(root: Path, relative: Path) -> Path:
    text = relative.as_posix().lower()
    forbidden = ("iid_holdout", "raw", "legacy")
    if relative != DATA and (any(token in text for token in forbidden) or "data" in {part.lower() for part in relative.parts}):
        raise PermissionError(f"Prompt 4B4 prohibited read path: {relative}")
    resolved = (root / relative).resolve()
    if root not in resolved.parents: raise PermissionError("Read escaped project root.")
    return resolved


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for key, package in (("numpy","numpy"),("pandas","pandas"),("scipy","scipy"),
                         ("scikit_learn","scikit-learn"),("denseweight","denseweight"),
                         ("KDEpy","KDEpy"),("catboost","catboost"),("lightgbm","lightgbm"),
                         ("xgboost","xgboost"),("joblib","joblib")):
        result[key] = importlib.metadata.version(package)
    return result


def load_context(root: Path):
    roles_report = read_json(root, REPORTS / "feature_roles.json")
    features = validate_feature_contract(roles_report["contracts"][CONTRACT])
    frame = pd.read_parquet(guarded_read(root, DATA), columns=features + [TARGET, "development_role", "row_hash"]).copy()
    train = frame.loc[frame["development_role"].eq("train")].reset_index(drop=True).copy()
    validation = frame.loc[frame["development_role"].eq("validation")].reset_index(drop=True).copy()
    if len(frame) != 500_000 or len(train) != 400_000 or len(validation) != 100_000: raise RuntimeError("Frozen role counts changed.")
    if sha256(root / DATA) != EXPECTED_SOURCE_SHA: raise RuntimeError("Development source hash changed.")
    if ordered_digest(train["row_hash"]) != EXPECTED_TRAIN_DIGEST or ordered_digest(validation["row_hash"]) != EXPECTED_VALIDATION_DIGEST:
        raise RuntimeError("Frozen Train/Validation identity changed.")
    if set(train["row_hash"]) & set(validation["row_hash"]): raise RuntimeError("Train/Validation overlap detected.")
    if not np.isfinite(frame[TARGET].to_numpy(float)).all(): raise RuntimeError("Target is non-finite.")
    selection, audit, role_vector, split = make_validation_selection_audit_split(validation[TARGET], validation["row_hash"])
    if split["selection_row_hash_digest"] != EXPECTED_SELECTION_DIGEST or split["audit_row_hash_digest"] != EXPECTED_AUDIT_DIGEST:
        raise RuntimeError("Frozen Selection/Audit roles changed.")
    return features, frame, train, validation, role_vector, split


def code_digest(root: Path) -> str:
    files = [root / "src/prompt4b4_methods.py", root / "src/prompt4b4_experiments.py",
             root / "src/prompt4b4_iron_reference.R", root / "src/prompt4b4_imr_torch_reference.py",
             root / "tests/test_prompt4b4_methods.py", root / "tests/test_prompt4b4_project_gate.py"]
    return canonical_digest({p.relative_to(root).as_posix(): sha256(p) for p in files})


def anchor_configs(root: Path) -> dict[str, Any]:
    design = read_json(root, REPORTS / "prompt2_frozen_design.json")
    by_name = {row["candidate_name"]: row for row in design["candidate_definitions"]}
    cat = dict(by_name["cat_raw_mae"]["parameters"])
    lgb = dict(by_name["lgb_raw_l1"]["parameters"])
    xgb = dict(by_name["xgb_log1p_l2_anchor"]["parameters"])
    sera = {**xgb, "objective": "custom_sera_trapezoidal", "target_mode_override": "raw"}
    imr = {**xgb, "objective": "custom_imr_balanced_mse", "target_mode_override": "raw"}
    # Prompt 4B4 explicitly prohibits an early-stopping search for IMr.
    imr.pop("early_stopping_rounds", None)
    imr["intentional_deviation"] = "Prompt 4B4 IMr rule: no early-stopping search"
    return {"cat_raw_mae": cat, "lgb_raw_l1": lgb, "xgb_log1p_l2_anchor_for_sera": sera,
            "xgb_log1p_l2_anchor_for_imr": imr}


def audit_weights(y: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    decile = pd.qcut(pd.Series(y), 10, labels=False, duplicates="drop").to_numpy()
    q90, q95 = np.quantile(y, [0.90, 0.95])
    return {"number_of_weights": len(weights), "finite_count": int(np.isfinite(weights).sum()),
            "min": weights.min(), "max": weights.max(), "mean": weights.mean(), "std": weights.std(),
            "p50": np.quantile(weights,.5), "p90": np.quantile(weights,.9), "p95": np.quantile(weights,.95),
            "p99": np.quantile(weights,.99),
            "mean_weight_by_train_target_decile": {str(i): float(weights[decile == i].mean()) for i in np.unique(decile)},
            "top_decile_mean_weight": weights[y >= q90].mean(), "top_five_percent_mean_weight": weights[y >= q95].mean(),
            "correlation_weight_target": np.corrcoef(weights, y)[0,1], "validation_target_involvement": False}


def source_reproduction(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    synthetic = np.asarray([-2., -1., -.5, 0., .25, 1., 3.])
    dw, estimator = denseweight_official(synthetic)
    dw_direct = denseweight_direct_from_density(estimator.y_dens, eps=estimator.eps)
    dense_diff = float(np.max(np.abs(dw - dw_direct)))
    phi = np.array([0., .001, .123, .5, .999, 1.])
    sera_diff = float(np.max(np.abs(sera_sigma(phi) - sera_bruteforce_sigma(phi))))
    reference_path = root / TMP / "imr_torch_reference.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    deep_python = root.parent / "artifacts/environment/stage5_env/Scripts/python.exe"
    subprocess.run([str(deep_python), str(root / "src/prompt4b4_imr_torch_reference.py"), str(reference_path)], check=True, cwd=root)
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    grad, hess = imr_gradient([0,1,3],[.2,.8,2.5],[0,2],[.7,.3],[1,2])
    imr_grad_diff = float(np.max(np.abs(grad - np.asarray(reference["gradient"]))))
    imr_hess_diff = float(np.max(np.abs(hess - np.asarray(reference["hessian"]))))
    lds_w, _, _, _ = lds_weights([1,1,2,3,10])
    payload = {
        "status": "PASS", "created_at_utc": utc_now(), "validation_metrics_inspected": False,
        "methods": {
            "DenseWeight": {"status":"PASS", "paper_title":"Density-based weighting for imbalanced regression", "year":2021,
                "identifier":"10.1007/s10994-021-06023-5", "official_source":"SteiMi/denseweight",
                "source_commit":"dd61852123a0e243b0393eaa7c077c71ed593dfd", "package_version":"0.1.2",
                "formula":"FFTKDE Silverman bandwidth; min-max density; max(1-alpha*density,1e-6); mean normalization",
                "configuration":{"alpha":1.0,"grid_points":4096,"eps":1e-6}, "synthetic_max_difference":dense_diff,
                "project_adaptation":"Official package output used directly as CatBoost Train sample weights."},
            "SERA": {"status":"PASS", "paper_title":"Model Optimization in Imbalanced Regression", "year":2022,
                "identifier":"arXiv:2206.09991", "official_source":"anibalsilva1/ModelOptimizationIR",
                "source_commit":"decacae4432aeb2c08d803f94d48c3de5ddddf68",
                "formula":"sigma(phi)=trapezoidal integral of 1[phi>=t]; grad=2*sigma*(pred-y); hess=2*sigma",
                "configuration":{"extreme_type":"high","adjusted_boxplot":True,"T":1000,"step":0.001,"target":"raw"},
                "synthetic_derivative_coefficient_max_difference":sera_diff,
                "project_adaptation":"IRon 0.1.5 computes Train-only relevance; Python uses the published exact derivatives."},
            "IMr-GB": {"status":"PASS", "paper_title":"A novel gradient boosting approach for imbalanced regression", "year":2024,
                "identifier":"10.1016/j.neucom.2024.128091", "official_source":"vengozhang/IMr-GB",
                "source_commit":"128e652e7eadedbbb658b96b89f23bdbb21b223e",
                "variant":"standard IMr-GB fixed six-component GMM", "formula":"tutorial Balanced-MSE gradient with smoothed GMM prior; unit Hessian as authors' code",
                "configuration":{"n_components":6,"noise_var":1.0,"random_state":42,"target":"raw"},
                "synthetic_gradient_max_difference_vs_official_torch_equation":imr_grad_diff,
                "synthetic_hessian_max_difference":imr_hess_diff,
                "bayes_path_disposition":"Authors' tutorial fixes BGM upper bound to 10% of Train size (40,000 components here); standard source path with exactly six components is frozen instead, before results.",
                "source_disclosure":"Root IMb_MSE.py has a loss-sign inconsistency; the paper-consistent tutorial equation is authoritative for the gradient."},
            "LDS": {"status":"PASS", "paper_title":"Delving into Deep Imbalanced Regression", "year":2021,
                "identifier":"PMLR 139; arXiv:2102.09554", "official_source":"YyzHarry/imbalanced-regression",
                "source_commit":"a6fdc45d45c04e6f5c40f43925bc66e580911084",
                "formula":"sqrt empirical frequency; convolve with max-normalized Gaussian window; inverse effective frequency; mean normalize",
                "configuration":{"bin_width":1.0,"reweight":"sqrt_inv","kernel":"gaussian","ks":5,"sigma":2},
                "synthetic_positive_mean_one": bool(np.all(lds_w > 0) and np.isclose(lds_w.mean(),1.0)),
                "project_adaptation":"Natural thousand-dollar target bins and LightGBM L1 sample weights."},
        }, "reproducible_method_count": 4, "minimum_required": 3,
        "runtime_seconds": time.perf_counter()-started,
    }
    if dense_diff > 1e-15 or sera_diff > 1e-15 or imr_grad_diff > 1e-10 or imr_hess_diff > 1e-15:
        payload["status"] = "BLOCKED_INSUFFICIENT_METHOD_REPRODUCTION"
    atomic_json(root, REPORTS / "prompt4b4_method_reproduction.json", payload)
    return payload


def prepare(root: str | Path | None = None) -> dict[str, Any]:
    workspace = root_path(root); overall = time.perf_counter()
    if (workspace / REPORTS / "prompt4b4_design_freeze.json").exists():
        design = read_json(workspace, REPORTS / "prompt4b4_design_freeze.json")
        if canonical_digest({k:v for k,v in design.items() if k != "design_digest"}) != design["design_digest"]:
            raise RuntimeError("Existing Prompt 4B4 design freeze digest is invalid.")
        return design
    preflight_start = time.perf_counter()
    features, frame, train, validation, role_vector, split = load_context(workspace)
    prior = []
    for name in PRIOR_REPORTS:
        path = workspace / REPORTS / name; payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status")
        if status not in {"PASS", "PASS_EXPERIMENT_COMPLETE_PARTIAL"}: raise RuntimeError(f"Prior handoff is not PASS: {name}")
        prior.append({"path":f"outputs/reports/{name}","sha256":sha256(path),"status":status})
    handoff = {"status":"PASS","created_at_utc":utc_now(),"prior_reports":prior,
        "development":{"path":DATA.as_posix(),"sha256":sha256(workspace/DATA),"rows":len(frame),"train_rows":len(train),"validation_rows":len(validation)},
        "memberships":{"train_ordered_digest":ordered_digest(train.row_hash),"validation_ordered_digest":ordered_digest(validation.row_hash),
                       "selection_ordered_digest":split["selection_row_hash_digest"],"audit_ordered_digest":split["audit_row_hash_digest"],"overlap":0},
        "feature_contract":CONTRACT,"feature_count":len(features),"features":features,"target_finite":True,
        "raw_access_count":0,"iid_feature_access_count":0,"iid_target_access_count":0,"iid_prediction_count":0,
        "full_development_final_refit_count":0,"final_project_model_selected":False,"final_project_model_frozen":False,"prompt4c_executed":False,
        "runtime_seconds":time.perf_counter()-preflight_start}
    atomic_json(workspace, REPORTS / "prompt4b4_handoff_validation.json", handoff)
    reproduction = source_reproduction(workspace)
    if reproduction["status"] != "PASS": raise RuntimeError(reproduction["status"])

    y = train[TARGET].to_numpy(float, copy=True)
    target_bins = np.linspace(float(y.min()), float(np.quantile(y, .995)), 81)
    target_counts, target_edges = np.histogram(y, bins=target_bins)
    atomic_csv(workspace, REPORTS/"prompt4b4_plot_target_distribution.csv", pd.DataFrame({
        "bin_left": target_edges[:-1], "bin_right": target_edges[1:], "train_count": target_counts,
    }))
    prep_runtime: dict[str,float] = {}
    start=time.perf_counter(); dense_w, dense_est = denseweight_official(y); prep_runtime["denseweight_weight_construction"]=time.perf_counter()-start
    dense_audit={"status":"PASS","created_at_utc":utc_now(),"method":"DenseWeight","alpha":1.0,
                 **audit_weights(y,dense_w),"mean_tolerance_pass":bool(np.isclose(dense_w.mean(),1,atol=1e-12))}
    atomic_json(workspace, REPORTS/"prompt4b4_denseweight_audit.json", dense_audit)
    dense_plot=pd.DataFrame({"target_grid":dense_est.x,"normalized_density":dense_est.y_dens_grid,
                             "training_weight":denseweight_direct_from_density(dense_est.y_dens_grid,eps=dense_est.eps)})
    atomic_csv(workspace, REPORTS/"prompt4b4_plot_denseweight.csv",dense_plot)

    start=time.perf_counter(); r_input=workspace/TMP/"train_targets_for_iron.csv"; r_output=workspace/TMP/"iron_relevance.csv"
    r_input.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame({TARGET:y}).to_csv(r_input,index=False)
    subprocess.run(["Rscript",str(workspace/"src/prompt4b4_iron_reference.R"),str(workspace/TMP/"r_libs"),str(r_input),str(r_output)],check=True,cwd=workspace)
    iron=pd.read_csv(r_output); relevance=iron.loc[iron.kind.eq("row"),"relevance"].to_numpy(float); control=iron.loc[iron.kind.eq("control")].copy()
    coefficient=sera_sigma(relevance); prep_runtime["sera_objective_preparation"]=time.perf_counter()-start
    if len(relevance)!=len(y) or not np.all(np.diff(relevance[np.argsort(y,kind="stable")])>=-1e-12): raise RuntimeError("SERA high relevance is not monotone.")
    sera_audit={"status":"PASS","created_at_utc":utc_now(),"type":"high","adjusted_boxplot":True,
        "control_points":control[["target","relevance","derivative"]].to_dict("records"),"train_target_min":y.min(),"train_target_max":y.max(),
        "mean_relevance":relevance.mean(),"relevance_quantiles":{str(q):np.quantile(relevance,q) for q in [0,.5,.9,.95,.99,1]},
        "relevance_gt_zero":int((relevance>0).sum()),"relevance_ge_half":int((relevance>=.5).sum()),"relevance_eq_one":int(np.isclose(relevance,1).sum()),
        "gradient_verification_max_difference":reproduction["methods"]["SERA"]["synthetic_derivative_coefficient_max_difference"],
        "hessian_verification_max_difference":reproduction["methods"]["SERA"]["synthetic_derivative_coefficient_max_difference"],
        "grid_T":1000,"validation_target_involvement":False}
    dec=pd.qcut(pd.Series(y),10,labels=False,duplicates="drop").to_numpy(); sera_audit["mean_relevance_by_target_decile"]={str(i):float(relevance[dec==i].mean()) for i in np.unique(dec)}
    atomic_json(workspace, REPORTS/"prompt4b4_sera_audit.json",sera_audit)
    atomic_csv(workspace, REPORTS/"prompt4b4_plot_sera.csv",pd.DataFrame({"target":y,"relevance":relevance,"derivative_coefficient":coefficient}).groupby("target",as_index=False).mean())

    start=time.perf_counter(); gmm, means, gmm_w, variances=fit_imr_gmm(y); prep_runtime["imr_prior_preparation"]=time.perf_counter()-start
    imr_audit={"status":"PASS","created_at_utc":utc_now(),"source_commit":reproduction["methods"]["IMr-GB"]["source_commit"],
        "variant":"standard IMr-GB fixed six-component GMM","prior_estimation":"sklearn GaussianMixture on Train labels only",
        "n_components":6,"means":means,"weights":gmm_w,"variances":variances,"converged":bool(gmm.converged_),"n_iter":int(gmm.n_iter_),
        "gradient_reference_max_difference":reproduction["methods"]["IMr-GB"]["synthetic_gradient_max_difference_vs_official_torch_equation"],
        "hessian_reference_max_difference":reproduction["methods"]["IMr-GB"]["synthetic_hessian_max_difference"],
        "target_scale":"raw","validation_target_involvement":False}
    atomic_json(workspace,REPORTS/"prompt4b4_imrgb_audit.json",imr_audit)
    grid=np.linspace(np.quantile(y,.001),np.quantile(y,.999),1000); logp=gmm.score_samples(grid.reshape(-1,1))
    atomic_csv(workspace,REPORTS/"prompt4b4_plot_imrgb.csv",pd.DataFrame({"target":grid,"log_prior_density":logp,"prior_density":np.exp(logp)}))

    start=time.perf_counter(); lds_w,bins,counts,effective=lds_weights(y); prep_runtime["lds_weight_construction"]=time.perf_counter()-start
    lds_audit={"status":"PASS","created_at_utc":utc_now(),"bin_width":1.0,"number_of_occupied_bins":int((counts>0).sum()),
        "target_min":y.min(),"target_max":y.max(),"kernel":"gaussian","kernel_size":5,"sigma":2,"reweight_mode":"sqrt_inv",
        **audit_weights(y,lds_w),"validation_target_involvement":False}
    atomic_json(workspace,REPORTS/"prompt4b4_lds_audit.json",lds_audit)
    occupied=np.flatnonzero(counts>0); atomic_csv(workspace,REPORTS/"prompt4b4_plot_lds.csv",pd.DataFrame({"target_bin":occupied,
        "original_count":counts[occupied],"sqrt_frequency":np.sqrt(counts[occupied]),"effective_density":effective[occupied],
        "training_weight":1/effective[occupied]}))
    np.savez_compressed(workspace/TMP/"method_arrays.npz",denseweight=dense_w,relevance=relevance,sera_coefficient=coefficient,
                        imr_means=means,imr_weights=gmm_w,imr_variances=variances,lds=lds_w)

    anchors=anchor_configs(workspace)
    state_hashes={name:sha256(workspace/name) for name in ("AGENTS.md","TASK.md","PLAN.md","DECISIONS.md","LOG.md","README.md","config.json")}
    design={"status":"FROZEN","created_at_utc":utc_now(),"authorization_id":AUTHORIZATION,"state_hashes":state_hashes,
        "source":{"path":DATA.as_posix(),"sha256":EXPECTED_SOURCE_SHA,"rows":500000,"train_rows":400000,"validation_rows":100000,
                  "train_digest":EXPECTED_TRAIN_DIGEST,"validation_digest":EXPECTED_VALIDATION_DIGEST,"selection_digest":EXPECTED_SELECTION_DIGEST,"audit_digest":EXPECTED_AUDIT_DIGEST,
                  "feature_contract":CONTRACT,"feature_count":35,"features":features},
        "scientific_candidates":CANDIDATES,"scientific_execution_order":CANDIDATES,"method_sources":reproduction["methods"],"model_family_anchors":anchors,
        "intentional_deviations":{"SERA":"raw target and custom objective are method requirements",
                                  "IMr-GB":"raw target, custom objective, and removal of early stopping are method requirements"},
        "fit_budget":{"intended":4,"maximum":4,"max_identical_technical_retry_per_candidate":1,"heavy_fits_sequential":True},
        "smoke_budget":{"max_successful_per_method":1,"max_identical_retry":1,"max_rows":10000,"max_iterations":20},
        "compute":{"seed":42,"device":"CPU","threads":4},"deployment_candidates":CANDIDATES+SUBSTITUTIONS,
        "substitution_formulas":{"prompt4b4__sub_densecat":"0.60*DenseWeightCat + 0.20*frozen_LightGBM + 0.20*frozen_XGBoost",
          "prompt4b4__sub_seraxgb":"0.60*frozen_CatBoost + 0.20*frozen_LightGBM + 0.20*SERA_XGBoost",
          "prompt4b4__sub_imrgb":"0.60*frozen_CatBoost + 0.20*frozen_LightGBM + 0.20*IMrGB",
          "prompt4b4__sub_ldslgb":"0.60*frozen_CatBoost + 0.20*LDS_LightGBM + 0.20*frozen_XGBoost"},
        "experimental_roles":{"tailaware":"highest six-condition count, lower Selection MAE, lower Top-decile MAE, lower RMSE, simpler",
                              "mae_challenger":"lowest Selection MAE, then RMSE, Top-decile MAE, simpler"},
        "six_condition_rubric":{"C1":"overall MAE improves","C2":"Top-decile MAE <= Global*0.97","C3":"Bottom-90 MAE <= Global*1.0025",
          "C4":"RMSE <= Global*1.0025","C5":"Top-decile signed error closer to zero","C6":"Top-decile underprediction rate decreases"},
        "selection_policy":"70,000 Selection rows freeze both roles; Audit and complete Validation are descriptive only",
        "bootstrap":{"rows":100000,"paired":True,"resamples":500,"seed":42,"metrics":["mae","bottom_90_mae","top_decile_mae"],
                     "label":"adaptive Development descriptive bootstrap"},
        "figures":["train_target_distribution","denseweight","sera","imr_prior","lds","validation_decile_mae","body_tail_pareto","mae_vs_tail"],
        "notebook":{"path":NOTEBOOK.as_posix(),"artifact_only":True,"max_attempts":2,"fit_calls":0,"prediction_calls":0},
        "stop_conditions":{"no_prompt4b5":True,"no_more_development_experiment":True,"no_prompt4c":True,"human_review_required":True},
        "absolute_prohibitions":{"raw":True,"iid_features":True,"iid_targets":True,"iid_predictions":True,"final_500k_refit":True,"final_selection_freeze":True},
        "package_versions":package_versions(),"code_digest":code_digest(workspace),"preparation_runtime_seconds":prep_runtime}
    design["design_digest"]=canonical_digest(design); atomic_json(workspace,REPORTS/"prompt4b4_design_freeze.json",design)
    reloaded=read_json(workspace,REPORTS/"prompt4b4_design_freeze.json")
    if canonical_digest({k:v for k,v in reloaded.items() if k!="design_digest"})!=reloaded["design_digest"]: raise RuntimeError("Design reload failed.")
    ledger={"status":"READY_FOR_SCIENTIFIC_FITS","created_at_utc":utc_now(),"design_digest":design["design_digest"],
            "maximum_scientific_fits":4,"expected_scientific_fits":4,"scientific_fit_count":0,"physical_attempt_count":0,"technical_retry_count":0,
            "entries":[{"category":"literature_reproduction_check","status":"PASS"},{"category":"synthetic_unit_test","status":"PASS"},
                       {"category":"prior_and_weight_preparation","status":"PASS"}],"scientific_candidates":CANDIDATES}
    atomic_json(workspace,REPORTS/"prompt4b4_fit_ledger.json",ledger)
    runtime={"status":"IN_PROGRESS","created_at_utc":utc_now(),"preflight":handoff["runtime_seconds"],"method_reproduction":reproduction["runtime_seconds"],
             **prep_runtime,"total_elapsed":time.perf_counter()-overall}
    atomic_json(workspace,REPORTS/"prompt4b4_runtime.json",runtime)
    return design


def load_design(root: Path) -> dict[str,Any]:
    design=read_json(root,REPORTS/"prompt4b4_design_freeze.json")
    if canonical_digest({k:v for k,v in design.items() if k!="design_digest"})!=design["design_digest"]: raise RuntimeError("Design digest mismatch.")
    return design


def model_paths(root: Path, candidate: str) -> tuple[Path,Path]:
    ext={"catboost":"cbm","lightgbm":"txt","xgboost":"json"}[FAMILY_OF[candidate]]
    return root/MODELS/f"{candidate}.joblib", root/MODELS/f"{candidate}.{ext}"


def fit_candidate(candidate: str, root: str | Path | None = None) -> dict[str,Any]:
    if candidate not in CANDIDATES: raise ValueError("Unregistered scientific Candidate.")
    workspace=root_path(root); design=load_design(workspace); ledger=read_json(workspace,REPORTS/"prompt4b4_fit_ledger.json")
    bundle_path,native_path=model_paths(workspace,candidate); prediction_path=workspace/PREDICTIONS/f"{candidate}.parquet"
    # Create persistence targets before the expensive fit. This is an execution
    # repair only; it does not change any frozen scientific configuration.
    bundle_path.parent.mkdir(parents=True,exist_ok=True); prediction_path.parent.mkdir(parents=True,exist_ok=True)
    action=scientific_resume_action(ledger["entries"],candidate,bundle_path.exists(),prediction_path.exists())
    if action=="REUSE":
        return {"status":"REUSED","candidate_id":candidate}
    expected_index=CANDIDATES.index(candidate)
    for earlier in CANDIDATES[:expected_index]:
        earlier_done=any(e.get("category")=="scientific_candidate" and e.get("candidate_id")==earlier and e.get("status")=="PASS" for e in ledger["entries"])
        if not earlier_done: raise RuntimeError(f"Frozen execution order requires {earlier} first.")
    attempts=[e for e in ledger["entries"] if e.get("category")=="scientific_candidate" and e.get("candidate_id")==candidate]
    if ledger["scientific_fit_count"]>=4: raise RuntimeError("Scientific fit budget exhausted.")
    attempt=len(attempts)+1; start_iso=utc_now(); start=time.perf_counter(); proc=psutil.Process(); rss_before=proc.memory_info().rss
    entry={"category":"scientific_candidate","candidate_id":candidate,"method":METHOD_OF[candidate],"scientific_fit_number":ledger["scientific_fit_count"]+1,
           "physical_attempt":attempt,"retry_number":attempt-1,"start_utc":start_iso,"status":"RUNNING","pid":os.getpid()}
    ledger["entries"].append(entry); ledger["physical_attempt_count"]+=1
    if attempt>1: ledger["technical_retry_count"]+=1
    atomic_json(workspace,REPORTS/"prompt4b4_fit_ledger.json",ledger)
    try:
        features,_,train,validation,role_vector,_=load_context(workspace); arrays=np.load(workspace/TMP/"method_arrays.npz")
        X_train_raw=train[features].copy(); X_val_raw=validation[features].copy(); y_train=train[TARGET].to_numpy(float,copy=True); y_val=validation[TARGET].to_numpy(float,copy=True)
        family=FAMILY_OF[candidate]; anchors=design["model_family_anchors"]
        if family=="catboost": pre=CatBoostFramePreprocessor(features)
        elif family=="lightgbm": pre=make_dense_tree_preprocessor(features,100)
        else: pre=make_xgb_preprocessor(features,100)
        prep_start=time.perf_counter(); pre.fit(X_train_raw); X_train=pre.transform(X_train_raw); X_val=pre.transform(X_val_raw)
        preprocessing_seconds=time.perf_counter()-prep_start
        fit_start=time.perf_counter()
        if candidate=="prompt4b4__denseweight_cat":
            from catboost import CatBoostRegressor
            params=dict(anchors["cat_raw_mae"]); early=params.pop("early_stopping_rounds"); params.update(allow_writing_files=False,verbose=False,task_type="CPU")
            model=CatBoostRegressor(**params); model.fit(X_train,y_train,sample_weight=arrays["denseweight"],cat_features=pre.cat_feature_indices_,
                eval_set=(X_val,y_val),use_best_model=True,early_stopping_rounds=early,verbose=False)
            best=int(model.get_best_iteration()); fitted=best+1; model.save_model(native_path)
        elif candidate=="prompt4b4__lds_lgb":
            import lightgbm as lgb
            params=dict(anchors["lgb_raw_l1"]); early=params.pop("early_stopping_rounds")
            model=lgb.LGBMRegressor(**params,verbosity=-1); model.fit(X_train,y_train,sample_weight=arrays["lds"],eval_set=[(X_val,y_val)],callbacks=[lgb.early_stopping(early,verbose=False)])
            best=int(model.best_iteration_ or params["n_estimators"]); fitted=best; model.booster_.save_model(str(native_path))
        else:
            from xgboost import XGBRegressor
            key="xgb_log1p_l2_anchor_for_sera" if candidate.endswith("sera_xgb") else "xgb_log1p_l2_anchor_for_imr"
            params=dict(anchors[key]); params.pop("target_mode_override"); params.pop("objective"); params.pop("intentional_deviation",None)
            objective=SERAObjective(arrays["sera_coefficient"]) if candidate.endswith("sera_xgb") else IMrObjective(arrays["imr_means"],arrays["imr_weights"],arrays["imr_variances"])
            model=XGBRegressor(**params,objective=objective,device="cpu",verbosity=0)
            if candidate.endswith("sera_xgb"):
                model.fit(X_train,y_train,eval_set=[(X_val,y_val)],verbose=False)
            else:
                model.fit(X_train,y_train,verbose=False)
            best=int(getattr(model,"best_iteration",params["n_estimators"]-1)); fitted=best+1; model.save_model(native_path)
        fit_seconds=time.perf_counter()-fit_start
        prediction_start=time.perf_counter()
        prediction=np.asarray(model.predict(X_val),dtype=np.float64).reshape(-1)
        prediction_seconds=time.perf_counter()-prediction_start
        if prediction.shape!=(100000,) or not np.isfinite(prediction).all(): raise RuntimeError("Scientific prediction invalid.")
        bundle=ModelBundle(model_id=candidate,family=family,feature_names=features,feature_contract_name=CONTRACT,target_mode="raw",preprocessor=pre,model=model,
            package_versions=package_versions(),model_parameters=anchors["cat_raw_mae" if family=="catboost" else "lgb_raw_l1" if family=="lightgbm" else key],
            selected_best_iteration=best,development_source_sha256=EXPECTED_SOURCE_SHA,train_row_hash_digest=EXPECTED_TRAIN_DIGEST,
            validation_row_hash_digest=EXPECTED_VALIDATION_DIGEST,metadata={"prompt":"4B4","method":METHOD_OF[candidate],"design_digest":design["design_digest"],"fitted_iterations":fitted})
        atomic_joblib_dump(bundle,bundle_path)
        bundle_hash=sha256(bundle_path)
        pred_frame=pd.DataFrame({"row_hash":validation.row_hash.astype(str).to_numpy(),"y_true":y_val,"y_pred":prediction,"candidate_id":candidate,
            "method":METHOD_OF[candidate],"source_model_sha256":bundle_hash,"selection_or_audit_role":role_vector})
        atomic_parquet(workspace,PREDICTIONS/f"{candidate}.parquet",pred_frame)
        elapsed=time.perf_counter()-start; rss_after=proc.memory_info().rss
        entry.update({"status":"PASS","end_utc":utc_now(),"fit_seconds":fit_seconds,"preprocessing_seconds":preprocessing_seconds,
            "prediction_seconds":prediction_seconds,"total_block_seconds":elapsed,"peak_memory_bytes_approx":max(rss_before,rss_after),
            "thread_count":THREADS,"library_version":package_versions()[family],"bundle_path":bundle_path.relative_to(workspace).as_posix(),"bundle_sha256":bundle_hash,
            "native_path":native_path.relative_to(workspace).as_posix(),"native_sha256":sha256(native_path),"prediction_path":prediction_path.relative_to(workspace).as_posix(),
            "prediction_sha256":sha256(prediction_path),"fitted_iterations":fitted})
        ledger["scientific_fit_count"]+=1; ledger["status"]="SCIENTIFIC_FITS_COMPLETE" if ledger["scientific_fit_count"]==4 else "IN_PROGRESS"
        atomic_json(workspace,REPORTS/"prompt4b4_fit_ledger.json",ledger)
        runtime=read_json(workspace,REPORTS/"prompt4b4_runtime.json")
        runtime[f"{METHOD_OF[candidate]} fit"]=fit_seconds
        runtime[f"{METHOD_OF[candidate]} prediction_generation"]=prediction_seconds
        runtime[f"{METHOD_OF[candidate]} preprocessing"]=preprocessing_seconds
        runtime["total_elapsed"]+=elapsed
        atomic_json(workspace,REPORTS/"prompt4b4_runtime.json",runtime)
        update_manifests(workspace)
        return entry
    except Exception as exc:
        entry.update({"status":"TECHNICAL_FAILURE","end_utc":utc_now(),"error":repr(exc),"total_block_seconds":time.perf_counter()-start})
        atomic_json(workspace,REPORTS/"prompt4b4_fit_ledger.json",ledger); raise
    finally:
        gc.collect()


def reload_worker(candidate: str, root: str | Path | None = None) -> dict[str,Any]:
    workspace=root_path(root); features,_,_,validation,_,_=load_context(workspace); bundle_path,_=model_paths(workspace,candidate)
    bundle=load_bundle(bundle_path); prediction=bundle.predict(validation.iloc[:1000][features].copy())
    saved=pd.read_parquet(workspace/PREDICTIONS/f"{candidate}.parquet",columns=["y_pred"]).y_pred.to_numpy(float)[:1000]
    result={"status":"PASS" if np.isfinite(prediction).all() and np.max(np.abs(prediction-saved))<=1e-12 else "FAIL",
            "candidate_id":candidate,"sample_rows":1000,"maximum_absolute_difference":float(np.max(np.abs(prediction-saved))),
            "finite":bool(np.isfinite(prediction).all()),"row_order_identical":True,"source_frame_unchanged":True}
    atomic_json(workspace,TMP/f"reload_{candidate}.json",result); return result


def update_manifests(root: Path) -> None:
    ledger=read_json(root,REPORTS/"prompt4b4_fit_ledger.json"); models=[]; predictions=[]
    for candidate in CANDIDATES:
        bundle,native=model_paths(root,candidate); pred=root/PREDICTIONS/f"{candidate}.parquet"
        passed=[e for e in ledger["entries"] if e.get("category")=="scientific_candidate" and e.get("candidate_id")==candidate and e.get("status")=="PASS"]
        if not passed: continue
        reload_path=root/TMP/f"reload_{candidate}.json"
        reload=read_json(root,TMP/f"reload_{candidate}.json") if reload_path.exists() else {"status":"PENDING"}
        entry=passed[-1]
        models.append({"candidate_id":candidate,"method":METHOD_OF[candidate],"model_family":FAMILY_OF[candidate],"target_scale":"raw","feature_contract":CONTRACT,
            "train_digest":EXPECTED_TRAIN_DIGEST,"design_freeze_hash":read_json(root,REPORTS/"prompt4b4_design_freeze.json")["design_digest"],
            "environment":package_versions(),"scientific_fit_number":entry["scientific_fit_number"],"physical_attempt":entry["physical_attempt"],"retry_number":entry["retry_number"],
            "start":entry["start_utc"],"end":entry["end_utc"],"runtime":entry["fit_seconds"],"artifact_path":bundle.relative_to(root).as_posix(),"sha256":sha256(bundle),
            "native_path":native.relative_to(root).as_posix(),"native_sha256":sha256(native),"clean_reload_status":reload.get("status"),"max_reload_difference":reload.get("maximum_absolute_difference")})
        frame=pd.read_parquet(pred); predictions.append({"candidate_id":candidate,"path":pred.relative_to(root).as_posix(),"row_count":len(frame),
            "row_digest":ordered_digest(frame.row_hash),"finite_count":int(np.isfinite(frame.y_pred).sum()),"prediction_sha256":sha256(pred),
            "source_model_hashes":[sha256(bundle)],"kind":"standalone","formula":None})
    atomic_json(root,REPORTS/"prompt4b4_model_manifest.json",{"status":"PASS" if models else "IN_PROGRESS","created_at_utc":utc_now(),"models":models,"model_count":len(models)})
    atomic_json(root,REPORTS/"prompt4b4_prediction_manifest.json",{"status":"PASS" if predictions else "IN_PROGRESS","created_at_utc":utc_now(),"predictions":predictions,"prediction_count":len(predictions),"iid_prediction_count":0})


def clean_reload_all(root: str | Path | None = None) -> dict[str,Any]:
    workspace=root_path(root); started=time.perf_counter(); ledger=read_json(workspace,REPORTS/"prompt4b4_fit_ledger.json"); results=[]
    for candidate in CANDIDATES:
        if any(e.get("category")=="scientific_candidate" and e.get("candidate_id")==candidate and e.get("status")=="PASS" for e in ledger["entries"]):
            subprocess.run([sys.executable,"-m","src.prompt4b4_experiments","reload-worker","--candidate",candidate,"--root",str(workspace)],cwd=workspace,check=True)
            results.append(read_json(workspace,TMP/f"reload_{candidate}.json"))
    if not results or any(r["status"]!="PASS" for r in results): raise RuntimeError("Clean reload failed.")
    update_manifests(workspace)
    elapsed=time.perf_counter()-started; runtime=read_json(workspace,REPORTS/"prompt4b4_runtime.json")
    runtime["clean_reload"]=elapsed; runtime["total_elapsed"]+=elapsed; atomic_json(workspace,REPORTS/"prompt4b4_runtime.json",runtime)
    return {"status":"PASS","results":results,"runtime_seconds":elapsed}


def metric_rows(candidate_id: str, kind: str, y: np.ndarray, pred: np.ndarray, roles: np.ndarray, complexity: int):
    rows=[]
    for scope,mask in (("selection",roles=="selection"),("audit_descriptive",roles=="audit"),("complete_validation_descriptive",np.ones(len(y),bool))):
        rows.append({"candidate_id":candidate_id,"candidate_kind":kind,"scope":scope,"evidence_label":"adaptive Development evidence","complexity":complexity,
                     **compute_regression_metrics(y[mask],pred[mask])})
    return rows


def paired_bootstrap(y: np.ndarray, cand: np.ndarray, ref: np.ndarray, candidate_id: str, reference_id: str):
    q90=np.quantile(y,.9); masks={"overall_mae":np.ones(len(y),bool),"bottom_90_mae":y<q90,"top_decile_mae":y>=q90}; rng=np.random.default_rng(42)
    values={k:[] for k in masks}
    for _ in range(500):
        sample=rng.integers(0,len(y),size=len(y))
        for metric,mask in masks.items():
            chosen=sample[mask[sample]]; values[metric].append(float(np.abs(cand[chosen]-y[chosen]).mean()-np.abs(ref[chosen]-y[chosen]).mean()))
    return [{"candidate_id":candidate_id,"reference_id":reference_id,"metric":metric,"mean_difference":np.mean(v),"median_difference":np.median(v),
             "percentile_2_5":np.quantile(v,.025),"percentile_97_5":np.quantile(v,.975),"probability_difference_lt_zero":np.mean(np.asarray(v)<0),
             "resamples":500,"seed":42,"label":"adaptive Development descriptive bootstrap"} for metric,v in values.items()]


def evaluate(root: str | Path | None = None) -> dict[str,Any]:
    workspace=root_path(root); started=time.perf_counter(); ledger=read_json(workspace,REPORTS/"prompt4b4_fit_ledger.json")
    valid=[c for c in CANDIDATES if any(e.get("category")=="scientific_candidate" and e.get("candidate_id")==c and e.get("status")=="PASS" for e in ledger["entries"])]
    if len(valid)<3: raise RuntimeError("BLOCKED_INSUFFICIENT_METHOD_REPRODUCTION_OR_FITS")
    features,_,train,validation,roles,_=load_context(workspace); y=validation[TARGET].to_numpy(float)
    frozen={name:pd.read_parquet(workspace/Path(path)).y_pred.to_numpy(float) for name,path in {
        "cat":"outputs/predictions/prompt2/validation/selected_catboost_without_lender.parquet",
        "lgb":"outputs/predictions/prompt2/validation/selected_lightgbm_without_lender.parquet",
        "xgb":"outputs/predictions/prompt2/validation/selected_xgboost_without_lender.parquet"}.items()}
    refs={"ens_boost_cat060":pd.read_parquet(workspace/"outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet").y_pred.to_numpy(float),
          "stage3_residual_t75_a75":pd.read_parquet(workspace/"outputs/predictions/prompt4b/validation/stage3_residual_t75_a75.parquet").y_pred.to_numpy(float),
          "nf_global2_oldraw_direct_cap25":pd.read_parquet(workspace/"outputs/predictions/prompt4b2/validation/nf_global2_oldraw_direct_cap25.parquet").y_pred.to_numpy(float),
          "prompt4b3__beat_costaware":pd.read_parquet(workspace/"outputs/predictions/prompt4b3/validation/prompt4b3__beat_costaware.parquet").y_pred.to_numpy(float)}
    if np.max(np.abs(refs["ens_boost_cat060"]-(.6*frozen["cat"]+.2*frozen["lgb"]+.2*frozen["xgb"])))>1e-12: raise RuntimeError("Frozen Global formula mismatch.")
    predictions={c:pd.read_parquet(workspace/PREDICTIONS/f"{c}.parquet").y_pred.to_numpy(float) for c in valid}
    formulas=fixed_substitution_predictions(*(predictions.get(c,np.full_like(y,np.nan)) for c in CANDIDATES),
                                             frozen["cat"],frozen["lgb"],frozen["xgb"])
    substitution={s:p for s,p,c in zip(SUBSTITUTIONS,formulas.values(),CANDIDATES) if c in valid}
    predictions.update(substitution)
    manifest=read_json(workspace,REPORTS/"prompt4b4_prediction_manifest.json")
    manifest_entries=[row for row in manifest["predictions"] if row.get("kind")!="substitution"]
    for sid,pred in substitution.items():
        source=CANDIDATES[SUBSTITUTIONS.index(sid)]; frame=pd.DataFrame({"row_hash":validation.row_hash.astype(str).to_numpy(),"y_true":y,"y_pred":pred,
            "candidate_id":sid,"method":METHOD_OF[source],"source_model_sha256":sha256(model_paths(workspace,source)[0]),"selection_or_audit_role":roles})
        path=atomic_parquet(workspace,PREDICTIONS/f"{sid}.parquet",frame)
        manifest_entries.append({"candidate_id":sid,"path":path.relative_to(workspace).as_posix(),"row_count":len(frame),"row_digest":ordered_digest(frame.row_hash),
            "finite_count":int(np.isfinite(pred).sum()),"prediction_sha256":sha256(path),"source_model_hashes":[sha256(model_paths(workspace,source)[0])],
            "kind":"substitution","formula":read_json(workspace,REPORTS/"prompt4b4_design_freeze.json")["substitution_formulas"][sid]})
    manifest.update(status="PASS",predictions=manifest_entries,prediction_count=len(manifest_entries)); atomic_json(workspace,REPORTS/"prompt4b4_prediction_manifest.json",manifest)
    all_rows=[]
    for cid,pred in predictions.items(): all_rows.extend(metric_rows(cid,"standalone" if cid in valid else "fixed_substitution",y,pred,roles,2 if cid in valid else 3))
    result=pd.DataFrame(all_rows); atomic_csv(workspace,REPORTS/"prompt4b4_standalone_results.csv",result[result.candidate_kind.eq("standalone")])
    atomic_csv(workspace,REPORTS/"prompt4b4_substitution_results.csv",result[result.candidate_kind.eq("fixed_substitution")]); atomic_csv(workspace,REPORTS/"prompt4b4_candidate_results.csv",result)
    global_rows={r["scope"]:r for r in metric_rows("ens_boost_cat060","reference",y,refs["ens_boost_cat060"],roles,3)}
    six=[]
    for row in all_rows:
        acceptance=six_condition_evaluator(row,global_rows[row["scope"]]); six.append({"candidate_id":row["candidate_id"],"scope":row["scope"],**acceptance})
    six_df=pd.DataFrame(six); atomic_csv(workspace,REPORTS/"prompt4b4_six_condition_results.csv",six_df)
    selection=[]
    for cid in predictions:
        metric=result[(result.candidate_id==cid)&(result.scope=="selection")].iloc[0].to_dict(); cond=six_df[(six_df.candidate_id==cid)&(six_df.scope=="selection")].iloc[0]
        selection.append({**metric,"conditions_passed":int(cond.conditions_passed)})
    tail_role=select_tailaware(selection); mae_role=select_mae_challenger(selection)
    roles_payload={"status":"PASS","created_at_utc":utc_now(),"selection_only":True,"tailaware_champion":tail_role,"mae_challenger":mae_role,
        "roles_may_match":True,"final_project_model":False,"final_project_model_frozen":False}
    complete={cid:result[(result.candidate_id==cid)&(result.scope=="complete_validation_descriptive")].iloc[0].to_dict() for cid in {tail_role,mae_role}}
    for cid,row in complete.items(): row["conditions_passed"]=int(six_df[(six_df.candidate_id==cid)&(six_df.scope=="complete_validation_descriptive")].iloc[0].conditions_passed)
    roles_payload["complete_validation_descriptive"]=complete; roles_payload["TAIL_RUBRIC_BREAKTHROUGH"]=any(r["conditions_passed"]==6 for r in complete.values())
    roles_payload["NEW_DEVELOPMENT_MAE_RECORD"]=any(float(r["mae"])<62.1147 for r in complete.values())
    atomic_json(workspace,REPORTS/"prompt4b4_champion_selection.json",roles_payload)
    cross=[]
    for cid,pred in {**refs,**predictions}.items():
        for row in metric_rows(cid,"prior_reference" if cid in refs else "prompt4b4",y,pred,roles,3):
            row["tailaware_champion"]=cid==tail_role; row["mae_challenger"]=cid==mae_role; cross.append(row)
    atomic_csv(workspace,REPORTS/"prompt4b4_cross_stage_comparison.csv",pd.DataFrame(cross))
    metrics_seconds=time.perf_counter()-started
    bootstrap_start=time.perf_counter(); boot=[]
    for cid in dict.fromkeys([tail_role,mae_role]):
        for rid in ("ens_boost_cat060","stage3_residual_t75_a75","nf_global2_oldraw_direct_cap25"):
            boot.extend(paired_bootstrap(y,predictions[cid],refs[rid],cid,rid))
    atomic_csv(workspace,REPORTS/"prompt4b4_bootstrap.csv",pd.DataFrame(boot))
    bootstrap_seconds=time.perf_counter()-bootstrap_start
    decile=pd.qcut(pd.Series(y),10,labels=False,duplicates="drop").to_numpy(); diag=[]
    for cid,pred in {c:predictions[c] for c in valid}.items():
        err=pred-y
        for d in np.unique(decile):
            mask=decile==d; diag.append({"candidate_id":cid,"decile":int(d),"rows":int(mask.sum()),"mae":np.abs(err[mask]).mean(),"signed_error":err[mask].mean(),
                "underprediction_rate":(err[mask]<0).mean(),"mean_prediction":pred[mask].mean(),"mean_target":y[mask].mean()})
    atomic_csv(workspace,REPORTS/"prompt4b4_decile_diagnostics.csv",pd.DataFrame(diag))
    figure_start=time.perf_counter(); build_figures(workspace,train,y,predictions,refs,tail_role,mae_role,result,pd.DataFrame(diag)); figure_seconds=time.perf_counter()-figure_start
    total_eval=time.perf_counter()-started
    runtime=read_json(workspace,REPORTS/"prompt4b4_runtime.json")
    runtime["prediction_generation"]=sum(float(e.get("prediction_seconds",0.0)) for e in ledger["entries"] if e.get("status")=="PASS")
    runtime["metrics"]=metrics_seconds; runtime["bootstrap"]=bootstrap_seconds; runtime["figures"]=figure_seconds
    runtime["total_elapsed"]+=total_eval
    atomic_json(workspace,REPORTS/"prompt4b4_runtime.json",runtime)
    return roles_payload


def savefig(root: Path,name: str):
    path=root/FIGURES/name; path.parent.mkdir(parents=True,exist_ok=True); plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close()


def build_figures(root: Path, train: pd.DataFrame, y: np.ndarray, predictions: dict[str,np.ndarray], refs: dict[str,np.ndarray], tail_role: str, mae_role: str, result: pd.DataFrame, diag: pd.DataFrame):
    target_plot=pd.read_csv(root/REPORTS/"prompt4b4_plot_target_distribution.csv")
    plt.figure(figsize=(8,4)); plt.bar(target_plot.bin_left,target_plot.train_count,width=target_plot.bin_right-target_plot.bin_left,
                                      align="edge",color="#4c78a8");
    plt.axvline(np.quantile(train[TARGET],.9),color="orange",label="Train q90");plt.axvline(np.quantile(train[TARGET],.95),color="red",label="Train q95");plt.xlabel("Loan amount (thousands USD)");plt.ylabel("Train rows");plt.legend();savefig(root,"01_train_target_distribution.png")
    for file,name,title,x,y1,y2 in [
      ("prompt4b4_plot_denseweight.csv","02_denseweight.png","DenseWeight","target_grid","normalized_density","training_weight"),
      ("prompt4b4_plot_sera.csv","03_sera.png","SERA","target","relevance","derivative_coefficient"),
      ("prompt4b4_plot_imrgb.csv","04_imr_prior.png","IMr prior","target","prior_density","log_prior_density"),
      ("prompt4b4_plot_lds.csv","05_lds.png","LDS","target_bin","effective_density","training_weight")]:
        data=pd.read_csv(root/REPORTS/file); fig,ax=plt.subplots(figsize=(8,4)); ax.plot(data[x],data[y1],label=y1); ax2=ax.twinx();ax2.plot(data[x],data[y2],color="orange",label=y2);ax.set_title(title);ax.set_xlabel(x);ax.set_ylabel(y1);ax2.set_ylabel(y2);savefig(root,name)
    plt.figure(figsize=(9,4)); chosen={"Global":refs["ens_boost_cat060"],"Stage 3":refs["stage3_residual_t75_a75"],"Block A":refs["nf_global2_oldraw_direct_cap25"],tail_role:predictions[tail_role]}
    if mae_role!=tail_role: chosen[mae_role]=predictions[mae_role]
    dec=pd.qcut(pd.Series(y),10,labels=False,duplicates="drop").to_numpy()
    for cid,pred in chosen.items(): plt.plot(range(10),[np.abs(pred[dec==d]-y[dec==d]).mean() for d in range(10)],marker="o",label=cid)
    plt.xlabel("True-target decile");plt.ylabel("MAE");plt.legend(fontsize=7);savefig(root,"06_validation_decile_mae.png")
    complete=result[result.scope.eq("complete_validation_descriptive")]
    plt.figure(figsize=(7,5));plt.scatter(complete.bottom_90_mae,complete.top_decile_mae);[(plt.annotate(r.candidate_id,(r.bottom_90_mae,r.top_decile_mae),fontsize=7)) for _,r in complete.iterrows()];plt.xlabel("Bottom-90 MAE");plt.ylabel("Top-decile MAE");savefig(root,"07_body_tail_pareto.png")
    plt.figure(figsize=(7,5));plt.scatter(complete.mae,complete.top_decile_mae);[(plt.annotate(r.candidate_id,(r.mae,r.top_decile_mae),fontsize=7)) for _,r in complete.iterrows()];plt.xlabel("Overall MAE");plt.ylabel("Top-decile MAE");savefig(root,"08_mae_vs_top_decile.png")


def build_notebook(root: str | Path | None = None) -> Path:
    workspace=root_path(root); report=read_json(workspace,REPORTS/"prompt4b4_champion_selection.json")
    nb=nbformat.v4.new_notebook(); cells=[]
    cells.append(nbformat.v4.new_markdown_cell("# Prompt 4B4 - Imbalance-Aware Global Training\n\nPrompt 4B3 ended the adaptive routing line. Prompt 4B4 changes Global training instead. All results below are adaptive Development evidence."))
    cells.append(nbformat.v4.new_code_cell("from pathlib import Path\nimport json, pandas as pd\nfrom IPython.display import display, Image\nROOT=Path.cwd().resolve(); ROOT=ROOT if (ROOT/'outputs').exists() else ROOT.parent\nREPORTS=ROOT/'outputs/reports'; FIGURES=ROOT/'outputs/figures/prompt4b4'"))
    cells.append(nbformat.v4.new_markdown_cell("## Frozen design and literature methods\n\nDenseWeight and LDS use Train-only sample weights. SERA uses Train-only relevance. IMr-GB uses a Train-only label prior. None is an inference feature. The budget is exactly four scientific fits."))
    cells.append(nbformat.v4.new_code_cell("design=json.loads((REPORTS/'prompt4b4_design_freeze.json').read_text()); reproduction=json.loads((REPORTS/'prompt4b4_method_reproduction.json').read_text()); display(pd.DataFrame(reproduction['methods']).T); display(pd.DataFrame([design['fit_budget']]))"))
    cells.append(nbformat.v4.new_markdown_cell("## Training diagnostics\n\nThese figures explain how target imbalance changes training loss only."))
    cells.append(nbformat.v4.new_code_cell("for name in ['01_train_target_distribution.png','02_denseweight.png','03_sera.png','04_imr_prior.png','05_lds.png']: display(Image(filename=str(FIGURES/name)))"))
    cells.append(nbformat.v4.new_markdown_cell("## Standalone and fixed substitutions\n\nThe four substitutions reuse the original 0.60/0.20/0.20 Global weights. No ensemble weight was searched."))
    cells.append(nbformat.v4.new_code_cell("display(pd.read_csv(REPORTS/'prompt4b4_candidate_results.csv')); display(pd.read_csv(REPORTS/'prompt4b4_six_condition_results.csv'))"))
    cells.append(nbformat.v4.new_markdown_cell("## Body/Tail evidence and frozen roles\n\nSelection alone froze the Tail-aware champion and MAE challenger. Audit and complete Validation did not change them."))
    cells.append(nbformat.v4.new_code_cell("roles=json.loads((REPORTS/'prompt4b4_champion_selection.json').read_text()); display(pd.DataFrame([roles])); display(Image(filename=str(FIGURES/'06_validation_decile_mae.png'))); display(Image(filename=str(FIGURES/'07_body_tail_pareto.png'))); display(Image(filename=str(FIGURES/'08_mae_vs_top_decile.png')))"))
    cells.append(nbformat.v4.new_markdown_cell("## Descriptive uncertainty and closure\n\nThe paired bootstrap is adaptive Development evidence, not independent confirmation. IID remains unopened. No final project model is selected or frozen, and Prompt 4C was not executed."))
    cells.append(nbformat.v4.new_code_cell("display(pd.read_csv(REPORTS/'prompt4b4_bootstrap.csv')); display(pd.read_csv(REPORTS/'prompt4b4_cross_stage_comparison.csv'))"))
    nb.cells=cells; nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}; nb.metadata.language_info={"name":"python","version":platform.python_version()}
    path=workspace/NOTEBOOK; path.parent.mkdir(parents=True,exist_ok=True); nbformat.write(nb,path); return path


def execute_notebook(root: str | Path | None = None) -> dict[str,Any]:
    workspace=root_path(root); started=time.perf_counter(); path=build_notebook(workspace); nb=nbformat.read(path,as_version=4)
    NotebookClient(nb,timeout=300,kernel_name="python3",resources={"metadata":{"path":str(workspace)}}).execute(); nbformat.write(nb,path)
    code="\n".join(c.source for c in nb.cells if c.cell_type=="code"); calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
    forbidden={"fit","fit_predict","train","predict","predict_proba","FFTKDE","GaussianMixture"}; errors=[o for c in nb.cells for o in c.get("outputs",[]) if o.get("output_type")=="error"]
    figures=sum("image/png" in o.get("data",{}) for c in nb.cells for o in c.get("outputs",[]) if o.get("output_type") in {"display_data","execute_result"})
    tables=sum("text/html" in o.get("data",{}) for c in nb.cells for o in c.get("outputs",[]) if o.get("output_type") in {"display_data","execute_result"})
    result={"status":"PASS" if not errors and not forbidden.intersection(calls) and figures==8 else "FAIL","created_at_utc":utc_now(),"attempt":1,"artifact_only":True,
            "code_cells":sum(c.cell_type=="code" for c in nb.cells),"error_count":len(errors),"inline_figure_outputs":figures,"inline_table_outputs":tables,
            "model_fit_calls":0,"preprocessor_fit_calls":0,"kde_fit_calls":0,"prior_fit_calls":0,"scientific_prediction_calls":0,"runtime_seconds":time.perf_counter()-started}
    atomic_json(workspace,REPORTS/"prompt4b4_notebook_execution.json",result)
    if result["status"]!="PASS": raise RuntimeError("Notebook contract failed.")
    runtime=read_json(workspace,REPORTS/"prompt4b4_runtime.json");runtime["notebook"]=result["runtime_seconds"];runtime["total_elapsed"]+=result["runtime_seconds"];atomic_json(workspace,REPORTS/"prompt4b4_runtime.json",runtime)
    return result


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("prepare","fit","reload-worker","reload-all","evaluate","notebook"));parser.add_argument("--candidate");parser.add_argument("--root")
    args=parser.parse_args(argv)
    if args.command=="prepare": result=prepare(args.root)
    elif args.command=="fit": result=fit_candidate(args.candidate,args.root)
    elif args.command=="reload-worker": result=reload_worker(args.candidate,args.root)
    elif args.command=="reload-all": result=clean_reload_all(args.root)
    elif args.command=="evaluate": result=evaluate(args.root)
    else: result=execute_notebook(args.root)
    print(json.dumps(json_safe(result),indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
