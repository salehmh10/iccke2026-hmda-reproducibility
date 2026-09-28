"""Stage 8 frozen-model explainability utilities.

The protection phase never parses explanation values, source feature values,
or serialized model contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "artifacts/results/stage8/explainability"
FIGURES = ROOT / "artifacts/figures/stage8"
PLOTTING = FIGURES / "plotting_data"
MANIFESTS = ROOT / "artifacts/manifests/stage8"
REPORTS = ROOT / "artifacts/reports"
BACKUPS = ROOT / "artifacts/backups"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
LABEL = "Post-Test Explainability and Feature Interpretation"

CANDIDATES = [
    "stage4l__blend__without_sensitive",
    "stage5c__realmlp__without_sensitive__test_evaluation",
    "stage5c__realmlp__with_sensitive__test_evaluation",
]
MODELS = [
    {"id": "stage4l__catboost__without_sensitive", "family": "CatBoost", "mode": "without_sensitive", "target_mode": "log1p", "path": "artifacts/models/catboost/final/catboost_final_without_sensitive.joblib", "sha256": "57672c581ce91dfabb83a75ceaf074adaaab1b3ace7f94e394626e8f8a324ccf"},
    {"id": "stage4l__lightgbm__without_sensitive", "family": "LightGBM", "mode": "without_sensitive", "target_mode": "raw", "path": "artifacts/models/lightgbm/final/lightgbm_final_without_sensitive.joblib", "sha256": "b1216ea51eafcd7b53289b23a53a594b67b1f5f3015e9555972cbd6c118d3002"},
    {"id": "stage4l__xgboost__without_sensitive", "family": "XGBoost", "mode": "without_sensitive", "target_mode": "log1p", "path": "artifacts/models/xgboost/final/xgboost_final_without_sensitive.joblib", "sha256": "2a6bb80dacae81510592e9aea3477820cd3a7b66676d6e0047a0bb3c85ae4430"},
    {"id": "stage5c__realmlp__without_sensitive__test_evaluation", "family": "RealMLP", "mode": "without_sensitive", "target_mode": "raw", "path": "artifacts/models/deep/core_final/stage5a2__realmlp__full_train__without_sensitive__direct_no_refit_recovery2.joblib", "sha256": "0d2dc108578512022608fee31676ced4d3d65d178f3d77b2418011057eff7006"},
    {"id": "stage5c__realmlp__with_sensitive__test_evaluation", "family": "RealMLP", "mode": "with_sensitive", "target_mode": "raw", "path": "artifacts/models/deep/core_final/stage5a2__realmlp__full_train__with_sensitive__fixed_epoch30__technical_retry1.joblib", "sha256": "d8b74180385cae7c0cfb9570ef124e9061c4ee398a68a7a9266aca3de299f600"},
]
PREDICTIONS = [
    {"id": CANDIDATES[0], "path": "artifacts/predictions/final_test/stage4l__blend__without_sensitive.csv", "sha256": "9f9efa21d95a466b8271cd0db0a1e6b2c1ed2b5f1cabfbbb7e081137b9e4b7ed"},
    {"id": CANDIDATES[1], "path": "artifacts/predictions/stage5/posttest_evaluation/stage5c_test_predictions_without_sensitive.csv", "sha256": "972eaa799c00eaa0ed486739636fb643f8f3e46e6890dc1964da542fd6108ee5"},
    {"id": CANDIDATES[2], "path": "artifacts/predictions/stage5/posttest_evaluation/stage5c_test_predictions_with_sensitive.csv", "sha256": "b4b11779a2d85209b2082c003ce79db2b657acd52c816c5e5345aaa6671f5e99"},
]
EXPLAIN_METADATA = [
    "artifacts/manifests/stage4/catboost/catboost_shap_metadata_without_sensitive.json",
    "artifacts/manifests/stage4/lightgbm/stage4h_shap_metadata_without_sensitive.json",
    "artifacts/manifests/stage4/xgboost/stage4k_shap_metadata_without_sensitive.json",
    "artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv",
]
FIGURE_TITLES = [
    "Explainability artifact coverage and provenance",
    "Official Stage 4L blend grouped permutation Importance — Top 20",
    "RealMLP without-sensitive grouped permutation Importance — Top 20",
    "RealMLP with-sensitive grouped permutation Importance — Top 20",
    "Cross-model Feature-rank heatmap",
    "Feature-family importance-share comparison",
    "CatBoost native Importance versus SHAP rank agreement",
    "LightGBM native Importance versus SHAP rank agreement",
    "XGBoost native Importance versus SHAP rank agreement",
    "Saved Deep attribution versus Stage 8 Deep permutation agreement",
    "Explicit-sensitive and contextual-sensitive aggregate importance shares",
    "Local effects for one frozen common-large-error case",
    "Local effects for one frozen Stage 4L-versus-RealMLP disagreement case",
    "Local effects and background stability for the frozen sensitive-improvement case",
    "Stage 8 global and local explainability dashboard",
]
REGISTRY_IDS = [
    "stage8__stage4l_official__global_explanation",
    "stage8__realmlp_without_sensitive__global_explanation",
    "stage8__realmlp_with_sensitive__global_explanation",
    "stage8__tree_shap__synthesis",
    "stage8__cross_model__feature_comparison",
    "stage8__local_cases__explanations",
    "stage8__explainability_summary",
    "stage8__stage9_handoff",
]
EXPECTED = {"rows": 99948, "row_hash": "e58e4d078c761f60405e644d4dd7ba368f364daffb73b44abb39095938ece95e", "target_hash": "889e4253fb584c2a52a06d8b8e956beefad997ba18e4d736af0cd1738fb34a1a"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def value_hash(values: np.ndarray, dtype: object) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def record(rel: str) -> dict:
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(rel)
    return {"path": rel, "sha256": sha(path), "size_bytes": path.stat().st_size}


def validate_prerequisites() -> dict:
    checks = {}
    files = {
        "stage4l": "artifacts/reports/stage4l_verification.json",
        "stage5a": "artifacts/reports/stage5a_verification.json",
        "stage5b": "artifacts/reports/stage5b_verification.json",
        "stage5c": "artifacts/reports/stage5c_verification.json",
        "stage6": "artifacts/reports/stage6_verification.json",
        "stage7": "artifacts/reports/stage7_verification.json",
        "stage7_recheck": "artifacts/reports/stage7_protected_recheck.json",
        "stage7_handoff": "artifacts/manifests/stage7/stage7_stage8_handoff.json",
    }
    for key, rel in files.items():
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        checks[key] = {"path": rel, "status": data.get("status", data.get("stage7_status", "")), "sha256": sha(ROOT / rel)}
    reviewer = (ROOT / "artifacts/reports/stage7_reviewer.md").read_text(encoding="utf-8")
    checks["stage7_reviewer"] = {**record("artifacts/reports/stage7_reviewer.md"), "pass_text": "PASS" in reviewer}
    checks["stage7_notebook"] = record("REGRESSION_PART7_FAIRNESS_SENSITIVE_ANALYSIS.ipynb")
    checks["stage7_notebook"]["expected_sha256"] = "4eb4f349eb49a059710463de4c28e67e21832d12e5c7193ab5d8fae18cfe6e5d"
    if checks["stage7_notebook"]["sha256"] != checks["stage7_notebook"]["expected_sha256"]:
        raise RuntimeError("Stage 7 Notebook hash differs")
    if checks["stage7"]["status"] != "PASS" or checks["stage7_recheck"]["status"] != "PASS" or not checks["stage7_reviewer"]["pass_text"]:
        raise RuntimeError("Stage 7 prerequisite is not PASS")
    for item in MODELS + PREDICTIONS:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Frozen hash mismatch: {item['id']}")
    return checks


def select_samples() -> dict:
    pred = pd.read_csv(ROOT / PREDICTIONS[1]["path"], usecols=["row_id", "y_true"])
    if len(pred) != EXPECTED["rows"] or pred.row_id.nunique() != EXPECTED["rows"]:
        raise RuntimeError("Stage 5C row contract failed")
    if value_hash(pred.row_id, np.int64) != EXPECTED["row_hash"] or value_hash(pred.y_true, np.float64) != EXPECTED["target_hash"]:
        raise RuntimeError("Stage 5C row or target hash failed")
    pred["target_decile"] = pd.qcut(pred.y_true.rank(method="first"), 10, labels=False) + 1
    sample = pd.concat([g.sample(n=200, random_state=42) for _, g in pred.groupby("target_decile", sort=True)]).sort_values("row_id").reset_index(drop=True)
    sample[["row_id", "target_decile"]].to_csv(MANIFESTS / "stage8_global_explanation_sample_row_ids.csv", index=False)
    cases = pd.read_csv(ROOT / "artifacts/results/stage7/fairness/stage7_representative_cases_public.csv").sort_values(["case_type", "case_rank"]).drop_duplicates("row_id").head(20).copy()
    public = cases[["case_type", "case_rank", "row_id", "target_decile", "future_use", "analysis_label"]].copy()
    semantic = {"common_large_error": "common_large_error", "stage4l_beats_deep_without": "stage4l_beats_realmlp_without", "deep_without_beats_stage4l": "realmlp_without_beats_stage4l", "deep_with_improves_over_without": "sensitive_improvement"}
    public["semantic_case_type"] = public.case_type.map(semantic)
    public["visualization_case"] = False
    for case_type in semantic:
        idx = public.index[public.case_type == case_type]
        if len(idx):
            public.loc[idx[0], "visualization_case"] = True
    if len(public) != 20 or int(public.visualization_case.sum()) != 4:
        raise RuntimeError("Local-case contract failed")
    public.to_csv(MANIFESTS / "stage8_local_case_manifest.csv", index=False)
    background = pd.concat([g.sample(n=4, random_state=42) for _, g in sample[~sample.row_id.isin(set(public.row_id))].groupby("target_decile", sort=True)]).sort_values("row_id").reset_index(drop=True)
    background[["row_id", "target_decile"]].to_csv(MANIFESTS / "stage8_local_background_row_ids.csv", index=False)
    return {"global_rows": len(sample), "global_row_hash": value_hash(sample.row_id, np.int64), "global_target_hash": value_hash(sample.y_true, np.float64), "decile_counts": {str(k): int(v) for k, v in sample.target_decile.value_counts().sort_index().items()}, "local_cases": len(public), "visualization_cases": int(public.visualization_case.sum()), "background_rows": len(background), "seed": 42, "selected_at_utc": now(), "selection_before_feature_access": True}


def protected_files() -> list[Path]:
    prior = json.loads((ROOT / "artifacts/manifests/stage7/stage7_protected_hashes_before.json").read_text(encoding="utf-8"))
    paths = [ROOT / item["path"] for item in prior["entries"] if (ROOT / item["path"]).exists()]
    for base in [ROOT / "artifacts/results/stage7", ROOT / "artifacts/figures/stage7", ROOT / "artifacts/manifests/stage7", ROOT / "artifacts/reports"]:
        if base.exists():
            paths.extend(p for p in base.rglob("*") if p.is_file() and ("stage7" in p.name.lower() or "stage7" in str(p.parent).lower()))
    paths.extend([ROOT / "data/regression_without_sensitive_features.csv", ROOT / "data/regression_with_sensitive_features.csv", ROOT / "REGRESSION_PART7_FAIRNESS_SENSITIVE_ANALYSIS.ipynb", REGISTRY])
    return sorted(set(paths), key=lambda p: p.as_posix().lower())


def preflight() -> None:
    start = time.perf_counter()
    for directory in [RESULTS, FIGURES, PLOTTING, MANIFESTS, REPORTS, BACKUPS]:
        directory.mkdir(parents=True, exist_ok=True)
    checks = validate_prerequisites()
    notebook = ROOT / "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb"
    if notebook.exists():
        shutil.copy2(notebook, BACKUPS / f"REGRESSION_PART8_FINAL_EXPLAINABILITY.{datetime.now().strftime('%Y%m%dT%H%M%S')}.ipynb")
    samples = select_samples()
    entries = []
    for p in protected_files():
        try:
            label = str(p.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            label = str(p.resolve()).replace("\\", "/")
        entries.append({"path": label, "sha256": sha(p), "size_bytes": p.stat().st_size})
    reg = pd.read_csv(REGISTRY)
    baseline = {"stage_id": "stage8", "created_at_utc": now(), "entries": entries, "protected_file_count": len(entries), "registry_rows_before": len(reg), "registry_ids_before": reg["experiment_id"].astype(str).tolist(), "registry_sha256_before": sha(REGISTRY)}
    dump(baseline, MANIFESTS / "stage8_protected_hashes_before.json")
    freeze = {
        "stage_id": "stage8", "created_at_utc": now(), "analysis_label": LABEL, "prerequisites": checks,
        "stage4l_official_candidate": CANDIDATES[0], "stage4l_pretest_freeze": record("artifacts/reports/stage4l_pretest_freeze.json"),
        "stage5a_governance_exception": "PASS_WITH_DOCUMENTED_GOVERNANCE_EXCEPTION", "stage5a_deep_attribution": record(EXPLAIN_METADATA[-1]),
        "stage5b_ensemble_status": "rejected", "stage9_started": False, "candidate_ids": CANDIDATES, "model_identities": MODELS, "prediction_artifacts": PREDICTIONS,
        "official_blend_weights": {"catboost": 0.60, "lightgbm": 0.20, "xgboost": 0.20}, "expected_test_contract": EXPECTED,
        "existing_explainability_metadata": [record(rel) for rel in EXPLAIN_METADATA],
        "methods": ["reused native tree importance", "reused native tree SHAP", "reused Stage 5A attribution", "common grouped permutation importance", "local reference substitution"],
        "sample_contract": {**samples, "global_policy": "200 rows per target decile; seed 42; sorted row_id", "background_policy": "4 rows per decile from global sample excluding local cases; seed 42", "local_policy": "all 20 Stage 7 public cases", "local_stability_cases": 4},
        "feature_unit_policy": "Canonical semantic units fixed before feature access; one shared row permutation or reference replacement per unit; ratios remain separate.",
        "permutation_repeats": 2, "permutation_seeds": [42, 43], "local_background_rows": 40, "local_case_maximum": 20,
        "local_replacement_formula": "effect = prediction(case) - mean prediction(case with one semantic unit replaced by each of 40 reference rows)",
        "stability_calculations": ["repeat Spearman", "top-10 overlap", "top-20 overlap", "rank movement", "four-case background-half stability"],
        "dense_plotting_sample_limit": 2000, "figures": [{"figure_id": i + 1, "title": title} for i, title in enumerate(FIGURE_TITLES)],
        "registry_ids": REGISTRY_IDS, "notebook_attempt_limit": 3, "reviewer_cycle_limit": 2,
        "privacy_policy": "No public raw sensitive values; explicit sensitive local features are joint blocks; aggregate-only sensitive dependence.",
        "prohibitions": {"model_fit": True, "preprocessing_fit": True, "surrogate_fit": True, "new_prediction_file": True, "full_test_shap": True, "global_shap_recomputation": True},
        "stage9_next_step": "Begin Stage 9 — Model Card and Final Technical Report only after Stage 8 PASS.",
        "no_explainability_value_parsed_yet": True, "no_source_feature_value_materialized_yet": True, "no_model_deserialized_yet": True,
        "protected_baseline_sha256": sha(MANIFESTS / "stage8_protected_hashes_before.json"),
    }
    dump(freeze, REPORTS / "stage8_preexplainability_freeze.json")
    reloaded = json.loads((REPORTS / "stage8_preexplainability_freeze.json").read_text(encoding="utf-8"))
    if len(reloaded["candidate_ids"]) != 3 or len(reloaded["model_identities"]) != 5 or len(reloaded["figures"]) != 15:
        raise RuntimeError("Freeze reload validation failed")
    dump({"phase": "preflight", "status": "PASS", "seconds": time.perf_counter() - start, "freeze_sha256": sha(REPORTS / "stage8_preexplainability_freeze.json")}, REPORTS / "stage8_runtime_preflight.json")
    print(json.dumps({"status": "PASS", "protected": len(entries), "samples": samples, "freeze_sha256": sha(REPORTS / "stage8_preexplainability_freeze.json")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight"])
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()


if __name__ == "__main__":
    main()
