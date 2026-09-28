"""Independent saved-artifact verification and Prompt 5B promotion."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


STATUS = "PASS_FINAL_FAIRNESS_AND_EXPLAINABILITY"
AUTHORIZATION_ID = "regression_v2_prompt5b_fairness_explainability"
EXPECTED_ROWS = 75_000
SAMPLE_SIZE = 4_000
FREEZE_SHA = "8f2b8e4fda80056770b916b7859ad0f6f89f2948236320e6815e082fadca35c1"
PRIMARY_SHA = "5349ab15fd1c8182ef539f435cc4f065e71ee7da047de09a4dc271c9097e0c08"
GLOBAL_SHA = "6f61a0be1fc90d2331b08dada63a453f6c5410d5783f46f1e0cbcbf425f75597"
REPORTS = Path("outputs/reports")
EXPLAIN = Path("outputs/explainability/prompt5b")
FIGURES = Path("outputs/figures/prompt5b")
NOTEBOOK = Path("notebooks/05B_FAIRNESS_AND_FINAL_EXPLAINABILITY.ipynb")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def _labels(series: pd.Series) -> pd.Series:
    return series.map(lambda value: "__MISSING__" if pd.isna(value) else str(value))


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.failures: list[str] = []

    def add(self, name: str, condition: bool, evidence: Any) -> None:
        passed = bool(condition)
        self.items.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})
        if not passed:
            self.failures.append(name)


def verify(root: Path) -> dict[str, Any]:
    checks = Checks()
    ready5a = read_json(root / REPORTS / "PROMPT5A_READY.json")
    checks.add("Prompt 5A readiness PASS", ready5a.get("status") == "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS", ready5a.get("status"))
    checks.add("FINAL_PRE_IID_FREEZE immutable", sha256(root / REPORTS / "FINAL_PRE_IID_FREEZE.json") == FREEZE_SHA, sha256(root / REPORTS / "FINAL_PRE_IID_FREEZE.json"))
    checks.add("Primary bundle immutable", sha256(root / "outputs/models/final_pre_iid/primary_stage3/bundle.joblib") == PRIMARY_SHA, sha256(root / "outputs/models/final_pre_iid/primary_stage3/bundle.joblib"))
    checks.add("Global bundle immutable", sha256(root / "outputs/models/final_pre_iid/global_comparator/bundle.joblib") == GLOBAL_SHA, sha256(root / "outputs/models/final_pre_iid/global_comparator/bundle.joblib"))

    snapshot_manifest = read_json(root / REPORTS / "prompt5a_post_iid_snapshot_manifest.json")
    snapshot_failures = []
    for name, item in snapshot_manifest["artifacts"].items():
        path = root / item["path"]
        if sha256(path) != item["sha256"] or pq.ParquetFile(path).metadata.num_rows != item["rows"]:
            snapshot_failures.append(name)
    checks.add("all post-IID snapshot hashes unchanged", not snapshot_failures, snapshot_failures)

    evaluation = pd.read_parquet(root / "outputs/data/post_iid/iid_evaluation_frame.parquet")
    fairness = pd.read_parquet(root / "outputs/data/post_iid/iid_fairness_audit_snapshot.parquet")
    merged = evaluation[["row_hash"]].merge(fairness, on="row_hash", how="outer", validate="one_to_one", indicator=True)
    checks.add("75,000 fairness/evaluation alignment", len(merged) == EXPECTED_ROWS and merged["_merge"].eq("both").all(), {"rows": len(merged), "unique": int(merged["row_hash"].nunique())})

    roles = read_json(root / REPORTS / "feature_roles.json")
    sensitive = list(roles["sensitive_fields"])
    feature_contract = list(roles["contracts"]["main_without_sensitive_without_lender"])
    contract = read_json(root / REPORTS / "prompt5b_sensitive_contract.json")
    checks.add("exact frozen sensitive contract", contract["exact_column_names"] == sensitive and fairness.columns.tolist()[1:] == sensitive and len(sensitive) == 8, sensitive)
    checks.add("sensitive fields not predictive", not set(sensitive).intersection(feature_contract), sorted(set(sensitive).intersection(feature_contract)))
    checks.add("respondent_id not predictive", "respondent_id" not in feature_contract, "respondent_id" in feature_contract)

    inventory = pd.read_csv(root / REPORTS / "prompt5b_group_inventory.csv")
    expected_levels = sum(fairness[field].nunique(dropna=False) for field in sensitive)
    expected_counts = {
        (field, str(label)): int(count)
        for field in sensitive
        for label, count in _labels(fairness[field]).value_counts(dropna=False).items()
    }
    reported_counts = {(row.sensitive_field, row.group_label): int(row.n) for row in inventory.itertuples(index=False)}
    checks.add("group inventory complete", len(inventory) == expected_levels and reported_counts == expected_counts, {"expected_levels": expected_levels, "reported_levels": len(inventory)})
    composition_columns = {"mean_target", "median_target", "d10_fraction", "top5_target_fraction"}
    checks.add("group inventory includes target composition", composition_columns.issubset(inventory.columns), sorted(composition_columns.intersection(inventory.columns)))

    primary = pd.read_csv(root / REPORTS / "prompt5b_primary_group_metrics.csv")
    global_metrics = pd.read_csv(root / REPORTS / "prompt5b_global_group_metrics.csv")
    comparison = pd.read_csv(root / REPORTS / "prompt5b_group_primary_vs_global.csv")
    small = primary[primary["n"] < 200]
    eligible = primary[primary["n"] >= 200]
    checks.add("small-group rule enforced", small["analysis_status"].eq("SMALL_GROUP").all() and small["mae"].isna().all() and eligible["analysis_status"].eq("ELIGIBLE").all(), {"small": len(small), "eligible": len(eligible)})

    analysis = evaluation.merge(fairness, on="row_hash", validate="one_to_one")
    q90 = float(np.quantile(analysis["y_true"], .90, method="linear"))
    q95 = float(np.quantile(analysis["y_true"], .95, method="linear"))
    analysis["iid_global_top_decile"] = analysis["y_true"].ge(q90)
    analysis["iid_top5_target"] = analysis["y_true"].ge(q95)
    formula_max = 0.0
    comparison_max = 0.0
    for row in eligible.itertuples(index=False):
        part = analysis[_labels(analysis[row.sensitive_field]).eq(row.group_label)]
        signed = part["primary_prediction"].to_numpy(float) - part["y_true"].to_numpy(float)
        formula_max = max(formula_max, abs(float(np.abs(signed).mean()) - row.mae), abs(float(signed.mean()) - row.mean_signed_error), abs(float(np.mean(signed < 0)) - row.underprediction_rate))
        comp = comparison[(comparison["sensitive_field"].eq(row.sensitive_field)) & (comparison["group_label"].eq(row.group_label))].iloc[0]
        comparison_max = max(comparison_max, abs(float((part["primary_abs_error"] - part["global_abs_error"]).mean()) - comp["primary_minus_global_mae"]))
    checks.add("eligible Primary metric formulas reproducible", formula_max < 1e-10, formula_max)
    checks.add("Primary-versus-Global formulas reproducible", comparison_max < 1e-10, comparison_max)
    checks.add("Global metric row parity", len(global_metrics) == len(primary) and global_metrics["analysis_status"].equals(primary["analysis_status"]), len(global_metrics))

    tail = pd.read_csv(root / REPORTS / "prompt5b_tail_group_metrics.csv")
    tail_n_mismatch = 0
    for row in tail.itertuples(index=False):
        mask = analysis["iid_global_top_decile"] if row.tail_scope == "IID_GLOBAL_TOP_DECILE" else analysis["iid_top5_target"]
        expected_n = int((_labels(analysis.loc[mask, row.sensitive_field]).eq(row.group_label)).sum())
        tail_n_mismatch += int(expected_n != row.n)
    checks.add("common IID-global Tail masks", tail_n_mismatch == 0, tail_n_mismatch)
    checks.add("Tail n>=50 display rule", tail[tail["n"] < 50]["primary_tail_mae"].isna().all() and tail[tail["n"] >= 50]["primary_tail_mae"].notna().all(), {"eligible": int((tail["n"] >= 50).sum())})

    decile = pd.read_csv(root / REPORTS / "prompt5b_group_decile_metrics.csv")
    checks.add("group-decile n>=30 display rule", decile[decile["n"] < 30]["primary_mae"].isna().all() and decile[decile["n"] >= 30]["primary_mae"].notna().all(), {"cells": len(decile)})
    intersections = pd.read_csv(root / REPORTS / "prompt5b_intersectional_metrics.csv")
    expected_intersections = {"applicant_race_x_applicant_sex", "applicant_ethnicity_x_applicant_sex"}
    checks.add("intersectional analysis bounded", set(intersections["intersection"]) == expected_intersections and set(intersections["scope"]) == {"OVERALL", "IID_GLOBAL_TOP_DECILE", "IID_TOP5_TARGET"}, sorted(intersections["intersection"].unique()))

    bootstrap = pd.read_csv(root / REPORTS / "prompt5b_fairness_bootstrap.csv")
    checks.add("fairness bootstrap settings", set(bootstrap["resamples"]) == {500} and set(bootstrap["seed"]) == {42}, {"rows": len(bootstrap), "resamples": sorted(bootstrap["resamples"].unique()), "seed": sorted(bootstrap["seed"].unique())})
    checks.add("fairness flags are descriptive", pd.read_csv(root / REPORTS / "prompt5b_fairness_flags.csv")["legal_fairness_determination"].eq(False).all(), "all false")

    features = pd.read_parquet(root / "outputs/data/post_iid/iid_model_features_snapshot.parquet", columns=["row_hash"])
    score = features["row_hash"].astype(str).map(lambda value: hashlib.sha256(("prompt5b_explainability_seed42" + value).encode()).hexdigest())
    expected_sample = features.loc[score.sort_values(kind="mergesort").head(SAMPLE_SIZE).index, "row_hash"].astype(str).tolist()
    sample = pd.read_parquet(root / EXPLAIN / "sample_row_hashes.parquet")
    checks.add("deterministic 4,000-row explainability sample", sample["row_hash"].astype(str).tolist() == expected_sample and len(sample) == SAMPLE_SIZE, {"rows": len(sample)})
    sample_manifest = read_json(root / REPORTS / "prompt5b_explainability_sample_manifest.json")
    checks.add("sample independent of target and sensitive fields", sample_manifest["sensitive_fields_used_for_selection"] == [] and not sample_manifest["target_or_error_used_for_main_selection"], sample_manifest["sample_rule"])

    attribution_names = ["catboost", "lightgbm", "xgboost", "gate", "residual"]
    attribution_evidence = {}
    valid_attributions = True
    for name in attribution_names:
        path = root / EXPLAIN / f"{name}_shap.parquet"
        artifact = pd.read_parquet(path)
        numeric = artifact.drop(columns="row_hash").to_numpy(float)
        condition = len(artifact) == SAMPLE_SIZE and np.isfinite(numeric).all() and artifact["row_hash"].astype(str).tolist() == expected_sample
        valid_attributions &= condition
        attribution_evidence[name] = {"rows": len(artifact), "columns": len(artifact.columns), "finite": bool(np.isfinite(numeric).all())}
    checks.add("component attribution artifacts valid", valid_attributions, attribution_evidence)
    errors = sample_manifest["maximum_native_additivity_errors"]
    checks.add("component native additivity", errors["xgboost"] < 1e-3 and all(errors[name] < 1e-8 for name in ("catboost", "lightgbm", "gate", "residual")), errors)
    checks.add("XGBoost native-scale disclosure", "log1p" in sample_manifest["xgboost_warning"] and "not raw-scale comparable" in sample_manifest["xgboost_warning"], sample_manifest["xgboost_warning"])
    checks.add("no single composite SHAP claim", "No single additive SHAP" in sample_manifest["composite_warning"], sample_manifest["composite_warning"])

    importance = pd.read_csv(root / REPORTS / "prompt5b_global_feature_importance.csv")
    reproduced = .6 * importance["catboost_normalized_rank_score"] + .2 * importance["lightgbm_normalized_rank_score"] + .2 * importance["xgboost_normalized_rank_score"]
    checks.add("Global rank consensus reproducible", float(np.max(np.abs(reproduced - importance["consensus_rank_score"]))) < 1e-12 and not importance["raw_shap_values_summed_across_models"].any(), float(np.max(np.abs(reproduced - importance["consensus_rank_score"]))))
    gate = pd.read_csv(root / REPORTS / "prompt5b_gate_feature_importance.csv")
    residual = pd.read_csv(root / REPORTS / "prompt5b_residual_feature_importance.csv")
    checks.add("Gate importance correctly labeled", gate["importance_type"].eq("Gate importance").all() and "global_prediction_feature" in set(gate["feature"]), gate.head(3)["feature"].tolist())
    checks.add("Residual importance correctly labeled", residual["importance_type"].eq("Residual correction importance").all(), residual.head(3)["feature"].tolist())
    stability = pd.read_csv(root / REPORTS / "prompt5b_explainability_stability.csv")
    checks.add("SHAP stability complete", set(stability["component"]) == set(attribution_names) and stability["spearman_rank_correlation"].notna().all(), stability[["component", "top10_overlap_count", "spearman_rank_correlation"]].to_dict(orient="records"))

    local = pd.read_csv(root / REPORTS / "prompt5b_local_cases.csv")
    expected_roles = [
        "CASE_1_LARGEST_PRIMARY_ABSOLUTE_ERROR", "CASE_2_LARGEST_PRIMARY_IMPROVEMENT", "CASE_3_LARGEST_PRIMARY_DAMAGE",
        "CASE_4_D10_CLOSEST_SIGNED_ERROR_TO_ZERO", "CASE_5_D10_LARGEST_UNDERPREDICTION", "CASE_6_BODY_LARGEST_APPLIED_CORRECTION",
    ]
    checks.add("six deterministic local case roles", local["case_role"].tolist() == expected_roles and len(local) == 6 and not local["sensitive_values_disclosed"].any(), local["case_role"].tolist())
    case_map = pd.read_parquet(root / EXPLAIN / "local_case_row_hashes.parquet")
    expected_hashes = [
        evaluation.loc[evaluation["primary_abs_error"].idxmax(), "row_hash"],
        evaluation.loc[evaluation["delta_abs_error"].idxmin(), "row_hash"],
        evaluation.loc[evaluation["delta_abs_error"].idxmax(), "row_hash"],
        evaluation.loc[evaluation[evaluation["iid_local_decile"].eq("D10")]["primary_signed_error"].abs().idxmin(), "row_hash"],
        evaluation.loc[evaluation[evaluation["iid_local_decile"].eq("D10")]["primary_signed_error"].idxmin(), "row_hash"],
        evaluation.loc[evaluation[~evaluation["iid_local_decile"].eq("D10")]["applied_residual_correction"].abs().idxmax(), "row_hash"],
    ]
    checks.add("local case selection reproducible", case_map["row_hash"].astype(str).tolist() == [str(x) for x in expected_hashes], case_map["anonymized_row_id"].tolist())

    notebook_report = read_json(root / REPORTS / "prompt5b_notebook_execution.json")
    notebook = nbformat.read(root / NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    prohibited_notebook = [token for token in ("iid_holdout_features", "iid_holdout_targets", ".fit(", "joblib.load", "get_feature_importance", "pred_contrib") if token in source]
    checks.add("artifact-only inline notebook", notebook_report["status"] == "PASS" and notebook_report["inline_images"] >= 15 and notebook_report["inline_tables"] >= 15 and not prohibited_notebook, {"images": notebook_report["inline_images"], "tables": notebook_report["inline_tables"], "prohibited": prohibited_notebook})
    checks.add("required statistical figures", len(list((root / FIGURES).glob("*.png"))) >= 15, len(list((root / FIGURES).glob("*.png"))))

    reviewer = read_json(root / REPORTS / "prompt5b_reviewer.json")
    checks.add("exactly one reviewer PASS", reviewer["status"] == "PASS" and reviewer["reviewer_count"] == 1 and reviewer["critical_findings"] == 0 and reviewer["major_findings"] == 0, {"status": reviewer["status"], "count": reviewer["reviewer_count"], "findings": reviewer["findings"]})
    runtime = read_json(root / REPORTS / "prompt5b_runtime.json")
    checks.add("zero fit/refit/tuning/model change", all(runtime[key] == 0 for key in ("original_iid_rereads", "model_fit_count", "preprocessor_fit_count", "calibration_fit_count", "hpo_count", "threshold_search_count", "model_change_count")), {key: runtime[key] for key in ("original_iid_rereads", "model_fit_count", "preprocessor_fit_count", "calibration_fit_count", "hpo_count", "threshold_search_count", "model_change_count")})
    tree = ast.parse((root / "src/prompt5b_analysis.py").read_text(encoding="utf-8"))
    training_calls = [(node.lineno, node.func.attr) for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform", "partial_fit"}]
    checks.add("static no-training implementation", training_calls == [], training_calls)

    candidate = read_json(root / REPORTS / "prompt5b_final_candidate.json")
    candidate_hash_failures = [name for name, expected in candidate["fairness_report_hashes"].items() if sha256(root / REPORTS / name) != expected]
    candidate_explain_failures = [name for name, expected in candidate["explainability_artifact_hashes"].items() if sha256(root / EXPLAIN / name) != expected]
    checks.add("candidate report hashes", not candidate_hash_failures and not candidate_explain_failures and sha256(root / NOTEBOOK) == candidate["notebook_sha256"], {"report_failures": candidate_hash_failures, "explain_failures": candidate_explain_failures})
    checks.add("no model-selection recommendation", "recommend" not in json.dumps(candidate).lower(), "candidate contains no recommendation")

    payload = {
        "status": "PASS" if not checks.failures else "FAIL",
        "created_at_utc": utc_now(), "authorization_id": AUTHORIZATION_ID,
        "independent_of_analysis_orchestration": True,
        "check_count": len(checks.items), "checks": checks.items, "failures": checks.failures,
        "original_iid_rereads": 0, "model_fits": 0, "model_refits": 0, "tuning": 0, "model_changes": 0,
    }
    atomic_json(root / REPORTS / "prompt5b_verification.json", payload)
    if checks.failures:
        raise RuntimeError(f"Prompt 5B verification failed: {checks.failures}")
    return payload


def promote(root: Path) -> dict[str, Any]:
    reviewer = read_json(root / REPORTS / "prompt5b_reviewer.json")
    verification = read_json(root / REPORTS / "prompt5b_verification.json")
    candidate_path = root / REPORTS / "prompt5b_final_candidate.json"
    candidate = read_json(candidate_path)
    if reviewer["status"] != "PASS" or verification["status"] != "PASS":
        raise RuntimeError("Reviewer and verification must pass before promotion.")
    final_path = root / REPORTS / "FINAL_FAIRNESS_EXPLAINABILITY.json"
    if final_path.exists():
        raise RuntimeError("FINAL_FAIRNESS_EXPLAINABILITY.json already exists and is immutable.")
    payload = {
        **candidate,
        "status": STATUS,
        "promoted_at_utc": utc_now(),
        "promotion": {
            "candidate_sha256": sha256(candidate_path),
            "reviewer_sha256": sha256(root / REPORTS / "prompt5b_reviewer.json"),
            "verification_sha256": sha256(root / REPORTS / "prompt5b_verification.json"),
            "reviewer_status": reviewer["status"],
            "verification_status": verification["status"],
        },
    }
    atomic_json(final_path, payload)
    reloaded = read_json(final_path)
    if reloaded["status"] != STATUS:
        raise RuntimeError("Final Prompt 5B handoff reload failed.")
    return {"status": reloaded["status"], "path": final_path.as_posix(), "sha256": sha256(final_path)}


def readiness(root: Path) -> dict[str, Any]:
    final_path = root / REPORTS / "FINAL_FAIRNESS_EXPLAINABILITY.json"
    final = read_json(final_path)
    reviewer = read_json(root / REPORTS / "prompt5b_reviewer.json")
    verification = read_json(root / REPORTS / "prompt5b_verification.json")
    if final["status"] != STATUS or reviewer["status"] != "PASS" or verification["status"] != "PASS":
        raise RuntimeError("Prompt 5B is not readiness-eligible.")
    path = root / REPORTS / "PROMPT5B_READY.json"
    if path.exists():
        raise RuntimeError("PROMPT5B_READY.json already exists; no later Prompt 5B report is authorized.")
    payload = {
        "status": STATUS, "created_at_utc": utc_now(), "authorization_id": AUTHORIZATION_ID,
        "prompt5b_complete": True,
        "final_iid_evaluation_sha256": sha256(root / REPORTS / "FINAL_IID_EVALUATION.json"),
        "final_fairness_explainability_sha256": sha256(final_path),
        "reviewer_status": reviewer["status"], "verification_status": verification["status"],
        "zero_fit": True, "zero_refit": True, "zero_tuning": True, "zero_model_change": True,
        "original_iid_files_status": "CLOSED_AFTER_ONE_TIME_EVALUATION",
        "next_step": "Prompt 5C — Final Model Card, Technical Report, and Paper-Ready Outputs",
        "report_creation_rule": "This is the last Prompt 5B report artifact.",
    }
    atomic_json(path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify", "promote", "ready"))
    parser.add_argument("--root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    if args.command == "verify":
        result = verify(root)
    elif args.command == "promote":
        result = promote(root)
    else:
        result = readiness(root)
    print(json.dumps(result, indent=2, default=_default))


if __name__ == "__main__":
    main()
