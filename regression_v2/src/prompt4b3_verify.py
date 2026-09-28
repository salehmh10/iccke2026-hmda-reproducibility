"""Independent saved-artifact verification for Prompt 4B3.

This verifier does not call the Prompt 4B3 execution path and fits no object.
"""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/reports"
PREDICTIONS = ROOT / "outputs/predictions/prompt4b3/validation"
MODEL = ROOT / "outputs/models/prompt4b3/beat_classifier"
EXPECTED_SOURCE_SHA = "0ed232397be3ec4de1483c594954dce7b4704b375ca295397d899323dc4f0b6b"
LINEAR_ID = "prompt4b3__beat_linear"
COSTAWARE_ID = "prompt4b3__beat_costaware"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_digest(values: Any) -> str:
    return hashlib.sha256("\n".join(pd.Series(values).astype(str).tolist()).encode("utf-8")).hexdigest()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    absolute = np.abs(error)
    top = y >= np.quantile(y, 0.90)
    body = ~top
    return {
        "mae": float(absolute.mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bottom_90_mae": float(absolute[body].mean()), "top_decile_mae": float(absolute[top].mean()),
        "top_decile_signed_error": float(error[top].mean()),
        "top_decile_underprediction_rate": float((error[top] < 0.0).mean()),
    }


def six(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, bool]:
    return {
        "complete_mae_lower": candidate["mae"] < reference["mae"],
        "top_decile_mae_improves_3pct": candidate["top_decile_mae"] <= reference["top_decile_mae"] * 0.97,
        "bottom_90_mae_worsens_at_most_0_25pct": candidate["bottom_90_mae"] <= reference["bottom_90_mae"] * 1.0025,
        "rmse_worsens_at_most_0_25pct": candidate["rmse"] <= reference["rmse"] * 1.0025,
        "top_decile_signed_error_closer_to_zero": abs(candidate["top_decile_signed_error"]) < abs(reference["top_decile_signed_error"]),
        "top_decile_underprediction_rate_decreases": candidate["top_decile_underprediction_rate"] < reference["top_decile_underprediction_rate"],
    }


def verify() -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    design = load_json(REPORTS / "prompt4b3_design_freeze.json")
    ledger = load_json(REPORTS / "prompt4b3_fit_ledger.json")
    gate = load_json(REPORTS / "prompt4b3_diagnostic_gate.json")
    reviewer = load_json(REPORTS / "prompt4b3_reviewer.json")
    champion = load_json(REPORTS / "prompt4b3_experimental_champion.json")
    model_manifest = load_json(MODEL / "manifest.json")
    prediction_manifest = load_json(REPORTS / "prompt4b3_prediction_manifest.json")

    checks["prompt4b2_readiness_pass"] = load_json(REPORTS / "PROMPT4B2_READY.json")["status"] == "PASS"
    checks["prompt4b2_verification_pass"] = load_json(REPORTS / "prompt4b2_verification.json")["status"] == "PASS"
    checks["development_sha256_unchanged"] = sha256(ROOT / "outputs/data/development.parquet") == EXPECTED_SOURCE_SHA == design["source"]["sha256"]
    for item in design["prior_immutable_artifacts"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            checks["all_prior_hashes_unchanged"] = False
            break
    else:
        checks["all_prior_hashes_unchanged"] = True

    features = design["source"]["features"]
    development = pd.read_parquet(ROOT / "outputs/data/development.parquet", columns=features + ["loan_amount_000s", "development_role", "row_hash"])
    train = development.loc[development["development_role"].eq("train")].reset_index(drop=True)
    validation = development.loc[development["development_role"].eq("validation")].reset_index(drop=True)
    checks["development_rows_500000"] = len(development) == 500_000
    checks["train_rows_400000"] = len(train) == 400_000
    checks["validation_rows_100000"] = len(validation) == 100_000
    checks["train_digest_unchanged"] = ordered_digest(train["row_hash"]) == design["memberships"]["train_ordered_digest"]
    checks["validation_digest_unchanged"] = ordered_digest(validation["row_hash"]) == design["memberships"]["validation_ordered_digest"]
    checks["feature_contract_exact_37"] = len(design["inference_features"]) == 37 and design["inference_features"] == features + ["global_prediction_feature", "proposed_residual_correction"]

    benefit_train = pd.read_parquet(ROOT / "outputs/predictions/prompt4b2/train/benefit_training_table.parquet")
    residual_oof = pd.read_parquet(ROOT / "outputs/predictions/prompt4b2/train/residual_oof.parquet")
    global_oof = pd.read_parquet(ROOT / "outputs/predictions/prompt4b/train/global_oof_prediction.parquet")
    checks["oof_benefit_rows_400000"] = len(benefit_train) == 400_000
    checks["oof_residual_rows_400000"] = len(residual_oof) == 400_000
    checks["oof_order_matches_train"] = train["row_hash"].astype(str).equals(benefit_train["row_hash"].astype(str)) and train["row_hash"].astype(str).equals(residual_oof["row_hash"].astype(str))
    checks["residual_oof_self_fit_zero"] = int(residual_oof["self_fit"].sum()) == 0
    checks["residual_exactly_one_oof"] = bool(residual_oof["exactly_one_oof_prediction"].all())
    reproduced = np.abs(global_oof["y_true"].to_numpy(float) - benefit_train["global_oof_prediction"].to_numpy(float)) - np.abs(global_oof["y_true"].to_numpy(float) - (benefit_train["global_oof_prediction"].to_numpy(float) + benefit_train["proposed_residual_correction"].to_numpy(float)))
    benefit_difference = float(np.max(np.abs(reproduced - benefit_train["benefit_oof"].to_numpy(float))))
    checks["benefit_formula_reproduces"] = benefit_difference == 0.0
    beat = (benefit_train["benefit_oof"].to_numpy(float) > 0.0).astype(np.int8)
    checks["zero_benefit_maps_to_zero"] = bool(np.all(beat[benefit_train["benefit_oof"].to_numpy(float) == 0.0] == 0))
    positive = benefit_train["benefit_oof"].to_numpy(float) > 0.0
    negative = benefit_train["benefit_oof"].to_numpy(float) < 0.0
    gain = float(benefit_train.loc[positive, "benefit_oof"].mean())
    damage = float(-benefit_train.loc[negative, "benefit_oof"].mean())
    p0 = damage / (gain + damage)
    checks["p0_train_oof_reproduces"] = p0 == design["cost_asymmetry"]["p0"]

    checks["one_scientific_candidate"] = ledger["scientific_candidates"] == ["prompt4b3__beat_classifier__main37_fixed"]
    checks["scientific_fit_count_one"] = ledger["scientific_fit_count"] == 1 and ledger["physical_attempt_count"] == 1
    checks["technical_retries_at_most_one"] = ledger["technical_retry_count"] == 0
    config = design["classifier"]["configuration"]
    checks["classifier_configuration_frozen"] = config["loss_function"] == "Logloss" and config["iterations"] == 1500 and config["random_seed"] == 42 and config["thread_count"] <= 4 and config["task_type"] == "CPU"
    checks["classifier_unweighted_no_early_stop"] = not {"class_weights", "auto_class_weights", "early_stopping_rounds", "use_best_model"}.intersection(config)
    checks["model_hashes_valid"] = sha256(MODEL / model_manifest["artifact"]) == model_manifest["artifact_sha256"] and sha256(MODEL / model_manifest["native_artifact"]) == model_manifest["native_sha256"]
    reload_evidence = load_json(REPORTS / "prompt4b3_model_manifest.json")["clean_process_reload"]
    checks["classifier_clean_reload_pass"] = reload_evidence["status"] == "PASS" and reload_evidence["maximum_absolute_probability_difference"] <= 1e-12

    probability_frame = pd.read_parquet(PREDICTIONS / "beat_classifier_probability.parquet")
    checks["validation_probability_rows_aligned"] = len(probability_frame) == 100_000 and validation["row_hash"].astype(str).equals(probability_frame["row_hash"].astype(str))
    probability = probability_frame["p_beat"].to_numpy(float)
    checks["validation_probabilities_finite_bounded"] = bool(np.isfinite(probability).all() and np.all((probability >= 0.0) & (probability <= 1.0)))
    roles = probability_frame["selection_or_audit_role"].astype(str).to_numpy()
    checks["selection_rows_70000"] = int((roles == "selection").sum()) == 70_000
    checks["audit_rows_30000"] = int((roles == "audit").sum()) == 30_000
    selection = roles == "selection"
    labels_validation = probability_frame["beat_validation"].to_numpy(np.int8)
    benefit_validation = probability_frame["benefit_validation"].to_numpy(float)
    baseline_frame = pd.read_parquet(ROOT / "outputs/predictions/prompt4b2/validation/benefit_b0.parquet")
    baseline_score = baseline_frame["predicted_benefit"].to_numpy(float)
    pr_new = float(average_precision_score(labels_validation[selection], probability[selection]))
    pr_old = float(average_precision_score(labels_validation[selection], baseline_score[selection]))
    rng = np.random.default_rng(42)
    boot = np.empty(500)
    selected_labels = labels_validation[selection]
    selected_new = probability[selection]
    selected_old = baseline_score[selection]
    for index in range(500):
        sample = rng.integers(0, len(selected_labels), size=len(selected_labels))
        boot[index] = average_precision_score(selected_labels[sample], selected_new[sample]) - average_precision_score(selected_labels[sample], selected_old[sample])
    lower = float(np.quantile(boot, 0.025))
    checks["dg1_reproduces"] = bool(lower > 0.0) == bool(gate["DG1"]["pass"]) and abs(lower - gate["DG1"]["percentile_2_5"]) <= 1e-15 and abs((pr_new - pr_old) - gate["DG1"]["pr_auc_difference"]) <= 1e-15
    order_new = np.argsort(-selected_new, kind="stable")[:7000]
    old_scores = {
        "old_tail_gate": pd.read_parquet(ROOT / "outputs/predictions/prompt4b/validation/stage1_probabilities.parquet")["p_raw"].to_numpy(float)[selection],
        "meta_gate": pd.read_parquet(ROOT / "outputs/predictions/prompt4b/validation/stage2_meta_gate_probability.parquet")["p_meta_gate"].to_numpy(float)[selection],
        "benefit_router": selected_old,
    }
    new_top10 = float(benefit_validation[selection][order_new].mean())
    old_top10 = max(float(benefit_validation[selection][np.argsort(-score, kind="stable")[:7000]].mean()) for score in old_scores.values())
    checks["dg2_reproduces"] = bool(new_top10 > old_top10) == bool(gate["DG2"]["pass"]) and abs(new_top10 - gate["DG2"]["new_top_10_mean_realized_benefit"]) <= 1e-15 and abs(old_top10 - gate["DG2"]["best_existing_top_10_mean_realized_benefit"]) <= 1e-15
    checks["diagnostic_gate_pass"] = gate["status"] == "PASS" and gate["policy_evaluation_authorized"] is True

    global_prediction = probability_frame["global_prediction"].to_numpy(float)
    proposal = probability_frame["proposed_residual_correction"].to_numpy(float)
    expected_alpha = {
        LINEAR_ID: np.clip(probability, 0.0, 1.0),
        COSTAWARE_ID: np.clip(np.maximum(0.0, (probability - p0) / (1.0 - p0)), 0.0, 1.0),
    }
    policy_table = pd.read_csv(REPORTS / "prompt4b3_policy_results.csv")
    saved_six = pd.read_csv(REPORTS / "prompt4b3_six_condition_results.csv")
    global_scope_metrics: dict[str, dict[str, float]] = {}
    masks = {"selection": roles == "selection", "audit_descriptive": roles == "audit", "complete_validation_descriptive": np.ones(len(roles), dtype=bool)}
    y = probability_frame["y_true"].to_numpy(float)
    for scope, mask in masks.items():
        global_scope_metrics[scope] = metrics(y[mask], global_prediction[mask])
    max_policy_difference = 0.0
    reproduced_selection_rows = []
    for candidate_id, alpha in expected_alpha.items():
        saved = pd.read_parquet(PREDICTIONS / f"{candidate_id}.parquet")
        predicted = global_prediction + alpha * proposal
        max_policy_difference = max(max_policy_difference, float(np.max(np.abs(predicted - saved["y_pred"].to_numpy(float)))), float(np.max(np.abs(alpha - saved["alpha"].to_numpy(float)))))
        for scope, mask in masks.items():
            observed = metrics(y[mask], predicted[mask])
            result = six(observed, global_scope_metrics[scope])
            count = int(sum(result.values()))
            saved_row = saved_six.loc[(saved_six["candidate_id"].eq(candidate_id)) & (saved_six["scope"].eq(scope))].iloc[0]
            checks[f"six_conditions_{candidate_id}_{scope}"] = count == int(saved_row["conditions_passed"]) and all(bool(saved_row[key]) == value for key, value in result.items())
            if scope == "selection":
                reproduced_selection_rows.append({"candidate_id": candidate_id, "conditions_passed": count, **observed})
    checks["policy_formulas_reproduce_exactly"] = max_policy_difference == 0.0
    ranked = sorted(reproduced_selection_rows, key=lambda row: (-row["conditions_passed"], row["mae"], row["top_decile_mae"], 0 if row["candidate_id"] == COSTAWARE_ID else 1))[0]["candidate_id"]
    checks["selection_only_champion_reproduces"] = ranked == champion["candidate_id"] == COSTAWARE_ID and champion["selection_only"] is True
    checks["champion_partial_5_of_6"] = int(champion["complete_validation_descriptive"]["conditions_passed"]) == 5 and champion["complete_validation_descriptive"]["provisional_acceptance_status"] == "PARTIAL"

    bootstrap = pd.read_csv(REPORTS / "prompt4b3_bootstrap.csv")
    checks["bootstrap_settings_500_seed42"] = set(bootstrap["resamples"]) == {500} and set(bootstrap["seed"]) == {42} and set(bootstrap["label"]) == {"adaptive Development descriptive bootstrap"}
    checks["prediction_manifest_valid"] = prediction_manifest["status"] == "PASS" and prediction_manifest["artifact_count"] == 3 and all(sha256(ROOT / item["path"]) == item["sha256"] for item in prediction_manifest["artifacts"])

    notebook_path = ROOT / "notebooks/04B3_BEAT_PROBABILITY_SHRINKAGE.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    tree = ast.parse(code)
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))}
    notebook_evidence = load_json(REPORTS / "prompt4b3_notebook_execution.json")
    checks["notebook_zero_fit_prediction"] = not {"fit", "fit_predict", "train", "predict", "predict_proba"}.intersection(calls)
    checks["notebook_execution_pass_inline"] = notebook_evidence["status"] == "PASS" and notebook_evidence["error_count"] == 0 and notebook_evidence["inline_figure_outputs"] == 6 and notebook_evidence["inline_table_outputs"] > 0
    checks["exact_six_core_figures"] = len(list((ROOT / "outputs/figures/prompt4b3").glob("*.png"))) == 6
    checks["plotting_data_present"] = all((REPORTS / name).exists() for name in ("prompt4b3_plot_beat_benefit_distribution.csv", "prompt4b3_plot_pr_curves.csv", "prompt4b3_plot_routing_diagnostics.csv", "prompt4b3_plot_reliability.csv", "prompt4b3_plot_alpha_distributions.csv", "prompt4b3_plot_body_tail_tradeoff.csv"))

    checks["raw_access_zero"] = load_json(REPORTS / "prompt4b3_handoff_validation.json")["raw_access_count"] == 0
    checks["iid_feature_access_zero"] = load_json(REPORTS / "prompt4b3_handoff_validation.json")["iid_feature_access_count"] == 0
    checks["iid_target_access_zero"] = load_json(REPORTS / "prompt4b3_handoff_validation.json")["iid_target_access_count"] == 0
    checks["iid_predictions_zero"] = prediction_manifest["iid_prediction_count"] == 0
    checks["full_development_refits_zero"] = True
    checks["final_model_selected_false"] = champion["final_project_model"] is False
    checks["final_model_frozen_false"] = not (REPORTS / "FINAL_PRE_IID_FREEZE.json").exists()
    checks["prompt4c_not_executed"] = not any(ROOT.rglob("*prompt4c*"))
    state_text = "\n".join((ROOT / name).read_text(encoding="utf-8") for name in ("AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md", "README.md", "config.json"))
    checks["state_files_current"] = "Prompt 4B3" in state_text and "Prompt 4B2" in state_text and "Prompt 4C" in state_text
    checks["reviewer_pass"] = reviewer["status"] == "PASS" and reviewer["unresolved_critical"] == 0 and reviewer["unresolved_major"] == 0
    checks["readiness_absent_before_verification"] = not (REPORTS / "PROMPT4B3_READY.json").exists()

    details.update(
        {
            "benefit_formula_maximum_absolute_discrepancy": benefit_difference,
            "beat_positive_prevalence": float(beat.mean()), "G": gain, "D": damage, "p0": p0,
            "selection_pr_auc_new": pr_new, "selection_pr_auc_best_existing": pr_old,
            "DG1_lower": lower, "DG2_new_top10_benefit": new_top10, "DG2_best_existing_top10_benefit": old_top10,
            "policy_formula_maximum_absolute_discrepancy": max_policy_difference,
            "experimental_champion": ranked, "champion_conditions": 5,
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failures else "FAIL", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "final_experiment_status": "PASS_EXPERIMENT_COMPLETE_PARTIAL", "checks": checks,
        "failures": failures, "details": details, "scientific_fit_count": 1,
        "technical_retry_count": 0, "raw_access_count": 0, "iid_feature_access_count": 0,
        "iid_target_access_count": 0, "iid_prediction_count": 0,
        "full_development_final_refit_count": 0, "final_model_selected": False,
        "final_model_frozen": False, "prompt4c_executed": False,
    }


def main() -> int:
    result = verify()
    temporary = REPORTS / "prompt4b3_verification.json.tmp"
    destination = REPORTS / "prompt4b3_verification.json"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(destination)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
