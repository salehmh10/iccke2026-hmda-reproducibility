"""Resumable execution and reporting for Prompt 4B3.

The only real model fit in this module is the one frozen Beat classifier.
All other candidate construction uses saved predictions and fixed formulas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from nbclient import NotebookClient
from sklearn.metrics import precision_recall_curve

try:
    from .deep_preprocessing import ordered_digest
    from .prompt4_metrics import compute_operational_tail_metrics, compute_regression_metrics, quantile_membership
    from .prompt4a_experiments import (
        _roles_from_design, atomic_csv, atomic_json, atomic_parquet, file_sha256,
        load_aligned_predictions, load_development, package_versions, regression_v2_root,
    )
    from .prompt4b3_beat import (
        BOOTSTRAP_RESAMPLES, CANDIDATE_ID, COSTAWARE_ID, FEATURE_COUNT, LINEAR_ID,
        RELIABILITY_BINS, ROUTED_FRACTIONS, BeatClassifierBundle, alpha_costaware,
        alpha_linear, beat_target, binary_score_metrics, classifier_config_from_anchor,
        classifier_metrics, cost_asymmetry, fixed_fraction_diagnostics, load_bundle,
        make_feature_frame, paired_pr_auc_bootstrap, policy_prediction, reliability_table,
        reproduce_benefit, resumable_model_status, save_bundle, select_policy,
        sha256_file, six_condition_evaluator, validate_feature_contract,
        validate_oof_membership, validate_read_path,
    )
    from .prompt4b2_benefit import BenefitPreprocessor
except ImportError:
    from deep_preprocessing import ordered_digest
    from prompt4_metrics import compute_operational_tail_metrics, compute_regression_metrics, quantile_membership
    from prompt4a_experiments import (
        _roles_from_design, atomic_csv, atomic_json, atomic_parquet, file_sha256,
        load_aligned_predictions, load_development, package_versions, regression_v2_root,
    )
    from prompt4b3_beat import (
        BOOTSTRAP_RESAMPLES, CANDIDATE_ID, COSTAWARE_ID, FEATURE_COUNT, LINEAR_ID,
        RELIABILITY_BINS, ROUTED_FRACTIONS, BeatClassifierBundle, alpha_costaware,
        alpha_linear, beat_target, binary_score_metrics, classifier_config_from_anchor,
        classifier_metrics, cost_asymmetry, fixed_fraction_diagnostics, load_bundle,
        make_feature_frame, paired_pr_auc_bootstrap, policy_prediction, reliability_table,
        reproduce_benefit, resumable_model_status, save_bundle, select_policy,
        sha256_file, six_condition_evaluator, validate_feature_contract,
        validate_oof_membership, validate_read_path,
    )
    from prompt4b2_benefit import BenefitPreprocessor


REPORTS = Path("outputs/reports")
MODELS = Path("outputs/models/prompt4b3/beat_classifier")
PREDICTIONS = Path("outputs/predictions/prompt4b3/validation")
FIGURES = Path("outputs/figures/prompt4b3")
TMP = Path("outputs/tmp/prompt4b3")
NOTEBOOK = Path("notebooks/04B3_BEAT_PROBABILITY_SHRINKAGE.ipynb")
DESIGN = REPORTS / "prompt4b3_design_freeze.json"
LEDGER = REPORTS / "prompt4b3_fit_ledger.json"
TARGET = "loan_amount_000s"
Q90_TRAIN = 438.0
CAP25 = 109.5

SCORE_PATHS = {
    "global": Path("outputs/predictions/prompt4a/validation/ens_boost_cat060.parquet"),
    "old_tail_gate": Path("outputs/predictions/prompt4b/validation/stage1_probabilities.parquet"),
    "meta_gate": Path("outputs/predictions/prompt4b/validation/stage2_meta_gate_probability.parquet"),
    "residual": Path("outputs/predictions/prompt4b/validation/stage3_residual_prediction.parquet"),
    "benefit_router": Path("outputs/predictions/prompt4b2/validation/benefit_b0.parquet"),
}
COMPARATOR_PATHS = {
    "ens_boost_cat060": SCORE_PATHS["global"],
    "stage3_residual_t75_a75": Path("outputs/predictions/prompt4b/validation/stage3_residual_t75_a75.parquet"),
    "nf_global2_oldraw_direct_cap25": Path("outputs/predictions/prompt4b2/validation/nf_global2_oldraw_direct_cap25.parquet"),
    "quantile_q60_meta_symmetric": Path("outputs/predictions/prompt4b2/validation/quantile_q60_meta_symmetric.parquet"),
    "benefit_b0": Path("outputs/predictions/prompt4b2/validation/benefit_b0.parquet"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def code_digest(root: Path) -> str:
    paths = [
        root / "src/prompt4b3_beat.py", root / "src/prompt4b3_experiments.py",
        root / "tests/test_prompt4b3_beat.py",
    ]
    evidence = {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True).encode("utf-8")).hexdigest()


def canonical_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def guarded_parquet(root: Path, relative: Path) -> pd.DataFrame:
    return pd.read_parquet(validate_read_path(root, relative))


def validation_state(root: Path) -> dict[str, Any]:
    train, validation, source = load_development(root)
    design4b = read_json(root / REPORTS / "prompt4b_frozen_design.json")
    aligned, alignment = load_aligned_predictions(root, validation)
    roles = _roles_from_design(aligned, design4b)
    aligned = aligned.copy()
    aligned["role"] = roles
    return {
        "train": train, "validation": validation, "source": source, "aligned": aligned,
        "roles": roles, "alignment": alignment, "design4b": design4b,
        "features": list(source["features"]),
    }


def load_scientific_inputs(root: Path, state: dict[str, Any]) -> dict[str, Any]:
    train = state["train"]
    aligned = state["aligned"]
    benefit_train = guarded_parquet(root, Path("outputs/predictions/prompt4b2/train/benefit_training_table.parquet"))
    residual_oof = guarded_parquet(root, Path("outputs/predictions/prompt4b2/train/residual_oof.parquet"))
    global_oof = guarded_parquet(root, Path("outputs/predictions/prompt4b/train/global_oof_prediction.parquet"))
    membership = validate_oof_membership(train["row_hash"], benefit_train, residual_oof)
    if not train["row_hash"].astype(str).equals(global_oof["row_hash"].astype(str)):
        raise RuntimeError("Global OOF row order differs from frozen Train membership.")
    if set(train["row_hash"].astype(str)) & set(aligned["row_hash"].astype(str)):
        raise RuntimeError("Train and Validation memberships overlap.")
    if not np.array_equal(global_oof["global_oof_prediction"].to_numpy(float), benefit_train["global_oof_prediction"].to_numpy(float)):
        raise RuntimeError("Saved Global OOF predictions disagree across frozen artifacts.")
    if not np.array_equal(residual_oof["residual_oof_prediction"].to_numpy(float), benefit_train["residual_oof_prediction"].to_numpy(float)):
        raise RuntimeError("Saved Residual OOF predictions disagree across frozen artifacts.")
    reproduced = reproduce_benefit(
        global_oof["y_true"], benefit_train["global_oof_prediction"],
        benefit_train["proposed_residual_correction"],
    )
    benefit_difference = float(np.max(np.abs(reproduced - benefit_train["benefit_oof"].to_numpy(float))))
    if benefit_difference != 0.0:
        raise RuntimeError("Frozen OOF Benefit formula did not reproduce exactly.")

    saved = {name: guarded_parquet(root, path) for name, path in SCORE_PATHS.items()}
    for name, frame in saved.items():
        if not aligned["row_hash"].astype(str).equals(frame["row_hash"].astype(str)):
            raise RuntimeError(f"Validation score source is misaligned: {name}")
    global_prediction = saved["global"]["y_pred"].to_numpy(float)
    proposal = np.clip(np.maximum(saved["residual"]["predicted_residual"].to_numpy(float), 0.0), 0.0, CAP25)
    validation_benefit = reproduce_benefit(aligned["y_true"], global_prediction, proposal)
    if float(np.max(np.abs(proposal - saved["benefit_router"]["proposed_residual_correction"].to_numpy(float)))) != 0.0:
        raise RuntimeError("Frozen Validation residual proposal did not reproduce exactly.")
    if float(np.max(np.abs(validation_benefit - saved["benefit_router"]["realized_benefit"].to_numpy(float)))) != 0.0:
        raise RuntimeError("Frozen Validation Benefit did not reproduce exactly.")
    scores = {
        "old_tail_gate": saved["old_tail_gate"]["p_raw"].to_numpy(float),
        "meta_gate": saved["meta_gate"]["p_meta_gate"].to_numpy(float),
        "benefit_router": saved["benefit_router"]["predicted_benefit"].to_numpy(float),
    }
    return {
        "benefit_train": benefit_train, "residual_oof": residual_oof, "global_oof": global_oof,
        "membership": membership, "benefit_difference": benefit_difference,
        "global_prediction": global_prediction, "proposal": proposal,
        "validation_benefit": validation_benefit, "validation_beat": beat_target(validation_benefit),
        "scores": scores,
    }


def prior_hash_inventory(root: Path) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for manifest_name in ("prompt4b2_prediction_manifest.json", "prompt4b2_model_manifest.json"):
        manifest = read_json(root / REPORTS / manifest_name)
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"Prior manifest is not PASS: {manifest_name}")
        for item in manifest["artifacts"]:
            path = root / item["path"]
            actual = file_sha256(path)
            if actual != item["sha256"]:
                raise RuntimeError(f"Prior immutable artifact hash changed: {item['path']}")
            artifacts.append({"path": item["path"], "sha256": actual})
    return artifacts


def zero_fit_baselines(state: dict[str, Any], inputs: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection = state["roles"] == "selection"
    labels = inputs["validation_beat"][selection]
    benefit = inputs["validation_benefit"][selection]
    rows = []
    summary: dict[str, Any] = {}
    for score_id, all_score in inputs["scores"].items():
        score = all_score[selection]
        metrics = binary_score_metrics(labels, score)
        routed = fixed_fraction_diagnostics(labels, benefit, score)
        for item in routed.to_dict("records"):
            rows.append({"score_id": score_id, **metrics, **item})
        top10 = routed.loc[np.isclose(routed["routed_fraction"], 0.10)].iloc[0]
        summary[score_id] = {**metrics, "top_10_mean_realized_benefit": float(top10["mean_realized_benefit"])}
    table = pd.DataFrame(rows)
    best = sorted(summary, key=lambda name: (-summary[name]["pr_auc_average_precision"], -summary[name]["roc_auc"], name))[0]
    best_top10 = max(summary, key=lambda name: (summary[name]["top_10_mean_realized_benefit"], name))
    return table, {
        "evaluation_scope": "selection", "rows": int(selection.sum()),
        "base_beat_prevalence": float(labels.mean()), "scores": summary,
        "best_existing_baseline": best,
        "best_existing_selection_rule": "highest PR-AUC, then ROC-AUC, then score ID",
        "best_top10_mean_benefit_baseline": best_top10,
        "best_top10_mean_realized_benefit": float(summary[best_top10]["top_10_mean_realized_benefit"]),
    }


def prepare_design(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    if (workspace / DESIGN).exists():
        design = read_json(workspace / DESIGN)
        core = {key: value for key, value in design.items() if key != "design_digest"}
        if design.get("status") != "FROZEN" or canonical_digest(core) != design.get("design_digest"):
            raise RuntimeError("Existing Prompt 4B3 design freeze is invalid.")
        if design.get("code_digest") != code_digest(workspace):
            raise RuntimeError("Existing Prompt 4B3 design does not match frozen implementation.")
        return design

    for name in ("PROMPT4B2_READY.json", "prompt4b2_verification.json", "PROMPT4B_READY.json", "prompt4b_verification.json", "DATA_READY.json", "final_verification.json"):
        if read_json(workspace / REPORTS / name).get("status") != "PASS":
            raise RuntimeError(f"Required prerequisite is not PASS: {name}")
    if (workspace / REPORTS / "FINAL_PRE_IID_FREEZE.json").exists():
        raise RuntimeError("FINAL_PRE_IID_FREEZE.json must remain absent.")
    if list(workspace.rglob("*prompt4c*")):
        raise RuntimeError("Prompt 4C evidence exists; Prompt 4B3 cannot start.")

    state = validation_state(workspace)
    inputs = load_scientific_inputs(workspace, state)
    prior_hashes = prior_hash_inventory(workspace)
    feature_roles = read_json(workspace / REPORTS / "feature_roles.json")
    base_features = feature_roles["contracts"]["main_without_sensitive_without_lender"]
    router_manifest = read_json(workspace / "outputs/models/prompt4b2/benefit_router/manifest.json")
    feature_contract = validate_feature_contract(base_features, router_manifest["metadata"]["feature_contract"])
    classifier_config = classifier_config_from_anchor(router_manifest["metadata"]["model_configuration"])
    asymmetry = cost_asymmetry(inputs["benefit_train"]["benefit_oof"])
    baseline_table, baseline_summary = zero_fit_baselines(state, inputs)
    atomic_csv(workspace, REPORTS / "prompt4b3_zero_fit_beat_baselines.csv", baseline_table)
    atomic_json(workspace, REPORTS / "prompt4b3_zero_fit_beat_baselines_summary.json", baseline_summary)

    benefit_values = inputs["benefit_train"]["benefit_oof"].to_numpy(float)
    target_audit = {
        "status": "PASS", "created_at_utc": utc_now(), "rows": len(benefit_values),
        "positive_count": int((benefit_values > 0.0).sum()),
        "negative_or_zero_count": int((benefit_values <= 0.0).sum()),
        "zero_count": int((benefit_values == 0.0).sum()),
        "positive_prevalence": float((benefit_values > 0.0).mean()),
        "row_hash_digest": ordered_digest(inputs["benefit_train"]["row_hash"]),
        "benefit_formula": "abs(y-global_oof)-abs(y-(global_oof+residual_proposal_oof))",
        "benefit_formula_maximum_absolute_discrepancy": inputs["benefit_difference"],
        "strict_positive_target": True, "zero_benefit_label": 0,
    }
    atomic_json(workspace, REPORTS / "prompt4b3_beat_target_audit.json", target_audit)

    state_hashes = {name: file_sha256(workspace / name) for name in ("AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md", "README.md", "config.json")}
    core = {
        "status": "FROZEN", "created_at_utc": utc_now(), "prompt": "Prompt 4B3",
        "development_stage": "adaptive Development evidence",
        "code_digest": code_digest(workspace), "state_file_hashes_at_freeze": state_hashes,
        "source": state["source"],
        "memberships": {
            "development_rows": 500_000, "train_rows": 400_000, "validation_rows": 100_000,
            "selection_rows": 70_000, "audit_rows": 30_000,
            "train_ordered_digest": state["source"]["train_row_hash_digest"],
            "validation_ordered_digest": state["source"]["validation_row_hash_digest"],
            "selection_ordered_digest": state["design4b"]["validation_split"]["selection_row_hash_digest"],
            "audit_ordered_digest": state["design4b"]["validation_split"]["audit_row_hash_digest"],
            "reload_sample_rows": 1_000,
            "reload_sample_ordered_digest": ordered_digest(state["aligned"].iloc[:1000]["row_hash"]),
        },
        "prior_immutable_artifacts": prior_hashes,
        "feature_contract_name": "main_without_sensitive_without_lender_plus_frozen_global_and_residual_proposal",
        "base_feature_count": 35, "inference_feature_count": 37,
        "inference_features": feature_contract,
        "beat_formula": "beat = 1 if benefit > 0 else 0; benefit == 0 maps to 0",
        "benefit_formula": "abs(y-global_oof)-abs(y-(global_oof+residual_proposal_oof))",
        "classifier": {
            "candidate_id": CANDIDATE_ID, "model_class": "CatBoostClassifier",
            "configuration": classifier_config, "prediction_output": "class-1 probability-like score",
            "no_class_weights": True, "no_auto_class_weights": True, "no_early_stopping": True,
            "no_use_best_model": True,
        },
        "scientific_fit_budget": {"candidates": 1, "fits": 1, "maximum_exact_candidate_technical_retries": 1},
        "zero_fit_baselines": {
            "score_ids": ["old_tail_gate", "meta_gate", "benefit_router"],
            "diagnostic_fractions": list(ROUTED_FRACTIONS),
            "table_sha256": file_sha256(workspace / REPORTS / "prompt4b3_zero_fit_beat_baselines.csv"),
            **baseline_summary,
        },
        "cost_asymmetry": {**asymmetry, "source": "400000-row Train OOF Benefit only"},
        "shrinkage_policies": {
            LINEAR_ID: "alpha=clip(p_beat,0,1)",
            COSTAWARE_ID: "alpha=clip(max(0,(p_beat-p0)/(1-p0)),0,1)",
        },
        "diagnostic_gate": {
            "DG1": "paired 500-resample seed-42 Selection bootstrap lower 95% bound of PR_AUC_new - PR_AUC_best_existing > 0",
            "DG2": "Selection top-10% mean realized Benefit new > best old/meta/Benefit-Router top-10% mean realized Benefit",
            "failure_status": "PASS_DIAGNOSTIC_STOP",
        },
        "candidate_selection_rule": [
            "higher six-condition pass count", "lower Selection MAE", "lower Selection Top-decile MAE",
            "if tied choose prompt4b3__beat_costaware",
        ],
        "six_condition_rubric": {
            "C1": "overall MAE improves", "C2": "Top-decile MAE improves by at least 3%",
            "C3": "Bottom-90 MAE worsens by no more than 0.25%", "C4": "RMSE worsens by no more than 0.25%",
            "C5": "Top-decile signed error moves closer to zero", "C6": "Top-decile underprediction rate decreases",
            "PASS": "6/6", "PARTIAL": "1-5/6", "FAIL": "0/6",
        },
        "bootstrap": {"resamples": 500, "seed": 42, "paired_rows": True, "label": "adaptive Development descriptive bootstrap"},
        "notebook": {"path": NOTEBOOK.as_posix(), "artifact_only": True, "maximum_attempts": 2, "fit_count": 0},
        "required_reports": [
            "prompt4b3_handoff_validation.json", "prompt4b3_zero_fit_beat_baselines.csv",
            "prompt4b3_beat_target_audit.json", "prompt4b3_classifier_diagnostics.json",
            "prompt4b3_classifier_routing_rates.csv", "prompt4b3_diagnostic_gate.json",
            "prompt4b3_cross_stage_comparison.csv", "prompt4b3_fit_ledger.json",
            "prompt4b3_model_manifest.json", "prompt4b3_prediction_manifest.json",
            "prompt4b3_runtime.json", "prompt4b3_reviewer.json", "prompt4b3_verification.json",
        ],
        "stop_conditions": [
            "invalid Prompt 4B2 prerequisite", "missing or corrupt prior scientific artifact",
            "Development identity change", "Raw or IID access required", "37-feature contract failure",
            "OOF Benefit reproduction failure", "ambiguous structural classifier configuration",
            "leakage", "more than one scientific Candidate required", "post-result design change required",
        ],
        "prohibitions": {
            "raw_access": True, "iid_feature_access": True, "iid_target_access": True,
            "iid_prediction": True, "new_global_fit": True, "new_specialist_fit": True,
            "threshold_search": True, "feature_search": True, "final_500k_refit": True,
            "final_model_selection": True, "final_model_freeze": True, "prompt4c": True,
        },
    }
    design = {**core, "design_digest": canonical_digest(core)}
    atomic_json(workspace, DESIGN, design)
    reloaded = read_json(workspace / DESIGN)
    if reloaded != design or canonical_digest({key: value for key, value in reloaded.items() if key != "design_digest"}) != reloaded["design_digest"]:
        raise RuntimeError("Prompt 4B3 design freeze reload failed.")

    handoff = {
        "status": "PASS", "created_at_utc": utc_now(), "prompt4b2_readiness": "PASS",
        "prompt4b2_verification": "PASS", "prompt4b_readiness": "PASS",
        "prior_manifest_hashes_valid": True, "development_identity_unchanged": True,
        "development_rows": 500_000, "train_rows": 400_000, "validation_rows": 100_000,
        "selection_rows": 70_000, "audit_rows": 30_000,
        "oof_benefit_rows": 400_000, "oof_residual_rows": 400_000,
        "benefit_formula_maximum_absolute_discrepancy": 0.0,
        "residual_oof_self_fit_rows": 0, "feature_count": 37,
        "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0,
        "iid_prediction_count": 0, "final_pre_iid_freeze_absent": True, "prompt4c_not_executed": True,
    }
    atomic_json(workspace, REPORTS / "prompt4b3_handoff_validation.json", handoff)
    ledger = {
        "status": "READY_FOR_SINGLE_FIT", "created_at_utc": utc_now(),
        "scientific_candidates": [CANDIDATE_ID], "scientific_candidate_count": 1,
        "scientific_fit_count": 0, "physical_attempt_count": 0, "technical_retry_count": 0,
        "synthetic_unit_model_construction_count": 1,
        "synthetic_unit_model_note": "One fake probability model was serialized; no CatBoost fit was used in tests.",
        "reporting_only_work_count": 1, "physical_attempts": [],
    }
    atomic_json(workspace, LEDGER, ledger)
    return design


def load_design(root: Path) -> dict[str, Any]:
    design = read_json(root / DESIGN)
    core = {key: value for key, value in design.items() if key != "design_digest"}
    if design.get("status") != "FROZEN" or canonical_digest(core) != design.get("design_digest"):
        raise RuntimeError("Prompt 4B3 design freeze is invalid.")
    if design.get("code_digest") != code_digest(root):
        raise RuntimeError("Prompt 4B3 implementation changed after design freeze.")
    return design


def fit_classifier(root: str | Path | None = None) -> dict[str, Any]:
    from catboost import CatBoostClassifier

    workspace = Path(root or regression_v2_root()).resolve()
    design = load_design(workspace)
    state = validation_state(workspace)
    inputs = load_scientific_inputs(workspace, state)
    ledger = read_json(workspace / LEDGER)
    model_directory = workspace / MODELS
    resume = resumable_model_status(model_directory, ledger)
    if resume == "INSPECT_ACTIVE_ATTEMPT":
        raise RuntimeError("An incomplete Prompt 4B3 physical attempt requires process inspection.")
    if resume == "BLOCKED_MISSING_COMPLETED_MODEL":
        raise RuntimeError("A completed fit is recorded but its model artifact is missing.")

    features = design["inference_features"]
    X_train = make_feature_frame(
        state["train"], design["source"]["features"],
        inputs["benefit_train"]["global_oof_prediction"],
        inputs["benefit_train"]["proposed_residual_correction"],
    )
    labels = beat_target(inputs["benefit_train"]["benefit_oof"])
    if list(X_train.columns) != features or len(X_train) != 400_000 or set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("Prompt 4B3 training contract failed before fit.")
    numeric_columns = X_train.select_dtypes(include=[np.number, "bool"]).columns
    if not np.isfinite(X_train[numeric_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Prompt 4B3 numeric inference features are not finite.")

    if resume == "REUSE_VALID_MODEL":
        bundle = load_bundle(model_directory)
        fit_seconds = float(bundle.metadata["fit_runtime_seconds"])
    else:
        attempts = list(ledger.get("physical_attempts", []))
        if len(attempts) >= 2:
            raise RuntimeError("Prompt 4B3 exact-Candidate technical retry budget is exhausted.")
        attempt_number = len(attempts) + 1
        attempt = {
            "candidate_id": CANDIDATE_ID, "attempt_number": attempt_number,
            "technical_retry": attempt_number == 2, "status": "IN_PROGRESS",
            "process_id": os.getpid(), "started_at_utc": utc_now(),
            "design_digest": design["design_digest"],
        }
        attempts.append(attempt)
        ledger.update(
            {
                "status": "FIT_IN_PROGRESS", "physical_attempts": attempts,
                "physical_attempt_count": len(attempts),
                "technical_retry_count": int(sum(bool(item["technical_retry"]) for item in attempts)),
            }
        )
        atomic_json(workspace, LEDGER, ledger)
        started = time.perf_counter()
        try:
            preprocessor = BenefitPreprocessor(design["source"]["features"]).fit(X_train)
            model = CatBoostClassifier(**design["classifier"]["configuration"])
            model.fit(
                preprocessor.transform(X_train), labels,
                cat_features=preprocessor.cat_feature_indices_, verbose=False,
            )
            fit_seconds = float(time.perf_counter() - started)
            metadata = {
                "candidate_id": CANDIDATE_ID, "model_class": "CatBoostClassifier",
                "model_configuration": design["classifier"]["configuration"],
                "feature_contract": features, "training_rows": 400_000,
                "training_row_digest": design["memberships"]["train_ordered_digest"],
                "beat_positive_count": int(labels.sum()), "beat_prevalence": float(labels.mean()),
                "design_digest": design["design_digest"], "development_source_sha256": design["source"]["sha256"],
                "fit_runtime_seconds": fit_seconds, "physical_attempt_number": attempt_number,
                "random_seed": 42, "thread_count": 4, "device": "CPU",
                "class_weights": None, "auto_class_weights": None, "early_stopping": False,
                "use_best_model": False, "package_versions": package_versions(),
            }
            bundle = BeatClassifierBundle(preprocessor, model, features, metadata)
            manifest = save_bundle(bundle, model_directory)
            attempt.update(
                {
                    "status": "COMPLETE", "completed_at_utc": utc_now(),
                    "fit_runtime_seconds": fit_seconds,
                    "bundle_sha256": manifest["artifact_sha256"], "native_sha256": manifest["native_sha256"],
                }
            )
            ledger.update(
                {
                    "status": "FIT_COMPLETE", "physical_attempts": attempts,
                    "scientific_fit_count": 1, "physical_attempt_count": len(attempts),
                    "technical_retry_count": int(sum(bool(item["technical_retry"]) for item in attempts)),
                }
            )
            atomic_json(workspace, LEDGER, ledger)
        except Exception as error:
            attempt.update(
                {
                    "status": "TECHNICAL_FAILURE", "completed_at_utc": utc_now(),
                    "error_type": type(error).__name__, "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            ledger.update({"status": "TECHNICAL_FAILURE", "physical_attempts": attempts})
            atomic_json(workspace, LEDGER, ledger)
            atomic_json(workspace, TMP / f"attempt_{attempt_number}_technical_failure.json", attempt)
            raise

    global_prediction = inputs["global_prediction"]
    proposal = inputs["proposal"]
    X_validation = make_feature_frame(state["validation"], design["source"]["features"], global_prediction, proposal)
    source_before = X_validation.copy(deep=True)
    probability = bundle.predict_probability(X_validation)
    if not X_validation.equals(source_before):
        raise RuntimeError("Beat probability prediction changed the source frame.")
    prediction_frame = pd.DataFrame(
        {
            "row_hash": state["aligned"]["row_hash"].astype(str),
            "y_true": state["aligned"]["y_true"].to_numpy(float),
            "p_beat": probability, "benefit_validation": inputs["validation_benefit"],
            "beat_validation": inputs["validation_beat"], "global_prediction": global_prediction,
            "proposed_residual_correction": proposal,
            "selection_or_audit_role": state["roles"], "candidate_id": CANDIDATE_ID,
        }
    )
    atomic_parquet(workspace, PREDICTIONS / "beat_classifier_probability.parquet", prediction_frame)
    worker_result = workspace / TMP / "clean_reload_worker.json"
    command = [sys.executable, str(workspace / "src/prompt4b3_experiments.py"), "reload-worker", "--root", str(workspace)]
    completed = subprocess.run(command, cwd=workspace, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0:
        raise RuntimeError(f"Clean-process reload worker failed: {completed.stderr}")
    reload_evidence = read_json(worker_result)
    if reload_evidence.get("status") != "PASS":
        raise RuntimeError("Clean-process Beat classifier reload did not pass.")

    manifest = read_json(model_directory / "manifest.json")
    model_report = {
        "status": "PASS", "created_at_utc": utc_now(), "scientific_model_count": 1,
        "candidate_id": CANDIDATE_ID, "path": MODELS.as_posix(),
        "bundle_path": (MODELS / manifest["artifact"]).as_posix(),
        "bundle_sha256": manifest["artifact_sha256"],
        "native_path": (MODELS / manifest["native_artifact"]).as_posix(),
        "native_sha256": manifest["native_sha256"], "feature_count": 37,
        "configuration": design["classifier"]["configuration"], "clean_process_reload": reload_evidence,
    }
    atomic_json(workspace, REPORTS / "prompt4b3_model_manifest.json", model_report)
    atomic_json(
        workspace, REPORTS / "prompt4b3_runtime.json",
        {
            "status": "FIT_COMPLETE", "created_at_utc": utc_now(), "fit_runtime_seconds": fit_seconds,
            "scientific_fit_count": 1, "technical_retry_count": read_json(workspace / LEDGER)["technical_retry_count"],
            "reporting_only_continuation": resume == "REUSE_VALID_MODEL",
        },
    )
    return model_report


def reload_worker(root: str | Path) -> dict[str, Any]:
    workspace = Path(root).resolve()
    design = read_json(workspace / DESIGN)
    state = validation_state(workspace)
    inputs = load_scientific_inputs(workspace, state)
    reference = guarded_parquet(workspace, PREDICTIONS / "beat_classifier_probability.parquet")
    sample = slice(0, 1000)
    X = make_feature_frame(
        state["validation"].iloc[sample], design["source"]["features"],
        inputs["global_prediction"][sample], inputs["proposal"][sample],
    )
    before = X.copy(deep=True)
    probability = load_bundle(workspace / MODELS).predict_probability(X)
    difference = float(np.max(np.abs(probability - reference.iloc[sample]["p_beat"].to_numpy(float))))
    result = {
        "status": "PASS" if difference <= 1e-12 else "FAIL", "rows": 1000,
        "sample_row_digest": ordered_digest(reference.iloc[sample]["row_hash"]),
        "raw_named_dataframe_accepted": True, "exact_feature_order_enforced": list(X.columns) == design["inference_features"],
        "finite_probabilities": bool(np.isfinite(probability).all()),
        "probabilities_bounded_0_1": bool(np.all((probability >= 0.0) & (probability <= 1.0))),
        "row_order_unchanged": ordered_digest(reference.iloc[sample]["row_hash"]) == design["memberships"]["reload_sample_ordered_digest"],
        "source_frame_unchanged": X.equals(before), "maximum_absolute_probability_difference": difference,
        "tolerance": 1e-12,
    }
    atomic_json(workspace, TMP / "clean_reload_worker.json", result)
    return result


def metric_scopes(state: dict[str, Any], prediction: np.ndarray, candidate_id: str) -> list[dict[str, Any]]:
    y = state["aligned"]["y_true"].to_numpy(float)
    rows = []
    for scope, mask in (
        ("selection", state["roles"] == "selection"),
        ("audit_descriptive", state["roles"] == "audit"),
        ("complete_validation_descriptive", np.ones(len(y), dtype=bool)),
    ):
        rows.append(
            {
                "candidate_id": candidate_id, "scope": scope,
                **compute_regression_metrics(y[mask], prediction[mask]),
                **compute_operational_tail_metrics(y[mask], prediction[mask], Q90_TRAIN),
            }
        )
    return rows


def descriptive_bootstrap(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray, fixed_tail: np.ndarray, reference_id: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    overall_pointwise = np.abs(candidate - y) - np.abs(reference - y)
    tail_pointwise = overall_pointwise[fixed_tail]
    overall = np.empty(500)
    tail = np.empty(500)
    for index in range(500):
        sample = rng.integers(0, len(y), size=len(y))
        tail_sample = rng.integers(0, len(tail_pointwise), size=len(tail_pointwise))
        overall[index] = overall_pointwise[sample].mean()
        tail[index] = tail_pointwise[tail_sample].mean()
    rows = []
    for metric, pointwise, distribution in (
        ("overall_mae_difference", overall_pointwise, overall),
        ("fixed_top_decile_mae_difference", tail_pointwise, tail),
    ):
        rows.append(
            {
                "reference_id": reference_id, "metric": metric, "rows": int(len(pointwise)),
                "difference_champion_minus_reference": float(pointwise.mean()),
                "percentile_2_5": float(np.quantile(distribution, 0.025)),
                "median": float(np.median(distribution)), "percentile_97_5": float(np.quantile(distribution, 0.975)),
                "resamples": 500, "seed": 42, "label": "adaptive Development descriptive bootstrap",
            }
        )
    return rows


def evaluate(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    design = load_design(workspace)
    state = validation_state(workspace)
    inputs = load_scientific_inputs(workspace, state)
    probability_frame = guarded_parquet(workspace, PREDICTIONS / "beat_classifier_probability.parquet")
    if not state["aligned"]["row_hash"].astype(str).equals(probability_frame["row_hash"].astype(str)):
        raise RuntimeError("Beat classifier Validation probabilities are misaligned.")
    probability = probability_frame["p_beat"].to_numpy(float)
    labels = inputs["validation_beat"]
    benefit = inputs["validation_benefit"]

    diagnostics: dict[str, Any] = {
        "status": "PASS", "created_at_utc": utc_now(), "candidate_id": CANDIDATE_ID,
        "score_interpretation": "Logloss Beat probability-like score; not assumed perfectly calibrated",
        "scopes": {}, "reliability_bins": 10,
    }
    routing_rows = []
    reliability_rows = []
    score_sets = {**inputs["scores"], "beat_classifier": probability}
    for scope, mask in (
        ("selection", state["roles"] == "selection"),
        ("audit_descriptive", state["roles"] == "audit"),
        ("complete_validation_descriptive", np.ones(len(labels), dtype=bool)),
    ):
        diagnostics["scopes"][scope] = classifier_metrics(labels[mask], probability[mask])
        reliability = reliability_table(labels[mask], probability[mask])
        reliability.insert(0, "scope", scope)
        reliability_rows.extend(reliability.to_dict("records"))
        for score_id, score in score_sets.items():
            routed = fixed_fraction_diagnostics(labels[mask], benefit[mask], score[mask])
            routed.insert(0, "score_id", score_id)
            routed.insert(0, "scope", scope)
            routing_rows.extend(routed.to_dict("records"))
    routing = pd.DataFrame(routing_rows)
    reliability = pd.DataFrame(reliability_rows)
    atomic_csv(workspace, REPORTS / "prompt4b3_classifier_routing_rates.csv", routing)
    atomic_csv(workspace, REPORTS / "prompt4b3_classifier_reliability.csv", reliability)

    selection = state["roles"] == "selection"
    best_existing = design["zero_fit_baselines"]["best_existing_baseline"]
    bootstrap = paired_pr_auc_bootstrap(labels[selection], probability[selection], inputs["scores"][best_existing][selection])
    new_top10 = fixed_fraction_diagnostics(labels[selection], benefit[selection], probability[selection])
    new_top10_value = float(new_top10.loc[np.isclose(new_top10["routed_fraction"], 0.10), "mean_realized_benefit"].iloc[0])
    best_old_top10 = float(design["zero_fit_baselines"]["best_top10_mean_realized_benefit"])
    dg1 = bool(bootstrap["percentile_2_5"] > 0.0)
    dg2 = bool(new_top10_value > best_old_top10)
    gate = {
        "status": "PASS" if dg1 and dg2 else "FAIL", "created_at_utc": utc_now(),
        "best_existing_baseline": best_existing, "new_classifier": CANDIDATE_ID,
        "DG1": {"pass": dg1, **bootstrap, "strict_rule": "percentile_2_5 > 0"},
        "DG2": {
            "pass": dg2, "new_top_10_mean_realized_benefit": new_top10_value,
            "best_existing_top_10_mean_realized_benefit": best_old_top10,
            "best_existing_top_10_score": design["zero_fit_baselines"]["best_top10_mean_benefit_baseline"],
            "strict_rule": "new > best existing",
        },
        "policy_evaluation_authorized": bool(dg1 and dg2),
        "failure_final_status": "PASS_DIAGNOSTIC_STOP",
    }
    diagnostics["diagnostic_gate"] = gate
    atomic_json(workspace, REPORTS / "prompt4b3_classifier_diagnostics.json", diagnostics)
    atomic_json(workspace, REPORTS / "prompt4b3_diagnostic_gate.json", gate)

    policy_rows: list[dict[str, Any]] = []
    six_rows: list[dict[str, Any]] = []
    champion = None
    bootstrap_rows: list[dict[str, Any]] = []
    if gate["policy_evaluation_authorized"]:
        p0 = float(design["cost_asymmetry"]["p0"])
        alpha_map = {LINEAR_ID: alpha_linear(probability), COSTAWARE_ID: alpha_costaware(probability, p0)}
        global_rows = {row["scope"]: row for row in metric_scopes(state, inputs["global_prediction"], "ens_boost_cat060")}
        policy_predictions: dict[str, np.ndarray] = {}
        for candidate_id, alpha in alpha_map.items():
            prediction = policy_prediction(inputs["global_prediction"], inputs["proposal"], alpha)
            policy_predictions[candidate_id] = prediction
            rows = metric_scopes(state, prediction, candidate_id)
            for row in rows:
                acceptance = six_condition_evaluator(row, global_rows[row["scope"]])
                row.update(acceptance)
                policy_rows.append(row)
                six_rows.append({"candidate_id": candidate_id, "scope": row["scope"], **acceptance})
            atomic_parquet(
                workspace, PREDICTIONS / f"{candidate_id}.parquet",
                pd.DataFrame(
                    {
                        "row_hash": state["aligned"]["row_hash"].astype(str), "y_true": state["aligned"]["y_true"].to_numpy(float),
                        "y_pred": prediction, "p_beat": probability, "alpha": alpha,
                        "proposed_residual_correction": inputs["proposal"], "candidate_id": candidate_id,
                        "selection_or_audit_role": state["roles"],
                    }
                ),
            )
        policy_table = pd.DataFrame(policy_rows)
        six_table = pd.DataFrame(six_rows)
        atomic_csv(workspace, REPORTS / "prompt4b3_policy_results.csv", policy_table)
        atomic_csv(workspace, REPORTS / "prompt4b3_six_condition_results.csv", six_table)
        selection_rows = policy_table.loc[policy_table["scope"].eq("selection")].to_dict("records")
        champion = select_policy(selection_rows)
        champion_row = policy_table.loc[(policy_table["candidate_id"].eq(champion)) & (policy_table["scope"].eq("complete_validation_descriptive"))].iloc[0].to_dict()
        atomic_json(
            workspace, REPORTS / "prompt4b3_experimental_champion.json",
            {
                "status": "FROZEN_EXPERIMENTAL_DEVELOPMENT_REFERENCE", "created_at_utc": utc_now(),
                "candidate_id": champion, "selection_only": True, "complete_validation_descriptive": champion_row,
                "final_project_model": False,
            },
        )
        y = state["aligned"]["y_true"].to_numpy(float)
        fixed_tail = quantile_membership(y, 0.90)
        comparator_predictions = {
            candidate_id: guarded_parquet(workspace, path)["y_pred"].to_numpy(float)
            for candidate_id, path in {
                "ens_boost_cat060": COMPARATOR_PATHS["ens_boost_cat060"],
                "stage3_residual_t75_a75": COMPARATOR_PATHS["stage3_residual_t75_a75"],
                "nf_global2_oldraw_direct_cap25": COMPARATOR_PATHS["nf_global2_oldraw_direct_cap25"],
            }.items()
        }
        for reference_id, reference_prediction in comparator_predictions.items():
            bootstrap_rows.extend(descriptive_bootstrap(y, policy_predictions[champion], reference_prediction, fixed_tail, reference_id))
        atomic_csv(workspace, REPORTS / "prompt4b3_bootstrap.csv", pd.DataFrame(bootstrap_rows))

    build_cross_stage(workspace, state, policy_rows)
    build_figures(workspace, state, inputs, probability, routing, reliability, bool(gate["policy_evaluation_authorized"]))
    build_manifests(workspace)
    return {
        "status": "POLICY_EVALUATION_COMPLETE" if gate["policy_evaluation_authorized"] else "PASS_DIAGNOSTIC_STOP",
        "gate": gate, "champion": champion,
    }


def build_cross_stage(root: Path, state: dict[str, Any], policy_rows: list[dict[str, Any]]) -> pd.DataFrame:
    prior = pd.read_csv(validate_read_path(root, REPORTS / "prompt4b2_cross_block_comparison.csv"))
    wanted = ["ens_boost_cat060", "stage3_residual_t75_a75", "nf_global2_oldraw_direct_cap25", "quantile_q60_meta_symmetric", "benefit_b0"]
    prior = prior.loc[prior["candidate_id"].isin(wanted)].copy()
    rows = []
    for item in prior.to_dict("records"):
        rows.append(
            {
                "reference": item["reference"], "candidate_id": item["candidate_id"],
                "selection_mae": item["selection_mae"], "audit_mae": item["audit_mae"],
                "complete_validation_mae": item["complete_validation_mae"], "rmse": item["rmse"],
                "bottom_90_mae": item["bottom_90_mae"], "top_decile_mae": item["top_decile_mae"],
                "top_five_percent_mae": item["top_five_percent_mae"],
                "top_decile_signed_error": item["top_decile_signed_error"],
                "top_decile_underprediction_rate": item["top_decile_underprediction_rate"],
                "conditions_passed": item["conditions_passed"], "status": item["provisional_status"],
            }
        )
    if policy_rows:
        table = pd.DataFrame(policy_rows)
        for candidate_id in (LINEAR_ID, COSTAWARE_ID):
            by_scope = {row["scope"]: row for row in table.loc[table["candidate_id"].eq(candidate_id)].to_dict("records")}
            complete = by_scope["complete_validation_descriptive"]
            rows.append(
                {
                    "reference": "Prompt 4B3 policy", "candidate_id": candidate_id,
                    "selection_mae": by_scope["selection"]["mae"], "audit_mae": by_scope["audit_descriptive"]["mae"],
                    "complete_validation_mae": complete["mae"], "rmse": complete["rmse"],
                    "bottom_90_mae": complete["bottom_90_mae"], "top_decile_mae": complete["top_decile_mae"],
                    "top_five_percent_mae": complete["top_five_percent_mae"],
                    "top_decile_signed_error": complete["top_decile_signed_error"],
                    "top_decile_underprediction_rate": complete["top_decile_underprediction_rate"],
                    "conditions_passed": complete["conditions_passed"], "status": complete["provisional_acceptance_status"],
                }
            )
    result = pd.DataFrame(rows)
    atomic_csv(root, REPORTS / "prompt4b3_cross_stage_comparison.csv", result)
    return result


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def build_figures(root: Path, state: dict[str, Any], inputs: dict[str, Any], probability: np.ndarray, routing: pd.DataFrame, reliability: pd.DataFrame, policies: bool) -> None:
    figure_root = root / FIGURES
    figure_root.mkdir(parents=True, exist_ok=True)
    selection = state["roles"] == "selection"
    benefit = inputs["validation_benefit"][selection]
    labels = inputs["validation_beat"][selection]
    counts, edges = np.histogram(benefit, bins=60)
    histogram = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})
    atomic_csv(root, REPORTS / "prompt4b3_plot_beat_benefit_distribution.csv", histogram)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(["Beat=0", "Beat=1"], [int((labels == 0).sum()), int((labels == 1).sum())], color=["#9aa0a6", "#1f77b4"])
    axes[0].set_title("Selection Beat prevalence")
    axes[0].set_ylabel("Rows")
    axes[1].hist(benefit, bins=60, color="#4c78a8")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Realized Benefit distribution")
    axes[1].set_xlabel("Benefit")
    save_figure(figure_root / "01_beat_prevalence_benefit_distribution.png")

    curve_rows = []
    for score_id, values in {**inputs["scores"], "beat_classifier": probability}.items():
        precision, recall, thresholds = precision_recall_curve(labels, values[selection])
        for index in range(len(precision)):
            curve_rows.append(
                {
                    "score_id": score_id, "precision": precision[index], "recall": recall[index],
                    "threshold": thresholds[index] if index < len(thresholds) else np.nan,
                }
            )
    curves = pd.DataFrame(curve_rows)
    atomic_csv(root, REPORTS / "prompt4b3_plot_pr_curves.csv", curves)
    plt.figure(figsize=(7, 5))
    for score_id, part in curves.groupby("score_id"):
        plt.plot(part["recall"], part["precision"], label=score_id)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Beat precision-recall curves (Selection)")
    plt.legend()
    save_figure(figure_root / "02_beat_pr_curves.png")

    selection_routing = routing.loc[routing["scope"].eq("selection")].copy()
    atomic_csv(root, REPORTS / "prompt4b3_plot_routing_diagnostics.csv", selection_routing)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for score_id, part in selection_routing.groupby("score_id"):
        axes[0].plot(part["routed_fraction"] * 100, part["precision"], marker="o", label=score_id)
        axes[1].plot(part["routed_fraction"] * 100, part["mean_realized_benefit"], marker="o", label=score_id)
    axes[0].set_title("Precision at fixed routed fractions")
    axes[0].set_xlabel("Routed rows (%)")
    axes[0].set_ylabel("Precision")
    axes[1].set_title("Realized Benefit at fixed fractions")
    axes[1].set_xlabel("Routed rows (%)")
    axes[1].set_ylabel("Mean Benefit")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[0].legend(fontsize=8)
    save_figure(figure_root / "03_fixed_fraction_precision_benefit.png")

    rel = reliability.loc[reliability["scope"].eq("selection")].copy()
    atomic_csv(root, REPORTS / "prompt4b3_plot_reliability.csv", rel)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    plt.plot(rel["mean_probability"], rel["observed_beat_rate"], marker="o", label="Beat classifier")
    plt.xlabel("Mean probability-like score")
    plt.ylabel("Observed Beat rate")
    plt.title("Beat classifier reliability (Selection)")
    plt.legend()
    save_figure(figure_root / "04_classifier_reliability.png")

    if policies:
        p0 = read_json(root / DESIGN)["cost_asymmetry"]["p0"]
        alpha_values = {LINEAR_ID: alpha_linear(probability), COSTAWARE_ID: alpha_costaware(probability, p0)}
        alpha_rows = []
        plt.figure(figsize=(7, 5))
        for candidate_id, values in alpha_values.items():
            count, edge = np.histogram(values, bins=np.linspace(0, 1, 41))
            alpha_rows.extend({"candidate_id": candidate_id, "left": edge[i], "right": edge[i + 1], "count": count[i]} for i in range(len(count)))
            plt.hist(values, bins=np.linspace(0, 1, 41), alpha=0.5, label=candidate_id)
        atomic_csv(root, REPORTS / "prompt4b3_plot_alpha_distributions.csv", pd.DataFrame(alpha_rows))
        plt.xlabel("alpha")
        plt.ylabel("Rows")
        plt.title("Frozen shrinkage alpha distributions")
        plt.legend(fontsize=8)
        save_figure(figure_root / "05_alpha_distributions.png")

        cross = pd.read_csv(root / REPORTS / "prompt4b3_cross_stage_comparison.csv")
        compare = cross.loc[cross["candidate_id"].isin(["ens_boost_cat060", "stage3_residual_t75_a75", "nf_global2_oldraw_direct_cap25", LINEAR_ID, COSTAWARE_ID])].copy()
        atomic_csv(root, REPORTS / "prompt4b3_plot_body_tail_tradeoff.csv", compare)
        plt.figure(figsize=(7, 5))
        plt.scatter(compare["bottom_90_mae"], compare["top_decile_mae"], s=60)
        for _, row in compare.iterrows():
            plt.annotate(row["candidate_id"], (row["bottom_90_mae"], row["top_decile_mae"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
        plt.xlabel("Bottom-90 MAE")
        plt.ylabel("Top-decile MAE")
        plt.title("Body/Tail trade-off (complete Validation)")
        save_figure(figure_root / "06_body_tail_tradeoff.png")


def build_manifests(root: Path) -> None:
    model_manifest = read_json(root / REPORTS / "prompt4b3_model_manifest.json")
    model_manifest["status"] = "PASS"
    atomic_json(root, REPORTS / "prompt4b3_model_manifest.json", model_manifest)
    artifacts = []
    for path in sorted((root / PREDICTIONS).glob("*.parquet")):
        frame = pd.read_parquet(path)
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(float)
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(), "sha256": file_sha256(path),
                "rows": len(frame), "columns": frame.columns.tolist(),
                "row_hash_unique": bool(frame["row_hash"].is_unique),
                "finite_numeric": bool(np.isfinite(numeric).all()),
                "compression": sorted({pq.ParquetFile(path).metadata.row_group(0).column(i).compression for i in range(pq.ParquetFile(path).metadata.row_group(0).num_columns)}),
            }
        )
    atomic_json(
        root, REPORTS / "prompt4b3_prediction_manifest.json",
        {"status": "PASS", "created_at_utc": utc_now(), "artifact_count": len(artifacts), "artifacts": artifacts, "iid_prediction_count": 0},
    )


def build_notebook(root: str | Path | None = None) -> Path:
    workspace = Path(root or regression_v2_root()).resolve()
    gate = read_json(workspace / REPORTS / "prompt4b3_diagnostic_gate.json")
    policy = bool(gate["policy_evaluation_authorized"])
    notebook = nbformat.v4.new_notebook()
    cells = [
        nbformat.v4.new_markdown_cell("# Prompt 4B3 - Beat-Probability Residual Shrinkage\n\nAll results are adaptive Development evidence. IID stayed closed. No final model was selected or frozen."),
        nbformat.v4.new_markdown_cell("## Scientific question\n\nGlobal prediction is already strong, and the frozen residual correction has real Tail signal. Tail membership is not the same as correction Benefit. Prompt 4B2 Benefit regression ranked Benefit weakly. Prompt 4B3 changes only the routing target and shrinkage policy; the Global prediction and residual proposal stay frozen."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nREPORTS = ROOT / 'outputs/reports'\nFIGURES = ROOT / 'outputs/figures/prompt4b3'"),
        nbformat.v4.new_markdown_cell("## Frozen design and handoff"),
        nbformat.v4.new_code_cell("design = json.loads((REPORTS / 'prompt4b3_design_freeze.json').read_text())\nhandoff = json.loads((REPORTS / 'prompt4b3_handoff_validation.json').read_text())\ndisplay(pd.DataFrame([{'handoff_status': handoff['status'], 'train_rows': handoff['train_rows'], 'validation_rows': handoff['validation_rows'], 'feature_count': handoff['feature_count'], 'raw_accesses': handoff['raw_access_count'], 'iid_accesses': handoff['iid_feature_access_count'] + handoff['iid_target_access_count']}]))"),
        nbformat.v4.new_markdown_cell("## Beat target and existing score baselines\n\nBeat is one only when the frozen correction reduces absolute error. A zero Benefit is labeled zero. Existing scores are compared on this same target; PR-AUC is not compared with earlier Spearman correlation."),
        nbformat.v4.new_code_cell("audit = json.loads((REPORTS / 'prompt4b3_beat_target_audit.json').read_text())\nbaselines = pd.read_csv(REPORTS / 'prompt4b3_zero_fit_beat_baselines.csv')\ndisplay(pd.DataFrame([audit]))\ndisplay(baselines.loc[baselines['routed_fraction'].eq(0.10), ['score_id','roc_auc','pr_auc_average_precision','precision','recall','mean_realized_benefit','lift_over_base_prevalence']])\ndisplay(Image(filename=str(FIGURES / '01_beat_prevalence_benefit_distribution.png')))\ndisplay(Image(filename=str(FIGURES / '02_beat_pr_curves.png')))"),
        nbformat.v4.new_markdown_cell("## Beat classifier diagnostics and hard gate\n\nThe classifier score is probability-like, but perfect calibration is not assumed. Policies are evaluated only if both frozen diagnostic conditions pass."),
        nbformat.v4.new_code_cell("diagnostics = json.loads((REPORTS / 'prompt4b3_classifier_diagnostics.json').read_text())\ngate = json.loads((REPORTS / 'prompt4b3_diagnostic_gate.json').read_text())\ndisplay(pd.DataFrame(diagnostics['scopes']).T.reset_index(names='scope'))\ndisplay(pd.DataFrame([{'DG1': gate['DG1']['pass'], 'DG1_lower': gate['DG1']['percentile_2_5'], 'DG2': gate['DG2']['pass'], 'new_top10_benefit': gate['DG2']['new_top_10_mean_realized_benefit'], 'best_existing_top10_benefit': gate['DG2']['best_existing_top_10_mean_realized_benefit'], 'policy_evaluation': gate['policy_evaluation_authorized']}]))\ndisplay(Image(filename=str(FIGURES / '03_fixed_fraction_precision_benefit.png')))\ndisplay(Image(filename=str(FIGURES / '04_classifier_reliability.png')))"),
    ]
    if policy:
        cells.extend(
            [
                nbformat.v4.new_markdown_cell("## Two frozen shrinkage policies\n\nBoth policies use the same classifier score and unchanged residual proposal. Policy B uses Train-OOF `p0` as a cost-informed pre-frozen rule, not as a perfect Bayesian threshold."),
                nbformat.v4.new_code_cell("policies = pd.read_csv(REPORTS / 'prompt4b3_policy_results.csv')\nsix = pd.read_csv(REPORTS / 'prompt4b3_six_condition_results.csv')\nchampion = json.loads((REPORTS / 'prompt4b3_experimental_champion.json').read_text())\ndisplay(policies)\ndisplay(six)\ndisplay(pd.DataFrame([{'experimental_champion': champion['candidate_id'], 'selection_only': champion['selection_only'], 'final_project_model': champion['final_project_model']}]))\ndisplay(Image(filename=str(FIGURES / '05_alpha_distributions.png')))\ndisplay(Image(filename=str(FIGURES / '06_body_tail_tradeoff.png')))"),
                nbformat.v4.new_markdown_cell("## Adaptive Development bootstrap\n\nThis paired bootstrap is descriptive. It does not repair adaptive-selection bias."),
                nbformat.v4.new_code_cell("display(pd.read_csv(REPORTS / 'prompt4b3_bootstrap.csv'))"),
            ]
        )
    else:
        cells.append(nbformat.v4.new_markdown_cell("## Diagnostic stop\n\nAt least one frozen diagnostic condition failed, so no shrinkage policy was evaluated. No threshold, feature, classifier, or architecture was added after seeing the result."))
    cells.extend(
        [
            nbformat.v4.new_markdown_cell("## Cross-stage context"),
            nbformat.v4.new_code_cell("display(pd.read_csv(REPORTS / 'prompt4b3_cross_stage_comparison.csv'))"),
            nbformat.v4.new_markdown_cell("## Boundary\n\nThe result is adaptive Development evidence. IID remains completely unopened. No final project model was selected or frozen, and Prompt 4C was not executed. Human review is the next step."),
        ]
    )
    notebook["cells"] = cells
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    path = workspace / NOTEBOOK
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return path


def execute_notebook(root: str | Path | None = None) -> dict[str, Any]:
    workspace = Path(root or regression_v2_root()).resolve()
    path = build_notebook(workspace)
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    client = NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(workspace)}})
    executed = client.execute()
    nbformat.write(executed, path)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
    figures = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "display_data" and "image/png" in output.get("data", {})]
    tables = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type in {"display_data", "execute_result"} and "text/html" in output.get("data", {})]
    evidence = {
        "status": "PASS" if not errors else "FAIL", "created_at_utc": utc_now(),
        "attempt": 1, "artifact_only": True, "code_cells": len(code_cells),
        "executed_code_cells": int(sum(cell.execution_count is not None for cell in code_cells)),
        "error_count": len(errors), "inline_figure_outputs": len(figures), "inline_table_outputs": len(tables),
        "model_fit_calls": 0, "preprocessor_fit_calls": 0, "new_prediction_calls": 0,
        "raw_access_count": 0, "iid_feature_access_count": 0, "iid_target_access_count": 0,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    atomic_json(workspace, REPORTS / "prompt4b3_notebook_execution.json", evidence)
    if evidence["status"] != "PASS":
        raise RuntimeError("Prompt 4B3 artifact-only notebook execution failed.")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "fit", "reload-worker", "evaluate", "notebook"))
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare_design(args.root)
    elif args.command == "fit":
        result = fit_classifier(args.root)
    elif args.command == "reload-worker":
        result = reload_worker(args.root or regression_v2_root())
    elif args.command == "evaluate":
        result = evaluate(args.root)
    else:
        result = execute_notebook(args.root)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
