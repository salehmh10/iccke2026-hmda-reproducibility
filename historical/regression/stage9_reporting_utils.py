"""Stage 9 reporting helpers.

This module only reads aggregate, public, saved evidence. It never reads source
CSV values, model files, bundles, restricted sensitive data, or predictions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
STAGE9_RESULTS = ROOT / "artifacts/results/stage9/reporting"
STAGE9_MANIFESTS = ROOT / "artifacts/manifests/stage9"
STAGE9_REPORTS = ROOT / "artifacts/reports"
STAGE9_FIGURES = ROOT / "artifacts/figures/stage9"


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_stage9_directories() -> None:
    for path in (
        STAGE9_RESULTS,
        STAGE9_MANIFESTS,
        STAGE9_REPORTS,
        STAGE9_FIGURES / "report",
        STAGE9_FIGURES / "slides",
        STAGE9_FIGURES / "vector",
        STAGE9_FIGURES / "plotting_data",
        ROOT / "artifacts/backups",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _is_protected(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(("artifacts/backups/", "artifacts/environment/")):
        return False
    if "/stage9/" in rel or rel.startswith("artifacts/results/stage9/"):
        return False
    if rel in {"TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md", "AGENTS.md"}:
        return False
    if rel.startswith("data/"):
        return path.is_file()
    if rel.startswith("artifacts/"):
        return path.is_file()
    if path.parent == ROOT and path.suffix == ".ipynb" and "PART9" not in path.name:
        return True
    return False


def protected_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if path.is_file() and _is_protected(path):
            yield path


def create_protected_baseline() -> dict[str, Any]:
    target = STAGE9_MANIFESTS / "stage9_protected_hashes_before.json"
    if target.exists():
        return read_json(target)
    files = []
    for path in sorted(protected_files(), key=lambda item: posix(item)):
        files.append({"path": posix(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    registry = ROOT / "artifacts/results/experiment_results.csv"
    baseline = {
        "stage_id": "stage9",
        "created_at_utc": utc_now(),
        "policy": "All listed Stage 1-8 and source artifacts are immutable during Stage 9.",
        "file_count": len(files),
        "registry_path": posix(registry),
        "registry_size_bytes": registry.stat().st_size,
        "registry_sha256": sha256_file(registry),
        "files": files,
    }
    write_json(target, baseline)
    return baseline


def create_prereport_freeze() -> dict[str, Any]:
    target = STAGE9_REPORTS / "stage9_prereport_freeze.json"
    if target.exists():
        return read_json(target)
    handoff = ROOT / "artifacts/manifests/stage8/recovery/stage8_recovery_stage9_handoff.json"
    sections = [
        "Stage Objective and Audience", "Imports and Reporting Configuration", "State Reconstruction",
        "Stage 8 Recovery Verification and Handoff", "Reporting Scope and Non-Technical Contract",
        "Protected File Baseline", "Pre-Report Freeze", "Evidence Inventory and Claim Matrix",
        "Project Question in Plain Language", "Dataset Scope and Target", "Data Split and Evaluation Governance",
        "Metric Glossary", "Complete Project Workflow", "Modeling Journey", "Model Roles and Comparability",
        "Official Test Result", "Post-Test Deep Comparison", "Ensemble Decision",
        "Error Distribution and Concentration", "Target-Decile and Tail Analysis",
        "Underprediction and Overprediction", "Fairness Scope", "Fairness Findings", "Explainability Scope",
        "Global Feature Interpretation", "Local Case Interpretation", "Governance Incidents",
        "Responsible-Use Recommendation", "Non-Technical FAQ", "Final Story Figures",
        "Visual and Axis Integrity Audit", "Executive Summary and One-Page Brief", "Model Card",
        "Final Technical Report", "Presentation Storyboard and Speaker Notes",
        "Registry and Stage 10 Handoff", "Independent Review, Verification, and Completion",
    ]
    figures = [
        "Project Workflow", "Dataset Scope and Split", "Modeling Journey and Decision Timeline",
        "Comparable Final Selection Validation Performance", "Final Test MAE Comparison",
        "Final Multi-Metric Comparison", "Paired Uncertainty", "Why the Ensemble Was Rejected",
        "Error by Target Decile", "Tail and Underprediction Risk", "Error Concentration",
        "Fairness Scope and Group Disparity Summary", "Accuracy Versus Disparity Trade-Off",
        "Global Feature Importance Consensus", "Privacy-Safe Local Explanation Cases",
        "Final Project Decision and Governance Dashboard",
    ]
    chart_types = [
        "workflow", "proportional_flow", "timeline", "dot_plot", "focused_dot_plot", "small_multiples",
        "interval_plot", "decision_gate", "line_plot", "small_multiples", "dot_plot", "aggregate_dashboard",
        "tradeoff_dot_plot", "rank_dot_plot", "privacy_safe_case_panels", "governance_dashboard",
    ]
    freeze = {
        "stage_id": "stage9",
        "stage_name": "Stage 9 — Model Card and Final Technical Report",
        "reporting_label": "Final Project Synthesis with Post-Test Disclosures",
        "created_at_utc": utc_now(),
        "stage8_recovery_handoff_path": posix(handoff),
        "stage8_recovery_handoff_sha256": sha256_file(handoff),
        "official_model_identity": "stage4l__blend__without_sensitive",
        "post_test_candidate_identities": [
            "stage5c__realmlp__without_sensitive__test_evaluation",
            "stage5c__realmlp__with_sensitive__test_evaluation",
        ],
        "rejected_ensemble_identity": "stage5b__realmlp_0.50__boosting_0.50",
        "report_audience": ["non-technical reader", "technical reviewer", "maintainer", "presentation audience", "decision-maker"],
        "report_section_list": sections,
        "model_card_section_list": [
            "Model Details", "Model Purpose", "Intended Use", "Out-of-Scope Use",
            "Training and Evaluation Data", "Evaluation", "Error Characteristics", "Fairness",
            "Explainability", "Governance", "Limitations", "Monitoring Recommendations", "Reproducibility",
        ],
        "figure_list": figures,
        "chart_type_list": chart_types,
        "axis_policies": {
            "filled_bars_start_at_zero": True,
            "close_values_use_focused_points_or_intervals": True,
            "focused_axes_must_be_labelled": True,
            "dual_axes_prohibited": True,
            "all_axes_show_units": True,
        },
        "metric_direction_policies": {"MAE": "lower_is_better", "RMSE": "lower_is_better", "RMSLE": "lower_is_better", "R2": "higher_is_better", "mean_signed_error": "closer_to_zero"},
        "unit_conversion_policy": "loan_amount_000s × 1000 = US dollars; label both units explicitly",
        "number_format_policy": "technical metrics retain at least six decimals; main report uses three decimals; dollar explanations use sensible rounding",
        "official_post_test_label_policy": "Stage 4L is official_pre_registered_primary; Stage 5C candidates are Post-Test Extensions and cannot be promoted",
        "fairness_wording_policy": "descriptive observational disparity analysis for approved applications only; no approval-fairness, causal, legal, or compliance conclusion",
        "explainability_wording_policy": "importance and local substitution describe model reliance, not causality; reference substitution is not SHAP and is non-additive",
        "governance_disclosure_policy": "always disclose Stage 5A procedural materialization exception and Stage 8 Registry Governance Path B",
        "presentation_slide_count": 14,
        "presentation_story_order": list(range(1, 15)),
        "registry_ids": [
            "stage9__executive_summary", "stage9__model_card", "stage9__final_technical_report",
            "stage9__final_visual_package", "stage9__presentation_storyboard", "stage9__stage10_handoff",
        ],
        "notebook_attempt_limit": 3,
        "reviewer_cycle_limit": 2,
        "stage10_prohibition": "Stage 10 must not start during Stage 9.",
    }
    write_json(target, freeze)
    return freeze


def validate_prereport_freeze(freeze: dict[str, Any]) -> None:
    assert freeze["stage_id"] == "stage9"
    assert len(freeze["figure_list"]) == 16
    assert len(freeze["report_section_list"]) == 37
    assert freeze["presentation_slide_count"] == 14
    assert len(freeze["registry_ids"]) == 6
    handoff = ROOT / freeze["stage8_recovery_handoff_path"]
    assert sha256_file(handoff) == freeze["stage8_recovery_handoff_sha256"]


if __name__ == "__main__":
    ensure_stage9_directories()
    baseline = create_protected_baseline()
    freeze = create_prereport_freeze()
    validate_prereport_freeze(freeze)
    print(json.dumps({
        "status": "PASS",
        "protected_file_count": baseline["file_count"],
        "protected_baseline_sha256": sha256_file(STAGE9_MANIFESTS / "stage9_protected_hashes_before.json"),
        "prereport_freeze_sha256": sha256_file(STAGE9_REPORTS / "stage9_prereport_freeze.json"),
        "stage8_handoff_sha256": freeze["stage8_recovery_handoff_sha256"],
    }, indent=2))
