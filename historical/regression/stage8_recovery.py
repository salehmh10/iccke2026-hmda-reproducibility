"""Bounded Stage 8 saved-decile recovery.

This module keeps the failed Stage 8 attempt immutable.  Recovery outputs use
their own namespace and no source row or model is opened by the preflight and
sample-freeze commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage8_explainability_utils import CANDIDATES, EXPECTED, MODELS, PREDICTIONS


ROOT = Path(__file__).resolve().parent
AUTHORIZATION_ID = "stage8_saved_decile_registry_recovery_20260716"
REPORTS = ROOT / "artifacts/reports"
RECOVERY_MANIFESTS = ROOT / "artifacts/manifests/stage8/recovery"
RECOVERY_RESULTS = ROOT / "artifacts/results/stage8/recovery"
RECOVERY_FIGURES = ROOT / "artifacts/figures/stage8/recovery"
RECOVERY_PLOTTING = RECOVERY_FIGURES / "plotting_data"
REGISTRY = ROOT / "artifacts/results/experiment_results.csv"
SOURCE_HASHES = {
    "data/regression_without_sensitive_features.csv": "e90f7bb49cce5584c7ab250c1db6a107de5cf640c7839f318d7f3cb995edd93c",
    "data/regression_with_sensitive_features.csv": "6dc52dca5a8a7196a75213fab4a5a5c0a541f84390219459afb0b2be7b77aede",
}
OLD_REGISTRY_HASH = "36faee6d39850d11b0c2e2798ebc3a2b8679aafc2bd9eb69b6aa1194105ed879"
OLD_REGISTRY_SIZE = 259_114


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def value_hash(values: object, dtype: object) -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def record(path: Path) -> dict:
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = path.resolve().as_posix()
    return {
        "path": label,
        "sha256": sha(path),
        "size_bytes": path.stat().st_size,
    }


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def validate_prerequisites() -> dict:
    checks: dict[str, object] = {}
    required_status = {
        "artifacts/reports/stage4l_verification.json": "PASS",
        "artifacts/reports/stage5b_verification.json": "PASS",
        "artifacts/reports/stage5c_verification.json": "PASS",
        "artifacts/reports/stage6_verification.json": "PASS",
        "artifacts/reports/stage7_verification.json": "PASS",
        "artifacts/reports/stage7_protected_recheck.json": "PASS",
    }
    for relative, expected in required_status.items():
        payload = read_json(relative)
        actual = payload.get("status", payload.get("stage7_status"))
        checks[relative] = {"actual": actual, "expected": expected, **record(ROOT / relative)}
        if actual != expected:
            raise RuntimeError(f"Prerequisite status failed: {relative} = {actual}")

    stage4l = read_json("artifacts/reports/stage4l_verification.json")
    if not stage4l.get("checks", {}).get("official_primary_unchanged", True):
        raise RuntimeError("Stage 4L official role is not validated")
    stage5b = read_json("artifacts/reports/stage5b_verification.json")
    status_text = json.dumps(stage5b).lower()
    if "rejected" not in status_text:
        raise RuntimeError("Stage 5B rejected ensemble was not validated")

    freeze = read_json("artifacts/reports/stage8_preexplainability_freeze.json")
    if freeze.get("candidate_ids") != CANDIDATES or len(freeze.get("model_identities", [])) != 5:
        raise RuntimeError("Frozen candidate or model identity contract changed")
    checks["candidate_predictors"] = 3
    checks["frozen_model_identities"] = 5

    for item in MODELS + PREDICTIONS:
        path = ROOT / item["path"]
        actual = sha(path)
        checks[item["id"]] = {"path": item["path"], "expected": item["sha256"], "actual": actual}
        if actual != item["sha256"]:
            raise RuntimeError(f"Frozen artifact hash mismatch: {item['id']}")
    for relative, expected in SOURCE_HASHES.items():
        actual = sha(ROOT / relative)
        checks[relative] = {"expected": expected, "actual": actual}
        if actual != expected:
            raise RuntimeError(f"Source hash mismatch: {relative}")

    blocked = read_json("artifacts/reports/stage8_verification.json")
    counters = blocked.get("counters", {})
    zero_contracts = {
        "model_fit_calls": counters.get("model_fit_calls"),
        "preprocessing_fit_calls": counters.get("preprocessing_fit_calls"),
        "global_shap_recomputations": counters.get("global_shap_recomputations"),
        "new_evaluation_prediction_files": counters.get("new_evaluation_prediction_files"),
    }
    if any(value != 0 for value in zero_contracts.values()):
        raise RuntimeError(f"Historical zero-operation contract failed: {zero_contracts}")
    checks["historical_zero_operations"] = zero_contracts
    checks["stage9_started"] = bool(counters.get("stage9_started", False))
    if checks["stage9_started"]:
        raise RuntimeError("Stage 9 has started")
    return checks


def backup_paths() -> tuple[Path, list[dict]]:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    root = ROOT / "artifacts/backups" / f"stage8_recovery_{stamp}"
    required = [
        "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb",
        "artifacts/results/experiment_results.csv",
        "TASK.md",
        "PLAN.md",
        "DECISIONS.md",
        "LOG.md",
        "AGENTS.md",
        "artifacts/reports/stage8_reviewer.md",
        "artifacts/reports/stage8_verification.json",
        "artifacts/manifests/stage8/stage8_stage9_handoff.json",
        "artifacts/manifests/stage8/stage8_visualization_manifest.json",
        "artifacts/results/stage8/explainability/stage8_global_explanation_summary.json",
    ]
    entries = []
    for relative in required:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(relative)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entries.append({**record(source), "backup_path": target.relative_to(ROOT).as_posix()})
    dump({"authorization_id": AUTHORIZATION_ID, "created_at_utc": now(), "entries": entries}, root / "backup_manifest.json")
    return root, entries


def protected_paths(backup_root: Path) -> list[Path]:
    paths: set[Path] = set()
    previous = read_json("artifacts/manifests/stage8/stage8_protected_hashes_before.json")
    for item in previous.get("entries", []):
        candidate = ROOT / item["path"]
        if candidate.exists() and candidate.is_file():
            paths.add(candidate)
    stage8_roots = [
        ROOT / "artifacts/results/stage8",
        ROOT / "artifacts/figures/stage8",
        ROOT / "artifacts/manifests/stage8",
    ]
    for base in stage8_roots:
        if base.exists():
            paths.update(path for path in base.rglob("*") if path.is_file())
    paths.update(path for path in REPORTS.glob("stage8*") if path.is_file())
    paths.update(ROOT / relative for relative in SOURCE_HASHES)
    paths.update(ROOT / item["path"] for item in MODELS + PREDICTIONS)
    extra = [
        "artifacts/results/stage4/catboost/final/catboost_final_importance_without_sensitive.csv",
        "artifacts/results/stage4/catboost/final/catboost_final_shap_without_sensitive.csv",
        "artifacts/features/stage4/lightgbm/stage4h_importance_source_without_sensitive.csv",
        "artifacts/features/stage4/lightgbm/stage4h_shap_mean_absolute_without_sensitive.csv",
        "artifacts/results/stage4/xgboost/final/stage4k_importance_aggregated_without_sensitive.csv",
        "artifacts/results/stage4/xgboost/final/stage4k_shap_complete_without_sensitive.csv",
        "artifacts/results/stage5/deep_core/summary/stage5a2_feature_attribution.csv",
        "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb",
        "artifacts/results/experiment_results.csv",
    ]
    paths.update(ROOT / relative for relative in extra)
    paths.update(path for path in backup_root.rglob("*") if path.is_file())
    return sorted((path for path in paths if path.exists()), key=lambda path: path.as_posix().lower())


def invalidation_manifest() -> dict:
    affected: list[Path] = []
    result_dir = ROOT / "artifacts/results/stage8/explainability"
    affected_names = {
        "stage8_common_permutation_importance.csv",
        "stage8_permutation_repeat_stability.csv",
        "stage8_cross_model_feature_comparison.csv",
        "stage8_cross_model_agreement.csv",
        "stage8_deep_attribution_comparison.csv",
        "stage8_cross_method_agreement.csv",
        "stage8_feature_family_summary.csv",
        "stage8_sensitive_feature_dependence.csv",
        "stage8_potential_proxy_overlap.csv",
        "stage8_prediction_reconciliation.csv",
        "stage8_local_attributions_public.csv",
        "stage8_local_prediction_reconciliation.csv",
        "stage8_local_explanation_stability.csv",
        "stage8_local_explanation_stability_detail.csv",
        "stage8_case_explanation_synthesis.csv",
        "stage8_global_explanation_summary.json",
        "stage8_feature_interpretation_report.md",
        "stage8_registry_rows.csv",
    }
    affected.extend(result_dir / name for name in affected_names if (result_dir / name).exists())
    affected.extend(path for path in (ROOT / "artifacts/figures/stage8").rglob("*") if path.is_file())
    for relative in [
        "artifacts/manifests/stage8/stage8_global_explanation_sample_row_ids.csv",
        "artifacts/manifests/stage8/stage8_local_background_row_ids.csv",
        "artifacts/manifests/stage8/stage8_visualization_manifest.json",
        "artifacts/manifests/stage8/stage8_stage9_handoff.json",
        "artifacts/reports/stage8_registry_update.json",
    ]:
        path = ROOT / relative
        if path.exists():
            affected.append(path)
    entries = []
    for path in sorted(set(affected), key=lambda item: item.as_posix().lower()):
        entries.append({
            **record(path),
            "original_status": "BLOCKED_or_scientifically_affected",
            "invalidated_reason": "invalidated_by_stage8_saved_decile_recovery",
            "invalidated_scope": "initial Stage 8 sample/background-dependent evidence",
            "scientifically_reusable": False,
            "audit_only": True,
            "superseding_recovery_artifact_path": None,
        })
    payload = {
        "authorization_id": AUTHORIZATION_ID,
        "created_at_utc": now(),
        "status": "invalidated_by_stage8_saved_decile_recovery",
        "entry_count": len(entries),
        "entries": entries,
    }
    dump(payload, RECOVERY_MANIFESTS / "stage8_initial_attempt_invalidation_manifest.json")
    return payload


def registry_search() -> dict:
    candidates = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size == OLD_REGISTRY_SIZE:
                digest = sha(path)
                candidates.append({**record(path), "exact_match": digest == OLD_REGISTRY_HASH})
        except OSError:
            continue
    exact = [item for item in candidates if item["exact_match"]]
    payload = {
        "authorization_id": AUTHORIZATION_ID,
        "searched_at_utc": now(),
        "root": str(ROOT),
        "target_sha256": OLD_REGISTRY_HASH,
        "target_size_bytes": OLD_REGISTRY_SIZE,
        "size_matched_candidates": candidates,
        "exact_matches": exact,
        "exact_bytes_found": bool(exact),
        "search_scope": [
            "artifacts/backups",
            "temporary Notebook outputs",
            "Stage 7 execution backups",
            "Registry snapshots",
            "completion backups",
            "recovery folders",
            "hidden files under project root",
        ],
        "status": "PASS",
    }
    dump(payload, REPORTS / "stage8_registry_recovery_search.json")
    return payload


def preflight() -> None:
    for directory in [RECOVERY_MANIFESTS, RECOVERY_RESULTS, RECOVERY_FIGURES, RECOVERY_PLOTTING]:
        directory.mkdir(parents=True, exist_ok=True)
    checks = validate_prerequisites()
    backup_root, backups = backup_paths()
    registry_before = REGISTRY.read_bytes()
    registry = pd.read_csv(REGISTRY)
    invalidation = invalidation_manifest()
    search = registry_search()
    entries = [record(path) for path in protected_paths(backup_root)]
    baseline = {
        "authorization_id": AUTHORIZATION_ID,
        "created_at_utc": now(),
        "status": "PASS",
        "prerequisite_checks": checks,
        "backup_root": backup_root.relative_to(ROOT).as_posix(),
        "backup_entries": backups,
        "protected_file_count": len(entries),
        "entries": entries,
        "registry": {
            "path": REGISTRY.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(registry_before).hexdigest(),
            "size_bytes": len(registry_before),
            "row_count": len(registry),
            "experiment_ids": registry.experiment_id.astype(str).tolist(),
        },
        "blocked_notebook": record(ROOT / "REGRESSION_PART8_FINAL_EXPLAINABILITY.ipynb"),
        "invalidation_manifest": record(RECOVERY_MANIFESTS / "stage8_initial_attempt_invalidation_manifest.json"),
        "registry_search": record(REPORTS / "stage8_registry_recovery_search.json"),
        "exact_registry_bytes_found": search["exact_bytes_found"],
        "invalidation_entry_count": invalidation["entry_count"],
        "stage9_started": False,
    }
    dump(baseline, RECOVERY_MANIFESTS / "stage8_recovery_protected_baseline.json")
    print(json.dumps({
        "status": "PASS",
        "authorization_id": AUTHORIZATION_ID,
        "backup_root": baseline["backup_root"],
        "protected_file_count": len(entries),
        "registry_sha256": baseline["registry"]["sha256"],
        "registry_rows": baseline["registry"]["row_count"],
        "exact_registry_bytes_found": search["exact_bytes_found"],
        "invalidation_entries": invalidation["entry_count"],
    }, indent=2))


def select_per_decile(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    parts = []
    for decile, group in frame.groupby("target_decile", sort=True):
        if len(group) < count:
            raise RuntimeError(f"Decile {decile} has fewer than {count} rows")
        parts.append(group.sample(n=count, random_state=42))
    return pd.concat(parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)


def sample_freeze() -> None:
    baseline_path = RECOVERY_MANIFESTS / "stage8_recovery_protected_baseline.json"
    if not baseline_path.exists():
        raise RuntimeError("Recovery protected baseline is missing")
    without = pd.read_csv(ROOT / PREDICTIONS[1]["path"], usecols=["row_id", "y_true", "target_decile"])
    with_sensitive = pd.read_csv(ROOT / PREDICTIONS[2]["path"], usecols=["row_id", "y_true", "target_decile"])
    for frame in [without, with_sensitive]:
        frame["row_id"] = frame.row_id.astype(np.int64)
        frame["y_true"] = frame.y_true.astype(np.float64)
        frame["target_decile"] = frame.target_decile.astype(int)
    if len(without) != EXPECTED["rows"] or without.row_id.nunique() != EXPECTED["rows"]:
        raise RuntimeError("Authoritative Stage 5C row count failed")
    if not without.row_id.equals(with_sensitive.row_id):
        raise RuntimeError("Stage 5C Deep row IDs do not align")
    if not np.array_equal(without.y_true.to_numpy(), with_sensitive.y_true.to_numpy()):
        raise RuntimeError("Stage 5C Deep targets do not align")
    if not np.array_equal(without.target_decile.to_numpy(), with_sensitive.target_decile.to_numpy()):
        raise RuntimeError("Stage 5C saved deciles do not align")
    if value_hash(without.row_id, np.int64) != EXPECTED["row_hash"]:
        raise RuntimeError("Frozen Test row hash failed")
    if value_hash(without.y_true, np.float64) != EXPECTED["target_hash"]:
        raise RuntimeError("Frozen Test target hash failed")
    deciles = without.target_decile
    if sorted(deciles.unique().tolist()) != list(range(1, 11)) or deciles.isna().any():
        raise RuntimeError("Saved deciles are incomplete or invalid")

    initial = pd.read_csv(ROOT / "artifacts/manifests/stage8/stage8_global_explanation_sample_row_ids.csv")
    recomputed = pd.qcut(without.y_true.rank(method="first"), 10, labels=False).astype(int) + 1
    mismatch_count = int((recomputed.to_numpy() != deciles.to_numpy()).sum())
    if mismatch_count != 878:
        raise RuntimeError(f"Expected 878 saved-decile mismatches, found {mismatch_count}")

    sample = select_per_decile(without, 200)
    sample_path = RECOVERY_MANIFESTS / "stage8_recovery_global_sample_row_ids.csv"
    sample[["row_id", "target_decile"]].to_csv(sample_path, index=False)
    sample_ids = set(sample.row_id.astype(int))
    initial_overlap = len(sample_ids & set(initial.row_id.astype(int)))
    if initial_overlap != 206:
        raise RuntimeError(f"Expected initial sample overlap 206, found {initial_overlap}")

    cases = pd.read_csv(ROOT / "artifacts/manifests/stage8/stage8_local_case_manifest.csv")
    if len(cases) > 20 or cases.row_id.duplicated().any():
        raise RuntimeError("Frozen local case contract failed")
    if not set(cases.row_id.astype(int)).issubset(set(without.row_id.astype(int))):
        raise RuntimeError("A frozen local case is outside Test")
    eligible = sample[~sample.row_id.astype(int).isin(set(cases.row_id.astype(int)))].copy()
    background = select_per_decile(eligible, 4)
    background_path = RECOVERY_MANIFESTS / "stage8_recovery_local_background_row_ids.csv"
    background[["row_id", "target_decile"]].to_csv(background_path, index=False)
    if set(background.row_id.astype(int)) & set(cases.row_id.astype(int)):
        raise RuntimeError("Background overlaps frozen local cases")

    original_freeze = ROOT / "artifacts/reports/stage8_preexplainability_freeze.json"
    incident = ROOT / "artifacts/reports/stage8_sample_contract_incident.json"
    validation = {
        "authorization_id": AUTHORIZATION_ID,
        "created_at_utc": now(),
        "status": "PASS",
        "authoritative_sources": [record(ROOT / PREDICTIONS[1]["path"]), record(ROOT / PREDICTIONS[2]["path"])],
        "row_count": len(without),
        "row_id_hash": value_hash(without.row_id, np.int64),
        "target_hash": value_hash(without.y_true, np.float64),
        "saved_target_decile_values_align": True,
        "saved_target_decile_complete": True,
        "saved_target_decile_counts": {str(k): int(v) for k, v in deciles.value_counts().sort_index().items()},
        "invalid_recomputed_decile_mismatch_count": mismatch_count,
        "correct_sample_overlap_with_invalid_initial": initial_overlap,
        "correct_background_overlap_with_invalid_initial": len(set(background.row_id.astype(int)) & set(pd.read_csv(ROOT / "artifacts/manifests/stage8/stage8_local_background_row_ids.csv").row_id.astype(int))),
    }
    dump(validation, REPORTS / "stage8_recovery_saved_decile_validation.json")

    freeze = {
        "authorization_id": AUTHORIZATION_ID,
        "created_at_utc": now(),
        "status": "PASS",
        "supersedes_only_invalid_sample_and_exhausted_recovery_contracts": True,
        "original_freeze": record(original_freeze),
        "sample_contract_incident": record(incident),
        "reason_for_supersession": "Initial Stage 8 recomputed target deciles instead of reusing immutable Stage 5C target_decile.",
        "authoritative_saved_decile_source": record(ROOT / PREDICTIONS[1]["path"]),
        "candidate_ids": CANDIDATES,
        "model_identities": MODELS,
        "methods": read_json("artifacts/reports/stage8_preexplainability_freeze.json")["methods"],
        "local_cases": record(ROOT / "artifacts/manifests/stage8/stage8_local_case_manifest.csv"),
        "privacy_policy": read_json("artifacts/reports/stage8_preexplainability_freeze.json")["privacy_policy"],
        "figure_scope": read_json("artifacts/reports/stage8_preexplainability_freeze.json")["figures"],
        "registry_recovery_policy": "Path A exact restoration when found; otherwise authorized Path B semantic-preservation adjudication.",
        "sample": {**record(sample_path), "row_count": len(sample), "row_id_hash": value_hash(sample.row_id, np.int64), "target_hash": value_hash(sample.y_true, np.float64), "counts": {str(k): int(v) for k, v in sample.target_decile.value_counts().sort_index().items()}},
        "background": {**record(background_path), "row_count": len(background), "row_id_hash": value_hash(background.row_id, np.int64), "counts": {str(k): int(v) for k, v in background.target_decile.value_counts().sort_index().items()}, "local_case_overlap": 0},
        "permutation_repeats": 2,
        "permutation_seeds": [42, 43],
        "blend_weights": {"catboost": 0.60, "lightgbm": 0.20, "xgboost": 0.20},
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "global_shap_recomputations": 0,
        "new_evaluation_prediction_files": 0,
        "stage9_started": False,
    }
    freeze_path = REPORTS / "stage8_recovery_sample_freeze.json"
    dump(freeze, freeze_path)
    reloaded = read_json("artifacts/reports/stage8_recovery_sample_freeze.json")
    if reloaded["sample"]["row_count"] != 2000 or reloaded["background"]["row_count"] != 40:
        raise RuntimeError("Recovery freeze reload failed")
    print(json.dumps({
        "status": "PASS",
        "mismatch_count": mismatch_count,
        "sample_rows": len(sample),
        "sample_hash": freeze["sample"]["row_id_hash"],
        "sample_counts": freeze["sample"]["counts"],
        "background_rows": len(background),
        "background_hash": freeze["background"]["row_id_hash"],
        "background_counts": freeze["background"]["counts"],
        "local_cases": len(cases),
        "freeze_sha256": sha(freeze_path),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "sample-freeze"])
    args = parser.parse_args()
    if args.command == "preflight":
        preflight()
    else:
        sample_freeze()


if __name__ == "__main__":
    main()
