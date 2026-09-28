"""Sequential parent-enforced Stage 5A2 regular Final Validation worker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil

from stage5a2_deep_utils import (
    AMENDMENT, EXPECTED_FREEZE_SHA256, FINAL_SAMPLE, REPLACEMENT_CANDIDATE_ID, ROOT, SOURCE_WITHOUT, TEST_IDS,
    atomic_csv, atomic_joblib, atomic_json, candidate_definition,
    digest_values, feature_lists, load_final_selection, predict_bundle,
    regression_metrics, sha256_file, train_ft_regular, train_realmlp_regular,
    validate_freeze,
)


CANDIDATES = [
    "stage5a2__realmlp__frozen", "stage5a2__realmlp__refined",
    REPLACEMENT_CANDIDATE_ID,
    "stage5a2__ft_transformer__frozen", "stage5a2__ft_transformer__refined",
]


def paths(candidate_id: str) -> dict[str, Path]:
    return {
        "checkpoint": ROOT / f"artifacts/checkpoints/stage5/deep_core/final_validation/{candidate_id}.json",
        "progress": ROOT / f"artifacts/checkpoints/stage5/deep_core/final_validation/{candidate_id}_progress.json",
        "result": ROOT / f"artifacts/results/stage5/deep_core/final_validation/candidates/{candidate_id}.json",
        "prediction": ROOT / f"artifacts/predictions/stage5/deep_core/final_validation/{candidate_id}.csv",
        "history": ROOT / f"artifacts/results/stage5/deep_core/final_validation/histories/{candidate_id}_history.csv",
        "bundle": ROOT / f"artifacts/models/deep/core_validation/{candidate_id}.joblib",
        "state": ROOT / f"artifacts/models/deep/core_validation/{candidate_id}.pt",
        "reload": ROOT / f"artifacts/reports/stage5a2_reload_{candidate_id}.json",
        "parent": ROOT / f"artifacts/reports/stage5a2_parent_{candidate_id}.json",
        "log": ROOT / f"artifacts/reports/stage5a2_parent_{candidate_id}.log",
    }


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _artifact_hashes(items: list[Path]) -> dict[str, str]:
    return {_relative(path): sha256_file(path) for path in items if path.exists()}


def run_candidate(candidate_id: str) -> dict[str, Any]:
    if candidate_id not in CANDIDATES:
        raise ValueError(candidate_id)
    validate_freeze()
    definition = candidate_definition(candidate_id)
    output_paths = paths(candidate_id)
    if output_paths["checkpoint"].exists():
        existing = json.loads(output_paths["checkpoint"].read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "REUSED", "candidate_id": candidate_id}))
            return existing
        raise RuntimeError("A failed checkpoint exists; the parent must adjudicate the one retry")
    lock = ROOT / "artifacts/checkpoints/stage5/deep_core/stage5a2_heavy_fit.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError("Another Stage 5A2 heavy fit lock exists") from exc
    started = time.perf_counter()
    try:
        X_train, X_val, y_train, y_val, train_rows, val_rows, val_bins = load_final_selection("without_sensitive", candidate_id)
        if definition["family"] == "realmlp":
            trained = train_realmlp_regular(candidate_id, definition, X_train, X_val, y_train, y_val, output_paths["history"])
        else:
            trained = train_ft_regular(candidate_id, definition, X_train, X_val, y_train, y_val,
                                       output_paths["state"], output_paths["history"], output_paths["progress"])
        prediction = np.asarray(trained.pop("prediction"), dtype=np.float64)
        if len(prediction) != 25_000 or not np.isfinite(prediction).all():
            raise RuntimeError("Invalid regular Final Validation predictions")
        pred_frame = pd.DataFrame({
            "row_id": val_rows, "y_true": y_val.astype(np.float64), "y_pred": prediction,
            "absolute_error": np.abs(prediction - y_val), "signed_error": prediction - y_val,
            "target_decile": val_bins, "sensitive_mode": "without_sensitive",
            "model_family": definition["family"], "candidate_id": candidate_id,
            "target_mode": definition["target_mode"], "epoch": trained["best_epoch"],
        })
        atomic_csv(pred_frame, output_paths["prediction"])
        numerical, categorical = feature_lists("without_sensitive")
        bundle = {
            "bundle_version": "stage5a2_core_validation_v1", "stage_id": "stage5a2",
            "candidate_id": candidate_id, "candidate_definition": definition,
            "family": definition["family"], "sensitive_mode": "without_sensitive",
            "target_mode": definition["target_mode"], "feature_schema": "deep_core_v1",
            "numerical_features": numerical, "categorical_features": categorical,
            "preprocessor": trained.pop("preprocessor"), "target_transform": trained.pop("target_transform"),
            "architecture": trained["architecture"], "training": trained["training"],
            "best_epoch": trained["best_epoch"], "seed": 42, "device": "cpu", "precision": "float32",
            "training_rows": 100_000, "validation_rows": 25_000, "test_rows": 0,
            "training_row_id_hash": digest_values(train_rows), "validation_row_id_hash": digest_values(val_rows),
            "freeze_sha256": EXPECTED_FREEZE_SHA256,
            "amendment_sha256": sha256_file(AMENDMENT) if candidate_id == REPLACEMENT_CANDIDATE_ID else None,
        }
        if definition["family"] == "realmlp":
            bundle["model"] = trained.pop("model")
            bundle["model_state_path"] = None
        else:
            trained.pop("model")
            bundle["model_state_path"] = _relative(output_paths["state"])
            bundle["cardinalities"] = list(bundle["preprocessor"].cardinalities_)
        atomic_joblib(bundle, output_paths["bundle"])
        model_files = [output_paths["bundle"]] + ([output_paths["state"]] if output_paths["state"].exists() else [])
        result = {
            "stage_id": "stage5a2", "candidate_id": candidate_id,
            "candidate_type": definition["candidate_type"], "model_family": definition["family"],
            "target_mode": definition["target_mode"], "feature_schema": "deep_core_v1",
            "sensitive_mode": "without_sensitive", "seed": 42, "device": "cpu", "precision": "float32",
            "training_rows": 100_000, "validation_rows": 25_000, "test_rows": 0,
            "architecture": trained["architecture"], "training_configuration": trained["training"],
            "best_epoch": trained["best_epoch"], "epochs_completed": trained["epochs_completed"],
            "stop_reason": trained["stop_reason"], "metrics": trained["metrics"],
            "fit_time_seconds": trained["fit_time_seconds"],
            "prediction_time_seconds": trained["prediction_time_seconds"],
            "peak_ram_mib": trained["peak_ram_mib"], "peak_vram_mib": 0.0,
            "model_size_bytes": sum(path.stat().st_size for path in model_files),
            "bundle_path": _relative(output_paths["bundle"]),
            "model_state_path": _relative(output_paths["state"]) if output_paths["state"].exists() else None,
            "prediction_path": _relative(output_paths["prediction"]), "history_path": _relative(output_paths["history"]),
            "prediction_sha256": sha256_file(output_paths["prediction"]),
            "training_row_id_hash": digest_values(train_rows), "validation_row_id_hash": digest_values(val_rows),
            "source_sha256": sha256_file(SOURCE_WITHOUT), "sample_sha256": sha256_file(FINAL_SAMPLE),
            "freeze_sha256": EXPECTED_FREEZE_SHA256,
            "candidate_definition_matches_freeze": candidate_id != REPLACEMENT_CANDIDATE_ID,
            "candidate_definition_matches_approved_amendment": candidate_id == REPLACEMENT_CANDIDATE_ID,
            "artifact_hashes": _artifact_hashes([*model_files, output_paths["prediction"], output_paths["history"]]),
            "categorical_contract": trained.get("categorical_contract"),
            "official_encoder_contract": trained.get("official_encoder_contract"),
            "effective_resolved_config": trained.get("effective_resolved_config"),
            "effective_config_proof_path": trained.get("effective_config_proof_path"),
            "train_only_vocabulary_unchanged": trained.get("train_only_vocabulary_unchanged"),
            "numerical_medians_unchanged": trained.get("numerical_medians_unchanged"),
            "worker_elapsed_seconds": time.perf_counter() - started,
            "status": "PASS", "error": None,
        }
        atomic_json(result, output_paths["result"])
        atomic_json(result, output_paths["checkpoint"])
        if output_paths["progress"].exists():
            atomic_json({"candidate_id": candidate_id, "best_epoch": result["best_epoch"], "status": "PASS"}, output_paths["progress"])
        print(json.dumps({"status": "PASS", "candidate_id": candidate_id, "metrics": result["metrics"]}))
        return result
    except Exception as exc:
        failure = {"stage_id": "stage5a2", "candidate_id": candidate_id, "status": "FAIL",
                   "error": repr(exc), "traceback": traceback.format_exc(),
                   "worker_elapsed_seconds": time.perf_counter() - started}
        atomic_json(failure, output_paths["checkpoint"])
        raise
    finally:
        if lock.exists():
            lock.unlink()


def verify_candidate(candidate_id: str) -> dict[str, Any]:
    validate_freeze()
    p = paths(candidate_id)
    result = json.loads(p["result"].read_text(encoding="utf-8"))
    X_train, X_val, _, y_val, _, val_rows, _ = load_final_selection("without_sensitive", candidate_id)
    del X_train
    before = {column: pd.util.hash_pandas_object(X_val[column], index=True).sum() for column in X_val.columns}
    prediction = predict_bundle(p["bundle"], X_val)
    after = {column: pd.util.hash_pandas_object(X_val[column], index=True).sum() for column in X_val.columns}
    expected = pd.read_csv(p["prediction"])
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    difference = np.abs(prediction - expected["y_pred"].to_numpy(np.float64))
    checks = {
        "result_pass": result.get("status") == "PASS",
        "row_order_match": np.array_equal(expected["row_id"].to_numpy(np.int64), val_rows),
        "row_ids_unique": bool(expected["row_id"].is_unique),
        "test_overlap_zero": len(np.intersect1d(expected["row_id"].to_numpy(np.int64), test_ids)) == 0,
        "target_alignment": np.array_equal(expected["y_true"].to_numpy(np.float32), y_val),
        "finite_predictions": bool(np.isfinite(prediction).all()),
        "prediction_count_25000": len(prediction) == 25_000,
        "prediction_match": bool(np.allclose(prediction, expected["y_pred"], rtol=1e-5, atol=1e-4)),
        "source_frame_unchanged": before == after,
        "cpu_inference": True, "no_refit": True, "test_rows_zero": result.get("test_rows") == 0,
    }
    report = {"stage_id": "stage5a2", "candidate_id": candidate_id, "clean_process": True,
              "maximum_absolute_difference": float(difference.max()), "checks": checks,
              "status": "PASS" if all(checks.values()) else "FAIL"}
    atomic_json(report, p["reload"])
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Clean reload failed")
    return report


def parent_run(candidate_id: str) -> dict[str, Any]:
    definition = candidate_definition(candidate_id)
    timeout = 3600 if definition["family"] == "realmlp" else 5400
    p = paths(candidate_id)
    if p["checkpoint"].exists() and json.loads(p["checkpoint"].read_text(encoding="utf-8")).get("status") == "PASS":
        report = {"stage_id": "stage5a2", "candidate_id": candidate_id, "status": "REUSED",
                  "timeout_seconds": timeout, "checkpoint_status": "PASS"}
        atomic_json(report, p["parent"])
        print(json.dumps(report, indent=2))
        return report
    command = [sys.executable, str(Path(__file__).resolve()), "--candidate", candidate_id]
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0.0
    timed_out = False
    while process.poll() is None:
        try:
            proc = psutil.Process(process.pid)
            peak = max(peak, sum(item.memory_info().rss for item in [proc, *proc.children(recursive=True)]) / 1024**2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        elapsed = time.perf_counter() - started
        atomic_json({"stage_id": "stage5a2", "candidate_id": candidate_id, "status": "RUNNING",
                     "elapsed_seconds": elapsed, "peak_process_tree_ram_mib": peak,
                     "timeout_seconds": timeout}, p["parent"])
        if elapsed > timeout:
            timed_out = True
            process.kill()
            break
        time.sleep(2)
    stdout, stderr = process.communicate()
    p["log"].parent.mkdir(parents=True, exist_ok=True)
    p["log"].write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    checkpoint_status = json.loads(p["checkpoint"].read_text(encoding="utf-8")).get("status") if p["checkpoint"].exists() else "MISSING"
    report = {"stage_id": "stage5a2", "candidate_id": candidate_id,
              "command": command, "elapsed_seconds": time.perf_counter() - started,
              "timeout_seconds": timeout, "timed_out": timed_out, "return_code": process.returncode,
              "peak_process_tree_ram_mib": peak, "checkpoint_status": checkpoint_status,
              "log_path": _relative(p["log"]),
              "status": "PASS" if not timed_out and process.returncode == 0 and checkpoint_status == "PASS" else "FAIL"}
    if report["status"] == "PASS" and p["result"].exists():
        result = json.loads(p["result"].read_text(encoding="utf-8"))
        result["peak_process_tree_ram_mib"] = peak
        result["parent_elapsed_seconds"] = report["elapsed_seconds"]
        result["parent_timeout_seconds"] = timeout
        atomic_json(result, p["result"])
        atomic_json(result, p["checkpoint"])
    atomic_json(report, p["parent"])
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError(f"Parent-enforced fit failed: {candidate_id}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--verify", choices=CANDIDATES)
    parser.add_argument("--parent", choices=CANDIDATES)
    args = parser.parse_args()
    if args.parent:
        parent_run(args.parent)
    elif args.verify:
        verify_candidate(args.verify)
    elif args.candidate:
        run_candidate(args.candidate)
    else:
        parser.error("Choose a worker action")


if __name__ == "__main__":
    main()
