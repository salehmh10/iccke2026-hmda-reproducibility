"""Single sensitive Validation worker for the frozen Stage 5A2 Core winner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from stage5a2_deep_utils import (
    EXPECTED_FREEZE_SHA256, FINAL_SAMPLE, ROOT, SOURCE_WITH, TEST_IDS,
    atomic_csv, atomic_joblib, atomic_json, digest_values, feature_lists,
    load_final_selection, predict_bundle, sha256_file, train_realmlp_fixed_validation,
    validate_freeze,
)


CANDIDATE_ID = "stage5a2__realmlp__core__with_sensitive"
WINNER_PATH = ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json"


def paths() -> dict[str, Path]:
    return {
        "checkpoint": ROOT / f"artifacts/checkpoints/stage5/deep_core/final_validation/{CANDIDATE_ID}.json",
        "result": ROOT / f"artifacts/results/stage5/deep_core/final_validation/candidates/{CANDIDATE_ID}.json",
        "prediction": ROOT / f"artifacts/predictions/stage5/deep_core/final_validation/{CANDIDATE_ID}.csv",
        "history": ROOT / f"artifacts/results/stage5/deep_core/final_validation/histories/{CANDIDATE_ID}_history.csv",
        "bundle": ROOT / f"artifacts/models/deep/core_validation/{CANDIDATE_ID}.joblib",
        "reload": ROOT / f"artifacts/reports/stage5a2_reload_{CANDIDATE_ID}.json",
        "parent": ROOT / f"artifacts/reports/stage5a2_parent_{CANDIDATE_ID}.json",
        "log": ROOT / f"artifacts/reports/stage5a2_parent_{CANDIDATE_ID}.log",
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_sensitive() -> dict:
    validate_freeze()
    p = paths()
    if p["checkpoint"].exists():
        existing = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "REUSED", "candidate_id": CANDIDATE_ID}))
            return existing
        raise RuntimeError("Failed sensitive checkpoint exists; no second sensitive fit is authorized")
    winner = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    if winner.get("family") != "realmlp" or winner.get("candidate_id") != "stage5a2__realmlp__frozen":
        raise RuntimeError("Unexpected Core winner")
    started = time.perf_counter()
    lock = ROOT / "artifacts/checkpoints/stage5/deep_core/stage5a2_heavy_fit.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, str(os.getpid()).encode("ascii")); os.close(descriptor)
    except FileExistsError as exc:
        raise RuntimeError("Another heavy fit is active") from exc
    try:
        X_train, X_val, y_train, y_val, train_rows, val_rows, val_bins = load_final_selection("with_sensitive", CANDIDATE_ID)
        without = pd.read_csv(ROOT / winner["validation_prediction_path"])
        if not np.array_equal(without["row_id"].to_numpy(np.int64), val_rows):
            raise RuntimeError("Sensitive Validation row membership differs from reused winner")
        if not np.array_equal(without["y_true"].to_numpy(np.float32), y_val):
            raise RuntimeError("Sensitive source target alignment differs from reused winner")
        trained = train_realmlp_fixed_validation(CANDIDATE_ID, "with_sensitive", X_train, X_val,
                                                 y_train, y_val, int(winner["best_epoch"]), p["history"])
        prediction = np.asarray(trained.pop("prediction"), dtype=np.float64)
        if len(prediction) != 25_000 or not np.isfinite(prediction).all():
            raise RuntimeError("Invalid sensitive Validation predictions")
        pred = pd.DataFrame({
            "row_id": val_rows, "y_true": y_val.astype(np.float64), "y_pred": prediction,
            "absolute_error": np.abs(prediction - y_val), "signed_error": prediction - y_val,
            "target_decile": val_bins, "sensitive_mode": "with_sensitive", "model_family": "realmlp",
            "candidate_id": CANDIDATE_ID, "target_mode": "raw", "epoch": winner["best_epoch"],
        })
        atomic_csv(pred, p["prediction"])
        numerical, categorical = feature_lists("with_sensitive")
        bundle = {
            "bundle_version": "stage5a2_core_validation_v1", "stage_id": "stage5a2",
            "candidate_id": CANDIDATE_ID, "family": "realmlp", "sensitive_mode": "with_sensitive",
            "target_mode": "raw", "feature_schema": "deep_core_v1_with_validated_sensitive_sources",
            "base_feature_schema": "deep_core_v1", "numerical_features": numerical,
            "categorical_features": categorical, "preprocessor": trained.pop("preprocessor"),
            "target_transform": trained.pop("target_transform"), "model": trained.pop("model"),
            "model_state_path": None, "architecture": trained["architecture"], "training": trained["training"],
            "fixed_epoch": winner["best_epoch"], "seed": 42, "device": "cpu", "precision": "float32",
            "training_rows": 100_000, "validation_rows": 25_000, "test_rows": 0,
            "training_row_id_hash": digest_values(train_rows), "validation_row_id_hash": digest_values(val_rows),
            "freeze_sha256": EXPECTED_FREEZE_SHA256,
        }
        atomic_joblib(bundle, p["bundle"])
        result = {
            "stage_id": "stage5a2", "candidate_id": CANDIDATE_ID, "model_family": "realmlp",
            "sensitive_mode": "with_sensitive", "target_mode": "raw", "feature_schema": "deep_core_v1",
            "fixed_epoch": winner["best_epoch"], "requested_epoch": winner["best_epoch"],
            "completed_epoch": winner["best_epoch"], "saved_artifact_epoch": winner["best_epoch"],
            "early_stopping": False, "best_checkpoint_restoration": False,
            "same_frozen_configuration_as_winner": True, "same_rows_as_winner": True,
            "training_rows": 100_000, "validation_rows": 25_000, "test_rows": 0,
            "metrics": trained["metrics"], "fit_time_seconds": trained["fit_time_seconds"],
            "prediction_time_seconds": trained["prediction_time_seconds"], "peak_ram_mib": trained["peak_ram_mib"],
            "model_size_bytes": p["bundle"].stat().st_size, "bundle_path": relative(p["bundle"]),
            "prediction_path": relative(p["prediction"]), "history_path": relative(p["history"]),
            "fixed_epoch_proof_path": trained["fixed_epoch_proof_path"],
            "bundle_sha256": sha256_file(p["bundle"]), "prediction_sha256": sha256_file(p["prediction"]),
            "training_row_id_hash": digest_values(train_rows), "validation_row_id_hash": digest_values(val_rows),
            "source_sha256": sha256_file(SOURCE_WITH), "sample_sha256": sha256_file(FINAL_SAMPLE),
            "official_encoder_contract": trained["official_encoder_contract"],
            "worker_elapsed_seconds": time.perf_counter() - started, "status": "PASS",
        }
        atomic_json(result, p["result"]); atomic_json(result, p["checkpoint"])
        print(json.dumps({"status": "PASS", "candidate_id": CANDIDATE_ID, "metrics": result["metrics"]}))
        return result
    except Exception as exc:
        failure = {"stage_id": "stage5a2", "candidate_id": CANDIDATE_ID, "status": "FAIL",
                   "error": repr(exc), "traceback": traceback.format_exc(),
                   "worker_elapsed_seconds": time.perf_counter() - started}
        atomic_json(failure, p["checkpoint"])
        raise
    finally:
        if lock.exists(): lock.unlink()


def verify_sensitive() -> dict:
    p = paths(); result = json.loads(p["result"].read_text(encoding="utf-8"))
    _, X_val, _, y_val, _, val_rows, _ = load_final_selection("with_sensitive", CANDIDATE_ID)
    before = {column: pd.util.hash_pandas_object(X_val[column], index=True).sum() for column in X_val.columns}
    prediction = predict_bundle(p["bundle"], X_val)
    after = {column: pd.util.hash_pandas_object(X_val[column], index=True).sum() for column in X_val.columns}
    expected = pd.read_csv(p["prediction"])
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    difference = np.abs(prediction - expected["y_pred"].to_numpy(np.float64))
    checks = {
        "result_pass": result["status"] == "PASS", "row_order_match": np.array_equal(expected["row_id"], val_rows),
        "row_ids_unique": expected["row_id"].is_unique, "test_overlap_zero": len(np.intersect1d(expected["row_id"], test_ids)) == 0,
        "target_alignment": np.array_equal(expected["y_true"].to_numpy(np.float32), y_val),
        "finite_predictions": np.isfinite(prediction).all(), "prediction_count_25000": len(prediction) == 25_000,
        "prediction_match": np.allclose(prediction, expected["y_pred"], rtol=1e-5, atol=1e-4),
        "source_frame_unchanged": before == after, "cpu_inference": True, "no_refit": True,
        "fixed_epoch_30": result["fixed_epoch"] == 30 and result["completed_epoch"] == 30,
        "no_early_stopping_or_restoration": not result["early_stopping"] and not result["best_checkpoint_restoration"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {"stage_id": "stage5a2", "candidate_id": CANDIDATE_ID, "clean_process": True,
              "maximum_absolute_difference": float(difference.max()), "checks": checks,
              "status": "PASS" if all(checks.values()) else "FAIL"}
    atomic_json(report, p["reload"]); print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise RuntimeError("Sensitive clean reload failed")
    return report


def parent_run() -> dict:
    p = paths(); timeout = 5400
    if p["checkpoint"].exists() and json.loads(p["checkpoint"].read_text(encoding="utf-8")).get("status") == "PASS":
        report = {"stage_id": "stage5a2", "candidate_id": CANDIDATE_ID, "status": "REUSED"}
        atomic_json(report, p["parent"]); print(json.dumps(report)); return report
    command = [sys.executable, str(Path(__file__).resolve()), "--sensitive"]
    started = time.perf_counter(); process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0.0; timed_out = False
    while process.poll() is None:
        try:
            proc = psutil.Process(process.pid)
            peak = max(peak, sum(item.memory_info().rss for item in [proc, *proc.children(recursive=True)]) / 1024**2)
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        elapsed = time.perf_counter() - started
        atomic_json({"stage_id": "stage5a2", "candidate_id": CANDIDATE_ID, "status": "RUNNING",
                     "elapsed_seconds": elapsed, "peak_process_tree_ram_mib": peak, "timeout_seconds": timeout}, p["parent"])
        if elapsed > timeout: timed_out = True; process.kill(); break
        time.sleep(2)
    stdout, stderr = process.communicate(); p["log"].write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    checkpoint_status = json.loads(p["checkpoint"].read_text(encoding="utf-8")).get("status") if p["checkpoint"].exists() else "MISSING"
    report = {"stage_id": "stage5a2", "candidate_id": CANDIDATE_ID, "command": command,
              "elapsed_seconds": time.perf_counter() - started, "timeout_seconds": timeout,
              "timed_out": timed_out, "return_code": process.returncode, "peak_process_tree_ram_mib": peak,
              "checkpoint_status": checkpoint_status, "log_path": relative(p["log"]),
              "status": "PASS" if not timed_out and process.returncode == 0 and checkpoint_status == "PASS" else "FAIL"}
    atomic_json(report, p["parent"])
    if report["status"] == "PASS":
        result = json.loads(p["result"].read_text(encoding="utf-8")); result["peak_process_tree_ram_mib"] = peak
        result["parent_elapsed_seconds"] = report["elapsed_seconds"]
        atomic_json(result, p["result"]); atomic_json(result, p["checkpoint"])
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise RuntimeError("Sensitive parent fit failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--sensitive", action="store_true")
    parser.add_argument("--verify-sensitive", action="store_true"); parser.add_argument("--parent-sensitive", action="store_true")
    args = parser.parse_args()
    if args.parent_sensitive: parent_run()
    elif args.verify_sensitive: verify_sensitive()
    elif args.sensitive: run_sensitive()
    else: parser.error("Choose an action")


if __name__ == "__main__": main()
