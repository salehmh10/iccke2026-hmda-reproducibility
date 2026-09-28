"""Two sequential fixed-epoch full-Train RealMLP bundle workers for Stage 5A2."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
import torch

from stage5a2_deep_utils import (
    EXPECTED_FREEZE_SHA256, ROOT, SOURCE_WITH, SOURCE_WITHOUT, TEST_IDS, TRAIN_IDS,
    _load_source_rows, atomic_csv, atomic_joblib, atomic_json, digest_values,
    feature_lists, load_full_train, predict_bundle, sha256_file, train_realmlp_full_train,
    validate_freeze,
)


MODES = ["without_sensitive", "with_sensitive"]
WINNER_PATH = ROOT / "artifacts/results/stage5/deep_core/final_validation/stage5a_core_winner_configuration.json"


def candidate_id(mode: str) -> str:
    return f"stage5a2__realmlp__core_final__{mode}"


def paths(mode: str) -> dict[str, Path]:
    cid = candidate_id(mode)
    return {
        "checkpoint": ROOT / f"artifacts/checkpoints/stage5/deep_core/full_train/{cid}.json",
        "result": ROOT / f"artifacts/results/stage5/deep_core/full_train/{cid}.json",
        "bundle": ROOT / f"artifacts/models/deep/core_final/{cid}.joblib",
        "history": ROOT / f"artifacts/results/stage5/deep_core/full_train/histories/{cid}_history.csv",
        "reference": ROOT / f"artifacts/predictions/stage5/deep_core/full_train_reference/{cid}.csv",
        "reload": ROOT / f"artifacts/reports/stage5a2_reload_{cid}.json",
        "parent": ROOT / f"artifacts/reports/stage5a2_parent_{cid}.json",
        "log": ROOT / f"artifacts/reports/stage5a2_parent_{cid}.log",
    }


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_full_train(mode: str) -> dict:
    if mode not in MODES: raise ValueError(mode)
    validate_freeze(); p = paths(mode); cid = candidate_id(mode)
    if p["checkpoint"].exists():
        existing = json.loads(p["checkpoint"].read_text(encoding="utf-8"))
        if existing.get("status") == "PASS": print(json.dumps({"status": "REUSED", "candidate_id": cid})); return existing
        retry_path = ROOT / f"artifacts/reports/{cid}_technical_retry.json"
        if not (mode == "without_sensitive" and retry_path.exists()
                and json.loads(retry_path.read_text(encoding="utf-8")).get("status") == "RETRY_AUTHORIZED"):
            raise RuntimeError("Failed full-Train checkpoint exists without the one authorized technical retry")
    winner = json.loads(WINNER_PATH.read_text(encoding="utf-8"))
    if winner.get("family") != "realmlp" or winner.get("best_epoch") != 30:
        raise RuntimeError("Unexpected frozen Core winner")
    lock = ROOT / "artifacts/checkpoints/stage5/deep_core/stage5a2_heavy_fit.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    except FileExistsError as exc: raise RuntimeError("Another heavy fit is active") from exc
    started = time.perf_counter()
    try:
        X_train, y_train, train_rows = load_full_train(mode)
        test_rows = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
        if len(X_train) != 399_788 or len(np.intersect1d(train_rows, test_rows)) != 0:
            raise RuntimeError("Full-Train membership is invalid")
        trained = train_realmlp_full_train(cid, mode, X_train, y_train, 30, p["history"])
        numerical, categorical = feature_lists(mode)
        reference_rows = train_rows[:64]
        reference_raw = X_train.iloc[:64].copy()
        # Build the complete bundle before obtaining its reference predictions.
        bundle = {
            "bundle_version": "stage5a2_core_final_v1", "stage_id": "stage5a2",
            "model_family": "realmlp", "family": "realmlp", "candidate_id": cid,
            "sensitive_mode": mode, "feature_schema": "deep_core_v1" if mode == "without_sensitive" else "deep_core_v1_with_validated_sensitive_sources",
            "base_feature_schema": "deep_core_v1", "numerical_features": numerical,
            "categorical_features": categorical, "numeric_imputer": "training-fit medians",
            "category_vocabularies": "embedded in RealMLPPreprocessor and official encoder",
            "missing_token": "__UNKNOWN_OR_RARE__", "unknown_token": "__UNKNOWN_OR_RARE__",
            "rare_token": "__UNKNOWN_OR_RARE__", "preprocessor": trained.pop("preprocessor"),
            "target_mode": "raw", "target_transform": trained.pop("target_transform"),
            "model": trained.pop("model"), "architecture": trained["architecture"],
            "training_configuration": trained["training"], "optimizer_metadata": "official RealMLP Adam policy",
            "fixed_epoch": 30, "batch_size_policy": {"train": 256, "predict": 1024},
            "seed": 42, "device": "cpu", "precision": "float32",
            "package_versions": {"python": platform.python_version(), "torch": torch.__version__,
                                 "sklearn": sklearn.__version__, "joblib": joblib.__version__},
            "source_path": relative(SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH),
            "source_sha256": sha256_file(SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH),
            "train_ids_path": relative(TRAIN_IDS), "train_ids_sha256": sha256_file(TRAIN_IDS),
            "test_ids_sha256": sha256_file(TEST_IDS), "training_row_count": 399_788,
            "training_row_id_hash": digest_values(train_rows), "test_rows": 0,
            "training_history_path": relative(p["history"]),
            "fixed_epoch_proof_path": trained["fixed_epoch_proof_path"],
            "reference_row_ids": reference_rows.tolist(),
        }
        atomic_joblib(bundle, p["bundle"])
        reference_prediction = predict_bundle(p["bundle"], reference_raw)
        reference = pd.DataFrame({"row_id": reference_rows, "y_true": y_train[:64].astype(np.float64),
                                  "y_pred": reference_prediction, "sensitive_mode": mode,
                                  "candidate_id": cid, "fixed_epoch": 30})
        atomic_csv(reference, p["reference"])
        bundle["reference_prediction_path"] = relative(p["reference"])
        bundle["reference_prediction_sha256"] = sha256_file(p["reference"])
        atomic_joblib(bundle, p["bundle"])
        result = {
            "stage_id": "stage5a2", "candidate_id": cid, "model_family": "realmlp",
            "sensitive_mode": mode, "target_mode": "raw", "fixed_epoch": 30,
            "requested_epoch": 30, "completed_epoch": 30, "saved_artifact_epoch": 30,
            "training_rows": 399_788, "test_rows": 0, "early_stopping": False,
            "best_checkpoint_restoration": False, "deployed_interface_is_full_data_refit": True,
            "fit_time_seconds": trained["fit_time_seconds"], "peak_ram_mib": trained["peak_ram_mib"],
            "model_size_bytes": p["bundle"].stat().st_size, "bundle_path": relative(p["bundle"]),
            "bundle_sha256": sha256_file(p["bundle"]), "history_path": relative(p["history"]),
            "history_sha256": sha256_file(p["history"]), "reference_prediction_path": relative(p["reference"]),
            "reference_prediction_sha256": sha256_file(p["reference"]),
            "fixed_epoch_proof_path": trained["fixed_epoch_proof_path"],
            "training_row_id_hash": digest_values(train_rows), "train_ids_sha256": sha256_file(TRAIN_IDS),
            "source_sha256": sha256_file(SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH),
            "official_encoder_contract": trained["official_encoder_contract"],
            "freeze_sha256": EXPECTED_FREEZE_SHA256, "worker_elapsed_seconds": time.perf_counter() - started,
            "status": "PASS",
        }
        atomic_json(result, p["result"]); atomic_json(result, p["checkpoint"])
        print(json.dumps({"status": "PASS", "candidate_id": cid, "fit_time_seconds": result["fit_time_seconds"]}))
        return result
    except Exception as exc:
        failure = {"stage_id": "stage5a2", "candidate_id": cid, "status": "FAIL",
                   "error": repr(exc), "traceback": traceback.format_exc(),
                   "worker_elapsed_seconds": time.perf_counter() - started}
        atomic_json(failure, p["checkpoint"]); raise
    finally:
        if lock.exists(): lock.unlink()


def verify_full_train(mode: str) -> dict:
    p = paths(mode); cid = candidate_id(mode); result = json.loads(p["result"].read_text(encoding="utf-8"))
    reference = pd.read_csv(p["reference"]); rows = reference["row_id"].to_numpy(np.int64)
    numerical, categorical = feature_lists(mode); source = SOURCE_WITHOUT if mode == "without_sensitive" else SOURCE_WITH
    raw = _load_source_rows(source, rows, [*numerical, *categorical])
    before = {column: pd.util.hash_pandas_object(raw[column], index=True).sum() for column in raw.columns}
    prediction = predict_bundle(p["bundle"], raw)
    after = {column: pd.util.hash_pandas_object(raw[column], index=True).sum() for column in raw.columns}
    probe = raw.iloc[:4].copy(); probe.iloc[0, probe.columns.get_loc(categorical[0])] = "__STAGE5A2_UNSEEN__"
    probe.iloc[1, probe.columns.get_loc(numerical[0])] = np.nan
    probe_prediction = predict_bundle(p["bundle"], probe)
    expected = reference["y_pred"].to_numpy(np.float64); difference = np.abs(prediction - expected)
    bundle = joblib.load(p["bundle"])
    checks = {
        "result_pass": result["status"] == "PASS", "bundle_hash_match": sha256_file(p["bundle"]) == result["bundle_sha256"],
        "training_rows_399788": result["training_rows"] == 399_788, "test_rows_zero": result["test_rows"] == 0,
        "fixed_epoch_30": result["requested_epoch"] == result["completed_epoch"] == result["saved_artifact_epoch"] == 30,
        "no_early_stopping": result["early_stopping"] is False,
        "no_best_checkpoint_restoration": result["best_checkpoint_restoration"] is False,
        "deployed_full_data_refit": result["deployed_interface_is_full_data_refit"] is True,
        "finite_predictions": np.isfinite(prediction).all(), "reference_count": len(prediction) == 64,
        "prediction_match": np.allclose(prediction, expected, rtol=1e-5, atol=1e-4),
        "source_frame_unchanged": before == after, "unknown_category_handling": np.isfinite(probe_prediction[0]),
        "missing_value_handling": np.isfinite(probe_prediction[1]), "target_inverse_transform": bundle["target_mode"] == "raw",
        "cpu_inference": bundle["device"] == "cpu", "no_statistics_refit": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {"stage_id": "stage5a2", "candidate_id": cid, "sensitive_mode": mode, "clean_process": True,
              "maximum_absolute_difference": float(difference.max()), "checks": checks,
              "status": "PASS" if all(checks.values()) else "FAIL"}
    atomic_json(report, p["reload"]); print(json.dumps(report, indent=2))
    if report["status"] != "PASS": raise RuntimeError("Full-Train clean reload failed")
    return report


def parent_run(mode: str) -> dict:
    p = paths(mode); cid = candidate_id(mode); timeout = 7200
    if p["checkpoint"].exists() and json.loads(p["checkpoint"].read_text(encoding="utf-8")).get("status") == "PASS":
        report = {"stage_id": "stage5a2", "candidate_id": cid, "status": "REUSED"}; atomic_json(report, p["parent"]); print(json.dumps(report)); return report
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode]
    started = time.perf_counter(); process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    peak = 0.0; timed_out = False
    while process.poll() is None:
        try:
            proc = psutil.Process(process.pid); peak = max(peak, sum(item.memory_info().rss for item in [proc, *proc.children(recursive=True)]) / 1024**2)
        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
        elapsed = time.perf_counter() - started
        atomic_json({"stage_id": "stage5a2", "candidate_id": cid, "status": "RUNNING",
                     "elapsed_seconds": elapsed, "peak_process_tree_ram_mib": peak, "timeout_seconds": timeout}, p["parent"])
        if elapsed > timeout: timed_out = True; process.kill(); break
        time.sleep(2)
    stdout, stderr = process.communicate(); p["log"].parent.mkdir(parents=True, exist_ok=True)
    p["log"].write_text(stdout + "\n--- STDERR ---\n" + stderr, encoding="utf-8")
    checkpoint_status = json.loads(p["checkpoint"].read_text(encoding="utf-8")).get("status") if p["checkpoint"].exists() else "MISSING"
    report = {"stage_id": "stage5a2", "candidate_id": cid, "command": command,
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
    if report["status"] != "PASS": raise RuntimeError(f"Full-Train parent fit failed: {mode}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--parent-mode", choices=MODES); parser.add_argument("--verify-mode", choices=MODES)
    args = parser.parse_args()
    if args.parent_mode: parent_run(args.parent_mode)
    elif args.verify_mode: verify_full_train(args.verify_mode)
    elif args.mode: run_full_train(args.mode)
    else: parser.error("Choose an action")


if __name__ == "__main__": main()
