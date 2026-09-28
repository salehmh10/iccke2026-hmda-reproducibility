"""Clean-process frozen RealMLP Test inference for Stage 5C.

This worker loads one exact Test Feature frame at the CSV parser boundary,
uses one frozen bundle, and writes one immutable prediction artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil

from stage5_safe_row_loader import load_allowed_source_rows


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/reports/stage5c_matplotlib_cache"))
FREEZE = ROOT / "artifacts/reports/stage5c_preevaluation_freeze.json"
SENTINEL = ROOT / "artifacts/reports/stage5c_safe_loader_sentinel.json"
COMPARATOR_VALIDATION = ROOT / "artifacts/reports/stage5c_stage4l_primary_validation.json"
TEST_IDS = ROOT / "artifacts/splits/test_row_ids.csv"
TRAIN_IDS = ROOT / "artifacts/splits/train_row_ids.csv"
COMPARATOR = ROOT / "artifacts/predictions/final_test/stage4l__blend__without_sensitive.csv"
PRED_DIR = ROOT / "artifacts/predictions/stage5/posttest_evaluation"
REPORT_DIR = ROOT / "artifacts/reports"
MANIFEST_DIR = ROOT / "artifacts/manifests/stage5"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(values: np.ndarray, dtype: Any) -> str:
    array = np.ascontiguousarray(values, dtype=dtype)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def frame_hash(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(np.uint64)
    return array_hash(values, np.uint64)


class MemoryMonitor:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.peak_bytes = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self.stop_event.is_set():
            try:
                processes = [process] + process.children(recursive=True)
                total = sum(item.memory_info().rss for item in processes if item.is_running())
                self.peak_bytes = max(self.peak_bytes, total)
            except (psutil.Error, OSError):
                pass
            self.stop_event.wait(0.05)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        return self.peak_bytes / (1024.0 * 1024.0)


def mode_contract(freeze: dict[str, Any], mode: str) -> dict[str, Any]:
    contract = freeze["deep_modes"][mode]
    return {
        **contract,
        "bundle_path": ROOT / contract["bundle_path"],
        "model_path": ROOT / contract["model_path"],
        "source_path": ROOT / contract["source_path"],
        "output_path": ROOT / contract["prediction_path"],
        "manifest_path": ROOT / contract["prediction_manifest_path"],
        "epoch_proof_path": ROOT / contract["epoch_proof_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["without_sensitive", "with_sensitive"], required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    started_at = now_utc()
    report_path = REPORT_DIR / f"stage5c_prediction_attempt_{args.attempt_id}.json"
    monitor = MemoryMonitor()
    monitor.start()
    report: dict[str, Any] = {
        "attempt_id": args.attempt_id,
        "mode": args.mode,
        "started_at_utc": started_at,
        "status": "RUNNING",
        "model_fit_calls": 0,
        "preprocessing_fit_calls": 0,
        "prediction_calls": 0,
        "parent_timeout_seconds": 1200,
        "evaluation_label": "Post-Test Extension",
    }
    atomic_json(report, report_path)

    try:
        if sha256_file(FREEZE) != args.expected_freeze_sha256:
            raise RuntimeError("Stage 5C pre-evaluation freeze hash mismatch")
        freeze = load_json(FREEZE)
        if freeze.get("status") != "PASS" or freeze.get("ensemble_candidate_count") != 0:
            raise RuntimeError("Stage 5C freeze is not eligible")
        sentinel = load_json(SENTINEL)
        if sentinel.get("status") != "PASS":
            raise RuntimeError("Safe-loader sentinel has not passed")
        comparator_validation = load_json(COMPARATOR_VALIDATION)
        if comparator_validation.get("status") != "PASS":
            raise RuntimeError("Stage 4L official comparator validation has not passed")

        contract = mode_contract(freeze, args.mode)
        for key in ("bundle_path", "model_path", "source_path"):
            if not contract[key].exists():
                raise FileNotFoundError(contract[key])
        if contract["output_path"].exists() or contract["manifest_path"].exists():
            raise RuntimeError("A Stage 5C prediction artifact already exists; regeneration is prohibited")
        if sha256_file(contract["bundle_path"]) != contract["bundle_sha256"]:
            raise RuntimeError("Frozen bundle hash mismatch")
        if sha256_file(contract["model_path"]) != contract["model_sha256"]:
            raise RuntimeError("Underlying model hash mismatch")
        if sha256_file(contract["source_path"]) != contract["source_sha256"]:
            raise RuntimeError("Source hash mismatch")
        if sha256_file(TEST_IDS) != freeze["test_id_file_sha256"]:
            raise RuntimeError("Saved Test-ID file hash mismatch")

        test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
        train_ids = pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
        if len(test_ids) != freeze["expected_test_row_count"] or len(np.unique(test_ids)) != len(test_ids):
            raise RuntimeError("Saved Test membership is invalid")
        if np.intersect1d(test_ids, train_ids).size:
            raise RuntimeError("Saved Train/Test overlap is nonzero")
        ordered_ids = np.sort(test_ids)
        if array_hash(ordered_ids, np.int64) != freeze["sorted_test_row_id_hash"]:
            raise RuntimeError("Sorted Test row-ID hash mismatch")

        first_source_access = now_utc()
        raw = load_allowed_source_rows(
            contract["source_path"],
            ordered_ids,
            contract["required_feature_columns"],
            allowed_train_ids=set(int(value) for value in test_ids),
        )
        report.update({
            "first_source_feature_access_at_utc": first_source_access,
            "test_feature_rows_materialized": len(raw),
            "train_feature_rows_materialized": 0,
            "excluded_rows_converted": 0,
            "source_target_values_materialized": 0,
        })
        atomic_json(report, report_path)
        if list(raw.columns) != contract["required_feature_columns"]:
            raise RuntimeError("Loaded Feature schema mismatch")
        if not np.array_equal(raw.index.to_numpy(np.int64), ordered_ids):
            raise RuntimeError("Loaded Test Feature rows are misaligned")
        raw_before = frame_hash(raw)
        feature_column_hashes = {
            column: array_hash(
                pd.util.hash_pandas_object(raw[column], index=True).to_numpy(np.uint64),
                np.uint64,
            )
            for column in raw.columns
        }

        bundle_load_at = now_utc()
        bundle = joblib.load(contract["bundle_path"])
        epoch_proof = load_json(contract["epoch_proof_path"])
        expected_features = bundle["numerical_features"] + bundle["categorical_features"]
        checks = {
            "family_realmlp": bundle["family"] == "realmlp",
            "target_mode_raw": bundle["target_mode"] == "raw",
            "fixed_epoch_30": int(bundle["fixed_epoch"]) == 30,
            "device_cpu": bundle["device"] == "cpu",
            "precision_float32": bundle["precision"] == "float32",
            "sensitive_mode_match": bundle["sensitive_mode"] == args.mode,
            "features_exact": expected_features == contract["required_feature_columns"],
            "row_id_not_feature": "row_id" not in expected_features,
            "target_not_feature": freeze["source_target_column"] not in expected_features,
            "training_rows_exact": int(bundle["training_row_count"]) == 399788,
            "test_rows_in_fit_zero": int(bundle["test_rows"]) == 0,
            "early_stopping_false": epoch_proof["early_stopping"] is False,
            "best_checkpoint_restoration_false": epoch_proof["best_checkpoint_restoration"] is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Frozen bundle contract failed: {[k for k,v in checks.items() if not v]}")

        transformed = bundle["preprocessor"].transform(raw)
        prediction_at = now_utc()
        report["prediction_calls"] = 1
        standardized = np.asarray(bundle["model"].predict(transformed)).reshape(-1)
        prediction = np.asarray(
            bundle["target_transform"].inverse(standardized, standardized=True),
            dtype=np.float64,
        ).reshape(-1)
        if len(prediction) != len(ordered_ids) or not np.isfinite(prediction).all():
            raise RuntimeError("Deep prediction is invalid")
        if frame_hash(raw) != raw_before:
            raise RuntimeError("Source Feature frame was modified in place")

        comparator = pd.read_csv(COMPARATOR, usecols=["row_id", "y_true"])
        if len(comparator) != len(ordered_ids) or comparator["row_id"].duplicated().any():
            raise RuntimeError("Canonical Stage 4L target artifact is invalid")
        canonical = comparator.set_index("row_id").loc[ordered_ids, "y_true"].to_numpy(np.float64)
        if not np.isfinite(canonical).all():
            raise RuntimeError("Canonical target contains non-finite values")
        target_hash = array_hash(canonical, np.float64)
        if target_hash != comparator_validation["canonical_target_hash"]:
            raise RuntimeError("Canonical target hash mismatch")

        decile = np.asarray(
            pd.qcut(canonical, q=10, labels=False, duplicates="drop"), dtype=np.int64
        ) + 1
        top_five_threshold = float(np.quantile(canonical, 0.95))
        signed = prediction - canonical
        output = pd.DataFrame({
            "row_id": ordered_ids,
            "y_true": canonical,
            "y_pred": prediction,
            "absolute_error": np.abs(signed),
            "signed_error": signed,
            "target_decile": decile,
            "is_top_five_percent_target": canonical >= top_five_threshold,
            "sensitive_mode": args.mode,
            "candidate_id": contract["evaluation_candidate_id"],
            "bundle_candidate_id": contract["bundle_candidate_id"],
            "model_family": "RealMLP",
            "target_mode": "raw",
            "frozen_epoch": 30,
            "evaluation_label": "Post-Test Extension",
            "official_result_role": contract["official_result_role"],
            "bundle_sha256": contract["bundle_sha256"],
            "model_sha256": contract["model_sha256"],
            "source_sha256": contract["source_sha256"],
            "test_row_id_hash": freeze["sorted_test_row_id_hash"],
            "target_hash": target_hash,
        })
        atomic_csv(output, contract["output_path"])
        prediction_sha256 = sha256_file(contract["output_path"])

        peak_mib = monitor.stop()
        manifest = {
            "stage_id": "stage5c",
            "status": "PASS",
            "evaluation_label": "Post-Test Extension",
            "attempt_id": args.attempt_id,
            "attempt_lineage": [args.attempt_id],
            "sensitive_mode": args.mode,
            "candidate_id": contract["evaluation_candidate_id"],
            "bundle_candidate_id": contract["bundle_candidate_id"],
            "bundle_path": str(contract["bundle_path"].relative_to(ROOT)).replace("\\", "/"),
            "bundle_sha256": contract["bundle_sha256"],
            "model_path": str(contract["model_path"].relative_to(ROOT)).replace("\\", "/"),
            "model_sha256": contract["model_sha256"],
            "input_schema_hash": contract["feature_schema_sha256"],
            "source_path": str(contract["source_path"].relative_to(ROOT)).replace("\\", "/"),
            "source_sha256": contract["source_sha256"],
            "test_id_file_sha256": freeze["test_id_file_sha256"],
            "test_row_id_hash": freeze["sorted_test_row_id_hash"],
            "target_hash": target_hash,
            "prediction_path": str(contract["output_path"].relative_to(ROOT)).replace("\\", "/"),
            "prediction_sha256": prediction_sha256,
            "prediction_value_hash": array_hash(prediction, np.float64),
            "row_count": len(output),
            "runtime_seconds": time.perf_counter() - started,
            "peak_process_tree_ram_mib": peak_mib,
            "model_fit_calls": 0,
            "preprocessing_fit_calls": 0,
            "prediction_call_count": 1,
            "physical_attempt_count": 1,
            "source_scan_data_lines": 499736,
            "test_feature_rows_materialized": len(raw),
            "train_feature_rows_materialized": 0,
            "excluded_rows_converted": 0,
            "source_target_values_materialized": 0,
            "source_target_columns_requested": 0,
            "safe_loader_path": "stage5_safe_row_loader.py",
            "safe_loader_sha256": freeze["safe_loader_sha256"],
            "safe_loader_sentinel_status": "PASS",
            "source_frame_unchanged": True,
            "source_frame_hash": raw_before,
            "feature_column_hashes": feature_column_hashes,
            "predictions_finite": True,
            "predictions_original_scale": True,
            "bundle_checks": checks,
            "first_source_feature_access_at_utc": first_source_access,
            "bundle_loaded_at_utc": bundle_load_at,
            "prediction_started_at_utc": prediction_at,
            "completed_at_utc": now_utc(),
        }
        atomic_json(manifest, contract["manifest_path"])
        report.update({
            "status": "PASS",
            "completed_at_utc": now_utc(),
            "runtime_seconds": manifest["runtime_seconds"],
            "peak_process_tree_ram_mib": peak_mib,
            "prediction_calls": 1,
            "prediction_path": manifest["prediction_path"],
            "prediction_sha256": prediction_sha256,
            "manifest_path": str(contract["manifest_path"].relative_to(ROOT)).replace("\\", "/"),
            "test_feature_rows_materialized": len(raw),
            "train_feature_rows_materialized": 0,
            "excluded_rows_converted": 0,
            "source_target_values_materialized": 0,
        })
        atomic_json(report, report_path)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        peak_mib = monitor.stop()
        report.update({
            "status": "FAIL",
            "completed_at_utc": now_utc(),
            "runtime_seconds": time.perf_counter() - started,
            "peak_process_tree_ram_mib": peak_mib,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        atomic_json(report, report_path)
        raise


if __name__ == "__main__":
    main()
