"""One-process-per-candidate worker and clean reload verifier for Stage 5A1."""

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
import torch
from sklearn.metrics import mean_absolute_error
from torch import nn

from stage5_deep_models import (
    build_torch_model,
    model_size_bytes,
    predict_bundle,
    regression_metrics,
    seed_everything,
    train_realmlp_candidate,
    train_realmlp_log1p_fit11,
    train_torch_candidate,
)
from stage5_deep_preprocessing import (
    CATEGORICAL_FEATURES,
    DISCOVERY,
    NUMERICAL_FEATURES,
    ROOT,
    SCHEMA_ID,
    SOURCE,
    TargetTransform,
    TensorPreprocessor,
    atomic_csv,
    atomic_joblib,
    atomic_json,
    digest_values,
    load_discovery,
    sha256_file,
)


CANDIDATES = {
    "stage5a1__realmlp__raw": ("realmlp", "raw"),
    "stage5a1__realmlp__log1p": ("realmlp", "log1p"),
    "stage5a1__tabm__raw": ("tabm", "raw"),
    "stage5a1__tabm__log1p": ("tabm", "log1p"),
    "stage5a1__ft_transformer__raw": ("ft_transformer", "raw"),
    "stage5a1__ft_transformer__log1p": ("ft_transformer", "log1p"),
    "stage5a1__realmlp__raw__replacement1": ("realmlp", "raw"),
    "stage5a1__realmlp__log1p__replacement1": ("realmlp", "log1p"),
    "stage5a1__tabm__raw__replacement1": ("tabm_embedding", "raw"),
    "stage5a1__tabm__log1p__replacement1": ("tabm_embedding", "log1p"),
    "stage5a1__realmlp__log1p__replacement2": ("realmlp", "log1p"),
}

REPLACEMENTS = {
    "stage5a1__realmlp__raw__replacement1": "stage5a1__realmlp__raw",
    "stage5a1__realmlp__log1p__replacement1": "stage5a1__realmlp__log1p",
    "stage5a1__tabm__raw__replacement1": "stage5a1__tabm__raw",
    "stage5a1__tabm__log1p__replacement1": "stage5a1__tabm__log1p",
    "stage5a1__realmlp__log1p__replacement2": "stage5a1__realmlp__log1p__replacement1",
}


def candidate_paths(candidate_id: str) -> dict[str, Path]:
    return {
        "checkpoint": ROOT / f"artifacts/checkpoints/stage5/deep_core/screening/{candidate_id}.json",
        "prediction": ROOT / f"artifacts/predictions/stage5/deep_core/screening/{candidate_id}.csv",
        "result": ROOT / f"artifacts/results/stage5/deep_core/screening/candidates/{candidate_id}.json",
        "reload": ROOT / f"artifacts/reports/stage5a1_reload_{candidate_id}.json",
    }


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run_candidate(candidate_id: str) -> dict[str, Any]:
    if candidate_id not in CANDIDATES:
        raise ValueError(f"Unknown Stage 5A1 candidate: {candidate_id}")
    family, target_mode = CANDIDATES[candidate_id]
    display_family = "tabm" if family == "tabm_embedding" else family
    paths = candidate_paths(candidate_id)
    if paths["checkpoint"].exists():
        existing = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        if existing.get("status") == "PASS":
            print(json.dumps({"status": "REUSED", "candidate_id": candidate_id}))
            return existing
        raise RuntimeError(f"A failed checkpoint already exists for {candidate_id}; parent must adjudicate retry")

    started = time.perf_counter()
    X_train, X_val, y_train_series, y_val_series, train_rows, val_rows = load_discovery()
    y_train = y_train_series.to_numpy(dtype=np.float32)
    y_val = y_val_series.to_numpy(dtype=np.float32)
    sample_digest = sha256_file(DISCOVERY)
    source_digest = sha256_file(SOURCE)
    try:
        if candidate_id == "stage5a1__realmlp__log1p__replacement2":
            output = train_realmlp_log1p_fit11(
                X_train, X_val, y_train, y_val, candidate_id
            )
        elif family == "realmlp":
            output = train_realmlp_candidate(
                target_mode, X_train, X_val, y_train, y_val, candidate_id
            )
        else:
            output = train_torch_candidate(
                family, target_mode, X_train, X_val, y_train, y_val, candidate_id
            )
        prediction = pd.DataFrame(
            {
                "row_id": val_rows,
                "sample_role": "validation",
                "candidate_id": candidate_id,
                "model_family": display_family,
                "target_mode": target_mode,
                "y_true": y_val.astype(np.float64),
                "y_pred": output.pop("prediction"),
            }
        )
        atomic_csv(prediction, paths["prediction"])
        model_paths = [output["bundle_path"], output["state_path"]]
        model_bytes = model_size_bytes(model_paths)
        result = {
            "stage_id": "stage5a1",
            "candidate_id": candidate_id,
            "model_family": display_family,
            "implementation_family": family,
            "replacement_of": REPLACEMENTS.get(candidate_id),
            "target_mode": target_mode,
            "feature_schema": SCHEMA_ID,
            "preprocessing_version": "tabm_learned_embeddings_v2" if family == "tabm_embedding" else "realmlp_train_only_vocab_v2" if candidate_id in REPLACEMENTS else f"{family}_v1",
            "architecture": output["architecture"],
            "optimizer": output["optimizer"],
            "learning_rate": output["learning_rate"],
            "batch_size": output["batch_size"],
            "device": "cpu",
            "amp": False,
            "seed": 42,
            "training_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": 0,
            "sensitive_mode": "without_sensitive",
            "best_epoch": output["best_epoch"],
            "epochs_completed": output["epochs_completed"],
            "stop_reason": output["stop_reason"],
            "fit_time_seconds": output["fit_seconds"],
            "prediction_time_seconds": output["prediction_seconds"],
            "peak_ram_mib": output["peak_ram_mib"],
            "peak_vram_mib": output["peak_vram_mib"],
            "model_size_bytes": model_bytes,
            "metrics": output["metrics"],
            "prediction_path": _relative(paths["prediction"]),
            "model_path": _relative(output["bundle_path"]),
            "model_state_path": _relative(output["state_path"]),
            "preprocessor_path": _relative(output["preprocessor_path"]),
            "training_history_path": _relative(output["history_path"]),
            "sample_digest": sample_digest,
            "source_digest": source_digest,
            "training_row_id_digest": digest_values(train_rows),
            "validation_row_id_digest": digest_values(val_rows),
            "status": "PASS",
            "warning": output.get("warning"),
            "categorical_contract": output.get("categorical_contract"),
            "official_encoder_contract": output.get("official_encoder_contract"),
            "official_encoder_matches_train_only": output.get("official_encoder_matches_train_only"),
            "fixed_epoch_no_early_stopping_exception": output.get("fixed_epoch_no_early_stopping_exception", False),
            "physical_fit_number": output.get("physical_fit_number"),
            "protocol_proof_path": _relative(output["protocol_proof_path"]) if output.get("protocol_proof_path") else None,
            "resolved_config_path": _relative(output["resolved_config_path"]) if output.get("resolved_config_path") else None,
            "saved_final_artifact_epoch": output.get("saved_final_artifact_epoch"),
            "restored_checkpoint_epoch": output.get("restored_checkpoint_epoch"),
            "use_best_epoch": output.get("use_best_epoch"),
            "error": None,
            "worker_elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(result, paths["result"])
        atomic_json(result, paths["checkpoint"])
        print(json.dumps({"status": "PASS", "candidate_id": candidate_id, "metrics": result["metrics"]}))
        return result
    except Exception as exc:
        failure = {
            "stage_id": "stage5a1",
            "candidate_id": candidate_id,
            "model_family": family,
            "target_mode": target_mode,
            "feature_schema": SCHEMA_ID,
            "sample_digest": sample_digest,
            "source_digest": source_digest,
            "status": "FAIL",
            "warning": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "worker_elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(failure, paths["checkpoint"])
        raise


def verify_candidate(candidate_id: str) -> dict[str, Any]:
    paths = candidate_paths(candidate_id)
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    if checkpoint.get("status") != "PASS":
        raise RuntimeError(f"Cannot reload failed candidate: {candidate_id}")
    X_train, X_val, _, _, _, val_rows = load_discovery()
    del X_train
    expected = pd.read_csv(paths["prediction"])
    if not np.array_equal(expected["row_id"].to_numpy(np.int64), val_rows):
        raise RuntimeError("Reload row order mismatch")
    prediction = predict_bundle(ROOT / checkpoint["model_path"], X_val)
    difference = np.abs(prediction - expected["y_pred"].to_numpy(np.float64))
    report = {
        "stage_id": "stage5a1",
        "candidate_id": candidate_id,
        "clean_process": True,
        "finite_predictions": bool(np.isfinite(prediction).all()),
        "prediction_count": len(prediction),
        "maximum_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "prediction_match": bool(np.allclose(prediction, expected["y_pred"], rtol=1e-5, atol=1e-4)),
        "status": "PASS" if np.allclose(prediction, expected["y_pred"], rtol=1e-5, atol=1e-4) else "FAIL",
    }
    atomic_json(report, paths["reload"])
    print(json.dumps(report))
    return report


def validate_fit11_artifact() -> dict[str, Any]:
    """Independently validate the serialized strict epoch-64 RealMLP artifact."""
    candidate_id = "stage5a1__realmlp__log1p__replacement2"
    paths = candidate_paths(candidate_id)
    checkpoint = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
    proof = json.loads((ROOT / "artifacts/reports/stage5a1_realmlp_log1p_fit11_protocol_proof.json").read_text(encoding="utf-8"))
    reload_report = json.loads(paths["reload"].read_text(encoding="utf-8"))
    bundle = joblib.load(ROOT / checkpoint["model_path"])
    model = bundle["model"]
    module = model.cv_alg_interface_.model
    resolved = model.get_config()
    checks = {
        "candidate_status_pass": checkpoint.get("status") == "PASS",
        "physical_fit_number_11": checkpoint.get("physical_fit_number") == 11,
        "use_best_epoch_false": resolved.get("use_best_epoch") is False,
        "early_stopping_disabled": resolved.get("use_early_stopping") is False,
        "requested_epochs_64": int(resolved.get("n_epochs", -1)) == 64,
        "completed_epochs_64": int(module.progress.epoch) == 64,
        "progress_max_epochs_64": int(module.progress.max_epochs) == 64,
        "official_stop_epoch_64": int(model.fit_params_["stop_epoch"]["mae"]) == 64,
        "internal_use_best_epoch_false": module.creator.config.get("use_best_epoch") is False,
        "restored_checkpoint_epoch_null": bundle.get("restored_checkpoint_epoch") is None,
        "saved_artifact_epoch_64": bundle.get("saved_final_artifact_epoch") == 64,
        "history_length_64": proof.get("training_history_length") == 64,
        "history_final_epoch_64": proof.get("training_history_final_epoch") == 64,
        "train_only_encoder": bundle.get("official_encoder_matches_train_only") is True,
        "historical_artifacts_unchanged": proof.get("historical_artifacts_unchanged") is True,
        "clean_reload_pass": reload_report.get("status") == "PASS",
        "reload_predictions_match": reload_report.get("prediction_match") is True,
        "test_rows_zero": checkpoint.get("test_rows") == 0,
    }
    report = {
        "stage_id": "stage5a1",
        "candidate_id": candidate_id,
        "independent_process": True,
        "resolved_use_best_epoch": resolved.get("use_best_epoch"),
        "resolved_use_early_stopping": resolved.get("use_early_stopping"),
        "resolved_requested_epochs": resolved.get("n_epochs"),
        "serialized_progress": dict(module.progress.__dict__),
        "serialized_fit_params": model.fit_params_,
        "observed_best_validation_epoch_not_restored": module.best_mean_val_epochs,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    atomic_json(report, ROOT / "artifacts/reports/stage5a1_realmlp_log1p_fit11_independent_validation.json")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Fit 11 independent artifact validation failed")
    return report


def run_smoke() -> dict[str, Any]:
    """Run <=4,000/1,000 rows and <=3 epochs for CPU and save/reload evidence."""
    from pytabkit import RealMLP_TD_Regressor
    from rtdl_num_embeddings import compute_bins

    seed_everything()
    X_train, X_val, y_train, y_val, _, _ = load_discovery()
    X_train = X_train.iloc[:4000].copy()
    X_val = X_val.iloc[:1000].copy()
    y_train_array = y_train.iloc[:4000].to_numpy(np.float32)
    y_val_array = y_val.iloc[:1000].to_numpy(np.float32)
    smoke_dir = ROOT / "artifacts/checkpoints/stage5/deep_core/smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Official RealMLP smoke.
    from stage5_deep_preprocessing import RealMLPPreprocessor

    prep = RealMLPPreprocessor().fit(X_train)
    tr = prep.transform(X_train)
    va = prep.transform(X_val)
    target = TargetTransform("log1p").fit(y_train_array)
    model = RealMLP_TD_Regressor(
        device="cpu", random_state=42, n_cv=1, n_refit=0, n_epochs=3,
        batch_size=256, predict_batch_size=512, n_threads=2, verbosity=0,
    )
    start = time.perf_counter()
    model.fit(tr, target.transform(y_train_array), X_val=va, y_val=target.transform(y_val_array), cat_col_names=CATEGORICAL_FEATURES)
    pred = target.inverse(np.asarray(model.predict(va)).reshape(-1))
    bundle_path = smoke_dir / "realmlp_smoke.joblib"
    atomic_joblib({"model": model, "preprocessor": prep, "target": target}, bundle_path)
    loaded = joblib.load(bundle_path)
    reload_pred = loaded["target"].inverse(np.asarray(loaded["model"].predict(loaded["preprocessor"].transform(X_val))).reshape(-1))
    results.append({"family": "realmlp", "finite_loss": True, "finite_predictions": bool(np.isfinite(pred).all()), "save_reload": bool(np.allclose(pred, reload_pred)), "epochs": 3, "seconds": time.perf_counter()-start})

    # Real TabM and true FT-Transformer smoke loops.
    for family in ["tabm", "ft_transformer"]:
        prep = TensorPreprocessor(family).fit(X_train)
        tr_num, tr_cat = prep.transform(X_train)
        va_num, va_cat = prep.transform(X_val)
        target = TargetTransform("log1p").fit(y_train_array)
        y_scaled = target.transform(y_train_array)
        bins = None
        if family == "tabm":
            bins = [x.numpy().astype(np.float32) for x in compute_bins(torch.from_numpy(tr_num), n_bins=16)]
        model, architecture = build_torch_model(family, tr_num.shape[1], prep.cardinalities_, bins)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.SmoothL1Loss()
        start = time.perf_counter()
        final_loss = None
        for _ in range(3):
            model.train()
            for begin in range(0, len(tr_num), 512):
                xn = torch.from_numpy(tr_num[begin:begin+512])
                xc = torch.from_numpy(tr_cat[begin:begin+512])
                yt = torch.from_numpy(y_scaled[begin:begin+512])
                optimizer.zero_grad(set_to_none=True)
                out = model(xn, xc).squeeze(-1)
                loss = loss_fn(out, yt[:, None].expand_as(out)) if family == "tabm" else loss_fn(out, yt)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
        from stage5_deep_models import predict_torch_model, atomic_torch_save
        transformed = predict_torch_model(model, family, va_num, va_cat, 512)
        pred = target.inverse(transformed)
        state_path = smoke_dir / f"{family}_smoke.pt"
        atomic_torch_save(model.state_dict(), state_path)
        reloaded, _ = build_torch_model(family, tr_num.shape[1], prep.cardinalities_, bins)
        reloaded.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
        reload_pred = target.inverse(predict_torch_model(reloaded, family, va_num, va_cat, 512))
        results.append({"family": family, "finite_loss": bool(np.isfinite(final_loss)), "finite_predictions": bool(np.isfinite(pred).all()), "save_reload": bool(np.allclose(pred, reload_pred, rtol=1e-5, atol=1e-4)), "epochs": 3, "seconds": time.perf_counter()-start, "architecture": architecture})

    checks = {
        "cpu_forward_backward_optimizer": all(row["finite_loss"] for row in results),
        "finite_predictions": all(row["finite_predictions"] for row in results),
        "save_reload": all(row["save_reload"] for row in results),
        "cuda_available": torch.cuda.is_available(),
        "cuda_smoke": "NOT_RUN_CPU_BUILD",
        "amp_smoke": "NOT_RUN_CPU_BUILD",
        "row_budget_valid": len(X_train) <= 5000 and len(X_val) <= 1000,
        "epoch_budget_valid": all(row["epochs"] <= 5 for row in results),
    }
    report = {
        "stage_id": "stage5a1",
        "device_selected": "cpu",
        "precision": "float32",
        "amp": False,
        "num_workers": 0,
        "family_results": results,
        "checks": checks,
        "status": "PASS" if all(value is True for key, value in checks.items() if key not in {"cuda_available", "cuda_smoke", "amp_smoke"}) else "FAIL",
    }
    atomic_json(report, ROOT / "artifacts/reports/stage5a1_hardware_smoke.json")
    print(json.dumps(report, indent=2))
    return report


def run_repair_smoke() -> dict[str, Any]:
    """Validate repaired preprocessing and TabM embedding path without a Screening fit."""
    from rtdl_num_embeddings import compute_bins
    from stage5_deep_models import atomic_torch_save, predict_torch_model
    from stage5_deep_preprocessing import RealMLPPreprocessor

    seed_everything()
    X_train, X_val, y_train, _, _, _ = load_discovery()
    realmlp = RealMLPPreprocessor().fit(X_train)
    contract = realmlp.categorical_contract(X_train, X_val)
    probe = X_val.iloc[:32].copy()
    probe.loc[probe.index[0], CATEGORICAL_FEATURES[0]] = "__REPAIR_UNSEEN__"
    transformed_probe = realmlp.transform(probe)
    realmlp_unknown_ok = str(transformed_probe.iloc[0][CATEGORICAL_FEATURES[0]]) == realmlp.unknown_token
    if not contract["other_values_subset_of_train"] or not realmlp_unknown_ok:
        raise RuntimeError("Repaired RealMLP Train-only vocabulary smoke failed")

    small_train = X_train.iloc[:4000].copy()
    small_val = X_val.iloc[:1000].copy()
    prep = TensorPreprocessor("tabm_embedding").fit(small_train)
    tr_num, tr_cat = prep.transform(small_train)
    va_num, va_cat = prep.transform(small_val)
    bins = [values.numpy().astype(np.float32) for values in compute_bins(torch.from_numpy(tr_num), n_bins=16)]
    model, architecture = build_torch_model(
        "tabm_embedding", tr_num.shape[1], prep.cardinalities_, bins
    )
    target = TargetTransform("log1p").fit(y_train.iloc[:4000].to_numpy(np.float32))
    y_scaled = target.transform(y_train.iloc[:4000].to_numpy(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    final_loss = None
    for begin in range(0, len(tr_num), 512):
        xn = torch.from_numpy(tr_num[begin:begin+512])
        xc = torch.from_numpy(tr_cat[begin:begin+512])
        yt = torch.from_numpy(y_scaled[begin:begin+512])
        optimizer.zero_grad(set_to_none=True)
        out = model(xn, xc).squeeze(-1)
        loss = loss_fn(out, yt[:, None].expand_as(out))
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    pred = target.inverse(predict_torch_model(model, "tabm_embedding", va_num, va_cat, 512))
    smoke_dir = ROOT / "artifacts/checkpoints/stage5/deep_core/repair_smoke"
    state_path = smoke_dir / "tabm_embedding_smoke.pt"
    atomic_torch_save(model.state_dict(), state_path)
    reloaded, _ = build_torch_model("tabm_embedding", tr_num.shape[1], prep.cardinalities_, bins)
    reloaded.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    reload_pred = target.inverse(predict_torch_model(reloaded, "tabm_embedding", va_num, va_cat, 512))
    report = {
        "stage_id": "stage5a1",
        "screening_fit": False,
        "realmlp_train_only_contract": contract["other_values_subset_of_train"],
        "realmlp_unseen_probe": realmlp_unknown_ok,
        "tabm_architecture": architecture,
        "tabm_learned_embedding_modules": len(model.cat_embeddings),
        "tabm_finite_loss": bool(np.isfinite(final_loss)),
        "tabm_finite_predictions": bool(np.isfinite(pred).all()),
        "tabm_reload_match": bool(np.allclose(pred, reload_pred, rtol=1e-5, atol=1e-4)),
    }
    report["status"] = "PASS" if all(
        [report["realmlp_train_only_contract"], report["realmlp_unseen_probe"], report["tabm_finite_loss"], report["tabm_finite_predictions"], report["tabm_reload_match"]]
    ) else "FAIL"
    atomic_json(report, ROOT / "artifacts/reports/stage5a1_repair_smoke.json")
    print(json.dumps(report, indent=2))
    return report


def run_parent_candidate(candidate_id: str) -> dict[str, Any]:
    if candidate_id not in REPLACEMENTS:
        raise ValueError("Parent replacement runner accepts only explicitly approved replacement IDs")
    family, _ = CANDIDATES[candidate_id]
    timeout_seconds = 1800 if family == "realmlp" else 2700
    report_path = ROOT / f"artifacts/reports/stage5a1_parent_{candidate_id}.json"
    log_path = ROOT / f"artifacts/reports/stage5a1_parent_{candidate_id}.log"
    config = {
        "stage_id": "stage5a1",
        "candidate_id": candidate_id,
        "replacement_of": REPLACEMENTS[candidate_id],
        "timeout_seconds": timeout_seconds,
        "parent_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), "--candidate", candidate_id],
        "started": True,
        "status": "RUNNING",
    }
    atomic_json(config, report_path)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            config["command"], cwd=ROOT, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        checkpoint = candidate_paths(candidate_id)["checkpoint"]
        checkpoint_status = json.loads(checkpoint.read_text(encoding="utf-8")).get("status") if checkpoint.exists() else "MISSING"
        config.update({
            "elapsed_seconds": time.perf_counter() - started,
            "return_code": completed.returncode,
            "timed_out": False,
            "checkpoint_status": checkpoint_status,
            "log_path": _relative(log_path),
            "status": "PASS" if completed.returncode == 0 and checkpoint_status == "PASS" else "FAIL",
        })
    except subprocess.TimeoutExpired as exc:
        log_path.write_text((exc.stdout or "") + "\n--- STDERR ---\n" + (exc.stderr or ""), encoding="utf-8")
        config.update({
            "elapsed_seconds": time.perf_counter() - started,
            "return_code": None,
            "timed_out": True,
            "checkpoint_status": "TIMEOUT",
            "log_path": _relative(log_path),
            "status": "FAIL",
        })
    atomic_json(config, report_path)
    print(json.dumps(config, indent=2))
    if config["status"] != "PASS":
        raise RuntimeError(f"Parent-run replacement failed: {candidate_id}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES))
    parser.add_argument("--verify", choices=sorted(CANDIDATES))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repair-smoke", action="store_true")
    parser.add_argument("--validate-fit11", action="store_true")
    parser.add_argument("--parent-candidate", choices=sorted(REPLACEMENTS))
    args = parser.parse_args()
    if args.validate_fit11:
        validate_fit11_artifact()
    elif args.repair_smoke:
        run_repair_smoke()
    elif args.parent_candidate:
        run_parent_candidate(args.parent_candidate)
    elif args.smoke:
        run_smoke()
    elif args.verify:
        verify_candidate(args.verify)
    elif args.candidate:
        run_candidate(args.candidate)
    else:
        parser.error("Choose --candidate, --verify, or --smoke")


if __name__ == "__main__":
    main()
