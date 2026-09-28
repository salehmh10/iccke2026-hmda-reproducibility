"""One-process worker for a single Stage 3 development, Fold, final, or importance fit."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.inspection import permutation_importance

from stage3_recovery_utils import (
    DEV_PATH, MANIFESTS, MODELS, ROOT, atomic_save_csv, checkpoint_metadata,
    experiment_digest, fold_paths, load_ids, load_rows, save_fold_checkpoint, utc_now,
)
from stage3_tree_utils import (
    canonical_json, deterministic_experiment_id, evaluate_regression_predictions,
    feature_lists, finite_prediction_check, make_complete_pipeline, model_size_bytes,
    save_model, sha256_file,
)


def tree_summary(model):
    inner = model.regressor_ if hasattr(model, "regressor_") else model
    estimator = inner.named_steps["regressor"]
    trees = []
    if hasattr(estimator, "tree_"):
        trees = [estimator]
    elif hasattr(estimator, "estimators_"):
        trees = list(np.asarray(estimator.estimators_, dtype=object).ravel())
    nodes = int(sum(tree.tree_.node_count for tree in trees if hasattr(tree, "tree_")))
    return estimator, nodes


def fit_predict(configuration, mode, train_ids, valid_ids):
    all_ids = np.sort(np.unique(np.concatenate([train_ids, valid_ids])))
    X, y = load_rows(configuration, mode, all_ids)
    model = make_complete_pipeline(configuration, mode)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X.loc[train_ids], y.loc[train_ids].to_numpy(dtype=float))
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    prediction = finite_prediction_check(model, X.loc[valid_ids])
    prediction_seconds = time.perf_counter() - started
    metrics = evaluate_regression_predictions(y.loc[valid_ids], prediction)
    estimator, nodes = tree_summary(model)
    return model, y.loc[valid_ids].to_numpy(dtype=float), prediction, fit_seconds, prediction_seconds, metrics, nodes, caught


def run_development(args, configuration):
    development = pd.read_csv(DEV_PATH, dtype={"row_id": "int64"})
    train_ids = np.sort(development.loc[development["development_role"].eq("train"), "row_id"].to_numpy(dtype=np.int64))
    valid_ids = np.sort(development.loc[development["development_role"].eq("validation"), "row_id"].to_numpy(dtype=np.int64))
    model, y_true, prediction, fit_seconds, prediction_seconds, metrics, nodes, caught = fit_predict(
        configuration, "without_sensitive", train_ids, valid_ids
    )
    model_path = Path(args.model_path)
    save_model(model_path, model)
    reloaded = joblib.load(model_path)
    X_valid, _ = load_rows(configuration, "without_sensitive", valid_ids)
    reload_prediction = finite_prediction_check(reloaded, X_valid.loc[valid_ids])
    if not np.allclose(prediction, reload_prediction, rtol=1e-10, atol=1e-8):
        raise AssertionError("Serialized development Pipeline did not prediction-match.")
    payload = {
        "status": "success", "task": "development", "configuration": configuration,
        "configuration_json": canonical_json(configuration),
        "candidate_digest": experiment_digest(configuration, "without_sensitive", 0, "runtime_repair_development"),
        "model_name": configuration["model_name"], "sensitive_mode": "without_sensitive",
        "feature_pack": configuration["feature_pack"], "target_mode": configuration["target_mode"],
        "training_rows": len(train_ids), "validation_rows": len(valid_ids),
        **{key: value for key, value in metrics.items() if key != "metric_warnings"},
        "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
        "model_size_bytes": model_size_bytes(model_path), "tree_node_count": nodes,
        "prediction_std": float(np.std(prediction)), "finite_predictions": True,
        "serialized_reload_match": True, "warning_status": " | ".join(str(x.message) for x in caught),
        "model_path": str(model_path.relative_to(ROOT)), "completed_at_utc": utc_now(),
    }
    from stage3_tree_utils import write_json
    write_json(args.result_path, payload)


def run_fold(args, configuration):
    _, _, assignments = load_ids()
    fold = int(args.fold)
    train_ids = np.sort(assignments.loc[~assignments["fold"].eq(fold), "row_id"].to_numpy(dtype=np.int64))
    valid_ids = np.sort(assignments.loc[assignments["fold"].eq(fold), "row_id"].to_numpy(dtype=np.int64))
    started_at = utc_now()
    model, y_true, prediction, fit_seconds, prediction_seconds, metrics, nodes, caught = fit_predict(
        configuration, args.mode, train_ids, valid_ids
    )
    frame = pd.DataFrame({"row_id": valid_ids, "fold": fold, "y_true": y_true, "y_pred": prediction})
    result = {
        "experiment_id": deterministic_experiment_id(
            configuration["model_name"], args.mode, configuration["target_mode"], "cv_fold", fold,
            configuration, configuration["feature_pack"]),
        "model_name": configuration["model_name"], "sensitive_mode": args.mode,
        "target_mode": configuration["target_mode"], "feature_pack": configuration["feature_pack"],
        "configuration_json": canonical_json(configuration), "training_rows": len(train_ids),
        "validation_rows": len(valid_ids),
        **{key: value for key, value in metrics.items() if key != "metric_warnings"},
        "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
        "tree_node_count": nodes, "tree_storage_mb": None,
        "warning_status": " | ".join([str(x.message) for x in caught] + metrics["metric_warnings"]),
        "status": "success", "provenance": args.provenance,
        "start_time_utc": started_at, "end_time_utc": utc_now(),
    }
    save_fold_checkpoint(configuration, args.mode, fold, frame, result)


def run_pilot(args, configuration):
    _, _, assignments = load_ids(); fold = int(args.fold)
    train_ids = np.sort(assignments.loc[~assignments["fold"].eq(fold), "row_id"].to_numpy(dtype=np.int64))
    valid_ids = np.sort(assignments.loc[assignments["fold"].eq(fold), "row_id"].to_numpy(dtype=np.int64))
    started_at = utc_now()
    model, y_true, prediction, fit_seconds, prediction_seconds, metrics, nodes, caught = fit_predict(
        configuration, args.mode, train_ids, valid_ids
    )
    model_path = Path(args.model_path); save_model(model_path, model)
    reloaded = joblib.load(model_path)
    X_valid, _ = load_rows(configuration, args.mode, valid_ids)
    if not np.allclose(prediction, finite_prediction_check(reloaded, X_valid.loc[valid_ids]), rtol=1e-10, atol=1e-8):
        raise AssertionError("Pilot Pipeline did not prediction-match after reload.")
    frame = pd.DataFrame({"row_id": valid_ids, "fold": fold, "y_true": y_true, "y_pred": prediction})
    result = {
        "experiment_id": deterministic_experiment_id(
            configuration["model_name"], args.mode, configuration["target_mode"], "full_fold_pilot", fold,
            configuration, configuration["feature_pack"]),
        "task": "full_fold_pilot", "model_name": configuration["model_name"],
        "sensitive_mode": args.mode, "target_mode": configuration["target_mode"],
        "feature_pack": configuration["feature_pack"], "configuration_json": canonical_json(configuration),
        "training_rows": len(train_ids), "validation_rows": len(valid_ids),
        **{key: value for key, value in metrics.items() if key != "metric_warnings"},
        "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
        "tree_node_count": nodes, "tree_storage_mb": None,
        "warning_status": " | ".join([str(x.message) for x in caught] + metrics["metric_warnings"]),
        "status": "success", "provenance": "full_fold_runtime_pilot",
        "start_time_utc": started_at, "end_time_utc": utc_now(),
        "pilot_model_path": str(model_path.relative_to(ROOT)), "pilot_model_sha256": sha256_file(model_path),
        "serialized_reload_match": True,
    }
    # A passing pilot is also a valid Fold-0 checkpoint for this exact configuration.
    save_fold_checkpoint(configuration, args.mode, fold, frame, {**result, "experiment_id": deterministic_experiment_id(
        configuration["model_name"], args.mode, configuration["target_mode"], "cv_fold", fold,
        configuration, configuration["feature_pack"]), "task": "cv_fold", "provenance": "reused_full_fold_pilot"})
    from stage3_tree_utils import write_json
    write_json(args.result_path, result)


def run_final(args, configuration):
    train_ids, _, _ = load_ids()
    X, y = load_rows(configuration, args.mode, train_ids)
    model = make_complete_pipeline(configuration, args.mode)
    start_at = utc_now(); start = time.perf_counter()
    model.fit(X, y.to_numpy(dtype=float)); fit_seconds = time.perf_counter() - start
    sample_ids = train_ids[:128]
    start = time.perf_counter(); reference = finite_prediction_check(model, X.loc[sample_ids]); pred_seconds = time.perf_counter() - start
    model_path = Path(args.model_path); save_model(model_path, model)
    sample = X.loc[sample_ids].copy(); sample.insert(0, "row_id", sample_ids)
    sample_path = MANIFESTS / f"stage3_reload_sample__{configuration['model_name']}__{args.mode}.csv"
    atomic_save_csv(sample.reset_index(drop=True), sample_path)
    loaded = joblib.load(model_path); check = finite_prediction_check(loaded, X.loc[sample_ids])
    if not np.allclose(reference, check, rtol=1e-10, atol=1e-8):
        raise AssertionError("Final Pipeline reload did not prediction-match.")
    _, nodes = tree_summary(model)
    experiment_id = deterministic_experiment_id(
        configuration["model_name"], args.mode, configuration["target_mode"], "final_training_fit", None,
        configuration, configuration["feature_pack"])
    payload = {
        "status": "success", "experiment_id": experiment_id, "model_name": configuration["model_name"],
        "sensitive_mode": args.mode, "feature_pack": configuration["feature_pack"],
        "target_mode": configuration["target_mode"], "configuration": configuration,
        "configuration_json": canonical_json(configuration), "training_rows": len(train_ids),
        "fit_time_seconds": fit_seconds, "prediction_time_seconds": pred_seconds,
        "model_path": str(model_path.relative_to(ROOT)), "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_size_bytes(model_path), "tree_node_count": nodes,
        "reload_sample_path": str(sample_path.relative_to(ROOT)),
        "reload_reference_predictions": reference.tolist(), "reload_prediction_match": True,
        "feature_list": feature_lists(configuration["feature_pack"], args.mode)["raw"],
        "start_time_utc": start_at, "end_time_utc": utc_now(),
        "configuration_digest": experiment_digest(configuration, args.mode, None, "final_training_fit"),
    }
    from stage3_tree_utils import write_json
    write_json(args.result_path, payload)


def run_importance(args, configuration):
    train_ids, _, _ = load_ids(); sample_ids = np.sort(train_ids[::max(len(train_ids)//10000, 1)][:10000])
    X, y = load_rows(configuration, args.mode, sample_ids)
    model = joblib.load(args.model_path)
    result = permutation_importance(
        model, X.loc[sample_ids], y.loc[sample_ids].to_numpy(dtype=float),
        scoring="neg_mean_absolute_error", n_repeats=3, random_state=42, n_jobs=1,
    )
    raw = feature_lists(configuration["feature_pack"], args.mode)["raw"]
    frame = pd.DataFrame({
        "source_feature": raw, "importance": result.importances_mean,
        "importance_std": result.importances_std, "model_name": configuration["model_name"],
        "sensitive_mode": args.mode, "method": "permutation", "sample_rows": len(sample_ids),
        "n_repeats": 3, "associative_not_causal": True,
    }).sort_values("importance", ascending=False)
    atomic_save_csv(frame, args.result_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["development", "fold", "pilot", "final", "importance"], required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--mode", default="without_sensitive")
    parser.add_argument("--fold", type=int, default=0); parser.add_argument("--result-path", required=True)
    parser.add_argument("--model-path", default=""); parser.add_argument("--provenance", default="targeted_worker")
    args = parser.parse_args(); configuration = json.loads(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps({"event":"start","task":args.task,"model":configuration["model_name"],"mode":args.mode,"fold":args.fold,"time":utc_now()}), flush=True)
    if args.task == "development": run_development(args, configuration)
    elif args.task == "fold": run_fold(args, configuration)
    elif args.task == "pilot": run_pilot(args, configuration)
    elif args.task == "final": run_final(args, configuration)
    else: run_importance(args, configuration)
    print(json.dumps({"event":"complete","task":args.task,"model":configuration["model_name"],"mode":args.mode,"fold":args.fold,"time":utc_now(),"rss_mb":round(psutil.Process().memory_info().rss/2**20,1)}), flush=True)


if __name__ == "__main__":
    main()
