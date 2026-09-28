"""Train-only data, models, metrics, and bundles for Stage 5A2."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from stage5_deep_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RealMLPPreprocessor,
    TargetTransform,
    TensorPreprocessor,
)


ROOT = Path(__file__).resolve().parent
TARGET = "loan_amount_000s"
FINAL_SAMPLE = ROOT / "artifacts/splits/stage4/stage4_final_selection_sample.csv"
TRAIN_IDS = ROOT / "artifacts/splits/train_row_ids.csv"
TEST_IDS = ROOT / "artifacts/splits/test_row_ids.csv"
SOURCE_WITHOUT = ROOT / "data/regression_without_sensitive_features.csv"
SOURCE_WITH = ROOT / "data/regression_with_sensitive_features.csv"
FREEZE = ROOT / "artifacts/reports/stage5a2_prevalidation_freeze.json"
EXPECTED_FREEZE_SHA256 = "5d7406e179bf8554ea12c2ee2d9cc052b58bee4495b12cec331382a87b1ee4c4"
AMENDMENT = ROOT / "artifacts/reports/stage5a2_prevalidation_freeze_amendment_realmlp_dropout020.json"
AMENDMENT_LOCK = ROOT / "artifacts/manifests/stage5/stage5a2_freeze_amendment_lock.json"
REPLACEMENT_CANDIDATE_ID = "stage5a2__realmlp__refined__dropout020__replacement1"

SENSITIVE_NUMERICAL = ["minority_population"]
SENSITIVE_CATEGORICAL = [
    "applicant_ethnicity_name", "co_applicant_ethnicity_name",
    "applicant_race_name_1", "co_applicant_race_name_1",
    "applicant_sex_name", "co_applicant_sex_name", "majority_minority_tract",
]


def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def digest_values(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_joblib(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    os.replace(temporary, path)


def atomic_torch(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate_freeze() -> dict[str, Any]:
    if not FREEZE.exists():
        raise RuntimeError("Stage 5A2 pre-validation freeze is missing")
    if sha256_file(FREEZE) != EXPECTED_FREEZE_SHA256:
        raise RuntimeError("Stage 5A2 pre-validation freeze hash changed")
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or payload.get("regular_candidate_count") != 4:
        raise RuntimeError("Stage 5A2 pre-validation freeze is invalid")
    return payload


def candidate_definition(candidate_id: str) -> dict[str, Any]:
    freeze = validate_freeze()
    matches = [item for item in freeze["candidates"] if item["candidate_id"] == candidate_id]
    if candidate_id == REPLACEMENT_CANDIDATE_ID:
        if not AMENDMENT.exists() or not AMENDMENT_LOCK.exists():
            raise RuntimeError("The approved RealMLP amendment or its lock is missing")
        lock = json.loads(AMENDMENT_LOCK.read_text(encoding="utf-8"))
        if lock.get("status") != "LOCKED" or lock.get("original_freeze_sha256") != EXPECTED_FREEZE_SHA256:
            raise RuntimeError("The RealMLP amendment lock is invalid")
        if sha256_file(AMENDMENT) != lock.get("amendment_sha256"):
            raise RuntimeError("The RealMLP amendment hash changed")
        amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
        definition = amendment.get("replacement_candidate_definition")
        if amendment.get("status") != "PASS" or definition.get("candidate_id") != candidate_id:
            raise RuntimeError("The RealMLP amendment definition is invalid")
        return definition
    if len(matches) != 1:
        raise ValueError(f"Candidate is not uniquely frozen: {candidate_id}")
    return matches[0]


def feature_lists(sensitive_mode: str) -> tuple[list[str], list[str]]:
    if sensitive_mode == "without_sensitive":
        return NUMERICAL_FEATURES.copy(), CATEGORICAL_FEATURES.copy()
    if sensitive_mode == "with_sensitive":
        return NUMERICAL_FEATURES + SENSITIVE_NUMERICAL, CATEGORICAL_FEATURES + SENSITIVE_CATEGORICAL
    raise ValueError(f"Unknown sensitive mode: {sensitive_mode}")


def _load_source_rows(source: Path, row_ids: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Load exact original RangeIndex rows in bounded CSV chunks and restore requested order."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("Requested row IDs are not unique")
    wanted = set(int(value) for value in row_ids)
    pieces: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(source, usecols=columns, chunksize=50_000, low_memory=False):
        positions = np.arange(offset, offset + len(chunk), dtype=np.int64)
        mask = np.fromiter((int(value) in wanted for value in positions), dtype=bool, count=len(positions))
        if mask.any():
            selected = chunk.loc[mask].copy()
            selected.insert(0, "__row_id__", positions[mask])
            pieces.append(selected)
        offset += len(chunk)
    if not pieces:
        raise RuntimeError("No requested rows were loaded")
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.set_index("__row_id__", drop=True)
    missing = np.setdiff1d(row_ids, frame.index.to_numpy(np.int64))
    if len(missing):
        raise RuntimeError(f"Missing requested source rows: {missing[:10].tolist()}")
    return frame.loc[row_ids, columns].copy()


def record_target_access(candidate_id: str) -> None:
    validate_freeze()
    path = ROOT / "artifacts/reports/stage5a2_target_access_audit.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("freeze_sha256") != EXPECTED_FREEZE_SHA256:
            raise RuntimeError("Target-access audit freeze mismatch")
        return
    from datetime import datetime, timezone
    atomic_json({
        "stage_id": "stage5a2",
        "first_target_access_at": datetime.now(timezone.utc).isoformat(),
        "first_candidate_id": candidate_id,
        "freeze_path": str(FREEZE.relative_to(ROOT)),
        "freeze_sha256": EXPECTED_FREEZE_SHA256,
        "freeze_precedes_target_access": True,
        "test_rows_loaded": 0,
        "stage4l_test_metrics_loaded": False,
        "status": "PASS",
    }, path)


def load_final_selection(sensitive_mode: str, candidate_id: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_target_access(candidate_id)
    sample = pd.read_csv(FINAL_SAMPLE, usecols=["row_id", "sample_role", "target_bin"])
    train_rows = sample.loc[sample["sample_role"] == "train", "row_id"].to_numpy(np.int64)
    val_part = sample.loc[sample["sample_role"] == "validation"].copy()
    val_rows = val_part["row_id"].to_numpy(np.int64)
    val_bins = val_part["target_bin"].to_numpy()
    numerical, categorical = feature_lists(sensitive_mode)
    source = SOURCE_WITHOUT if sensitive_mode == "without_sensitive" else SOURCE_WITH
    columns = [TARGET, *numerical, *categorical]
    loaded = _load_source_rows(source, np.concatenate([train_rows, val_rows]), columns)
    train = loaded.loc[train_rows]
    val = loaded.loc[val_rows]
    return (
        train.loc[:, numerical + categorical].copy(),
        val.loc[:, numerical + categorical].copy(),
        train[TARGET].to_numpy(np.float32),
        val[TARGET].to_numpy(np.float32),
        train_rows, val_rows, val_bins,
    )


def load_full_train(sensitive_mode: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load all and only the immutable saved Train IDs; never load Test source rows."""
    train_ids = pd.read_csv(TRAIN_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    test_ids = pd.read_csv(TEST_IDS, usecols=["row_id"])["row_id"].to_numpy(np.int64)
    if len(train_ids) != 399_788 or len(np.unique(train_ids)) != len(train_ids):
        raise RuntimeError("Saved Train IDs are invalid")
    if len(np.intersect1d(train_ids, test_ids)) != 0:
        raise RuntimeError("Saved Train/Test IDs overlap")
    numerical, categorical = feature_lists(sensitive_mode)
    source = SOURCE_WITHOUT if sensitive_mode == "without_sensitive" else SOURCE_WITH
    columns = [TARGET, *numerical, *categorical]
    loaded = _load_source_rows(source, train_ids, columns)
    return loaded.loc[:, numerical + categorical].copy(), loaded[TARGET].to_numpy(np.float32), train_ids


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    absolute = np.abs(y_pred - y_true)
    mse = float(mean_squared_error(y_true, y_pred))
    top10 = y_true >= np.quantile(y_true, 0.90)
    top5 = y_true >= np.quantile(y_true, 0.95)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mape_percent": float(np.mean(absolute / np.maximum(np.abs(y_true), 1e-8)) * 100),
        "r_squared": float(r2_score(y_true, y_pred)),
        "rmsle": float(np.sqrt(np.mean(np.square(np.log1p(np.clip(y_true, 0, None)) - np.log1p(np.clip(y_pred, 0, None)))))),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "wape_percent": float(absolute.sum() / np.maximum(np.abs(y_true).sum(), 1e-8) * 100),
        "mean_signed_error": float(np.mean(y_pred - y_true)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "negative_prediction_rate": float(np.mean(y_pred < 0)),
        "top_decile_mae": float(np.mean(absolute[top10])),
        "top_five_percent_mae": float(np.mean(absolute[top5])),
    }


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, psutil.cpu_count(logical=False) or 2)))


def build_ft_model(n_num: int, cardinalities: list[int]) -> nn.Module:
    from rtdl_revisiting_models import FTTransformer
    return FTTransformer(
        n_cont_features=n_num, cat_cardinalities=cardinalities, d_out=1,
        n_blocks=3, d_block=96, attention_n_heads=8, attention_dropout=0.15,
        ffn_d_hidden=None, ffn_d_hidden_multiplier=4.0 / 3.0,
        ffn_dropout=0.10, residual_dropout=0.0,
    )


@torch.no_grad()
def predict_ft(model: nn.Module, x_num: np.ndarray, x_cat: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_cat)),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    for xn, xc in loader:
        output.append(model(xn, xc).squeeze(-1).detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def train_ft_regular(candidate_id: str, definition: dict[str, Any], X_train: pd.DataFrame, X_val: pd.DataFrame,
                     y_train: np.ndarray, y_val: np.ndarray, state_path: Path, history_path: Path,
                     progress_path: Path, seed: int = 42) -> dict[str, Any]:
    seed_everything(seed)
    numerical, categorical = feature_lists("without_sensitive")
    preprocessor = TensorPreprocessor("ft_transformer", numerical_features=numerical, categorical_features=categorical).fit(X_train)
    tr_num, tr_cat = preprocessor.transform(X_train)
    va_num, va_cat = preprocessor.transform(X_val)
    target = TargetTransform("log1p").fit(y_train)
    y_scaled = target.transform(y_train, standardize=True)
    model = build_ft_model(tr_num.shape[1], preprocessor.cardinalities_)
    training = definition["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(tr_num), torch.from_numpy(tr_cat), torch.from_numpy(y_scaled)),
                        batch_size=256, shuffle=True, generator=generator, num_workers=0)
    history: list[dict[str, Any]] = []
    best_state = None
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    peak_ram = psutil.Process().memory_info().rss / 1024**2
    started = time.perf_counter()
    for epoch in range(1, 25):
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        for xn, xc, yt in loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(xn, xc).squeeze(-1)
            loss = loss_fn(out, yt)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite FT-Transformer loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        transformed = predict_ft(model, va_num, va_cat)
        prediction = target.inverse(transformed, standardized=True)
        metrics = regression_metrics(y_val, prediction)
        scheduler.step(metrics["mae"])
        peak_ram = max(peak_ram, psutil.Process().memory_info().rss / 1024**2)
        row = {
            "epoch": epoch, "training_loss": float(np.mean(losses)),
            "validation_loss": float(np.mean(np.abs(prediction - y_val))),
            "validation_mae": metrics["mae"], "validation_rmse": metrics["rmse"],
            "validation_rmsle": metrics["rmsle"], "validation_r_squared": metrics["r_squared"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_start, "ram_mib": peak_ram, "vram_mib": 0.0,
        }
        history.append(row)
        atomic_csv(pd.DataFrame(history), history_path)
        if metrics["mae"] < best_mae - 1e-8:
            best_mae, best_epoch, stale = metrics["mae"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            atomic_torch(best_state, state_path)
        else:
            stale += 1
        atomic_json({"candidate_id": candidate_id, "epoch": epoch, "best_epoch": best_epoch,
                     "best_mae": best_mae, "status": "RUNNING"}, progress_path)
        if stale >= 5:
            break
    if best_state is None:
        raise RuntimeError("No valid FT-Transformer checkpoint")
    model.load_state_dict(best_state)
    predict_start = time.perf_counter()
    prediction = target.inverse(predict_ft(model, va_num, va_cat), standardized=True)
    predict_seconds = time.perf_counter() - predict_start
    return {
        "model": model, "preprocessor": preprocessor, "target_transform": target,
        "prediction": prediction, "metrics": regression_metrics(y_val, prediction),
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "fit_time_seconds": time.perf_counter() - started,
        "prediction_time_seconds": predict_seconds, "peak_ram_mib": peak_ram,
        "architecture": definition["architecture"], "training": training,
        "stop_reason": "early_stopping" if len(history) < 24 else "max_epochs",
    }


def train_realmlp_regular(candidate_id: str, definition: dict[str, Any], X_train: pd.DataFrame, X_val: pd.DataFrame,
                          y_train: np.ndarray, y_val: np.ndarray, history_path: Path, seed: int = 42) -> dict[str, Any]:
    from pytabkit import RealMLP_TD_Regressor
    seed_everything(seed)
    numerical, categorical = feature_lists("without_sensitive")
    preprocessor = RealMLPPreprocessor(numerical_features=numerical, categorical_features=categorical).fit(X_train)
    tr = preprocessor.transform(X_train)
    va = preprocessor.transform(X_val)
    contract = preprocessor.categorical_contract(X_train, X_val)
    if not contract["other_values_subset_of_train"]:
        raise RuntimeError("RealMLP Validation category contract failed")
    target = TargetTransform("raw").fit(y_train)
    config = definition["architecture_and_training"]
    frozen_preprocessor_match = None
    frozen_vocabulary_match = None
    frozen_numeric_medians_match = None
    if candidate_id == REPLACEMENT_CANDIDATE_ID:
        frozen_bundle = joblib.load(ROOT / "artifacts/models/deep/core_validation/stage5a2__realmlp__frozen.joblib")
        frozen_preprocessor = frozen_bundle["preprocessor"]
        frozen_vocabulary_match = all(
            preprocessor.vocabularies_[column] == frozen_preprocessor.vocabularies_[column]
            and preprocessor.rare_values_[column] == frozen_preprocessor.rare_values_[column]
            for column in categorical
        )
        frozen_numeric_medians_match = all(
            float(preprocessor.medians_[column]) == float(frozen_preprocessor.medians_[column])
            for column in numerical
        )
        frozen_preprocessor_match = bool(frozen_vocabulary_match and frozen_numeric_medians_match)
        if not frozen_preprocessor_match:
            raise RuntimeError("Replacement preprocessing differs from frozen RealMLP on identical Train rows")
    model = RealMLP_TD_Regressor(
        device="cpu", random_state=seed, n_cv=1, n_refit=0,
        n_repeats=int(config["n_repeats"]), n_threads=4, verbosity=0,
        n_epochs=64, batch_size=256, predict_batch_size=1024,
        train_metric_name="mae", val_metric_name="mae", use_early_stopping=True,
        p_drop=float(config["p_drop"]),
    )
    resolved_config = model.get_config()
    proof_path = ROOT / f"artifacts/reports/{candidate_id}_effective_resolved_config.json"
    proof = {
        "stage_id": "stage5a2", "candidate_id": candidate_id,
        "official_estimator_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "resolved_config": json.loads(json.dumps(resolved_config, default=str)),
        "n_repeats": resolved_config.get("n_repeats"), "p_drop": resolved_config.get("p_drop"),
        "external_validation_supplied": True, "external_validation_rows": len(X_val),
        "external_validation_row_id_hash": json.loads(FREEZE.read_text(encoding="utf-8"))["final_selection_validation_row_id_hash"],
        "training_row_id_hash": json.loads(FREEZE.read_text(encoding="utf-8"))["final_selection_train_row_id_hash"],
        "train_only_vocabulary_unchanged_from_frozen_realmlp": frozen_vocabulary_match,
        "numerical_medians_unchanged_from_frozen_realmlp": frozen_numeric_medians_match,
        "complete_preprocessor_policy_unchanged": frozen_preprocessor_match,
        "original_freeze_sha256": EXPECTED_FREEZE_SHA256,
        "amendment_sha256": sha256_file(AMENDMENT) if candidate_id == REPLACEMENT_CANDIDATE_ID else None,
        "test_or_stage4l_test_evidence_used": False,
        "fit_started": False, "status": "PREFIT_PASS",
    }
    if candidate_id == REPLACEMENT_CANDIDATE_ID:
        if int(resolved_config.get("n_repeats", -1)) != 1 or float(resolved_config.get("p_drop", -1)) != 0.20:
            raise RuntimeError("Replacement effective RealMLP configuration does not match the approved amendment")
        if frozen_preprocessor_match is not True:
            raise RuntimeError("Replacement Train-only preprocessing proof failed")
    atomic_json(proof, proof_path)
    proof["fit_started"] = True
    proof["status"] = "RUNNING"
    atomic_json(proof, proof_path)
    started = time.perf_counter()
    model.fit(tr, target.transform(y_train), X_val=va, y_val=target.transform(y_val),
              cat_col_names=categorical, time_to_fit_in_seconds=3550)
    fit_seconds = time.perf_counter() - started
    stop_value = model.fit_params_["stop_epoch"]["mae"]
    if isinstance(stop_value, (list, tuple, np.ndarray)):
        best_epoch = int(np.median(np.asarray(stop_value, dtype=float)))
    else:
        best_epoch = int(stop_value)
    predict_start = time.perf_counter()
    prediction = target.inverse(np.asarray(model.predict(va)).reshape(-1), standardized=True)
    predict_seconds = time.perf_counter() - predict_start
    metrics = regression_metrics(y_val, prediction)
    history = pd.DataFrame([{
        "epoch": best_epoch, "training_loss": np.nan, "validation_loss": np.nan,
        "validation_mae": metrics["mae"], "validation_rmse": metrics["rmse"],
        "validation_rmsle": metrics["rmsle"], "validation_r_squared": metrics["r_squared"],
        "learning_rate": np.nan, "epoch_seconds": np.nan,
        "ram_mib": psutil.Process().memory_info().rss / 1024**2, "vram_mib": 0.0,
        "note": "Official public API exposes the selected epoch but not per-epoch history.",
    }])
    atomic_csv(history, history_path)
    encoder = model.x_converter_.cat_tf.named_transformers_["categorical"]
    official_contract = {}
    for index, column in enumerate(categorical):
        expected = set(tr[column].astype(str).unique())
        learned = {str(value) for value in encoder.categories_[index]}
        official_contract[column] = {
            "train_only_cardinality": len(expected), "official_encoder_cardinality": len(learned),
            "unexpected_official_categories": sorted(learned - expected),
            "missing_train_categories": sorted(expected - learned),
        }
    if any(row["unexpected_official_categories"] or row["missing_train_categories"] for row in official_contract.values()):
        raise RuntimeError("Official RealMLP encoder is not Train-only")
    proof.update({
        "fit_completed": True,
        "official_encoder_matches_train_only": True,
        "completed_epoch_count": int(model.cv_alg_interface_.model.progress.epoch),
        "selected_best_epoch": best_epoch,
        "status": "PASS",
    })
    atomic_json(proof, proof_path)
    return {
        "model": model, "preprocessor": preprocessor, "target_transform": target,
        "prediction": prediction, "metrics": metrics, "best_epoch": best_epoch,
        "epochs_completed": int(model.cv_alg_interface_.model.progress.epoch),
        "fit_time_seconds": fit_seconds, "prediction_time_seconds": predict_seconds,
        "peak_ram_mib": psutil.Process().memory_info().rss / 1024**2,
        "architecture": {"implementation": "pytabkit.RealMLP_TD_Regressor", "resolved_config": model.get_config()},
        "training": {"checkpoint_metric": "original_scale_validation_mae", "n_repeats": int(config["n_repeats"])},
        "stop_reason": "official_early_stopping", "categorical_contract": contract,
        "official_encoder_contract": official_contract,
        "effective_resolved_config": resolved_config,
        "effective_config_proof_path": str(proof_path.relative_to(ROOT)),
        "train_only_vocabulary_unchanged": frozen_vocabulary_match,
        "numerical_medians_unchanged": frozen_numeric_medians_match,
    }


def train_realmlp_fixed_validation(candidate_id: str, sensitive_mode: str, X_train: pd.DataFrame,
                                   X_val: pd.DataFrame, y_train: np.ndarray, y_val: np.ndarray,
                                   fixed_epoch: int, history_path: Path, seed: int = 42) -> dict[str, Any]:
    """Fit the one fixed-epoch matched Validation model without restoration."""
    from stage5_deep_models import StrictFinalEpochRealMLPRegressor
    seed_everything(seed)
    numerical, categorical = feature_lists(sensitive_mode)
    preprocessor = RealMLPPreprocessor(numerical_features=numerical, categorical_features=categorical).fit(X_train)
    tr = preprocessor.transform(X_train)
    va = preprocessor.transform(X_val)
    contract = preprocessor.categorical_contract(X_train, X_val)
    if not contract["other_values_subset_of_train"]:
        raise RuntimeError("Fixed-epoch RealMLP Validation category contract failed")
    target = TargetTransform("raw").fit(y_train)
    model = StrictFinalEpochRealMLPRegressor(
        device="cpu", random_state=seed, n_cv=1, n_refit=0, n_repeats=1,
        n_threads=4, verbosity=0, n_epochs=int(fixed_epoch), batch_size=256,
        predict_batch_size=1024, train_metric_name="mae", val_metric_name="mae",
        use_early_stopping=False, p_drop=0.15,
    )
    resolved = model.get_config()
    proof_path = ROOT / f"artifacts/reports/{candidate_id}_fixed_epoch_proof.json"
    prefit = {
        "stage_id": "stage5a2", "candidate_id": candidate_id, "sensitive_mode": sensitive_mode,
        "official_estimator_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "resolved_config": json.loads(json.dumps(resolved, default=str)),
        "requested_epoch": int(fixed_epoch), "use_best_epoch": resolved.get("use_best_epoch"),
        "use_early_stopping": resolved.get("use_early_stopping"), "n_repeats": resolved.get("n_repeats"),
        "p_drop": resolved.get("p_drop"), "external_validation_rows": len(X_val),
        "best_checkpoint_restoration_requested": False, "test_rows": 0,
        "test_or_stage4l_test_evidence_used": False, "fit_started": False, "status": "PREFIT_PASS",
    }
    if not (resolved.get("use_best_epoch") is False and resolved.get("use_early_stopping") is False
            and int(resolved.get("n_epochs", -1)) == int(fixed_epoch)
            and int(resolved.get("n_repeats", -1)) == 1 and float(resolved.get("p_drop", -1)) == 0.15):
        raise RuntimeError("Fixed-epoch effective RealMLP configuration proof failed before fit")
    atomic_json(prefit, proof_path)
    prefit.update({"fit_started": True, "status": "RUNNING"})
    atomic_json(prefit, proof_path)
    started = time.perf_counter()
    model.fit(tr, target.transform(y_train), X_val=va, y_val=target.transform(y_val),
              cat_col_names=categorical, time_to_fit_in_seconds=5350)
    fit_seconds = time.perf_counter() - started
    module = model.cv_alg_interface_.model
    completed_epoch = int(module.progress.epoch)
    max_epoch = int(module.progress.max_epochs)
    stop_epoch = int(model.fit_params_["stop_epoch"]["mae"])
    internal_use_best = module.creator.config.get("use_best_epoch")
    internal_early = module.creator.config.get("use_early_stopping")
    callbacks_created = bool(getattr(module, "ckpt_callbacks", None))
    if not (completed_epoch == fixed_epoch and max_epoch == fixed_epoch and stop_epoch == fixed_epoch
            and internal_use_best is False and internal_early is False and not callbacks_created):
        raise RuntimeError("Fixed-epoch RealMLP completed/restoration proof failed")
    predict_start = time.perf_counter()
    prediction = target.inverse(np.asarray(model.predict(va)).reshape(-1), standardized=True)
    prediction_seconds = time.perf_counter() - predict_start
    metrics = regression_metrics(y_val, prediction)
    atomic_csv(pd.DataFrame([{
        "epoch": fixed_epoch, "training_loss": np.nan, "validation_loss": np.nan,
        "validation_mae": metrics["mae"], "validation_rmse": metrics["rmse"],
        "validation_rmsle": metrics["rmsle"], "validation_r_squared": metrics["r_squared"],
        "learning_rate": np.nan, "epoch_seconds": fit_seconds,
        "ram_mib": psutil.Process().memory_info().rss / 1024**2, "vram_mib": 0.0,
        "note": "Official public API exposes final fixed epoch; per-epoch loss history is unavailable.",
    }]), history_path)
    encoder = model.x_converter_.cat_tf.named_transformers_["categorical"]
    official_contract = {}
    for index, column in enumerate(categorical):
        expected = set(tr[column].astype(str).unique())
        learned = {str(value) for value in encoder.categories_[index]}
        official_contract[column] = {
            "train_only_cardinality": len(expected), "official_encoder_cardinality": len(learned),
            "unexpected_official_categories": sorted(learned - expected),
            "missing_train_categories": sorted(expected - learned),
        }
    encoder_pass = not any(row["unexpected_official_categories"] or row["missing_train_categories"]
                           for row in official_contract.values())
    if not encoder_pass:
        raise RuntimeError("Fixed-epoch official encoder is not Train-only")
    proof = {
        **prefit, "completed_epoch": completed_epoch, "saved_artifact_epoch": completed_epoch,
        "progress_max_epoch": max_epoch, "official_stop_epoch": stop_epoch,
        "internal_use_best_epoch": internal_use_best, "internal_use_early_stopping": internal_early,
        "restoration_callback_created": callbacks_created, "restored_checkpoint_epoch": None,
        "official_encoder_matches_train_only": encoder_pass, "fit_completed": True, "status": "PASS",
    }
    atomic_json(proof, proof_path)
    return {
        "model": model, "preprocessor": preprocessor, "target_transform": target,
        "prediction": prediction, "metrics": metrics, "fixed_epoch": fixed_epoch,
        "fit_time_seconds": fit_seconds, "prediction_time_seconds": prediction_seconds,
        "peak_ram_mib": psutil.Process().memory_info().rss / 1024**2,
        "architecture": {"implementation": "pytabkit.RealMLP_TD_Regressor", "resolved_config": resolved},
        "training": {"fixed_epoch": fixed_epoch, "early_stopping": False,
                     "use_best_epoch": False, "best_checkpoint_restoration": False},
        "categorical_contract": contract, "official_encoder_contract": official_contract,
        "fixed_epoch_proof_path": str(proof_path.relative_to(ROOT)),
    }


def train_realmlp_full_train(candidate_id: str, sensitive_mode: str, X_train: pd.DataFrame,
                             y_train: np.ndarray, fixed_epoch: int, history_path: Path,
                             seed: int = 42) -> dict[str, Any]:
    """Fit a strict fixed-epoch RealMLP refit interface on every supplied Train row."""
    from stage5_deep_models import StrictFinalEpochRealMLPRegressor
    seed_everything(seed)
    numerical, categorical = feature_lists(sensitive_mode)
    preprocessor = RealMLPPreprocessor(numerical_features=numerical, categorical_features=categorical).fit(X_train)
    transformed = preprocessor.transform(X_train)
    target = TargetTransform("raw").fit(y_train)
    model = StrictFinalEpochRealMLPRegressor(
        device="cpu", random_state=seed, n_cv=1, n_refit=1, n_repeats=1,
        val_fraction=0.20, n_threads=4, verbosity=0, n_epochs=int(fixed_epoch),
        batch_size=256, predict_batch_size=1024, train_metric_name="mae",
        val_metric_name="mae", use_early_stopping=False, p_drop=0.15,
    )
    resolved = model.get_config()
    proof_path = ROOT / f"artifacts/reports/{candidate_id}_fixed_epoch_proof.json"
    prefit = {
        "stage_id": "stage5a2", "candidate_id": candidate_id, "sensitive_mode": sensitive_mode,
        "official_estimator_class": model.__class__.__module__ + "." + model.__class__.__name__,
        "resolved_config": json.loads(json.dumps(resolved, default=str)),
        "input_training_rows": len(X_train), "requested_epoch": int(fixed_epoch),
        "n_cv": resolved.get("n_cv"), "n_refit": resolved.get("n_refit"),
        "n_repeats": resolved.get("n_repeats"), "use_best_epoch": resolved.get("use_best_epoch"),
        "use_early_stopping": resolved.get("use_early_stopping"), "p_drop": resolved.get("p_drop"),
        "full_train_refit_policy": "Official n_refit=1 fits the deployed interface on all rows supplied to fit().",
        "early_stopping": False, "best_checkpoint_restoration": False,
        "test_rows": 0, "test_or_stage4l_test_evidence_used": False,
        "fit_started": False, "status": "PREFIT_PASS",
    }
    if not (len(X_train) == 399_788 and resolved.get("use_best_epoch") is False
            and resolved.get("use_early_stopping") is False and int(resolved.get("n_epochs", -1)) == fixed_epoch
            and int(resolved.get("n_cv", -1)) == 1 and int(resolved.get("n_refit", -1)) == 1
            and int(resolved.get("n_repeats", -1)) == 1 and float(resolved.get("p_drop", -1)) == 0.15):
        raise RuntimeError("Full-Train effective RealMLP configuration proof failed before fit")
    atomic_json(prefit, proof_path)
    prefit.update({"fit_started": True, "status": "RUNNING"}); atomic_json(prefit, proof_path)
    started = time.perf_counter()
    model.fit(transformed, target.transform(y_train), cat_col_names=categorical, time_to_fit_in_seconds=7150)
    fit_seconds = time.perf_counter() - started
    if not hasattr(model, "refit_alg_interface_") or model.alg_interface_ is not model.refit_alg_interface_:
        raise RuntimeError("Official full-Train refit interface is missing or not deployed")
    cv_module = model.cv_alg_interface_.model
    refit_module = model.refit_alg_interface_.model
    cv_completed = int(cv_module.progress.epoch)
    refit_completed = int(refit_module.progress.epoch)
    refit_max = int(refit_module.progress.max_epochs)
    stop_epoch = int(model.fit_params_["stop_epoch"]["mae"])
    internal_use_best = refit_module.creator.config.get("use_best_epoch")
    internal_early = refit_module.creator.config.get("use_early_stopping")
    callbacks_created = bool(getattr(refit_module, "ckpt_callbacks", None))
    postfit_audit = {
        **prefit, "fit_completed": True, "cv_completed_epoch": cv_completed,
        "refit_completed_epoch": refit_completed, "refit_progress_max_epoch": refit_max,
        "official_stop_epoch": stop_epoch, "internal_use_best_epoch": internal_use_best,
        "internal_use_early_stopping": internal_early,
        "restoration_callback_created": callbacks_created,
        "deployed_interface_is_full_data_refit": model.alg_interface_ is model.refit_alg_interface_,
        "deployed_refit_training_rows": len(X_train), "status": "POSTFIT_AUDIT",
    }
    atomic_json(postfit_audit, proof_path)
    # The CV model is only an official setup pass used to construct the refit interface.
    # Deployment correctness depends on the all-row refit interface, which must satisfy every invariant below.
    if not (refit_completed == fixed_epoch and refit_max == fixed_epoch
            and stop_epoch == fixed_epoch and internal_use_best is False and internal_early is False
            and not callbacks_created and model.alg_interface_ is model.refit_alg_interface_):
        postfit_audit["status"] = "FAIL"
        atomic_json(postfit_audit, proof_path)
        raise RuntimeError("Full-Train deployed-refit fixed-epoch proof failed")
    model.to("cpu")
    atomic_csv(pd.DataFrame([{
        "epoch": fixed_epoch, "training_loss": np.nan, "validation_loss": np.nan,
        "learning_rate": np.nan, "epoch_seconds": fit_seconds,
        "ram_mib": psutil.Process().memory_info().rss / 1024**2, "vram_mib": 0.0,
        "input_training_rows": len(X_train), "deployed_refit_training_rows": len(X_train),
        "note": "Official fixed-epoch refit interface trained on all supplied Train rows; public per-epoch loss history is unavailable.",
    }]), history_path)
    encoder = model.x_converter_.cat_tf.named_transformers_["categorical"]
    official_contract = {}
    for index, column in enumerate(categorical):
        expected = set(transformed[column].astype(str).unique())
        learned = {str(value) for value in encoder.categories_[index]}
        official_contract[column] = {
            "train_only_cardinality": len(expected), "official_encoder_cardinality": len(learned),
            "unexpected_official_categories": sorted(learned - expected),
            "missing_train_categories": sorted(expected - learned),
        }
    encoder_pass = not any(row["unexpected_official_categories"] or row["missing_train_categories"]
                           for row in official_contract.values())
    if not encoder_pass:
        raise RuntimeError("Full-Train official encoder is not Train-only")
    proof = {
        **prefit, "fit_completed": True, "cv_completed_epoch": cv_completed,
        "refit_completed_epoch": refit_completed, "completed_epoch": refit_completed,
        "saved_artifact_epoch": refit_completed, "refit_progress_max_epoch": refit_max,
        "official_stop_epoch": stop_epoch, "internal_use_best_epoch": internal_use_best,
        "internal_use_early_stopping": internal_early,
        "restoration_callback_created": callbacks_created, "restored_checkpoint_epoch": None,
        "deployed_interface_is_full_data_refit": True, "deployed_refit_training_rows": len(X_train),
        "official_encoder_matches_train_only": encoder_pass, "model_moved_to_cpu": True,
        "status": "PASS",
    }
    atomic_json(proof, proof_path)
    return {
        "model": model, "preprocessor": preprocessor, "target_transform": target,
        "fit_time_seconds": fit_seconds, "peak_ram_mib": psutil.Process().memory_info().rss / 1024**2,
        "architecture": {"implementation": "pytabkit.RealMLP_TD_Regressor", "resolved_config": resolved},
        "training": {"fixed_epoch": fixed_epoch, "early_stopping": False, "use_best_epoch": False,
                     "best_checkpoint_restoration": False, "n_refit": 1},
        "fixed_epoch_proof_path": str(proof_path.relative_to(ROOT)),
        "official_encoder_contract": official_contract,
    }


def predict_bundle(bundle_path: Path, X_raw: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    features = bundle["numerical_features"] + bundle["categorical_features"]
    raw = X_raw.loc[:, features].copy()
    if bundle["family"] == "realmlp":
        transformed = bundle["preprocessor"].transform(raw)
        values = np.asarray(bundle["model"].predict(transformed)).reshape(-1)
    elif bundle["family"] == "ft_transformer":
        x_num, x_cat = bundle["preprocessor"].transform(raw)
        model = build_ft_model(len(bundle["numerical_features"]), bundle["cardinalities"])
        state = torch.load(ROOT / bundle["model_state_path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        values = predict_ft(model, x_num, x_cat)
    else:
        raise ValueError(bundle["family"])
    return bundle["target_transform"].inverse(values, standardized=True)
