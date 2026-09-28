"""Prediction-only worker for Stage 4L.

This module loads one frozen model bundle and predicts the exact saved Test rows.
It never trains, tunes, or changes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TARGET = "loan_amount_000s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def install_no_fit_guard() -> list[str]:
    """Block common training entry points before a bundle is loaded."""

    guarded: list[str] = []

    def blocked(*_args, **_kwargs):
        raise RuntimeError("Stage 4L no-fit guard blocked a training call")

    from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
    from sklearn.pipeline import Pipeline

    for cls, names in [
        (Pipeline, ["fit"]),
        (ColumnTransformer, ["fit", "fit_transform"]),
        (TransformedTargetRegressor, ["fit"]),
    ]:
        for name in names:
            if hasattr(cls, name):
                setattr(cls, name, blocked)
                guarded.append(f"{cls.__module__}.{cls.__name__}.{name}")

    optional = [
        ("catboost", "CatBoostRegressor"),
        ("lightgbm", "LGBMRegressor"),
        ("xgboost", "XGBRegressor"),
    ]
    for module_name, class_name in optional:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            setattr(cls, "fit", blocked)
            guarded.append(f"{module_name}.{class_name}.fit")
        except (ImportError, AttributeError):
            continue
    return guarded


def load_test_view(source: Path, raw_columns: list[str], test_ids: np.ndarray) -> pd.DataFrame:
    """Read only requested Test rows and columns from a canonical CSV."""

    wanted = set(int(value) for value in test_ids)
    usecols = list(dict.fromkeys([*raw_columns, TARGET]))
    frame = pd.read_csv(
        source,
        usecols=usecols,
        skiprows=lambda line_number: line_number > 0 and (line_number - 1) not in wanted,
    )
    sorted_ids = np.array(sorted(wanted), dtype=np.int64)
    if len(frame) != len(sorted_ids):
        raise RuntimeError(f"Test view has {len(frame)} rows, expected {len(sorted_ids)}")
    frame.insert(0, "row_id", sorted_ids)
    frame = frame.set_index("row_id", drop=False).loc[test_ids].reset_index(drop=True)
    if frame["row_id"].tolist() != test_ids.tolist():
        raise RuntimeError("Test ordering does not match the saved row-ID file")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    args = parser.parse_args()

    sys.path[:0] = [str(ROOT), str(ROOT / "artifacts" / "environment" / "stage4_packages")]
    manifest_path = ROOT / args.candidate_manifest
    freeze_path = ROOT / args.freeze_manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_hash = sha256_file(freeze_path)
    candidate = next(item for item in manifest["candidates"] if item["candidate_id"] == args.candidate_id)

    if candidate["candidate_type"] != "saved_model":
        raise RuntimeError("The prediction worker accepts saved-model candidates only")
    if candidate["candidate_id"] not in freeze["candidate_ids"]:
        raise RuntimeError("Candidate is not present in the frozen set")
    if freeze.get("test_target_not_loaded") is not True:
        raise RuntimeError("Invalid pre-Test freeze statement")
    contract_payload = dict(freeze)
    contract_expected = contract_payload.pop("self_sha256_contract")
    contract_actual = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if contract_actual != contract_expected:
        raise RuntimeError("Freeze-manifest hash contract does not match")

    model_path = ROOT / candidate["model_bundle_path"]
    if sha256_file(model_path) != candidate["model_sha256"]:
        raise RuntimeError("Model hash changed after the pre-Test freeze")

    test_ids = pd.read_csv(ROOT / "artifacts/splits/test_row_ids.csv")["row_id"].to_numpy(np.int64)
    source_name = (
        "regression_with_sensitive_features.csv"
        if candidate["sensitive_mode"] == "with_sensitive"
        else "regression_without_sensitive_features.csv"
    )
    source = ROOT / "data" / source_name
    view = load_test_view(source, candidate["required_raw_input_columns"], test_ids)
    y_true = view[TARGET].to_numpy(float)
    features = view[candidate["required_raw_input_columns"]].copy(deep=True)

    guarded = install_no_fit_guard()
    load_started = time.perf_counter()
    model = joblib.load(model_path)
    load_seconds = time.perf_counter() - load_started
    before = features.copy(deep=True)
    prediction_started = time.perf_counter()
    prediction = np.asarray(model.predict(features), dtype=float).reshape(-1)
    prediction_seconds = time.perf_counter() - prediction_started

    if not features.equals(before):
        raise RuntimeError("The model changed the source Feature frame")
    if len(prediction) != len(test_ids):
        raise RuntimeError("Prediction length is incorrect")
    if not np.isfinite(prediction).all():
        raise RuntimeError("Prediction contains a missing or non-finite value")

    signed_error = prediction - y_true
    output = pd.DataFrame(
        {
            "candidate_id": candidate["candidate_id"],
            "row_id": test_ids,
            "y_true": y_true,
            "y_pred": prediction,
            "absolute_error": np.abs(signed_error),
            "signed_error": signed_error,
            "model_family": candidate["model_family"],
            "sensitive_mode": candidate["sensitive_mode"],
            "prediction_time_seconds": prediction_seconds,
            "model_sha256": candidate["model_sha256"],
            "freeze_manifest_sha256": freeze_hash,
        }
    )
    output_path = ROOT / "artifacts/predictions/final_test" / f"{candidate['candidate_id']}.csv"
    atomic_csv(output, output_path)
    metadata = {
        "candidate_id": candidate["candidate_id"],
        "status": "PASS",
        "row_count": int(len(output)),
        "prediction_path": str(output_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(output_path),
        "model_sha256": candidate["model_sha256"],
        "freeze_manifest_sha256": freeze_hash,
        "model_load_seconds": load_seconds,
        "prediction_seconds": prediction_seconds,
        "fit_call_count": 0,
        "guarded_training_methods": guarded,
        "finite_predictions": True,
        "source_frame_unchanged": True,
    }
    atomic_json(metadata, output_path.with_suffix(".metadata.json"))
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
