"""Reload all Prompt 2 models in a clean process and compare predictions."""

from pathlib import Path
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline


root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
manifest = json.loads((root / "artifacts/manifests/prompt2_model_manifest.json").read_text(encoding="utf-8"))
reference = pd.read_csv(root / "artifacts/manifests/prompt2_reload_reference.csv")
sample_ids = np.sort(reference["row_id"].unique().astype(int))
source_path = root / "data/regression_with_sensitive_features.csv"
sample_header = pd.read_csv(source_path, nrows=5000, encoding="utf-8-sig")
dtype_map = {column: "category" for column in sample_header.select_dtypes(include="object").columns}
source = pd.read_csv(source_path, dtype=dtype_map, encoding="utf-8-sig", low_memory=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


rows = []
for item in manifest["models"]:
    model_path = root / item["model_path"]
    model = joblib.load(model_path)
    complete_pipeline = (
        isinstance(model, Pipeline)
        or (isinstance(model, TransformedTargetRegressor) and isinstance(model.regressor_, Pipeline))
    )
    features = item["feature_set"]
    X_sample = source.iloc[sample_ids][features].copy()
    before_fingerprint = int(pd.util.hash_pandas_object(X_sample, index=True).sum().sum())
    prediction = np.asarray(model.predict(X_sample), dtype=float)
    after_fingerprint = int(pd.util.hash_pandas_object(X_sample, index=True).sum().sum())
    expected = reference.loc[
        (reference["model_name"] == item["model_name"])
        & (reference["sensitive_mode"] == item["sensitive_mode"])
    ].sort_values("row_id")["reference_prediction"].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(prediction - expected)))
    passed = bool(
        complete_pipeline and np.isfinite(prediction).all()
        and np.allclose(prediction, expected, rtol=1e-12, atol=1e-12)
        and before_fingerprint == after_fingerprint
        and sha256(model_path) == item["model_sha256"]
    )
    rows.append({
        "model_name": item["model_name"], "sensitive_mode": item["sensitive_mode"],
        "target_mode": item["target_mode"], "model_path": item["model_path"],
        "artifact_sha256": sha256(model_path), "sample_row_id_hash": hashlib.sha256(sample_ids.tobytes()).hexdigest(),
        "prediction_count": len(prediction), "max_abs_difference": max_abs_diff,
        "rtol": 1e-12, "atol": 1e-12, "finite_predictions": bool(np.isfinite(prediction).all()),
        "raw_dataframe_input": True, "complete_pipeline": complete_pipeline,
        "source_frame_unchanged": before_fingerprint == after_fingerprint,
        "passed": passed, "error": "" if passed else "reload comparison failed",
    })

result = pd.DataFrame(rows)
output = root / "artifacts/reports/prompt2_model_reload_verification.csv"
result.to_csv(output, index=False)
print(result[["model_name", "sensitive_mode", "max_abs_difference", "passed"]].to_string(index=False))
if len(result) != 12 or not result["passed"].all():
    raise SystemExit(1)
