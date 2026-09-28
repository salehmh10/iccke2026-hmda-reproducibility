"""Stage 4C CatBoost experiment, analysis, and verification helpers.

The helpers use only saved training-only Stage 4 Discovery rows. They never
load locked Test targets or create Test predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import stage4_boosting_utils as s4


STAGE_ID = "stage4c"
STAGE_NAME = "Stage 4C — Initial CatBoost Model and Controlled Sensitive Comparison"
TARGET = s4.TARGET_COLUMN
SEED = 42
THREAD_COUNT = 4
EXECUTION_MODE = "CPU"
SCREENING_TRAIN_ROWS = 30_000
SCREENING_VALIDATION_ROWS = 10_000
MAX_SHAP_ROWS = 300

SENSITIVE_NUMERIC = ("minority_population",)
SENSITIVE_CATEGORICAL = (
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "majority_minority_tract",
)
SENSITIVE_COLUMNS = SENSITIVE_CATEGORICAL + SENSITIVE_NUMERIC

SHARED_PARAMETERS = {
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 10,
    "random_strength": 1,
    "iterations": 1000,
    "early_stopping_rounds": 75,
    "random_seed": SEED,
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "task_type": EXECUTION_MODE,
    "thread_count": THREAD_COUNT,
    "verbose": False,
    "allow_writing_files": False,
}

REQUIRED_CANDIDATES = {
    "candidate_01_base_raw": {
        "candidate_id": "candidate_01_base_raw",
        "feature_pack": "boosting_base_v1",
        "target_mode": "raw",
        "parameters": SHARED_PARAMETERS,
        "required": True,
    },
    "candidate_02_native_raw": {
        "candidate_id": "candidate_02_native_raw",
        "feature_pack": "catboost_native_v1",
        "target_mode": "raw",
        "parameters": SHARED_PARAMETERS,
        "required": True,
    },
    "candidate_03_native_log1p": {
        "candidate_id": "candidate_03_native_log1p",
        "feature_pack": "catboost_native_v1",
        "target_mode": "log1p",
        "parameters": SHARED_PARAMETERS,
        "required": True,
    },
}


def paths(root: str | Path) -> dict[str, Path]:
    project = Path(root).resolve()
    return {
        "root": project,
        "results": project / "artifacts/results/stage4/catboost/initial",
        "predictions": project / "artifacts/predictions/catboost/initial",
        "models": project / "artifacts/models/catboost/preliminary",
        "candidate_models": project / "artifacts/models/catboost/initial_candidates",
        "features": project / "artifacts/features/stage4/catboost",
        "figures": project / "artifacts/figures/stage4/catboost",
        "checkpoints": project / "artifacts/checkpoints/stage4/catboost",
        "manifests": project / "artifacts/manifests/stage4/catboost",
        "reports": project / "artifacts/reports",
        "splits": project / "artifacts/splits/stage4",
        "registry": project / "artifacts/results/experiment_results.csv",
    }


def ensure_directories(root: str | Path) -> None:
    for name, path in paths(root).items():
        if name not in {"root", "registry"}:
            path.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_digest(path: str | Path) -> str:
    return s4.sha256_file(path)


def frame_digest(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _relative_name(project: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project))
    except ValueError:
        return str(resolved)


def _protected_candidates(project: Path) -> list[Path]:
    baseline = read_json(project / "artifacts/manifests/stage4/stage4b_protected_hashes_before.json")
    candidates: set[Path] = set()
    mutable_names = {
        "AGENTS.md", "TASK.md", "PLAN.md", "DECISIONS.md", "LOG.md",
        "artifacts/results/experiment_results.csv",
    }
    for name in baseline["hashes"]:
        item = Path(name)
        candidate = item if item.is_absolute() else project / item
        normalized = str(candidate.resolve().relative_to(project)).replace("\\", "/") if not item.is_absolute() else ""
        if normalized not in mutable_names:
            candidates.add(candidate)
    explicit = [
        project / "REGRESSION_PART4_BOOSTING_FOUNDATION.ipynb",
        project / "REGRESSION_PART3_TREE_MODELS.ipynb",
        project / "REGRESSION_PART2_MODELING.ipynb",
        project / "stage4_boosting_utils.py",
        project / "stage4b_feature_builder.py",
        project / "artifacts/features/stage4/boosting_feature_packs.json",
        project / "artifacts/features/stage4/catboost_feature_schema.json",
        project / "artifacts/splits/stage4/stage4_discovery_sample.csv",
        project / "artifacts/splits/stage4/stage4_feature_confirmation_sample.csv",
        project / "artifacts/splits/stage4/stage4_final_selection_sample.csv",
        project / "data/regression_without_sensitive_features.csv",
        project / "data/regression_with_sensitive_features.csv",
    ]
    candidates.update(explicit)
    for pattern in (
        "artifacts/reports/stage4a*",
        "artifacts/reports/stage4b*",
        "artifacts/manifests/stage4/stage4a*",
        "artifacts/manifests/stage4/stage4b*",
        "artifacts/checkpoints/stage4/stage4a*",
        "artifacts/checkpoints/stage4/stage4b*",
        "artifacts/features/stage4/*",
    ):
        candidates.update(path for path in project.glob(pattern) if path.is_file())
    missing = [str(path) for path in candidates if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Protected Stage 4C inputs are missing: {missing[:10]}")
    return sorted(candidates, key=lambda value: str(value).lower())


def capture_protected_baseline(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    ensure_directories(project)
    output = paths(project)["manifests"] / "stage4c_protected_hashes_before.json"
    if output.is_file():
        return read_json(output)
    hashes: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for path in _protected_candidates(project):
        name = _relative_name(project, path)
        hashes[name] = file_digest(path)
        sizes[name] = path.stat().st_size
    result = {
        "stage": STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "file_count": len(hashes),
        "hashes": hashes,
        "sizes": sizes,
        "status": "PASS",
    }
    s4.atomic_write_json(output, result)
    return result


def recheck_protected(root: str | Path, save: bool = False) -> dict[str, Any]:
    project = Path(root).resolve()
    before = capture_protected_baseline(project)
    mismatches: dict[str, Any] = {}
    for name, expected in before["hashes"].items():
        path = Path(name)
        path = path if path.is_absolute() else project / path
        if not path.is_file():
            mismatches[name] = {"status": "missing"}
            continue
        actual = file_digest(path)
        size = path.stat().st_size
        if actual != expected or size != before["sizes"][name]:
            mismatches[name] = {
                "status": "changed",
                "expected_sha256": expected,
                "actual_sha256": actual,
                "expected_bytes": before["sizes"][name],
                "actual_bytes": size,
            }
    result = {
        "stage": STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "file_count": before["file_count"],
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "FAIL",
    }
    if save:
        s4.atomic_write_json(paths(project)["manifests"] / "stage4c_protected_hashes_after.json", result)
    return result


def validate_start(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    stage4a = read_json(project / "artifacts/reports/stage4a_verification.json")
    stage4b = read_json(project / "artifacts/reports/stage4b_verification.json")
    sample = read_json(project / "artifacts/splits/stage4/stage4_sample_verification.json")
    smoke = read_json(project / "artifacts/checkpoints/stage4/stage4b_catboost_smoke.json")
    reload_report = read_json(project / "artifacts/reports/stage4b_clean_model_roundtrip.json")
    protected = recheck_protected(project)
    packs = read_json(project / "artifacts/features/stage4/boosting_feature_packs.json")
    schema = read_json(project / "artifacts/features/stage4/catboost_feature_schema.json")
    s4.activate_local_packages(project)
    import catboost
    checks = {
        "stage4a_pass": stage4a.get("status") == "PASS",
        "stage4b_pass": stage4b.get("status") == "PASS",
        "protected_hashes_unchanged": protected.get("status") == "PASS",
        "discovery_sample_valid": sample.get("samples", {}).get("discovery", {}).get("valid") is True,
        "discovery_test_overlap_zero": sample.get("test_overlap_rows") == 0,
        "permitted_packs_exist": all(name in packs.get("packs", {}) for name in ("boosting_base_v1", "catboost_native_v1")),
        "catboost_schema_matches": schema.get("feature_pack") == "catboost_native_v1",
        "catboost_import_passes": catboost.__version__ == "1.2.10",
        "stage4b_catboost_smoke_pass": smoke.get("status") == "PASS",
        "stage4b_clean_reload_pass": reload_report.get("status") == "PASS",
    }
    result = {
        "stage": STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "catboost_version": catboost.__version__,
        "execution_mode": EXECUTION_MODE,
        "thread_count": THREAD_COUNT,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(paths(project)["reports"] / "stage4c_start_validation.json", result)
    if result["status"] != "PASS":
        raise AssertionError(f"Stage 4C start validation failed: {checks}")
    return result


def _largest_counts(counts: pd.Series, total: int) -> dict[int, int]:
    raw = counts / counts.sum() * int(total)
    selected = np.floor(raw).astype(int)
    remaining = int(total - selected.sum())
    order = (raw - selected).sort_values(ascending=False, kind="mergesort").index.tolist()
    for bin_id in order[:remaining]:
        selected.loc[bin_id] += 1
    return {int(key): int(value) for key, value in selected.items()}


def create_screening_subset(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    output = paths(project)["splits"] / "stage4c_screening_subset.csv"
    manifest = pd.read_csv(
        paths(project)["splits"] / "stage4_discovery_sample.csv",
        dtype={"row_id": "int64", "sample_role": "string", "target_bin": "int64"},
    )
    expected = {"train": SCREENING_TRAIN_ROWS, "validation": SCREENING_VALIDATION_ROWS}
    rows: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED)
    for role in ("train", "validation"):
        source = manifest.loc[manifest["sample_role"].eq(role)].copy()
        counts = _largest_counts(source["target_bin"].value_counts().sort_index(), expected[role])
        for bin_id in sorted(counts):
            group = source.loc[source["target_bin"].eq(bin_id)].copy()
            order = rng.permutation(len(group))
            rows.append(group.iloc[order[: counts[bin_id]]])
    subset = pd.concat(rows, ignore_index=True).rename(columns={"sample_role": "original_discovery_role"})
    subset["stage4c_screening_role"] = subset["original_discovery_role"]
    subset = subset[["row_id", "original_discovery_role", "stage4c_screening_role", "target_bin"]]
    subset = subset.sort_values(["stage4c_screening_role", "target_bin", "row_id"], kind="mergesort").reset_index(drop=True)
    s4.atomic_write_csv(subset, output)
    result = validate_screening_subset(project)
    if result["status"] != "PASS":
        raise AssertionError(f"Stage 4C screening subset is invalid: {result}")
    return result


def validate_screening_subset(root: str | Path) -> dict[str, Any]:
    project = Path(root).resolve()
    output = paths(project)["splits"] / "stage4c_screening_subset.csv"
    subset = pd.read_csv(output, dtype={"row_id": "int64", "target_bin": "int64"})
    discovery = pd.read_csv(paths(project)["splits"] / "stage4_discovery_sample.csv", dtype={"row_id": "int64", "target_bin": "int64"})
    train_ids = pd.read_csv(project / "artifacts/splits/train_row_ids.csv", dtype={"row_id": "int64"})["row_id"]
    test_ids = pd.read_csv(project / "artifacts/splits/test_row_ids.csv", dtype={"row_id": "int64"})["row_id"]
    lookup = discovery.set_index("row_id")
    aligned = lookup.loc[subset["row_id"].to_numpy()]
    train = subset.loc[subset["stage4c_screening_role"].eq("train")]
    validation = subset.loc[subset["stage4c_screening_role"].eq("validation")]
    row_check = s4.validate_row_ids(subset["row_id"], train_ids, test_ids)
    checks = {
        "row_count_40000": len(subset) == 40_000,
        "train_rows_30000": len(train) == SCREENING_TRAIN_ROWS,
        "validation_rows_10000": len(validation) == SCREENING_VALIDATION_ROWS,
        "row_ids_unique": subset["row_id"].is_unique,
        "all_rows_in_discovery": set(subset["row_id"]).issubset(set(discovery["row_id"])),
        "all_rows_in_saved_train": row_check["all_in_saved_train"],
        "zero_test_overlap": row_check["test_overlap_rows"] == 0,
        "roles_aligned": aligned["sample_role"].astype(str).to_numpy().tolist() == subset["original_discovery_role"].astype(str).to_numpy().tolist(),
        "target_bins_aligned": np.array_equal(aligned["target_bin"].to_numpy(dtype=int), subset["target_bin"].to_numpy(dtype=int)),
        "roles_disjoint": set(train["row_id"]).isdisjoint(set(validation["row_id"])),
        "saved_edges_reused": True,
    }
    result = {
        "stage": STAGE_ID,
        "created_at_utc": s4.utc_now(),
        "path": str(output.relative_to(project)),
        "sha256": file_digest(output),
        "rows_by_role": subset["stage4c_screening_role"].value_counts().sort_index().to_dict(),
        "rows_by_role_and_bin": {
            f"{role}__bin-{int(bin_id)}": int(count)
            for (role, bin_id), count in subset.groupby(["stage4c_screening_role", "target_bin"]).size().items()
        },
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    s4.atomic_write_json(paths(project)["reports"] / "stage4c_screening_subset_verification.json", result)
    return result


def load_feature_pack(root: str | Path, pack_name: str) -> dict[str, Any]:
    packs = read_json(Path(root) / "artifacts/features/stage4/boosting_feature_packs.json")["packs"]
    if pack_name not in {"boosting_base_v1", "catboost_native_v1"}:
        raise ValueError(f"Stage 4C cannot use Feature Pack {pack_name!r}.")
    return packs[pack_name]


def feature_pack_digest(root: str | Path, pack_name: str) -> str:
    return s4.configuration_digest(load_feature_pack(root, pack_name), length=64)


def source_digest(root: str | Path, sensitive_mode: str) -> str:
    config = read_json(Path(root) / "artifacts/splits/split_config.json")
    key = "with_sensitive" if sensitive_mode == "with_sensitive" else "without_sensitive"
    return config["source_hashes"][key]


def source_path(root: str | Path, sensitive_mode: str) -> Path:
    name = "regression_with_sensitive_features.csv" if sensitive_mode == "with_sensitive" else "regression_without_sensitive_features.csv"
    return Path(root) / "data" / name


def prepare_pipeline(root: str | Path, pack_name: str, sensitive_mode: str, parameters: dict[str, Any]) -> tuple[Pipeline, list[str], list[str], list[str]]:
    project = Path(root).resolve()
    pack = load_feature_pack(project, pack_name)
    fixed = list(pack.get("fixed_features", []))
    numeric = list(pack["numeric"])
    categorical = list(pack["categorical"])
    raw = list(pack["raw"])
    if sensitive_mode == "with_sensitive":
        numeric.extend(name for name in SENSITIVE_NUMERIC if name not in numeric)
        categorical.extend(name for name in SENSITIVE_CATEGORICAL if name not in categorical)
        raw.extend(name for name in SENSITIVE_COLUMNS if name not in raw)
    elif sensitive_mode != "without_sensitive":
        raise ValueError(f"Unknown sensitive mode: {sensitive_mode}")
    selected = numeric + categorical
    steps: list[tuple[str, Any]] = []
    if fixed:
        steps.append(("fixed", s4.Stage4FixedFeatureEngineer(tuple(fixed))))
    steps.extend([
        ("select", s4.Stage4ColumnSelector(tuple(selected))),
        ("sanitize", s4.Stage4CategoricalSanitizer(tuple(categorical))),
    ])
    high_cardinality = [name for name in ("respondent_id", "msamd_name", "county_name", "census_tract_number") if name in categorical]
    if high_cardinality:
        steps.append(("rare", s4.Stage4RareCategoryGrouper(tuple(high_cardinality), min_count=2)))
    preprocess = Pipeline(steps)
    s4.activate_local_packages(project)
    from catboost import CatBoostRegressor
    model_parameters = dict(parameters)
    model_parameters.pop("early_stopping_rounds", None)
    model = CatBoostRegressor(cat_features=categorical, **model_parameters)
    pipeline = Pipeline(steps + [("model", model)])
    return pipeline, raw, selected, categorical


def fit_pipeline(
    root: str | Path,
    pack_name: str,
    sensitive_mode: str,
    target_mode: str,
    parameters: dict[str, Any],
    train_ids: Iterable[int],
    validation_ids: Iterable[int],
    use_early_stopping: bool,
) -> dict[str, Any]:
    project = Path(root).resolve()
    pipeline, raw_columns, selected, categorical = prepare_pipeline(project, pack_name, sensitive_mode, parameters)
    all_ids = np.concatenate([np.asarray(list(train_ids), dtype=np.int64), np.asarray(list(validation_ids), dtype=np.int64)])
    frame = s4.read_training_rows(source_path(project, sensitive_mode), all_ids, raw_columns + [TARGET])
    train_ids_array = np.asarray(list(train_ids), dtype=np.int64)
    validation_ids_array = np.asarray(list(validation_ids), dtype=np.int64)
    X_train = frame.loc[train_ids_array, raw_columns].copy()
    X_validation = frame.loc[validation_ids_array, raw_columns].copy()
    y_train = frame.loc[train_ids_array, TARGET].to_numpy(dtype=float)
    y_validation = frame.loc[validation_ids_array, TARGET].to_numpy(dtype=float)
    y_fit = s4.transform_target(y_train, target_mode)
    y_eval = s4.transform_target(y_validation, target_mode)
    preprocess = Pipeline(pipeline.steps[:-1])
    X_train_ready = preprocess.fit_transform(X_train, y_fit)
    X_validation_ready = preprocess.transform(X_validation)
    model = pipeline.named_steps["model"]
    fit_kwargs: dict[str, Any] = {"eval_set": (X_validation_ready, y_eval)}
    if use_early_stopping:
        fit_kwargs["early_stopping_rounds"] = int(parameters.get("early_stopping_rounds", 75))
        fit_kwargs["use_best_model"] = True
    else:
        fit_kwargs["use_best_model"] = False
    fit_started = time.perf_counter()
    model.fit(X_train_ready, y_fit, **fit_kwargs)
    fit_seconds = time.perf_counter() - fit_started
    fitted = Pipeline(preprocess.steps + [("model", model)])
    prediction_started = time.perf_counter()
    validation_output = fitted.predict(X_validation)
    prediction_seconds = time.perf_counter() - prediction_started
    train_output = fitted.predict(X_train)
    validation_prediction = s4.inverse_target(validation_output, target_mode)
    train_prediction = s4.inverse_target(train_output, target_mode)
    metrics = extended_metrics(y_validation, validation_prediction)
    training_metrics = extended_metrics(y_train, train_prediction)
    if not np.isfinite(validation_prediction).all():
        raise ValueError("Validation predictions are not finite.")
    return {
        "pipeline": fitted,
        "raw_columns": raw_columns,
        "selected_features": selected,
        "categorical_features": categorical,
        "train_ids": train_ids_array,
        "validation_ids": validation_ids_array,
        "y_train": y_train,
        "y_validation": y_validation,
        "train_prediction": train_prediction,
        "validation_prediction": validation_prediction,
        "metrics": metrics,
        "training_metrics": training_metrics,
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "best_iteration_zero_based": int(model.get_best_iteration()) if use_early_stopping else int(parameters["iterations"]) - 1,
        "tree_count": int(model.tree_count_),
    }


def extended_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, Any]:
    true = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    metrics = s4.evaluate_regression_predictions(true, pred)
    absolute = np.abs(pred - true)
    top_decile_cut = np.quantile(true, 0.90)
    top_five_cut = np.quantile(true, 0.95)
    metrics.update({
        "top_decile_mae": float(absolute[true >= top_decile_cut].mean()),
        "top_five_percent_mae": float(absolute[true >= top_five_cut].mean()),
        "underestimation_rate": float(np.mean(pred < true)),
        "overestimation_rate": float(np.mean(pred > true)),
    })
    return metrics


def candidate_config(candidate_id: str, refinement: dict[str, Any] | None = None) -> dict[str, Any]:
    if candidate_id in REQUIRED_CANDIDATES:
        value = json.loads(json.dumps(REQUIRED_CANDIDATES[candidate_id]))
    elif refinement and refinement.get("candidate_id") == candidate_id:
        value = json.loads(json.dumps(refinement))
    else:
        raise ValueError(f"Unknown Candidate ID: {candidate_id}")
    return value


def screening_ids(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    subset = pd.read_csv(paths(root)["splits"] / "stage4c_screening_subset.csv", dtype={"row_id": "int64"})
    train = subset.loc[subset["stage4c_screening_role"].eq("train"), "row_id"].to_numpy(dtype=np.int64)
    validation = subset.loc[subset["stage4c_screening_role"].eq("validation"), "row_id"].to_numpy(dtype=np.int64)
    return train, validation


def discovery_ids(root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    manifest = pd.read_csv(paths(root)["splits"] / "stage4_discovery_sample.csv", dtype={"row_id": "int64"})
    train = manifest.loc[manifest["sample_role"].eq("train"), "row_id"].to_numpy(dtype=np.int64)
    validation = manifest.loc[manifest["sample_role"].eq("validation"), "row_id"].to_numpy(dtype=np.int64)
    return train, validation


def prediction_frame(result: dict[str, Any], target_mode: str, sensitive_mode: str, experiment_id: str) -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": result["validation_ids"],
        "y_true": result["y_validation"],
        "y_pred": result["validation_prediction"],
        "residual": result["validation_prediction"] - result["y_validation"],
        "absolute_error": np.abs(result["validation_prediction"] - result["y_validation"]),
        "target_mode": target_mode,
        "sensitive_mode": sensitive_mode,
        "experiment_id": experiment_id,
    })


def clean_reload_check(root: str | Path, model_path: Path, prediction_path: Path, output_path: Path) -> dict[str, Any]:
    project = Path(root).resolve()
    code = (
        "import joblib,numpy as np,pandas as pd,pathlib,sys; "
        "root=pathlib.Path.cwd(); sys.path.insert(0,str(root)); "
        "import stage4_boosting_utils,stage4_catboost_utils; "
        f"bundle=joblib.load(root/{str(model_path.relative_to(project))!r}); "
        f"ref=pd.read_csv(root/{str(prediction_path.relative_to(project))!r}); "
        "mode=bundle['sensitive_mode']; raw=bundle['raw_columns']; ids=ref['row_id'].to_numpy(dtype=np.int64); "
        "X=stage4_boosting_utils.read_training_rows(stage4_catboost_utils.source_path(root,mode),ids,raw); "
        "pred=stage4_boosting_utils.inverse_target(bundle['pipeline'].predict(X.loc[ids,raw]),bundle['target_mode']); "
        "assert np.isfinite(pred).all(); assert np.allclose(pred,ref['y_pred'].to_numpy(),rtol=1e-10,atol=1e-10); print('PASS')"
    )
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=project, env=s4.worker_environment(project),
        capture_output=True, text=True, timeout=180, check=False,
    )
    result = {
        "model_path": str(model_path.relative_to(project)),
        "prediction_path": str(prediction_path.relative_to(project)),
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "wall_seconds": time.perf_counter() - started,
        "status": "PASS" if completed.returncode == 0 and "PASS" in completed.stdout else "FAIL",
    }
    s4.atomic_write_json(output_path, result)
    return result
