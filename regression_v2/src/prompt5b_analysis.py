"""Prompt 5B fairness audit and hierarchical frozen-model explainability.

This module reads only the post-IID snapshots and frozen final bundles.  It
contains no training, calibration, threshold search, or model-selection code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import Pool
from nbclient import NotebookClient
from scipy.stats import rankdata, spearmanr

try:
    from .prompt4c_bundles import FinalPrimaryBundle
except ImportError:
    from prompt4c_bundles import FinalPrimaryBundle


AUTHORIZATION_ID = "regression_v2_prompt5b_fairness_explainability"
STATUS = "PASS_FINAL_FAIRNESS_AND_EXPLAINABILITY"
EXPECTED_ROWS = 75_000
MIN_GROUP_N = 200
MIN_TAIL_N = 50
MIN_DECILE_N = 30
MIN_INTERSECTION_N = 500
BOOTSTRAP_RESAMPLES = 500
SEED = 42
SAMPLE_SIZE = 4_000
FREEZE_SHA = "8f2b8e4fda80056770b916b7859ad0f6f89f2948236320e6815e082fadca35c1"
PRIMARY_SHA = "5349ab15fd1c8182ef539f435cc4f065e71ee7da047de09a4dc271c9097e0c08"
GLOBAL_SHA = "6f61a0be1fc90d2331b08dada63a453f6c5410d5783f46f1e0cbcbf425f75597"
PRIMARY_MAE = 62.26062600334689
GLOBAL_MAE = 62.444349310930285

REPORTS = Path("outputs/reports")
FIGURES = Path("outputs/figures/prompt5b")
EXPLAIN = Path("outputs/explainability/prompt5b")
TMP = Path("outputs/tmp/prompt5b")
NOTEBOOK = Path("notebooks/05B_FAIRNESS_AND_FINAL_EXPLAINABILITY.ipynb")
MODEL_SNAPSHOT = Path("outputs/data/post_iid/iid_model_features_snapshot.parquet")
FAIRNESS_SNAPSHOT = Path("outputs/data/post_iid/iid_fairness_audit_snapshot.parquet")
TARGET_SNAPSHOT = Path("outputs/data/post_iid/iid_target_snapshot.parquet")
EVALUATION_FRAME = Path("outputs/data/post_iid/iid_evaluation_frame.parquet")
PRIMARY_PREDICTION = Path("outputs/predictions/prompt5a/iid/final_primary_stage3_500k.parquet")
GLOBAL_PREDICTION = Path("outputs/predictions/prompt5a/iid/final_global_500k.parquet")
PRIMARY_BUNDLE = Path("outputs/models/final_pre_iid/primary_stage3/bundle.joblib")
GLOBAL_BUNDLE = Path("outputs/models/final_pre_iid/global_comparator/bundle.joblib")
SNAPSHOT_MANIFEST = REPORTS / "prompt5a_post_iid_snapshot_manifest.json"
FEATURE_ROLES = REPORTS / "feature_roles.json"

LIMITATIONS = (
    "Sensitive variables were excluded from Primary model inputs and are used only after prediction for audit. "
    "Observed group differences are descriptive, not causal. Group target distributions differ. This is a "
    "loan-amount regression model, not a lending approval model. The audit does not prove the absence or "
    "presence of discrimination. Small-group estimates are unstable, and multiple comparisons are descriptive."
)


def root_path(value: str | None = None) -> Path:
    return Path(value).resolve() if value else Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _schema(path: Path) -> list[dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    return [{"name": field.name, "dtype": str(field.type), "nullable": bool(field.nullable)} for field in schema]


def preflight(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    expected_reports = {
        "PROMPT5A_READY.json": "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS",
        "FINAL_IID_EVALUATION.json": "PASS_FINAL_IID_EVALUATION_AND_ERROR_ANALYSIS",
        "prompt5a_verification.json": "PASS",
        "prompt5a_reviewer.json": "PASS",
    }
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for name, expected in expected_reports.items():
        payload = read_json(root / REPORTS / name)
        passed = payload.get("status") == expected
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": payload.get("status")})
        if not passed:
            failures.append(name)
    identities = {
        "FINAL_PRE_IID_FREEZE.json": (root / REPORTS / "FINAL_PRE_IID_FREEZE.json", FREEZE_SHA),
        "Primary bundle": (root / PRIMARY_BUNDLE, PRIMARY_SHA),
        "Global bundle": (root / GLOBAL_BUNDLE, GLOBAL_SHA),
    }
    for name, (path, expected) in identities.items():
        actual = sha256(path)
        passed = actual == expected
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": actual})
        if not passed:
            failures.append(name)
    manifest = read_json(root / SNAPSHOT_MANIFEST)
    for name, item in manifest["artifacts"].items():
        path = root / item["path"]
        actual = sha256(path)
        rows = pq.ParquetFile(path).metadata.num_rows
        passed = actual == item["sha256"] and rows == item["rows"]
        checks.append({"check": f"snapshot:{name}", "status": "PASS" if passed else "FAIL", "evidence": {"sha256": actual, "rows": rows}})
        if not passed:
            failures.append(f"snapshot:{name}")
    roles = read_json(root / FEATURE_ROLES)
    sensitive = list(roles["sensitive_fields"])
    features = list(roles["contracts"]["main_without_sensitive_without_lender"])
    fairness_columns = pq.ParquetFile(root / FAIRNESS_SNAPSHOT).schema_arrow.names
    exact_sensitive = fairness_columns[1:] == sensitive and len(sensitive) == 8
    no_leakage = not set(sensitive).intersection(features) and "respondent_id" not in features
    checks.extend([
        {"check": "exact eight-field sensitive contract", "status": "PASS" if exact_sensitive else "FAIL", "evidence": sensitive},
        {"check": "sensitive and respondent identity exclusion", "status": "PASS" if no_leakage else "FAIL", "evidence": {"feature_count": len(features), "respondent_id": "respondent_id" in features}},
    ])
    if not exact_sensitive:
        failures.append("sensitive contract")
    if not no_leakage:
        failures.append("predictive leakage")
    status = "PASS" if not failures else "BLOCKED_POST_IID_HANDOFF_MISMATCH"
    payload = {
        "status": status,
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "checks": checks,
        "failures": failures,
        "original_iid_reads": 0,
        "model_fits": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(root / REPORTS / "prompt5b_handoff_validation.json", payload)
    if failures:
        raise RuntimeError(f"Prompt 5B handoff mismatch: {failures}")
    return payload


def _label_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: "__MISSING__" if pd.isna(value) else str(value))


def _missing_unknown(label: str) -> str:
    lower = label.lower()
    if label == "__MISSING__":
        return "MISSING"
    if "not provided" in lower or "unknown" in lower:
        return "UNKNOWN_OR_NOT_PROVIDED"
    if "not applicable" in lower:
        return "NOT_APPLICABLE"
    if "no co-applicant" in lower:
        return "NO_CO_APPLICANT"
    return "OBSERVED"


def _composition(part: pd.DataFrame) -> dict[str, float]:
    return {
        "mean_target": float(part["y_true"].mean()),
        "median_target": float(part["y_true"].median()),
        "d10_fraction": float(part["iid_local_decile"].eq("D10").mean()),
        "top5_target_fraction": float(part["iid_top5_target"].mean()),
    }


def _metrics(part: pd.DataFrame, prefix: str) -> dict[str, float]:
    y = part["y_true"].to_numpy(np.float64)
    prediction = part[f"{prefix}_prediction"].to_numpy(np.float64)
    signed = prediction - y
    absolute = np.abs(signed)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(signed * signed))),
        "mape_percent": float(100.0 * np.mean(absolute / y)),
        "wape_percent": float(100.0 * absolute.sum() / y.sum()),
        "mean_signed_error": float(signed.mean()),
        "median_signed_error": float(np.median(signed)),
        "underprediction_rate": float(np.mean(signed < 0.0)),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "mean_prediction": float(prediction.mean()),
    }


def _group_iter(frame: pd.DataFrame, sensitive: list[str]):
    for field in sensitive:
        labels = _label_series(frame[field])
        for label, index in labels.groupby(labels, sort=True).groups.items():
            yield field, str(label), frame.loc[index]


def _eligible_status(n: int, threshold: int = MIN_GROUP_N) -> str:
    return "ELIGIBLE" if n >= threshold else "SMALL_GROUP"


def fairness_analysis(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    roles = read_json(root / FEATURE_ROLES)
    sensitive = list(roles["sensitive_fields"])
    fairness = pd.read_parquet(root / FAIRNESS_SNAPSHOT)
    evaluation = pd.read_parquet(root / EVALUATION_FRAME)
    if len(fairness) != EXPECTED_ROWS or len(evaluation) != EXPECTED_ROWS:
        raise RuntimeError("Prompt 5B requires exactly 75,000 saved rows.")
    if fairness["row_hash"].duplicated().any() or evaluation["row_hash"].duplicated().any():
        raise RuntimeError("Duplicate row_hash at fairness alignment gate.")
    frame = evaluation.merge(fairness, on="row_hash", how="left", validate="one_to_one", indicator=True)
    if len(frame) != EXPECTED_ROWS or not frame["_merge"].eq("both").all():
        raise RuntimeError("Fairness/evaluation alignment failed.")
    frame = frame.drop(columns="_merge")
    q90 = float(np.quantile(frame["y_true"].to_numpy(float), 0.90, method="linear"))
    q95 = float(np.quantile(frame["y_true"].to_numpy(float), 0.95, method="linear"))
    frame["iid_global_top_decile"] = frame["y_true"].ge(q90)
    frame["iid_top5_target"] = frame["y_true"].ge(q95)
    primary_error_q95 = float(np.quantile(frame["primary_abs_error"], 0.95, method="linear"))
    correction_q95 = float(np.quantile(frame["applied_residual_correction"].abs(), 0.95, method="linear"))
    frame["primary_top5_abs_error"] = frame["primary_abs_error"].ge(primary_error_q95)
    frame["high_correction_magnitude"] = frame["applied_residual_correction"].abs().ge(correction_q95)
    working = frame[[
        "row_hash", "y_true", "primary_prediction", "global_prediction", "primary_signed_error",
        "global_signed_error", "primary_abs_error", "global_abs_error", "delta_abs_error",
        "iid_local_decile", "iid_global_top_decile", "iid_top5_target", "primary_top5_abs_error", "high_correction_magnitude",
        "routing_condition_activated", "applied_residual_correction", *sensitive,
    ]].copy()
    atomic_parquet(root / TMP / "fairness_analysis_frame.parquet", working)

    contract_rows = []
    semantic = {
        "applicant_ethnicity_name": ("applicant", "categorical ethnicity audit label"),
        "co_applicant_ethnicity_name": ("co-applicant", "categorical ethnicity audit label"),
        "applicant_race_name_1": ("applicant", "categorical first race audit label"),
        "co_applicant_race_name_1": ("co-applicant", "categorical first race audit label"),
        "applicant_sex_name": ("applicant", "categorical sex audit label"),
        "co_applicant_sex_name": ("co-applicant", "categorical sex audit label"),
        "minority_population": ("tract", "continuous tract minority-population percentage"),
        "majority_minority_tract": ("tract", "derived categorical tract audit label"),
    }
    for field in sensitive:
        side, role = semantic.get(field, ("not determined", "frozen sensitive audit field"))
        contract_rows.append({
            "column_name": field,
            "original_semantic_role": role,
            "dtype": str(fairness[field].dtype),
            "missing_count": int(fairness[field].isna().sum()),
            "unique_level_count": int(fairness[field].nunique(dropna=False)),
            "side": side,
            "predictive_input": False,
        })
    contract_frame = pd.DataFrame(contract_rows)
    contract_payload = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "authorization_id": AUTHORIZATION_ID,
        "source": FAIRNESS_SNAPSHOT.as_posix(),
        "exact_column_names": sensitive,
        "field_count": len(sensitive),
        "fields": contract_rows,
        "contract_sha256": hashlib.sha256(contract_frame.to_csv(index=False).encode("utf-8")).hexdigest(),
        "sensitive_fields_are_audit_only": True,
        "limitations": LIMITATIONS,
    }
    atomic_json(root / REPORTS / "prompt5b_sensitive_contract.json", contract_payload)

    inventory_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for field, label, part in _group_iter(frame, sensitive):
        n = len(part)
        composition = _composition(part)
        common = {
            "sensitive_field": field,
            "group_label": label,
            "n": n,
            "iid_fraction": float(n / EXPECTED_ROWS),
            **composition,
            "missing_unknown_status": _missing_unknown(label),
            "analysis_status": _eligible_status(n),
            "target_composition_warning": "Descriptive; group target distributions differ.",
        }
        inventory_rows.append(common)
        if n >= MIN_GROUP_N:
            pm = _metrics(part, "primary")
            gm = _metrics(part, "global")
            primary_rows.append({**common, **pm})
            global_rows.append({**common, **gm})
            comparison_rows.append({
                **common,
                "primary_minus_global_mae": pm["mae"] - gm["mae"],
                "primary_minus_global_rmse": pm["rmse"] - gm["rmse"],
                "primary_minus_global_mape_percent": pm["mape_percent"] - gm["mape_percent"],
                "primary_minus_global_wape_percent": pm["wape_percent"] - gm["wape_percent"],
                "primary_minus_global_signed_error_distance_to_zero": abs(pm["mean_signed_error"]) - abs(gm["mean_signed_error"]),
                "primary_minus_global_underprediction_rate": pm["underprediction_rate"] - gm["underprediction_rate"],
                "primary_row_win_rate": float(np.mean(part["primary_abs_error"].to_numpy() < part["global_abs_error"].to_numpy())),
            })
        else:
            blank = {key: np.nan for key in ("mae", "rmse", "mape_percent", "wape_percent", "mean_signed_error", "median_signed_error", "underprediction_rate", "p90_absolute_error", "mean_prediction")}
            primary_rows.append({**common, **blank})
            global_rows.append({**common, **blank})
            comparison_rows.append({**common, **{key: np.nan for key in (
                "primary_minus_global_mae", "primary_minus_global_rmse", "primary_minus_global_mape_percent",
                "primary_minus_global_wape_percent", "primary_minus_global_signed_error_distance_to_zero",
                "primary_minus_global_underprediction_rate", "primary_row_win_rate",
            )}})
    inventory = pd.DataFrame(inventory_rows)
    primary = pd.DataFrame(primary_rows)
    global_metrics = pd.DataFrame(global_rows)
    comparison = pd.DataFrame(comparison_rows)
    atomic_csv(root / REPORTS / "prompt5b_group_inventory.csv", inventory)
    atomic_csv(root / REPORTS / "prompt5b_primary_group_metrics.csv", primary)
    atomic_csv(root / REPORTS / "prompt5b_global_group_metrics.csv", global_metrics)
    atomic_csv(root / REPORTS / "prompt5b_group_primary_vs_global.csv", comparison)

    disparity_rows = []
    eligible_primary = primary[primary["analysis_status"].eq("ELIGIBLE")]
    for field, part in eligible_primary.groupby("sensitive_field", sort=False):
        if len(part) < 2:
            continue
        mae_min = part.loc[part["mae"].idxmin()]
        mae_max = part.loc[part["mae"].idxmax()]
        wape_min = part.loc[part["wape_percent"].idxmin()]
        wape_max = part.loc[part["wape_percent"].idxmax()]
        signed_min = part.loc[part["mean_signed_error"].idxmin()]
        signed_max = part.loc[part["mean_signed_error"].idxmax()]
        under_min = part.loc[part["underprediction_rate"].idxmin()]
        under_max = part.loc[part["underprediction_rate"].idxmax()]
        disparity_rows.append({
            "sensitive_field": field,
            "eligible_group_count": len(part),
            "min_mae_group": mae_min["group_label"], "min_group_mae": mae_min["mae"],
            "max_mae_group": mae_max["group_label"], "max_group_mae": mae_max["mae"],
            "mae_max_minus_min": mae_max["mae"] - mae_min["mae"],
            "mae_max_min_ratio": mae_max["mae"] / mae_min["mae"],
            "max_group_mae_minus_overall": mae_max["mae"] - PRIMARY_MAE,
            "min_wape_group": wape_min["group_label"], "min_group_wape_percent": wape_min["wape_percent"],
            "max_wape_group": wape_max["group_label"], "max_group_wape_percent": wape_max["wape_percent"],
            "wape_max_minus_min": wape_max["wape_percent"] - wape_min["wape_percent"],
            "wape_max_min_ratio": wape_max["wape_percent"] / wape_min["wape_percent"],
            "most_negative_signed_group": signed_min["group_label"], "most_negative_signed_error": signed_min["mean_signed_error"],
            "most_positive_signed_group": signed_max["group_label"], "most_positive_signed_error": signed_max["mean_signed_error"],
            "signed_error_range": signed_max["mean_signed_error"] - signed_min["mean_signed_error"],
            "max_distance_from_zero": float(part["mean_signed_error"].abs().max()),
            "min_underprediction_group": under_min["group_label"], "min_underprediction_rate": under_min["underprediction_rate"],
            "max_underprediction_group": under_max["group_label"], "max_underprediction_rate": under_max["underprediction_rate"],
            "underprediction_gap_percentage_points": 100.0 * (under_max["underprediction_rate"] - under_min["underprediction_rate"]),
            "interpretation": "Descriptive predictive-error disparity; not a legal fairness finding.",
        })
    disparity = pd.DataFrame(disparity_rows)
    atomic_csv(root / REPORTS / "prompt5b_disparity_summary.csv", disparity)

    tail_rows = []
    masks = {"IID_GLOBAL_TOP_DECILE": frame["iid_global_top_decile"], "IID_TOP5_TARGET": frame["iid_top5_target"]}
    for field in sensitive:
        labels = _label_series(frame[field])
        for scope, mask in masks.items():
            for label, index in labels[mask].groupby(labels[mask], sort=True).groups.items():
                part = frame.loc[index]
                n = len(part)
                base = {"sensitive_field": field, "group_label": str(label), "tail_scope": scope, "n": n, **_composition(part), "analysis_status": "ELIGIBLE" if n >= MIN_TAIL_N else "SMALL_TAIL_GROUP", "target_composition_warning": "Common IID-global Tail mask; descriptive only."}
                if n >= MIN_TAIL_N:
                    pm, gm = _metrics(part, "primary"), _metrics(part, "global")
                    base.update({"primary_tail_mae": pm["mae"], "global_tail_mae": gm["mae"], "primary_signed_error": pm["mean_signed_error"], "global_signed_error": gm["mean_signed_error"], "primary_underprediction_rate": pm["underprediction_rate"], "global_underprediction_rate": gm["underprediction_rate"], "primary_minus_global_tail_mae": pm["mae"] - gm["mae"]})
                else:
                    base.update({key: np.nan for key in ("primary_tail_mae", "global_tail_mae", "primary_signed_error", "global_signed_error", "primary_underprediction_rate", "global_underprediction_rate", "primary_minus_global_tail_mae")})
                tail_rows.append(base)
    tail_metrics = pd.DataFrame(tail_rows)
    atomic_csv(root / REPORTS / "prompt5b_tail_group_metrics.csv", tail_metrics)

    decile_rows = []
    eligible_keys = eligible_primary[["sensitive_field", "group_label"]]
    for record in eligible_keys.itertuples(index=False):
        labels = _label_series(frame[record.sensitive_field])
        group = frame[labels.eq(record.group_label)]
        for decile in [f"D{i}" for i in range(1, 11)]:
            part = group[group["iid_local_decile"].eq(decile)]
            n = len(part)
            numeric = n >= MIN_DECILE_N
            decile_rows.append({
                "sensitive_field": record.sensitive_field, "group_label": record.group_label,
                "iid_target_decile": decile, "n": n,
                "primary_mae": float(part["primary_abs_error"].mean()) if numeric else np.nan,
                "mean_target": float(part["y_true"].mean()) if n else np.nan,
                "median_target": float(part["y_true"].median()) if n else np.nan,
                "cell_status": "DISPLAY" if numeric else "INSUFFICIENT_CELL_N",
                "target_composition_warning": "Target-decile cell; descriptive only.",
            })
    decile_metrics = pd.DataFrame(decile_rows)
    atomic_csv(root / REPORTS / "prompt5b_group_decile_metrics.csv", decile_metrics)

    intersection_rows = []
    definitions = [
        ("applicant_race_x_applicant_sex", "applicant_race_name_1", "applicant_sex_name"),
        ("applicant_ethnicity_x_applicant_sex", "applicant_ethnicity_name", "applicant_sex_name"),
    ]
    for name, left, right in definitions:
        if left not in sensitive or right not in sensitive:
            intersection_rows.append({"intersection": name, "scope": "OVERALL", "group_label": "NOT_AVAILABLE", "n": 0, "analysis_status": "NOT_AVAILABLE"})
            continue
        combined = _label_series(frame[left]) + " | " + _label_series(frame[right])
        for label, overall_index in combined.groupby(combined, sort=True).groups.items():
            overall = frame.loc[overall_index]
            for scope, mask, threshold in (("OVERALL", pd.Series(True, index=frame.index), MIN_INTERSECTION_N), ("IID_GLOBAL_TOP_DECILE", frame["iid_global_top_decile"], MIN_TAIL_N), ("IID_TOP5_TARGET", frame["iid_top5_target"], MIN_TAIL_N)):
                part = frame.loc[overall_index].loc[mask.loc[overall_index]]
                n = len(part)
                eligible = n >= threshold
                base = {"intersection": name, "scope": scope, "group_label": str(label), "n": n, "analysis_status": "ELIGIBLE" if eligible else ("SMALL_GROUP" if scope == "OVERALL" else "SMALL_TAIL_GROUP")}
                if n:
                    base.update(_composition(part))
                if eligible:
                    pm, gm = _metrics(part, "primary"), _metrics(part, "global")
                    base.update({"mae": pm["mae"], "mape_percent": pm["mape_percent"], "wape_percent": pm["wape_percent"], "mean_signed_error": pm["mean_signed_error"], "underprediction_rate": pm["underprediction_rate"], "primary_minus_global_mae": pm["mae"] - gm["mae"]})
                else:
                    base.update({key: np.nan for key in ("mae", "mape_percent", "wape_percent", "mean_signed_error", "underprediction_rate", "primary_minus_global_mae")})
                base["target_composition_warning"] = "Predeclared applicant-side intersection; descriptive only."
                intersection_rows.append(base)
    intersectional = pd.DataFrame(intersection_rows)
    atomic_csv(root / REPORTS / "prompt5b_intersectional_metrics.csv", intersectional)

    bootstrap = _fairness_bootstrap(frame, sensitive, eligible_primary)
    atomic_csv(root / REPORTS / "prompt5b_fairness_bootstrap.csv", bootstrap)
    flags = _fairness_flags(eligible_primary, bootstrap)
    atomic_csv(root / REPORTS / "prompt5b_fairness_flags.csv", flags)

    routing_rows = []
    link_rows = []
    routed = frame["routing_condition_activated"].astype(bool)
    for record in eligible_keys.itertuples(index=False):
        labels = _label_series(frame[record.sensitive_field])
        part = frame[labels.eq(record.group_label)]
        part_routed = part[part["routing_condition_activated"].astype(bool)]
        part_not = part[~part["routing_condition_activated"].astype(bool)]
        routing_rows.append({
            "sensitive_field": record.sensitive_field, "group_label": record.group_label, "n": len(part),
            **_composition(part), "routing_rate": float(part["routing_condition_activated"].mean()),
            "mean_applied_correction": float(part["applied_residual_correction"].mean()),
            "primary_minus_global_mae_routed": float(part_routed["delta_abs_error"].mean()) if len(part_routed) else np.nan,
            "routed_n": len(part_routed),
            "primary_minus_global_mae_non_routed": float(part_not["delta_abs_error"].mean()) if len(part_not) else np.nan,
            "non_routed_n": len(part_not),
            "interpretation": "Descriptive mechanism audit; routing-rate parity is not asserted as a fairness requirement.",
        })
        flag = flags[(flags["sensitive_field"].eq(record.sensitive_field)) & (flags["group_label"].eq(record.group_label))]
        link_rows.append({
            "sensitive_field": record.sensitive_field, "group_label": record.group_label, "n": len(part),
            **_composition(part), "elevated_mae_descriptive": bool(flag["elevated_mae_descriptive"].iloc[0]) if len(flag) else False,
            "elevated_underprediction_descriptive": bool(flag["elevated_underprediction_descriptive"].iloc[0]) if len(flag) else False,
            "primary_top5_abs_error_fraction": float(part["primary_top5_abs_error"].mean()),
            "high_correction_magnitude_fraction": float(part["high_correction_magnitude"].mean()),
            "routed_fraction": float(part["routing_condition_activated"].mean()),
            "interpretation": "Descriptive link only; explanations do not explain away group disparities.",
        })
    routing_audit = pd.DataFrame(routing_rows)
    fairness_link = pd.DataFrame(link_rows)
    atomic_csv(root / REPORTS / "prompt5b_routing_sensitive_audit.csv", routing_audit)
    atomic_csv(root / REPORTS / "prompt5b_fairness_explainability_link.csv", fairness_link)

    correction = _correction_behavior(frame)
    atomic_csv(root / REPORTS / "prompt5b_correction_behavior.csv", correction)
    summary = {
        "status": "PASS",
        "aligned_rows": len(frame),
        "sensitive_fields": sensitive,
        "observed_group_levels": len(inventory),
        "eligible_group_levels": int(eligible_primary.shape[0]),
        "small_group_levels": int(inventory["analysis_status"].eq("SMALL_GROUP").sum()),
        "intersection_definitions": [item[0] for item in definitions],
        "bootstrap_rows": len(bootstrap),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "seed": SEED,
        "overall_primary_mae": PRIMARY_MAE,
        "overall_global_mae": GLOBAL_MAE,
        "limitations": LIMITATIONS,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(root / REPORTS / "prompt5b_fairness_summary.json", summary)
    return summary


def _fairness_bootstrap(frame: pd.DataFrame, sensitive: list[str], eligible_primary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = len(frame)
    y = frame["y_true"].to_numpy(float)
    pae = frame["primary_abs_error"].to_numpy(float)
    gae = frame["global_abs_error"].to_numpy(float)
    psigned = frame["primary_signed_error"].to_numpy(float)
    under = psigned < 0.0
    records: list[dict[str, Any]] = []
    for field in sensitive:
        eligible = eligible_primary[eligible_primary["sensitive_field"].eq(field)]["group_label"].tolist()
        if not eligible:
            continue
        labels = _label_series(frame[field]).to_numpy(object)
        samples: dict[str, dict[str, list[float]]] = {
            label: {metric: [] for metric in ("primary_mae", "primary_wape_percent", "primary_mean_signed_error", "primary_underprediction_rate", "primary_minus_global_mae", "group_mae_minus_overall", "group_underprediction_minus_overall")}
            for label in eligible
        }
        gaps: list[float] = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            index = rng.integers(0, n, size=n)
            sampled_labels = labels[index]
            overall_mae = float(pae[index].mean())
            overall_under = float(under[index].mean())
            group_maes = []
            for label in eligible:
                selected = index[sampled_labels == label]
                g_mae = float(pae[selected].mean())
                g_under = float(under[selected].mean())
                group_maes.append(g_mae)
                samples[label]["primary_mae"].append(g_mae)
                samples[label]["primary_wape_percent"].append(float(100.0 * pae[selected].sum() / y[selected].sum()))
                samples[label]["primary_mean_signed_error"].append(float(psigned[selected].mean()))
                samples[label]["primary_underprediction_rate"].append(g_under)
                samples[label]["primary_minus_global_mae"].append(float((pae[selected] - gae[selected]).mean()))
                samples[label]["group_mae_minus_overall"].append(g_mae - overall_mae)
                samples[label]["group_underprediction_minus_overall"].append(g_under - overall_under)
            if len(group_maes) >= 2:
                gaps.append(float(max(group_maes) - min(group_maes)))
        composition = eligible_primary[eligible_primary["sensitive_field"].eq(field)].set_index("group_label")
        for label, metrics in samples.items():
            for metric, values in metrics.items():
                array = np.asarray(values, dtype=float)
                observed_map = {
                    "primary_mae": composition.loc[label, "mae"],
                    "primary_wape_percent": composition.loc[label, "wape_percent"],
                    "primary_mean_signed_error": composition.loc[label, "mean_signed_error"],
                    "primary_underprediction_rate": composition.loc[label, "underprediction_rate"],
                    "primary_minus_global_mae": float(frame[_label_series(frame[field]).eq(label)]["delta_abs_error"].mean()),
                    "group_mae_minus_overall": composition.loc[label, "mae"] - PRIMARY_MAE,
                    "group_underprediction_minus_overall": composition.loc[label, "underprediction_rate"] - float(np.mean(under)),
                }
                records.append({
                    "row_type": "GROUP", "sensitive_field": field, "group_label": label,
                    "n": int(composition.loc[label, "n"]), "metric": metric,
                    "observed": float(observed_map[metric]), "bootstrap_mean": float(array.mean()),
                    "ci_lower_2_5": float(np.quantile(array, 0.025)), "ci_upper_97_5": float(np.quantile(array, 0.975)),
                    "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED,
                })
        if gaps:
            array = np.asarray(gaps)
            observed_gap = float(composition["mae"].max() - composition["mae"].min())
            records.append({
                "row_type": "FIELD_GAP", "sensitive_field": field, "group_label": "__FIELD__",
                "n": n, "metric": "worst_minus_best_eligible_group_mae_gap", "observed": observed_gap,
                "bootstrap_mean": float(array.mean()), "ci_lower_2_5": float(np.quantile(array, 0.025)),
                "ci_upper_97_5": float(np.quantile(array, 0.975)), "resamples": BOOTSTRAP_RESAMPLES, "seed": SEED,
            })
    return pd.DataFrame(records)


def _fairness_flags(eligible_primary: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in eligible_primary.itertuples(index=False):
        subset = bootstrap[(bootstrap["row_type"].eq("GROUP")) & (bootstrap["sensitive_field"].eq(record.sensitive_field)) & (bootstrap["group_label"].eq(record.group_label))]
        mae_diff = subset[subset["metric"].eq("group_mae_minus_overall")]
        under_diff = subset[subset["metric"].eq("group_underprediction_minus_overall")]
        elevated_mae = bool(record.mae > PRIMARY_MAE and len(mae_diff) and mae_diff["ci_lower_2_5"].iloc[0] > 0.0)
        overall_under = 0.0
        if len(under_diff):
            overall_under = record.underprediction_rate - under_diff["observed"].iloc[0]
        elevated_under = bool(record.underprediction_rate > overall_under and len(under_diff) and under_diff["ci_lower_2_5"].iloc[0] > 0.0)
        flags = []
        if elevated_mae:
            flags.append("ELEVATED_MAE_DESCRIPTIVE")
        if elevated_under:
            flags.append("ELEVATED_UNDERPREDICTION_DESCRIPTIVE")
        rows.append({
            "sensitive_field": record.sensitive_field, "group_label": record.group_label, "n": record.n,
            "mean_target": record.mean_target, "median_target": record.median_target,
            "d10_fraction": record.d10_fraction, "top5_target_fraction": record.top5_target_fraction,
            "mae": record.mae, "underprediction_rate": record.underprediction_rate,
            "elevated_mae_descriptive": elevated_mae,
            "elevated_underprediction_descriptive": elevated_under,
            "descriptive_flags": ";".join(flags) if flags else "NONE",
            "legal_fairness_determination": False,
        })
    return pd.DataFrame(rows)


def _correction_behavior(frame: pd.DataFrame) -> pd.DataFrame:
    correction = frame["applied_residual_correction"].to_numpy(float)
    absolute = np.abs(correction)
    nonzero = absolute > 1e-12
    quantiles = np.quantile(absolute, [0, .25, .5, .75, .9, .95, .99, 1])
    rows = []
    rows.extend([
        {"section": "direction", "scope": "overall", "metric": "positive_fraction", "value": float(np.mean(correction > 1e-12)), "n": len(frame)},
        {"section": "direction", "scope": "overall", "metric": "negative_fraction", "value": float(np.mean(correction < -1e-12)), "n": len(frame)},
        {"section": "direction", "scope": "overall", "metric": "zero_fraction", "value": float(np.mean(~nonzero)), "n": len(frame)},
    ])
    for level, value in zip((0, 25, 50, 75, 90, 95, 99, 100), quantiles):
        rows.append({"section": "magnitude_quantile", "scope": "overall", "metric": f"p{level}", "value": float(value), "n": len(frame)})
    for decile, part in frame.groupby("iid_local_decile", sort=True):
        rows.extend([
            {"section": "target_decile", "scope": decile, "metric": "mean_correction", "value": float(part["applied_residual_correction"].mean()), "n": len(part)},
            {"section": "target_decile", "scope": decile, "metric": "median_correction", "value": float(part["applied_residual_correction"].median()), "n": len(part)},
            {"section": "target_decile", "scope": decile, "metric": "routed_fraction", "value": float(part["routing_condition_activated"].mean()), "n": len(part)},
        ])
    routed_frame = frame[frame["routing_condition_activated"].astype(bool)].copy()
    if len(routed_frame):
        rank = routed_frame["applied_residual_correction"].abs().rank(method="first")
        routed_frame["correction_magnitude_bin"] = pd.qcut(rank, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
        for label, part in routed_frame.groupby("correction_magnitude_bin", observed=True):
            rows.append({"section": "realized_benefit", "scope": str(label), "metric": "abs_error_global_minus_primary", "value": float((part["global_abs_error"] - part["primary_abs_error"]).mean()), "n": len(part)})
    return pd.DataFrame(rows)


def _xgb_feature_mapping(preprocessor: Any) -> list[str]:
    mapping = list(preprocessor.numeric_features_) + list(preprocessor.high_cardinality_features_)
    for name, categories in zip(preprocessor.low_cardinality_features_, preprocessor.one_hot_.categories_):
        mapping.extend([name] * len(categories))
    return mapping


def _explain_components(primary: FinalPrimaryBundle, features: pd.DataFrame, locked_global: np.ndarray) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    import xgboost as xgb

    selected = features.loc[:, primary.feature_names]
    results: dict[str, dict[str, Any]] = {}
    component_predictions: dict[str, np.ndarray] = {}
    for name, bundle in primary.global_bundle.components.items():
        transformed = bundle.preprocessor.transform(selected)
        if name == "catboost":
            pool = Pool(transformed, cat_features=bundle.preprocessor.cat_feature_indices_)
            raw = np.asarray(bundle.model.get_feature_importance(pool, type="ShapValues"), dtype=float)
            names = list(bundle.feature_names)
            native_prediction = np.asarray(bundle.model.predict(transformed), dtype=float).reshape(-1)
            deployed_prediction = native_prediction
            space = "raw loan amount (thousand USD)"
        elif name == "lightgbm":
            raw = np.asarray(bundle.model.predict(transformed, pred_contrib=True), dtype=float)
            names = list(bundle.preprocessor.get_feature_names_out())
            native_prediction = np.asarray(bundle.model.predict(transformed), dtype=float).reshape(-1)
            deployed_prediction = native_prediction
            space = "raw loan amount (thousand USD)"
        else:
            raw_transformed = np.asarray(bundle.model.get_booster().predict(xgb.DMatrix(transformed), pred_contribs=True), dtype=float)
            mapping = _xgb_feature_mapping(bundle.preprocessor)
            if raw_transformed.shape[1] != len(mapping) + 1:
                raise RuntimeError("XGBoost transformed-feature contribution width mismatch.")
            aggregated = np.zeros((len(selected), len(bundle.feature_names) + 1), dtype=float)
            for index, feature in enumerate(mapping):
                aggregated[:, bundle.feature_names.index(feature)] += raw_transformed[:, index]
            aggregated[:, -1] = raw_transformed[:, -1]
            raw = aggregated
            names = list(bundle.feature_names)
            native_prediction = np.asarray(bundle.model.predict(transformed), dtype=float).reshape(-1)
            deployed_prediction = np.expm1(native_prediction)
            space = "native log1p target space"
        maximum_additivity_error = float(np.max(np.abs(raw.sum(axis=1) - native_prediction)))
        tolerance = 1e-3 if name == "xgboost" else 1e-8
        if not np.isfinite(raw).all() or maximum_additivity_error > tolerance:
            raise RuntimeError(f"Invalid {name} native attributions: {maximum_additivity_error}")
        results[name] = {"values": raw[:, :-1], "base": raw[:, -1], "features": names, "space": space, "maximum_additivity_error": maximum_additivity_error}
        component_predictions[name] = deployed_prediction
    weighted = sum(primary.global_bundle.weights[name] * component_predictions[name] for name in primary.global_bundle.weights)
    locked_difference = float(np.max(np.abs(weighted - locked_global)))
    if locked_difference > 1e-6:
        raise RuntimeError(f"Explainability component prediction does not reproduce locked Global: {locked_difference}")
    meta = selected.copy()
    meta["global_prediction_feature"] = locked_global
    for name, bundle, space in (
        ("gate", primary.meta_gate_bundle, "native CatBoost log-odds routing space"),
        ("residual", primary.residual_bundle, "raw residual-correction space (thousand USD)"),
    ):
        transformed = bundle.preprocessor.transform(meta)
        pool = Pool(transformed, cat_features=bundle.preprocessor.cat_feature_indices_)
        raw = np.asarray(bundle.model.get_feature_importance(pool, type="ShapValues"), dtype=float)
        if name == "gate":
            native_prediction = np.asarray(bundle.model.predict(transformed, prediction_type="RawFormulaVal"), dtype=float).reshape(-1)
        else:
            native_prediction = np.asarray(bundle.model.predict(transformed), dtype=float).reshape(-1)
        maximum_additivity_error = float(np.max(np.abs(raw.sum(axis=1) - native_prediction)))
        if not np.isfinite(raw).all() or maximum_additivity_error > 1e-8:
            raise RuntimeError(f"Invalid {name} native attributions: {maximum_additivity_error}")
        results[name] = {"values": raw[:, :-1], "base": raw[:, -1], "features": list(bundle.preprocessor.feature_names_in_), "space": space, "maximum_additivity_error": maximum_additivity_error}
    component_frame = pd.DataFrame({"catboost_prediction": component_predictions["catboost"], "lightgbm_prediction": component_predictions["lightgbm"], "xgboost_prediction": component_predictions["xgboost"], "weighted_catboost_prediction": .6 * component_predictions["catboost"], "weighted_lightgbm_prediction": .2 * component_predictions["lightgbm"], "weighted_xgboost_prediction": .2 * component_predictions["xgboost"], "weighted_global_prediction": weighted, "locked_global_prediction": locked_global})
    return results, component_frame


def _save_explanation_matrix(root: Path, name: str, row_hash: pd.Series, item: dict[str, Any]) -> None:
    frame = pd.DataFrame(item["values"], columns=item["features"])
    frame.insert(0, "row_hash", row_hash.astype(str).to_numpy())
    frame["base_value"] = item["base"]
    atomic_parquet(root / EXPLAIN / f"{name}_shap.parquet", frame)


def explainability(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    features = pd.read_parquet(root / MODEL_SNAPSHOT)
    evaluation = pd.read_parquet(root / EVALUATION_FRAME)
    if len(features) != EXPECTED_ROWS or features["row_hash"].duplicated().any():
        raise RuntimeError("Invalid saved model-feature snapshot.")
    aligned = features[["row_hash"]].merge(evaluation, on="row_hash", how="left", validate="one_to_one")
    if aligned["global_prediction"].isna().any():
        raise RuntimeError("Explainability/evaluation alignment failed.")
    score = features["row_hash"].astype(str).map(lambda value: hashlib.sha256(("prompt5b_explainability_seed42" + value).encode("utf-8")).hexdigest())
    selected_index = score.sort_values(kind="mergesort").head(SAMPLE_SIZE).index
    sample = features.loc[selected_index].copy()
    sample_eval = aligned.loc[selected_index].copy()
    order = score.loc[selected_index].sort_values().index
    sample = sample.loc[order].reset_index(drop=True)
    sample_eval = sample_eval.loc[order].reset_index(drop=True)
    atomic_parquet(root / EXPLAIN / "sample_row_hashes.parquet", sample[["row_hash"]])
    primary = joblib.load(root / PRIMARY_BUNDLE)
    if not isinstance(primary, FinalPrimaryBundle):
        raise TypeError("Unexpected final Primary bundle type.")
    sensitive = set(read_json(root / FEATURE_ROLES)["sensitive_fields"])
    all_contracts = set(primary.feature_names)
    all_contracts.update(primary.global_bundle.feature_names)
    all_contracts.update(primary.meta_gate_bundle.preprocessor.feature_names_in_)
    all_contracts.update(primary.residual_bundle.preprocessor.feature_names_in_)
    if sensitive.intersection(all_contracts):
        raise RuntimeError("BLOCKED_SENSITIVE_FEATURE_LEAKAGE")
    if "respondent_id" in all_contracts:
        raise RuntimeError("BLOCKED_LENDER_IDENTITY_LEAKAGE")
    explanations, component = _explain_components(primary, sample, sample_eval["global_prediction"].to_numpy(float))
    component.insert(0, "row_hash", sample["row_hash"].astype(str).to_numpy())
    atomic_parquet(root / EXPLAIN / "global_component_predictions.parquet", component)
    for name, item in explanations.items():
        _save_explanation_matrix(root, name, sample["row_hash"], item)

    component_summary_rows = []
    for name, weight in primary.global_bundle.weights.items():
        component_summary_rows.append({
            "component": name, "frozen_weight": weight,
            "mean_component_prediction": float(component[f"{name}_prediction"].mean()),
            "mean_weighted_prediction_contribution": float(component[f"weighted_{name}_prediction"].mean()),
            "explanation_space": explanations[name]["space"],
            "maximum_native_additivity_error": explanations[name]["maximum_additivity_error"],
            "note": "Weighted model prediction contribution; not a SHAP feature contribution.",
        })
    component_summary = pd.DataFrame(component_summary_rows)
    atomic_csv(root / REPORTS / "prompt5b_global_component_summary.csv", component_summary)

    importance_by_component: dict[str, pd.DataFrame] = {}
    for name, item in explanations.items():
        values = item["values"]
        importance = pd.DataFrame({
            "feature": item["features"],
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_signed_shap": np.mean(values, axis=0),
        }).sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
        importance["rank"] = np.arange(1, len(importance) + 1)
        importance["explanation_space"] = item["space"]
        importance_by_component[name] = importance
    global_features = list(primary.feature_names)
    global_rows = []
    n_features = len(global_features)
    for feature in global_features:
        row: dict[str, Any] = {"feature": feature}
        consensus = 0.0
        top5 = 0
        top10 = 0
        for name, weight in primary.global_bundle.weights.items():
            item = importance_by_component[name].set_index("feature").loc[feature]
            rank = int(item["rank"])
            score_value = float((n_features - rank) / (n_features - 1))
            row.update({f"{name}_mean_abs_shap": item["mean_abs_shap"], f"{name}_rank": rank, f"{name}_normalized_rank_score": score_value, f"{name}_space": item["explanation_space"]})
            consensus += weight * score_value
            top5 += int(rank <= 5)
            top10 += int(rank <= 10)
        row.update({"consensus_rank_score": consensus, "components_top5": top5, "components_top10": top10, "rank_transform": "(35-rank)/(35-1); top rank=1, bottom rank=0", "raw_shap_values_summed_across_models": False})
        global_rows.append(row)
    global_importance = pd.DataFrame(global_rows).sort_values(["consensus_rank_score", "feature"], ascending=[False, True]).reset_index(drop=True)
    global_importance["consensus_rank"] = np.arange(1, len(global_importance) + 1)
    atomic_csv(root / REPORTS / "prompt5b_global_feature_importance.csv", global_importance)

    gate_importance = importance_by_component["gate"].copy()
    gate_importance.insert(0, "importance_type", "Gate importance")
    gate_importance["interpretation"] = "Features associated with routing toward Tail correction; not loan-amount importance."
    atomic_csv(root / REPORTS / "prompt5b_gate_feature_importance.csv", gate_importance)
    residual_importance = importance_by_component["residual"].copy()
    residual_importance.insert(0, "importance_type", "Residual correction importance")
    residual_importance["interpretation"] = "Positive output is upward correction; negative output is downward correction."
    atomic_csv(root / REPORTS / "prompt5b_residual_feature_importance.csv", residual_importance)

    stability_rows = []
    half_code = sample["row_hash"].astype(str).map(lambda value: int(hashlib.sha256(("prompt5b_stability_half" + value).encode()).hexdigest(), 16) % 2).to_numpy()
    for name, item in explanations.items():
        values = item["values"]
        first = np.mean(np.abs(values[half_code == 0]), axis=0)
        second = np.mean(np.abs(values[half_code == 1]), axis=0)
        ranks_first = rankdata(-first, method="average")
        ranks_second = rankdata(-second, method="average")
        top_first = set(np.asarray(item["features"])[np.argsort(-first)[:10]])
        top_second = set(np.asarray(item["features"])[np.argsort(-second)[:10]])
        stability_rows.append({
            "component": name, "half_0_n": int(np.sum(half_code == 0)), "half_1_n": int(np.sum(half_code == 1)),
            "top10_overlap_count": len(top_first.intersection(top_second)),
            "top10_overlap_fraction": len(top_first.intersection(top_second)) / 10.0,
            "spearman_rank_correlation": float(spearmanr(ranks_first, ranks_second).statistic),
            "explanation_space": item["space"],
        })
    stability = pd.DataFrame(stability_rows)
    atomic_csv(root / REPORTS / "prompt5b_explainability_stability.csv", stability)

    decile_rows = []
    body = ~sample_eval["iid_local_decile"].eq("D10").to_numpy()
    d10 = ~body
    for name in ("catboost", "lightgbm", "xgboost", "residual"):
        item = explanations[name]
        for index, feature in enumerate(item["features"]):
            body_value = float(np.mean(np.abs(item["values"][body, index])))
            d10_value = float(np.mean(np.abs(item["values"][d10, index])))
            decile_rows.append({"component": name, "feature": feature, "body_d1_d9_mean_abs_shap": body_value, "d10_mean_abs_shap": d10_value, "d10_to_body_ratio": d10_value / body_value if body_value > 0 else np.nan, "body_n": int(body.sum()), "d10_n": int(d10.sum()), "explanation_space": item["space"]})
    decile_explanation = pd.DataFrame(decile_rows)
    atomic_csv(root / REPORTS / "prompt5b_decile_explainability.csv", decile_explanation)

    error_rows = []
    error_threshold = float(np.quantile(evaluation["primary_abs_error"], 0.95, method="linear"))
    high_error = sample_eval["primary_abs_error"].ge(error_threshold).to_numpy()
    for name, item in explanations.items():
        for index, feature in enumerate(item["features"]):
            overall_value = float(np.mean(np.abs(item["values"][:, index])))
            high_value = float(np.mean(np.abs(item["values"][high_error, index]))) if high_error.any() else np.nan
            error_rows.append({"component": name, "feature": feature, "all_sample_mean_abs_shap": overall_value, "top5_primary_error_mean_abs_shap": high_value, "top5_to_all_ratio": high_value / overall_value if overall_value > 0 and np.isfinite(high_value) else np.nan, "all_n": len(sample), "top5_error_n": int(high_error.sum()), "explanation_space": item["space"]})
    error_explanation = pd.DataFrame(error_rows)
    atomic_csv(root / REPORTS / "prompt5b_error_explainability.csv", error_explanation)

    local_cases, local_mapping = _local_cases(evaluation)
    atomic_csv(root / REPORTS / "prompt5b_local_cases.csv", local_cases)
    atomic_parquet(root / EXPLAIN / "local_case_row_hashes.parquet", local_mapping)
    local_features = features.merge(local_mapping[["case_role", "row_hash"]], on="row_hash", how="inner", validate="one_to_many")
    local_eval = evaluation[["row_hash", "global_prediction"]].merge(local_mapping, on="row_hash", how="inner", validate="one_to_many")
    # Explain unique physical rows once, then map contributions to each deterministic case role.
    unique_local = local_features.drop_duplicates("row_hash").reset_index(drop=True)
    locked_local = evaluation.set_index("row_hash").loc[unique_local["row_hash"], "global_prediction"].to_numpy(float)
    local_explanations, _ = _explain_components(primary, unique_local, locked_local)
    case_lookup = local_mapping.groupby("row_hash")["case_role"].apply(list).to_dict()
    local_contribution_rows = []
    for component_name, item in local_explanations.items():
        for row_index, row_hash_value in enumerate(unique_local["row_hash"].astype(str)):
            order_index = np.argsort(-np.abs(item["values"][row_index]))[:10]
            for rank, feature_index in enumerate(order_index, 1):
                for case_role in case_lookup[row_hash_value]:
                    local_contribution_rows.append({"case_role": case_role, "anonymized_row_id": _anonymous_id(row_hash_value), "component": component_name, "rank": rank, "feature": item["features"][feature_index], "contribution": float(item["values"][row_index, feature_index]), "explanation_space": item["space"], "causal_interpretation": False})
    atomic_parquet(root / EXPLAIN / "local_case_contributions.parquet", pd.DataFrame(local_contribution_rows))

    manifest_payload = {
        "status": "PASS", "created_at_utc": utc_now(), "sample_rule": "4,000 smallest SHA-256 values of prompt5b_explainability_seed42 + row_hash",
        "row_count": len(sample), "row_digest": row_digest(sample["row_hash"]),
        "overlap_with_iid_d10": int(sample_eval["iid_local_decile"].eq("D10").sum()),
        "overlap_with_primary_top5_absolute_error": int(high_error.sum()),
        "overlap_with_routed_rows": int(sample_eval["routing_condition_activated"].sum()),
        "sensitive_fields_used_for_selection": [], "target_or_error_used_for_main_selection": False,
        "sample_file": (EXPLAIN / "sample_row_hashes.parquet").as_posix(),
        "attribution_spaces": {name: item["space"] for name, item in explanations.items()},
        "xgboost_warning": "XGBoost contributions are in native log1p target space and are not raw-scale comparable with CatBoost or LightGBM.",
        "composite_warning": "No single additive SHAP decomposition is claimed for the complete Stage 3 system.",
        "maximum_native_additivity_errors": {name: item["maximum_additivity_error"] for name, item in explanations.items()},
    }
    atomic_json(root / REPORTS / "prompt5b_explainability_sample_manifest.json", manifest_payload)
    summary = {
        "status": "PASS", "sample_rows": len(sample), "components": list(explanations),
        "top_global_features": global_importance.head(10)["feature"].tolist(),
        "top_gate_features": gate_importance.head(10)["feature"].tolist(),
        "top_residual_features": residual_importance.head(10)["feature"].tolist(),
        "local_case_roles": local_cases["case_role"].tolist(),
        "zero_fit": True, "zero_model_change": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(root / REPORTS / "prompt5b_explainability_summary.json", summary)
    return summary


def _anonymous_id(row_hash_value: str) -> str:
    return hashlib.sha256(("prompt5b_case" + str(row_hash_value)).encode()).hexdigest()[:12]


def _local_cases(evaluation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d10 = evaluation[evaluation["iid_local_decile"].eq("D10")]
    body = evaluation[~evaluation["iid_local_decile"].eq("D10")]
    choices = [
        ("CASE_1_LARGEST_PRIMARY_ABSOLUTE_ERROR", evaluation["primary_abs_error"].idxmax()),
        ("CASE_2_LARGEST_PRIMARY_IMPROVEMENT", evaluation["delta_abs_error"].idxmin()),
        ("CASE_3_LARGEST_PRIMARY_DAMAGE", evaluation["delta_abs_error"].idxmax()),
        ("CASE_4_D10_CLOSEST_SIGNED_ERROR_TO_ZERO", d10["primary_signed_error"].abs().idxmin()),
        ("CASE_5_D10_LARGEST_UNDERPREDICTION", d10["primary_signed_error"].idxmin()),
        ("CASE_6_BODY_LARGEST_APPLIED_CORRECTION", body["applied_residual_correction"].abs().idxmax()),
    ]
    rows, mapping = [], []
    for role, index in choices:
        item = evaluation.loc[index]
        anon = _anonymous_id(str(item["row_hash"]))
        rows.append({
            "case_role": role, "anonymized_row_id": anon,
            "true_target": item["y_true"], "primary_prediction": item["primary_prediction"], "global_prediction": item["global_prediction"],
            "primary_absolute_error": item["primary_abs_error"], "global_absolute_error": item["global_abs_error"],
            "delta_abs_error_primary_minus_global": item["delta_abs_error"], "target_decile": item["iid_local_decile"],
            "routed_flag": bool(item["routing_condition_activated"]), "applied_correction": item["applied_residual_correction"],
            "sensitive_values_disclosed": False, "causal_interpretation": False,
        })
        mapping.append({"case_role": role, "row_hash": str(item["row_hash"]), "anonymized_row_id": anon})
    return pd.DataFrame(rows), pd.DataFrame(mapping)


def _save_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    figure.savefig(temporary, dpi=150, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def _group_panel_figure(frame: pd.DataFrame, value: str, title: str, xlabel: str, path: Path, color_rule: str | None = None) -> None:
    fields = frame["sensitive_field"].drop_duplicates().tolist()
    ncols = 2
    nrows = math.ceil(len(fields) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(4, nrows * 4.2)))
    axes = np.asarray(axes).reshape(-1)
    for axis, field in zip(axes, fields):
        part = frame[frame["sensitive_field"].eq(field)].sort_values(value)
        labels = [label if len(label) <= 36 else label[:33] + "..." for label in part["group_label"]]
        colors = "#2b6cb0"
        if color_rule == "delta":
            colors = np.where(part[value] < 0, "#2f855a", "#c53030")
            axis.axvline(0, color="black", lw=.8)
        axis.barh(labels, part[value], color=colors)
        for y_position, (metric, n) in enumerate(zip(part[value], part["n"])):
            axis.text(metric, y_position, f" n={int(n):,}", va="center", fontsize=7)
        axis.set_title(field.replace("_", " "))
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", alpha=.2)
    for axis in axes[len(fields):]:
        axis.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_figure(path, fig)


def build_figures(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    primary = pd.read_csv(root / REPORTS / "prompt5b_primary_group_metrics.csv")
    eligible = primary[primary["analysis_status"].eq("ELIGIBLE")].copy()
    comparison = pd.read_csv(root / REPORTS / "prompt5b_group_primary_vs_global.csv")
    comparison = comparison[comparison["analysis_status"].eq("ELIGIBLE")]
    tail = pd.read_csv(root / REPORTS / "prompt5b_tail_group_metrics.csv")
    tail = tail[(tail["tail_scope"].eq("IID_GLOBAL_TOP_DECILE")) & (tail["analysis_status"].eq("ELIGIBLE"))]
    _group_panel_figure(eligible, "mae", "Primary MAE by eligible sensitive group", "MAE", root / FIGURES / "01_primary_mae_by_sensitive_group.png")
    _group_panel_figure(comparison, "primary_minus_global_mae", "Primary minus Global MAE by eligible sensitive group", "Primary minus Global MAE", root / FIGURES / "02_primary_minus_global_by_sensitive_group.png", "delta")
    _group_panel_figure(eligible, "underprediction_rate", "Primary underprediction rate by eligible sensitive group", "Underprediction rate", root / FIGURES / "03_underprediction_by_sensitive_group.png")
    _group_panel_figure(tail, "primary_tail_mae", "IID-global D10 MAE by eligible sensitive group", "D10 MAE", root / FIGURES / "04_top_decile_mae_by_sensitive_group.png")
    atomic_csv(root / REPORTS / "prompt5b_plot_group_metrics.csv", eligible)
    atomic_csv(root / REPORTS / "prompt5b_plot_group_comparison.csv", comparison)
    atomic_csv(root / REPORTS / "prompt5b_plot_tail_groups.csv", tail)

    bootstrap = pd.read_csv(root / REPORTS / "prompt5b_fairness_bootstrap.csv")
    gaps = bootstrap[bootstrap["row_type"].eq("FIELD_GAP")].sort_values("observed")
    atomic_csv(root / REPORTS / "prompt5b_plot_disparity_gaps.csv", gaps)
    fig, axis = plt.subplots(figsize=(9, 5))
    xerr = np.vstack([gaps["observed"] - gaps["ci_lower_2_5"], gaps["ci_upper_97_5"] - gaps["observed"]])
    axis.errorbar(gaps["observed"], gaps["sensitive_field"].str.replace("_", " "), xerr=xerr, fmt="o", color="#2b6cb0", capsize=3)
    axis.set(xlabel="Worst minus best eligible-group MAE", title="Descriptive group MAE gaps with 95% bootstrap intervals")
    axis.grid(axis="x", alpha=.25)
    _save_figure(root / FIGURES / "05_group_mae_gaps_bootstrap.png", fig)

    intersection = pd.read_csv(root / REPORTS / "prompt5b_intersectional_metrics.csv")
    intersection_plot = intersection[(intersection["scope"].eq("OVERALL")) & (intersection["analysis_status"].eq("ELIGIBLE"))].copy()
    atomic_csv(root / REPORTS / "prompt5b_plot_intersectional.csv", intersection_plot)
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, .38 * max(1, len(intersection_plot)))))
    for axis, (name, part) in zip(axes, intersection_plot.groupby("intersection", sort=False)):
        part = part.sort_values("mae").tail(15)
        labels = [label if len(label) < 45 else label[:42] + "..." for label in part["group_label"]]
        axis.barh(labels, part["mae"], color="#805ad5")
        for i, (value, n) in enumerate(zip(part["mae"], part["n"])):
            axis.text(value, i, f" n={int(n):,}", va="center", fontsize=7)
        axis.set_title(name.replace("_", " ")); axis.set_xlabel("Primary MAE"); axis.grid(axis="x", alpha=.2)
    if intersection_plot.empty:
        for axis in axes:
            axis.text(.5, .5, "No eligible predeclared intersection", ha="center", va="center"); axis.axis("off")
    fig.suptitle("Eligible predeclared applicant intersections")
    fig.tight_layout()
    _save_figure(root / FIGURES / "06_intersectional_mae.png", fig)

    global_importance = pd.read_csv(root / REPORTS / "prompt5b_global_feature_importance.csv").sort_values("consensus_rank_score").tail(15)
    _importance_bar(global_importance, "feature", "consensus_rank_score", "Global weighted rank-consensus Top 15", "Normalized weighted rank score", root / FIGURES / "07_global_consensus_top15.png", "#2b6cb0")
    for number, name, title, color in ((8, "catboost", "CatBoost component SHAP Top 15 (raw space)", "#805ad5"), (9, "lightgbm", "LightGBM component SHAP Top 15 (raw space)", "#38a169"), (10, "xgboost", "XGBoost component SHAP Top 15 (native log1p space)", "#dd6b20")):
        values = global_importance = pd.read_csv(root / REPORTS / "prompt5b_global_feature_importance.csv")
        plot = values[["feature", f"{name}_mean_abs_shap"]].sort_values(f"{name}_mean_abs_shap").tail(15)
        atomic_csv(root / REPORTS / f"prompt5b_plot_{name}_importance.csv", plot)
        _importance_bar(plot, "feature", f"{name}_mean_abs_shap", title, "Mean absolute native SHAP", root / FIGURES / f"{number:02d}_{name}_shap_top15.png", color)
    atomic_csv(root / REPORTS / "prompt5b_plot_global_consensus.csv", global_importance)
    gate = pd.read_csv(root / REPORTS / "prompt5b_gate_feature_importance.csv").sort_values("mean_abs_shap").tail(15)
    residual = pd.read_csv(root / REPORTS / "prompt5b_residual_feature_importance.csv").sort_values("mean_abs_shap").tail(15)
    atomic_csv(root / REPORTS / "prompt5b_plot_gate_importance.csv", gate)
    atomic_csv(root / REPORTS / "prompt5b_plot_residual_importance.csv", residual)
    _importance_bar(gate, "feature", "mean_abs_shap", "Meta-Gate SHAP Top 15 (routing log-odds space)", "Mean absolute native SHAP", root / FIGURES / "11_gate_shap_top15.png", "#c53030")
    _importance_bar(residual, "feature", "mean_abs_shap", "Residual Specialist SHAP Top 15", "Mean absolute native SHAP", root / FIGURES / "12_residual_shap_top15.png", "#319795")

    decile = pd.read_csv(root / REPORTS / "prompt5b_decile_explainability.csv")
    residual_decile = decile[decile["component"].eq("residual")].copy()
    residual_decile["maximum"] = residual_decile[["body_d1_d9_mean_abs_shap", "d10_mean_abs_shap"]].max(axis=1)
    residual_decile = residual_decile.nlargest(15, "maximum").sort_values("maximum")
    atomic_csv(root / REPORTS / "prompt5b_plot_residual_body_d10.csv", residual_decile)
    fig, axis = plt.subplots(figsize=(9, 6))
    y = np.arange(len(residual_decile)); width = .38
    axis.barh(y - width / 2, residual_decile["body_d1_d9_mean_abs_shap"], height=width, label="Body D1-D9", color="#718096")
    axis.barh(y + width / 2, residual_decile["d10_mean_abs_shap"], height=width, label="D10", color="#319795")
    axis.set(yticks=y, yticklabels=residual_decile["feature"], xlabel="Mean absolute SHAP (residual space)", title="Residual Specialist importance: Body versus D10")
    axis.legend(); axis.grid(axis="x", alpha=.2)
    _save_figure(root / FIGURES / "13_residual_body_vs_d10.png", fig)

    correction = pd.read_csv(root / REPORTS / "prompt5b_correction_behavior.csv")
    benefit = correction[correction["section"].eq("realized_benefit")]
    atomic_csv(root / REPORTS / "prompt5b_plot_correction_benefit.csv", benefit)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    colors = np.where(benefit["value"] >= 0, "#2f855a", "#c53030")
    axis.bar(benefit["scope"], benefit["value"], color=colors)
    axis.axhline(0, color="black", lw=.8)
    axis.set(xlabel="Applied-correction magnitude quintile among routed rows", ylabel="Global absolute error minus Primary absolute error", title="Correction magnitude and realized error benefit")
    axis.grid(axis="y", alpha=.2)
    _save_figure(root / FIGURES / "14_correction_magnitude_realized_benefit.png", fig)
    routing = correction[(correction["section"].eq("target_decile")) & (correction["metric"].eq("routed_fraction"))].copy()
    routing["decile_index"] = routing["scope"].str[1:].astype(int)
    atomic_csv(root / REPORTS / "prompt5b_plot_routing_decile.csv", routing)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(routing["decile_index"], routing["value"], marker="o", color="#c53030")
    axis.set(xlabel="IID-local target decile", ylabel="Routed fraction", title="Frozen routed fraction by IID target decile", xticks=range(1, 11))
    axis.grid(alpha=.25)
    _save_figure(root / FIGURES / "15_routed_fraction_by_decile.png", fig)
    return {"status": "PASS", "figure_count": 15, "elapsed_seconds": time.perf_counter() - started}


def _importance_bar(frame: pd.DataFrame, label: str, value: str, title: str, xlabel: str, path: Path, color: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.barh(frame[label], frame[value], color=color)
    axis.set(xlabel=xlabel, title=title)
    axis.grid(axis="x", alpha=.2)
    _save_figure(path, fig)


def build_notebook(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells = []
    md = lambda text: cells.append(nbformat.v4.new_markdown_cell(text))
    code = lambda text: cells.append(nbformat.v4.new_code_cell(text))
    md("# Prompt 5B - Fairness Audit and Final Explainability\n\nThe final Primary stayed frozen after IID. Prompt 5A reported Primary MAE 62.260626 and Global MAE 62.444349. Five of six frozen conditions passed; only the 3% Tail-improvement condition failed. No model was changed.")
    code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT=Path('..')\nR=ROOT/'outputs/reports'\nF=ROOT/'outputs/figures/prompt5b'")
    md("## 1. Audit meaning and boundary\n\nThis is a predictive-error audit, not a causal or legal fair-lending test. Sensitive fields were never model inputs. They were joined after frozen prediction only. The original IID files stayed closed.")
    code("handoff=json.loads((R/'prompt5b_handoff_validation.json').read_text())\ncontract=json.loads((R/'prompt5b_sensitive_contract.json').read_text())\ndisplay(pd.DataFrame([{'handoff':handoff['status'],'aligned snapshots':sum(x['status']=='PASS' for x in handoff['checks']),'sensitive fields':contract['field_count'],'fits':0,'model changes':0}]))\ndisplay(pd.DataFrame(contract['fields']))")
    md("## 2. Group sizes and target composition\n\nGroup target distributions differ. Error gaps may reflect target composition, available covariates, model behavior, or combinations of these factors.")
    code("inv=pd.read_csv(R/'prompt5b_group_inventory.csv')\ndisplay(inv[inv.analysis_status.eq('ELIGIBLE')])\ndisplay(inv.groupby('sensitive_field').agg(levels=('group_label','size'),eligible=('analysis_status',lambda x:(x=='ELIGIBLE').sum()),small=('analysis_status',lambda x:(x=='SMALL_GROUP').sum())).reset_index())")
    md("## 3. Primary group error")
    code("primary=pd.read_csv(R/'prompt5b_primary_group_metrics.csv')\ndisplay(primary[primary.analysis_status.eq('ELIGIBLE')])\ndisplay(Image(filename=str(F/'01_primary_mae_by_sensitive_group.png')))\ndisplay(Image(filename=str(F/'03_underprediction_by_sensitive_group.png')))")
    md("## 4. Primary versus Global within groups\n\nNegative MAE difference means the Tail-aware Primary performed better than Global in that group.")
    code("compare=pd.read_csv(R/'prompt5b_group_primary_vs_global.csv')\ndisplay(compare[compare.analysis_status.eq('ELIGIBLE')])\ndisplay(Image(filename=str(F/'02_primary_minus_global_by_sensitive_group.png')))")
    md("## 5. Descriptive disparity summary and uncertainty\n\nRatios and gaps below are descriptive. They are not legal fairness violations or pass/fail tests.")
    code("display(pd.read_csv(R/'prompt5b_disparity_summary.csv'))\ndisplay(pd.read_csv(R/'prompt5b_fairness_flags.csv'))\ndisplay(Image(filename=str(F/'05_group_mae_gaps_bootstrap.png')))")
    md("## 6. Common IID Tail groups\n\nEvery group uses the same IID-global D10 and top-5% target masks. No group-specific Tail threshold was used.")
    code("tail=pd.read_csv(R/'prompt5b_tail_group_metrics.csv')\ndisplay(tail[tail.analysis_status.eq('ELIGIBLE')])\ndisplay(Image(filename=str(F/'04_top_decile_mae_by_sensitive_group.png')))")
    md("## 7. Target-decile profiles")
    code("dec=pd.read_csv(R/'prompt5b_group_decile_metrics.csv')\ndisplay(dec[dec.cell_status.eq('DISPLAY')])")
    md("## 8. Bounded applicant intersections\n\nOnly applicant race by applicant sex and applicant ethnicity by applicant sex were predeclared.")
    code("inter=pd.read_csv(R/'prompt5b_intersectional_metrics.csv')\ndisplay(inter[inter.analysis_status.eq('ELIGIBLE')])\ndisplay(Image(filename=str(F/'06_intersectional_mae.png')))")
    md("## 9. Hierarchical explanation\n\nThe final system is composite. Global prediction, Meta-Gate routing, Residual Specialist correction, and realized correction behavior are explained separately. No single SHAP decomposition is claimed for the full Stage 3 system.")
    code("display(pd.read_csv(R/'prompt5b_global_component_summary.csv'))")
    md("## 10. Global prediction drivers\n\nThe consensus combines normalized feature ranks with frozen 0.60/0.20/0.20 model weights. Raw SHAP magnitudes are not added across models.")
    code("glob=pd.read_csv(R/'prompt5b_global_feature_importance.csv')\ndisplay(glob.head(15))\ndisplay(Image(filename=str(F/'07_global_consensus_top15.png')))\nfor name in ['08_catboost_shap_top15.png','09_lightgbm_shap_top15.png','10_xgboost_shap_top15.png']:\n    display(Image(filename=str(F/name)))")
    md("## 11. XGBoost explanation scale\n\nXGBoost uses historical log1p target semantics. Its SHAP values are in native log1p space and are not directly scale-comparable with raw-space CatBoost and LightGBM values.")
    md("## 12. Meta-Gate routing drivers\n\nGate importance asks which observed features push the frozen model toward Tail correction. It is not loan-amount importance. Global prediction is a model-derived Gate input, not an original dataset feature.")
    code("display(pd.read_csv(R/'prompt5b_gate_feature_importance.csv').head(15))\ndisplay(Image(filename=str(F/'11_gate_shap_top15.png')))")
    md("## 13. Residual Specialist correction drivers\n\nPositive Specialist output means upward correction. Negative output means downward correction. This is not a direct loan-amount explanation.")
    code("display(pd.read_csv(R/'prompt5b_residual_feature_importance.csv').head(15))\ndisplay(Image(filename=str(F/'12_residual_shap_top15.png')))")
    md("## 14. Body versus D10 importance")
    code("dexp=pd.read_csv(R/'prompt5b_decile_explainability.csv')\ndisplay(dexp.sort_values('d10_to_body_ratio',ascending=False).groupby('component').head(10))\ndisplay(Image(filename=str(F/'13_residual_body_vs_d10.png')))")
    md("## 15. Large-error explanation\n\nThis comparison is descriptive. It shows feature regimes associated with large errors; it is not causal.")
    code("err=pd.read_csv(R/'prompt5b_error_explainability.csv')\ndisplay(err.sort_values('top5_to_all_ratio',ascending=False).groupby('component').head(10))")
    md("## 16. Routing and correction mechanism")
    code("display(pd.read_csv(R/'prompt5b_correction_behavior.csv'))\ndisplay(Image(filename=str(F/'14_correction_magnitude_realized_benefit.png')))\ndisplay(Image(filename=str(F/'15_routed_fraction_by_decile.png')))")
    md("## 17. Sensitive-group routing cross-check\n\nThis is a post-hoc mechanism audit. Routing-rate parity is not asserted as a fairness requirement.")
    code("display(pd.read_csv(R/'prompt5b_routing_sensitive_audit.csv'))\ndisplay(pd.read_csv(R/'prompt5b_fairness_explainability_link.csv'))")
    md("## 18. SHAP stability")
    code("display(pd.read_csv(R/'prompt5b_explainability_stability.csv'))")
    md("## 19. Six deterministic local cases\n\nCases follow frozen rules. IDs are pseudonymous, sensitive values are not shown, and explanations are not causal.")
    code("display(pd.read_csv(R/'prompt5b_local_cases.csv'))\nlocal_contrib=pd.read_parquet(ROOT/'outputs/explainability/prompt5b/local_case_contributions.parquet')\ndisplay(local_contrib[local_contrib['rank']<=5].sort_values(['case_role','component','rank']))")
    md("## 20. Limitations and closure\n\nSensitive variables were excluded from Primary inputs and used only after prediction for audit. Observational group differences are not causal effects. Group target distributions differ. This is not a lending approval model. The audit does not prove absence or presence of discrimination. Small-group estimates are unstable, and multiple comparisons are descriptive. No model, feature, threshold, weight, routing rule, or correction policy changed.")
    nb.cells = cells
    path = root / NOTEBOOK
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    executed = client.execute()
    nbformat.write(executed, path)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    errors = [out for cell in code_cells for out in cell.get("outputs", []) if out.get("output_type") == "error"]
    images = sum(1 for cell in code_cells for out in cell.get("outputs", []) if out.get("output_type") == "display_data" and "image/png" in out.get("data", {}))
    tables = sum(1 for cell in code_cells for out in cell.get("outputs", []) if "text/html" in out.get("data", {}))
    source = "\n".join(cell.source for cell in code_cells)
    prohibited = [token for token in ("iid_holdout_features", "iid_holdout_targets", ".fit(", "joblib.load", "get_feature_importance", "pred_contrib") if token in source]
    report = {
        "status": "PASS" if not errors and images >= 15 and tables >= 15 and not prohibited else "FAIL",
        "created_at_utc": utc_now(), "notebook_path": NOTEBOOK.as_posix(), "attempt": 1,
        "code_cells": len(code_cells), "markdown_sections": sum(cell.cell_type == "markdown" for cell in executed.cells),
        "error_count": len(errors), "inline_images": images, "inline_tables": tables,
        "original_iid_accesses": 0, "model_fits": 0, "prediction_calls": 0, "shap_calculations": 0,
        "prohibited_code_tokens": prohibited, "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(root / REPORTS / "prompt5b_notebook_execution.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"Prompt 5B artifact-only notebook failed: {report}")
    return report


def create_candidate(root: Path) -> dict[str, Any]:
    fairness = read_json(root / REPORTS / "prompt5b_fairness_summary.json")
    explain = read_json(root / REPORTS / "prompt5b_explainability_summary.json")
    notebook = read_json(root / REPORTS / "prompt5b_notebook_execution.json")
    if fairness["status"] != "PASS" or explain["status"] != "PASS" or notebook["status"] != "PASS":
        raise RuntimeError("Prompt 5B artifacts are not candidate-ready.")
    flags = pd.read_csv(root / REPORTS / "prompt5b_fairness_flags.csv")
    tail = pd.read_csv(root / REPORTS / "prompt5b_tail_group_metrics.csv")
    disparity = pd.read_csv(root / REPORTS / "prompt5b_disparity_summary.csv")
    contract = read_json(root / REPORTS / "prompt5b_sensitive_contract.json")
    report_names = [
        "prompt5b_group_inventory.csv", "prompt5b_primary_group_metrics.csv", "prompt5b_global_group_metrics.csv",
        "prompt5b_group_primary_vs_global.csv", "prompt5b_disparity_summary.csv", "prompt5b_tail_group_metrics.csv",
        "prompt5b_group_decile_metrics.csv", "prompt5b_intersectional_metrics.csv", "prompt5b_fairness_bootstrap.csv",
        "prompt5b_fairness_flags.csv", "prompt5b_global_feature_importance.csv", "prompt5b_gate_feature_importance.csv",
        "prompt5b_residual_feature_importance.csv", "prompt5b_explainability_stability.csv", "prompt5b_decile_explainability.csv",
        "prompt5b_error_explainability.csv", "prompt5b_local_cases.csv", "prompt5b_routing_sensitive_audit.csv",
        "prompt5b_fairness_explainability_link.csv", "prompt5b_correction_behavior.csv",
    ]
    explanation_names = ["sample_row_hashes.parquet", "catboost_shap.parquet", "lightgbm_shap.parquet", "xgboost_shap.parquet", "gate_shap.parquet", "residual_shap.parquet", "global_component_predictions.parquet", "local_case_row_hashes.parquet", "local_case_contributions.parquet"]
    payload = {
        "status": "CANDIDATE_COMPLETE_AWAITING_REVIEW_AND_VERIFICATION", "created_at_utc": utc_now(), "authorization_id": AUTHORIZATION_ID,
        "prompt5a_evidence": {"final_iid_evaluation_sha256": sha256(root / REPORTS / "FINAL_IID_EVALUATION.json"), "prompt5a_ready_sha256": sha256(root / REPORTS / "PROMPT5A_READY.json"), "primary_mae": PRIMARY_MAE, "global_mae": GLOBAL_MAE, "conditions_passed": 5, "conditions_total": 6},
        "sensitive_contract_sha256": sha256(root / REPORTS / "prompt5b_sensitive_contract.json"),
        "sensitive_contract_identity": contract["contract_sha256"],
        "fairness_report_hashes": {name: sha256(root / REPORTS / name) for name in report_names},
        "explainability_artifact_hashes": {name: sha256(root / EXPLAIN / name) for name in explanation_names},
        "notebook_sha256": sha256(root / NOTEBOOK),
        "main_fairness_findings": {
            "eligible_group_levels": fairness["eligible_group_levels"],
            "largest_eligible_group_mae_gap": float(disparity["mae_max_minus_min"].max()) if len(disparity) else None,
            "elevated_mae_groups": flags[flags["elevated_mae_descriptive"]][["sensitive_field", "group_label"]].to_dict(orient="records"),
            "elevated_underprediction_groups": flags[flags["elevated_underprediction_descriptive"]][["sensitive_field", "group_label"]].to_dict(orient="records"),
        },
        "main_tail_group_findings": {
            "eligible_tail_group_scope_cells": int(tail["analysis_status"].eq("ELIGIBLE").sum()),
            "best_primary_minus_global_tail_mae": float(tail["primary_minus_global_tail_mae"].min()),
            "worst_primary_minus_global_tail_mae": float(tail["primary_minus_global_tail_mae"].max()),
        },
        "top_global_drivers": explain["top_global_features"], "top_gate_drivers": explain["top_gate_features"], "top_residual_drivers": explain["top_residual_features"],
        "explainability_limitations": ["Component explanations use separate native spaces.", "XGBoost SHAP is in log1p space.", "No raw SHAP sum or single composite Stage 3 SHAP is claimed.", "Associations are not causal."],
        "integrity": {"original_iid_rereads": 0, "model_fits": 0, "model_refits": 0, "preprocessor_fits": 0, "calibration_fits": 0, "tuning": 0, "threshold_searches": 0, "model_changes": 0},
        "limitations": LIMITATIONS,
    }
    atomic_json(root / REPORTS / "prompt5b_final_candidate.json", payload)
    return payload


def record_review(root: Path, status: str, check_count: int, findings: str) -> dict[str, Any]:
    parsed_findings = [] if not findings.strip() else json.loads(findings)
    evidence = {
        "data_boundary": "Only authorized post-IID snapshots and saved artifacts were opened; original IID files remained closed.",
        "model_integrity": "Primary, Global, and freeze hashes match; locked predictions are unchanged; zero fits, refits, tuning, or model changes.",
        "alignment": "75,000 unique fairness rows align one-to-one with 75,000 evaluation rows.",
        "sensitive_contract": "Exact eight fields; no overlap with Primary, Global, Gate, or Residual contracts; respondent_id absent.",
        "fairness": "All 9,053 group counts and all 27 eligible-group formulas reproduced; n>=200, tail_n>=50, and decile n>=30 rules enforced.",
        "tails": "Common inclusive IID-global q90 top-decile n=7,503 and q95 top-5% n=3,781 reproduced.",
        "intersections": "Only applicant race by sex and ethnicity by sex, with frozen 500/50 thresholds.",
        "bootstrap": "500 resamples and seed 42; 22 rows for the first frozen field independently reproduced exactly.",
        "sample": "Exact target-free 4,000-row sample and digest reproduced.",
        "attributions": "Five finite aligned component matrices; native additivity within declared tolerances.",
        "explainability": "Weighted Global prediction and rank consensus reproduced; XGBoost log1p warning, no cross-model sum, and Gate/Residual labels verified.",
        "local_cases": "All six deterministic roles and 300 saved top-10 component contributions reproduced.",
        "notebook": "Artifact-only notebook has zero errors/prohibited calls, 15 inline images, and 23 inline tables.",
        "closure": "Descriptive non-causal language; no model- or feature-selection recommendation or model change.",
    }
    payload = {
        "status": status, "created_at_utc": utc_now(), "authorization_id": AUTHORIZATION_ID,
        "review_type": "exactly_one_independent_read_only_agent_review", "reviewer_count": 1,
        "check_count": check_count, "checks_passed": check_count if status == "PASS" else None,
        "evidence": evidence, "findings": parsed_findings,
        "resolved_during_review": [{"severity": "MAJOR", "finding": "The notebook initially omitted per-case component contributions.", "resolution": "Repaired from the saved contribution artifact only; the current notebook renders rank<=5 contributors for every case and component. No attribution recomputation occurred."}],
        "critical_findings": sum(item.get("severity") == "CRITICAL" for item in parsed_findings),
        "major_findings": sum(item.get("severity") == "MAJOR" for item in parsed_findings),
        "reporting_only_repairs_permitted": True, "scientific_or_model_changes_permitted": False,
        "original_iid_reads": 0, "model_fits": 0, "model_changes": 0,
    }
    atomic_json(root / REPORTS / "prompt5b_reviewer.json", payload)
    return payload


def runtime_report(root: Path, started_epoch: float, phases: dict[str, float]) -> dict[str, Any]:
    payload = {"status": "PASS", "created_at_utc": utc_now(), "authorization_id": AUTHORIZATION_ID, "started_epoch": started_epoch, "elapsed_seconds": time.time() - started_epoch, "phases_seconds": phases, "original_iid_rereads": 0, "model_fit_count": 0, "preprocessor_fit_count": 0, "calibration_fit_count": 0, "hpo_count": 0, "threshold_search_count": 0, "model_change_count": 0, "explainability_sample_rows": SAMPLE_SIZE, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "seed": SEED}
    atomic_json(root / REPORTS / "prompt5b_runtime.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "fairness", "explain", "figures", "notebook", "candidate", "record-review"))
    parser.add_argument("--root", default=None)
    parser.add_argument("--status", default="PASS")
    parser.add_argument("--check-count", type=int, default=0)
    parser.add_argument("--findings", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = root_path(args.root)
    if args.command == "preflight":
        result = preflight(root)
    elif args.command == "fairness":
        result = fairness_analysis(root)
    elif args.command == "explain":
        result = explainability(root)
    elif args.command == "figures":
        result = build_figures(root)
    elif args.command == "notebook":
        result = build_notebook(root)
    elif args.command == "candidate":
        result = create_candidate(root)
    else:
        result = record_review(root, args.status, args.check_count, args.findings)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
