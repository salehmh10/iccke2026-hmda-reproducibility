"""Append-only CSV experiment tracking helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


LEDGER_COLUMNS = [
    "experiment_id",
    "timestamp_utc",
    "git_commit",
    "dataset_variant",
    "sample_size",
    "years",
    "feature_set",
    "model",
    "hyperparameters",
    "seed",
    "class_weighting",
    "sampler",
    "train_runtime_seconds",
    "threshold",
    "status",
    "error",
    "accuracy",
    "balanced_accuracy",
    "f1_denial",
    "f1_macro",
    "mcc",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier_score",
    "recall_denial",
    "recall_approval",
    "precision_denial",
    "precision_approval",
    "specificity",
    "predicted_positive_rate",
    "predicted_negative_rate",
    "unique_prediction_classes",
    "degenerate",
    "tn",
    "fp",
    "fn",
    "tp",
]


def git_commit_or_none(root: Path) -> str:
    """Return the repository commit or `not-a-git-repository`."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not-a-git-repository"


def build_record(
    *,
    root: Path,
    experiment_id: str,
    dataset_variant: str,
    sample_size: int,
    feature_set: str,
    model: str,
    hyperparameters: Mapping[str, Any],
    seed: int,
    class_weighting: str,
    sampler: str,
    train_runtime_seconds: float,
    threshold: float,
    status: str,
    metrics: Mapping[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Create one schema-complete experiment row."""

    record: dict[str, Any] = {column: "" for column in LEDGER_COLUMNS}
    record.update(
        {
            "experiment_id": experiment_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit_or_none(root),
            "dataset_variant": dataset_variant,
            "sample_size": int(sample_size),
            "years": "unavailable",
            "feature_set": feature_set,
            "model": model,
            "hyperparameters": json.dumps(dict(hyperparameters), sort_keys=True, default=str),
            "seed": int(seed),
            "class_weighting": class_weighting,
            "sampler": sampler,
            "train_runtime_seconds": round(float(train_runtime_seconds), 6),
            "threshold": float(threshold),
            "status": status,
            "error": error,
        }
    )
    if metrics:
        for key, value in metrics.items():
            if key in record:
                record[key] = value
    return record


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append without overwriting earlier runs and reject duplicate IDs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = {row["experiment_id"] for row in csv.DictReader(handle)}
        if str(record["experiment_id"]) in existing:
            raise ValueError(f"duplicate experiment_id: {record['experiment_id']}")
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({column: record.get(column, "") for column in LEDGER_COLUMNS})


def experiment_exists(path: Path, experiment_id: str) -> bool:
    """Check an experiment ID by parsed exact equality, never substring search."""

    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        return any(row.get("experiment_id") == experiment_id for row in csv.DictReader(handle))


def experiment_contract_fingerprint(
    root: Path,
    contract: Mapping[str, Any],
    relative_code_files: list[str],
) -> str:
    """Bind a run identity to data/split/feature metadata and executable code."""

    digest = hashlib.sha256()
    digest.update(json.dumps(dict(contract), sort_keys=True, default=str).encode("utf-8"))
    for relative in sorted(relative_code_files):
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def write_experiment_provenance(
    root: Path,
    experiment_id: str,
    fingerprint: str,
    contract: Mapping[str, Any],
    relative_code_files: list[str],
    artifact_dir: Path | None = None,
) -> Path:
    """Write a sidecar that allows exact resume-time contract validation."""

    destination = artifact_dir or (root / "artifacts" / "models")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{experiment_id}.provenance.json"
    payload = {
        "experiment_id": experiment_id,
        "contract_fingerprint": fingerprint,
        "contract": dict(contract),
        "code_files": sorted(relative_code_files),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def cumulative_training_seconds(path: Path) -> float:
    """Sum completed/failed measured training runtimes in the ledger."""

    if not path.exists():
        return 0.0
    total = 0.0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                total += float(row.get("train_runtime_seconds", 0.0) or 0.0)
            except ValueError:
                continue
    return total
