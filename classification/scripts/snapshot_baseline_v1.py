"""Create a non-destructive, hash-verified snapshot of the frozen v1 baseline."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DEST = ROOT / "reports" / "generations" / "baseline_v1"
ARTIFACT_DEST = ROOT / "artifacts" / "generations" / "baseline_v1"

REPORT_FILES = {
    "reports/FINAL_REPORT_FA.md": "BASELINE_REPORT_FA.md",
    "reports/FINAL_REPORT_FA.pdf": "BASELINE_REPORT_FA.pdf",
    "reports/FINAL_REPORT.md": "BASELINE_REPORT_EN.md",
    "reports/MODEL_COMPARISON.md": "MODEL_COMPARISON.md",
    "reports/FINAL_TEST_RESULTS.csv": "FINAL_TEST_RESULTS.csv",
    "reports/CALIBRATION_RESULTS.json": "CALIBRATION_RESULTS.json",
    "reports/BOOTSTRAP_INTERVALS.json": "BOOTSTRAP_INTERVALS.json",
    "reports/FAIRNESS_AUDIT.md": "FAIRNESS_AUDIT.md",
    "reports/REVIEW.md": "REVIEW.md",
    "reports/TRAINING_BUDGET.md": "TRAINING_BUDGET.md",
}

ARTIFACT_FILES = {
    "artifacts/models/best_predictive_model.joblib": "best_predictive_model.joblib",
    "artifacts/models/best_practical_model.joblib": "best_practical_model.joblib",
    "artifacts/models/finalist_selection.json": "finalist_selection.json",
    "artifacts/models/FINAL_ARTIFACT_MANIFEST.json": "FINAL_ARTIFACT_MANIFEST.json",
    "artifacts/models/FINAL_GENERATION.lock.json": "FINAL_GENERATION.lock.json",
    "artifacts/predictions/best_predictive_hybrid_test.npz": "best_predictive_hybrid_test.npz",
    "artifacts/predictions/best_single_catboost_test.npz": "best_single_catboost_test.npz",
    "artifacts/predictions/best_practical_lightgbm_test.npz": "best_practical_lightgbm_test.npz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copy_group(mapping: dict[str, str], destination: Path) -> list[dict[str, object]]:
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source_relative, destination_name in mapping.items():
        source = ROOT / source_relative
        target = destination / destination_name
        if not source.is_file():
            raise FileNotFoundError(f"baseline source missing: {source_relative}")
        if target.exists():
            raise FileExistsError(f"baseline snapshot refuses overwrite: {target}")
        shutil.copy2(source, target)
        source_hash = sha256(source)
        target_hash = sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"snapshot hash mismatch: {source_relative}")
        records.append(
            {
                "source": source_relative,
                "snapshot": target.relative_to(ROOT).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": target_hash,
            }
        )
    return records


def main() -> int:
    manifest_path = REPORT_DEST / "BASELINE_SNAPSHOT.json"
    if manifest_path.exists() or REPORT_DEST.exists() or ARTIFACT_DEST.exists():
        raise FileExistsError("baseline_v1 snapshot already exists; refusing overwrite")
    records = _copy_group(REPORT_FILES, REPORT_DEST)
    records.extend(_copy_group(ARTIFACT_FILES, ARTIFACT_DEST))
    payload = {
        "generation_id": "baseline_v1_final_v1_versioned_equal_60k",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "immutable rollback snapshot; feature-v2 work must not overwrite these files",
        "files": records,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": manifest_path.relative_to(ROOT).as_posix(),
                "files": len(records),
                "all_hashes_verified": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
