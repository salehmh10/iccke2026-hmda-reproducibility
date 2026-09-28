"""Irreversibly freeze the current final test-bearing generation by hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "artifacts" / "models" / "FINAL_GENERATION.lock.json"
FROZEN_PATHS = (
    "artifacts/models/best_predictive_model.joblib",
    "artifacts/models/best_practical_model.joblib",
    "artifacts/models/finalist_selection.json",
    "artifacts/models/FINAL_ARTIFACT_MANIFEST.json",
    "artifacts/predictions/best_predictive_hybrid_test.npz",
    "artifacts/predictions/best_single_catboost_test.npz",
    "artifacts/predictions/best_practical_lightgbm_test.npz",
    "reports/FINAL_TEST_RESULTS.csv",
    "reports/CALIBRATION_RESULTS.json",
    "reports/BOOTSTRAP_INTERVALS.json",
    "reports/FAIRNESS_AUDIT.md",
    "reports/MODEL_COMPARISON.md",
    "reports/figures/final_calibration_curves.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if LOCK.exists():
        raise RuntimeError(f"final generation is already frozen: {LOCK}")
    missing = [relative for relative in FROZEN_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze incomplete generation: {missing}")
    payload = {
        "generation_id": "final_v1_versioned_equal_60k",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "evaluate_all refuses test re-access or overwrite while this verified lock exists",
        "sha256": {relative: sha256(ROOT / relative) for relative in FROZEN_PATHS},
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"lock": str(LOCK), "files_frozen": len(FROZEN_PATHS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
