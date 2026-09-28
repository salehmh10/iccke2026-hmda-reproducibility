"""Hash-lock the complete test-bearing Feature V2 generation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "artifacts" / "generations" / "feature_v2" / "FINAL_GENERATION.lock.json"
FROZEN_PATHS = (
    "artifacts/generations/feature_v2/PRETEST_SELECTION.json",
    "artifacts/generations/feature_v2/TEST_ACCESS_GUARD.json",
    "artifacts/generations/feature_v2/FINAL_ARTIFACT_MANIFEST.json",
    "artifacts/generations/feature_v2/best_predictive_model.joblib",
    "artifacts/generations/feature_v2/best_single_model.joblib",
    "artifacts/generations/feature_v2/best_practical_model.joblib",
    "artifacts/generations/feature_v2/predictions/best_predictive_hybrid_test.npz",
    "artifacts/generations/feature_v2/predictions/best_single_model_test.npz",
    "artifacts/generations/feature_v2/predictions/best_practical_model_test.npz",
    "reports/generations/feature_v2/ACCEPTANCE_RULE.json",
    "reports/generations/feature_v2/CONFIRMED_FEATURE_CONFIG.json",
    "reports/generations/feature_v2/FINAL_FEATURE_SELECTION.json",
    "reports/generations/feature_v2/FEATURE_MI.csv",
    "reports/generations/feature_v2/FEATURE_VALIDATION.csv",
    "reports/generations/feature_v2/EXPERIMENT_RESULTS.csv",
    "reports/generations/feature_v2/VALIDATION_V1_V2_COMPARISON.csv",
    "reports/generations/feature_v2/FINAL_TEST_RESULTS.csv",
    "reports/generations/feature_v2/FINAL_TEST_V1_V2_COMPARISON.csv",
    "reports/generations/feature_v2/CALIBRATION_RESULTS.json",
    "reports/generations/feature_v2/BOOTSTRAP_INTERVALS.json",
    "reports/generations/feature_v2/ARTIFACT_RELOAD_AUDIT.json",
    "reports/generations/feature_v2/FAIRNESS_AUDIT.md",
    "reports/generations/feature_v2/FEATURE_ENGINEERING_V2_REPORT_FA.md",
    "reports/generations/feature_v2/FEATURE_ENGINEERING_V2_REPORT_FA.pdf",
    "reports/generations/feature_v2/REPORT_MANIFEST.json",
    "reports/generations/feature_v2/figures/oof_ablation_pr_auc.png",
    "reports/generations/feature_v2/figures/validation_delta_by_family.png",
    "reports/generations/feature_v2/figures/v1_v2_test_predictive.png",
    "reports/generations/feature_v2/figures/v2_final_calibration_curves.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if LOCK.exists():
        raise RuntimeError(f"V2 generation is already frozen: {LOCK}")
    missing = [relative for relative in FROZEN_PATHS if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot freeze incomplete V2 generation: {missing}")
    guard = json.loads(
        (ROOT / "artifacts/generations/feature_v2/TEST_ACCESS_GUARD.json").read_text(encoding="utf-8")
    )
    if guard.get("status") != "complete":
        raise RuntimeError("cannot freeze an incomplete V2 test generation")
    payload = {
        "generation_id": "feature_v2_onehot_numeric_equal_60k",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "immutable final V2 generation; evaluator refuses test re-access/overwrite",
        "files_frozen": len(FROZEN_PATHS),
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
