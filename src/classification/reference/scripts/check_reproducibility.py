"""Deterministic rerun, manifest, and final-artifact integrity verification."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import optimize_threshold
from src.models.classical import denial_probability, make_classical_model
from src.training.context import build_experiment_context, transform_variant_training
from src.training.tracking import append_record, build_record, experiment_exists


SEED = 20260809


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != str(SEED):
        raise RuntimeError(f"set PYTHONHASHSEED={SEED} before starting Python")
    context = build_experiment_context(root=ROOT, max_fit_rows=60000, seed=SEED)
    variant = context.variants["undersampled"]
    x_train, y_train, _, _ = transform_variant_training(
        context, variant, max_rows=12000, seed=SEED
    )
    ledger = ROOT / "reports" / "EXPERIMENT_RESULTS.csv"
    probabilities = []
    runtimes = []
    for label in ("a", "b"):
        experiment_id = f"repro_lightgbm_{label}_undersampled_s{SEED}_n12000"
        model = make_classical_model("lightgbm", seed=SEED, n_jobs=1)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        runtime = time.perf_counter() - started
        probability = denial_probability(model, context.validation_features)
        probabilities.append(probability)
        runtimes.append(runtime)
        if not experiment_exists(ledger, experiment_id):
            threshold = optimize_threshold(context.validation_target, probability)
            append_record(
                ledger,
                build_record(
                    root=ROOT,
                    experiment_id=experiment_id,
                    dataset_variant="undersampled",
                    sample_size=len(y_train),
                    feature_set="primary_financial_v1_repro_check",
                    model="lightgbm_repro",
                    hyperparameters={"n_jobs": 1, "purpose": "deterministic_rerun"},
                    seed=SEED,
                    class_weighting="none",
                    sampler=variant.sampler,
                    train_runtime_seconds=runtime,
                    threshold=threshold.threshold,
                    status="completed",
                    metrics=threshold.metrics,
                ),
            )
    max_difference = float(np.max(np.abs(probabilities[0] - probabilities[1])))
    if max_difference != 0.0:
        raise RuntimeError(f"deterministic rerun prediction mismatch: {max_difference}")

    artifact_manifest_path = ROOT / "artifacts" / "models" / "FINAL_ARTIFACT_MANIFEST.json"
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    for name, expected in artifact_manifest["artifacts"].items():
        actual = _sha256(ROOT / "artifacts" / "models" / name)
        if actual != expected:
            raise RuntimeError(f"final artifact hash mismatch for {name}")
    result = {
        "seed": SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "rows": int(len(y_train)),
        "model": "lightgbm",
        "n_jobs": 1,
        "training_runtime_seconds": runtimes,
        "max_validation_probability_difference": max_difference,
        "validation_probability_sha256": hashlib.sha256(
            np.asarray(probabilities[0], dtype="<f8").tobytes()
        ).hexdigest(),
        "source_sha256": context.prepared.source_sha256,
        "split_index_sha256": context.splits.hashes(),
        "final_artifact_hashes_verified": True,
        "status": "passed",
    }
    (ROOT / "reports" / "REPRODUCIBILITY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
