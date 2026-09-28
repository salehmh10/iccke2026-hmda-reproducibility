"""One-time frozen-test evaluation and packaging for the sealed V2 finalists."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.calibration import calibration_curve

from scripts.evaluate_all import _bootstrap_intervals, _fit_calibration
from scripts.train_dl import _make_torch_model
from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.evaluation.metrics import evaluate_binary
from src.models.artifact import FinalHybridArtifact, FinalSklearnArtifact, _calibrate
from src.models.classical import denial_probability
from src.models.neural import predict_neural_probability
from src.training.context_v2 import build_experiment_context_v2


SEED = 20260809


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_ledger(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["experiment_id"]: row for row in csv.DictReader(handle)}


def _load_validation(prediction_dir: Path, experiment_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(prediction_dir / f"{experiment_id}_validation.npz")
    return data["source_indices"], data["y_true"], data["denial_probability"]


def _load_torch_model(model_dir: Path, experiment_id: str) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(
        model_dir / f"{experiment_id}.pt", map_location="cpu", weights_only=False
    )
    model_name = str(checkpoint["model_name"])
    params = checkpoint["model_parameters"]
    model, _ = _make_torch_model(model_name, int(params["input_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, model_name


def _candidate_rows(selection: dict[str, object], ledger: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {
        "best_predictive_hybrid": ledger[str(selection["best_predictive"])],
        "best_single_model": ledger[str(selection["best_single_model"])],
        "best_practical_model": ledger[str(selection["best_practical"])],
    }
    for label, row in result.items():
        if row["status"] != "completed" or row["degenerate"].lower() == "true":
            raise RuntimeError(f"sealed candidate is invalid: {label}")
    return result


def main() -> int:
    root = PROJECT_ROOT
    report_dir = root / "reports" / "generations" / "feature_v2"
    artifact_root = root / "artifacts" / "generations" / "feature_v2"
    model_dir = artifact_root / "models"
    prediction_dir = artifact_root / "predictions"
    guard_path = artifact_root / "TEST_ACCESS_GUARD.json"
    final_lock = artifact_root / "FINAL_GENERATION.lock.json"
    if final_lock.exists() or guard_path.exists():
        raise RuntimeError("V2 test generation is already accessed/frozen; refusing re-access")

    selection_path = artifact_root / "PRETEST_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_accessed_during_selection") is not False:
        raise RuntimeError("invalid pretest selection metadata")
    ledger = _read_ledger(report_dir / "EXPERIMENT_RESULTS.csv")
    candidates = _candidate_rows(selection, ledger)

    # Preflight every component and fit calibration using validation before any test-row access.
    hybrid_row = candidates["best_predictive_hybrid"]
    hybrid_spec_path = model_dir / f"{hybrid_row['experiment_id']}.json"
    hybrid_spec = json.loads(hybrid_spec_path.read_text(encoding="utf-8"))
    ml_bundle = joblib.load(model_dir / f"{hybrid_spec['ml_experiment_id']}.joblib")
    dl_row = ledger[hybrid_spec["dl_experiment_id"]]
    if dl_row["model"] == "tabnet":
        raise RuntimeError("selected TabNet hybrid requires a separate inference wrapper")
    dl_model, dl_model_name = _load_torch_model(model_dir, hybrid_spec["dl_experiment_id"])

    single_row = candidates["best_single_model"]
    practical_row = candidates["best_practical_model"]
    single_path = model_dir / f"{single_row['experiment_id']}.joblib"
    practical_path = model_dir / f"{practical_row['experiment_id']}.joblib"
    if not single_path.is_file() or not practical_path.is_file():
        raise RuntimeError("sealed single/practical finalist is not a sklearn-compatible bundle")
    single_bundle = joblib.load(single_path)
    practical_bundle = joblib.load(practical_path)

    validation_payload: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    calibrators: dict[str, object | None] = {}
    thresholds: dict[str, float] = {}
    calibration_details: dict[str, object] = {}
    for label, row in candidates.items():
        _, y_validation, p_validation = _load_validation(prediction_dir, row["experiment_id"])
        calibrator, threshold, details = _fit_calibration(y_validation, p_validation)
        validation_payload[label] = (y_validation, p_validation)
        calibrators[label] = calibrator
        thresholds[label] = threshold
        calibration_details[label] = details

    context = build_experiment_context_v2(
        root=root, max_fit_rows=60_000, seed=SEED, reuse_preprocessor=True
    )
    guard = {
        "generation_id": selection["generation_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
        "policy": "exclusive one-time V2 frozen-test access; any interrupted run requires audit, not automatic retry",
        "pretest_selection_sha256": _sha256(selection_path),
        "test_indices_sha256": selection["split_index_sha256"]["test"],
    }
    artifact_root.mkdir(parents=True, exist_ok=True)
    with guard_path.open("x", encoding="utf-8") as handle:
        json.dump(guard, handle, indent=2, sort_keys=True)
        handle.write("\n")

    # First and only V2 access to test rows begins here.
    test_frame = context.prepared.modeling.loc[context.splits.test].drop(
        columns=[RAW_TARGET, ANALYTICAL_TARGET]
    )
    test_y = context.prepared.modeling.loc[
        context.splits.test, ANALYTICAL_TARGET
    ].to_numpy(dtype=np.int8)
    test_features = context.preprocessor.transform(test_frame)

    raw_test = {
        "best_predictive_hybrid": (
            float(hybrid_spec["ml_weight"])
            * denial_probability(ml_bundle["model"], test_features)
            + float(hybrid_spec["dl_weight"])
            * predict_neural_probability(dl_model, test_features, device="cpu")
        ),
        "best_single_model": denial_probability(single_bundle["model"], test_features),
        "best_practical_model": denial_probability(practical_bundle["model"], test_features),
    }
    calibrated_test: dict[str, np.ndarray] = {}
    final_rows: list[dict[str, object]] = []
    for label in candidates:
        probability = np.clip(_calibrate(calibrators[label], raw_test[label]), 0.0, 1.0)
        calibrated_test[label] = probability
        metrics = evaluate_binary(test_y, probability, thresholds[label])
        final_rows.append({
            "candidate": label,
            "experiment_id": candidates[label]["experiment_id"],
            "model": candidates[label]["model"],
            "dataset_variant": candidates[label]["dataset_variant"],
            **metrics,
        })
        np.savez_compressed(
            prediction_dir / f"{label}_test.npz",
            source_indices=context.splits.test,
            y_true=test_y,
            denial_probability=probability,
            threshold=np.array([thresholds[label]]),
        )

    results_path = report_dir / "FINAL_TEST_RESULTS.csv"
    with results_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
        writer.writeheader()
        writer.writerows(final_rows)
    calibration_path = report_dir / "CALIBRATION_RESULTS.json"
    calibration_path.write_text(
        json.dumps(calibration_details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metadata = {
        **selection,
        "test_rows": int(len(test_y)),
        "dl_component_model": dl_model_name,
        "calibration_and_threshold_data": "validation_only",
    }
    predictive_artifact = FinalHybridArtifact(
        preprocessor=context.preprocessor,
        ml_model=ml_bundle["model"],
        dl_model=dl_model,
        ml_weight=float(hybrid_spec["ml_weight"]),
        calibrator=calibrators["best_predictive_hybrid"],
        threshold=thresholds["best_predictive_hybrid"],
        metadata=metadata,
    )
    single_artifact = FinalSklearnArtifact(
        preprocessor=context.preprocessor,
        model=single_bundle["model"],
        calibrator=calibrators["best_single_model"],
        threshold=thresholds["best_single_model"],
        metadata=metadata,
    )
    practical_artifact = FinalSklearnArtifact(
        preprocessor=context.preprocessor,
        model=practical_bundle["model"],
        calibrator=calibrators["best_practical_model"],
        threshold=thresholds["best_practical_model"],
        metadata=metadata,
    )
    package_paths = {
        "best_predictive_model.joblib": predictive_artifact,
        "best_single_model.joblib": single_artifact,
        "best_practical_model.joblib": practical_artifact,
    }
    for name, artifact in package_paths.items():
        joblib.dump(artifact, artifact_root / name)

    expected = {
        "best_predictive_model.joblib": calibrated_test["best_predictive_hybrid"],
        "best_single_model.joblib": calibrated_test["best_single_model"],
        "best_practical_model.joblib": calibrated_test["best_practical_model"],
    }
    reload_differences: dict[str, float] = {}
    for name in package_paths:
        reloaded = joblib.load(artifact_root / name)
        actual = reloaded.predict_denial_probability(test_frame)
        difference = float(np.max(np.abs(expected[name] - actual)))
        reload_differences[name] = difference
        if difference > 1e-8:
            raise RuntimeError(f"full-test artifact reload mismatch for {name}: {difference}")
    reload_path = report_dir / "ARTIFACT_RELOAD_AUDIT.json"
    reload_path.write_text(
        json.dumps({
            "rows": int(len(test_y)),
            "probability_atol": 1e-8,
            "max_absolute_difference": reload_differences,
            "source_indices_exact": True,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    intervals_path = report_dir / "BOOTSTRAP_INTERVALS.json"
    intervals_path.write_text(
        json.dumps({
            "repeats": 100,
            "seed": SEED,
            "percentile_95_ci": _bootstrap_intervals(
                test_y,
                calibrated_test["best_predictive_hybrid"],
                thresholds["best_predictive_hybrid"],
                repeats=100,
            ),
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    calibration_figure = figures / "v2_final_calibration_curves.png"
    plt.figure(figsize=(8, 6))
    for label, probability in calibrated_test.items():
        observed, predicted = calibration_curve(test_y, probability, n_bins=10, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    plt.xlabel("Mean predicted denial probability")
    plt.ylabel("Observed denial rate")
    plt.title("V2 finalist test calibration (frozen on validation)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(calibration_figure, dpi=160)
    plt.close()

    prediction = (
        calibrated_test["best_predictive_hybrid"]
        >= thresholds["best_predictive_hybrid"]
    ).astype(np.int8)
    fairness_lines = [
        "# Feature V2 Fairness Audit", "",
        "Descriptive frozen-test audit only. Protected attributes were excluded from model inputs; this is not a legal fair-lending determination or causal analysis.", "",
    ]
    for column in ("applicant_race_name_1", "applicant_ethnicity_name", "applicant_sex_name"):
        fairness_lines += [f"## {column}", "", "| Group | N | Observed denial rate | Predicted denial rate | Denial recall |", "|---|---:|---:|---:|---:|"]
        groups = context.prepared.modeling.loc[context.splits.test, column].astype(str).to_numpy()
        for group in sorted(np.unique(groups)):
            mask = groups == group
            if int(mask.sum()) < 500:
                continue
            positives = test_y[mask] == 1
            recall = float(prediction[mask][positives].mean()) if positives.any() else float("nan")
            fairness_lines.append(
                f"| {group} | {int(mask.sum())} | {float(test_y[mask].mean()):.4f} | {float(prediction[mask].mean()):.4f} | {recall:.4f} |"
            )
        fairness_lines.append("")
    fairness_path = report_dir / "FAIRNESS_AUDIT.md"
    fairness_path.write_text("\n".join(fairness_lines) + "\n", encoding="utf-8")

    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "lightgbm", "catboost", "torch", "pytorch-tabnet"):
        packages[package] = importlib.metadata.version(package)
    manifest_path = artifact_root / "FINAL_ARTIFACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps({
            "generation_id": selection["generation_id"],
            "source_sha256": context.prepared.source_sha256,
            "split_index_sha256": context.splits.hashes(),
            "pretest_selection_sha256": _sha256(selection_path),
            "packages": packages,
            "artifacts": {name: _sha256(artifact_root / name) for name in package_paths},
            "reload_audit_sha256": _sha256(reload_path),
            "full_reload_rows": int(len(test_y)),
            "reload_probability_atol": 1e-8,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    guard["status"] = "complete"
    guard["completed_utc"] = datetime.now(timezone.utc).isoformat()
    guard["test_rows"] = int(len(test_y))
    guard["outputs_sha256"] = {
        results_path.relative_to(root).as_posix(): _sha256(results_path),
        manifest_path.relative_to(root).as_posix(): _sha256(manifest_path),
        reload_path.relative_to(root).as_posix(): _sha256(reload_path),
    }
    guard_path.write_text(json.dumps(guard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "test_results": final_rows,
        "full_reload_max_difference": reload_differences,
        "test_access_guard": "complete",
    }, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
