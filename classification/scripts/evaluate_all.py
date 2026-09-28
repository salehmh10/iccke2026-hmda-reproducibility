"""Freeze validation-selected finalists, calibrate, evaluate test once, and serialize."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.evaluation.metrics import evaluate_binary, optimize_threshold
from src.models.artifact import FinalHybridArtifact, FinalSklearnArtifact, _calibrate
from src.models.classical import denial_probability
from src.models.neural import ModernHopfieldClassifier, predict_neural_probability
from src.training.context import build_experiment_context, feature_contract_sha256


SEED = 20260809
SINGLE_MODELS = {
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "bagging",
    "stacking",
}


def _load_prediction(root: Path, experiment_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(root / "artifacts" / "predictions" / f"{experiment_id}_validation.npz")
    return data["source_indices"], data["y_true"], data["denial_probability"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fit_calibration(y: np.ndarray, probability: np.ndarray) -> tuple[object | None, float, dict[str, object]]:
    positions = np.arange(len(y), dtype=np.int64)
    fit_positions, select_positions = train_test_split(
        positions, train_size=0.5, stratify=y, random_state=SEED, shuffle=True
    )
    calibrators: dict[str, object | None] = {"none": None}
    sigmoid = LogisticRegression(solver="lbfgs", random_state=SEED)
    sigmoid.fit(probability[fit_positions].reshape(-1, 1), y[fit_positions])
    calibrators["sigmoid"] = sigmoid
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(probability[fit_positions], y[fit_positions])
    calibrators["isotonic"] = isotonic

    scores: dict[str, dict[str, float]] = {}
    best_name = "none"
    best_objective = np.inf
    for name, calibrator in calibrators.items():
        selected_probability = np.clip(
            _calibrate(calibrator, probability[select_positions]), 1e-7, 1 - 1e-7
        )
        brier = float(brier_score_loss(y[select_positions], selected_probability))
        loss = float(log_loss(y[select_positions], selected_probability, labels=[0, 1]))
        objective = brier + 0.1 * loss
        scores[name] = {"brier_score": brier, "log_loss": loss, "objective": objective}
        if objective < best_objective:
            best_name = name
            best_objective = objective
    chosen = calibrators[best_name]
    threshold = optimize_threshold(
        y[select_positions], _calibrate(chosen, probability[select_positions])
    ).threshold
    return chosen, threshold, {
        "fit_rows": int(len(fit_positions)),
        "threshold_selection_rows": int(len(select_positions)),
        "selected": best_name,
        "scores": scores,
    }


def _bootstrap_intervals(
    y: np.ndarray, probability: np.ndarray, threshold: float, repeats: int = 100
) -> dict[str, list[float]]:
    rng = np.random.default_rng(SEED)
    values = {key: [] for key in ("pr_auc", "roc_auc", "mcc", "balanced_accuracy", "f1_macro")}
    for _ in range(repeats):
        sample = rng.integers(0, len(y), size=len(y))
        metrics = evaluate_binary(y[sample], probability[sample], threshold)
        for key in values:
            values[key].append(float(metrics[key]))
    return {
        key: [float(np.quantile(metric, 0.025)), float(np.quantile(metric, 0.975))]
        for key, metric in values.items()
    }


def _load_hopfield(root: Path, experiment_id: str) -> ModernHopfieldClassifier:
    checkpoint = torch.load(
        root / "artifacts" / "models" / f"{experiment_id}.pt",
        map_location="cpu",
        weights_only=False,
    )
    params = checkpoint["model_parameters"]
    model = ModernHopfieldClassifier(
        input_dim=int(params["input_dim"]),
        model_dim=int(params["model_dim"]),
        memory_patterns=int(params["memory_patterns"]),
        dropout=float(params["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main() -> int:
    root = PROJECT_ROOT
    generation_lock = root / "artifacts" / "models" / "FINAL_GENERATION.lock.json"
    if generation_lock.exists():
        frozen = json.loads(generation_lock.read_text(encoding="utf-8"))
        mismatches = []
        for relative, expected in frozen["sha256"].items():
            path = root / relative
            if not path.exists() or _sha256(path) != expected:
                mismatches.append(relative)
        if mismatches:
            raise RuntimeError(
                f"final generation lock integrity failure for: {mismatches}"
            )
        raise RuntimeError(
            "final test-bearing generation is frozen and verified; evaluate_all refuses overwrite/re-access"
        )
    ledger_path = root / "reports" / "EXPERIMENT_RESULTS.csv"
    with ledger_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    screening = [
        row for row in rows
        if row["status"] == "completed"
        and row["sample_size"] == "60000"
        and "_v" in row["experiment_id"]
    ]
    hybrid = [row for row in screening if row["model"] == "hybrid_ml_dl"]
    predictive_row = max(hybrid, key=lambda row: (float(row["pr_auc"]), float(row["mcc"])))
    single_row = max(
        (row for row in screening if row["model"] in SINGLE_MODELS),
        key=lambda row: (float(row["pr_auc"]), float(row["mcc"])),
    )
    practical_row = max(
        (row for row in screening if row["model"] in {"logistic_regression", "lightgbm"}),
        key=lambda row: (float(row["pr_auc"]), float(row["mcc"])),
    )
    selection = {
        "selection_data": "validation_only",
        "best_predictive": predictive_row["experiment_id"],
        "best_single_model": single_row["experiment_id"],
        "best_practical": practical_row["experiment_id"],
        "selection_rule": "PR-AUC primary, MCC tie-break; practical restricted to logistic/lightgbm",
        "test_accessed_during_selection": False,
    }
    context = build_experiment_context(root=root, max_fit_rows=60000, seed=SEED)
    selection.update(
        {
            "source_sha256": context.prepared.source_sha256,
            "split_index_sha256": context.splits.hashes(),
            "feature_contract_sha256": feature_contract_sha256(root),
        }
    )
    test_frame = context.prepared.modeling.loc[context.splits.test].drop(
        columns=[RAW_TARGET, ANALYTICAL_TARGET]
    )
    test_features = context.preprocessor.transform(test_frame)
    test_y = context.prepared.modeling.loc[context.splits.test, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)

    candidates: list[tuple[str, object, np.ndarray, np.ndarray, np.ndarray]] = []
    # Hybrid validation winner.
    hybrid_spec = json.loads(
        (root / "artifacts" / "models" / f"{predictive_row['experiment_id']}.json").read_text(encoding="utf-8")
    )
    selection["best_predictive_components"] = {
        "ml": hybrid_spec["ml_experiment_id"],
        "dl": hybrid_spec["dl_experiment_id"],
        "ml_weight": hybrid_spec["ml_weight"],
        "dl_weight": hybrid_spec["dl_weight"],
    }
    selection["best_practical_component"] = practical_row["experiment_id"]
    (root / "artifacts" / "models" / "finalist_selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ml_bundle = joblib.load(root / "artifacts" / "models" / f"{hybrid_spec['ml_experiment_id']}.joblib")
    dl_model = _load_hopfield(root, hybrid_spec["dl_experiment_id"])
    _, hybrid_y, hybrid_validation_probability = _load_prediction(root, predictive_row["experiment_id"])
    hybrid_test_probability = (
        hybrid_spec["ml_weight"] * denial_probability(ml_bundle["model"], test_features)
        + hybrid_spec["dl_weight"] * predict_neural_probability(dl_model, test_features, device="cpu")
    )
    candidates.append(("best_predictive_hybrid", (ml_bundle["model"], dl_model, hybrid_spec), hybrid_y, hybrid_validation_probability, hybrid_test_probability))

    for label, row in (("best_single_catboost", single_row), ("best_practical_lightgbm", practical_row)):
        bundle = joblib.load(root / "artifacts" / "models" / f"{row['experiment_id']}.joblib")
        _, validation_y, validation_probability = _load_prediction(root, row["experiment_id"])
        test_probability = denial_probability(bundle["model"], test_features)
        candidates.append((label, bundle["model"], validation_y, validation_probability, test_probability))

    final_rows: list[dict[str, object]] = []
    calibration_details: dict[str, object] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    calibrators: dict[str, object | None] = {}
    for label, _, validation_y, validation_probability, test_probability in candidates:
        calibrator, threshold, calibration = _fit_calibration(validation_y, validation_probability)
        calibrated_test = np.clip(_calibrate(calibrator, test_probability), 0.0, 1.0)
        metrics = evaluate_binary(test_y, calibrated_test, threshold)
        final_rows.append({"candidate": label, **metrics})
        calibration_details[label] = calibration
        test_probabilities[label] = calibrated_test
        thresholds[label] = threshold
        calibrators[label] = calibrator
        np.savez_compressed(
            root / "artifacts" / "predictions" / f"{label}_test.npz",
            source_indices=context.splits.test,
            y_true=test_y,
            denial_probability=calibrated_test,
            threshold=np.array([threshold]),
        )

    fieldnames = list(final_rows[0])
    with (root / "reports" / "FINAL_TEST_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)
    (root / "reports" / "CALIBRATION_RESULTS.json").write_text(
        json.dumps(calibration_details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    hybrid_model, hybrid_dl, hybrid_meta = candidates[0][1]
    best_predictive = FinalHybridArtifact(
        preprocessor=context.preprocessor,
        ml_model=hybrid_model,
        dl_model=hybrid_dl,
        ml_weight=float(hybrid_meta["ml_weight"]),
        calibrator=calibrators["best_predictive_hybrid"],
        threshold=thresholds["best_predictive_hybrid"],
        metadata=selection,
    )
    practical_model = candidates[2][1]
    best_practical = FinalSklearnArtifact(
        preprocessor=context.preprocessor,
        model=practical_model,
        calibrator=calibrators["best_practical_lightgbm"],
        threshold=thresholds["best_practical_lightgbm"],
        metadata=selection,
    )
    predictive_path = root / "artifacts" / "models" / "best_predictive_model.joblib"
    practical_path = root / "artifacts" / "models" / "best_practical_model.joblib"
    joblib.dump(best_predictive, predictive_path)
    joblib.dump(best_practical, practical_path)
    sample = test_frame.iloc[:128]
    expected_predictive = best_predictive.predict_proba(sample)
    expected_practical = best_practical.predict_proba(sample)
    reloaded_predictive = joblib.load(predictive_path)
    reloaded_practical = joblib.load(practical_path)
    if not np.allclose(expected_predictive, reloaded_predictive.predict_proba(sample), atol=1e-8):
        raise RuntimeError("best predictive artifact reload mismatch")
    if not np.allclose(expected_practical, reloaded_practical.predict_proba(sample), atol=1e-8):
        raise RuntimeError("best practical artifact reload mismatch")
    packages = {}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "catboost",
        "torch",
        "pytorch-tabnet",
    ):
        packages[package] = importlib.metadata.version(package)
    artifact_manifest = {
        "source_sha256": context.prepared.source_sha256,
        "split_index_sha256": context.splits.hashes(),
        "selection_manifest_sha256": _sha256(root / "artifacts" / "models" / "finalist_selection.json"),
        "packages": packages,
        "artifacts": {
            predictive_path.name: _sha256(predictive_path),
            practical_path.name: _sha256(practical_path),
        },
        "reload_smoke_rows": 128,
        "reload_probability_atol": 1e-8,
    }
    (root / "artifacts" / "models" / "FINAL_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    intervals = _bootstrap_intervals(
        test_y,
        test_probabilities["best_predictive_hybrid"],
        thresholds["best_predictive_hybrid"],
        repeats=100,
    )
    (root / "reports" / "BOOTSTRAP_INTERVALS.json").write_text(
        json.dumps({"repeats": 100, "seed": SEED, "percentile_95_ci": intervals}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure_path = root / "reports" / "figures" / "final_calibration_curves.png"
    plt.figure(figsize=(8, 6))
    for label, probability in test_probabilities.items():
        observed, predicted = calibration_curve(test_y, probability, n_bins=10, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=label)
    plt.plot([0, 1], [0, 1], "k--", label="ideal")
    plt.xlabel("Mean predicted denial probability")
    plt.ylabel("Observed denial rate")
    plt.title("Finalist test calibration (choices frozen on validation)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    fairness_lines = [
        "# Fairness Audit",
        "",
        "Descriptive test-split audit for the validation-selected predictive artifact. Protected attributes were not used as model inputs. These statistics are not a legal fair-lending determination and do not establish causality.",
        "",
    ]
    prediction = (test_probabilities["best_predictive_hybrid"] >= thresholds["best_predictive_hybrid"]).astype(np.int8)
    for column in ("applicant_race_name_1", "applicant_ethnicity_name", "applicant_sex_name"):
        fairness_lines.extend([f"## {column}", "", "| Group | N | Observed denial rate | Predicted denial rate | Denial recall |", "|---|---:|---:|---:|---:|"])
        groups = context.prepared.modeling.loc[context.splits.test, column].astype(str).to_numpy()
        for group in sorted(np.unique(groups)):
            mask = groups == group
            if int(mask.sum()) < 500:
                continue
            observed_rate = float(test_y[mask].mean())
            predicted_rate = float(prediction[mask].mean())
            positives = test_y[mask] == 1
            recall = float(prediction[mask][positives].mean()) if positives.any() else float("nan")
            fairness_lines.append(f"| {group} | {int(mask.sum())} | {observed_rate:.4f} | {predicted_rate:.4f} | {recall:.4f} |")
        fairness_lines.append("")
    (root / "reports" / "FAIRNESS_AUDIT.md").write_text("\n".join(fairness_lines) + "\n", encoding="utf-8")

    print(json.dumps({"selection": selection, "test_results": final_rows, "reload_smoke_rows": 128, "bootstrap_repeats": 100}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
