"""Pure Prompt 4B3 Beat-classification and shrinkage utilities.

Importing this module does not read project data or fit a model.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

try:
    from .prompt4_metrics import provisional_acceptance
    from .prompt4b2_benefit import BenefitPreprocessor
except ImportError:
    from prompt4_metrics import provisional_acceptance
    from prompt4b2_benefit import BenefitPreprocessor


SEED = 42
CANDIDATE_ID = "prompt4b3__beat_classifier__main37_fixed"
LINEAR_ID = "prompt4b3__beat_linear"
COSTAWARE_ID = "prompt4b3__beat_costaware"
BASE_FEATURE_COUNT = 35
FEATURE_COUNT = 37
GLOBAL_FEATURE = "global_prediction_feature"
PROPOSAL_FEATURE = "proposed_residual_correction"
ROUTED_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)
RELIABILITY_BINS = 10
BOOTSTRAP_RESAMPLES = 500

PROHIBITED_INFERENCE_FEATURES = frozenset(
    {
        "loan_amount_000s", "y_true", "target", "beat", "benefit", "benefit_oof",
        "realized_benefit", "realized_residual", "target_decile", "operational_tail",
        "p_tail", "component_disagreement", "ensemble_standard_deviation", "row_hash",
        "record_hash", "respondent_id", "minority_population", "majority_minority_tract",
        "applicant_ethnicity_name", "co_applicant_ethnicity_name", "applicant_race_name_1",
        "co_applicant_race_name_1", "applicant_sex_name", "co_applicant_sex_name",
    }
)


def finite_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and non-empty.")
    return array


def beat_target(benefit: Any) -> np.ndarray:
    """Use the frozen strict target: one only when Benefit is greater than zero."""
    values = finite_vector(benefit, "benefit")
    return (values > 0.0).astype(np.int8)


def reproduce_benefit(y_true: Any, global_prediction: Any, proposal: Any) -> np.ndarray:
    y = finite_vector(y_true, "y_true")
    global_values = finite_vector(global_prediction, "global_prediction")
    correction = finite_vector(proposal, "proposal")
    if not (y.shape == global_values.shape == correction.shape):
        raise ValueError("Benefit inputs must be aligned.")
    if np.any(correction < 0.0):
        raise ValueError("The frozen correction proposal must be non-negative.")
    return np.abs(y - global_values) - np.abs(y - (global_values + correction))


def cost_asymmetry(benefit: Any) -> dict[str, float | int]:
    values = finite_vector(benefit, "benefit")
    positive = values > 0.0
    negative = values < 0.0
    if not positive.any() or not negative.any():
        raise ValueError("p0 requires both positive and negative Benefit rows.")
    gain = float(np.mean(values[positive]))
    damage = float(np.mean(-values[negative]))
    p0 = damage / (gain + damage)
    return {
        "positive_rows": int(positive.sum()), "negative_rows": int(negative.sum()),
        "zero_rows": int((values == 0.0).sum()), "G": gain, "D": damage, "p0": float(p0),
    }


def validate_feature_contract(base_features: Sequence[str], router_features: Sequence[str] | None = None) -> list[str]:
    base = [str(item) for item in base_features]
    expected = base + [GLOBAL_FEATURE, PROPOSAL_FEATURE]
    if len(base) != BASE_FEATURE_COUNT or len(set(base)) != BASE_FEATURE_COUNT:
        raise ValueError("Prompt 4B3 requires exactly 35 unique clean base features.")
    if PROHIBITED_INFERENCE_FEATURES.intersection(base):
        raise ValueError("Prompt 4B3 base features contain a prohibited inference field.")
    actual = expected if router_features is None else [str(item) for item in router_features]
    if actual != expected or len(actual) != FEATURE_COUNT or len(set(actual)) != FEATURE_COUNT:
        raise ValueError("Prompt 4B3 requires the exact ordered Prompt 4B2 37-feature contract.")
    return expected


def make_feature_frame(
    base_frame: pd.DataFrame,
    base_features: Sequence[str],
    global_prediction: Any,
    proposal: Any,
) -> pd.DataFrame:
    contract = validate_feature_contract(base_features)
    global_values = finite_vector(global_prediction, "global_prediction")
    correction = finite_vector(proposal, "proposal")
    if len(base_frame) != len(global_values) or correction.shape != global_values.shape:
        raise ValueError("Prompt 4B3 feature inputs must be aligned.")
    if np.any(correction < 0.0):
        raise ValueError("Prompt 4B3 correction proposal must be non-negative.")
    missing = [name for name in base_features if name not in base_frame.columns]
    if missing:
        raise ValueError(f"Missing Prompt 4B3 features: {missing}")
    frame = base_frame.loc[:, list(base_features)].copy()
    frame[GLOBAL_FEATURE] = global_values
    frame[PROPOSAL_FEATURE] = correction
    return frame.loc[:, contract]


def validate_oof_membership(
    train_row_hashes: Sequence[Any], benefit_frame: pd.DataFrame, residual_frame: pd.DataFrame
) -> dict[str, Any]:
    hashes = pd.Series(train_row_hashes, copy=False).astype(str).reset_index(drop=True)
    if len(hashes) != 400_000 or hashes.duplicated().any():
        raise ValueError("Prompt 4B3 requires exactly 400,000 unique frozen Train rows.")
    for name, frame in (("Benefit", benefit_frame), ("Residual", residual_frame)):
        if len(frame) != len(hashes) or "row_hash" not in frame:
            raise ValueError(f"{name} OOF rows do not match frozen Train membership.")
        if not hashes.equals(frame["row_hash"].astype(str).reset_index(drop=True)):
            raise ValueError(f"{name} OOF row order differs from frozen Train membership.")
    if not residual_frame["exactly_one_oof_prediction"].astype(bool).all():
        raise ValueError("A Train row does not have exactly one Residual OOF prediction.")
    if residual_frame["self_fit"].astype(bool).any():
        raise ValueError("Residual OOF evidence contains a self-fit row.")
    return {
        "train_rows": len(hashes), "unique_train_rows": int(hashes.nunique()),
        "benefit_rows": len(benefit_frame), "residual_rows": len(residual_frame),
        "zero_self_fit_rows": int(residual_frame["self_fit"].astype(bool).sum()),
        "exactly_one_oof_all": True,
    }


def classifier_config_from_anchor(anchor: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only documented structural values, then apply Prompt 4B3 overrides."""
    required = ("depth", "learning_rate", "l2_leaf_reg", "random_strength")
    missing = [key for key in required if key not in anchor]
    if missing:
        raise ValueError(f"Benefit Router anchor is ambiguous; missing {missing}.")
    optional = ("bagging_temperature", "border_count", "grow_policy", "bootstrap_type")
    config = {key: anchor[key] for key in (*required, *optional) if key in anchor}
    config.update(
        {
            "loss_function": "Logloss", "iterations": 1500, "random_seed": SEED,
            "thread_count": 4, "task_type": "CPU", "allow_writing_files": False,
            "verbose": False,
        }
    )
    prohibited = {"class_weights", "auto_class_weights", "early_stopping_rounds", "use_best_model"}
    if prohibited.intersection(config):
        raise ValueError("Prompt 4B3 classifier configuration contains a prohibited option.")
    return config


def alpha_linear(p_beat: Any) -> np.ndarray:
    return np.clip(finite_vector(p_beat, "p_beat"), 0.0, 1.0)


def alpha_costaware(p_beat: Any, p0: float) -> np.ndarray:
    score = finite_vector(p_beat, "p_beat")
    threshold = float(p0)
    if not 0.0 < threshold < 1.0:
        raise ValueError("p0 must be strictly between zero and one.")
    return np.clip(np.maximum(0.0, (score - threshold) / (1.0 - threshold)), 0.0, 1.0)


def policy_prediction(global_prediction: Any, proposal: Any, alpha: Any) -> np.ndarray:
    global_values = finite_vector(global_prediction, "global_prediction")
    correction = finite_vector(proposal, "proposal")
    weight = finite_vector(alpha, "alpha")
    if not (global_values.shape == correction.shape == weight.shape):
        raise ValueError("Policy inputs must be aligned.")
    if np.any(correction < 0.0) or np.any((weight < 0.0) | (weight > 1.0)):
        raise ValueError("Policy correction and alpha bounds failed.")
    return global_values + weight * correction


def fixed_fraction_diagnostics(
    beat: Any,
    benefit: Any,
    score: Any,
    fractions: Sequence[float] = ROUTED_FRACTIONS,
) -> pd.DataFrame:
    labels = finite_vector(beat, "beat").astype(np.int8)
    realized = finite_vector(benefit, "benefit")
    ranking = finite_vector(score, "score")
    if not (labels.shape == realized.shape == ranking.shape) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Routing diagnostic inputs must be aligned and binary.")
    base = float(labels.mean())
    positive_total = int(labels.sum())
    order = np.argsort(-ranking, kind="stable")
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        value = float(fraction)
        if not 0.0 < value <= 1.0:
            raise ValueError("Routed fractions must be in (0, 1].")
        count = int(round(len(labels) * value))
        selected = order[:count]
        precision = float(labels[selected].mean())
        rows.append(
            {
                "routed_fraction": value, "routed_rows": count, "precision": precision,
                "recall": float(labels[selected].sum() / positive_total),
                "mean_realized_benefit": float(realized[selected].mean()),
                "positive_benefit_prevalence": precision,
                "lift_over_base_prevalence": float(precision / base),
                "base_beat_prevalence": base,
            }
        )
    return pd.DataFrame(rows)


def binary_score_metrics(beat: Any, score: Any) -> dict[str, float | int]:
    labels = finite_vector(beat, "beat").astype(np.int8)
    values = finite_vector(score, "score")
    if labels.shape != values.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary metrics need aligned binary labels.")
    return {
        "rows": int(len(labels)), "beat_positive_count": int(labels.sum()),
        "beat_prevalence": float(labels.mean()), "roc_auc": float(roc_auc_score(labels, values)),
        "pr_auc_average_precision": float(average_precision_score(labels, values)),
    }


def classifier_metrics(beat: Any, probability: Any) -> dict[str, float | int]:
    labels = finite_vector(beat, "beat").astype(np.int8)
    values = finite_vector(probability, "probability")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Classifier probabilities must be in [0, 1].")
    return {
        **binary_score_metrics(labels, values),
        "brier_score": float(brier_score_loss(labels, values)),
        "log_loss": float(log_loss(labels, values, labels=[0, 1])),
    }


def reliability_table(beat: Any, probability: Any, bins: int = RELIABILITY_BINS) -> pd.DataFrame:
    labels = finite_vector(beat, "beat").astype(np.int8)
    values = finite_vector(probability, "probability")
    if bins != 10 or labels.shape != values.shape or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Prompt 4B3 reliability uses exactly 10 bins and aligned probabilities.")
    index = np.minimum((values * bins).astype(int), bins - 1)
    rows = []
    for bin_id in range(bins):
        mask = index == bin_id
        rows.append(
            {
                "bin": bin_id, "lower": bin_id / bins, "upper": (bin_id + 1) / bins,
                "rows": int(mask.sum()),
                "mean_probability": float(values[mask].mean()) if mask.any() else float("nan"),
                "observed_beat_rate": float(labels[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def paired_pr_auc_bootstrap(
    beat: Any,
    new_score: Any,
    baseline_score: Any,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = SEED,
) -> dict[str, float | int]:
    labels = finite_vector(beat, "beat").astype(np.int8)
    new = finite_vector(new_score, "new_score")
    baseline = finite_vector(baseline_score, "baseline_score")
    if not (labels.shape == new.shape == baseline.shape) or n_resamples != 500 or random_state != 42:
        raise ValueError("Diagnostic bootstrap is frozen to aligned rows, 500 resamples, seed 42.")
    point = float(average_precision_score(labels, new) - average_precision_score(labels, baseline))
    rng = np.random.default_rng(random_state)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, len(labels), size=len(labels))
        differences[index] = average_precision_score(labels[sample], new[sample]) - average_precision_score(labels[sample], baseline[sample])
    return {
        "rows": int(len(labels)), "n_resamples": n_resamples, "random_state": random_state,
        "pr_auc_difference": point, "percentile_2_5": float(np.quantile(differences, 0.025)),
        "median": float(np.median(differences)), "percentile_97_5": float(np.quantile(differences, 0.975)),
    }


def six_condition_evaluator(candidate: Mapping[str, Any], global_reference: Mapping[str, Any]) -> dict[str, Any]:
    """The one shared six-condition evaluator for every Prompt 4B3 policy."""
    return provisional_acceptance(candidate, global_reference)


def select_policy(selection_rows: Sequence[Mapping[str, Any]]) -> str:
    rows = [dict(row) for row in selection_rows]
    if {str(row.get("candidate_id")) for row in rows} != {LINEAR_ID, COSTAWARE_ID}:
        raise ValueError("Policy selection requires exactly the two frozen Prompt 4B3 policies.")
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row["conditions_passed"]), float(row["mae"]), float(row["top_decile_mae"]),
            0 if str(row["candidate_id"]) == COSTAWARE_ID else 1,
        ),
    )
    return str(ordered[0]["candidate_id"])


@dataclass
class BeatClassifierBundle:
    preprocessor: BenefitPreprocessor
    model: Any
    feature_contract: list[str]
    metadata: dict[str, Any]

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame, pd.DataFrame) or list(frame.columns) != self.feature_contract:
            raise ValueError("Beat bundle input must use the exact ordered 37-feature contract.")
        transformed = self.preprocessor.transform(frame)
        probability = np.asarray(self.model.predict_proba(transformed), dtype=np.float64)[:, 1]
        if probability.shape != (len(frame),) or not np.isfinite(probability).all():
            raise RuntimeError("Beat classifier produced invalid probabilities.")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise RuntimeError("Beat classifier probabilities are outside [0, 1].")
        return probability

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_probability(frame)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_bundle(bundle: BeatClassifierBundle, destination: str | Path) -> dict[str, Any]:
    directory = Path(destination)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / "bundle.joblib"
    native = directory / "classifier.cbm"
    temp_artifact = directory / "bundle.joblib.tmp"
    temp_native = directory / "classifier.cbm.tmp"
    joblib.dump(bundle, temp_artifact, compress=3)
    reloaded = joblib.load(temp_artifact)
    if not isinstance(reloaded, BeatClassifierBundle) or reloaded.feature_contract != bundle.feature_contract:
        raise RuntimeError("Beat classifier temporary bundle reload failed.")
    bundle.model.save_model(str(temp_native), format="cbm")
    os.replace(temp_artifact, artifact)
    os.replace(temp_native, native)
    manifest = {
        "status": "COMPLETE", "bundle_type": "prompt4b3_beat_classifier", "bundle_format_version": 1,
        "artifact": artifact.name, "artifact_sha256": sha256_file(artifact),
        "native_artifact": native.name, "native_sha256": sha256_file(native),
        "feature_contract": bundle.feature_contract, "feature_count": len(bundle.feature_contract),
        "metadata": bundle.metadata, "preprocessing": bundle.preprocessor.evidence(),
    }
    temp_manifest = directory / "manifest.json.tmp"
    temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    json.loads(temp_manifest.read_text(encoding="utf-8"))
    os.replace(temp_manifest, directory / "manifest.json")
    return manifest


def load_bundle(destination: str | Path) -> BeatClassifierBundle:
    directory = Path(destination)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    artifact = directory / manifest["artifact"]
    native = directory / manifest["native_artifact"]
    if manifest.get("status") != "COMPLETE" or sha256_file(artifact) != manifest.get("artifact_sha256"):
        raise RuntimeError("Beat classifier bundle manifest or hash failed.")
    if sha256_file(native) != manifest.get("native_sha256"):
        raise RuntimeError("Beat classifier native model hash failed.")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, BeatClassifierBundle) or bundle.feature_contract != manifest.get("feature_contract"):
        raise RuntimeError("Beat classifier bundle payload contract failed.")
    return bundle


def validate_read_path(root: str | Path, path: str | Path) -> Path:
    workspace = Path(root).resolve()
    candidate = Path(path)
    resolved = (workspace / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise PermissionError("Prompt 4B3 cannot read outside regression_v2.") from error
    prohibited = {
        (workspace / "outputs/data/iid_holdout_features.parquet").resolve(),
        (workspace / "outputs/data/iid_holdout_targets.parquet").resolve(),
    }
    raw = (workspace / "data").resolve()
    if resolved in prohibited or resolved == raw or raw in resolved.parents:
        raise PermissionError("Prompt 4B3 read guard blocks Raw and IID paths.")
    return resolved


def resumable_model_status(model_directory: str | Path, ledger: Mapping[str, Any]) -> str:
    directory = Path(model_directory)
    if (directory / "manifest.json").exists() and (directory / "bundle.joblib").exists():
        load_bundle(directory)
        return "REUSE_VALID_MODEL"
    attempts = list(ledger.get("physical_attempts", []))
    if any(str(item.get("status")) == "IN_PROGRESS" for item in attempts):
        return "INSPECT_ACTIVE_ATTEMPT"
    completed = [item for item in attempts if str(item.get("status")) == "COMPLETE"]
    if completed:
        return "BLOCKED_MISSING_COMPLETED_MODEL"
    return "START_FIRST_ATTEMPT" if not attempts else "TECHNICAL_RETRY_ELIGIBLE"
