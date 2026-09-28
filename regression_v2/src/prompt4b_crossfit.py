"""Leakage-safe Global cross-fit and Meta-Gate components for Prompt 4B."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .deep_preprocessing import duplicate_safe_deciles, ordered_digest
    from .metrics import transform_target
    from .model_bundles import ModelBundle
    from .preprocessing import CatBoostFramePreprocessor, make_dense_tree_preprocessor, make_xgb_preprocessor
except ImportError:
    from deep_preprocessing import duplicate_safe_deciles, ordered_digest
    from metrics import transform_target
    from model_bundles import ModelBundle
    from preprocessing import CatBoostFramePreprocessor, make_dense_tree_preprocessor, make_xgb_preprocessor


SEED = 42
GLOBAL_FEATURE = "global_prediction_feature"


def membership_digest(values: Any) -> str:
    normalized = sorted(str(item) for item in values)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def deterministic_twofold_split(y_true: Any, row_hash: Any, q90_train: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target = np.asarray(y_true, dtype=np.float64).reshape(-1)
    hashes = pd.Series(row_hash).astype(str).reset_index(drop=True)
    if target.size != 400_000 or hashes.size != target.size or hashes.duplicated().any():
        raise ValueError("Prompt 4B cross-fit requires exactly 400,000 unique Train rows.")
    decile = duplicate_safe_deciles(target)
    tail = (target > float(q90_train)).astype(np.int8)
    stratify = np.asarray([f"{int(d)}_{int(t)}" for d, t in zip(decile, tail)], dtype=object)
    fold_a, fold_b = train_test_split(np.arange(target.size), train_size=200_000, test_size=200_000, random_state=SEED, shuffle=True, stratify=stratify)
    fold_a = np.sort(fold_a); fold_b = np.sort(fold_b)
    if np.intersect1d(fold_a, fold_b).size or np.union1d(fold_a, fold_b).size != target.size:
        raise RuntimeError("Cross-fit folds are not an exact partition.")
    evidence = {
        "status": "PASS", "random_state": SEED,
        "stratification": "duplicate-safe target decile plus operational Tail status",
        "fold_a_rows": int(fold_a.size), "fold_b_rows": int(fold_b.size), "overlap_rows": 0,
        "fold_a_row_hash_digest": ordered_digest(hashes.iloc[fold_a]),
        "fold_b_row_hash_digest": ordered_digest(hashes.iloc[fold_b]),
        "fold_a_membership_digest": membership_digest(hashes.iloc[fold_a]),
        "fold_b_membership_digest": membership_digest(hashes.iloc[fold_b]),
    }
    return fold_a, fold_b, evidence


def fixed_crossfit_parameters(saved_bundle: ModelBundle) -> dict[str, Any]:
    params = dict(saved_bundle.model_parameters)
    params.pop("early_stopping_rounds", None)
    fitted = int(saved_bundle.metadata.get("fitted_iterations", 0) or 0)
    if fitted < 1:
        fitted = int(saved_bundle.selected_best_iteration or 0) + (1 if saved_bundle.family in {"catboost", "xgboost"} else 0)
    if saved_bundle.family == "catboost":
        params["iterations"] = fitted
    else:
        params["n_estimators"] = fitted
    return params


def _fit_one_family(family: str, parameters: dict[str, Any], features: list[str], X_fit: pd.DataFrame, y_fit: np.ndarray, target_mode: str):
    transformed_target = transform_target(y_fit, target_mode)
    if family == "catboost":
        from catboost import CatBoostRegressor
        preprocessor = CatBoostFramePreprocessor(features).fit(X_fit)
        transformed = preprocessor.transform(X_fit)
        model = CatBoostRegressor(**parameters, allow_writing_files=False, task_type="CPU", verbose=False)
        model.fit(transformed, transformed_target, cat_features=preprocessor.cat_feature_indices_, verbose=False)
    elif family == "lightgbm":
        import lightgbm as lgb
        preprocessor = make_dense_tree_preprocessor(features, 100).fit(X_fit)
        transformed = preprocessor.transform(X_fit)
        model = lgb.LGBMRegressor(**parameters, verbosity=-1)
        model.fit(transformed, transformed_target)
    elif family == "xgboost":
        from xgboost import XGBRegressor
        preprocessor = make_xgb_preprocessor(features, 100).fit(X_fit)
        transformed = preprocessor.transform(X_fit)
        model = XGBRegressor(**parameters, device="cpu", verbosity=0)
        model.fit(transformed, transformed_target, verbose=False)
    else:
        raise ValueError(f"Unauthorized cross-fit family: {family}")
    return preprocessor, model


def _atomic_bundle(bundle: ModelBundle, destination: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "bundle.joblib"
    temporary = destination / "bundle.joblib.tmp"
    joblib.dump(bundle, temporary, compress=3)
    reloaded = joblib.load(temporary)
    if not isinstance(reloaded, ModelBundle):
        raise RuntimeError("Cross-fit bundle reload failed.")
    os.replace(temporary, artifact)
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = {"status": "COMPLETE", "artifact": artifact.name, "artifact_sha256": digest, **manifest}
    temp_manifest = destination / "manifest.json.tmp"
    temp_manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_manifest, destination / "manifest.json")
    return payload


def build_global_oof(train: pd.DataFrame, features: list[str], q90_train: float, saved_bundles: dict[str, ModelBundle], model_root: Path, package_versions: dict[str, str], development_sha256: str, validation_row_hash_digest: str, fit_callback: Callable[[str, Callable[[], Any]], Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_a, fold_b, split = deterministic_twofold_split(train["loan_amount_000s"], train["row_hash"], q90_train)
    components = {family: np.full(len(train), np.nan, dtype=np.float64) for family in ("catboost", "lightgbm", "xgboost")}
    fit_records = []
    for train_label, fit_index, predict_index in (("fold_a", fold_a, fold_b), ("fold_b", fold_b, fold_a)):
        for family in ("catboost", "lightgbm", "xgboost"):
            role = f"crossfit_{family}_{train_label}"
            destination = model_root / role
            source_bundle = saved_bundles[family]
            parameters = fixed_crossfit_parameters(source_bundle)
            expected = {"family": family, "fit_membership_digest": membership_digest(train.iloc[fit_index]["row_hash"]), "predict_membership_digest": membership_digest(train.iloc[predict_index]["row_hash"]), "parameters": parameters, "target_mode": source_bundle.target_mode}

            def work():
                manifest_path = destination / "manifest.json"
                if manifest_path.exists() and (destination / "bundle.joblib").exists():
                    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if all(existing.get(k) == v for k, v in expected.items()):
                        return joblib.load(destination / "bundle.joblib"), existing, True
                preprocessor, model = _fit_one_family(family, parameters, features, train.iloc[fit_index][features], train.iloc[fit_index]["loan_amount_000s"].to_numpy(float), source_bundle.target_mode)
                bundle = ModelBundle(model_id=role, family=family, feature_names=features, feature_contract_name="main_without_sensitive_without_lender", target_mode=source_bundle.target_mode, preprocessor=preprocessor, model=model, package_versions=package_versions, model_parameters=parameters, selected_best_iteration=int(parameters.get("iterations", parameters.get("n_estimators"))), development_source_sha256=development_sha256, train_row_hash_digest=ordered_digest(train.iloc[fit_index]["row_hash"]), validation_row_hash_digest=validation_row_hash_digest, metadata={"fit_role": role, "fitted_iterations": int(parameters.get("iterations", parameters.get("n_estimators"))), "zero_self_prediction": True})
                manifest = _atomic_bundle(bundle, destination, {**expected, "feature_contract": features, "seed": SEED, "package_versions": package_versions, "training_row_count": int(len(fit_index)), "prediction_row_count": int(len(predict_index)), "ordered_fit_digest": ordered_digest(train.iloc[fit_index]["row_hash"]), "ordered_predict_digest": ordered_digest(train.iloc[predict_index]["row_hash"])})
                return bundle, manifest, False

            bundle, manifest, reused = fit_callback(role, work)
            prediction = bundle.predict(train.iloc[predict_index][features])
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"Non-finite cross-fit predictions: {role}")
            components[family][predict_index] = prediction
            fit_records.append({"role": role, "family": family, "reused": bool(reused), **{k: manifest[k] for k in ("training_row_count", "prediction_row_count", "fit_membership_digest", "predict_membership_digest")}})
    if any(not np.isfinite(values).all() for values in components.values()):
        raise RuntimeError("Every Train row must have one finite OOF prediction per family.")
    global_oof = 0.60 * components["catboost"] + 0.20 * components["lightgbm"] + 0.20 * components["xgboost"]
    folds = np.full(len(train), "", dtype=object); folds[fold_a] = "fold_a"; folds[fold_b] = "fold_b"
    frame = pd.DataFrame({"row_hash": train["row_hash"].astype(str), "y_true": train["loan_amount_000s"].to_numpy(float), "catboost_oof": components["catboost"], "lightgbm_oof": components["lightgbm"], "xgboost_oof": components["xgboost"], "global_oof_prediction": global_oof, "fold_id": folds})
    report = {"status": "PASS", "split": split, "rows": len(frame), "exactly_one_oof_per_row": True, "zero_self_fit_rows": True, "finite_predictions": True, "ensemble_formula": "0.60*CatBoost + 0.20*LightGBM + 0.20*XGBoost", "fit_records": fit_records}
    return frame, report


@dataclass
class MetaPreprocessor:
    base_features: list[str]
    global_feature: str = GLOBAL_FEATURE

    def fit(self, frame: pd.DataFrame):
        names = self.base_features + [self.global_feature]
        self.transformer_ = CatBoostFramePreprocessor(names).fit(frame[names])
        self.feature_names_in_ = names
        self.cat_feature_indices_ = list(self.transformer_.cat_feature_indices_)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.transformer_.transform(frame[self.feature_names_in_])

    def evidence(self) -> dict[str, Any]:
        return {"feature_names": self.feature_names_in_, "global_feature": self.global_feature, "cat_feature_indices": self.cat_feature_indices_}


@dataclass
class MetaGateBundle:
    preprocessor: MetaPreprocessor
    model: Any
    metadata: dict[str, Any]

    def predict_tail_probability(self, frame: pd.DataFrame) -> np.ndarray:
        probability = np.asarray(self.model.predict_proba(self.preprocessor.transform(frame)), dtype=np.float64)[:, 1]
        if probability.shape != (len(frame),) or not np.isfinite(probability).all():
            raise RuntimeError("Meta-Gate produced invalid probabilities.")
        return probability


def save_experimental_bundle(bundle: Any, destination: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "bundle.joblib"; temporary = destination / "bundle.joblib.tmp"
    joblib.dump(bundle, temporary, compress=3); joblib.load(temporary); os.replace(temporary, artifact)
    payload = {"status": "COMPLETE", "artifact": artifact.name, "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), **metadata}
    temp = destination / "manifest.json.tmp"; temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"); os.replace(temp, destination / "manifest.json")
    return payload


def load_experimental_bundle(destination: Path):
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    artifact = destination / manifest["artifact"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["artifact_sha256"]:
        raise RuntimeError("Experimental bundle hash mismatch.")
    return joblib.load(artifact), manifest
