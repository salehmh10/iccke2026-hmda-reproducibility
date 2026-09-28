"""Budget-conscious classical model factories for sparse tabular features."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_CLASSICAL_MODELS = (
    "dummy",
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "catboost",
    "bagging",
    "stacking",
)


def make_classical_model(name: str, seed: int, n_jobs: int = 8) -> BaseEstimator:
    """Construct a deterministic, screening-scale probabilistic classifier."""

    if name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(
            solver="saga", C=1.0, max_iter=500, tol=1e-3, random_state=seed, n_jobs=n_jobs
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=180,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=250,
            learning_rate=0.06,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            n_jobs=n_jobs,
            random_state=seed,
            verbosity=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=250,
            depth=7,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        )
    if name == "bagging":
        base = DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=5, class_weight=None, random_state=seed
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=80,
            max_samples=0.8,
            max_features=0.9,
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "stacking":
        from lightgbm import LGBMClassifier

        bases: list[tuple[str, BaseEstimator]] = [
            (
                "lr",
                LogisticRegression(
                    solver="liblinear", C=0.8, max_iter=300, random_state=seed
                ),
            ),
            (
                "extra",
                ExtraTreesClassifier(
                    n_estimators=100,
                    min_samples_leaf=4,
                    max_features="sqrt",
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            ),
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=140,
                    learning_rate=0.07,
                    num_leaves=25,
                    n_jobs=n_jobs,
                    random_state=seed,
                    verbosity=-1,
                ),
            ),
        ]
        return StackingClassifier(
            estimators=bases,
            final_estimator=LogisticRegression(
                solver="liblinear", C=1.0, max_iter=300, random_state=seed
            ),
            stack_method="predict_proba",
            cv=3,
            n_jobs=n_jobs,
            passthrough=False,
        )
    raise KeyError(f"unsupported classical model: {name}")


def estimator_parameters(estimator: BaseEstimator) -> dict[str, Any]:
    """Return shallow parameters suitable for compact experiment metadata."""

    return estimator.get_params(deep=False)


def denial_probability(estimator: BaseEstimator, features: Any) -> Any:
    """Return probability for class label 1 (Denial)."""

    probabilities = estimator.predict_proba(features)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError(f"model classes do not contain denial label 1: {classes}")
    return probabilities[:, classes.index(1)]
