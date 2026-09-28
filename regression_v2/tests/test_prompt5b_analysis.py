from __future__ import annotations

import ast
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prompt5b_analysis import (  # noqa: E402
    _anonymous_id,
    _composition,
    _eligible_status,
    _label_series,
    _metrics,
    _missing_unknown,
    _xgb_feature_mapping,
)


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [100.0, 200.0, 300.0, 400.0],
            "primary_prediction": [90.0, 220.0, 270.0, 440.0],
            "global_prediction": [80.0, 230.0, 260.0, 450.0],
            "iid_local_decile": ["D1", "D4", "D10", "D10"],
            "iid_top5_target": [False, False, False, True],
        }
    )


def test_group_metric_formulas() -> None:
    frame = synthetic_frame()
    result = _metrics(frame, "primary")
    signed = np.array([-10.0, 20.0, -30.0, 40.0])
    assert result["mae"] == 25.0
    assert np.isclose(result["rmse"], np.sqrt(np.mean(signed**2)))
    assert result["mean_signed_error"] == 5.0
    assert result["underprediction_rate"] == 0.5
    assert np.isclose(result["wape_percent"], 10.0)


def test_target_composition_is_reported() -> None:
    result = _composition(synthetic_frame())
    assert result == {
        "mean_target": 250.0,
        "median_target": 250.0,
        "d10_fraction": 0.5,
        "top5_target_fraction": 0.25,
    }


def test_small_group_threshold_is_frozen() -> None:
    assert _eligible_status(199) == "SMALL_GROUP"
    assert _eligible_status(200) == "ELIGIBLE"


def test_missing_and_unknown_labels() -> None:
    labels = _label_series(pd.Series([None, "Unknown", "No co-applicant", "Observed"]))
    assert labels.tolist()[0] == "__MISSING__"
    assert _missing_unknown(labels.iloc[0]) == "MISSING"
    assert _missing_unknown(labels.iloc[1]) == "UNKNOWN_OR_NOT_PROVIDED"
    assert _missing_unknown(labels.iloc[2]) == "NO_CO_APPLICANT"
    assert _missing_unknown(labels.iloc[3]) == "OBSERVED"


def test_anonymous_case_id_is_deterministic_and_bounded() -> None:
    first = _anonymous_id("abc")
    assert first == _anonymous_id("abc")
    assert first != _anonymous_id("abd")
    assert len(first) == 12


def test_xgboost_mapping_aggregates_one_hot_levels() -> None:
    class OneHot:
        categories_ = [np.array(["a", "b"]), np.array(["x", "y", "z"])]

    class Preprocessor:
        numeric_features_ = ["income"]
        high_cardinality_features_ = ["county"]
        low_cardinality_features_ = ["purpose", "region"]
        one_hot_ = OneHot()

    assert _xgb_feature_mapping(Preprocessor()) == [
        "income", "county", "purpose", "purpose", "region", "region", "region"
    ]


def test_prompt5b_source_contains_no_training_method_calls() -> None:
    tree = ast.parse((ROOT / "src" / "prompt5b_analysis.py").read_text(encoding="utf-8"))
    prohibited = {"fit", "fit_transform", "partial_fit"}
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in prohibited
    ]
    assert calls == []


def test_notebook_code_is_artifact_only_when_present() -> None:
    path = ROOT / "notebooks" / "05B_FAIRNESS_AND_FINAL_EXPLAINABILITY.ipynb"
    if not path.exists():
        return
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for token in (
        "iid_holdout_features",
        "iid_holdout_targets",
        ".fit(",
        "joblib.load",
        "get_feature_importance",
        "pred_contrib",
    ):
        assert token not in source

