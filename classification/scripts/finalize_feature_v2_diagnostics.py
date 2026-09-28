"""Regenerate final train-only MI/disposition tables for the confirmed V2 set."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_feature_v2 import (
    _candidate_validation_table,
    _mutual_information_table,
)
from src.data.loader import load_and_prepare
from src.data.schema import ANALYTICAL_TARGET, RAW_TARGET
from src.data.splitting import load_split_manifest
from src.data.variants import load_variant
from src.features.advanced import FEATURE_FAMILIES, AdvancedFinancialFeatureEngineer
from src.training.context import stratified_variant_sample


def main() -> int:
    root = PROJECT_ROOT
    output_dir = root / "reports" / "generations" / "feature_v2"
    confirmed_path = output_dir / "CONFIRMED_FEATURE_CONFIG.json"
    confirmed = json.loads(confirmed_path.read_text(encoding="utf-8"))
    if confirmed.get("test_rows_used") is not False:
        raise RuntimeError("confirmed V2 configuration is not test-blind")
    selected_families = tuple(confirmed["selected_families"])

    prepared = load_and_prepare(root / "hmda_classification_stratified_500k.csv")
    splits = load_split_manifest(root / "data" / "manifests")
    variant = load_variant("original_weighted", root / "data" / "manifests", splits)
    indices, _ = stratified_variant_sample(
        prepared.modeling, variant, max_rows=60_000, seed=20260809
    )
    if not set(indices).issubset(set(splits.train)):
        raise RuntimeError("feature diagnostics attempted to use non-training rows")
    frame = prepared.modeling.loc[indices].drop(columns=[RAW_TARGET, ANALYTICAL_TARGET])
    target = prepared.modeling.loc[indices, ANALYTICAL_TARGET].to_numpy(dtype=np.int8)

    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=20260809)
    _, mi_positions = next(splitter.split(frame, target))
    mi_frame = frame.iloc[mi_positions]
    mi_target = target[mi_positions]
    engineer = AdvancedFinancialFeatureEngineer(families=FEATURE_FAMILIES).fit(mi_frame)
    engineered = engineer.transform(mi_frame)
    mi_inputs = engineered.copy()
    mi_inputs["hud_median_family_income"] = mi_frame[
        "hud_median_family_income"
    ].astype("float64")
    mi_table = _mutual_information_table(mi_inputs, mi_target, seed=20260809)
    validation = _candidate_validation_table(
        engineered, mi_table, selected_families
    )

    for name in ("FEATURE_MI.csv", "FEATURE_VALIDATION.csv"):
        current = output_dir / name
        archive = output_dir / name.replace(".csv", "_EXPLORATORY.csv")
        if current.exists() and not archive.exists():
            shutil.copy2(current, archive)
    mi_table.to_csv(output_dir / "FEATURE_MI.csv", index=False)
    validation.to_csv(output_dir / "FEATURE_VALIDATION.csv", index=False)

    selected = validation[validation["selected"]]
    rejected = validation[~validation["selected"]]
    payload = {
        "selection_source": confirmed_path.relative_to(root).as_posix(),
        "selected_configuration": confirmed["selected_configuration"],
        "selected_families": list(selected_families),
        "encoded_features": int(confirmed["encoded_features"]),
        "candidate_features_selected": int(len(selected)),
        "candidate_features_rejected": int(len(rejected)),
        "mi_rows": int(len(mi_frame)),
        "mi_source": "original-weighted training split only",
        "category_statistics": "X-only and fitted only on the MI training sample",
        "test_rows_used": False,
    }
    (output_dir / "FINAL_FEATURE_SELECTION.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
