from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/HMDA_Feature_V2_Standalone_FA.ipynb"
RESULTS = ROOT / "notebook_outputs/feature_v2_standalone/notebook_validation_results.csv"


def test_standalone_v2_notebook_is_fully_executed_without_local_imports() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert len(payload["cells"]) == 38
    assert len(code_cells) == 19
    assert [cell["execution_count"] for cell in code_cells] == list(range(1, 20))
    assert not any(
        output.get("output_type") == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    for forbidden in ("from src", "from scripts", "import src", "import scripts"):
        assert forbidden not in source
    assert "ALLOW_TEST_ACCESS = False" in source


def test_standalone_v2_notebook_executed_every_model_family_and_variant() -> None:
    with RESULTS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 42
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    assert counts == {"classical": 27, "dl": 12, "hybrid": 3}
    assert {row["variant"] for row in rows} == {
        "original_weighted", "oversampled", "undersampled"
    }
    assert all(row["pr_auc"] and row["mcc"] and row["balanced_accuracy"] for row in rows)
