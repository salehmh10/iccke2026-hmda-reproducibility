# Classification pipeline generations

| Source generation | Role and limits |
|---|---|
| Original `binary classification part.ipynb` | historical; CatBoost early stopping uses Test; resampled Test and repeated model comparisons; inconsistent holdout populations |
| `binary classification par1t.ipynb` | intermediate notebook; excluded from retained paper grid |
| `binary classification corrected v2.ipynb` / `classification_fixed.py` | intermediate improvements: deduplication and Validation-driven choices; approval-positive AP and repeated Validation roles; not the authoritative denial-positive generation |
| `hmda_binary_pipeline_v2.py` and its tests | development/reference: five-way Train/Validation/calibration/threshold/Test split, lock, separate approval/denial AP; no final paper-authoritative evidence was established for it |
| `new new project` V1 `final_v1_versioned_equal_60k` | retained 42-run denial-positive Validation grid, baseline rollback, three Test finalists |
| `new new project` V2 `feature_v2_onehot_numeric_equal_60k` | retained 42-run denial-positive Validation grid and three Test finalists; reused physical Test |
| Flattened `regresionpart2/classification_v2` | partial reporting copy; generic reports often describe V1; full local retained project has more complete V2 evidence |

The old classification audit's statement that label-construction code was absent is superseded as a source-availability statement: `main/DATA_CLEANING_FOR_14M.ipynb` now supplies that code. Its leakage findings remain relevant. A recovered historical implementation does not prove an immutable execution chain to the exact final extract.

Source evidence (relative to the read-only source collection):
- `binary classification part.ipynb`
- `HMDA_pipeline_review/project/binary classification par1t.ipynb`
- `HMDA_pipeline_review/project/binary classification corrected v2.ipynb`
- `HMDA_pipeline_review/project/classification_fixed.py`
- `HMDA_pipeline_review/project/hmda_binary_pipeline_v2.py`
- `HMDA_pipeline_review/project/HMDA_BINARY_CLASSIFICATION_AUDIT.md`
- `HMDA_pipeline_review/project/new new project/reports/V1_V2_RETENTION_CLEANUP.md`
