# Classification metric orientation

Paper comparisons use **denial-positive Average Precision (AP)**. The saved column `pr_auc` calls sklearn `average_precision_score(y_denial, p_denial)`. AP sums precision weighted by recall increments; it is not trapezoidal integration of the precision-recall curve. `target_denied = 1 - loan_approved`; classifier probability for class label 1 is denial probability.

MCC, balanced accuracy, denial recall, F1 and confusion counts refer to this same orientation. Balanced accuracy averages approval and denial recall. Thresholds operate on denial probability. Test table metrics are after Validation-derived calibration/threshold choice; raw Validation leaderboard metrics and calibrated Test metrics have different evidence roles.

The intermediate `classification_fixed.py` uses `loan_approved=1` and reports approval-positive AP. The five-way development/reference pipeline reports approval and denial AP separately. Historical or intermediate `pr_auc` columns are never silently promoted to denial-positive AP.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/evaluation/metrics.py`
- `HMDA_pipeline_review/project/new new project/src/models/classical.py`
- `HMDA_pipeline_review/project/classification_fixed.py`
- `HMDA_pipeline_review/project/hmda_binary_pipeline_v2.py`
