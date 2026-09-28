# Calibration, thresholds, and hybrids

Within each strategy, hybrid selection uses Validation probabilities to choose classical/neural bases and soft-blend weights. The selected undersampled hybrid combines CatBoost and Modern Hopfield: V1 weights .85/.15; V2 weights .775/.225. All strategy weights and source IDs are in [hybrid_weights.csv](../results/classification/hybrid_weights.csv). These are not fixed across the V1/V2 comparison.

Final evaluation divides Validation into 49,974 calibration-fit and 49,974 threshold-selection rows. Candidate calibration methods are none, sigmoid, and isotonic. The objective is Brier score + .1 * log loss; all six saved finalists select isotonic. The fitted calibration functions differ across runs even when their method names match. Validation is reused across earlier model/weight selection and later calibration roles, so the full selection process is not independent of those holdouts.

Threshold search considers .5 plus 199 probability quantiles from .005 to .995, excluding 0 and 1. It maximizes MCC + .001 * balanced accuracy. Final thresholds in [thresholds.csv](../results/classification/thresholds.csv) are calibrated thresholds; hybrid ledger thresholds are earlier uncalibrated Validation values.

The four V1/V2 CatBoost/Hybrid cases are `descriptive_partially_confounded_comparison`: hybrid weights, fitted calibrators, and thresholds differ; exact capped training-membership evidence is asymmetric; and physical Test had already been used. This is not a controlled causal ablation or fresh holdout validation.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/scripts/train_hybrid.py`
- `HMDA_pipeline_review/project/new new project/scripts/train_feature_v2_hybrid.py`
- `HMDA_pipeline_review/project/new new project/scripts/evaluate_all.py`
- `HMDA_pipeline_review/project/new new project/scripts/evaluate_feature_v2.py`
- `HMDA_pipeline_review/project/new new project/src/evaluation/metrics.py`
