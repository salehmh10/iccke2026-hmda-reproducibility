# Regression V2 evaluation

Only Global and Primary were evaluated on the same one-time 75,000-row internal IID holdout after freeze. Saved Global MAE is 62.444349310930285 kUSD and Primary MAE is 62.26062600334689 kUSD. Saved Primary-minus-Global MAE is -0.18372330758339217 kUSD. The approximate .29% overall reduction, 1.40% top-decile reduction, 2.69% top-5% reduction, and .20% body increase are arithmetic from saved aggregate values, not new experiments. RMSE also improves.

The saved paired bootstrap uses 500 row resamples with replacement, seed 42, the same resampled positions for both models, fixed original tail masks, and percentile intervals. Its MAE 95% interval is [-0.26298875484432377, -0.1048113958689481] kUSD. It does not retrain models, measure training-seed variability, or account for lender/geographic dependence through cluster resampling. Tail masks are not re-estimated within each resample.

Five of six frozen conditions pass. C2 requires at least 3% top-decile improvement and fails: Primary 191.9663851471891 versus Global 194.68268274850158 kUSD. Body MAE rises from 47.744670572881226 to 47.84246948000145 kUSD, within the .25% permitted worsening. Target ties make reported top-decile/top-5% row counts 7,503 and 3,781 rather than exact fractions. See [six_condition_check.json](../results/regression/six_condition_check.json).

One-time internal IID does not mean external validation. The final Primary was not changed after this mixed result. No fresh IID, predictions, uncertainty, or metrics were generated for this repository.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_overall_metrics.csv`
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_bootstrap.csv`
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_six_condition_check.json`
- `regresionpart2/regression_v2/src/prompt5a_evaluation.py`
