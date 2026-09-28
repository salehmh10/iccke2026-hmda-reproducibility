# Regression

This update packages existing work. No research, notebook, software test or CI workflow was rerun.

## Regression V2 features

The [35-predictor dictionary](../results/regression/feature_dictionary_35.csv) enumerates the exact main contract, with all formulas. It contains 20 retained raw predictors and 15 engineered predictors. Numeric transformations include five log1p variables, applicant/area income, tract-income ratio, housing-stock ratios, and per-1,000-person unit counts. Invalid divisions produce missing values for downstream handling.

`has_co_applicant` is derived from the recorded co-applicant-sex field by testing whether its text contains `No co-applicant`. Direct sex is excluded from the main contract, but information derived from a sensitive field remains. No claim that all sensitive-derived information is absent is justified.

Loan-program grouping uses FHA/VA/FSA/RHS text matching. Applicant-income and tract-income groups use right-closed .5/.8/1.2 cut points (Very low/Low/Moderate/High). Region follows the saved state-to-region mapping, with unknown and Puerto Rico assigned Other. The full mapping is in [feature_contract.yaml](../configs/regression/feature_contract.yaml). Geography remains among V2 predictors; excluding lender identity and explicit demographics does not remove proxies.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/feature_roles.json`
- `regresionpart2/regression_v2/src/feature_engineering.py`

## Regression V2 models

The saved Development base grid contains 13 candidates across Lasso, histogram gradient boosting (HGB), CatBoost, LightGBM, and XGBoost. The saved deep grid contains four RealMLP/FT-Transformer candidates. Exact designs are exported in [the base design](../configs/regression/prompt2_frozen_design.json) and [the deep design](../configs/regression/prompt3_frozen_design.json). Results preserve failed/status fields rather than selecting only favorable rows.

Preprocessing is family-specific: scaled/imputed one-hot features for Lasso; frequency encoding for high-cardinality and ordinal encoding for other HGB categorical fields; native categorical text for CatBoost; Train-fitted frequency/ordinal encoding for LightGBM; encoded XGBoost; dedicated deep preprocessing in the unchanged source modules. Raw and log1p-target variants have separate candidate IDs. Neural architecture and training specifications are preserved in the frozen design and `prompt3_deep_models.py`; saved recovery metadata distinguishes selected epochs, persistence, and technical retries. These are historical configurations, not a newly verified installation recipe.

All 120 advanced candidates from 13 aggregate tables are present, including weak/rejected ensemble, direct-tail, residual, calibration, benefit, beat-probability and imbalance-aware directions. The wide table separates Selection MAE, Audit MAE, full Validation MAE, and their tail MAEs. The companion 360-row table retains all saved metrics and source scopes. Later Audit/full-Validation labels ending in `_descriptive` are preserved. No rejected candidate receives a fabricated IID value.

The frozen Global recipe `ens_boost_cat060` is .6 CatBoost + .2 LightGBM + .2 XGBoost, with each component prediction mapped into raw kUSD before blending. It is a V2 fit, distinct from the similarly weighted historical Stage4L blend. The final refit plan records fixed component counts and target modes and is included in configurations.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/prompt2_frozen_design.json`
- `regresionpart2/regression_v2/outputs/reports/prompt3_frozen_design.json`
- `regresionpart2/regression_v2/outputs/reports/prompt4c_refit_plan.json`
- `regresionpart2/regression_v2/src/preprocessing.py`
- `regresionpart2/regression_v2/src/deep_preprocessing.py`

## Regression V2 routing

Frozen Primary recipe: `stage3_residual_t75_a75`.

```text
G = 0.6*CatBoost + 0.2*LightGBM + 0.2*XGBoost
P = G + s(p)*residual
s(p) = 0                         when p <= 0.75
s(p) = 0.75*(p - 0.75)/0.25      when p > 0.75
```

The residual is signed and uncapped. In particular, p=.875 gives strength .375, not .75. The frozen implementation validates probability bounds; routing never reads the inference target. A lightweight scalar implementation and synthetic tests verify this documentation formula only; no saved model is loaded.

The gate is a CatBoost binary Logloss classifier for the strict Development training-tail event (amount >438 kUSD). The residual expert is CatBoost MAE regression of `y - global_OOF_prediction`, fitted on the strict tail. Both receive the 35 source predictors plus `global_prediction_feature`. Global OOF predictions are generated on the opposite deterministic fold during the historical fit; the final Development refit plan records two 250,000-row folds for six component OOF roles, three full-Development global fits, one gate fit, and one tail-residual fit (11 roles total). OOF residual targets prevent a component from using its own in-sample global prediction for that residual target.

Final five roles and saved fixed parameters:

| Role | Family | Fixed configuration |
|---|---|---|
| prompt4c_full_catboost_500k | catboost | {"loss_function": "MAE", "iterations": 2000, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20, "random_strength": 1, "random_seed": 42, "thread_count": 4} |
| prompt4c_full_lightgbm_500k | lightgbm | {"objective": "regression_l1", "n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 47, "min_child_samples": 50, "reg_lambda": 5, "random_state": 42, "n_jobs": 4} |
| prompt4c_full_xgboost_500k | xgboost | {"objective": "reg:squarederror", "n_estimators": 1995, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 10, "reg_lambda": 5, "subsample": 1.0, "colsample_bytree": 1.0, "tree_method": "hist", "random_state": 42, "n_jobs": 4} |
| prompt4c_meta_gate_500k | catboost_classifier | {"loss_function": "Logloss", "eval_metric": "PRAUC", "iterations": 976, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 10, "random_seed": 42, "thread_count": 4, "verbose": false} |
| prompt4c_residual_specialist_500k | catboost_regressor | {"loss_function": "MAE", "iterations": 791, "depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 20, "random_strength": 1, "random_seed": 42, "thread_count": 4, "verbose": false} |

No residual clipping/cap is applied in the frozen Primary formula. Small negative final predictions remain in aggregate IID diagnostics; they were not silently clipped during packaging. Original paper prose can read as a constant .75 correction; this repository records the exact source implementation and leaves the manuscript untouched.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/src/prompt4c_bundles.py`
- `regresionpart2/regression_v2/src/prompt4b_residual.py`
- `regresionpart2/regression_v2/outputs/reports/prompt4c_refit_plan.json`
- `regresionpart2/regression_v2/outputs/reports/prompt4c_stage3_recipe_reproduction.json`

## Regression V2 evaluation

Only Global and Primary were evaluated on the same one-time 75,000-row internal IID holdout after freeze. Saved Global MAE is 62.444349310930285 kUSD and Primary MAE is 62.26062600334689 kUSD. Saved Primary-minus-Global MAE is -0.18372330758339217 kUSD. The approximate .29% overall reduction, 1.40% top-decile reduction, 2.69% top-5% reduction, and .20% body increase are arithmetic from saved aggregate values, not new experiments. RMSE also improves.

The saved paired bootstrap uses 500 row resamples with replacement, seed 42, the same resampled positions for both models, fixed original tail masks, and percentile intervals. Its MAE 95% interval is [-0.26298875484432377, -0.1048113958689481] kUSD. It does not retrain models, measure training-seed variability, or account for lender/geographic dependence through cluster resampling. Tail masks are not re-estimated within each resample.

Five of six frozen conditions pass. C2 requires at least 3% top-decile improvement and fails: Primary 191.9663851471891 versus Global 194.68268274850158 kUSD. Body MAE rises from 47.744670572881226 to 47.84246948000145 kUSD, within the .25% permitted worsening. Target ties make reported top-decile/top-5% row counts 7,503 and 3,781 rather than exact fractions. See [six_condition_check.json](../results/regression/six_condition_check.json).

One-time internal IID does not mean external validation. The final Primary was not changed after this mixed result. No fresh IID, predictions, uncertainty, or metrics were generated for this repository.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_overall_metrics.csv`
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_bootstrap.csv`
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_six_condition_check.json`
- `regresionpart2/regression_v2/src/prompt5a_evaluation.py`

## Historical regression and post-Test work

The historical official model `stage4l__blend__without_sensitive` has locked-Test MAE 61.511631217701016 kUSD on 99,948 rows. It belongs to the legacy 499,736-row regression sample, not Regression V2. Historical CatBoost/LightGBM/XGBoost development and the final blend are separate from V2 even when nominal blend weights coincide.

Stage5C RealMLP without-sensitive (MAE 62.15974508887792 kUSD) and with-sensitive (61.82111373018834 kUSD) are post-Test extensions, the latter accuracy-only. These values are historical descriptive context, not independent model-selection evidence. They are deliberately not exported into the V2 leaderboard.

Historical Stage7 fairness is descriptive; Stage8 explainability includes recovery and governance restrictions; Stage9 reporting documents the closure. Metadata-only governance recovery or post-Test amendment does not reset consumed Test status. No old models or row-level artifacts are copied. No numerical ranking between historical locked-Test MAE and V2 IID MAE is valid because population, split, features, selection history, and evidence roles differ.

Source evidence (relative to the read-only source collection):
- `regresionpart2/artifacts/results/stage9/reporting/stage9_final_test_comparison.csv`
- `regresionpart2/artifacts/results/stage9/reporting/FINAL_TECHNICAL_REPORT.md`
- `regresionpart2/artifacts/results/stage9/reporting/MODEL_CARD.md`

## Explainability scope

Regression V2 explanations are post-IID descriptive aggregates. A deterministic sample takes the 4,000 lowest hashes of `prompt5b_explainability_seed42` plus record hash, without using target/error/sensitive fields for primary selection. No record hashes or local cases are exported.

CatBoost component SHAP is in raw loan-amount space; LightGBM uses native contributions in raw kUSD; XGBoost attributions are in native log1p-target space. Gate SHAP is in native log-odds routing space; residual SHAP is in signed raw residual-correction space. These raw/log/log-odds values cannot be averaged as common-unit effects. Normalized importance ranks provide a descriptive consensus, not a full-system additive decomposition.

The component summary contains weighted component prediction means, explicitly not feature-SHAP contributions. Saved half-sample stability (1,968/2,032 rows) reports top-ten overlap 10 for each component and rank correlations near one; it does not establish training-seed or population stability. These aggregates are in `results/regression/explainability_component_summary.csv` and `explainability_stability.csv`.

No SHAP, permutation importance, ALE, or feature-dependence computation occurred in packaging. No causal determinant, lender rule, intervention effect, or additive explanation of the whole routed Primary is claimed.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/prompt5b_explainability_sample_manifest.json`
- `regresionpart2/regression_v2/outputs/reports/prompt5b_global_component_summary.csv`
- `regresionpart2/regression_v2/outputs/reports/prompt5b_explainability_stability.csv`
- `regresionpart2/regression_v2/outputs/final/FINAL_TECHNICAL_REPORT.md`

## Complete saved importance and subgroup tables

The [original report directory](../regression_v2/outputs/reports/README.md) contains the complete 35-row Global feature table, 36-row gate table and 36-row residual table. Global columns keep CatBoost/LightGBM raw-kUSD SHAP separate from XGBoost log1p-target SHAP. The weighted normalized rank consensus is a rank summary, not an average of raw attribution scales. Gate SHAP is in routing log-odds; residual SHAP is in kUSD correction space. None is an additive explanation of the whole routed predictor.

The three-row global_component_summary (also exported as explainability_component_summary) contains frozen weights and component prediction means. It is not a feature-importance table.

Subgroup publication retains only rows already marked ELIGIBLE, and group-decile rows already marked DISPLAY. Original suppression rules are n>=200 for groups, n>=50 for tail groups, n>=30 for group-decile cells and n>=500 for intersections. No metric or rank was recalculated. Excluded small-group rows and local case files are not published. Fairness results remain descriptive, non-causal and non-legal.

The base family comparison includes Train-mean and Train-median baselines. The saved lender ablation and separate final deep-anchor record are included where available. Later development Audit comparisons are adaptive/descriptive, not another independent Test.
