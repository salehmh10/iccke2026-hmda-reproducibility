# Regression V2 routing

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
