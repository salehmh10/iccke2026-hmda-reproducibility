# Regression V2 models

The saved Development base grid contains 13 candidates across Lasso, histogram gradient boosting (HGB), CatBoost, LightGBM, and XGBoost. The saved deep grid contains four RealMLP/FT-Transformer candidates. Exact designs are exported in [the base design](../configs/regression/prompt2_frozen_design.json) and [the deep design](../configs/regression/prompt3_frozen_design.json). Results preserve failed/status fields rather than selecting only favorable rows.

Preprocessing is family-specific: scaled/imputed one-hot features for Lasso; frequency encoding for high-cardinality and ordinal encoding for other HGB categorical fields; native categorical text for CatBoost; category-preserving LightGBM; encoded XGBoost; dedicated deep preprocessing in the preserved reference modules. Raw and log1p-target variants have separate candidate IDs. Neural architecture and training specifications are preserved in the frozen design and `prompt3_deep_models.py`; saved recovery metadata distinguishes selected epochs, persistence, and technical retries. These are historical configurations, not a newly verified installation recipe.

All 120 advanced candidates from 13 aggregate tables are present, including weak/rejected ensemble, direct-tail, residual, calibration, benefit, beat-probability and imbalance-aware directions. The wide table separates Selection MAE, Audit MAE, full Validation MAE, and their tail MAEs. The companion 360-row table retains all saved metrics and source scopes. Later Audit/full-Validation labels ending in `_descriptive` are preserved. No rejected candidate receives a fabricated IID value.

The frozen Global recipe `ens_boost_cat060` is .6 CatBoost + .2 LightGBM + .2 XGBoost, with each component prediction mapped into raw kUSD before blending. It is a V2 fit, distinct from the similarly weighted historical Stage4L blend. The final refit plan records fixed component counts and target modes and is included in configurations.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/prompt2_frozen_design.json`
- `regresionpart2/regression_v2/outputs/reports/prompt3_frozen_design.json`
- `regresionpart2/regression_v2/outputs/reports/prompt4c_refit_plan.json`
- `regresionpart2/regression_v2/src/preprocessing.py`
- `regresionpart2/regression_v2/src/deep_preprocessing.py`
