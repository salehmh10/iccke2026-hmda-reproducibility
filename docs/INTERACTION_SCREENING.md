# Interaction screening

The [98-row candidate dictionary](../results/classification/interaction_candidates_98.csv) preserves exact saved formulas, source pairs, statistics, fallbacks, and keep/reject decisions. Family sizes are 10 numeric-relative, 15 nonlinear, 40 category-context, and 33 onehot-numeric candidates. Base features are not counted among the 98.

Numeric-relative features compare loan/income with area and tract context and include gaps, normalized differences, and shares. Nonlinear features include square roots, log1p, fixed indicators, and five train-median/IQR standardized features. Category-context includes frequency and rare/unseen indicators for seven source fields and two conditional transforms for each of 13 category/numeric pairs. Rare means training support below 500. Missing category keys use `__MISSING__`; unseen frequency is zero. Conditional medians/IQR fall back to global train values; nonfinite median becomes zero and degenerate global IQR becomes one, with numerical denominator guards. Onehot-numeric expands eleven pair templates over training levels.

Seven screening configurations: `all_candidates`, `baseline_v1_equivalent`, `categorical_numeric_combined`, `category_context`, `nonlinear`, `numeric_relative`, `onehot_numeric`. The first screening uses seeds 20260809 and 20260810, three Train-only OOF folds, and LightGBM/logistic anchors. Third-seed confirmation uses 20260811 and checks improvement for both models without additional logistic convergence warnings. The eligible list is empty. `confirm_feature_v2.py` explicitly selects `onehot_numeric` in that case. The retained family must not be called confirmation-validated. MI/redundancy statistics remain supporting screening evidence, not causal effects.

These screening seeds do not establish final-model seed robustness. Validation and Test results are separate from these OOF tables.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/features/advanced.py`
- `HMDA_pipeline_review/project/new new project/scripts/analyze_feature_v2.py`
- `HMDA_pipeline_review/project/new new project/scripts/confirm_feature_v2.py`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/CONFIRMED_FEATURE_CONFIG.json`
