# Explainability scope

Regression V2 explanations are post-IID descriptive aggregates. A deterministic sample takes the 4,000 lowest hashes of `prompt5b_explainability_seed42` plus record hash, without using target/error/sensitive fields for primary selection. No record hashes or local cases are exported.

CatBoost component SHAP is in raw loan-amount space; LightGBM uses native contributions in raw kUSD; XGBoost attributions are in native log1p-target space. Gate SHAP is in native log-odds routing space; residual SHAP is in signed raw residual-correction space. These raw/log/log-odds values cannot be averaged as common-unit effects. Normalized importance ranks provide a descriptive consensus, not a full-system additive decomposition.

The component summary contains weighted component prediction means, explicitly not feature-SHAP contributions. Saved half-sample stability (1,968/2,032 rows) reports top-ten overlap 10 for each component and rank correlations near one; it does not establish training-seed or population stability. These aggregates are in `results/regression/explainability_component_summary.csv` and `explainability_stability.csv`.

No SHAP, permutation importance, ALE, or feature-dependence computation occurred in packaging. No causal determinant, lender rule, intervention effect, or additive explanation of the whole routed Primary is claimed.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/prompt5b_explainability_sample_manifest.json`
- `regresionpart2/regression_v2/outputs/reports/prompt5b_global_component_summary.csv`
- `regresionpart2/regression_v2/outputs/reports/prompt5b_explainability_stability.csv`
- `regresionpart2/regression_v2/outputs/final/FINAL_TECHNICAL_REPORT.md`
