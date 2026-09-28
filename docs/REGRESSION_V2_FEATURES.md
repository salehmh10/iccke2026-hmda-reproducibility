# Regression V2 features

The [35-predictor dictionary](../results/regression/feature_dictionary_35.csv) enumerates the exact main contract, with all formulas. It contains 20 retained raw predictors and 15 engineered predictors. Numeric transformations include five log1p variables, applicant/area income, tract-income ratio, housing-stock ratios, and per-1,000-person unit counts. Invalid divisions produce missing values for downstream handling.

`has_co_applicant` is derived from the recorded co-applicant-sex field by testing whether its text contains `No co-applicant`. Direct sex is excluded from the main contract, but information derived from a sensitive field remains. No claim that all sensitive-derived information is absent is justified.

Loan-program grouping uses FHA/VA/FSA/RHS text matching. Applicant-income and tract-income groups use right-closed .5/.8/1.2 cut points (Very low/Low/Moderate/High). Region follows the saved state-to-region mapping, with unknown and Puerto Rico assigned Other. The full mapping is in [feature_contract.yaml](../configs/regression/feature_contract.yaml). Geography remains among V2 predictors; excluding lender identity and explicit demographics does not remove proxies.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/feature_roles.json`
- `regresionpart2/regression_v2/src/feature_engineering.py`
