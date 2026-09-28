# Populations and targets

| Generation | Population and target | Evidence role |
|---|---|---|
| Historical classification | supplied favorable/denial sample; legacy protocols | historical only |
| Classification V1/V2 | transformed binary snapshot, positive `target_denied=1` | Validation model choice; reused final Test descriptive |
| Historical Stage4L | favorable-action legacy sample; `loan_amount_000s` | historical locked Test |
| Regression V2 | legacy-excluded favorable actions 1/2/8; `loan_amount_000s` | Development selection and one-time internal IID |

Regression target units are thousands of US dollars (kUSD). The target is reported loan amount among favorable-action records, not necessarily disbursed funds. It is not default, interest rate, pricing, profit, or loss. A difference in MAE is a difference in prediction error and must not be described as monetary savings.

Missingness filters and historical sampling restrict generalization. No new national-representativeness, causal, legal, automated-lending, or production-readiness claim follows from the package.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/reports/DATASET_REPORT.md`
- `regresionpart2/regression_v2/src/data_cleaning.py`
- `regresionpart2/regression_v2/src/feature_engineering.py`
