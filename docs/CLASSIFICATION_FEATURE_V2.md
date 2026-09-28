# Classification Feature-V2

Feature-V2 appends 33 selected interactions to the 33-column base, giving 66 encoded columns. Every interaction has the generic form `J(X,C=c) = X * 1[C=c]`; [the exact saved names](../results/classification/feature_v2_interactions_33.csv) are published with their source pair, formula, selection reason, and scope.

There are eleven category/numeric pair templates: loan type with amount and LTI; property type with amount and LTI; loan purpose with amount and LTI; owner occupancy with income and LTI; preapproval with amount; lien status with amount and LTI. Fitted levels expand these templates into 33 features. Names contain the numeric source, category source, readable category slug truncated to 28 characters, and the first six SHA-1 hex characters of the category label. This hash describes a category name, not an applicant identifier.

Unseen categories activate none of their fitted interactions. The code uses X-only training categories; no target is used in interaction construction. These are selected interactions. The confirmation result had no eligible configuration and retained this family through the documented fallback rule.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/features/advanced.py`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/FEATURE_VALIDATION.csv`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/CONFIRMED_FEATURE_CONFIG.json`
