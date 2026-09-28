# Classification Feature-V1

Feature-V1 contains nine numeric encoded outputs plus 24 category indicators, for 33 columns. The ordered [dictionary](../results/classification/feature_v1_dictionary.csv) enumerates all 33 names. Numeric formulas and source fields are explicit. Categorical source fields are agency, loan type, property type, loan purpose, owner occupancy, preapproval, and lien status.

The dictionary recovers complete encoded labels from saved standalone smoke metadata, not by deserializing a production encoder. Therefore the observed smoke vocabulary is complete, while exact production vocabulary identity is unresolved. Positions reconstruct the declared numeric order and alphabetic OneHotEncoder order; they are not a recovered production serialized feature list.

Names use `numeric__<field>` or `categorical__<source>_<level>`. Invalid negative numeric inputs become missing; ratios require positive denominators. Explicit zero flags distinguish observed zero from invalid/missing values. Numeric median imputation (with train-observed missingness indicators) and StandardScaler are train-fitted. Categoricals use mode imputation and OneHotEncoder with unknown categories ignored. `Not applicable` remains a domain level. Extra missingness columns are possible in a different dataset, so 33 is the saved extract's contract, not a universal dimensionality guarantee.

The raw target, direct demographic fields, respondent ID, minority-population field, and fine-grained geography are dropped in this classification contract. Loan-to-income is an LTI proxy, not debt-to-income. Socioeconomic proxies remain.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/features/financial.py`
- `HMDA_pipeline_review/project/new new project/src/features/preprocessing.py`
- `HMDA_pipeline_review/project/new new project/notebook_outputs/feature_v2_standalone/mutual_information_smoke.csv`
- `HMDA_pipeline_review/project/new new project/reports/FEATURE_REPORT.md`
