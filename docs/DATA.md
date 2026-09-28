# Data and targets

This update packages existing work. No research, notebook, software test or CI workflow was rerun.

## Populations and targets

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

## Action-taken mapping

| HMDA action | Binary classification | Regression V2 |
|---|---|---|
| 1: loan originated | favorable (`loan_approved=1`) | retained |
| 2: approved but not accepted | favorable (`loan_approved=1`) | retained |
| 3: application denied | denial (`loan_approved=0`) | excluded |
| 7: preapproval request denied | denial (`loan_approved=0`) | excluded |
| 8: preapproval request approved but not accepted | favorable (`loan_approved=1`) | retained |
| 4/5/6 and other codes | excluded from binary population | excluded |

The historical notebook explicitly maps 1/2/8 to one and 3/7 to zero. The later classification adapter uses `target_denied = 1 - loan_approved`, so positive class 1 means denial. Favorable action does not imply loan origination, customer acceptance, or disbursement. In particular, codes 2 and 8 are not accepted/originated loans.

Source evidence (relative to the read-only source collection):
- `main/DATA_CLEANING_FOR_14M.ipynb`
- `regresionpart2/regression_v2/src/data_cleaning.py`
- `HMDA_pipeline_review/project/new new project/src/data/schema.py`

## Data provenance

The historical cleaning notebook is now available. It names the nationwide labelled HMDA 2017 file and documents the binary action mapping and historical sampler. This corrects earlier claims that no label-construction implementation was available. A saved source report for Regression V2 records 14,285,496 raw rows and 78 original columns. The historical classification notebook records the same stated source; a complete immutable chain from that historical notebook run to the retained classification extract is not proven here.

Classification uses the supplied transformed 500,000-row, 29-column extract. Its recorded file SHA-256 is `b9c8373a9234f0a6083d0b92e87574f37c0c2be858812b2e1cbde66ae5af63fe`. Later exact full-row deduplication removes 260 excess copies before splitting, leaving 499,740 rows. The original class counts are 401,226 favorable and 98,774 denial records. Source code recovery is stronger provenance evidence than the old flattened audit, but does not establish national representativeness or authenticate every historical intermediate file.

Regression V2 independently documents the labelled nationwide 2017 source: recorded SHA-256 `dd35f6a877c5882bbe7260ce65ca842b18ca6aa16514cba256feeeb316d4b7c3`, size 11,237,068,086 bytes. This is a saved identity, not a new hash of raw data. It retains favorable-action records, removes exact duplicates, contradictory-target groups, and overlap with the legacy regression sample before selecting 575,000 records. Raw files and individual membership remain excluded.

See [filters](DATA.md), [sampling](DATA.md), and [target definitions](DATA.md).

Source evidence (relative to the read-only source collection):
- `main/DATA_CLEANING_FOR_14M.ipynb`
- `HMDA_pipeline_review/project/new new project/reports/DATASET_REPORT.md`
- `regresionpart2/regression_v2/config.json`
- `regresionpart2/regression_v2/outputs/reports/prompt1a_source_report.json`

## Data filters

The historical notebook drops 27 administrative, detailed-race, denial-reason, and rate-spread columns from the 78-column source, then removes missing-like values in required fields. It retains only classification actions 1/2/3/7/8 and creates the binary label. Its hash-based deduplication branch writes `hmda_2017_no_duplicates.csv`, but subsequent validity filtering reads `hmda_2017_classification_ready.csv`; that branch did not feed the final sampler. The later V1/V2 project therefore removes exact duplicates again before splitting.

Regression V2 applies these fixed required fields:

`applicant_income_000s`, `msamd_name`, `msamd`, `tract_to_msamd_income`, `number_of_owner_occupied_units`, `number_of_1_to_4_family_units`, `census_tract_number`, `population`, `minority_population`, `hud_median_family_income`, `county_name`, `county_code`, `state_name`, `state_abbr`, `state_code`, `loan_amount_000s`.

After trimming and lowercasing for missing checks, tokens are: ``, `-`, `--`, `.`, `n/a`, `na`, `nan`, `none`, `null`, including the empty string. Numeric fields are parsed and must be finite. Loan amount, income, population, HUD median, tract-income ratio, and owner/family-unit counts must be positive; minority population must be in [0,100]. Present category codes must match the fixed allowed sets in the static cleaning source. These rules select a restricted complete-case population.

The saved ordered cleaning report records 6,500,536 action exclusions, 1,305,782 required-missing exclusions, and 7,548 invalid-range exclusions, leaving 6,471,630 records. Exact comparison over 30 physical fields removes 6,442 duplicate copies (5,790 groups), leaving 6,465,188. All 56,675 records in 27,691 contradictory-target groups are removed, followed by 495,750 records overlapping legacy keys. The remaining eligible population is 5,912,763.

Legacy matching trims strings and canonicalizes numeric values to ten decimal places; it matches 499,736 unique legacy keys and conservatively excludes all matching V2 records after conflict removal. Hash collisions alone never decide exact equality. Aggregate reports are included under `data/evidence/`; no keys or rows are included.

Source evidence (relative to the read-only source collection):
- `main/DATA_CLEANING_FOR_14M.ipynb`
- `regresionpart2/regression_v2/src/data_cleaning.py`
- `regresionpart2/regression_v2/outputs/reports/prompt1a_cleaning_report.csv`
- `regresionpart2/regression_v2/outputs/reports/prompt1b_deduplication_report.json`
- `regresionpart2/regression_v2/outputs/reports/prompt1b_conflict_report.json`
- `regresionpart2/regression_v2/outputs/reports/prompt1b_legacy_exclusion_report.json`

## Sampling and split roles

Classification historical sampling first counts the binary class population. It rounds class-proportional quotas to a total of 500,000 and assigns a rounding difference to the largest class. For each 200,000-row input chunk, it appends that chunk's records to the retained pool for each class. Whenever the pool exceeds its quota, it randomly caps the combined pool with seed `42 + chunk_id + class_label`. It finally shuffles with seed 42. This is a chunk-wise capped sampler, not uniform sampling from the whole population, reservoir sampling, or hash sampling. Inclusion can depend on source order and chunk position. The analogous historical favorable-only regression sampler is not Regression V2.

The V1/V2 transformed snapshot is deduplicated and stratified with seed 20260809 into Train 299,844, Validation 99,948, Test 99,948. Saved class counts are in `results/classification/split_summary.csv`. Train-only resampling produces weighted-original, oversampled, and undersampled variants. Each final family/strategy run has a 60,000-row cap. Physical Test was reused across generations and repair checks.

Regression V2 ranks eligible records by SHA-256 of `regression_v2_seed_42` concatenated with the target-excluding canonical record hash, with a deterministic hash tie-breaker; it retains the lowest 575,000. Target-decile-stratified shuffle splits (seed 42, duplicate-safe qcut bins) then reserve 75,000 IID rows and split 500,000 Development into 400,000 Train and 100,000 Validation. Validation later has 70,000 Selection and 30,000 Audit roles. Selection was adaptive; later Audit and full Validation rows are explicitly descriptive in the saved scope fields. Final model refits use Development only, before a one-time paired IID evaluation.

No membership was recreated or published. Internal IID is not an external, temporal, lender-held-out, or geographic transfer evaluation.

Source evidence (relative to the read-only source collection):
- `main/DATA_CLEANING_FOR_14M.ipynb`
- `HMDA_pipeline_review/project/new new project/data/manifests/split_manifest.json`
- `regresionpart2/regression_v2/src/final_dataset_builder.py`
- `regresionpart2/regression_v2/config.json`
