# Data filters

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
