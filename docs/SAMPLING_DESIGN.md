# Sampling and split roles

Classification historical sampling first counts the binary class population. It rounds class-proportional quotas to a total of 500,000 and assigns a rounding difference to the largest class. For each 200,000-row input chunk, it appends that chunk's records to the retained pool for each class. Whenever the pool exceeds its quota, it randomly caps the combined pool with seed `42 + chunk_id + class_label`. It finally shuffles with seed 42. This is a chunk-wise capped sampler, not uniform sampling from the whole population, reservoir sampling, or hash sampling. Inclusion can depend on source order and chunk position. The analogous historical favorable-only regression sampler is not Regression V2.

The V1/V2 transformed snapshot is deduplicated and stratified with seed 20260809 into Train 299,844, Validation 99,948, Test 99,948. Saved class counts are in `results/classification/split_summary.csv`. Train-only resampling produces weighted-original, oversampled, and undersampled variants. Each final family/strategy run has a 60,000-row cap. Physical Test was reused across generations and repair checks.

Regression V2 ranks eligible records by SHA-256 of `regression_v2_seed_42` concatenated with the target-excluding canonical record hash, with a deterministic hash tie-breaker; it retains the lowest 575,000. Target-decile-stratified shuffle splits (seed 42, duplicate-safe qcut bins) then reserve 75,000 IID rows and split 500,000 Development into 400,000 Train and 100,000 Validation. Validation later has 70,000 Selection and 30,000 Audit roles. Selection was adaptive; later Audit and full Validation rows are explicitly descriptive in the saved scope fields. Final model refits use Development only, before a one-time paired IID evaluation.

No membership was recreated or published. Internal IID is not an external, temporal, lender-held-out, or geographic transfer evaluation.

Source evidence (relative to the read-only source collection):
- `main/DATA_CLEANING_FOR_14M.ipynb`
- `HMDA_pipeline_review/project/new new project/data/manifests/split_manifest.json`
- `regresionpart2/regression_v2/src/final_dataset_builder.py`
- `regresionpart2/regression_v2/config.json`
