# Invalid, superseded, and failed artifacts

The unversioned `ml_stacking_oversampled_s20260809_n60000` is invalid because oversampling preceded ordinary Stacking CV, allowing cloned source observations to cross folds. The retained versioned replacement uses source-group-disjoint folds. Only the explicitly listed experiment is labelled invalid; other models are not invalidated by association.

The 73 non-final V1 ledger rows remain labelled historical/superseded/invalid in `results/classification/invalid_and_superseded.csv`. Screening pilots and deterministic repeats do not enter the 84-row final Validation grid. V2 confirmation's empty eligible list is preserved; the fallback is not reinterpreted as successful confirmation. Rejected HPO/feature variants are not promoted to final evidence.

Regression candidate tables preserve actual status and descriptive scope. Technical failures and retries recorded in designs/ledgers do not receive invented metrics. The historical audit's missing-label-code assertion and the older paper evidence map's missing-V2-artifact assertions are superseded by the files now present. Scientific limitations about reused Test and provenance remain.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/reports/INVALID_EXPERIMENTS.json`
- `HMDA_pipeline_review/project/new new project/reports/EXPERIMENT_RESULTS.csv`
- `HMDA_pipeline_review/project/new new project/reports/V1_V2_RETENTION_CLEANUP.md`
- `regresionpart2/regression_v2/outputs/reports/prompt3_attempt_ledger.json`
- `regresionpart2/regression_v2/outputs/reports/prompt4b4_fit_ledger.json`
