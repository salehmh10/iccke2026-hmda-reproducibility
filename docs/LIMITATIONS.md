# Limitations and excluded files

This update packages existing work. No research, notebook, software test or CI workflow was rerun.

## Unresolved evidence and manuscript discrepancies

1. Historical classification cleaning code is recovered, but a complete immutable execution/hash chain through every intermediate to the supplied extract is unresolved. National sampling representativeness is not established.
2. All 33 V1 encoded names are available from saved standalone smoke metadata; exact production categorical vocabulary and its column order have not been independently verified without serialized objects. The dictionary labels this distinction.
3. Exact V1/V2 capped training membership is asymmetric; no membership reconstruction is authorized. Hybrid weights, calibrated functions and thresholds differ, so the comparison is partially confounded.
4. TabNet's explicitly saved settings and version are available; unrecorded library defaults remain unresolved. Historical environments are not demonstrated as a clean, fully locked rebuild.
5. Manuscript prose calls the interactions validated; saved confirmation has zero eligible configurations and selects a fallback. Repository wording uses selected interactions. The paper is untouched.
6. Manuscript routing prose is simplified; the frozen implementation uses a probability-dependent ramp, not a constant .75 residual weight. Repository formula follows source. The paper is untouched.
7. Direct sensitive fields are omitted from the main Regression contract, but `has_co_applicant` derives from co-applicant sex and geographic proxies remain.
8. Paper code 1059/acceptance status is owner-supplied administrative context. A local acceptance letter was not identified.
9. No new multi-seed, clustered-bootstrap, temporal, lender-held-out, external-source, causal, or legal evidence was produced. Classification Test reuse and Regression C2 failure remain scientific limitations, not packaging defects.
10. No project license was found; author selection/permission is required before public publication. Data redistribution and code ownership require author review.

## Regression V2 limitations

The overall MAE effect is small, C2 fails, body MAE worsens, and the largest loans retain substantial underprediction. Reported amounts need not be disbursed funds. Complete-case filtering and favorable-action conditioning limit population interpretation. Legacy exclusion improves separation from the old sample but does not create an external source.

Development was adaptive and several later Audit/full-Validation tables are descriptive. There is no new multi-seed, temporal, lender-held-out, geographic-held-out, or external evaluation. Paired row-bootstrap uncertainty conditions on fixed fitted models and ignores clustered dependence. Predictor proxies remain, including sensitive-derived co-applicant information. No production or lending-decision suitability is established.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/final/FINAL_TECHNICAL_REPORT.md`
- `regresionpart2/regression_v2/outputs/final/MODEL_CARD_FINAL.md`
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_six_condition_check.json`

## Feature interpretation and fairness limitations

Importance reflects dependence of fitted components in declared native spaces, conditional on included features and sample. Correlated predictors and transformed copies can share importance. Large attribution is not evidence that changing a feature would change an outcome. A gate explanation concerns routing, while a residual explanation concerns the proposed correction; neither alone explains the complete Primary.

Historical fairness and Regression V2 fairness are distinct post-Test/post-IID descriptive analyses. Protected fields can be audit-only yet proxies remain; V2 co-applicant presence is sensitive-derived. Subgroup errors mix target-distribution and model-performance differences. No discrimination proof, legal finding, compliance certification, or causal fairness claim is supported. Individual sensitive rows and small/local-case exports are excluded.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/final/MODEL_CARD_FINAL.md`
- `regresionpart2/regression_v2/outputs/reports/prompt5b_sensitive_contract.json`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/FAIRNESS_AUDIT.md`

## Invalid, superseded, and failed artifacts

The unversioned `ml_stacking_oversampled_s20260809_n60000` is invalid because oversampling preceded ordinary Stacking CV, allowing cloned source observations to cross folds. The retained versioned replacement uses source-group-disjoint folds. Only the explicitly listed experiment is labelled invalid; other models are not invalidated by association.

The 73 non-final V1 ledger rows remain labelled historical/superseded/invalid in `results/classification/invalid_and_superseded.csv`. Screening pilots and deterministic repeats do not enter the 84-row final Validation grid. V2 confirmation's empty eligible list is preserved; the fallback is not reinterpreted as successful confirmation. Rejected HPO/feature variants are not promoted to final evidence.

Regression candidate tables preserve actual status and descriptive scope. Technical failures and retries recorded in designs/ledgers do not receive invented metrics. The historical audit's missing-label-code assertion and the older paper evidence map's missing-V2-artifact assertions are superseded by the files now present. Scientific limitations about reused Test and provenance remain.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/reports/INVALID_EXPERIMENTS.json`
- `HMDA_pipeline_review/project/new new project/reports/EXPERIMENT_RESULTS.csv`
- `HMDA_pipeline_review/project/new new project/reports/V1_V2_RETENTION_CLEANUP.md`
- `regresionpart2/regression_v2/outputs/reports/prompt3_attempt_ledger.json`
- `regresionpart2/regression_v2/outputs/reports/prompt4b4_fit_ledger.json`

## Publication exclusions and portability

The original data-cleaning notebook has personal filesystem paths in code, so its earlier code-redacted publication copy was removed. The historical Stage3 tree notebook has the same blocker. No replacement notebook was created. Five historical helpers also contain personal paths in code and are excluded. The [exclusion table](../data/manifests/publication_exclusions.csv) gives source filenames and reasons without publishing private paths.

Duplicate notebook copies, backups, the redundant historical visual-summary notebook, and the newer five-way-split classifier are not promoted as manuscript Feature-V2. Only the retained generation supplies paper results. Missing historical helpers and protected runtime dependencies limit portability. Code and Markdown were not rewritten to hide these limits.

The two authors must resolve ownership and licensing before any public release. Repository visibility stays private and releases stay draft. No new DOI, publication record or acceptance letter was created. Historical PASS outputs retain their original scope; this update makes no software-test or fresh reproduction claim.
