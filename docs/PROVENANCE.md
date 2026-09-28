# Evidence sources and publication copies

This update packages existing work. No research, notebook, software test or CI workflow was rerun.

## Manuscript identity and untouched boundary

The root PDF `ICCKE2026_Boosting_vs_Deep_Tabular_HMDA.pdf` is byte-identical to `paper_revision/main_FINAL_6PAGES_TARGETED_EDITS.pdf`. Its matching editable source is `paper_revision/main_FINAL_6PAGES_TARGETED_EDITS.tex`. Static text inspection confirms the exact title, author order, affiliation, classification AP values 0.4606/0.4621, and Regression V2 MAE values 62.444/62.261 kUSD. The older `paper_revision/main.tex` is a different revision and was not used to choose the accepted version.

Title: Boosting and Deep Tabular Learning on HMDA: Feature-Interaction Classification and Tail-Aware Loan-Amount Regression

Authors: Saleh Mohammad Hasani and Reza Kazemeiny Moghaddam. Affiliation: Department of Electrical Engineering, Sharif University of Technology, Tehran, Iran. The equal-contribution note and corresponding-author order remain as recorded in the manuscript.

ICCKE 2026 and paper code 1059 are administrative context supplied by the repository owner. A local acceptance letter was not identified, so this package does not independently certify acceptance.

The paper, editable sources, figures, bibliography, author metadata, and ZIPs are excluded from the repository and remain unchanged. Related Work change 13 is excluded. Documentation records implementation discrepancies without changing manuscript prose: interaction selection used a fallback; AP uses average precision; routing strength depends on gate probability; `has_co_applicant` derives from a sensitive source field.

Source evidence (relative to the read-only source collection):
- `ICCKE2026_Boosting_vs_Deep_Tabular_HMDA.pdf`
- `paper_revision/main_FINAL_6PAGES_TARGETED_EDITS.pdf`
- `paper_revision/main_FINAL_6PAGES_TARGETED_EDITS.tex`

## Classification evidence hierarchy

Use the retained full project under `HMDA_pipeline_review/project/new new project`. V1 lock `artifacts/generations/baseline_v1/FINAL_GENERATION.lock.json` identifies `final_v1_versioned_equal_60k`. V2 lock identifies `feature_v2_onehot_numeric_equal_60k`. Generation-specific final Test and calibration tables control over generic report titles. V1 final Validation rows are the 42 version-suffixed equal-60k records in the 115-row ledger; V2's retained ledger has 42 rows. All 14 families and three strategies occur once per representation.

Prior flattened reports can omit these artifacts and cannot override the recovered full project. Lock existence establishes a declared freeze; it does not prove an unused physical Test or erase earlier test access. Exact source hashes for every exported table are recorded, but model bytes and row membership were not reloaded.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/artifacts/generations/baseline_v1/FINAL_GENERATION.lock.json`
- `HMDA_pipeline_review/project/new new project/artifacts/generations/feature_v2/FINAL_GENERATION.lock.json`
- `HMDA_pipeline_review/project/new new project/reports/EXPERIMENT_RESULTS.csv`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/EXPERIMENT_RESULTS.csv`

## Regression V2 evidence hierarchy

Use `regresionpart2/regression_v2`, its immutable final freeze/evaluation/completion metadata, and saved aggregate reports. `FINAL_PRE_IID_FREEZE.json` fixes Primary `stage3_residual_t75_a75` and comparator `ens_boost_cat060`; `FINAL_IID_EVALUATION.json` fixes one-time IID evidence; `FINAL_PROJECT_COMPLETE.json` closes reporting. Saved Prompt5A metrics and six-condition JSON carry the exact evaluation values. Final figure/table derivatives are supporting corroboration.

Prompt2/3 Development candidates and all Prompt4 candidates retain their own selection scopes. Prompt5B explanations/fairness are post-IID descriptive. Historical Stage4L and post-Test deep extensions are separate sources and never ranked with V2 IID results.

Source evidence (relative to the read-only source collection):
- `regresionpart2/regression_v2/outputs/reports/FINAL_PRE_IID_FREEZE.json`
- `regresionpart2/regression_v2/outputs/reports/FINAL_IID_EVALUATION.json`
- `regresionpart2/regression_v2/outputs/reports/FINAL_PROJECT_COMPLETE.json`
- `regresionpart2/regression_v2/outputs/reports/prompt5a_iid_overall_metrics.csv`

## Experiment lineage

The [machine-readable lineage](../data/manifests/experiment_lineage.csv) separates populations, tasks, generations, configurations, implementations, result artifacts, and limitations. Start with [manuscript identity](PROVENANCE.md), then [classification evidence](PROVENANCE.md) and [regression evidence](PROVENANCE.md).

Authoritative manuscript result is an evidence role, not a guarantee of fresh holdout or causal interpretation. A final classification Test result can be paper-authoritative and still reused/descriptive. Regression V2's independently reserved internal IID is separate from adaptive Development. Historical, post-Test, invalid, failed-technical, and not-executed roles never inherit final status from their filename.

## Result traceability

Every CSV under `results/` has `source_artifact`, `source_row_key`, `source_generation`, `source_evaluation_scope`, `evaluation_scope`, and `scientific_status`. Source paths are relative to the original read-only source collection, not links to files shipped here. The [source index](../data/manifests/source_artifact_index.csv) supplies SHA-256 and byte size of each source artifact. Transformed tables carry source hashes where applicable.

Classification Validation rows use experiment IDs; Test rows use finalist roles within generation. Candidate dictionaries use feature names. Regression Development uses candidate ID plus original scope; the 120-row pivot points to all three saved scope rows and is accompanied by the 360-row long form. IID rows use model IDs; bootstrap rows use metric names. JSON six-condition evidence points to the original `conditions` array. No table was computed from row-level predictions.

The public artifact manifest identifies each packaged file's role and transformation. Source-availability caveats and implementation/manuscript differences are tracked in [unresolved gaps](LIMITATIONS.md). Static row mappings and hashes identify the saved sources. They do not recreate missing data or model evidence.

## This publication update

[Publication copies](../data/manifests/publication_copies.json) records source-relative paths, original and publication SHA-256 values, exact-copy or redaction status, and original CSV line mappings. CSV line 1 is the header. Filtered tables retain every value of the selected source row. [Notebook provenance](../data/manifests/notebook_provenance.csv) and [output/metadata redactions](../data/manifests/notebook_redactions.json) preserve the distinction between an exact copy and a sanitized copy. The two public artifact manifest files are excluded from their own hash set to avoid recursive self-hashing.

Source-relative paths identify the original read-only collection; they are not all files redistributed in this repository. The historical repository tree was inspected read-only and contains the older notebooks, not the Regression V2 notebook set. The fuller local retained projects supply the current generations. Detailed source inventory and operational logs stay outside tracked content.

This update uses static inspection only: source-byte and notebook-cell comparison, saved-table string comparison, link inspection, file counts and checksums. No project validator, test, application, notebook or CI job was run. Historical rc1 tests and saved notebook status messages must not be assigned to the new commit.

The old completion report, release-readiness certificate, self-review dashboard and separate safety report were removed. Thirty-seven short documentation files were consolidated into six technical guides. Scientific caveats and source lineage remain in these guides and machine-readable tables.
