# HMDA reproducibility package

**Boosting and Deep Tabular Learning on HMDA: Feature-Interaction Classification and Tail-Aware Loan-Amount Regression**

Saleh Mohammad Hasani and Reza Kazemeiny Moghaddam
Department of Electrical Engineering, Sharif University of Technology, Tehran, Iran
ICCKE 2026, paper code **1059** (owner-supplied conference context)

This **private, pre-publication release candidate** records how the study was built and where its results came from. It contains static source, configurations, saved aggregate results, and lightweight checks. It does not train models or provide applicant data. The paper and Related Work change 13 were not edited.

## Start here

- [Data provenance](docs/DATA_PROVENANCE.md), [filters](docs/DATA_FILTERS.md), and [sampling](docs/SAMPLING_DESIGN.md)
- [Experiment lineage](docs/EXPERIMENT_LINEAGE.md) and [result traceability](docs/RESULT_TRACEABILITY.md)
- [Classification features](docs/CLASSIFICATION_FEATURE_V1.md), [selected interactions](docs/CLASSIFICATION_FEATURE_V2.md), and [models](docs/CLASSIFICATION_MODELS.md)
- [Regression features](docs/REGRESSION_V2_FEATURES.md), [routing](docs/REGRESSION_V2_ROUTING.md), and [evaluation](docs/REGRESSION_V2_EVALUATION.md)
- [Unresolved gaps](docs/UNRESOLVED_EVIDENCE_GAPS.md) and [release readiness](docs/RELEASE_READINESS.md)

## Targets and evidence

Classification maps action codes **1/2/8 to favorable** and **3/7 to denial**; other actions are excluded. Favorable does not mean an originated, accepted, or disbursed loan. Paper comparisons use **denial-positive Average Precision (AP)**. The source column `pr_auc` uses `average_precision_score`, not trapezoidal area. See [action mapping](docs/ACTION_TAKEN_MAPPING.md) and [metrics](docs/CLASSIFICATION_METRICS.md).

Feature-V1 and V2 have 33 and 66 encoded columns. Their reused-Test Hybrid AP is 0.4606068466 and 0.4620686424, respectively. The [six Test finalists](results/classification/final_test_results_6.csv) are separate from the [84 Validation rows](results/classification/validation_results_84.csv). V1/V2 differences are descriptive and partially confounded by calibration, thresholds, blend weights, and training-membership evidence. No fresh classification Test or seed robustness is claimed.

Regression V2 predicts reported `loan_amount_000s`, in **thousands of US dollars (kUSD)**, among favorable-action records. Its one-time internal IID evaluation has 75,000 rows. [Saved IID results](results/regression/iid_overall_metrics.csv) give Global MAE 62.4443493 and Primary MAE 62.2606260 kUSD. The effect is small; **five of six conditions pass and C2 fails**. Body MAE worsens slightly. The paired row-bootstrap does not account for lender/geographic clustering. Historical Stage4L results are [separate](docs/HISTORICAL_REGRESSION.md).

## Repository map

| Folder | Contents |
|---|---|
| `docs/` | provenance, methods, limitations, release review |
| `configs/` | recovered feature/model configurations |
| `results/` | scoped, traceable saved aggregates |
| `data/` | schemas, aggregate construction metadata, manifests |
| `src/` | static scientific reference plus small synthetic-check helpers |
| `notebooks/` | output-cleared historical construction notebook |
| `archive/` | labelled classification reference code |
| `scripts/`, `tests/` | privacy, scope, traceability, manifest, link checks |

Open CSV files in a text editor or spreadsheet to inspect results. The 120-row Regression candidate table separates Selection, Audit, and full Validation metrics; the companion 360-row table keeps the original scopes. No aggregate table is a shared historical/V2 leaderboard.

## Run lightweight checks

Python 3.11+ and pytest are sufficient. No ML framework or HMDA download is needed.

```bash
python -m pytest -q -p no:cacheprovider
python scripts/validate_repository.py
```

CI installs pytest and runs only these checks. It never executes scientific notebooks or reference pipelines. [Environment notes](docs/ENVIRONMENT.md) distinguish validation dependencies from historical scientific environments.

## Status and limitations

Scientific statuses: `authoritative_manuscript_result`, `supporting_analysis`, `development_only`, `historical`, `post_test_descriptive`, `superseded`, `invalid`, `failed_technical`, `not_executed`, `unresolved`. Evaluation roles separately distinguish screening, Development, Train-only validation, OOF, Validation, calibration, threshold selection, final Test, IID, post-Test, historical Test, and aggregate explanations. See the [schema](data/schemas/result_schema.json).

The historical nationwide 2017 cleaning code is recovered, but its complete execution chain to the classification snapshot remains unresolved. Production category vocabulary is not independently confirmed. Row-level data are excluded to avoid exposing applicants or sensitive values, even though the underlying HMDA source is public. Exact scientific reruns require separately authorized data, complete environments, and a new protocol. Fairness and feature explanations are descriptive and non-causal; no legal or automated-lending suitability claim is made.

## Citation and review

Use [CITATION.cff](CITATION.cff) for the paper authors and repository title. Candidate version: **v1.0.0-rc1**. No DOI is assigned. This code is not yet publicly available; repository visibility must remain private until author review. [A license decision is required](docs/LICENSE_DECISION_REQUIRED.md). See the [human-review checklist](docs/RELEASE_READINESS.md) before any public release.
