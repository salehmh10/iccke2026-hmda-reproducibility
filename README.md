# HMDA paper repository

**Boosting and Deep Tabular Learning on HMDA: Feature-Interaction Classification and Tail-Aware Loan-Amount Regression**

Saleh Mohammad Hasani and Reza Kazemeiny Moghaddam  
Department of Electrical Engineering, Sharif University of Technology, Tehran, Iran  
ICCKE 2026, paper code 1059 (conference context supplied by the authors).

This repository contains the code, original notebooks, configurations, and saved results for our HMDA classification and loan-amount regression study. Start with the [notebook index](notebooks/README.md) to explore the experiments. Applicant-level data and fitted models are not included.

Version **1.0.0-rc3** is a private release candidate; releases remain drafts. Saved outputs come from earlier runs and were not regenerated during this publication update.

## Read the work

- [Notebook index](notebooks/README.md): all 29 notebooks, including 13 Regression V2 notebooks and historical benchmarks.
- Regression V2: [05A IID evaluation](regression_v2/notebooks/05A_ONE_TIME_IID_EVALUATION_AND_ERROR_ANALYSIS.ipynb), [05B fairness and explanations](regression_v2/notebooks/05B_FAIRNESS_AND_FINAL_EXPLAINABILITY.ipynb), and [05C final reporting](regression_v2/notebooks/05C_FINAL_PROJECT_REPORTING.ipynb).
- [Classification demonstration notebook](classification/notebooks/HMDA_Feature_V2_Standalone_FA.ipynb): a standalone demo, not the full experiment record.
- [Data and targets](docs/DATA.md): source, cleaning, sampling, exclusions and remaining lineage gaps.
- [Classification](docs/CLASSIFICATION.md): features, interactions, models, calibration and reused-Test limits.
- [Regression](docs/REGRESSION.md): features, models, routing, IID evaluation and explanation spaces.
- [Existing workflow and environments](docs/USAGE.md): original entry points, working directories and missing runtime materials.
- [Limitations](docs/LIMITATIONS.md) and [artifact provenance](docs/PROVENANCE.md).

## Saved main results

| Task and evidence | Model | Metric | Saved value |
|---|---|---|---|
| Classification Feature-V1, reused Test | CatBoost + Modern Hopfield | Denial-positive Average Precision | 0.4606068465893668 |
| Classification Feature-V2, reused Test | CatBoost + Modern Hopfield | Denial-positive Average Precision | 0.462068642391656 |
| Regression V2, one-time internal IID | Global, ens_boost_cat060 | MAE, thousand USD | 62.444349310930285 |
| Regression V2, one-time internal IID | Primary, stage3_residual_t75_a75 | MAE, thousand USD | 62.26062600334689 |

Sources: [six Classification Test results](results/classification/final_test_results_6.csv) and [original saved IID table](regression_v2/outputs/reports/prompt5a_iid_overall_metrics.csv). Classification uses denial as the positive class; source files named PR-AUC calculate Average Precision. V1/V2 differences are descriptive and partly confounded. Regression has a small improvement, slightly worse body MAE, and **five of six conditions passed; C2 failed**. Historical Stage4L results are separate.

## Files and tables

| Location | Contents |
|---|---|
| [classification/](classification/README.md) | Original src, scripts, runtime YAMLs, requirements and existing demonstration notebook |
| [regression_v2/](regression_v2/README.md) | Original src, runtime config, 13 existing notebooks, saved tables and figures |
| [historical/](historical/README.md) | Earlier benchmark notebooks and available unchanged helpers |
| [results/classification/](results/classification/README.md) | 33 V1 columns, 33 interactions, 98 candidates, 84 Validation rows, six Test finalists and selection details |
| [results/regression/](results/regression/README.md) | 35-feature dictionary, 120 advanced candidate identities and 360 candidate-by-scope rows |
| [Regression original reports](regression_v2/outputs/reports/README.md) | Complete importance tables, baselines, lender diagnostics, IID errors, subgroup results and uncertainty |
| [Saved final figures](regression_v2/outputs/final/figures/) | Existing aggregate figures, copied without regeneration |

Direct table links:

- Feature definitions: [Classification V1](results/classification/feature_v1_dictionary.csv), [V2 interactions](results/classification/feature_v2_interactions_33.csv), and [Regression V2](results/regression/feature_dictionary_35.csv).
- Classification results: [complete Validation grid](results/classification/validation_results_84.csv), [all Test finalists](results/classification/final_test_results_6.csv), and [selection/calibration details](results/classification/README.md).
- Regression importance: [Global](regression_v2/outputs/reports/prompt5b_global_feature_importance.csv), [Gate](regression_v2/outputs/reports/prompt5b_gate_feature_importance.csv), and [Residual](regression_v2/outputs/reports/prompt5b_residual_feature_importance.csv).
- Regression errors and uncertainty: [body/tail](results/regression/iid_body_tail_metrics.csv), [IID bootstrap](regression_v2/outputs/reports/prompt5a_iid_bootstrap.csv), [subgroup comparison](regression_v2/outputs/reports/prompt5b_group_primary_vs_global.csv), and [subgroup bootstrap](regression_v2/outputs/reports/prompt5b_fairness_bootstrap.csv).

## Data access and limits

The original study uses HMDA 2017 data. Applicant records, split membership, individual predictions/explanations, models and local environments are not included. Obtain the source through its official provider under the applicable terms; the exact author extract and legacy exclusions are also needed for a future reproduction. A public raw dataset alone does not reconstruct these frozen samples. See [data notes](docs/DATA.md).

Read notebooks on GitHub or as JSON without running them. Most Regression V2 notebooks present saved reports; some cells still access protected artifacts or create figures if executed. The standalone classifier has smoke/demo outputs and hard-coded frozen reference results; it is not the execution record for all retained experiments.

The main limitations are reused Classification Test data, fallback interaction selection, partly confounded V1/V2 comparisons, adaptive Regression Development/Audit use, incomplete source lineage and environments, private runtime dependencies, and fixed-model row-bootstrap uncertainty. Fairness is descriptive and non-legal; explanations are non-causal and component-specific. No production lending claim is made.

## Citation and license

Use [CITATION.cff](CITATION.cff) for the author names and version. The [repository](https://github.com/salehmh10/iccke2026-hmda-reproducibility) currently requires authorized access. No project license has been selected; ownership and licensing remain an author decision before public release.
