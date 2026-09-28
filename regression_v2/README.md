# Regression V2

Original source, tests and config.json are copied byte-for-byte. The 13 existing notebooks cover cleaning through final reporting. Safe saved aggregate outputs remain inline. Exact runtime reproduction still requires excluded private dependencies.

No code, notebook or tests were run in this update. See the [notebook index](../notebooks/README.md), [workflow notes](../docs/USAGE.md), and [limitations](../docs/LIMITATIONS.md).

- [01A_CLEAN_RAW_DATA.ipynb](notebooks/01A_CLEAN_RAW_DATA.ipynb): Saved cleaning-report presentation.
- [01B_BUILD_FINAL_DATASETS.ipynb](notebooks/01B_BUILD_FINAL_DATASETS.ipynb): Saved dataset-construction report; reads storage metadata if executed.
- [02_BASELINES_AND_BOOSTING.ipynb](notebooks/02_BASELINES_AND_BOOSTING.ipynb): Saved baseline and boosting Development results.
- [03_DEEP_TABULAR_MODELS.ipynb](notebooks/03_DEEP_TABULAR_MODELS.ipynb): Saved deep-model Development and refit report.
- [04A_INITIAL_ENSEMBLE_AND_TAIL_EXPERIMENTS.ipynb](notebooks/04A_INITIAL_ENSEMBLE_AND_TAIL_EXPERIMENTS.ipynb): Development ensemble and tail experiments.
- [04B2_BENEFIT_AWARE_SELECTIVE_CORRECTION.ipynb](notebooks/04B2_BENEFIT_AWARE_SELECTIVE_CORRECTION.ipynb): Adaptive benefit-aware Development analysis.
- [04B3_BEAT_PROBABILITY_SHRINKAGE.ipynb](notebooks/04B3_BEAT_PROBABILITY_SHRINKAGE.ipynb): Adaptive beat-probability Development report.
- [04B4_IMBALANCE_AWARE_GLOBAL_TRAINING.ipynb](notebooks/04B4_IMBALANCE_AWARE_GLOBAL_TRAINING.ipynb): Imbalance-aware Development report.
- [04B_STEPWISE_TAIL_IMPROVEMENT.ipynb](notebooks/04B_STEPWISE_TAIL_IMPROVEMENT.ipynb): Stepwise Development tail analysis.
- [04C_FINAL_SELECTION_AND_PRE_IID_FREEZE.ipynb](notebooks/04C_FINAL_SELECTION_AND_PRE_IID_FREEZE.ipynb): Frozen final selection and refit report.
- [05A_ONE_TIME_IID_EVALUATION_AND_ERROR_ANALYSIS.ipynb](notebooks/05A_ONE_TIME_IID_EVALUATION_AND_ERROR_ANALYSIS.ipynb): Saved one-time IID evaluation report.
- [05B_FAIRNESS_AND_FINAL_EXPLAINABILITY.ipynb](notebooks/05B_FAIRNESS_AND_FINAL_EXPLAINABILITY.ipynb): Saved aggregate fairness and component explanations.
- [05C_FINAL_PROJECT_REPORTING.ipynb](notebooks/05C_FINAL_PROJECT_REPORTING.ipynb): Saved final paper tables and figures.
