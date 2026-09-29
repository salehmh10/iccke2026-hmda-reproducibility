# Existing workflow and environments

This is a faithful publication copy, not a tested portable application. Nothing below was executed in this update, including --help. Reading the saved outputs needs no Python environment. Do not rerun the frozen Test/IID workflows.

## Classification project

The original subproject is copied under [classification](../classification/README.md). Its src, scripts, config and requirements remain together. Use that subproject as the working directory when studying relative paths. The loader resolves config/data.yaml and the configured hmda_classification_stratified_500k.csv relative to the project root. That applicant file, data splits and trained artifacts are excluded.

The existing stage order is data inspection and dataset construction, V1 feature construction, V1 classical/deep/hybrid development, baseline snapshot, interaction analysis and confirmation, V2 classical/deep/hybrid development, finalist selection, calibration/evaluation and generation freeze. It is not a new end-to-end command.

Static CLI examples below name arguments found in the copied source. They document the original interfaces and are not instructions to execute them on frozen data:

| Entry point, from classification/ | Existing arguments |
|---|---|
| scripts/train_ml.py | --root ., --models catboost, --variants undersampled, --max-train-rows 60000, --seed 20260809, --n-jobs 8 |
| scripts/train_feature_v2_ml.py | --root ., --max-train-rows 60000, --seed 20260809, --n-jobs 8 |

The [original requirements](../classification/requirements.txt) and [saved environment report](../classification/reports/ENVIRONMENT.md) are copied unchanged. The report describes a mixed Conda/user-site environment and NumPy-before-Torch import ordering. Its historical import checks are not checks of this repository. The exact production categorical vocabulary remains unresolved.

## Regression V2 project

The [regression_v2](../regression_v2/README.md) directory preserves config.json, src, notebooks and selected outputs in their original relative structure. Use regression_v2/ as the notebook working directory; some notebooks also detect a notebooks/ working directory, but not all do.

The code stages are data_cleaning.py, final_dataset_builder.py, prompt2_modeling.py, prompt3_deep_models.py, prompt4a_experiments.py, prompt4b_experiments.py, prompt4b2_experiments.py, prompt4b3_experiments.py, prompt4b4_experiments.py, prompt4c_final.py, prompt5a_evaluation.py, prompt5b_analysis.py and prompt5c_reporting.py. The last stages describe frozen refits, the one-time IID evaluation, interpretation and reporting.

| Entry point, from regression_v2/ | Existing interface |
|---|---|
| src/data_cleaning.py | Subcommands are defined at the end of the copied file; its source reader expects exactly one extracted CSV under data/ |
| src/final_dataset_builder.py | validate-handoff, smoke, full |
| src/prompt2_modeling.py | run / verify / ready / clean-check; --root |
| src/prompt3_deep_models.py | --root followed by prepare-design / smoke / fit-candidates / refit / promote / report / build-notebook / verify / readiness / run |
| src/prompt4c_final.py | prepare / fit / finalize / all / blocka-worker / clean-reload-worker; --root |
| src/prompt5b_analysis.py | preflight / fairness / explain / figures / notebook / candidate / record-review; --root |

The real config names the nationwide raw CSV and ZIP. No raw file is supplied. Later scripts depend on outputs/data, predictions, models, staging files and original governance files. final_dataset_builder.py also uses the original parent-directory layout for legacy sources. Those dependencies were not replaced with fake files or new path logic. Some historical modules are excluded because their code contains personal paths. A future portable execution requires separately authorized work.

[config.json](../regression_v2/config.json) is an exact copy. [The saved Prompt3 environment](../regression_v2/outputs/reports/prompt3_environment.json) preserves versions but replaces python_executable and cache_paths with user-local placeholders. It is an environment observation, not a lockfile. No complete Regression V2 dependency lock was found; different stages report different environments. No packages were installed or versions updated.

## Historical work and old checks

[Historical notebooks](../historical/README.md) keep their original code and stored outputs where safe. They can need original parent paths, data and unavailable helpers. Saved figures remain viewable inline even when their source artifact files are excluded.

The root scripts/, tests/ and src/common/ belong to the earlier rc1 packaging policy. They are retained for reference and were not executed for this update. Their blanket notebook-output rule is not applicable to the saved-output notebooks distributed here. Original task tests under classification/tests and regression_v2/tests are also unchanged and unexecuted. The historical rc1 maintenance workflow retains its original job logic and only a workflow_dispatch trigger; repository Actions remain disabled.
