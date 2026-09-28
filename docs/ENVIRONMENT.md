# Environment scope

Package validators use Python standard library only; tests use pytest. This run used Python 3.12.3 and pytest 9.1.1 without installing dependencies. CI uses Python 3.12 and pinned pytest 9.1.1. Reference source is never imported by validators.

The saved Classification report lists Python 3.13.9, numpy 2.3.5, pandas 2.3.3, scipy 1.16.3, sklearn 1.7.2, imbalanced-learn .14.0, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, torch 2.11.0+cu128 and pytorch-tabnet 4.1.0. It notes a mixed Conda/user-site environment and NumPy-before-Torch import ordering. These are historical saved observations, not installation recommendations verified now.

Regression recipe metadata separately records Python 3.12.3, numpy 2.2.6, pandas 2.2.2, pyarrow 25.0.0, sklearn 1.9.0 and scipy 1.13.0 for that artifact lineage. Do not combine these into an invented common lockfile. Exact clean scientific environments and unrecorded defaults remain unresolved. No environment or model binary is bundled.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/reports/ENVIRONMENT.md`
- `regresionpart2/regression_v2/outputs/reports/prompt4c_stage3_recipe_reproduction.json`
