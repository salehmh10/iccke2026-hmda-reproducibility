# Classification models

The 14 retained families are Dummy, Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, Bagging, Stacking, MLP, Tabular Transformer, Modern Hopfield, TabNet, and ML+DL Hybrid. Each has three strategies in each representation. Full saved per-run hyperparameters are in [model_families.yaml](../configs/classification/model_families.yaml); the 84 Validation rows preserve runtime, seed, threshold, sampler, and diagnostic values.

Classical factories: stratified Dummy; saga Logistic (C=1, max_iter=500, tol=0.001); RF/Extra Trees (180 trees, leaf minimum 3, sqrt features); XGBoost (250 trees, depth 6, rate .08, row/column fractions .85); LightGBM (250 trees, 31 leaves, rate .06); CatBoost (250 iterations, depth 7, rate .08, Logloss); Bagging (80 depth-10 trees); Stacking (logistic, Extra Trees, LightGBM bases and logistic meta-model, CV=3). Saved per-run metadata controls over factory defaults when a run differs. The retained oversampled Stacking implementation groups source clones to prevent cross-fold duplication leakage.

MLP has widths 128/64, BatchNorm, ReLU, dropout .2. Transformer uses scalar feature tokens of width 16, four heads, two encoder layers, feedforward width 64, GELU, pre-norm, CLS token, dropout .1. Hopfield uses a 64-wide projection, 32 learned memory patterns, scaled softmax retrieval, residual LayerNorm and a 32-wide head, dropout .15. These custom implementations do not establish fidelity to a separate reference implementation.

Torch models use AdamW (rate .001, weight decay .0001), weighted BCEWithLogitsLoss, gradient clipping 5, deterministic seed 20260809, and best in-memory Validation-AP state. MLP: 18 epochs/patience 4/batch 1024; Transformer: 10/3/512; Hopfield: 16/4/1024. Saved best epochs are included per run. Default scheduler is none and label smoothing zero.

TabNet explicitly sets n_d=n_a=16, three steps, gamma 1.3, sparse penalty .0001, max epochs 25, patience 4, batch 1024, virtual batch 128, and uses Validation ROC-AUC stopping. Its optimizer and other unrecorded library defaults are unresolved; the saved environment names pytorch-tabnet 4.1.0. No defaults were inferred by importing that package. All final family/strategy fits have a 60,000-row cap.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/models/classical.py`
- `HMDA_pipeline_review/project/new new project/src/models/neural.py`
- `HMDA_pipeline_review/project/new new project/scripts/train_dl.py`
- `HMDA_pipeline_review/project/new new project/reports/EXPERIMENT_RESULTS.csv`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/EXPERIMENT_RESULTS.csv`
- `HMDA_pipeline_review/project/new new project/reports/ENVIRONMENT.md`
