# Classification

## Classification pipeline generations

| Source generation | Role and limits |
|---|---|
| Original `binary classification part.ipynb` | historical; CatBoost early stopping uses Test; resampled Test and repeated model comparisons; inconsistent holdout populations |
| `binary classification par1t.ipynb` | intermediate notebook; excluded from retained paper grid |
| `binary classification corrected v2.ipynb` / `classification_fixed.py` | intermediate improvements: deduplication and Validation-driven choices; approval-positive AP and repeated Validation roles; not the authoritative denial-positive generation |
| `hmda_binary_pipeline_v2.py` and its tests | development/reference: five-way Train/Validation/calibration/threshold/Test split, lock, separate approval/denial AP; no final paper-authoritative evidence was established for it |
| `new new project` V1 `final_v1_versioned_equal_60k` | retained 42-run denial-positive Validation grid, baseline rollback, three Test finalists |
| `new new project` V2 `feature_v2_onehot_numeric_equal_60k` | retained 42-run denial-positive Validation grid and three Test finalists; reused physical Test |
| Flattened `regresionpart2/classification_v2` | partial reporting copy; generic reports often describe V1; full local retained project has more complete V2 evidence |

The old classification audit's statement that label-construction code was absent is superseded as a source-availability statement: `main/DATA_CLEANING_FOR_14M.ipynb` now supplies that code. Its leakage findings remain relevant. A recovered historical implementation does not prove an immutable execution chain to the exact final extract.

Source evidence (relative to the read-only source collection):
- `binary classification part.ipynb`
- `HMDA_pipeline_review/project/binary classification par1t.ipynb`
- `HMDA_pipeline_review/project/binary classification corrected v2.ipynb`
- `HMDA_pipeline_review/project/classification_fixed.py`
- `HMDA_pipeline_review/project/hmda_binary_pipeline_v2.py`
- `HMDA_pipeline_review/project/HMDA_BINARY_CLASSIFICATION_AUDIT.md`
- `HMDA_pipeline_review/project/new new project/reports/V1_V2_RETENTION_CLEANUP.md`

## Classification Feature-V1

Feature-V1 contains nine numeric encoded outputs plus 24 category indicators, for 33 columns. The ordered [dictionary](../results/classification/feature_v1_dictionary.csv) enumerates all 33 names. Numeric formulas and source fields are explicit. Categorical source fields are agency, loan type, property type, loan purpose, owner occupancy, preapproval, and lien status.

The dictionary recovers complete encoded labels from saved standalone smoke metadata, not by deserializing a production encoder. Therefore the observed smoke vocabulary is complete, while exact production vocabulary identity is unresolved. Positions reconstruct the declared numeric order and alphabetic OneHotEncoder order; they are not a recovered production serialized feature list.

Names use `numeric__<field>` or `categorical__<source>_<level>`. Invalid negative numeric inputs become missing; ratios require positive denominators. Explicit zero flags distinguish observed zero from invalid/missing values. Numeric median imputation (with train-observed missingness indicators) and StandardScaler are train-fitted. Categoricals use mode imputation and OneHotEncoder with unknown categories ignored. `Not applicable` remains a domain level. Extra missingness columns are possible in a different dataset, so 33 is the saved extract's contract, not a universal dimensionality guarantee.

The raw target, direct demographic fields, respondent ID, minority-population field, and fine-grained geography are dropped in this classification contract. Loan-to-income is an LTI proxy, not debt-to-income. Socioeconomic proxies remain.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/features/financial.py`
- `HMDA_pipeline_review/project/new new project/src/features/preprocessing.py`
- `HMDA_pipeline_review/project/new new project/notebook_outputs/feature_v2_standalone/mutual_information_smoke.csv`
- `HMDA_pipeline_review/project/new new project/reports/FEATURE_REPORT.md`

## Classification preprocessing

All authoritative classifier families consume the shared engineered/encoded V1 or V2 representation. This classifier CatBoost uses encoded features; it must not inherit the native-categorical description of Regression V2 CatBoost. Numerical median/missingness handling and scaling, category mode imputation and unknown-safe one-hot encoding are fitted on Train only. The saved primary preprocessor fit cap is 60,000 original-weighted Train rows.

Sparse matrices are used where supported by classical models. Torch and TabNet explicitly convert encoded matrices to contiguous dense float32. Imbalance sampling is applied only to training adapters; Validation and Test preserve their original prevalence. Class/sample weights apply in weighted-original; neural positive weights use the Train class ratio. Oversampled/undersampled variants use balanced resampling rather than additional class weighting.

The V1 metadata and V2 metadata are asymmetric: the preserved metadata does not provide symmetric per-model training-membership proof for the two generations. Thus matching caps and preprocessing semantics do not certify identical V1/V2 training membership. No row membership is opened or regenerated here.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/training/context.py`
- `HMDA_pipeline_review/project/new new project/src/training/context_v2.py`
- `HMDA_pipeline_review/project/new new project/src/data/variants.py`
- `HMDA_pipeline_review/project/new new project/src/features/preprocessing.py`
- `HMDA_pipeline_review/project/new new project/src/models/neural.py`

## Classification Feature-V2

Feature-V2 appends 33 selected interactions to the 33-column base, giving 66 encoded columns. Every interaction has the generic form `J(X,C=c) = X * 1[C=c]`; [the exact saved names](../results/classification/feature_v2_interactions_33.csv) are published with their source pair, formula, selection reason, and scope.

There are eleven category/numeric pair templates: loan type with amount and LTI; property type with amount and LTI; loan purpose with amount and LTI; owner occupancy with income and LTI; preapproval with amount; lien status with amount and LTI. Fitted levels expand these templates into 33 features. Names contain the numeric source, category source, readable category slug truncated to 28 characters, and the first six SHA-1 hex characters of the category label. This hash describes a category name, not an applicant identifier.

Unseen categories activate none of their fitted interactions. The code uses X-only training categories; no target is used in interaction construction. These are selected interactions. The confirmation result had no eligible configuration and retained this family through the documented fallback rule.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/features/advanced.py`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/FEATURE_VALIDATION.csv`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/CONFIRMED_FEATURE_CONFIG.json`

## Interaction screening

The [98-row candidate dictionary](../results/classification/interaction_candidates_98.csv) preserves exact saved formulas, source pairs, statistics, fallbacks, and keep/reject decisions. Family sizes are 10 numeric-relative, 15 nonlinear, 40 category-context, and 33 onehot-numeric candidates. Base features are not counted among the 98.

Numeric-relative features compare loan/income with area and tract context and include gaps, normalized differences, and shares. Nonlinear features include square roots, log1p, fixed indicators, and five train-median/IQR standardized features. Category-context includes frequency and rare/unseen indicators for seven source fields and two conditional transforms for each of 13 category/numeric pairs. Rare means training support below 500. Missing category keys use `__MISSING__`; unseen frequency is zero. Conditional medians/IQR fall back to global train values; nonfinite median becomes zero and degenerate global IQR becomes one, with numerical denominator guards. Onehot-numeric expands eleven pair templates over training levels.

Seven screening configurations: `all_candidates`, `baseline_v1_equivalent`, `categorical_numeric_combined`, `category_context`, `nonlinear`, `numeric_relative`, `onehot_numeric`. The first screening uses seeds 20260809 and 20260810, three Train-only OOF folds, and LightGBM/logistic anchors. Third-seed confirmation uses 20260811 and checks improvement for both models without additional logistic convergence warnings. The eligible list is empty. `confirm_feature_v2.py` explicitly selects `onehot_numeric` in that case. The retained family must not be called confirmation-validated. MI/redundancy statistics remain supporting screening evidence, not causal effects.

These screening seeds do not establish final-model seed robustness. Validation and Test results are separate from these OOF tables.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/features/advanced.py`
- `HMDA_pipeline_review/project/new new project/scripts/analyze_feature_v2.py`
- `HMDA_pipeline_review/project/new new project/scripts/confirm_feature_v2.py`
- `HMDA_pipeline_review/project/new new project/reports/generations/feature_v2/CONFIRMED_FEATURE_CONFIG.json`

## Classification models

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
- `HMDA_pipeline_review/project/new new project/reports/USAGE.md`

## Classification metric orientation

Paper comparisons use **denial-positive Average Precision (AP)**. The saved column `pr_auc` calls sklearn `average_precision_score(y_denial, p_denial)`. AP sums precision weighted by recall increments; it is not trapezoidal integration of the precision-recall curve. `target_denied = 1 - loan_approved`; classifier probability for class label 1 is denial probability.

MCC, balanced accuracy, denial recall, F1 and confusion counts refer to this same orientation. Balanced accuracy averages approval and denial recall. Thresholds operate on denial probability. Test table metrics are after Validation-derived calibration/threshold choice; raw Validation leaderboard metrics and calibrated Test metrics have different evidence roles.

The intermediate `classification_fixed.py` uses `loan_approved=1` and reports approval-positive AP. The five-way development/reference pipeline reports approval and denial AP separately. Historical or intermediate `pr_auc` columns are never silently promoted to denial-positive AP.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/evaluation/metrics.py`
- `HMDA_pipeline_review/project/new new project/src/models/classical.py`
- `HMDA_pipeline_review/project/classification_fixed.py`
- `HMDA_pipeline_review/project/hmda_binary_pipeline_v2.py`

## Calibration, thresholds, and hybrids

Within each strategy, hybrid selection uses Validation probabilities to choose classical/neural bases and soft-blend weights. The selected undersampled hybrid combines CatBoost and Modern Hopfield: V1 weights .85/.15; V2 weights .775/.225. All strategy weights and source IDs are in [hybrid_weights.csv](../results/classification/hybrid_weights.csv). These are not fixed across the V1/V2 comparison.

Final evaluation divides Validation into 49,974 calibration-fit and 49,974 threshold-selection rows. Candidate calibration methods are none, sigmoid, and isotonic. The objective is Brier score + .1 * log loss; all six saved finalists select isotonic. The fitted calibration functions differ across runs even when their method names match. Validation is reused across earlier model/weight selection and later calibration roles, so the full selection process is not independent of those holdouts.

Threshold search considers .5 plus 199 probability quantiles from .005 to .995, excluding 0 and 1. It maximizes MCC + .001 * balanced accuracy. Final thresholds in [thresholds.csv](../results/classification/thresholds.csv) are calibrated thresholds; hybrid ledger thresholds are earlier uncalibrated Validation values.

The four V1/V2 CatBoost/Hybrid cases are `descriptive_partially_confounded_comparison`: hybrid weights, fitted calibrators, and thresholds differ; exact capped training-membership evidence is asymmetric; and physical Test had already been used. This is not a controlled causal ablation or fresh holdout validation.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/scripts/train_hybrid.py`
- `HMDA_pipeline_review/project/new new project/scripts/train_feature_v2_hybrid.py`
- `HMDA_pipeline_review/project/new new project/scripts/evaluate_all.py`
- `HMDA_pipeline_review/project/new new project/scripts/evaluate_feature_v2.py`
- `HMDA_pipeline_review/project/new new project/src/evaluation/metrics.py`
