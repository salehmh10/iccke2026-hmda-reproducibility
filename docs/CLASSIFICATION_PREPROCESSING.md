# Classification preprocessing

All authoritative classifier families consume the shared engineered/encoded V1 or V2 representation. This classifier CatBoost uses encoded features; it must not inherit the native-categorical description of Regression V2 CatBoost. Numerical median/missingness handling and scaling, category mode imputation and unknown-safe one-hot encoding are fitted on Train only. The saved primary preprocessor fit cap is 60,000 original-weighted Train rows.

Sparse matrices are used where supported by classical models. Torch and TabNet explicitly convert encoded matrices to contiguous dense float32. Imbalance sampling is applied only to training adapters; Validation and Test preserve their original prevalence. Class/sample weights apply in weighted-original; neural positive weights use the Train class ratio. Oversampled/undersampled variants use balanced resampling rather than additional class weighting.

The V1 metadata and V2 metadata are asymmetric: the preserved metadata does not provide symmetric per-model training-membership proof for the two generations. Thus matching caps and preprocessing semantics do not certify identical V1/V2 training membership. No row membership is opened or regenerated here.

Source evidence (relative to the read-only source collection):
- `HMDA_pipeline_review/project/new new project/src/training/context.py`
- `HMDA_pipeline_review/project/new new project/src/training/context_v2.py`
- `HMDA_pipeline_review/project/new new project/src/data/variants.py`
- `HMDA_pipeline_review/project/new new project/src/features/preprocessing.py`
- `HMDA_pipeline_review/project/new new project/src/models/neural.py`
