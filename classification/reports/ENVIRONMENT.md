# Environment Report

## Host and data

- Windows build 26200.8875, PowerShell 5.1, CPython 3.13.9 (Conda base plus user site).
- CPU: Intel Core i7-14650HX, 16 physical / 24 logical cores.
- RAM: 15.61 GiB total; about 4.28 GiB free during the independent audit.
- Disk: about 348.58 GiB free.
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8,151 MiB VRAM; driver 592.00; CUDA available from PyTorch.
- Raw CSV: 240,374,656 bytes; SHA-256 `b9c8373a9234f0a6083d0b92e87574f37c0c2be858812b2e1cbde66ae5af63fe`.

## Core packages

| Package | Version/status |
|---|---|
| numpy | 2.3.5 |
| pandas | 2.3.3 |
| scipy | 1.16.3 |
| scikit-learn | 1.7.2 |
| imbalanced-learn | 0.14.0 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| torch | 2.11.0+cu128 |
| pytorch-tabnet | 4.1.0 |
| tensorflow | 2.21.0, CPU only |
| pytest | 8.4.2 |
| polars | missing, not required |

## Verified import behavior

- Core NumPy/pandas/scikit-learn/imbalanced-learn/XGBoost/LightGBM/CatBoost imports pass.
- Direct isolated `import torch` fails with Intel OpenMP duplicate-runtime error #15.
- `import numpy` before `import torch` passes, reports CUDA available, and permits `TabNetClassifier` import.
- The unsafe `KMP_DUPLICATE_LIB_OK` workaround is prohibited for this project.
- TensorFlow imports but reports no native-Windows GPU.

## Risks and actions

- Conda base and user-site packages are mixed and unpinned; user site precedes Conda on `sys.path`.
- `pip check` reports a pre-existing Streamlit/protobuf mismatch unrelated to this project.
- Neural scripts must follow the NumPy-before-PyTorch import contract and have isolated smoke tests.
- Exact versions will be captured in the project requirements; a fresh environment remains the preferred final reproduction target.
- No model training was performed during environment discovery.

