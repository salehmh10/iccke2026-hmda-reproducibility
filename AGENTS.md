# Repository guardrails

This private release candidate packages existing scientific evidence only.

- The historical ADS-PROJECT repository and the original local source tree are read-only. All generated files belong in this repository or its separate automation workspace.
- Never edit the accepted manuscript, figures, bibliography, author metadata, or camera-ready package. Related Work change 13 is excluded.
- Never train, refit, tune, predict, execute scientific notebooks, reconstruct split membership, or recompute metrics, fairness, bootstrap, or explanations. Static source inspection and arithmetic on saved aggregates are allowed.
- Never publish raw or applicant-level data, membership identifiers, row hashes, predictions, sensitive row values, executable model objects, personal paths, credentials, or caches.
- Keep Classification V1/V2, historical classification, Regression V2, and historical Stage4L separate. Label historical, development-only, invalid, superseded, and post-Test evidence explicitly.
- Classification paper metrics use denial-positive average precision; reused Test comparisons are descriptive and partially confounded. Regression V2 passed five of six conditions, with C2 failed.
- Every result needs relative source lineage, row keys, scientific status, and evaluation scope. Mark missing evidence unresolved instead of guessing.
- Use lightweight synthetic-only tests and publication, traceability, scope, manifest, notebook-output, and link validators. No ML-framework import is needed for package checks.
- Work on codex/build-complete-reproducibility-package. No direct main push before validation. Merge only after observed successful remote CI and local safety/source-integrity checks.
- Before any remote mutation verify the GitHub account is salehmh10. Never mutate ADS-PROJECT. Keep this repository private and releases draft. Do not assign a license without existing permission.
- One agent performs this task; do not delegate scientific or release work.
