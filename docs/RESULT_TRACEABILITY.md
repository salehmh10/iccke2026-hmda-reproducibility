# Result traceability

Every CSV under `results/` has `source_artifact`, `source_row_key`, `source_generation`, `source_evaluation_scope`, `evaluation_scope`, and `scientific_status`. Source paths are relative to the original read-only source collection, not links to files shipped here. The [source index](../data/manifests/source_artifact_index.csv) supplies SHA-256 and byte size of each source artifact. Transformed tables carry source hashes where applicable.

Classification Validation rows use experiment IDs; Test rows use finalist roles within generation. Candidate dictionaries use feature names. Regression Development uses candidate ID plus original scope; the 120-row pivot points to all three saved scope rows and is accompanied by the 360-row long form. IID rows use model IDs; bootstrap rows use metric names. JSON six-condition evidence points to the original `conditions` array. No table was computed from row-level predictions.

The public artifact manifest identifies each packaged file's role and transformation. Source-availability caveats and implementation/manuscript differences are tracked in [unresolved gaps](UNRESOLVED_EVIDENCE_GAPS.md). Traceability checks verify schemas, source-index coverage, row keys, allowed roles, and cross-table consistency; they cannot recreate absent row-level or model evidence.
