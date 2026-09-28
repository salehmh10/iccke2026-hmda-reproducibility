# Public artifact manifest

The CSV and JSON manifests cover every packaged project file other than the two generated manifests themselves. Each entry gives relative path, SHA-256, bytes, artifact role, generation, evaluation scope, scientific status, privacy class, source evidence, copy/transformation type, and transformation description. Mixed-generation tables carry exact row-level *aggregate-result* provenance.

Self-hashing a content manifest is circular. The two manifest files are the only self-exempt generated build metadata. Their bytes are authenticated by Git commits; the validator checks CSV/JSON parity and exact coverage of all other files. Git internals and local Python/pytest caches are also excluded from the file walk and are not tracked. No other tracked file is exempt.

Run `python scripts/generate_public_manifest.py` only after deliberate package edits, then `python scripts/validate_repository.py`. The generator uses the reviewed allowlisted provenance map in `data/manifests/packaging_provenance.json`. It does not load source datasets or scientific pipelines. Validation never regenerates a manifest to make a mismatch disappear.
