# Publication safety review

The package is constructed from allowlisted static source, documentation, configurations and saved aggregate tables. No raw/applicant table, original Test/IID file, row-level predictions, model serialization, or scientific notebook execution was used. One historical cleaning notebook has every output, execution count and original metadata removed; code paths are sanitized. Reference source is inspection-only and is never imported by CI.

Automated checks reject local absolute paths, credential-shaped values/private keys, forbidden binaries and data filenames, applicant-level CSV/JSON fields, notebook outputs/attachments, files over 2 MB, missing result scopes, duplicate source row keys, broken links, and manifest coverage/hash mismatches. Test examples are synthetic and create credential/path patterns at runtime rather than embedding apparent secrets in tracked fixtures.

Manual review distinguishes legitimate implementation mentions of predictions, sensitive fields, IDs, environment variable names, and serialization APIs from actual stored records or credentials. Exact scan results and counts are recorded in CODEX_COMPLETION_REPORT.md. A static validator cannot prove every conceivable privacy property; the repository remains private for author review.
