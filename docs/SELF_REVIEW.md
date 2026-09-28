# Release-candidate self-review

The sole release engineer performed a separate review after package construction. No independent agent or independent scientific reproduction is claimed.

## Findings and repairs

| Finding | Severity | Resolution |
|---|---|---|
| Historical first-stage column-drop count was misstated | Minor | Corrected to 27 after AST inspection of the source notebook literal |
| Initial metadata search could choose the older main.tex | Major | Root PDF byte identity establishes the TARGETED_EDITS PDF and matching LaTeX source; metadata documentation corrected |
| V1/V2 cap-seed difference was assumed without support | Major | Removed; only the supported asymmetric membership-evidence limitation remains |
| HGB preprocessing summary omitted high-cardinality frequency encoding | Minor | Added alongside ordinal encoding |
| Cross-platform line endings could invalidate manifest hashes | Major | New package text normalized to LF and .gitattributes fixes LF checkout |
| Wide-to-long body/tail export needs explicit source-field check | Minor | Added validator comparing each exported band with its corresponding saved IID column |

All listed repairs were verified. No unresolved Critical or Major repository defect was found in this self-review. Scientific gaps remain in UNRESOLVED_EVIDENCE_GAPS.md; they were not repaired by inventing evidence.

## Review coverage

Checked source roles, generation separation, action/target definitions, denial-positive AP, reused Test, non-causal V1/V2 comparison, fallback interaction selection, probability-dependent routing, C2 failure, historical Stage4L isolation, source immutability, source hashes, aggregate export parity, static-code syntax, cleared notebook outputs, credential/local-path patterns, result headers/scopes, duplicate keys, file sizes/extensions, manifest coverage, and internal links. Keyword hits for tokens, predictions, sensitive fields, and IDs are implementation/configuration/aggregate/documentation terms, not disclosed applicant rows or credentials. No manuscript or Related Work change was made.
