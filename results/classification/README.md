# Classification evidence tables

These existing exports retain source_artifact, source_row_key, source_generation, evidence scope and scientific status. The original scientific values were not recalculated in this update.

- [33 encoded V1 names](feature_v1_dictionary.csv): saved smoke evidence; production category order is not independently confirmed.
- [33 selected interactions and formulas](feature_v2_interactions_33.csv), [98 candidates](interaction_candidates_98.csv), [screening](screening_seed_results.csv), [confirmation](confirmation_results.csv).
- [84 Validation rows](validation_results_84.csv): 14 families x three imbalance strategies x two generations.
- [Six Test finalists](final_test_results_6.csv), [calibration](calibration_summary.csv), [thresholds](thresholds.csv), [hybrid weights](hybrid_weights.csv).
- [Invalid and superseded rows](invalid_and_superseded.csv).

Confirmation had no eligible configuration and used a fallback. AP is denial-positive; source pr_auc means Average Precision. Reused-Test comparisons are descriptive and partially confounded. See [Classification](../../docs/CLASSIFICATION.md).
