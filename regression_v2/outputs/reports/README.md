# Saved Regression V2 reports

These files were copied from the existing Regression V2 outputs/reports directory. No metrics, rankings, intervals or plots were recomputed. The original filenames and precision are retained. [Publication source/row mappings](../../../data/manifests/publication_copies.json) identify every copy and any suppressed rows.

| Evidence | Files |
|---|---|
| Full feature importance | [Global, 35 features](prompt5b_global_feature_importance.csv), [gate, 36 inputs](prompt5b_gate_feature_importance.csv), [residual, 36 inputs](prompt5b_residual_feature_importance.csv) |
| Component prediction means and weights, not feature importance | [Component summary](prompt5b_global_component_summary.csv) |
| Base candidates and Train-mean/median baselines | [Candidates](prompt2_candidate_results.csv), [family comparison](prompt2_family_comparison.csv) |
| Lender-feature diagnostics | [Lender ablation](prompt2_lender_ablation.csv) |
| Separate deep refit | [FT full-Train refit result](../models/prompt3/refits/selected_ft_full_train/result.json), [saved deep anchor](prompt3_deep_anchor.json) |
| Deep candidates | [Candidate results](prompt3_candidate_results.csv), [common Validation leaderboard](prompt3_common_validation_leaderboard.csv) |
| IID metrics and signed error | [Overall](prompt5a_iid_overall_metrics.csv), [deciles](prompt5a_iid_local_decile_metrics.csv), [underprediction](prompt5a_underprediction_analysis.csv), [body/tail bands](prompt5a_development_frozen_band_metrics.csv) |
| Paired uncertainty | [Bootstrap intervals](prompt5a_iid_bootstrap.csv) |
| Aggregate comparison despite the historical filename “rowwise” | [Primary versus Global summary](prompt5a_primary_vs_global_rowwise.csv) |
| Group errors | [Primary](prompt5b_primary_group_metrics.csv), [Global](prompt5b_global_group_metrics.csv), [comparison](prompt5b_group_primary_vs_global.csv) |
| Tail and intersections | [Tail groups](prompt5b_tail_group_metrics.csv), [intersections](prompt5b_intersectional_metrics.csv), [group-decile cells](prompt5b_group_decile_metrics.csv) |
| Group uncertainty and disparity | [Bootstrap](prompt5b_fairness_bootstrap.csv), [disparity summary](prompt5b_disparity_summary.csv) |

Files beginning prompt2/prompt3 contain Development results. prompt4 tables preserve Selection, Audit and complete-Validation scopes, including later adaptive/descriptive use. prompt5a is the frozen one-time IID evaluation. prompt5b is post-IID descriptive analysis. They are not one common leaderboard.

Global SHAP columns keep raw-kUSD CatBoost/LightGBM and native log1p XGBoost separate. The weighted normalized-rank consensus is not raw SHAP addition. Gate importance concerns routing log-odds; residual importance concerns signed kUSD correction. Individual cases and explanations are excluded.

Publication filtering uses the saved flags: only ELIGIBLE subgroup rows and DISPLAY group-decile rows remain. Ordinary groups require n>=200, tail groups n>=50, and group-decile cells n>=30. Overall intersectional groups require n>=500; tail-restricted intersectional groups require n>=50. Publication retains the recorded eligibility/display flags. Small-group labels, compositions and metrics are excluded together. All retained cell strings match their saved source rows. Fairness is descriptive, non-causal and non-legal.

The environment JSON has path-only publication redactions. Historical status fields and saved PASS labels refer to the original work; no tests were executed for this update.
