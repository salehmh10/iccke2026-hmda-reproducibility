# Historical regression and post-Test work

The historical official model `stage4l__blend__without_sensitive` has locked-Test MAE 61.511631217701016 kUSD on 99,948 rows. It belongs to the legacy 499,736-row regression sample, not Regression V2. Historical CatBoost/LightGBM/XGBoost development and the final blend are separate from V2 even when nominal blend weights coincide.

Stage5C RealMLP without-sensitive (MAE 62.15974508887792 kUSD) and with-sensitive (61.82111373018834 kUSD) are post-Test extensions, the latter accuracy-only. These values are historical descriptive context, not independent model-selection evidence. They are deliberately not exported into the V2 leaderboard.

Historical Stage7 fairness is descriptive; Stage8 explainability includes recovery and governance restrictions; Stage9 reporting documents the closure. Metadata-only governance recovery or post-Test amendment does not reset consumed Test status. No old models or row-level artifacts are copied. No numerical ranking between historical locked-Test MAE and V2 IID MAE is valid because population, split, features, selection history, and evidence roles differ.

Source evidence (relative to the read-only source collection):
- `regresionpart2/artifacts/results/stage9/reporting/stage9_final_test_comparison.csv`
- `regresionpart2/artifacts/results/stage9/reporting/FINAL_TECHNICAL_REPORT.md`
- `regresionpart2/artifacts/results/stage9/reporting/MODEL_CARD.md`
