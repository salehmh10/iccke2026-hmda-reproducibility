"""Build the evidence-backed validation and frozen-finalist comparison report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]


def _markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def main() -> int:
    ledger = pd.read_csv(ROOT / "reports" / "EXPERIMENT_RESULTS.csv")
    invalid_path = ROOT / "reports" / "INVALID_EXPERIMENTS.json"
    invalid_ids = set()
    if invalid_path.exists():
        invalid_ids = set(json.loads(invalid_path.read_text(encoding="utf-8"))["excluded_experiment_ids"])
    screening = ledger[
        (ledger["status"] == "completed")
        & (ledger["sample_size"] == 60000)
        & (ledger["experiment_id"].str.contains("_v", regex=False))
        & (~ledger["experiment_id"].isin(invalid_ids))
    ].copy()
    screening = screening.sort_values(["pr_auc", "mcc"], ascending=False)
    master = screening[
        [
            "model",
            "dataset_variant",
            "pr_auc",
            "roc_auc",
            "mcc",
            "balanced_accuracy",
            "f1_macro",
            "recall_denial",
            "recall_approval",
            "train_runtime_seconds",
            "degenerate",
        ]
    ].rename(
        columns={
            "model": "Model",
            "dataset_variant": "Dataset Variant",
            "pr_auc": "PR-AUC",
            "roc_auc": "ROC-AUC",
            "mcc": "MCC",
            "balanced_accuracy": "Balanced Acc",
            "f1_macro": "Macro F1",
            "recall_denial": "Denial Recall",
            "recall_approval": "Approval Recall",
            "train_runtime_seconds": "Runtime (s)",
            "degenerate": "Degenerate?",
        }
    )
    variant_summary = (
        screening[screening["model"] != "dummy"]
        .groupby("dataset_variant")[["pr_auc", "mcc", "balanced_accuracy", "f1_macro"]]
        .mean()
        .reset_index()
        .sort_values("pr_auc", ascending=False)
    )
    best_by_model = (
        screening.sort_values(["pr_auc", "mcc"], ascending=False)
        .groupby("model", as_index=False)
        .first()[["model", "dataset_variant", "pr_auc", "mcc", "train_runtime_seconds"]]
        .sort_values("pr_auc", ascending=False)
    )
    final_test = pd.read_csv(ROOT / "reports" / "FINAL_TEST_RESULTS.csv")
    selection = json.loads(
        (ROOT / "artifacts" / "models" / "finalist_selection.json").read_text(encoding="utf-8")
    )
    intervals = json.loads(
        (ROOT / "reports" / "BOOTSTRAP_INTERVALS.json").read_text(encoding="utf-8")
    )["percentile_95_ci"]

    cat = screening[screening["experiment_id"] == selection["best_single_model"]].iloc[0]
    hybrid = screening[screening["experiment_id"] == selection["best_predictive"]].iloc[0]
    practical = screening[screening["experiment_id"] == selection["best_practical"]].iloc[0]
    best_dl = screening[screening["model"].isin(["mlp", "tabnet", "tabular_transformer", "modern_hopfield"])].iloc[0]
    best_variant = str(variant_summary.iloc[0]["dataset_variant"])
    predictive_dl = screening[
        screening["experiment_id"] == selection["best_predictive_components"]["dl"]
    ].iloc[0]

    lines = [
        "# Model Comparison",
        "",
        "## Scope",
        "",
        "The master table reports validation results after validation-only threshold optimization. Classical and neural models each use 60,000 training rows per variant. All share the same train-fitted 33-feature pipeline and the same untouched 99,948-row validation split. Test results appear only for finalists frozen from validation.",
        "",
        "## Validation master table",
        "",
        _markdown_table(master),
        "",
        "## Weighted vs oversampled vs undersampled",
        "",
        _markdown_table(variant_summary.rename(columns={"dataset_variant": "Variant", "pr_auc": "Mean PR-AUC", "mcc": "Mean MCC", "balanced_accuracy": "Mean Balanced Acc", "f1_macro": "Mean Macro F1"})),
        "",
        f"Across non-dummy families, `{best_variant}` produced the highest mean validation PR-AUC in this equal-60k-row screening. Differences are small and family-dependent: undersampling supplies unique balanced rows, oversampling can repeat minority rows, and weighted-original preserves natural prevalence with loss/sample weighting.",
        "",
        "## Best variant for each model",
        "",
        _markdown_table(best_by_model.rename(columns={"model": "Model", "dataset_variant": "Best Variant", "pr_auc": "PR-AUC", "mcc": "MCC", "train_runtime_seconds": "Fit seconds"})),
        "",
        "## Family findings",
        "",
        f"- Best single model: CatBoost/undersampled, validation PR-AUC {cat.pr_auc:.6f}, MCC {cat.mcc:.6f}.",
        f"- Best DL: {best_dl.model}/{best_dl.dataset_variant}, PR-AUC {best_dl.pr_auc:.6f}; it did not beat CatBoost (gap {cat.pr_auc - best_dl.pr_auc:.6f}).",
        "- The tabular Transformer did not justify its added architectural complexity: its best PR-AUC remained below the best boosting, stacking, and Hopfield results.",
        f"- `{best_dl.model}` was the strongest neural family by validation PR-AUC. The predictive hybrid uses `{predictive_dl.model}` on the selected undersampled variant because that was the best DL base for that variant; no broader architectural superiority claim is made.",
        f"- The validation-selected ML+DL hybrid improved over CatBoost by only {hybrid.pr_auc - cat.pr_auc:.6f} PR-AUC and {hybrid.mcc - cat.mcc:.6f} MCC. This is not practically material relative to its multi-model inference cost.",
        f"- LightGBM/undersampled is the production-friendly candidate: PR-AUC {practical.pr_auc:.6f}, MCC {practical.mcc:.6f}, fit time {practical.train_runtime_seconds:.3f}s.",
        "",
        "## Frozen finalist test results",
        "",
        _markdown_table(final_test[["candidate", "pr_auc", "roc_auc", "mcc", "balanced_accuracy", "f1_macro", "recall_denial", "recall_approval", "log_loss", "brier_score", "threshold", "degenerate"]].rename(columns={"candidate": "Candidate", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC", "mcc": "MCC", "balanced_accuracy": "Balanced Acc", "f1_macro": "Macro F1", "recall_denial": "Denial Recall", "recall_approval": "Approval Recall", "log_loss": "Log Loss", "brier_score": "Brier", "threshold": "Threshold", "degenerate": "Degenerate?"})),
        "",
        "Finalist identities and the no-test-selection flag are persisted in `artifacts/models/finalist_selection.json`. Calibration was fit on one validation half, calibration method/threshold chosen on the other half, then frozen before test evaluation.",
        "",
        "## Best models",
        "",
        "- **Best Predictive Model:** validation-selected undersampled CatBoost + modern Hopfield soft blend. It leads test PR-AUC, balanced accuracy, and denial recall among frozen finalists, but not test MCC/Macro-F1.",
        "- **Best Practical Model:** undersampled LightGBM. It is much simpler/faster and its test performance is close to the single CatBoost and hybrid finalists.",
        "",
        "## Statistical robustness",
        "",
        f"For the predictive finalist, 100 seeded test bootstraps gave 95% percentile intervals: PR-AUC [{intervals['pr_auc'][0]:.4f}, {intervals['pr_auc'][1]:.4f}], ROC-AUC [{intervals['roc_auc'][0]:.4f}, {intervals['roc_auc'][1]:.4f}], MCC [{intervals['mcc'][0]:.4f}, {intervals['mcc'][1]:.4f}], balanced accuracy [{intervals['balanced_accuracy'][0]:.4f}, {intervals['balanced_accuracy'][1]:.4f}], and Macro-F1 [{intervals['f1_macro'][0]:.4f}, {intervals['f1_macro'][1]:.4f}].",
        "",
        "## Interpretation boundary",
        "",
        "The label lineage, source years, upstream stratification, and temporal field are unavailable. These are within-sample predictive comparisons for the supplied transformed extract, not forward-time estimates, national prevalence/calibration claims, causal underwriting evidence, or a fair-lending compliance determination.",
    ]
    (ROOT / "reports" / "MODEL_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_dir = ROOT / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_data = screening[screening["model"] != "dummy"].copy()
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_data, x="model", y="pr_auc", hue="dataset_variant")
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Validation PR-AUC (Denial)")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig(figure_dir / "model_pr_auc_by_variant.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=plot_data,
        x="train_runtime_seconds",
        y="pr_auc",
        hue="dataset_variant",
        style="model",
        s=90,
    )
    plt.xscale("log")
    plt.xlabel("Measured fit seconds (log scale)")
    plt.ylabel("Validation PR-AUC (Denial)")
    plt.tight_layout()
    plt.savefig(figure_dir / "runtime_vs_pr_auc.png", dpi=160)
    plt.close()
    print(f"wrote {len(master)} master rows and {len(final_test)} finalist test rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
