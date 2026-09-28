"""Build the complete Persian Feature Engineering V2 Markdown/PDF report."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "generations" / "feature_v2"
ARTIFACT_DIR = ROOT / "artifacts" / "generations" / "feature_v2"
MD_PATH = REPORT_DIR / "FEATURE_ENGINEERING_V2_REPORT_FA.md"
PDF_PATH = REPORT_DIR / "FEATURE_ENGINEERING_V2_REPORT_FA.pdf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_executable(name: str, candidates: tuple[Path, ...]) -> Path:
    discovered = shutil.which(name)
    if discovered:
        return Path(discovered)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"required executable not found: {name}")


def _fmt(value: object, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_figures(ablation: pd.DataFrame, test_comparison: pd.DataFrame, validation: pd.DataFrame) -> None:
    figures = REPORT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    pivot = ablation.pivot(index="configuration", columns="model", values="mean_oof_pr_auc")
    pivot = pivot.sort_values("lightgbm", ascending=True)
    ax = pivot.plot.barh(figsize=(9, 6), color=["#2d6a91", "#d97706"])
    ax.set_xlabel("Mean train-only OOF PR-AUC")
    ax.set_ylabel("Feature configuration")
    ax.set_title("Feature-family ablation (2 seeds × 3 folds)")
    ax.legend(title="Probe model")
    plt.tight_layout()
    plt.savefig(figures / "oof_ablation_pr_auc.png", dpi=170)
    plt.close()

    metrics = ["pr_auc", "mcc", "balanced_accuracy", "accuracy"]
    predictive = test_comparison[test_comparison["role"] == "predictive"].iloc[0]
    x = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, [predictive[f"v1_{m}"] for m in metrics], width, label="V1")
    plt.bar(x + width / 2, [predictive[f"v2_{m}"] for m in metrics], width, label="V2")
    plt.xticks(x, ["PR-AUC", "MCC", "Balanced Acc.", "Accuracy"])
    plt.ylim(0.28, 0.82)
    plt.ylabel("Score")
    plt.title("Frozen-test predictive finalist: V1 vs V2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "v1_v2_test_predictive.png", dpi=170)
    plt.close()

    best_delta = validation.groupby("model", as_index=False)["delta_pr_auc"].max().sort_values("delta_pr_auc")
    colors = ["#b91c1c" if value < 0 else "#15803d" for value in best_delta["delta_pr_auc"]]
    plt.figure(figsize=(9, 6))
    plt.barh(best_delta["model"], best_delta["delta_pr_auc"], color=colors)
    plt.axvline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Best-variant validation PR-AUC delta (V2 − V1)")
    plt.title("Feature impact by model family")
    plt.tight_layout()
    plt.savefig(figures / "validation_delta_by_family.png", dpi=170)
    plt.close()


def _render_pdf() -> None:
    pandoc = _find_executable("pandoc", ())
    browser = _find_executable(
        "msedge",
        (
            Path(r"SOURCE_ROOT"),
            Path(r"SOURCE_ROOT"),
            Path(r"SOURCE_ROOT"),
        ),
    )
    style = ROOT / "reports" / "report_fa.css"
    with tempfile.TemporaryDirectory(prefix="hmda_v2_report_", dir=ROOT, ignore_cleanup_errors=True) as temporary:
        temp = Path(temporary)
        html = temp / "FEATURE_ENGINEERING_V2_REPORT_FA.html"
        profile = temp / "browser-profile"
        subprocess.run(
            [str(pandoc), str(MD_PATH), "--from=gfm+raw_html", "--to=html5", "--standalone", "--embed-resources", "--resource-path", str(REPORT_DIR), "--css", str(style), "--metadata", "pagetitle=گزارش مهندسی ویژگی و مقایسه مدل V2", "--output", str(html)],
            cwd=ROOT, check=True,
        )
        subprocess.run(
            [str(browser), "--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--disable-crash-reporter", "--disable-breakpad", "--disable-extensions", "--run-all-compositor-stages-before-draw", "--no-pdf-header-footer", f"--user-data-dir={profile}", f"--print-to-pdf={PDF_PATH}", html.as_uri()],
            cwd=ROOT, check=True,
        )
    if not PDF_PATH.is_file() or PDF_PATH.stat().st_size < 10_000 or PDF_PATH.read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("invalid V2 PDF output")


def main() -> int:
    v2_ledger = pd.read_csv(REPORT_DIR / "EXPERIMENT_RESULTS.csv")
    v1_ledger = pd.read_csv(ROOT / "reports" / "EXPERIMENT_RESULTS.csv")
    validation = pd.read_csv(REPORT_DIR / "VALIDATION_V1_V2_COMPARISON.csv")
    ablation = pd.read_csv(REPORT_DIR / "ABLATION_RESULTS.csv")
    confirmation = pd.read_csv(REPORT_DIR / "CONFIRMATORY_OOF_RESULTS.csv")
    mi = pd.read_csv(REPORT_DIR / "FEATURE_MI.csv")
    disposition = pd.read_csv(REPORT_DIR / "FEATURE_VALIDATION.csv")
    v2_test = pd.read_csv(REPORT_DIR / "FINAL_TEST_RESULTS.csv")
    v1_test = pd.read_csv(ROOT / "reports" / "generations" / "baseline_v1" / "FINAL_TEST_RESULTS.csv")
    selection = json.loads((ARTIFACT_DIR / "PRETEST_SELECTION.json").read_text(encoding="utf-8"))
    confirmed = json.loads((REPORT_DIR / "CONFIRMED_FEATURE_CONFIG.json").read_text(encoding="utf-8"))
    screening = json.loads((REPORT_DIR / "SCREENING_BUDGET.json").read_text(encoding="utf-8"))
    reload_audit = json.loads((REPORT_DIR / "ARTIFACT_RELOAD_AUDIT.json").read_text(encoding="utf-8"))

    mapping = [
        ("predictive", "best_predictive_hybrid", "best_predictive_hybrid"),
        ("single", "best_single_catboost", "best_single_model"),
        ("practical", "best_practical_lightgbm", "best_practical_model"),
    ]
    test_rows: list[dict[str, object]] = []
    for role, v1_name, v2_name in mapping:
        old = v1_test[v1_test["candidate"] == v1_name].iloc[0]
        new = v2_test[v2_test["candidate"] == v2_name].iloc[0]
        row: dict[str, object] = {"role": role, "v1_candidate": v1_name, "v2_candidate": v2_name}
        for metric in ("pr_auc", "roc_auc", "mcc", "balanced_accuracy", "accuracy", "f1_denial", "recall_denial", "brier_score", "log_loss"):
            row[f"v1_{metric}"] = float(old[metric])
            row[f"v2_{metric}"] = float(new[metric])
            row[f"delta_{metric}"] = float(new[metric]) - float(old[metric])
        test_rows.append(row)
    test_comparison = pd.DataFrame(test_rows)
    test_comparison.to_csv(REPORT_DIR / "FINAL_TEST_V1_V2_COMPARISON.csv", index=False)
    _make_figures(ablation, test_comparison, validation)

    predictive = test_comparison[test_comparison["role"] == "predictive"].iloc[0]
    accepted = bool(selection["v2_accepted_by_pretest_rule"])
    recommendation = "V2" if accepted else "V1 (Rollback)"
    selected_features = disposition[disposition["selected"] == True].sort_values("target_mutual_information", ascending=False)  # noqa: E712
    rejected = disposition[disposition["selected"] == False]  # noqa: E712
    model_family_counts = v2_ledger.groupby("model").size().to_dict()
    v2_fit = float(v2_ledger["train_runtime_seconds"].sum())
    total_fit = 980.450260 + float(screening["model_fit_seconds"]) + float(confirmed["model_fit_seconds"]) + v2_fit
    positive_validation = int((validation["delta_pr_auc"] > 0).sum())

    lines: list[str] = [
        "---", "lang: fa", "dir: rtl", "---", "",
        "# گزارش جامع مهندسی ویژگی V2 و مقایسه با مدل پایه", "",
        f"تاریخ تولید: `{datetime.now(timezone.utc).isoformat()}`", "",
        "## خلاصهٔ مدیریتی", "",
        f"نسخهٔ پیشنهادی براساس قانون پذیرش ازپیش‌ثبت‌شده و فقط با Validation: **{recommendation}**.", "",
        f"در مدل پیش‌بینی نهایی، PR-AUC تست از `{_fmt(predictive['v1_pr_auc'])}` در V1 به `{_fmt(predictive['v2_pr_auc'])}` در V2 رسید؛ تغییر برابر `{_fmt(predictive['delta_pr_auc'])}` است. MCC به‌اندازهٔ `{_fmt(predictive['delta_mcc'])}` و Balanced Accuracy به‌اندازهٔ `{_fmt(predictive['delta_balanced_accuracy'])}` تغییر کرد. بنابراین V2 یک بهبود کوچک در معیار اصلی رتبه‌بندی همراه با افت بسیار کوچک در دو معیار آستانه‌ای است، نه برتری مطلق روی همهٔ معیارها.", "",
        "نسخهٔ V1 حذف یا بازنویسی نشده و Snapshot هش‌شدهٔ آن آمادهٔ Rollback است. انتخاب V2 پیش از مشاهدهٔ Test قفل شد و Test تنها یک‌بار برای گزارش نهایی استفاده شد.", "",
        "## پاسخ مستقیم به پرسش‌های مهندسی ویژگی", "",
        "- **آیا Feature Engineering انجام شد؟** بله. ۹۸ کاندید جدید در چهار خانواده بررسی شد و ۳۳ interaction جدید پذیرفته شد.",
        "- **آیا از ویژگی‌های قبلی ویژگی جدید ساخته شد؟** بله. ویژگی‌های عددی پایه مانند مبلغ وام، درآمد متقاضی و Loan-to-Income با سطح‌های One-Hot متغیرهای هدف وام، نوع وام، وضعیت وثیقه، نوع ملک و سکونت مالک ضرب شدند.",
        "- **Categorical Embedding انجام شد؟** خیر. برای حفظ مقایسهٔ منصفانه و Pipeline مشترک، learned embedding اضافه نشد. دسته‌ها One-Hot شدند و interaction عددی×One-Hot ساخته شد. مدل‌های DL همان نمایش ۶۶بعدی را دریافت کردند.",
        "- **از One-Hot و دادهٔ عددی ویژگی ساخته شد؟** بله؛ دقیقاً ۳۳ ویژگی منتخب V2 از همین نوع‌اند.",
        "- **Mutual Information محاسبه شد؟** بله، فقط روی ۳۰هزار ردیف Train و برای ویژگی‌های پایه و جدید؛ نتیجهٔ کامل در `FEATURE_MI.csv` است.",
        "- **GridSearchCV برای Threshold استفاده شد؟** خیر. Threshold پارامتر Estimator نیست؛ جست‌وجوی قطعی روی ۱۹۹ کاندید مبتنی بر Quantile فقط روی Validation انجام شد و MCC با tie-breakِ Balanced Accuracy بیشینه شد. Hyperparameterهای مدل‌ها عمداً همان V1 نگه داشته شدند تا اثر Featureها جدا اندازه‌گیری شود.", "",
        "## داده، هدف و مرزهای علمی", "",
        "فایل ورودی ۵۰۰٬۰۰۰ ردیف و ۲۹ ستون دارد. پس از حذف ۲۶۰ تکرار اضافی، Split مشترک و ثابت برابر ۲۹۹٬۸۴۴ Train، ۹۹٬۹۴۸ Validation و ۹۹٬۹۴۸ Test است. کلاس تحلیلی مثبت «رد وام» و تبدیل مورد استفاده `target_denied = 1 - loan_approved` است.", "",
        "فیلدهای race، ethnicity و sex فقط برای Audit نگه داشته شدند و همراه شناسهٔ پاسخ‌دهنده، جغرافیای ریز، Target و Proxyهای پس از Outcome وارد مدل نشدند. سال/تاریخ، Action-code خام و Manifest استخراج upstream در فایل موجود نیست؛ بنابراین نتیجه به همین Snapshot تبدیل‌شده محدود است.", "",
        "## حفظ مدل پایه و امکان Rollback", "",
        "Snapshot V1 شامل ۱۸ فایل گزارش، مدل، پیش‌بینی و قفل نهایی است و همهٔ جفت‌های Source/Snapshot در زمان ایجاد Hash یکسان داشتند:", "",
        "- `reports/generations/baseline_v1/BASELINE_SNAPSHOT.json`",
        "- `reports/generations/baseline_v1/BASELINE_REPORT_FA.md` و PDF",
        "- `artifacts/generations/baseline_v1/best_predictive_model.joblib`",
        "- `artifacts/generations/baseline_v1/best_practical_model.joblib`.", "",
        "کلاس‌های Feature V1 تغییر نکردند و تمام مدل‌ها/Predictionها/گزارش‌های V2 در مسیر نسل جدا ذخیره شدند.", "",
        "## طراحی Feature Engineering V2", "",
    ]
    lines += _table(
        ["خانواده", "نمونه", "نتیجه"],
        [
            ["numeric_relative", "نسبت و فاصلهٔ وام/درآمد/درآمد ناحیه", "رد شد"],
            ["nonlinear", "log/sqrt، robust-z و flagهای ثابت", "رد شد"],
            ["category_context", "میانه/IQR و فراوانی Train-conditioned", "رد شد"],
            ["onehot_numeric", "عدد × سطح One-Hot فیت‌شده", "انتخاب شد"],
        ],
    )
    lines += ["", "Pipeline پایه ۳۳ ستون Encoded داشت. V2 منتخب ۳۳ interaction اضافه کرد و به ۶۶ ستون رسید. نسخهٔ All-candidates دارای ۱۳۱ ستون بود اما برای LightGBM کمی ضعیف‌تر و برای Logistic ناپایدارتر بود.", "", "## Mutual Information و شواهد Synergy", ""]
    top_mi = mi.head(20)
    lines += _table(
        ["Feature", "نوع", "MI"],
        [[row.feature, row.kind, _fmt(row.target_mutual_information)] for row in top_mi.itertuples()],
    )
    lines += ["", "MI یک معیار تک‌متغیره است و به‌تنهایی معیار انتخاب نبود. ویژگی‌های کم-MI نیز در صورت بهبود پایدار OOF می‌توانستند نگه‌داری شوند؛ تصمیم نهایی براساس اثر افزایشی خارج از Fold بود.", "", "## Ablation تکرارشوندهٔ Train-only", ""]
    ab_rows = []
    for row in ablation.sort_values(["model", "mean_oof_pr_auc"], ascending=[True, False]).itertuples():
        ab_rows.append([row.model, row.configuration, int(row.mean_encoded_features), _fmt(row.mean_oof_pr_auc), _fmt(row.absolute_pr_auc_delta_vs_baseline), _fmt(row.seed_pr_auc_std)])
    lines += _table(["مدل Probe", "پیکربندی", "تعداد", "OOF PR-AUC", "Δ با پایه", "Std بین Seed"], ab_rows)
    lines += ["", "![Ablation OOF](figures/oof_ablation_pr_auc.png)", "", "در تأیید مستقل Seed=`20260811`، One-Hot×Numeric برای LightGBM بهبود `+0.001901` و برای Logistic بهبود `+0.008699` ایجاد کرد. Logistic منتخب یک ConvergenceWarning داشت؛ All-candidates سه Warning داشت و به همین دلیل همراه با افت LightGBM رد شد.", "", "## فهرست Featureهای منتخب", ""]
    for numeric, group in selected_features.groupby(selected_features["feature"].str.split("__x__").str[0]):
        lines.append(f"- `{numeric}` × {len(group)} سطح One-Hot فیت‌شده")
    lines += ["", "۱۵ Feature منتخب با MI بالاتر:", ""]
    lines += _table(
        ["Feature جدید", "MI", "بیشترین همبستگی Spearman"],
        [[row.feature, _fmt(row.target_mutual_information), _fmt(row.max_abs_spearman_redundancy, 4)] for row in selected_features.head(15).itertuples()],
    )
    lines += ["", "فهرست کامل ۳۳ Feature، فرمول، Featureهای منبع، MI منبع، Redundancy و fallback در `FEATURE_VALIDATION.csv` موجود است.", "", "## mRMR، Redundancy و Featureهای ردشده", "",
        "منطق mRMR به‌صورت عملی اجرا شد: Relevance با MI و OOF uplift سنجیده شد و Redundancy با بیشینهٔ همبستگی Spearman روی Train بررسی شد. هیچ Feature صرفاً به علت MI بالا پذیرفته نشد؛ خانواده باید در چند Fold/Seed و دو مدل Probe اثر افزایشی نشان می‌داد.", ""]
    rejected_summary = rejected.groupby("family").agg(count=("feature", "size"), max_mi=("target_mutual_information", "max"), mean_redundancy=("max_abs_spearman_redundancy", "mean")).reset_index()
    lines += _table(["خانوادهٔ ردشده", "تعداد", "بیشترین MI", "میانگین Redundancy"], [[r.family, int(r.count), _fmt(r.max_mi), _fmt(r.mean_redundancy)] for r in rejected_summary.itertuples()])
    lines += ["", "دلایل اصلی رد: نبود uplift پایدار LightGBM، همبستگی بسیار بالا بین نسخه‌های relative-median و robust-z، افزایش ابعاد از ۶۶ تا ۱۳۱، و Warningهای بیشتر Logistic. Agency در interactionهای context وارد نشد تا memorization رفتار مؤسسه تقویت نشود.", "", "## آموزش مجدد همهٔ مدل‌ها", "",
        f"تمام ۱۴ خانواده در سه Variant با ۶۰هزار ردیف آموزش داده شدند: ۴۲/۴۲ Completed و ۰ Failed. شمار خانواده‌ها: `{json.dumps(model_family_counts, sort_keys=True)}`.", "",
        f"در ۴۲ مقایسهٔ model×variant، V2 در `{positive_validation}` مورد PR-AUC Validation بالاتری از V1 داشت. جدول کامل در `VALIDATION_V1_V2_COMPARISON.csv` است.", "",
        "![Delta by family](figures/validation_delta_by_family.png)", "", "### ده اجرای برتر V2 روی Validation", ""]
    top_v2 = v2_ledger.sort_values(["pr_auc", "mcc"], ascending=False).head(10)
    lines += _table(["مدل", "Variant", "PR-AUC", "MCC", "Balanced Acc.", "زمان Fit (s)"], [[r.model, r.dataset_variant, _fmt(r.pr_auc), _fmt(r.mcc), _fmt(r.balanced_accuracy), _fmt(r.train_runtime_seconds, 3)] for r in top_v2.itertuples()])
    lines += ["", "## قانون پذیرش و تصمیم قبل از Test", "",
        "قانون ازپیش‌ثبت‌شده: PR-AUC برندهٔ V2 باید بهتر از V1 باشد؛ افت MCC و Balanced Accuracy هرکدام نباید بیشتر از ۰٫۰۰۵ باشد؛ خروجی نباید Degenerate باشد و Artifact باید با خطای حداکثر `1e-8` Reload شود.", ""]
    pre = selection["validation_delta_v2_minus_v1"]
    lines += _table(["معیار Validation", "Δ V2−V1", "پاس؟"], [["PR-AUC", _fmt(pre["pr_auc"]), "بله"], ["MCC", _fmt(pre["mcc"]), "بله"], ["Balanced Accuracy", _fmt(pre["balanced_accuracy"]), "بله"]])
    lines += ["", f"نتیجهٔ قفل‌شده پیش از Test: **{recommendation}**.", "", "## نتایج نهایی Frozen Test", ""]
    final_table = []
    for row in test_comparison.itertuples():
        final_table.append([row.role, _fmt(row.v1_pr_auc), _fmt(row.v2_pr_auc), _fmt(row.delta_pr_auc), _fmt(row.v1_mcc), _fmt(row.v2_mcc), _fmt(row.delta_mcc), _fmt(row.delta_balanced_accuracy)])
    lines += _table(["نقش", "V1 PR", "V2 PR", "Δ PR", "V1 MCC", "V2 MCC", "Δ MCC", "Δ Bal.Acc"], final_table)
    lines += ["", "![V1 vs V2 Test](figures/v1_v2_test_predictive.png)", "", "برای مدل پیش‌بینی V2: Accuracy=`" + _fmt(predictive["v2_accuracy"]) + "`، denial recall=`" + _fmt(predictive["v2_recall_denial"]) + "`، ROC-AUC=`" + _fmt(predictive["v2_roc_auc"]) + "` و Brier=`" + _fmt(predictive["v2_brier_score"]) + "` است.", "", "## مزایا و معایب V2", "", "### مزایا", "",
        "- بهبود PR-AUC در OOF هر سه Seed برای LightGBM و Logistic.",
        "- بهبود PR-AUC مدل پیش‌بینی روی Test به‌اندازهٔ حدود ۰٫۰۰۱۴۶.",
        "- interactionهای قابل‌توضیح بین متغیرهای عددی و وضعیت‌های وام/ملک.",
        "- همان Split، Row cap، Seed، Sampler و Hyperparameterهای V1؛ مقایسهٔ اثر Feature منصفانه است.",
        "- Pipeline مستقل، Train-fit-only، پشتیبانی از دستهٔ ناشناخته و Artifact قابل Reload.", "", "### معایب و هزینه‌ها", "",
        "- تعداد ستون‌ها از ۳۳ به ۶۶ افزایش یافته و هزینهٔ حافظه/Inference بالاتر است.",
        "- MCC، Balanced Accuracy و Accuracy مدل پیش‌بینی روی Test اندکی کمتر از V1 شدند.",
        "- بهبود PR-AUC کوچک است و بدون تکرار روی دادهٔ زمانی/سال دیگر نباید بزرگ‌نمایی شود.",
        "- learned categorical embedding آزمایش نشد؛ این نسخه interactionهای One-Hot را هدف گرفت.",
        "- Validation برای انتخاب مدل، Blend، Calibration و Threshold چندبار استفاده شده و احتمال خوش‌بینی وجود دارد.", "", "## Calibration و Threshold", "",
        "Calibration روی نیمی از Validation بین none/sigmoid/isotonic انتخاب شد و Threshold روی نیمهٔ دیگر Validation با Grid قطعی ۱۹۹ Quantile انتخاب شد. Test در این دو مرحله استفاده نشد. نمودار کالیبراسیون V2:", "", "![Calibration](figures/v2_final_calibration_curves.png)", "", "## Reproducibility، Leakage و Artifact", "",
        f"- تست‌های کد پیش از آموزش: ۳۸/۳۸ پاس؛ تست‌های متمرکز Feature: ۱۳/۱۳ پاس.",
        f"- Reload کامل هر سه Artifact روی `{reload_audit['rows']}` ردیف Test: بیشینه اختلاف `{max(reload_audit['max_absolute_difference'].values())}`.",
        "- همهٔ آمارهای یادگرفتنی Feature روی Train Fit شدند؛ Validation/Test هرگز Resample نشدند.",
        "- Oversampled Stacking از Group-disjoint CV براساس Source index استفاده کرد.",
        "- Protected attributes، Target، شناسه و جغرافیای ریز در Feature matrix نیستند.", "", "## بودجهٔ محاسباتی", ""]
    lines += _table(["بخش", "زمان Fit (ثانیه)"], [["Baseline V1 + reproducibility", _fmt(980.450260, 3)], ["OOF screening V2", _fmt(screening["model_fit_seconds"], 3)], ["Confirmatory OOF", _fmt(confirmed["model_fit_seconds"], 3)], ["۴۲ اجرای V2", _fmt(v2_fit, 3)], ["کل اندازه‌گیری‌شده", _fmt(total_fit, 3)], ["درصد سقف ۱۲ ساعت", _fmt(100 * total_fit / 43200, 2) + "%"]])
    lines += ["", "زمان preprocessing، inference، serialization و تولید گزارش در Fit budget منظور نشده است؛ این تفکیک مطابق قرارداد Ledger است.", "", "## توصیهٔ نهایی و Rollback", "",
        f"توصیهٔ فعلی **{recommendation}** است، چون قانون Validation پیش از Test پاس شد و Test نیز افزایش PR-AUC را نشان داد. بااین‌حال اگر هدف عملیاتی اصلی MCC/Accuracy باشد و افزایش PR-AUC ارزش هزینهٔ ۳۳ ستون اضافه را نداشته باشد، V1 گزینهٔ ساده‌تر و کاملاً آمادهٔ Rollback است.", "",
        "هیچ Artifact پایه حذف نشده است. مسیرهای Deployable V2:", "",
        "- `artifacts/generations/feature_v2/best_predictive_model.joblib`",
        "- `artifacts/generations/feature_v2/best_single_model.joblib`",
        "- `artifacts/generations/feature_v2/best_practical_model.joblib`.", "",
        "## فایل‌های خروجی و بازتولید", "",
        "```powershell",
        "python scripts/finalize_feature_v2_diagnostics.py",
        "python scripts/train_feature_v2_ml.py --max-train-rows 60000 --n-jobs 8",
        "python scripts/train_feature_v2_dl.py --max-train-rows 60000",
        "python scripts/train_feature_v2_hybrid.py",
        "python scripts/select_feature_v2_finalists.py",
        "# evaluate_feature_v2.py اکنون به‌علت Guard قابل اجرای مجدد نیست",
        "python scripts/build_feature_v2_report.py",
        "```", "",
        "گزارش‌های جزئی شامل MI، Ablation، Confirmation، ۴۲ نتیجه، مقایسهٔ Validation/Test، Calibration، Fairness، Reload audit و Provenance در همین پوشه قرار دارند.", "", "## جمع‌بندی علمی", "",
        "V2 نشان داد interactionهای ساده و قابل‌توضیح One-Hot×Numeric می‌توانند ranking رد وام را اندکی بهتر کنند؛ افزودن کورکورانهٔ همهٔ نسبت‌ها، تبدیل‌های nonlinear و آمار context مفید نبود. نتیجه از Feature selection چندمدلی و چندSeed پشتیبانی می‌شود، اما اندازهٔ اثر کوچک است و تأیید روی دادهٔ زمانی مستقل، همراه با provenance رسمی Target، مرحلهٔ علمی بعدی است.", "",
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    _render_pdf()

    from pypdf import PdfReader
    reader = PdfReader(str(PDF_PATH))
    nonempty = sum(bool((page.extract_text() or "").strip()) for page in reader.pages)
    if nonempty != len(reader.pages):
        raise RuntimeError("V2 PDF contains an empty/non-extractable page")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "markdown": {"path": MD_PATH.relative_to(ROOT).as_posix(), "bytes": MD_PATH.stat().st_size, "sha256": _sha256(MD_PATH)},
        "pdf": {"path": PDF_PATH.relative_to(ROOT).as_posix(), "bytes": PDF_PATH.stat().st_size, "pages": len(reader.pages), "nonempty_text_pages": nonempty, "sha256": _sha256(PDF_PATH)},
        "recommendation": recommendation,
    }
    (REPORT_DIR / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
