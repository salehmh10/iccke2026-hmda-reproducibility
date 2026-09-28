"""Reproducible quantitative EDA for the supplied binary HMDA extract.

This entry point is deliberately read-only with respect to raw data. It profiles
the full file, uses a deterministic exact-deduplicated view for relationships,
and writes only the EDA report and focused PNG figures.
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split


SEED = 20260809
TARGET = "loan_approved"
STRING_CODE_COLUMNS = [
    "respondent_id",
    "state_code",
    "county_code",
    "census_tract_number",
]
NUMERIC_MEASURES = [
    "loan_amount_000s",
    "applicant_income_000s",
    "population",
    "minority_population",
    "hud_median_family_income",
    "tract_to_msamd_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
]
PROTECTED_COLUMNS = [
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
]
SEMANTIC_MARKERS = {
    "not applicable": "Not applicable",
    "no co-applicant": "No co-applicant",
    "information not provided by applicant in mail, internet, or telephone application": (
        "Information not provided"
    ),
    "not available": "Not available",
    "unknown": "Unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/data.yaml"))
    parser.add_argument("--report", type=Path, default=Path("reports/EDA_REPORT.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or "raw_path" not in config:
        raise ValueError(f"Missing raw_path in {path}")
    return config


def markdown_table(frame: pd.DataFrame, *, index: bool = False) -> str:
    if frame.empty:
        return "_None._"
    return frame.to_markdown(index=index)


def compact_number(value: float | int) -> str:
    value = float(value)
    if not math.isfinite(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def column_role(column: str) -> str:
    if column == TARGET:
        return "raw target"
    if column in PROTECTED_COLUMNS:
        return "protected / audit-only"
    if column in {"respondent_id", "census_tract_number"}:
        return "identifier / audit-only"
    if column in {"state_code", "county_code"}:
        return "geographic code"
    if column in NUMERIC_MEASURES:
        return "numeric measure"
    return "categorical"


def profile_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        counts = df[column].value_counts(dropna=False)
        rows.append(
            {
                "column": column,
                "role": column_role(column),
                "dtype": str(df[column].dtype),
                "unique": int(df[column].nunique(dropna=False)),
                "machine/blank missing": int(df[column].isna().sum())
                + int((df[column].astype("string").str.strip() == "").sum()),
                "top share %": round(100 * counts.iloc[0] / len(df), 3),
            }
        )
    return pd.DataFrame(rows)


def semantic_missingness(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.select_dtypes(include=["object", "string"]).columns:
        normalized = df[column].astype("string").str.strip().str.casefold()
        for marker, display in SEMANTIC_MARKERS.items():
            count = int(normalized.eq(marker).sum())
            if count:
                rows.append(
                    {
                        "column": column,
                        "semantic marker": display,
                        "count": count,
                        "rate %": round(100 * count / len(df), 3),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["column", "semantic marker", "count", "rate %"])
    return pd.DataFrame(rows).sort_values(["count", "column"], ascending=[False, True])


def numeric_quantiles(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    quantiles = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]
    values = df[list(columns)].quantile(quantiles).T
    values.columns = ["min", "p01", "p25", "median", "p75", "p99", "max"]
    return values.reset_index(names="feature").round(3)


def outlier_profile(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        series = df[column].astype(float)
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "feature": column,
                "IQR lower": round(float(lower), 3),
                "IQR upper": round(float(upper), 3),
                "outside count": count,
                "outside %": round(100 * count / len(df), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("outside %", ascending=False)


def validity_profile(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "negative": int((series < 0).sum()),
                "zero": int((series == 0).sum()),
                "non-finite": int((~np.isfinite(series)).sum()),
            }
        )
    return pd.DataFrame(rows)


def categorical_relationships(
    df: pd.DataFrame, columns: Iterable[str], min_support: int = 500
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summaries: list[dict[str, Any]] = []
    details: dict[str, pd.DataFrame] = {}
    for column in columns:
        grouped = (
            df.groupby(column, dropna=False, observed=True)["target_denied"]
            .agg(["size", "mean"])
            .rename(columns={"size": "count", "mean": "denial_rate"})
            .reset_index()
        )
        eligible = grouped[grouped["count"] >= min_support].copy()
        details[column] = eligible
        if eligible.empty:
            continue
        weights = eligible["count"] / eligible["count"].sum()
        weighted_mean = float(np.average(eligible["denial_rate"], weights=weights))
        weighted_sd = float(
            np.sqrt(np.average((eligible["denial_rate"] - weighted_mean) ** 2, weights=weights))
        )
        summaries.append(
            {
                "feature": column,
                "all levels": int(grouped.shape[0]),
                f"levels n>={min_support}": int(eligible.shape[0]),
                "min denial %": round(100 * eligible["denial_rate"].min(), 2),
                "max denial %": round(100 * eligible["denial_rate"].max(), 2),
                "spread pp": round(
                    100 * (eligible["denial_rate"].max() - eligible["denial_rate"].min()), 2
                ),
                "weighted SD pp": round(100 * weighted_sd, 2),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("spread pp", ascending=False)
    return summary, details


def numeric_target_relationships(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        approved = df.loc[df["target_denied"] == 0, column].astype(float)
        denied = df.loc[df["target_denied"] == 1, column].astype(float)
        pooled_sd = math.sqrt((approved.var(ddof=1) + denied.var(ddof=1)) / 2)
        smd = (denied.mean() - approved.mean()) / pooled_sd if pooled_sd else 0.0
        rows.append(
            {
                "feature": column,
                "approval median": round(float(approved.median()), 3),
                "denial median": round(float(denied.median()), 3),
                "approval mean": round(float(approved.mean()), 3),
                "denial mean": round(float(denied.mean()), 3),
                "SMD (denial-approval)": round(float(smd), 4),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "SMD (denial-approval)", key=lambda s: s.abs(), ascending=False
    )


def population_stability_index(reference: pd.Series, comparison: pd.Series) -> float:
    values = pd.concat([reference, comparison], ignore_index=True).dropna().astype(float)
    edges = np.unique(values.quantile(np.linspace(0, 1, 11)).to_numpy())
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = pd.cut(reference, edges, include_lowest=True).value_counts(sort=False)
    cmp_counts = pd.cut(comparison, edges, include_lowest=True).value_counts(sort=False)
    eps = 1e-6
    ref_prop = np.clip(ref_counts.to_numpy() / len(reference), eps, None)
    cmp_prop = np.clip(cmp_counts.to_numpy() / len(comparison), eps, None)
    return float(np.sum((cmp_prop - ref_prop) * np.log(cmp_prop / ref_prop)))


def row_order_shift(
    df: pd.DataFrame, numeric_columns: Iterable[str], categorical_columns: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment_size = max(1, len(df) // 5)
    first = df.iloc[:segment_size]
    last = df.iloc[-segment_size:]
    numeric_rows: list[dict[str, Any]] = []
    for column in numeric_columns:
        statistic = ks_2samp(first[column], last[column], method="asymp").statistic
        numeric_rows.append(
            {
                "feature": column,
                "KS first-vs-last": round(float(statistic), 4),
                "PSI first-vs-last": round(
                    population_stability_index(first[column], last[column]), 4
                ),
            }
        )
    categorical_rows: list[dict[str, Any]] = []
    for column in categorical_columns:
        p = first[column].value_counts(normalize=True, dropna=False)
        q = last[column].value_counts(normalize=True, dropna=False)
        levels = p.index.union(q.index)
        tv = 0.5 * float((p.reindex(levels, fill_value=0) - q.reindex(levels, fill_value=0)).abs().sum())
        categorical_rows.append(
            {"feature": column, "total variation first-vs-last": round(tv, 4)}
        )
    numeric = pd.DataFrame(numeric_rows).sort_values("KS first-vs-last", ascending=False)
    categorical = pd.DataFrame(categorical_rows).sort_values(
        "total variation first-vs-last", ascending=False
    )
    return numeric, categorical


def prospective_group_overlap(df: pd.DataFrame) -> dict[str, float | int]:
    indices = np.arange(len(df))
    y = df["target_denied"].to_numpy()
    train_idx, holdout_idx = train_test_split(
        indices, test_size=0.40, random_state=SEED, stratify=y
    )
    validation_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=0.50,
        random_state=SEED,
        stratify=y[holdout_idx],
    )
    result: dict[str, float | int] = {
        "train rows": len(train_idx),
        "validation rows": len(validation_idx),
        "test rows": len(test_idx),
        "train denial %": 100 * float(y[train_idx].mean()),
        "validation denial %": 100 * float(y[validation_idx].mean()),
        "test denial %": 100 * float(y[test_idx].mean()),
    }
    for column in ["respondent_id", "census_tract_number", "county_name", "state_name"]:
        train_groups = set(df.iloc[train_idx][column])
        result[f"test {column} seen in train %"] = 100 * float(
            df.iloc[test_idx][column].isin(train_groups).mean()
        )
    return result


def plot_target(df: pd.DataFrame, output: Path) -> None:
    counts = df["target_denied"].value_counts().sort_index()
    labels = ["Approval (0)", "Denial (1)"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#E45756"])
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count:,}\n({100 * count / len(df):.2f}%)",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Rows")
    ax.set_title("Analytical target distribution (exact-deduplicated view)")
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, output)


def plot_semantic_missing(table: pd.DataFrame, row_count: int, output: Path) -> None:
    by_column = table.groupby("column", as_index=False)["count"].sum().nlargest(12, "count")
    by_column["rate"] = 100 * by_column["count"] / row_count
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=by_column, y="column", x="rate", color="#F2CF5B", ax=ax)
    ax.set_xlabel("Rows with enumerated semantic-missing labels (%)")
    ax.set_ylabel("")
    ax.set_title("Structural / semantic missingness (not machine nulls)")
    save_figure(fig, output)


def plot_numeric_distributions(df: pd.DataFrame, output: Path) -> None:
    sampled = df.sample(n=min(100_000, len(df)), random_state=SEED)
    features = [
        "loan_amount_000s",
        "applicant_income_000s",
        "population",
        "minority_population",
        "tract_to_msamd_income",
        "number_of_1_to_4_family_units",
    ]
    log_features = {"loan_amount_000s", "applicant_income_000s", "population", "number_of_1_to_4_family_units"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.flat, features):
        values = sampled[feature].astype(float)
        label = feature
        if feature in log_features:
            values = np.log1p(values)
            label = f"log1p({feature})"
        sns.histplot(values, bins=50, color="#4C78A8", ax=ax)
        ax.set_xlabel(label)
        ax.set_ylabel("Sample rows")
    fig.suptitle("Numeric distributions (deterministic 100k display sample)", y=1.01)
    save_figure(fig, output)


def plot_correlations(df: pd.DataFrame, columns: list[str], output: Path) -> pd.DataFrame:
    correlations = df[columns + ["target_denied"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlations,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Spearman rank correlations (full deduplicated data)")
    save_figure(fig, output)
    return correlations


def plot_categorical_rates(
    details: dict[str, pd.DataFrame],
    columns: list[str],
    output: Path,
    title: str,
    overall_rate: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, column in zip(axes.flat, columns):
        data = details[column].nlargest(12, "count").copy()
        data[column] = data[column].astype(str).map(lambda x: textwrap.shorten(x, width=40))
        data = data.sort_values("denial_rate")
        sns.barplot(data=data, x="denial_rate", y=column, color="#E45756", ax=ax)
        ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Denial rate")
        ax.set_ylabel("")
        ax.set_title(column)
        ax.xaxis.set_major_formatter(lambda x, _pos: f"{100*x:.0f}%")
    fig.suptitle(title, y=1.01)
    save_figure(fig, output)


def plot_protected_rates(
    details: dict[str, pd.DataFrame], output: Path, overall_rate: float
) -> None:
    columns = ["applicant_ethnicity_name", "applicant_race_name_1", "applicant_sex_name"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, column in zip(axes.flat, columns):
        data = details[column].nlargest(12, "count").copy()
        data[column] = data[column].astype(str).map(lambda x: textwrap.shorten(x, width=38))
        data = data.sort_values("denial_rate")
        sns.barplot(data=data, x="denial_rate", y=column, color="#B279A2", ax=ax)
        ax.axvline(overall_rate, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Denial rate")
        ax.set_ylabel("")
        ax.set_title(column)
        ax.xaxis.set_major_formatter(lambda x, _pos: f"{100*x:.0f}%")
    fig.suptitle("Protected attributes: descriptive audit only (not model features)", y=1.01)
    save_figure(fig, output)


def plot_numeric_deciles(df: pd.DataFrame, output: Path) -> None:
    features = [
        "loan_amount_000s",
        "applicant_income_000s",
        "loan_to_income",
        "minority_population",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, feature in zip(axes.flat, features):
        bins = pd.qcut(df[feature], q=10, duplicates="drop")
        grouped = (
            df.groupby(bins, observed=True)["target_denied"]
            .agg(["size", "mean"])
            .reset_index(drop=True)
        )
        ax.plot(np.arange(1, len(grouped) + 1), grouped["mean"], marker="o", color="#E45756")
        ax.axhline(df["target_denied"].mean(), color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Within-sample quantile bin (low to high)")
        ax.set_ylabel("Denial rate")
        ax.set_title(feature)
        ax.yaxis.set_major_formatter(lambda x, _pos: f"{100*x:.0f}%")
    fig.suptitle("Numeric-target relationships (descriptive, full deduplicated data)", y=1.01)
    save_figure(fig, output)


def plot_row_order_proxy(df: pd.DataFrame, output: Path) -> pd.DataFrame:
    block = np.minimum((np.arange(len(df)) * 10 // len(df)) + 1, 10)
    temp = pd.DataFrame(
        {
            "block": block,
            "target_denied": df["target_denied"].to_numpy(),
            "loan_amount_000s": df["loan_amount_000s"].to_numpy(),
            "applicant_income_000s": df["applicant_income_000s"].to_numpy(),
        }
    )
    summary = temp.groupby("block").agg(
        rows=("target_denied", "size"),
        denial_rate=("target_denied", "mean"),
        loan_median=("loan_amount_000s", "median"),
        income_median=("applicant_income_000s", "median"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(summary.index, summary["denial_rate"], marker="o", color="#E45756")
    axes[0].set_ylabel("Denial rate")
    axes[0].yaxis.set_major_formatter(lambda x, _pos: f"{100*x:.1f}%")
    axes[0].set_title("Target")
    axes[1].plot(summary.index, summary["loan_median"], marker="o", color="#4C78A8")
    axes[1].set_ylabel("Loan amount median ($000s)")
    axes[1].set_title("Loan amount")
    axes[2].plot(summary.index, summary["income_median"], marker="o", color="#59A14F")
    axes[2].set_ylabel("Applicant income median ($000s)")
    axes[2].set_title("Applicant income")
    for ax in axes:
        ax.set_xlabel("Contiguous source-row block")
        ax.set_xticks(range(1, 11))
    fig.suptitle("Row-order stability proxy — source order is not time", y=1.02)
    save_figure(fig, output)
    return summary.reset_index()


def build_report(
    *,
    raw_path: Path,
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    schema: pd.DataFrame,
    semantic: pd.DataFrame,
    quantiles: pd.DataFrame,
    outliers: pd.DataFrame,
    validity: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    numeric_target: pd.DataFrame,
    correlations: pd.DataFrame,
    shift_numeric: pd.DataFrame,
    shift_categorical: pd.DataFrame,
    block_summary: pd.DataFrame,
    split_proxy: dict[str, float | int],
    figure_names: list[str],
) -> str:
    raw_counts = raw_df[TARGET].value_counts().sort_index()
    counts = df["target_denied"].value_counts().sort_index()
    duplicate_excess = int(raw_df.duplicated(keep="first").sum())
    duplicate_rows = int(raw_df.duplicated(keep=False).sum())
    duplicate_groups = int(raw_df.loc[raw_df.duplicated(keep=False)].drop_duplicates().shape[0])
    constants = schema.loc[schema["unique"] == 1, "column"].tolist()
    near_constants = schema.loc[
        (schema["unique"] > 1) & (schema["top share %"] >= 95),
        ["column", "unique", "top share %"],
    ]
    high_cardinality = schema.loc[
        (schema["role"].isin(["categorical", "geographic code", "identifier / audit-only"]))
        & (schema["unique"] > 100),
        ["column", "role", "unique", "top share %"],
    ].sort_values("unique", ascending=False)
    machine_missing = int(schema["machine/blank missing"].sum())
    target_ratio = counts.loc[0] / counts.loc[1]
    top_corr = (
        correlations["target_denied"]
        .drop("target_denied")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .rename("Spearman with denial")
        .rename_axis("feature")
        .reset_index()
        .head(8)
        .round(4)
    )
    feature_corr = correlations.drop(index="target_denied", columns="target_denied")
    corr_pairs: list[dict[str, Any]] = []
    for left_idx, left in enumerate(feature_corr.columns):
        for right in feature_corr.columns[left_idx + 1 :]:
            corr_pairs.append(
                {"pair": f"{left} / {right}", "Spearman": feature_corr.loc[left, right]}
            )
    top_pairs = (
        pd.DataFrame(corr_pairs)
        .sort_values("Spearman", key=lambda s: s.abs(), ascending=False)
        .head(8)
        .round(4)
    )
    protected_summary = categorical_summary[
        categorical_summary["feature"].isin(PROTECTED_COLUMNS)
    ]
    nonprotected_summary = categorical_summary[
        ~categorical_summary["feature"].isin(PROTECTED_COLUMNS)
    ]
    raw_target_table = pd.DataFrame(
        [
            {
                "meaning": "Approval",
                "raw loan_approved": 1,
                "raw rows": int(raw_counts.loc[1]),
                "raw %": round(100 * raw_counts.loc[1] / len(raw_df), 4),
                "analytical target_denied": 0,
            },
            {
                "meaning": "Denial",
                "raw loan_approved": 0,
                "raw rows": int(raw_counts.loc[0]),
                "raw %": round(100 * raw_counts.loc[0] / len(raw_df), 4),
                "analytical target_denied": 1,
            },
        ]
    )
    split_rows = pd.DataFrame(
        [{"diagnostic": key, "value": compact_number(value)} for key, value in split_proxy.items()]
    )
    source_order_range = pd.DataFrame(
        [
            {
                "metric across 10 contiguous row blocks": "denial rate",
                "minimum": f"{100 * block_summary['denial_rate'].min():.3f}%",
                "maximum": f"{100 * block_summary['denial_rate'].max():.3f}%",
                "range": f"{100 * (block_summary['denial_rate'].max() - block_summary['denial_rate'].min()):.3f} pp",
            },
            {
                "metric across 10 contiguous row blocks": "loan median ($000s)",
                "minimum": compact_number(block_summary["loan_median"].min()),
                "maximum": compact_number(block_summary["loan_median"].max()),
                "range": compact_number(block_summary["loan_median"].max() - block_summary["loan_median"].min()),
            },
            {
                "metric across 10 contiguous row blocks": "income median ($000s)",
                "minimum": compact_number(block_summary["income_median"].min()),
                "maximum": compact_number(block_summary["income_median"].max()),
                "range": compact_number(block_summary["income_median"].max() - block_summary["income_median"].min()),
            },
        ]
    )
    income_flags = pd.DataFrame(
        [
            {
                "flag": "applicant_income_000s == 9999",
                "count": int((df["applicant_income_000s"] == 9999).sum()),
                "rate %": round(100 * (df["applicant_income_000s"] == 9999).mean(), 4),
            },
            {
                "flag": "applicant_income_000s > 9999",
                "count": int((df["applicant_income_000s"] > 9999).sum()),
                "rate %": round(100 * (df["applicant_income_000s"] > 9999).mean(), 4),
            },
            {
                "flag": "loan_to_income > 10",
                "count": int((df["loan_to_income"] > 10).sum()),
                "rate %": round(100 * (df["loan_to_income"] > 10).mean(), 4),
            },
        ]
    )
    plausibility_flags = pd.DataFrame(
        [
            {
                "check": "minority_population outside [0,100]",
                "count": int(
                    (
                        (df["minority_population"] < 0)
                        | (df["minority_population"] > 100)
                    ).sum()
                ),
                "rate %": round(
                    100
                    * (
                        (df["minority_population"] < 0)
                        | (df["minority_population"] > 100)
                    ).mean(),
                    4,
                ),
                "status": "invalid if nonzero",
            },
            {
                "check": "owner-occupied units > 1-to-4-family units",
                "count": int(
                    (
                        df["number_of_owner_occupied_units"]
                        > df["number_of_1_to_4_family_units"]
                    ).sum()
                ),
                "rate %": round(
                    100
                    * (
                        df["number_of_owner_occupied_units"]
                        > df["number_of_1_to_4_family_units"]
                    ).mean(),
                    4,
                ),
                "status": "review only; concepts have different denominators",
            },
        ]
    )

    lines = [
        "# HMDA Exploratory Data Analysis",
        "",
        "## Scope and reproducibility",
        "",
        f"- Input: `{raw_path.as_posix()}` (repository-relative).",
        f"- Raw shape: **{len(raw_df):,} rows x {raw_df.shape[1]} columns**.",
        f"- Exact-deduplicated analytical view: **{len(df):,} rows x {raw_df.shape[1]} raw columns**, plus derived `target_denied` and `loan_to_income` used only for EDA.",
        f"- Seed: `{SEED}`. Command: `python scripts/run_eda.py --config config/data.yaml`.",
        "- All tables are computed on the complete file. Only distribution plots use a deterministic 100,000-row display sample where stated.",
        "- No model was trained; no preprocessing statistic, sampler, split artifact, calibrator, or threshold was fit or saved.",
        "",
        "## Target semantics, prevalence, and imbalance",
        "",
        "The supplied binary field is interpreted under the documented project assumption `loan_approved=1 -> Approval`, `loan_approved=0 -> Denial`; analytical positive class is `target_denied=1`. The source action-code derivation is unavailable, so this is not a recovered official HMDA action mapping.",
        "",
        markdown_table(raw_target_table),
        "",
        f"After exact deduplication, there are **{int(counts.loc[0]):,} approvals** and **{int(counts.loc[1]):,} denials**; approval:denial is **{target_ratio:.4f}:1** and denial prevalence is **{100 * counts.loc[1] / len(df):.4f}%**.",
        "",
        "**Interpretation:** Accuracy alone can look strong for a majority-only rule, so it is not an adequate selection metric.",
        "",
        "**Action:** Preserve natural validation/test prevalence, apply imbalance methods only to training, and emphasize denial PR-AUC, both class recalls, balanced accuracy, macro-F1, and MCC.",
        "",
        "![Target distribution](figures/target_distribution.png)",
        "",
        "## Schema, types, uniqueness, and dominance",
        "",
        markdown_table(schema),
        "",
        f"- Constant columns: `{constants}`.",
        f"- Near-constant rule: top category >=95% but <100%; found **{len(near_constants)}**.",
        "",
        markdown_table(near_constants),
        "",
        "High-cardinality fields (>100 observed levels):",
        "",
        markdown_table(high_cardinality),
        "",
        "**Interpretation:** `respondent_id`, census tract, and geographic codes are labels, not magnitudes. Their high cardinality can memorize lender/geography and create optimistic row-random estimates. Near-constant fields, if any, offer little variation but are not removed by EDA alone.",
        "",
        "**Action:** Keep identifier/fine-geography fields audit-only for the primary benchmark, treat codes as strings, and let P05/P07 explicitly disposition every high-cardinality or near-constant feature. Any learned rare-category/frequency rule must be fit on training only.",
        "",
        "## Machine missingness and semantic missingness",
        "",
        f"Machine null/blank cells across the parsed schema: **{machine_missing:,}**. This does not mean the data are substantively complete; enumerated nonresponses are present:",
        "",
        markdown_table(semantic),
        "",
        "**Interpretation:** `Not applicable`, `No co-applicant`, and `Information not provided` have different meanings. Collapsing them into one generic missing value would erase application structure and may distort protected-group audits.",
        "",
        "**Action:** Preserve structural labels explicitly or map them to documented indicators within a train-fitted encoder; never compute a global imputation rule before splitting.",
        "",
        "![Semantic missingness](figures/semantic_missingness.png)",
        "",
        "## Exact duplicates",
        "",
        f"There are **{duplicate_excess:,} exact repeats beyond the first**, covering **{duplicate_rows:,} rows in {duplicate_groups:,} distinct repeated vectors**. The analytical EDA relationship tables use `drop_duplicates(keep='first')`; the raw file remains unchanged.",
        "",
        "**Interpretation:** Identical published vectors can land on both sides of a row-random split and contaminate evaluation, although removed application identifiers mean two identical rows could still represent distinct applications.",
        "",
        "**Action:** Deduplicate the modeling view before splitting (or enforce duplicate-group splitting), record removed row indices/hash, and do not delete or overwrite raw data.",
        "",
        "## Numeric ranges, impossible values, quantiles, and outliers",
        "",
        "Full-data quantiles (`*_000s` are thousands of dollars):",
        "",
        markdown_table(quantiles),
        "",
        "Non-negativity and finite-value checks:",
        "",
        markdown_table(validity),
        "",
        "Domain/range plausibility flags:",
        "",
        markdown_table(plausibility_flags),
        "",
        "IQR flags are descriptive, not deletion rules:",
        "",
        markdown_table(outliers),
        "",
        "Additional tail/coding flags:",
        "",
        markdown_table(income_flags),
        "",
        "**Interpretation:** No blanket negative-value failure is evident, but financial variables are strongly right-skewed and IQR flags are common. Extreme values can be legitimate, privacy-modified, capped, or coding artifacts; IQR alone cannot decide. Owner-unit/family-unit flags are plausibility prompts, not proven errors because the published concepts have different denominators.",
        "",
        "**Action:** Verify special-value/top-coding semantics against lineage if available. For linear/DL models, evaluate `log1p`, robust scaling, and train-fitted clipping; retain raw values for tree models and do not delete tails without a documented rule learned on training only.",
        "",
        "![Numeric distributions](figures/numeric_distributions.png)",
        "",
        "## Correlations",
        "",
        "Strongest feature-feature rank correlations:",
        "",
        markdown_table(top_pairs),
        "",
        "Rank correlations with denial:",
        "",
        markdown_table(top_corr),
        "",
        "**Interpretation:** Tract population/housing quantities may encode overlapping constructs, while marginal correlation with denial is neither causality nor a leakage test. Nonlinear and category interactions can exist despite small pairwise coefficients.",
        "",
        "**Action:** Check redundant ratios/features during P07 and use regularization or tree robustness as appropriate. P05 must independently assess leakage; do not drop a feature solely because its marginal correlation is large or small.",
        "",
        "![Correlation heatmap](figures/correlation_heatmap.png)",
        "",
        "## Numeric-target relationships",
        "",
        markdown_table(numeric_target),
        "",
        "`loan_to_income = loan_amount_000s / applicant_income_000s`; all denominators are positive in this file. SMD is a descriptive standardized difference (denial minus approval), not a causal effect.",
        "",
        "**Interpretation:** Class medians and quantile-bin rates show which raw financial/context variables have univariate separation and whether that separation is monotone. Means are more tail-sensitive than medians.",
        "",
        "**Action:** Carry only domain-supported ratios into P07, specify divide-by-zero handling, and validate incremental utility out of sample. Fit all binning/scaling boundaries on training only.",
        "",
        "![Numeric target relationships](figures/numeric_target_deciles.png)",
        "",
        "## Categorical-target relationships",
        "",
        "Rates below use only levels with at least 500 rows. The spread is descriptive and is not adjusted for other covariates.",
        "",
        "Non-protected fields:",
        "",
        markdown_table(nonprotected_summary.head(18)),
        "",
        "Protected fields (audit-only):",
        "",
        markdown_table(protected_summary),
        "",
        "**Interpretation:** Large raw rate spreads can reflect product mix, lender/geographic composition, missing-information structure, or sampling—not causal treatment. ID/geography spreads are especially vulnerable to memorization. Protected-group differences are screening signals for a later fairness audit, not proof of discrimination.",
        "",
        "**Action:** Exclude protected attributes from primary prediction, retain them for audit, exclude direct identifiers/fine geography by default, and require P05 to review suspicious near-pure groups or outcome proxies before training.",
        "",
        "![Loan categorical rates](figures/categorical_denial_rates.png)",
        "",
        "![Protected descriptive rates](figures/protected_attribute_denial_rates.png)",
        "",
        "## Temporal limitation and shift proxies",
        "",
        "There is **no year, date, month, quarter, period, or timestamp field** among the 29 supplied columns. Therefore class distribution by year, approval/denial trends through time, temporal drift, and a forward temporal split are **not executable** from this file. The requested approximate 2006–2017 coverage cannot be verified.",
        "",
        "The following diagnostics compare the first and last 20% of source-row order. Source-row order has no documented time meaning, so these are only ingestion/order stability proxies:",
        "",
        markdown_table(source_order_range),
        "",
        "Numeric first-vs-last row-order proxy:",
        "",
        markdown_table(shift_numeric),
        "",
        "Categorical first-vs-last row-order proxy (largest total-variation distances; sparse respondent/tract IDs excluded):",
        "",
        markdown_table(shift_categorical.head(12)),
        "",
        "**Interpretation:** Low row-order shift would only show that this extract is well mixed in file order; high shift would flag an undocumented ordering/source mixture. Neither result establishes temporal stability or future generalization.",
        "",
        "**Action:** Do not relabel row blocks as years. Use the documented stratified-random fallback for the benchmark, limit claims to within-sample generalization, and add true temporal validation only if a dated source extract/manifest is supplied.",
        "",
        "![Row-order proxy](figures/row_order_shift_proxy.png)",
        "",
        "## Prospective split and entity-overlap caveat",
        "",
        "This is a diagnostic simulation of the configured 60/20/20 stratified-random policy, not a persisted P06 split:",
        "",
        markdown_table(split_rows),
        "",
        "**Interpretation:** Stratification keeps class prevalence nearly identical, but high test-group overlap with training means a row-random evaluation can partially measure generalization to new applications from already-seen lenders/areas rather than unseen entities. Conversely, an entity holdout may change both geography and product mix.",
        "",
        "**Action:** P06 should persist one deduplicated stratified split and additionally report lender/tract overlap. A respondent-group or geography holdout is a sensitivity analysis—not a substitute for missing temporal validation—and must preserve both classes.",
        "",
        "## Potential leakage and proxy candidates for P05",
        "",
        "- `loan_approved` is the raw target and must never enter predictors; `target_denied` is derived only by the target adapter.",
        "- Exact duplicates must not cross evaluation boundaries.",
        "- `respondent_id`, census tract, MSA, county, and associated codes can memorize lender/place; fine geography also acts as a demographic proxy.",
        "- Direct race, ethnicity, and sex columns are audit-only. `minority_population` is also a strong protected/geographic proxy and is excluded from the primary feature policy.",
        "- `preapproval_name` describes preapproval request/applicability in this schema; its timing/derivation still requires P05 review before predictive use.",
        "",
        "**Interpretation:** EDA can nominate suspicious predictors but cannot certify when a field was known relative to the decision or how the transformed target was derived.",
        "",
        "**Action:** Block serious training until P05 records an explicit keep/exclude disposition and verifies train-only fit boundaries.",
        "",
        "## Artifacts and limitations",
        "",
        "Generated figures:",
        "",
        *[f"- `reports/figures/{name}`" for name in figure_names],
        "",
        "Limitations: the file is a transformed, stratified sample with no sampling weights, date, original action code, or derivation manifest. It is not population-representative evidence; associations are descriptive, privacy modification may affect values, and HMDA lacks major underwriting determinants. No causal, legal, fairness-compliance, production-credit, or future-time claim is supported.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_path = Path(config["raw_path"])
    if not raw_path.is_absolute():
        raw_path = Path.cwd() / raw_path
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    dtype_map = {column: "string" for column in STRING_CODE_COLUMNS}
    raw_df = pd.read_csv(raw_path, dtype=dtype_map, keep_default_na=False, low_memory=False)
    if TARGET not in raw_df.columns or set(raw_df[TARGET].unique()) != {0, 1}:
        raise ValueError(f"Expected binary {TARGET} with values {{0,1}}")
    if len(raw_df) != 500_000:
        raise ValueError(f"Expected 500,000 rows, found {len(raw_df):,}")

    schema = profile_schema(raw_df)
    semantic = semantic_missingness(raw_df)
    df = raw_df.drop_duplicates(keep="first").copy()
    df["target_denied"] = 1 - df[TARGET]
    if (df["applicant_income_000s"] <= 0).any():
        raise ValueError("loan_to_income denominator is not strictly positive")
    df["loan_to_income"] = df["loan_amount_000s"] / df["applicant_income_000s"]

    numeric_for_analysis = NUMERIC_MEASURES + ["loan_to_income"]
    quantiles = numeric_quantiles(df, numeric_for_analysis)
    outliers = outlier_profile(df, numeric_for_analysis)
    validity = validity_profile(df, NUMERIC_MEASURES)
    categorical_columns = [
        column
        for column in raw_df.columns
        if column not in NUMERIC_MEASURES + [TARGET]
    ]
    categorical_summary, categorical_details = categorical_relationships(
        df, categorical_columns
    )
    numeric_target = numeric_target_relationships(df, numeric_for_analysis)
    shift_numeric, shift_categorical = row_order_shift(
        df,
        numeric_for_analysis + ["target_denied"],
        [column for column in categorical_columns if column not in {"respondent_id", "census_tract_number"}],
    )
    split_proxy = prospective_group_overlap(df)

    sns.set_theme(style="whitegrid", context="notebook")
    figures = [
        "target_distribution.png",
        "semantic_missingness.png",
        "numeric_distributions.png",
        "correlation_heatmap.png",
        "categorical_denial_rates.png",
        "protected_attribute_denial_rates.png",
        "numeric_target_deciles.png",
        "row_order_shift_proxy.png",
    ]
    plot_target(df, args.figures / figures[0])
    plot_semantic_missing(semantic, len(raw_df), args.figures / figures[1])
    plot_numeric_distributions(df, args.figures / figures[2])
    correlations = plot_correlations(df, numeric_for_analysis, args.figures / figures[3])
    plot_categorical_rates(
        categorical_details,
        ["loan_type_name", "loan_purpose_name", "preapproval_name", "lien_status_name"],
        args.figures / figures[4],
        "Loan/application categorical denial rates (groups n>=500)",
        float(df["target_denied"].mean()),
    )
    plot_protected_rates(
        categorical_details, args.figures / figures[5], float(df["target_denied"].mean())
    )
    plot_numeric_deciles(df, args.figures / figures[6])
    block_summary = plot_row_order_proxy(df, args.figures / figures[7])

    report = build_report(
        raw_path=raw_path.relative_to(Path.cwd()),
        raw_df=raw_df,
        df=df,
        schema=schema,
        semantic=semantic,
        quantiles=quantiles,
        outliers=outliers,
        validity=validity,
        categorical_summary=categorical_summary,
        numeric_target=numeric_target,
        correlations=correlations,
        shift_numeric=shift_numeric,
        shift_categorical=shift_categorical,
        block_summary=block_summary,
        split_proxy=split_proxy,
        figure_names=figures,
    )
    args.report.write_text(report, encoding="utf-8")
    for figure in figures:
        path = args.figures / figure
        if not path.is_file() or path.stat().st_size < 10_000:
            raise RuntimeError(f"Missing or unexpectedly small plot: {path}")
    print(
        f"EDA complete: raw_rows={len(raw_df)}, analytical_rows={len(df)}, "
        f"report={args.report}, figures={len(figures)}"
    )


if __name__ == "__main__":
    main()
