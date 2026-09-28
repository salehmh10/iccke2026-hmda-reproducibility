"""Frozen Prompt 1B feature formulas for the final selected rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


NUMERIC_ENGINEERED_FEATURES = [
    "log1p_applicant_income",
    "log1p_population",
    "log1p_hud_median_family_income",
    "log1p_owner_occupied_units",
    "log1p_1_to_4_family_units",
    "applicant_income_to_area_income",
    "tract_income_ratio",
    "owner_occupied_unit_ratio",
    "family_units_per_1000_people",
    "owner_occupied_units_per_1000_people",
]

CATEGORICAL_ENGINEERED_FEATURES = [
    "has_co_applicant",
    "loan_program_group",
    "applicant_income_area_group",
    "tract_income_level",
    "us_region",
    "majority_minority_tract",
]

ENGINEERED_FEATURES = NUMERIC_ENGINEERED_FEATURES + CATEGORICAL_ENGINEERED_FEATURES

STATE_TO_REGION = {
    "Connecticut": "Northeast", "Maine": "Northeast", "Massachusetts": "Northeast",
    "New Hampshire": "Northeast", "Rhode Island": "Northeast", "Vermont": "Northeast",
    "New Jersey": "Northeast", "New York": "Northeast", "Pennsylvania": "Northeast",
    "Illinois": "Midwest", "Indiana": "Midwest", "Michigan": "Midwest", "Ohio": "Midwest",
    "Wisconsin": "Midwest", "Iowa": "Midwest", "Kansas": "Midwest", "Minnesota": "Midwest",
    "Missouri": "Midwest", "Nebraska": "Midwest", "North Dakota": "Midwest",
    "South Dakota": "Midwest", "Delaware": "South", "District of Columbia": "South",
    "Florida": "South", "Georgia": "South", "Maryland": "South", "North Carolina": "South",
    "South Carolina": "South", "Virginia": "South", "West Virginia": "South",
    "Alabama": "South", "Kentucky": "South", "Mississippi": "South", "Tennessee": "South",
    "Arkansas": "South", "Louisiana": "South", "Oklahoma": "South", "Texas": "South",
    "Arizona": "West", "Colorado": "West", "Idaho": "West", "Montana": "West",
    "Nevada": "West", "New Mexico": "West", "Utah": "West", "Wyoming": "West",
    "Alaska": "West", "California": "West", "Hawaii": "West", "Oregon": "West",
    "Washington": "West", "Puerto Rico": "Other",
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide numeric series and return missing values for unsafe results."""
    result = numerator.astype("float64").divide(denominator.astype("float64"))
    return result.where(np.isfinite(result), np.nan)


def _income_group(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 0.5, 0.8, 1.2, np.inf],
        labels=["Very low", "Low", "Moderate", "High"],
        include_lowest=True,
    ).astype("string")


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the exact 16 frozen engineered features."""
    result = frame.copy()
    result["log1p_applicant_income"] = np.log1p(result["applicant_income_000s"])
    result["log1p_population"] = np.log1p(result["population"])
    result["log1p_hud_median_family_income"] = np.log1p(result["hud_median_family_income"])
    result["log1p_owner_occupied_units"] = np.log1p(result["number_of_owner_occupied_units"])
    result["log1p_1_to_4_family_units"] = np.log1p(result["number_of_1_to_4_family_units"])
    result["applicant_income_to_area_income"] = safe_divide(
        result["applicant_income_000s"], result["hud_median_family_income"] / 1000.0
    )
    result["tract_income_ratio"] = result["tract_to_msamd_income"] / 100.0
    result["owner_occupied_unit_ratio"] = safe_divide(
        result["number_of_owner_occupied_units"], result["number_of_1_to_4_family_units"]
    )
    result["family_units_per_1000_people"] = safe_divide(
        result["number_of_1_to_4_family_units"], result["population"]
    ) * 1000.0
    result["owner_occupied_units_per_1000_people"] = safe_divide(
        result["number_of_owner_occupied_units"], result["population"]
    ) * 1000.0
    result["has_co_applicant"] = (~result["co_applicant_sex_name"].astype("string").str.contains(
        "No co-applicant", case=False, na=False
    )).astype("int8")
    result["loan_program_group"] = np.where(
        result["loan_type_name"].astype("string").str.contains("FHA|VA|FSA|RHS", case=False, na=False),
        "Government backed", "Conventional",
    )
    result["applicant_income_area_group"] = _income_group(result["applicant_income_to_area_income"])
    result["tract_income_level"] = _income_group(result["tract_income_ratio"])
    result["us_region"] = result["state_name"].map(STATE_TO_REGION).fillna("Other")
    result["majority_minority_tract"] = np.where(
        result["minority_population"] >= 50.0, "Majority minority", "Not majority minority"
    )
    return result


def duckdb_feature_expressions() -> list[str]:
    """Return DuckDB expressions for the same frozen formulas."""
    region_cases = " ".join(
        f"WHEN state_name = '{state.replace(chr(39), chr(39) * 2)}' THEN '{region}'"
        for state, region in STATE_TO_REGION.items()
    )
    income_ratio = "applicant_income_000s / NULLIF(hud_median_family_income / 1000.0, 0.0)"
    tract_ratio = "tract_to_msamd_income / 100.0"
    return [
        "ln(1.0 + applicant_income_000s) AS log1p_applicant_income",
        "ln(1.0 + population) AS log1p_population",
        "ln(1.0 + hud_median_family_income) AS log1p_hud_median_family_income",
        "ln(1.0 + number_of_owner_occupied_units) AS log1p_owner_occupied_units",
        "ln(1.0 + number_of_1_to_4_family_units) AS log1p_1_to_4_family_units",
        f"{income_ratio} AS applicant_income_to_area_income",
        f"{tract_ratio} AS tract_income_ratio",
        "number_of_owner_occupied_units / NULLIF(number_of_1_to_4_family_units, 0.0) AS owner_occupied_unit_ratio",
        "number_of_1_to_4_family_units / NULLIF(population, 0.0) * 1000.0 AS family_units_per_1000_people",
        "number_of_owner_occupied_units / NULLIF(population, 0.0) * 1000.0 AS owner_occupied_units_per_1000_people",
        "CASE WHEN lower(co_applicant_sex_name) LIKE '%no co-applicant%' THEN CAST(0 AS TINYINT) ELSE CAST(1 AS TINYINT) END AS has_co_applicant",
        "CASE WHEN regexp_matches(lower(loan_type_name), 'fha|va|fsa|rhs') THEN 'Government backed' ELSE 'Conventional' END AS loan_program_group",
        f"CASE WHEN {income_ratio} <= 0.5 THEN 'Very low' WHEN {income_ratio} <= 0.8 THEN 'Low' WHEN {income_ratio} <= 1.2 THEN 'Moderate' ELSE 'High' END AS applicant_income_area_group",
        f"CASE WHEN {tract_ratio} <= 0.5 THEN 'Very low' WHEN {tract_ratio} <= 0.8 THEN 'Low' WHEN {tract_ratio} <= 1.2 THEN 'Moderate' ELSE 'High' END AS tract_income_level",
        f"CASE {region_cases} ELSE 'Other' END AS us_region",
        "CASE WHEN minority_population >= 50.0 THEN 'Majority minority' ELSE 'Not majority minority' END AS majority_minority_tract",
    ]
