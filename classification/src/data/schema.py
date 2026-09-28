"""Explicit schema and column contracts for the supplied HMDA extract."""

from __future__ import annotations

RAW_TARGET = "loan_approved"
ANALYTICAL_TARGET = "target_denied"

EXPECTED_COLUMNS: tuple[str, ...] = (
    "respondent_id",
    "agency_name",
    "loan_type_name",
    "property_type_name",
    "loan_purpose_name",
    "owner_occupancy_name",
    "loan_amount_000s",
    "preapproval_name",
    "msamd_name",
    "state_name",
    "state_code",
    "county_name",
    "county_code",
    "census_tract_number",
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "applicant_income_000s",
    "lien_status_name",
    "population",
    "minority_population",
    "hud_median_family_income",
    "tract_to_msamd_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
    RAW_TARGET,
)

# Codes are strings on purpose: their numeric appearance does not make them
# measurements, and parsing them as numbers can destroy leading zeroes.
STRING_COLUMNS: tuple[str, ...] = (
    "respondent_id",
    "agency_name",
    "loan_type_name",
    "property_type_name",
    "loan_purpose_name",
    "owner_occupancy_name",
    "preapproval_name",
    "msamd_name",
    "state_name",
    "state_code",
    "county_name",
    "county_code",
    "census_tract_number",
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
    "lien_status_name",
)

INTEGER_COLUMNS: tuple[str, ...] = (
    "loan_amount_000s",
    "applicant_income_000s",
    "population",
    "hud_median_family_income",
    "number_of_owner_occupied_units",
    "number_of_1_to_4_family_units",
)

FLOAT_COLUMNS: tuple[str, ...] = (
    "minority_population",
    "tract_to_msamd_income",
)

PROTECTED_COLUMNS: tuple[str, ...] = (
    "applicant_ethnicity_name",
    "co_applicant_ethnicity_name",
    "applicant_race_name_1",
    "co_applicant_race_name_1",
    "applicant_sex_name",
    "co_applicant_sex_name",
)

# These columns remain available in the modeling view for audit, but are not
# approved primary predictive features. Geography can proxy protected traits;
# respondent_id is a direct entity identifier.
AUDIT_ONLY_COLUMNS: tuple[str, ...] = PROTECTED_COLUMNS + (
    "respondent_id",
    "msamd_name",
    "state_name",
    "state_code",
    "county_name",
    "county_code",
    "census_tract_number",
)

PRIMARY_EXCLUDED_COLUMNS: tuple[str, ...] = (
    RAW_TARGET,
    ANALYTICAL_TARGET,
) + AUDIT_ONLY_COLUMNS

READ_DTYPES: dict[str, str] = {
    **{column: "string" for column in STRING_COLUMNS},
    **{column: "Int64" for column in INTEGER_COLUMNS},
    **{column: "Float64" for column in FLOAT_COLUMNS},
    RAW_TARGET: "Int8",
}

assert set(READ_DTYPES) == set(EXPECTED_COLUMNS)


def primary_feature_columns(columns: tuple[str, ...] | list[str]) -> list[str]:
    """Return the ordered leakage-safe primary feature contract."""

    excluded = set(PRIMARY_EXCLUDED_COLUMNS)
    return [column for column in columns if column not in excluded]
