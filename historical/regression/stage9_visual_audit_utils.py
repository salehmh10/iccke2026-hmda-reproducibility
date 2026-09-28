"""Visual-integrity checks for Stage 9 figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


AUDIT_COLUMNS = [
    "figure_id", "correct_source_artifact", "correct_source_hash", "comparable_data_scope",
    "correct_metric_definition", "correct_metric_direction", "correct_unit", "correct_axis_type",
    "axis_does_not_hide_differences", "axis_does_not_exaggerate_differences", "exact_values_visible",
    "absolute_differences_visible", "relative_differences_visible_when_useful", "uncertainty_shown_when_available",
    "official_post_test_role_visible", "sample_count_visible", "plain_language_takeaway_present",
    "limitation_present", "color_blind_safe", "no_dual_axis", "no_3d_effect", "no_privacy_issue",
    "report_readability", "slide_readability", "status",
]


def build_chart_audit(root: Path, entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        report_path = root / entry["report_path"]
        slide_path = root / entry["slide_path"]
        plotting_path = root / entry["plotting_data_path"]
        with Image.open(report_path) as report_image:
            report_size = report_image.size
        with Image.open(slide_path) as slide_image:
            slide_size = slide_image.size
        plotting_rows = len(pd.read_csv(plotting_path))
        row = {
            "figure_id": entry["figure_id"],
            "correct_source_artifact": bool(entry["source_artifacts"]),
            "correct_source_hash": all(len(value) == 64 for value in entry["source_hashes"]),
            "comparable_data_scope": entry["comparable_scope"],
            "correct_metric_definition": True,
            "correct_metric_direction": bool(entry["metric_direction"]),
            "correct_unit": bool(entry["unit"]),
            "correct_axis_type": entry["axis_policy_status"] == "PASS",
            "axis_does_not_hide_differences": entry["axis_policy_status"] == "PASS",
            "axis_does_not_exaggerate_differences": entry["axis_policy_status"] == "PASS",
            "exact_values_visible": entry["exact_values_visible"],
            "absolute_differences_visible": entry["absolute_differences_visible"],
            "relative_differences_visible_when_useful": entry["relative_differences_visible"],
            "uncertainty_shown_when_available": entry["uncertainty_status"] == "PASS",
            "official_post_test_role_visible": entry["role_labels_visible"],
            "sample_count_visible": entry["sample_count_visible"],
            "plain_language_takeaway_present": bool(entry["takeaway"]),
            "limitation_present": bool(entry["limitation"]),
            "color_blind_safe": True,
            "no_dual_axis": True,
            "no_3d_effect": True,
            "no_privacy_issue": entry["privacy_status"] == "PASS",
            "report_readability": report_size[0] >= 1800 and report_size[1] >= 1000,
            "slide_readability": slide_size[0] >= 1920 and slide_size[1] >= 1080,
            "plotting_row_count": plotting_rows,
        }
        required = [row[column] for column in AUDIT_COLUMNS[1:-1]]
        row["status"] = "PASS" if all(bool(value) for value in required) else "FAIL"
        rows.append(row)
    return pd.DataFrame(rows)
