"""Future-safe CSV row loading for post-Test model development.

The parser skips excluded physical data rows before pandas converts requested
fields. This utility is for future work only. It does not change past fits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Sequence

import numpy as np
import pandas as pd


def load_allowed_source_rows(
    source: Path | str,
    row_ids: Sequence[int] | np.ndarray,
    columns: Sequence[str],
    *,
    allowed_train_ids: Collection[int],
    read_csv_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load requested saved-Train rows without parsing excluded row fields.

    Row IDs are zero-based physical data-row positions below the CSV header.
    The function rejects duplicate or non-Train requests. ``skiprows`` runs at
    the CSV parser boundary, before converters and dtype parsing for skipped
    data rows. Requested order is restored after the allowed rows are parsed.
    """

    source = Path(source)
    requested = np.asarray(row_ids, dtype=np.int64)
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError("row_ids must be a non-empty one-dimensional sequence")
    if np.any(requested < 0):
        raise ValueError("row_ids must be non-negative physical data-row positions")
    if len(np.unique(requested)) != len(requested):
        raise ValueError("Requested row IDs are not unique")

    allowed = {int(value) for value in allowed_train_ids}
    requested_set = {int(value) for value in requested}
    outside = sorted(requested_set.difference(allowed))
    if outside:
        raise PermissionError(f"Requested rows are outside saved Train membership: {outside[:10]}")

    ordered_columns = list(columns)
    if not ordered_columns or len(set(ordered_columns)) != len(ordered_columns):
        raise ValueError("columns must be a non-empty unique sequence")

    # pandas calls skiprows with physical CSV line numbers. Line zero is the
    # header, and data row ID N is physical line N + 1.
    def skip_excluded_line(line_number: int) -> bool:
        return line_number > 0 and (line_number - 1) not in requested_set

    kwargs = dict(read_csv_kwargs or {})
    forbidden = {"chunksize", "iterator", "skiprows", "usecols", "nrows"}.intersection(kwargs)
    if forbidden:
        raise ValueError(f"Unsupported read_csv options for the safe contract: {sorted(forbidden)}")

    selected = pd.read_csv(
        source,
        usecols=ordered_columns,
        skiprows=skip_excluded_line,
        **kwargs,
    )
    sorted_ids = np.sort(requested)
    if len(selected) != len(sorted_ids):
        raise RuntimeError(
            "Safe loader row-count mismatch; blank or malformed physical source rows are not allowed"
        )
    selected.insert(0, "__row_id__", sorted_ids)
    selected = selected.set_index("__row_id__", drop=True)
    return selected.loc[requested, ordered_columns].copy()

