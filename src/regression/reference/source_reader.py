"""Bounded source discovery and Arrow CSV streaming for Prompt 1A."""

from __future__ import annotations

import csv
import hashlib
import io
import threading
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow as pa
import pyarrow.csv as pacsv


def discover_raw_csv(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("*.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one extracted raw CSV, found {len(candidates)}")
    return candidates[0].resolve()


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)):
        raise ValueError("Raw CSV header contains duplicate column names")
    return header


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


class HashingReader(io.RawIOBase):
    """Sequential binary reader that hashes the exact bytes Arrow consumes."""

    def __init__(self, path: Path):
        super().__init__()
        self._handle = path.open("rb", buffering=0)
        self._digest = hashlib.sha256()
        self._read_lock = threading.Lock()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        with self._read_lock:
            count = self._handle.readinto(buffer)
            if count:
                self._digest.update(memoryview(buffer)[:count])
        return count

    @property
    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


def iter_csv_batches(
    path: Path,
    columns: Iterable[str],
    block_size: int,
    invalid_counter: dict[str, int] | None = None,
    hash_stream: bool = False,
) -> tuple[Iterator[pa.RecordBatch], HashingReader | None]:
    selected = list(columns)

    def handle_invalid(_row) -> str:
        if invalid_counter is not None:
            invalid_counter["structurally_malformed_rows"] = (
                invalid_counter.get("structurally_malformed_rows", 0) + 1
            )
        return "skip"

    read_options = pacsv.ReadOptions(
        block_size=block_size,
        encoding="utf8",
        use_threads=True,
    )
    parse_options = pacsv.ParseOptions(
        delimiter=",",
        invalid_row_handler=handle_invalid,
    )
    convert_options = pacsv.ConvertOptions(
        include_columns=selected,
        column_types={name: pa.string() for name in selected},
        strings_can_be_null=False,
        quoted_strings_can_be_null=False,
    )

    hashing_reader = HashingReader(path) if hash_stream else None
    if hashing_reader is None:
        source = str(path)
    else:
        source = pa.input_stream(hashing_reader)

    reader = pacsv.open_csv(
        source,
        read_options=read_options,
        parse_options=parse_options,
        convert_options=convert_options,
    )
    return iter(reader), hashing_reader
