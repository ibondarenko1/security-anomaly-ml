"""Shared public validation for CICFlowMeter-compatible product input."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import FeatureContract


TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"
RESERVED_COLUMNS = (
    "_source_row_id",
    "_event_ts",
    "_flow_bytes",
    "_flow_packets",
)


class InputContractError(ValueError):
    """Raised when input cannot satisfy the public v0.1 flow contract."""


@dataclass(frozen=True)
class ValidatedFlowBatch:
    """Normalized label-free rows plus their parsed source-defined timestamps."""

    frame: pd.DataFrame
    timestamps: pd.Series
    contract_version: str
    ignored_evaluation_columns: tuple[str, ...] = ()

    @property
    def row_count(self) -> int:
        return len(self.frame)

    @property
    def timestamp_min(self) -> pd.Timestamp | None:
        return None if self.timestamps.empty else pd.Timestamp(self.timestamps.min())

    @property
    def timestamp_max(self) -> pd.Timestamp | None:
        return None if self.timestamps.empty else pd.Timestamp(self.timestamps.max())


def _column_names(contract: FeatureContract) -> tuple[set[str], set[str], set[str]]:
    input_document = contract.document["input"]
    optional = set(input_document.get("optional_preserved_columns", ()))
    ignored = set(input_document.get("ignored_evaluation_columns", ()))
    derived = set(contract.model_features) - set(contract.baseline_features) - {
        "Src Port",
        "Dst Port",
        "Protocol",
    }
    return optional, ignored, derived


def validate_flow_records(
    records: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    contract: FeatureContract | None = None,
) -> ValidatedFlowBatch:
    """Validate and normalize records once for builder, CLI validate, and analyze.

    Known evaluation labels are ignored and removed. Other incompatible extra
    fields fail closed so the v0.1 input contract is not silently broadened.
    """

    contract = contract or FeatureContract.load()
    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    if frame.columns.has_duplicates:
        duplicates = list(dict.fromkeys(frame.columns[frame.columns.duplicated()].tolist()))
        raise InputContractError(f"Input contains duplicate columns: {duplicates}")
    if not all(isinstance(name, str) and name for name in frame.columns):
        raise InputContractError("Every input column must have a non-empty string name")

    optional, ignored, derived = _column_names(contract)
    collisions = sorted((set(RESERVED_COLUMNS) | derived) & set(frame.columns))
    if collisions:
        raise InputContractError(f"Input uses reserved/internal columns: {collisions}")

    missing = [name for name in contract.required_input_columns if name not in frame]
    if missing:
        raise InputContractError(f"Missing required CICFlowMeter columns: {missing}")

    allowed = set(contract.required_input_columns) | optional | ignored
    unexpected = sorted(set(frame.columns) - allowed)
    if unexpected:
        raise InputContractError(f"Unexpected columns for cicflow-v2-128: {unexpected}")

    ignored_present = tuple(name for name in ignored if name in frame.columns)
    if ignored_present:
        frame = frame.drop(columns=list(ignored_present))
    frame = frame.reset_index(drop=True)

    parsed = pd.to_datetime(frame["Timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
    if parsed.isna().any():
        bad_rows = frame.index[parsed.isna()].tolist()[:20]
        raise InputContractError(
            "Timestamp must use CICFlowMeter format "
            f"{TIMESTAMP_FORMAT!r}; invalid rows: {bad_rows}"
        )

    for name in ("Src IP", "Dst IP"):
        invalid = frame[name].isna() | frame[name].astype(str).str.strip().eq("")
        if invalid.any():
            bad_rows = frame.index[invalid].tolist()[:20]
            raise InputContractError(f"{name} contains missing/empty values at rows {bad_rows}")
        frame[name] = frame[name].astype(str)

    numeric_columns = list(
        dict.fromkeys(
            [*contract.baseline_features, "Src Port", "Dst Port", "Protocol"]
        )
    )
    for name in numeric_columns:
        converted = pd.to_numeric(frame[name], errors="coerce")
        if converted.isna().any():
            bad_rows = frame.index[converted.isna()].tolist()[:20]
            raise InputContractError(
                f"{name} contains missing or non-numeric values at rows {bad_rows}"
            )
        frame[name] = converted
    if numeric_columns and len(frame):
        numeric_matrix = frame[numeric_columns].to_numpy(dtype=float, copy=False)
        if not np.isfinite(numeric_matrix).all():
            raise InputContractError("Numeric input fields must be finite (no NaN/Infinity)")

    for name in ("Src Port", "Dst Port"):
        values = frame[name].to_numpy(dtype=float)
        invalid = (values < 0) | (values > 65535) | (values != np.floor(values))
        if invalid.any():
            bad_rows = np.flatnonzero(invalid).tolist()[:20]
            raise InputContractError(
                f"{name} must contain integers in [0, 65535]; invalid rows: {bad_rows}"
            )
        frame[name] = values.astype(np.int64)

    protocols = frame["Protocol"].to_numpy(dtype=float)
    invalid_protocol = (
        (protocols < 0) | (protocols > 255) | (protocols != np.floor(protocols))
    )
    if invalid_protocol.any():
        bad_rows = np.flatnonzero(invalid_protocol).tolist()[:20]
        raise InputContractError(
            "Protocol must contain integers in [0, 255]; "
            f"invalid rows: {bad_rows}"
        )
    frame["Protocol"] = protocols.astype(np.int64)

    return ValidatedFlowBatch(
        frame=frame,
        timestamps=parsed.reset_index(drop=True),
        contract_version=contract.version,
        ignored_evaluation_columns=tuple(sorted(ignored_present)),
    )


def validate_csv_header(path: str | Path) -> list[str]:
    """Read the raw header before pandas can mangle duplicate names."""

    source = Path(path)
    if not source.is_file():
        raise InputContractError(f"Input CSV not found: {source}")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle, strict=True), None)
    except (OSError, UnicodeError, csv.Error) as error:
        raise InputContractError(f"Cannot read CSV header: {error}") from error
    if header is None or not header or not any(value.strip() for value in header):
        raise InputContractError("Input CSV is empty or has no header")
    if any(value == "" for value in header):
        raise InputContractError("Input CSV header contains an empty column name")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise InputContractError(f"Input CSV header contains duplicate columns: {duplicates}")
    return header


def read_and_validate_flow_csv(
    path: str | Path,
    *,
    contract: FeatureContract | None = None,
) -> ValidatedFlowBatch:
    """Load a CSV only after lossless raw-header validation."""

    source = Path(path)
    raw_header = validate_csv_header(source)
    try:
        frame = pd.read_csv(source, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise InputContractError(f"Malformed or unreadable CSV: {error}") from error
    if frame.columns.tolist() != raw_header:
        raise InputContractError(
            "CSV columns changed during parsing; input header is not losslessly representable"
        )
    return validate_flow_records(frame, contract=contract)
