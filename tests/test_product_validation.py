from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.security_anomaly.validation import (
    InputContractError,
    read_and_validate_flow_csv,
    validate_flow_records,
)


def test_missing_columns_are_reported_together(valid_flow_frame: pd.DataFrame) -> None:
    frame = valid_flow_frame.drop(columns=["Flow Duration", "Idle Min"])
    with pytest.raises(InputContractError) as captured:
        validate_flow_records(frame)
    assert "Flow Duration" in str(captured.value)
    assert "Idle Min" in str(captured.value)


def test_dataframe_duplicate_columns_are_rejected(valid_flow_frame: pd.DataFrame) -> None:
    duplicate = pd.concat(
        [valid_flow_frame, valid_flow_frame[["Flow Duration"]]], axis=1
    )
    with pytest.raises(InputContractError, match="duplicate columns"):
        validate_flow_records(duplicate)


def test_raw_csv_duplicate_headers_are_rejected_before_pandas(
    tmp_path: Path, valid_flow_frame: pd.DataFrame
) -> None:
    source = tmp_path / "duplicate.csv"
    header = valid_flow_frame.columns.tolist()
    header[-1] = header[0]
    source.write_text(",".join(header) + "\n", encoding="utf-8")
    with pytest.raises(InputContractError, match="duplicate columns"):
        read_and_validate_flow_csv(source)


def test_reserved_derived_column_is_rejected(valid_flow_frame: pd.DataFrame) -> None:
    frame = valid_flow_frame.assign(src_conn_10s=10)
    with pytest.raises(InputContractError, match="reserved/internal"):
        validate_flow_records(frame)


def test_known_ground_truth_fields_are_ignored_not_consumed(
    valid_flow_frame: pd.DataFrame,
) -> None:
    frame = valid_flow_frame.assign(Label=["Benign", "Attack"], attack_cat=["", "X"])
    batch = validate_flow_records(frame)
    assert "Label" not in batch.frame
    assert "attack_cat" not in batch.frame
    assert batch.ignored_evaluation_columns == ("Label", "attack_cat")


def test_unknown_extra_column_fails_closed(valid_flow_frame: pd.DataFrame) -> None:
    frame = valid_flow_frame.assign(mystery=1)
    with pytest.raises(InputContractError, match="Unexpected columns"):
        validate_flow_records(frame)


def test_invalid_timestamp_is_rejected(valid_flow_frame: pd.DataFrame) -> None:
    frame = valid_flow_frame.copy()
    frame.loc[1, "Timestamp"] = "2015-02-17T20:00:01Z"
    with pytest.raises(InputContractError, match="Timestamp must use"):
        validate_flow_records(frame)


def test_non_numeric_model_field_is_rejected(valid_flow_frame: pd.DataFrame) -> None:
    frame = valid_flow_frame.copy()
    frame["Flow Duration"] = frame["Flow Duration"].astype(object)
    frame.loc[0, "Flow Duration"] = "not-a-number"
    with pytest.raises(InputContractError, match="Flow Duration.*non-numeric"):
        validate_flow_records(frame)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nan_and_infinity_are_rejected(
    valid_flow_frame: pd.DataFrame, value: float
) -> None:
    frame = valid_flow_frame.copy()
    frame.loc[0, "Flow Duration"] = value
    with pytest.raises(InputContractError):
        validate_flow_records(frame)


@pytest.mark.parametrize(
    ("column", "value"),
    [("Src Port", -1), ("Dst Port", 65536), ("Src Port", 12.5)],
)
def test_invalid_ports_are_rejected(
    valid_flow_frame: pd.DataFrame, column: str, value: float
) -> None:
    frame = valid_flow_frame.copy()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = value
    with pytest.raises(InputContractError, match=r"integers in \[0, 65535\]"):
        validate_flow_records(frame)


@pytest.mark.parametrize("value", [-1, 256, 6.5])
def test_invalid_protocol_is_rejected(
    valid_flow_frame: pd.DataFrame, value: float
) -> None:
    frame = valid_flow_frame.copy()
    frame["Protocol"] = frame["Protocol"].astype(float)
    frame.loc[0, "Protocol"] = value
    with pytest.raises(InputContractError, match="Protocol must contain"):
        validate_flow_records(frame)


def test_empty_file_without_header_fails(tmp_path: Path) -> None:
    source = tmp_path / "empty.csv"
    source.write_bytes(b"")
    with pytest.raises(InputContractError, match="empty or has no header"):
        read_and_validate_flow_csv(source)


def test_header_only_csv_is_a_valid_empty_batch(
    tmp_path: Path, valid_flow_frame: pd.DataFrame
) -> None:
    source = tmp_path / "header-only.csv"
    valid_flow_frame.head(0).to_csv(source, index=False)
    batch = read_and_validate_flow_csv(source)
    assert batch.row_count == 0
    assert batch.timestamp_min is None


def test_malformed_csv_is_rejected(tmp_path: Path, valid_flow_frame: pd.DataFrame) -> None:
    source = tmp_path / "malformed.csv"
    source.write_text(
        ",".join(valid_flow_frame.columns) + "\n\"unterminated",
        encoding="utf-8",
    )
    with pytest.raises(InputContractError, match="Malformed or unreadable CSV"):
        read_and_validate_flow_csv(source)
