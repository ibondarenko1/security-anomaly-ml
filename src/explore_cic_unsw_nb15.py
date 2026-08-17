"""Inspect CIC-UNSW-NB15 without modifying or preprocessing its flows.

The full CICFlowMeter export is roughly two gigabytes, so this script scans it
in chunks.  It is intentionally limited to dataset acquisition verification
and schema/timestamp inspection; it does not create splits or model features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cic_unsw_nb15"
FULL_DATA_PATH = DATA_DIR / "CICFlowMeter_out.csv"
SAMPLED_DATA_PATH = DATA_DIR / "Data.csv"
SAMPLED_LABEL_CANDIDATES = (
    DATA_DIR / "Label.csv",
    DATA_DIR / "Lable.csv",
)
README_PATH = DATA_DIR / "Readme.txt"
REPORT_PATH = PROJECT_ROOT / "models" / "cic_unsw_nb15_inspection.json"
TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"
METADATA_COLUMNS = (
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
)
LABEL_COLUMN = "Label"
EXPECTED_FULL_COLUMNS = (*METADATA_COLUMNS, LABEL_COLUMN)
SOURCE_PAGE = "https://www.unb.ca/cic/datasets/cic-unsw-nb15.html"
DOWNLOAD_MIRROR = (
    "https://huggingface.co/datasets/bencorn/CIC-UNSW-NB15"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="CSV rows processed per chunk (default: 200000).",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip file checksums when a faster local rescan is desired.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def datetime_to_epoch_seconds(values: pd.Series) -> pd.Series:
    """Return epoch seconds without assuming pandas' internal resolution."""
    return pd.Series(
        values.to_numpy(dtype="datetime64[s]").astype("int64"),
        index=values.index,
        dtype="int64",
    )


def resolved_label_path() -> Path:
    for path in SAMPLED_LABEL_CANDIDATES:
        if path.is_file():
            return path
    expected = ", ".join(str(path) for path in SAMPLED_LABEL_CANDIDATES)
    raise FileNotFoundError(f"Missing sampled label file; expected one of: {expected}")


def inspect_full_dataset(chunksize: int) -> dict[str, Any]:
    if not FULL_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing full CICFlowMeter export: {FULL_DATA_PATH}")

    row_count = 0
    columns: list[str] | None = None
    dtype_observations: dict[str, set[str]] = {}
    missing_counts: Counter[str] = Counter()
    infinite_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    protocol_counts: Counter[str] = Counter()
    src_ips: set[str] = set()
    dst_ips: set[str] = set()
    src_ports: set[str] = set()
    dst_ports: set[str] = set()
    timestamp_counts: Counter[int] = Counter()
    day_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    day_src_ips: dict[str, set[str]] = defaultdict(set)
    day_dst_ips: dict[str, set[str]] = defaultdict(set)
    day_attack_src_ips: dict[str, set[str]] = defaultdict(set)
    day_attack_dst_ips: dict[str, set[str]] = defaultdict(set)
    invalid_timestamp_count = 0
    timestamps_with_fractional_seconds = 0
    timestamp_reversals = 0
    previous_timestamp: pd.Timestamp | None = None
    first_rows: list[dict[str, Any]] = []

    reader = pd.read_csv(FULL_DATA_PATH, chunksize=chunksize, low_memory=False)
    for chunk_number, chunk in enumerate(reader, start=1):
        if columns is None:
            columns = chunk.columns.tolist()
            missing_required = [
                column for column in EXPECTED_FULL_COLUMNS if column not in columns
            ]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")
            first_rows = chunk.head(3).where(pd.notna(chunk.head(3)), None).to_dict(
                orient="records"
            )
        elif chunk.columns.tolist() != columns:
            raise ValueError(f"Schema changed in chunk {chunk_number}")

        row_count += len(chunk)
        for column, dtype in chunk.dtypes.items():
            dtype_observations.setdefault(column, set()).add(str(dtype))
        missing_counts.update(chunk.isna().sum().astype("int64").to_dict())

        numeric_columns = chunk.select_dtypes(include=[np.number]).columns
        if len(numeric_columns):
            numeric_values = chunk[numeric_columns].to_numpy(dtype="float64")
            infinite_counts.update(
                dict(
                    zip(
                        numeric_columns,
                        np.isinf(numeric_values).sum(axis=0).astype("int64"),
                        strict=True,
                    )
                )
            )

        labels = chunk[LABEL_COLUMN].fillna("<MISSING>").astype(str).str.strip()
        label_counts.update(labels.value_counts().to_dict())
        protocols = chunk["Protocol"].fillna("<MISSING>").astype(str).str.strip()
        protocol_counts.update(protocols.value_counts().to_dict())

        src_ips.update(chunk["Src IP"].dropna().astype(str).str.strip().unique())
        dst_ips.update(chunk["Dst IP"].dropna().astype(str).str.strip().unique())
        src_ports.update(chunk["Src Port"].dropna().astype(str).str.strip().unique())
        dst_ports.update(chunk["Dst Port"].dropna().astype(str).str.strip().unique())

        raw_timestamps = chunk["Timestamp"].astype("string")
        timestamps_with_fractional_seconds += int(
            raw_timestamps.str.contains(r"\d{2}:\d{2}:\d{2}\.\d+", na=False).sum()
        )
        timestamps = pd.to_datetime(
            raw_timestamps,
            format=TIMESTAMP_FORMAT,
            errors="coerce",
        )
        invalid_timestamp_count += int(timestamps.isna().sum())
        valid = timestamps.dropna()
        if not valid.empty:
            if previous_timestamp is not None and valid.iloc[0] < previous_timestamp:
                timestamp_reversals += 1
            timestamp_reversals += int((valid.diff().dropna() < pd.Timedelta(0)).sum())
            previous_timestamp = valid.iloc[-1]
            # Pandas may retain second-resolution datetime64 values for this
            # input.  Converting explicitly to datetime64[s] avoids assuming
            # that the integer representation is always nanoseconds.
            epoch_seconds = datetime_to_epoch_seconds(valid)
            timestamp_counts.update(epoch_seconds.value_counts().to_dict())

            day_values = timestamps.dt.strftime("%Y-%m-%d")
            temporal_frame = pd.DataFrame(
                {
                    "day": day_values,
                    "label": labels,
                    "src_ip": chunk["Src IP"].astype(str).str.strip(),
                    "dst_ip": chunk["Dst IP"].astype(str).str.strip(),
                }
            ).dropna(subset=["day"])
            for day, day_frame in temporal_frame.groupby("day", sort=False):
                day_label_counts[day].update(
                    day_frame["label"].value_counts().to_dict()
                )
                day_src_ips[day].update(day_frame["src_ip"].unique())
                day_dst_ips[day].update(day_frame["dst_ip"].unique())
                attacks = day_frame[day_frame["label"] != "Benign"]
                day_attack_src_ips[day].update(attacks["src_ip"].unique())
                day_attack_dst_ips[day].update(attacks["dst_ip"].unique())

        print(
            f"Scanned chunk {chunk_number}: {row_count:,} rows",
            flush=True,
        )

    if columns is None:
        raise ValueError("Full CICFlowMeter CSV is empty")

    unique_epoch_seconds = sorted(timestamp_counts)
    min_positive_delta_seconds: int | None = None
    largest_timestamp_gap_seconds: int | None = None
    gaps_over_60_seconds = 0
    gaps_over_1_hour = 0
    gaps_over_1_day = 0
    if len(unique_epoch_seconds) > 1:
        deltas = np.diff(np.asarray(unique_epoch_seconds, dtype="int64"))
        positive = deltas[deltas > 0]
        if len(positive):
            min_positive_delta_seconds = int(positive.min())
            largest_timestamp_gap_seconds = int(positive.max())
            gaps_over_60_seconds = int((positive > 60).sum())
            gaps_over_1_hour = int((positive > 3_600).sum())
            gaps_over_1_day = int((positive > 86_400).sum())

    capture_days = sorted(day_label_counts)
    host_overlap_by_day_pair: list[dict[str, Any]] = []
    for first_index, first_day in enumerate(capture_days):
        for second_day in capture_days[first_index + 1 :]:
            first_hosts = day_src_ips[first_day] | day_dst_ips[first_day]
            second_hosts = day_src_ips[second_day] | day_dst_ips[second_day]
            host_overlap_by_day_pair.append(
                {
                    "first_day": first_day,
                    "second_day": second_day,
                    "src_ip_overlap": len(
                        day_src_ips[first_day] & day_src_ips[second_day]
                    ),
                    "dst_ip_overlap": len(
                        day_dst_ips[first_day] & day_dst_ips[second_day]
                    ),
                    "any_endpoint_ip_overlap": len(first_hosts & second_hosts),
                }
            )

    numeric_features = [
        column
        for column in columns
        if column not in (*METADATA_COLUMNS, LABEL_COLUMN)
    ]
    label_total = sum(label_counts.values())
    benign_count = label_counts.get("Benign", 0)
    attack_count = label_total - benign_count - label_counts.get("<MISSING>", 0)

    return {
        "path": str(FULL_DATA_PATH.relative_to(PROJECT_ROOT)),
        "file_size_bytes": FULL_DATA_PATH.stat().st_size,
        "rows": row_count,
        "columns_count": len(columns),
        "columns": columns,
        "metadata_columns": list(METADATA_COLUMNS),
        "flow_feature_columns": numeric_features,
        "flow_feature_count": len(numeric_features),
        "label_column": LABEL_COLUMN,
        "dtype_observations": {
            column: sorted(values) for column, values in dtype_observations.items()
        },
        "missing_counts": dict(sorted(missing_counts.items())),
        "total_missing_cells": int(sum(missing_counts.values())),
        "infinite_counts": dict(sorted(infinite_counts.items())),
        "total_infinite_numeric_values": int(sum(infinite_counts.values())),
        "label_distribution": {
            label: {
                "count": int(count),
                "percentage": count / label_total * 100,
            }
            for label, count in label_counts.most_common()
        },
        "binary_distribution": {
            "benign": {
                "count": int(benign_count),
                "percentage": benign_count / label_total * 100,
            },
            "attack": {
                "count": int(attack_count),
                "percentage": attack_count / label_total * 100,
            },
        },
        "protocol_distribution": {
            protocol: int(count) for protocol, count in protocol_counts.most_common()
        },
        "cardinality": {
            "src_ip": len(src_ips),
            "dst_ip": len(dst_ips),
            "src_port": len(src_ports),
            "dst_port": len(dst_ports),
            "timestamp_seconds": len(timestamp_counts),
        },
        "timestamp": {
            "declared_parse_format": TIMESTAMP_FORMAT,
            "invalid_count": invalid_timestamp_count,
            "fractional_second_string_count": timestamps_with_fractional_seconds,
            "observed_string_precision": "seconds",
            "minimum": (
                pd.Timestamp(unique_epoch_seconds[0], unit="s").isoformat()
                if unique_epoch_seconds
                else None
            ),
            "maximum": (
                pd.Timestamp(unique_epoch_seconds[-1], unit="s").isoformat()
                if unique_epoch_seconds
                else None
            ),
            "unique_seconds": len(timestamp_counts),
            "minimum_positive_delta_seconds": min_positive_delta_seconds,
            "largest_gap_between_observed_seconds": largest_timestamp_gap_seconds,
            "gaps_over_60_seconds": gaps_over_60_seconds,
            "gaps_over_1_hour": gaps_over_1_hour,
            "gaps_over_1_day": gaps_over_1_day,
            "maximum_flows_sharing_one_second": (
                max(timestamp_counts.values()) if timestamp_counts else 0
            ),
            "rows_in_shared_timestamp_seconds": int(
                sum(count for count in timestamp_counts.values() if count > 1)
            ),
            "timestamp_order_reversals_in_file": timestamp_reversals,
            "file_is_chronologically_sorted": timestamp_reversals == 0,
            "timezone": "not specified by the published CSV",
        },
        "capture_days": {
            day: {
                "rows": int(sum(day_label_counts[day].values())),
                "label_distribution": {
                    label: int(count)
                    for label, count in day_label_counts[day].most_common()
                },
                "src_ip_count": len(day_src_ips[day]),
                "dst_ip_count": len(day_dst_ips[day]),
                "attack_src_ip_count": len(day_attack_src_ips[day]),
                "attack_dst_ip_count": len(day_attack_dst_ips[day]),
            }
            for day in capture_days
        },
        "host_overlap_between_capture_days": host_overlap_by_day_pair,
        "first_three_rows": first_rows,
    }


def count_csv_rows(path: Path, chunksize: int) -> tuple[int, list[str]]:
    row_count = 0
    columns: list[str] | None = None
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if columns is None:
            columns = chunk.columns.tolist()
        row_count += len(chunk)
    return row_count, columns or []


def inspect_companion_files(chunksize: int) -> dict[str, Any]:
    if not SAMPLED_DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing sampled feature file: {SAMPLED_DATA_PATH}")
    label_path = resolved_label_path()

    sampled_rows, sampled_columns = count_csv_rows(SAMPLED_DATA_PATH, chunksize)
    label_frame = pd.read_csv(label_path)
    if label_frame.shape[1] != 1:
        raise ValueError(
            f"Expected one sampled-label column, found {label_frame.shape[1]}"
        )
    label_column = label_frame.columns[0]
    label_counts = label_frame[label_column].value_counts(dropna=False)

    return {
        "sampled_data": {
            "path": str(SAMPLED_DATA_PATH.relative_to(PROJECT_ROOT)),
            "file_size_bytes": SAMPLED_DATA_PATH.stat().st_size,
            "rows": sampled_rows,
            "columns_count": len(sampled_columns),
            "columns": sampled_columns,
            "contains_temporal_or_endpoint_metadata": any(
                column in sampled_columns for column in METADATA_COLUMNS
            ),
        },
        "sampled_labels": {
            "path": str(label_path.relative_to(PROJECT_ROOT)),
            "published_filename_note": (
                "The acquired package spells the file Lable.csv; the UNB page "
                "documents it as Label.csv."
            ),
            "rows": len(label_frame),
            "column": label_column,
            "distribution": {
                str(label): int(count) for label, count in label_counts.items()
            },
            "row_count_matches_sampled_data": len(label_frame) == sampled_rows,
        },
        "readme": {
            "path": str(README_PATH.relative_to(PROJECT_ROOT)),
            "text": README_PATH.read_text(encoding="utf-8", errors="replace").strip(),
        },
    }


def print_report(report: dict[str, Any]) -> None:
    full = report["full_dataset"]
    timestamp = full["timestamp"]
    sampled = report["companion_files"]["sampled_data"]

    print("\nCIC-UNSW-NB15 FULL EXPORT")
    print(f"Rows x columns: {full['rows']:,} x {full['columns_count']}")
    print(f"Flow features: {full['flow_feature_count']}")
    print(f"Metadata: {full['metadata_columns']}")
    print(f"Labels: {json.dumps(full['label_distribution'], indent=2)}")
    print(f"Missing cells: {full['total_missing_cells']:,}")
    print(f"Infinite numeric values: {full['total_infinite_numeric_values']:,}")
    print("\nTIMESTAMP")
    print(f"Range: {timestamp['minimum']} through {timestamp['maximum']}")
    print(f"Observed precision: {timestamp['observed_string_precision']}")
    print(f"Unique timestamp seconds: {timestamp['unique_seconds']:,}")
    print(
        "Maximum flows sharing one timestamp second: "
        f"{timestamp['maximum_flows_sharing_one_second']:,}"
    )
    print(f"Invalid timestamps: {timestamp['invalid_count']:,}")
    print(f"Chronologically sorted on disk: {timestamp['file_is_chronologically_sorted']}")
    print(f"Timestamp order reversals: {timestamp['timestamp_order_reversals_in_file']:,}")
    print(f"Timezone: {timestamp['timezone']}")
    print("Capture days:")
    for day, details in full["capture_days"].items():
        print(
            f"  {day}: {details['rows']:,} rows, "
            f"{details['src_ip_count']} src IPs, {details['dst_ip_count']} dst IPs"
        )
    print("\nCARDINALITY")
    print(json.dumps(full["cardinality"], indent=2))
    print("\nCOLUMNS")
    for index, column in enumerate(full["columns"], start=1):
        print(f"{index:>2}. {column}")
    print("\nSAMPLED DATA.csv")
    print(f"Rows x columns: {sampled['rows']:,} x {sampled['columns_count']}")
    print(
        "Contains endpoint/timestamp metadata: "
        f"{sampled['contains_temporal_or_endpoint_metadata']}"
    )
    print(f"\nMachine-readable report: {REPORT_PATH}")


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")

    full = inspect_full_dataset(args.chunksize)
    companion_files = inspect_companion_files(args.chunksize)
    files = [FULL_DATA_PATH, SAMPLED_DATA_PATH, resolved_label_path(), README_PATH]
    checksums = (
        {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in files}
        if not args.skip_sha256
        else {}
    )

    report = {
        "report_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Security Anomaly ML v2 - Temporal Context Network Triage",
        "v1_status": "frozen; no v1 artifact was read or modified",
        "provenance": {
            "official_dataset_page": SOURCE_PAGE,
            "acquisition": (
                "The official CIC download endpoint requires a registration form. "
                "Files were acquired from a public mirror using the filenames and "
                "contents documented by UNB; local SHA-256 hashes are recorded."
            ),
            "download_mirror": DOWNLOAD_MIRROR,
            "raw_files_modified": False,
            "sha256": checksums,
        },
        "full_dataset": full,
        "companion_files": companion_files,
        "v2_readiness": {
            "use_for_temporal_context": "CICFlowMeter_out.csv",
            "do_not_use_for_temporal_context": "Data.csv",
            "available_context_keys": list(METADATA_COLUMNS),
            "timestamp_tie_policy_required": True,
            "timestamp_tie_reason": (
                "Published timestamps have second precision, so same-second flows "
                "have no trustworthy internal ordering."
            ),
            "unavailable_requested_signals": [
                "packet TTL (not present in the CSV)",
                "true RTT (not present; Flow IAT is not RTT)",
                "service name (can only be derived cautiously from ports)",
                "UNSW state field (TCP flags are available instead)",
            ],
            "split_created": False,
            "features_engineered": False,
            "model_trained": False,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=json_value),
        encoding="utf-8",
    )
    print_report(report)


if __name__ == "__main__":
    main()
