"""Label-free causal temporal features for CICFlowMeter-compatible batches.

This module is intentionally independent from the research dataset split and
evaluation scripts. A batch starts with empty temporal state, is ordered only
by timestamp, and excludes the complete current timestamp peer group from all
history windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from .contracts import FeatureContract


TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"
RESERVED_COLUMNS = ("_source_row_id", "_event_ts", "_flow_bytes", "_flow_packets")


class InputContractError(ValueError):
    """Raised when an input batch cannot satisfy the frozen feature contract."""


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _window_clause() -> str:
    return """
WINDOW
    src10 AS (
        PARTITION BY "Src IP" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src60 AS (
        PARTITION BY "Src IP" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_history AS (
        PARTITION BY "Src IP" ORDER BY _event_ts
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dst10 AS (
        PARTITION BY "Src IP", "Dst IP" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dst60 AS (
        PARTITION BY "Src IP", "Dst IP" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dst_history AS (
        PARTITION BY "Src IP", "Dst IP" ORDER BY _event_ts
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dport10 AS (
        PARTITION BY "Src IP", "Dst Port" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    src_dport60 AS (
        PARTITION BY "Src IP", "Dst Port" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst10 AS (
        PARTITION BY "Dst IP" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst60 AS (
        PARTITION BY "Dst IP" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst_history AS (
        PARTITION BY "Dst IP" ORDER BY _event_ts
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst_sport10 AS (
        PARTITION BY "Dst IP", "Src Port" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 10 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    ),
    dst_sport60 AS (
        PARTITION BY "Dst IP", "Src Port" ORDER BY _event_ts
        RANGE BETWEEN INTERVAL 60 SECOND PRECEDING AND CURRENT ROW
        EXCLUDE GROUP
    )
""".strip()


def _context_expressions() -> list[str]:
    """Frozen 43-feature SQL contract, independent from research labels/dates."""

    return [
        "(count(*) OVER src10)::BIGINT AS src_conn_10s",
        "(count(*) OVER src60)::BIGINT AS src_conn_60s",
        '(count(DISTINCT "Dst IP") OVER src10)::BIGINT AS src_unique_dst_10s',
        '(count(DISTINCT "Dst IP") OVER src60)::BIGINT AS src_unique_dst_60s',
        '(count(DISTINCT "Dst Port") OVER src10)::BIGINT AS src_unique_dport_10s',
        '(count(DISTINCT "Dst Port") OVER src60)::BIGINT AS src_unique_dport_60s',
        "(count(*) OVER src_dst10)::BIGINT AS src_dst_conn_10s",
        "(count(*) OVER src_dst60)::BIGINT AS src_dst_conn_60s",
        "(count(*) OVER src_dport10)::BIGINT AS src_dport_conn_10s",
        "(count(*) OVER src_dport60)::BIGINT AS src_dport_conn_60s",
        "coalesce(sum(_flow_bytes) OVER src10, 0.0) AS src_bytes_10s",
        "coalesce(sum(_flow_bytes) OVER src60, 0.0) AS src_bytes_60s",
        "coalesce(sum(_flow_packets) OVER src10, 0.0) AS src_packets_10s",
        "coalesce(sum(_flow_packets) OVER src60, 0.0) AS src_packets_60s",
        "coalesce(avg(_flow_bytes) OVER src60, 0.0) AS src_mean_bytes_per_flow_60s",
        "coalesce(avg(_flow_packets) OVER src60, 0.0) AS src_mean_packets_per_flow_60s",
        "coalesce(date_diff('second', max(_event_ts) OVER src_history, _event_ts), -1)::BIGINT AS seconds_since_src_last_flow",
        "coalesce(date_diff('second', max(_event_ts) OVER src_dst_history, _event_ts), -1)::BIGINT AS seconds_since_src_dst_last_flow",
        "coalesce(sum(CASE WHEN \"Protocol\" = 6 THEN 1 ELSE 0 END) OVER src60 / nullif(count(*) OVER src60, 0), 0.0)::DOUBLE AS src_tcp_ratio_60s",
        "coalesce(sum(CASE WHEN \"Protocol\" = 17 THEN 1 ELSE 0 END) OVER src60 / nullif(count(*) OVER src60, 0), 0.0)::DOUBLE AS src_udp_ratio_60s",
        'coalesce(sum("SYN Flag Count") OVER src60, 0)::DOUBLE AS src_syn_count_60s',
        'coalesce(sum("RST Flag Count") OVER src60, 0)::DOUBLE AS src_rst_count_60s',
        'coalesce(count(DISTINCT "Dst Port") OVER src60 / nullif(count(*) OVER src60, 0), 0.0)::DOUBLE AS src_dport_diversity_60s',
        "(count(*) OVER dst10)::BIGINT AS dst_conn_10s",
        "(count(*) OVER dst60)::BIGINT AS dst_conn_60s",
        '(count(DISTINCT "Src IP") OVER dst10)::BIGINT AS dst_unique_src_10s',
        '(count(DISTINCT "Src IP") OVER dst60)::BIGINT AS dst_unique_src_60s',
        '(count(DISTINCT "Src Port") OVER dst10)::BIGINT AS dst_unique_sport_10s',
        '(count(DISTINCT "Src Port") OVER dst60)::BIGINT AS dst_unique_sport_60s',
        "(count(*) OVER dst_sport10)::BIGINT AS dst_sport_conn_10s",
        "(count(*) OVER dst_sport60)::BIGINT AS dst_sport_conn_60s",
        "coalesce(sum(_flow_bytes) OVER dst10, 0.0) AS dst_bytes_10s",
        "coalesce(sum(_flow_bytes) OVER dst60, 0.0) AS dst_bytes_60s",
        "coalesce(sum(_flow_packets) OVER dst10, 0.0) AS dst_packets_10s",
        "coalesce(sum(_flow_packets) OVER dst60, 0.0) AS dst_packets_60s",
        "coalesce(avg(_flow_bytes) OVER dst60, 0.0) AS dst_mean_bytes_per_flow_60s",
        "coalesce(avg(_flow_packets) OVER dst60, 0.0) AS dst_mean_packets_per_flow_60s",
        "coalesce(date_diff('second', max(_event_ts) OVER dst_history, _event_ts), -1)::BIGINT AS seconds_since_dst_last_flow",
        "coalesce(sum(CASE WHEN \"Protocol\" = 6 THEN 1 ELSE 0 END) OVER dst60 / nullif(count(*) OVER dst60, 0), 0.0)::DOUBLE AS dst_tcp_ratio_60s",
        "coalesce(sum(CASE WHEN \"Protocol\" = 17 THEN 1 ELSE 0 END) OVER dst60 / nullif(count(*) OVER dst60, 0), 0.0)::DOUBLE AS dst_udp_ratio_60s",
        'coalesce(sum("SYN Flag Count") OVER dst60, 0)::DOUBLE AS dst_syn_count_60s',
        'coalesce(sum("RST Flag Count") OVER dst60, 0)::DOUBLE AS dst_rst_count_60s',
        'coalesce(count(DISTINCT "Src Port") OVER dst60 / nullif(count(*) OVER dst60, 0), 0.0)::DOUBLE AS dst_sport_diversity_60s',
    ]


@dataclass(frozen=True)
class FeatureBatch:
    """Timestamp-ordered identities plus the exact frozen model matrix."""

    identities: pd.DataFrame
    features: pd.DataFrame
    contract_version: str
    state_mode: str = "batch-empty"

    def to_numpy(self) -> np.ndarray:
        matrix = self.features.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(matrix).all():
            raise InputContractError("Generated model matrix contains NaN or infinity")
        return matrix

    def scores_in_source_order(self, ordered_scores: Sequence[float]) -> np.ndarray:
        scores = np.asarray(ordered_scores, dtype=float)
        if len(scores) != len(self.identities):
            raise ValueError("Score count does not match feature batch row count")
        restored = np.empty(len(scores), dtype=float)
        source_rows = self.identities["source_row_id"].to_numpy(dtype=np.int64)
        restored[source_rows] = scores
        return restored


class CausalTemporalFeatureBuilder:
    """Build frozen Context-v2 features from an unlabeled in-memory batch."""

    def __init__(self, contract: FeatureContract | None = None) -> None:
        self.contract = contract or FeatureContract.load()
        if len(_context_expressions()) != len(self.contract.temporal_features):
            raise RuntimeError("Temporal expression count differs from feature contract")

    def _normalize_input(
        self, records: pd.DataFrame | Sequence[Mapping[str, Any]]
    ) -> pd.DataFrame:
        frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        if frame.columns.has_duplicates:
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise InputContractError(f"Input contains duplicate columns: {duplicates}")
        collisions = sorted(set(RESERVED_COLUMNS) & set(frame.columns))
        if collisions:
            raise InputContractError(f"Input uses reserved columns: {collisions}")
        missing = [name for name in self.contract.required_input_columns if name not in frame]
        if missing:
            raise InputContractError(f"Missing required CICFlowMeter columns: {missing}")

        frame = frame.reset_index(drop=True)
        frame["_source_row_id"] = np.arange(len(frame), dtype=np.int64)
        parsed = pd.to_datetime(
            frame["Timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
        )
        if parsed.isna().any():
            bad_rows = frame.index[parsed.isna()].tolist()[:20]
            raise InputContractError(
                "Timestamp must use CICFlowMeter format "
                f"{TIMESTAMP_FORMAT!r}; invalid rows: {bad_rows}"
            )
        frame["_event_ts"] = parsed

        for name in ("Src IP", "Dst IP"):
            invalid = frame[name].isna() | frame[name].astype(str).str.strip().eq("")
            if invalid.any():
                raise InputContractError(f"{name} contains missing/empty values")
            frame[name] = frame[name].astype(str)

        numeric_columns = list(
            dict.fromkeys(
                [
                    *self.contract.baseline_features,
                    "Src Port",
                    "Dst Port",
                    "Protocol",
                ]
            )
        )
        for name in numeric_columns:
            converted = pd.to_numeric(frame[name], errors="coerce")
            if converted.isna().any():
                bad_rows = frame.index[converted.isna()].tolist()[:20]
                raise InputContractError(f"{name} is non-numeric at rows {bad_rows}")
            frame[name] = converted
        numeric_matrix = frame[numeric_columns].to_numpy(dtype=float, copy=False)
        if not np.isfinite(numeric_matrix).all():
            raise InputContractError("Numeric input fields must be finite")

        for name in ("Src Port", "Dst Port"):
            values = frame[name].to_numpy(dtype=float)
            if ((values < 0) | (values > 65535) | (values != np.floor(values))).any():
                raise InputContractError(f"{name} must contain integers in [0, 65535]")
            frame[name] = values.astype(np.int64)
        protocols = frame["Protocol"].to_numpy(dtype=float)
        if ((protocols < 0) | (protocols > 255) | (protocols != np.floor(protocols))).any():
            raise InputContractError("Protocol must contain integers in [0, 255]")
        frame["Protocol"] = protocols.astype(np.int64)

        frame["_flow_bytes"] = (
            frame["Total Length of Fwd Packet"]
            + frame["Total Length of Bwd Packet"]
        ).astype(float)
        frame["_flow_packets"] = (
            frame["Total Fwd Packet"] + frame["Total Bwd packets"]
        ).astype(float)
        return frame

    def _query(self, *, flow_id_present: bool) -> str:
        baseline_sql = ",\n                ".join(
            quote_identifier(name) for name in self.contract.baseline_features
        )
        context_sql = ",\n                ".join(_context_expressions())
        model_sql = ",\n            ".join(
            quote_identifier(name) for name in self.contract.model_features
        )
        flow_id_sql = (
            'CAST("Flow ID" AS VARCHAR) AS flow_id'
            if flow_id_present
            else "NULL::VARCHAR AS flow_id"
        )
        return f"""
        WITH enriched AS (
            SELECT
                _source_row_id,
                _event_ts,
                "Flow ID" AS _flow_id,
                "Src IP",
                "Dst IP",
                {baseline_sql},
                "Src Port",
                "Dst Port",
                "Protocol",
                CASE WHEN "Src Port" BETWEEN 0 AND 1023 THEN 1 ELSE 0 END::UTINYINT AS src_port_well_known,
                CASE WHEN "Src Port" BETWEEN 1024 AND 49151 THEN 1 ELSE 0 END::UTINYINT AS src_port_registered,
                CASE WHEN "Src Port" BETWEEN 49152 AND 65535 THEN 1 ELSE 0 END::UTINYINT AS src_port_ephemeral,
                CASE WHEN "Dst Port" BETWEEN 0 AND 1023 THEN 1 ELSE 0 END::UTINYINT AS dst_port_well_known,
                CASE WHEN "Dst Port" BETWEEN 1024 AND 49151 THEN 1 ELSE 0 END::UTINYINT AS dst_port_registered,
                CASE WHEN "Dst Port" BETWEEN 49152 AND 65535 THEN 1 ELSE 0 END::UTINYINT AS dst_port_ephemeral,
                {context_sql}
            FROM input_flows
            {_window_clause()}
        )
        SELECT
            _source_row_id AS source_row_id,
            _event_ts AS timestamp,
            {flow_id_sql.replace('"Flow ID"', '_flow_id')},
            "Src IP" AS src_ip,
            "Dst IP" AS dst_ip,
            "Src Port"::BIGINT AS src_port,
            "Dst Port"::BIGINT AS dst_port,
            "Protocol"::BIGINT AS identity_protocol,
            {model_sql}
        FROM enriched
        ORDER BY _event_ts, _source_row_id
        """

    def build(
        self, records: pd.DataFrame | Sequence[Mapping[str, Any]]
    ) -> FeatureBatch:
        """Build a deterministic, tie-safe batch with state initialized empty."""

        frame = self._normalize_input(records)
        flow_id_present = "Flow ID" in frame.columns
        if not flow_id_present:
            frame["Flow ID"] = pd.Series([None] * len(frame), dtype="string")
        connection = duckdb.connect()
        try:
            connection.execute("SET preserve_insertion_order = false")
            connection.register("input_flows", frame)
            result = connection.execute(
                self._query(flow_id_present=flow_id_present)
            ).fetchdf()
        finally:
            connection.close()

        identity_columns = [
            "source_row_id",
            "timestamp",
            "flow_id",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "identity_protocol",
        ]
        features = result[self.contract.model_features].copy()
        if list(features.columns) != self.contract.model_features:
            raise RuntimeError("Generated feature order differs from frozen contract")
        identities = result[identity_columns].copy().rename(
            columns={"identity_protocol": "protocol"}
        )
        batch = FeatureBatch(
            identities=identities,
            features=features,
            contract_version=self.contract.version,
        )
        batch.to_numpy()
        return batch
