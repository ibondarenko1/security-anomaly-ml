from __future__ import annotations

import duckdb

from src.build_temporal_features import (
    RAW_IDENTIFIER_COLUMNS,
    enriched_query,
    output_query,
)


def make_tie_fixture(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE TABLE flows (
            "Src IP" VARCHAR,
            "Dst IP" VARCHAR,
            "Src Port" INTEGER,
            "Dst Port" INTEGER,
            "Protocol" INTEGER,
            event_ts TIMESTAMP,
            flow_bytes DOUBLE,
            flow_packets DOUBLE,
            "SYN Flag Count" DOUBLE,
            "RST Flag Count" DOUBLE,
            "Flow Duration" BIGINT,
            "Label" VARCHAR
        )
        """
    )
    connection.execute(
        """
        INSERT INTO flows VALUES
          ('a', 'x', 50000, 80, 6, '2015-01-22 00:00:00', 100, 2, 1, 0, 1, 'Benign'),
          ('a', 'y', 50001, 53, 17, '2015-01-22 00:00:00', 200, 4, 0, 0, 2, 'Benign'),
          ('a', 'x', 50002, 80, 6, '2015-01-22 00:00:05', 300, 6, 1, 0, 3, 'Exploits'),
          ('a', 'z', 50003, 443, 6, '2015-01-22 00:00:05', 400, 8, 1, 1, 4, 'Benign'),
          ('a', 'x', 50004, 80, 6, '2015-01-22 00:00:11', 500, 10, 1, 0, 5, 'Benign')
        """
    )


def test_same_second_peers_never_enter_each_others_context() -> None:
    connection = duckdb.connect()
    make_tie_fixture(connection)
    frame = connection.execute(
        f"""
        WITH enriched AS ({enriched_query('flows', ['Flow Duration'])})
        SELECT event_ts, "Dst Port", src_conn_10s,
               src_unique_dst_10s, seconds_since_src_last_flow
        FROM enriched
        ORDER BY event_ts, "Dst Port"
        """
    ).fetchdf()

    assert frame["src_conn_10s"].tolist() == [0, 0, 2, 2, 2]
    assert frame["src_unique_dst_10s"].tolist() == [0, 0, 2, 2, 2]
    assert frame["seconds_since_src_last_flow"].tolist() == [-1, -1, 5, 5, 6]


def test_model_ready_output_excludes_raw_identifiers_and_timestamp() -> None:
    connection = duckdb.connect()
    make_tie_fixture(connection)
    columns = [
        row[0]
        for row in connection.execute(
            f"DESCRIBE {output_query('flows', ['Flow Duration'])}"
        ).fetchall()
    ]

    assert not (set(RAW_IDENTIFIER_COLUMNS) & set(columns))
    assert {"attack_cat", "label", "src_conn_10s"}.issubset(columns)
