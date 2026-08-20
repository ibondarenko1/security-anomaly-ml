from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.build_temporal_features import (
    CONTEXT_FEATURES,
    STATIC_BEHAVIORAL_FEATURES,
    enriched_query,
)
from src.security_anomaly.contracts import FeatureContract
from src.security_anomaly.temporal import (
    CausalTemporalFeatureBuilder,
    InputContractError,
)


def unlabeled_fixture(contract: FeatureContract) -> pd.DataFrame:
    rows = 5
    data: dict[str, list[object]] = {
        feature: [0.0] * rows for feature in contract.baseline_features
    }
    data.update(
        {
            "Flow ID": [f"fixture-{index}" for index in range(rows)],
            "Src IP": ["10.0.0.1"] * rows,
            "Src Port": [50000, 50001, 50002, 50003, 50004],
            "Dst IP": ["10.0.0.2", "10.0.0.3", "10.0.0.2", "10.0.0.4", "10.0.0.2"],
            "Dst Port": [80, 53, 80, 443, 80],
            "Protocol": [6, 17, 6, 6, 6],
            "Timestamp": [
                "19/08/2026 12:00:00 AM",
                "19/08/2026 12:00:00 AM",
                "19/08/2026 12:00:05 AM",
                "19/08/2026 12:00:05 AM",
                "19/08/2026 12:00:11 AM",
            ],
        }
    )
    frame = pd.DataFrame(data)
    frame["Flow Duration"] = [1, 2, 3, 4, 5]
    frame["Total Length of Fwd Packet"] = [60, 120, 180, 240, 300]
    frame["Total Length of Bwd Packet"] = [40, 80, 120, 160, 200]
    frame["Total Fwd Packet"] = [1, 2, 3, 4, 5]
    frame["Total Bwd packets"] = [1, 2, 3, 4, 5]
    frame["SYN Flag Count"] = [1, 0, 1, 1, 1]
    frame["RST Flag Count"] = [0, 0, 0, 1, 0]
    return frame


def research_features(frame: pd.DataFrame, contract: FeatureContract) -> pd.DataFrame:
    research = frame.copy()
    research["Label"] = "Benign"
    connection = duckdb.connect()
    try:
        connection.register("raw_fixture", research)
        connection.execute(
            """
            CREATE TABLE flows AS
            SELECT *,
                   strptime("Timestamp", '%d/%m/%Y %I:%M:%S %p') AS event_ts,
                   ("Total Length of Fwd Packet" + "Total Length of Bwd Packet")::DOUBLE AS flow_bytes,
                   ("Total Fwd Packet" + "Total Bwd packets")::DOUBLE AS flow_packets
            FROM raw_fixture
            """
        )
        selected = ", ".join(f'"{name}"' for name in contract.model_features)
        return connection.execute(
            f"""
            WITH enriched AS ({enriched_query('flows', contract.baseline_features)})
            SELECT {selected}
            FROM enriched
            ORDER BY event_ts, "Flow Duration"
            """
        ).fetchdf()
    finally:
        connection.close()


def test_contract_is_the_exact_frozen_76_plus_9_plus_43_order() -> None:
    contract = FeatureContract.load()
    assert len(contract.baseline_features) == 76
    assert contract.static_features == list(STATIC_BEHAVIORAL_FEATURES)
    assert contract.temporal_features == list(CONTEXT_FEATURES)
    assert contract.model_features == [
        *contract.baseline_features,
        *contract.static_features,
        *contract.temporal_features,
    ]


def test_label_free_builder_matches_every_research_feature() -> None:
    contract = FeatureContract.load()
    frame = unlabeled_fixture(contract)
    assert "Label" not in frame and "attack_cat" not in frame and "label" not in frame

    product = CausalTemporalFeatureBuilder(contract).build(frame)
    research = research_features(frame, contract)

    assert list(product.features.columns) == contract.model_features
    np.testing.assert_allclose(
        product.features.to_numpy(dtype=float),
        research.to_numpy(dtype=float),
        rtol=0,
        atol=0,
    )


def test_same_timestamp_peers_are_excluded_before_group_state_update() -> None:
    contract = FeatureContract.load()
    batch = CausalTemporalFeatureBuilder(contract).build(unlabeled_fixture(contract))

    assert batch.features["src_conn_10s"].tolist() == [0, 0, 2, 2, 2]
    assert batch.features["src_unique_dst_10s"].tolist() == [0, 0, 2, 2, 2]
    assert batch.features["seconds_since_src_last_flow"].tolist() == [-1, -1, 5, 5, 6]


def test_out_of_order_input_is_scored_by_time_and_scores_restore_to_source_order() -> None:
    contract = FeatureContract.load()
    frame = unlabeled_fixture(contract).iloc[[4, 0, 3, 1, 2]].reset_index(drop=True)
    batch = CausalTemporalFeatureBuilder(contract).build(frame)

    assert batch.identities["source_row_id"].tolist() == [1, 3, 2, 4, 0]
    restored = batch.scores_in_source_order([0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(restored, [0.5, 0.1, 0.3, 0.2, 0.4])


def test_missing_model_fields_fail_with_one_actionable_error() -> None:
    contract = FeatureContract.load()
    frame = unlabeled_fixture(contract).drop(columns=["Flow Duration", "Idle Min"])
    with pytest.raises(InputContractError, match="Flow Duration.*Idle Min"):
        CausalTemporalFeatureBuilder(contract).build(frame)
