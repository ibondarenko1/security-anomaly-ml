from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.security_anomaly.contracts import FeatureContract
from src.security_anomaly.temporal import CausalTemporalFeatureBuilder
from tools.verify_product_pipeline_parity import (
    PARITY_SOURCE_ROW,
    FeatureParityAccumulator,
    build_lossless_alignment,
    remove_evaluation_columns,
)


def raw_fixture(contract: FeatureContract) -> pd.DataFrame:
    rows = 4
    data: dict[str, list[object]] = {
        feature: [0.0] * rows for feature in contract.baseline_features
    }
    data.update(
        {
            "Flow ID": ["duplicate-a", "duplicate-b", "later-c", "later-d"],
            "Src IP": ["10.0.0.1"] * rows,
            "Src Port": [50000, 50000, 50001, 50002],
            "Dst IP": ["10.0.0.2"] * rows,
            "Dst Port": [80, 80, 443, 53],
            "Protocol": [6, 6, 6, 17],
            "Timestamp": [
                "17/02/2015 08:00:00 PM",
                "17/02/2015 08:00:00 PM",
                "17/02/2015 08:00:05 PM",
                "17/02/2015 08:00:10 PM",
            ],
        }
    )
    frame = pd.DataFrame(data)
    frame["Total Length of Fwd Packet"] = [100, 100, 200, 300]
    frame["Total Length of Bwd Packet"] = [50, 50, 100, 150]
    frame["Total Fwd Packet"] = [2, 2, 4, 6]
    frame["Total Bwd packets"] = [1, 1, 2, 3]
    frame["SYN Flag Count"] = [1, 1, 1, 0]
    frame["RST Flag Count"] = [0, 0, 1, 0]
    return frame


def test_ground_truth_and_evaluation_fields_are_removed_before_builder() -> None:
    contract = FeatureContract.load()
    frame = raw_fixture(contract)
    frame["Label"] = ["Benign", "Fuzzers", "Benign", "Benign"]
    frame["attack_score"] = [0.0, 1.0, 0.0, 0.0]

    stripped, removed = remove_evaluation_columns(frame, contract)

    assert removed == ["Label", "attack_score"]
    assert set(stripped.columns) == {
        *contract.required_input_columns,
        "Flow ID",
    }
    CausalTemporalFeatureBuilder(contract).build(stripped)


def test_lossless_alignment_handles_reordered_same_timestamp_duplicates(
    tmp_path: Path,
) -> None:
    contract = FeatureContract.load()
    raw = raw_fixture(contract)
    frozen = CausalTemporalFeatureBuilder(contract).build(raw).features.iloc[
        [2, 1, 3, 0]
    ]
    raw[PARITY_SOURCE_ROW] = np.arange(len(raw), dtype=np.int64)
    frozen_path = tmp_path / "frozen.parquet"

    connection = duckdb.connect()
    try:
        connection.register("frozen_input", frozen)
        connection.execute(
            f"COPY (SELECT * FROM frozen_input) TO '{frozen_path.as_posix()}' "
            "(FORMAT PARQUET)"
        )
        connection.unregister("frozen_input")
        connection.register("alignment_input", raw)
        validation_to_source, report = build_lossless_alignment(
            connection,
            frozen_features_path=frozen_path,
            contract=contract,
            expected_rows=4,
        )
    finally:
        connection.close()

    assert validation_to_source.tolist() == [2, 0, 3, 1]
    assert report["matched_rows"] == 4
    assert report["direct_value_mismatches"] == 0
    assert report["duplicate_identical_feature_groups"] == 1
    assert report["rows_in_duplicate_identical_feature_groups"] == 2


def test_feature_parity_accumulator_reports_cells_rows_and_temporal_gate() -> None:
    accumulator = FeatureParityAccumulator(["a", "temporal", "c"])
    accumulator.observe(
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0], [4.0, 7.0, 8.0]], dtype=np.float32),
    )

    report = accumulator.report(["temporal"])

    assert report["rows_compared"] == 2
    assert report["mismatched_rows"] == 1
    assert report["mismatched_cells"] == 2
    assert report["maximum_absolute_numeric_difference"] == 2.0
    assert report["per_feature_mismatch_summary"]["a"]["mismatched_cells"] == 0
    assert report["per_feature_mismatch_summary"]["temporal"][
        "mismatched_cells"
    ] == 1
    assert report["all_43_temporal_features_match"] is False
