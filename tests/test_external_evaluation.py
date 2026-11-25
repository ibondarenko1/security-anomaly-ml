from __future__ import annotations

import numpy as np
import pandas as pd

from src.security_anomaly.contracts import FeatureContract
from tools.evaluate_external_capture import (
    BEHAVIOR_THRESHOLD,
    audit_frozen_rf_compatibility,
    behavioural_scores,
    evaluate_scores,
    load_external_slice,
    top_k_mask,
)


def test_behavioral_score_and_equal_budget_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "Flow Pkts/s": [0.0, 1_000.0, 1_000.0],
            "Flow Byts/s": [0.0, 0.0, 1_000_000.0],
            "SYN Flag Cnt": [0.0, 0.0, 3.0],
            "RST Flag Cnt": [0.0, 0.0, 3.0],
        }
    )
    scores = behavioural_scores(frame)
    np.testing.assert_allclose(scores, [0.0, 0.25, 1.0])

    first = top_k_mask(scores, 2)
    second = top_k_mask(scores, 2)
    np.testing.assert_array_equal(first, [False, True, True])
    np.testing.assert_array_equal(first, second)

    result = evaluate_scores(
        np.array([0, 1, 1], dtype=np.int8),
        scores,
        threshold=BEHAVIOR_THRESHOLD,
        analyst_budget=2,
    )
    assert result["threshold_policy"]["confusion"] == {
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 1,
    }
    assert result["fixed_analyst_budget_policy"]["confusion"] == {
        "tp": 2,
        "fp": 0,
        "tn": 1,
        "fn": 0,
    }


def test_cse_style_schema_fails_closed_without_zero_fill() -> None:
    frame = pd.DataFrame(
        {
            "Dst Port": [80],
            "Protocol": [6],
            "Timestamp": ["01/03/2018 08:00:00"],
            "Flow Byts/s": [100.0],
            "Flow Pkts/s": [10.0],
            "SYN Flag Cnt": [0],
            "RST Flag Cnt": [0],
            "CWE Flag Count": [0],
            "Label": ["Benign"],
        }
    )
    compatibility = audit_frozen_rf_compatibility(
        frame, FeatureContract.load()
    )

    assert compatibility["compatible"] is False
    assert compatibility["no_zero_fill_or_implicit_rename"] is True
    assert compatibility["missing_identity_context_columns"] == [
        "Src IP",
        "Src Port",
        "Dst IP",
    ]
    assert compatibility["timestamp_audit"]["source_parseable_rows"] == 1
    assert compatibility["timestamp_audit"]["product_parseable_rows"] == 0
    assert {
        "source_column": "CWE Flag Count",
        "contract_column": "CWR Flag Count",
        "decision": (
            "upstream CICFlowMeter semantic alias is known, but the "
            "exact product column is not renamed automatically"
        ),
    } in compatibility["source_specific_name_differences"]


def test_external_slice_excludes_repeated_headers_and_nonfinite_rows(
    tmp_path,
) -> None:
    source = tmp_path / "external.csv"
    pd.DataFrame(
        {
            "Timestamp": [
                "Timestamp",
                "01/03/2018 08:00:00",
                "01/03/2018 08:00:01",
                "01/03/2018 08:00:02",
            ],
            "Flow Pkts/s": [
                "Flow Pkts/s",
                "10",
                "Infinity",
                "1000",
            ],
            "Flow Byts/s": ["Flow Byts/s", "10", "1", "1000"],
            "SYN Flag Cnt": ["SYN Flag Cnt", "0", "0", "3"],
            "RST Flag Cnt": ["RST Flag Cnt", "0", "0", "0"],
            "Label": ["Label", "Benign", "Infilteration", "Infilteration"],
        }
    ).to_csv(source, index=False)

    frame, info = load_external_slice(source)

    assert info["raw_rows"] == 4
    assert info["repeated_header_rows_excluded"] == 1
    assert info["nonfinite_behavior_rows_excluded"] == 1
    assert info["evaluated_rows"] == 2
    assert info["evaluated_label_counts"] == {
        "Benign": 1,
        "Infilteration": 1,
    }
    assert frame["_binary_label"].tolist() == [0, 1]
