"""Verify frozen Context scores against the existing Feb 17 reference artifact.

This is a label-free compatibility check. It does not fit a model, select a
threshold, or access the locked Feb 18 holdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security_anomaly.contracts import FeatureContract, sha256_file  # noqa: E402
from src.security_anomaly.model_bundle import FrozenModelBundle  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "models" / "v2_context_random_forest.joblib"
DEFAULT_MANIFEST = PROJECT_ROOT / "contracts" / "model-manifest-context-rf-v2.json"
DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"
DEFAULT_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cic_unsw_nb15_v2"
    / "validation_features.parquet"
)
DEFAULT_REFERENCE = PROJECT_ROOT / "models" / "v2_validation_scores.parquet"
EXPECTED_FEATURES_SHA256 = (
    "bb33d2d5379f801d87f4806e8543641bd5dffeaaaa37d045ce33d401c05aec0a"
)
EXPECTED_REFERENCE_SHA256 = (
    "e471dc6953851819345d9850e88e13d81aaaeaba701a5bb815c038b741e3649b"
)
EXPECTED_ROWS = 498_890
FLOW_THRESHOLD = 0.10


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_string(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--reference-scores", type=Path, default=DEFAULT_REFERENCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, expected, label in (
        (args.features, EXPECTED_FEATURES_SHA256, "Feb 17 frozen features"),
        (args.reference_scores, EXPECTED_REFERENCE_SHA256, "Feb 17 reference scores"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"{label} SHA256 mismatch: expected {expected}, got {actual}"
            )

    contract = FeatureContract.load(args.contract)
    bundle = FrozenModelBundle.load(args.model, args.manifest, args.contract)
    selected = ", ".join(quote_identifier(name) for name in contract.model_features)
    connection = duckdb.connect()
    rows = 0
    maximum_absolute_difference = 0.0
    differences_gt_1e_12 = 0
    threshold_disagreements = 0
    try:
        connection.execute(
            f"""
            SELECT {selected}, s.context_attack_score
            FROM (
                SELECT row_number() OVER () - 1 AS validation_row, *
                FROM read_parquet({sql_string(args.features)})
            ) f
            JOIN read_parquet({sql_string(args.reference_scores)}) s
              USING (validation_row)
            ORDER BY validation_row
            """
        )
        while True:
            chunk = connection.fetch_df_chunk(100)
            if chunk.empty:
                break
            current = bundle.predict_scores(
                chunk[contract.model_features].to_numpy(dtype=np.float32, copy=True)
            )
            reference = chunk["context_attack_score"].to_numpy(dtype=float)
            absolute = np.abs(current - reference)
            maximum_absolute_difference = max(
                maximum_absolute_difference, float(absolute.max(initial=0.0))
            )
            differences_gt_1e_12 += int(np.count_nonzero(absolute > 1e-12))
            threshold_disagreements += int(
                np.count_nonzero(
                    (current >= FLOW_THRESHOLD) != (reference >= FLOW_THRESHOLD)
                )
            )
            rows += len(chunk)
    finally:
        connection.close()
    if rows != EXPECTED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_ROWS:,} rows, compared {rows:,}")
    result = {
        "model_version": bundle.model_version,
        "feature_contract": contract.version,
        "comparison_data": "Feb 17 frozen validation; labels not loaded",
        "rows_compared": rows,
        "maximum_absolute_score_difference": maximum_absolute_difference,
        "score_differences_gt_1e_12": differences_gt_1e_12,
        "threshold_0_10_class_disagreements": threshold_disagreements,
        "score_parity_achieved": differences_gt_1e_12 == 0,
        "operational_predictions_identical": threshold_disagreements == 0,
        "model_retrained": False,
        "holdout_accessed": False,
    }
    print(json.dumps(result, indent=2))
    if not result["score_parity_achieved"]:
        raise AssertionError("Frozen reference score parity failed")


if __name__ == "__main__":
    main()
