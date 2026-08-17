from __future__ import annotations

import inspect

import numpy as np

from src import train_v2_ablation
from src.train_v2_ablation import sweep_thresholds


def test_threshold_policy_minimizes_fp_after_recall_gate() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    scores = np.array([0.10, 0.20, 0.80, 0.90, 0.70, 0.60])

    _, selected, gate_met = sweep_thresholds("fixture", y_true, scores)

    assert gate_met
    assert selected["threshold"] == 0.60
    assert selected["recall"] == 1.0
    assert selected["false_positives"] == 1
    assert np.isclose(selected["fp_per_100k_benign"], 100_000 / 3)


def test_evaluation_module_has_no_locked_holdout_data_path() -> None:
    source = inspect.getsource(train_v2_ablation)

    assert "locked_holdout_features.parquet" not in source
    assert train_v2_ablation.TRAIN_PATH.name == "train_features.parquet"
    assert train_v2_ablation.VALIDATION_PATH.name == "validation_features.parquet"
