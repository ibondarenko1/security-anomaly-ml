from __future__ import annotations

import inspect

import numpy as np
import pytest

from src import train_v2_fuzzer_weighting as weighting


def test_sample_weight_changes_only_fuzzer_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weighting, "EXPECTED_TRAIN_FUZZERS", 2)
    mask = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

    sample_weight = weighting.build_sample_weight(mask, 1.5)

    np.testing.assert_array_equal(sample_weight, [1.0, 1.5, 1.0, 1.5, 1.0])
    assert sample_weight.dtype == np.float32


def test_weight_grid_and_frozen_model_settings() -> None:
    assert weighting.FUZZER_WEIGHTS == (1.0, 1.25, 1.5, 2.0, 3.0)
    assert weighting.RECALL_GATE == 0.985
    assert weighting.RANDOM_FOREST_PARAMS == {
        "n_estimators": 300,
        "max_depth": 24,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "max_samples": 0.8,
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    }


def test_model_fit_receives_sample_weight() -> None:
    source = inspect.getsource(weighting.fit_and_score_weight)

    assert "sample_weight=sample_weight" in source


def test_weighting_module_has_no_locked_holdout_data_path() -> None:
    source = inspect.getsource(weighting)

    assert "locked_holdout_features.parquet" not in source
    assert weighting.TRAIN_PATH.name == "train_features.parquet"
    assert weighting.VALIDATION_PATH.name == "validation_features.parquet"


def test_new_model_names_do_not_overwrite_frozen_context() -> None:
    for weight in weighting.FUZZER_WEIGHTS:
        assert weighting.model_path(weight) != weighting.FROZEN_CONTEXT_MODEL_PATH
        assert "fuzzer_weight" in weighting.model_path(weight).name
