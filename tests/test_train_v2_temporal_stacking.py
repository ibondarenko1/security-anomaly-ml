from __future__ import annotations

import inspect

from src import train_v2_temporal_stacking as stacking


def test_temporal_folds_are_expanding_and_purged() -> None:
    folds = stacking.temporal_folds()

    assert len(folds) == 4
    assert sum(fold["prediction_rows"] for fold in folds) == stacking.EXPECTED_OOF_ROWS
    assert stacking.TEMPORAL_BOUNDARIES == (
        0,
        60_000,
        120_000,
        180_000,
        240_000,
        stacking.TRAIN_ROWS,
    )
    assert [fold["train_end_exclusive"] for fold in folds] == sorted(
        fold["train_end_exclusive"] for fold in folds
    )
    for fold in folds:
        assert fold["train_start"] == 0
        assert fold["prediction_start"] - fold["train_end_exclusive"] == 1_024
        assert fold["prediction_start"] > fold["train_end_exclusive"]
        assert fold["prediction_rows"] > 0


def test_timestamp_group_purge_is_strictly_larger_than_known_group() -> None:
    assert (
        stacking.BOUNDARY_PURGE_ROWS
        > stacking.MAX_FLOWS_PER_TIMESTAMP_SECOND
    )


def test_stacker_meta_features_are_small_and_predeclared() -> None:
    assert len(stacking.META_FEATURES) == 17
    assert stacking.SCORE_META_FEATURES == (
        "baseline_score",
        "context_score",
        "baseline_minus_context_score",
        "max_score",
        "min_score",
    )
    assert "Protocol" in stacking.CONTEXT_META_FEATURES
    assert "dst_unique_sport_60s" in stacking.CONTEXT_META_FEATURES
    assert "src_dst_conn_60s" in stacking.CONTEXT_META_FEATURES
    assert "src_dport_conn_60s" in stacking.CONTEXT_META_FEATURES


def test_base_rf_and_shallow_stacker_settings_are_frozen() -> None:
    assert stacking.RANDOM_FOREST_PARAMS["n_estimators"] == 300
    assert stacking.RANDOM_FOREST_PARAMS["random_state"] == 42
    assert stacking.LOGISTIC_PARAMS == {"max_iter": 1_000, "solver": "lbfgs"}
    assert stacking.HIST_GRADIENT_BOOSTING_PARAMS["max_depth"] == 3
    assert stacking.HIST_GRADIENT_BOOSTING_PARAMS["early_stopping"] is False


def test_stacking_module_has_no_random_fold_or_locked_holdout_path() -> None:
    source = inspect.getsource(stacking)

    assert "StratifiedKFold" not in source
    assert "KFold(" not in source
    assert "locked_holdout_features.parquet" not in source
    assert stacking.TRAIN_PATH.name == "train_features.parquet"
    assert stacking.VALIDATION_PATH.name == "validation_features.parquet"
