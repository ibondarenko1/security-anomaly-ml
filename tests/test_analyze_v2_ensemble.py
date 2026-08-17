from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from src import analyze_v2_ensemble
from src.analyze_v2_ensemble import add_overlap_labels, evaluate_ensembles


def fixture_scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "validation_row": np.arange(6),
            "attack_cat": ["Benign", "Benign", "Fuzzers", "Fuzzers", "DoS", "DoS"],
            "label": [0, 0, 1, 1, 1, 1],
            "baseline_attack_score": [0.8, 0.1, 0.9, 0.2, 0.8, 0.1],
            "context_attack_score": [0.7, 0.2, 0.2, 0.9, 0.8, 0.1],
        }
    )


def test_error_overlap_assigns_all_four_attack_cases() -> None:
    overlap = add_overlap_labels(fixture_scores(), 0.5, 0.5)

    assert set(overlap.loc[overlap["label"] == 1, "attack_overlap"]) == {
        "both_catch",
        "baseline_catch_context_miss",
        "baseline_miss_context_catch",
        "both_miss",
    }
    assert overlap.loc[0, "benign_overlap"] == "both_false_positive"
    assert overlap.loc[1, "benign_overlap"] == "both_true_negative"


def test_blend_endpoints_reproduce_saved_model_scores() -> None:
    overlap = add_overlap_labels(fixture_scores(), 0.5, 0.5)
    _, _, _, candidates = evaluate_ensembles(overlap)

    np.testing.assert_allclose(
        candidates["blend_alpha_0.00"], overlap["baseline_attack_score"]
    )
    np.testing.assert_allclose(
        candidates["blend_alpha_1.00"], overlap["context_attack_score"]
    )
    np.testing.assert_allclose(
        candidates["max_score"],
        np.maximum(
            overlap["baseline_attack_score"], overlap["context_attack_score"]
        ),
    )


def test_ensemble_module_has_no_locked_holdout_data_path() -> None:
    assert "locked_holdout_features.parquet" not in inspect.getsource(
        analyze_v2_ensemble
    )
