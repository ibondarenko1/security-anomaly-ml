"""Investigate post-hoc score and feature shift after locked evaluation.

This diagnostic stage does not tune, calibrate, retrain, or replace the final
candidate. Official-test labels are used only to explain the already-recorded
locked result; they are no longer an untouched model-selection resource.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .evaluate import attack_probabilities
    from .evaluate_final import (
        FINAL_MODEL_PATH,
        FINAL_PREPROCESSOR_PATH,
        FINAL_TEST_METRICS_PATH,
        LOCKED_THRESHOLD,
        calculate_threshold_metrics,
    )
    from .preprocess import (
        OFFICIAL_TEST_FILE,
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        assert_binary_target,
        build_preprocessor,
        identify_feature_groups,
        load_official_test,
        load_official_training,
        separate_features_and_target,
        transform_network_flows,
    )
    from .train_final import FROZEN_MODEL_HYPERPARAMETERS
except ImportError:
    from evaluate import attack_probabilities
    from evaluate_final import (
        FINAL_MODEL_PATH,
        FINAL_PREPROCESSOR_PATH,
        FINAL_TEST_METRICS_PATH,
        LOCKED_THRESHOLD,
        calculate_threshold_metrics,
    )
    from preprocess import (
        OFFICIAL_TEST_FILE,
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        assert_binary_target,
        build_preprocessor,
        identify_feature_groups,
        load_official_test,
        load_official_training,
        separate_features_and_target,
        transform_network_flows,
    )
    from train_final import FROZEN_MODEL_HYPERPARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DIAGNOSTICS_DIR = MODELS_DIR / "diagnostics"

REPORT_PATH = MODELS_DIR / "shift_investigation.json"
OOF_SCORES_PATH = DIAGNOSTICS_DIR / "oof_scores.csv"
TEST_SCORES_PATH = DIAGNOSTICS_DIR / "official_test_scores.csv"
SCORE_SUMMARY_PATH = DIAGNOSTICS_DIR / "score_distribution_summary.csv"
NUMERIC_DRIFT_PATH = DIAGNOSTICS_DIR / "numeric_feature_drift.csv"
CATEGORICAL_DRIFT_PATH = DIAGNOSTICS_DIR / "categorical_feature_drift.csv"
CATEGORICAL_PROFILES_PATH = (
    DIAGNOSTICS_DIR / "categorical_false_positive_profiles.csv"
)
FP_CLUSTERS_PATH = DIAGNOSTICS_DIR / "test_normal_fp_clusters.csv"
FUZZERS_NUMERIC_PATH = DIAGNOSTICS_DIR / "fuzzers_fn_vs_tp_features.csv"
FUZZERS_CATEGORICAL_PATH = (
    DIAGNOSTICS_DIR / "fuzzers_categorical_errors.csv"
)
SCORE_DISTRIBUTIONS_PLOT = DIAGNOSTICS_DIR / "score_distributions.png"
RELIABILITY_PLOT = DIAGNOSTICS_DIR / "reliability_curves.png"

CV_SPLITS = 3
CALIBRATION_BINS = 10
DIVERGENCE_BINS = 10
EPSILON = 1e-10
OOF_METRIC_REPRODUCTION_TOLERANCE = 0.002
TEST_SCORE_METRIC_REPRODUCTION_TOLERANCE = 1e-6

OUTPUT_PATHS = (
    REPORT_PATH,
    OOF_SCORES_PATH,
    TEST_SCORES_PATH,
    SCORE_SUMMARY_PATH,
    NUMERIC_DRIFT_PATH,
    CATEGORICAL_DRIFT_PATH,
    CATEGORICAL_PROFILES_PATH,
    FP_CLUSTERS_PATH,
    FUZZERS_NUMERIC_PATH,
    FUZZERS_CATEGORICAL_PATH,
    SCORE_DISTRIBUTIONS_PLOT,
    RELIABILITY_PLOT,
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_new_outputs() -> None:
    """Protect final and any existing diagnostic artifacts."""
    # A completed OOF score table is a resumable checkpoint because generating
    # it requires three expensive folds. Every other output remains protected.
    existing = [
        str(path)
        for path in OUTPUT_PATHS
        if path != OOF_SCORES_PATH and path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite diagnostic artifacts: "
            + ", ".join(existing)
        )
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def load_oof_checkpoint(
    training_data: pd.DataFrame, training_target: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Load and verify a completed per-row OOF score checkpoint."""
    checkpoint = pd.read_csv(OOF_SCORES_PATH)
    required_columns = {
        "official_training_row_index",
        "label",
        "fold",
        "attack_score",
        "prediction_at_0_45",
        "error_type",
    }
    assert set(checkpoint.columns) == required_columns
    assert len(checkpoint) == len(training_data)
    assert np.array_equal(
        checkpoint["official_training_row_index"].to_numpy(),
        training_data.index.to_numpy(),
    )
    assert np.array_equal(
        checkpoint["label"].to_numpy(), training_target.to_numpy()
    )
    scores = checkpoint["attack_score"].to_numpy(dtype=np.float64)
    folds = checkpoint["fold"].to_numpy(dtype=np.int8)
    predictions = (scores >= LOCKED_THRESHOLD).astype(np.int8)
    assert np.array_equal(
        predictions, checkpoint["prediction_at_0_45"].to_numpy()
    )
    assert np.array_equal(
        error_labels(training_target.to_numpy(), predictions),
        checkpoint["error_type"].to_numpy(),
    )
    assert np.isfinite(scores).all()
    assert set(np.unique(folds).tolist()) == {1, 2, 3}
    return scores, folds


def empirical_ks(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate the two-sample Kolmogorov-Smirnov statistic."""
    first_sorted = np.sort(np.asarray(first, dtype=np.float64))
    second_sorted = np.sort(np.asarray(second, dtype=np.float64))
    assert len(first_sorted) > 0 and len(second_sorted) > 0
    combined = np.sort(np.concatenate((first_sorted, second_sorted)))
    first_cdf = np.searchsorted(
        first_sorted, combined, side="right"
    ) / len(first_sorted)
    second_cdf = np.searchsorted(
        second_sorted, combined, side="right"
    ) / len(second_sorted)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def probability_divergences(
    reference_probabilities: np.ndarray,
    comparison_probabilities: np.ndarray,
) -> tuple[float, float, float]:
    """Calculate PSI, Jensen-Shannon divergence, and total variation."""
    reference = np.asarray(reference_probabilities, dtype=np.float64)
    comparison = np.asarray(comparison_probabilities, dtype=np.float64)
    reference = np.clip(reference, EPSILON, None)
    comparison = np.clip(comparison, EPSILON, None)
    reference /= reference.sum()
    comparison /= comparison.sum()
    midpoint = 0.5 * (reference + comparison)

    psi = float(
        np.sum((comparison - reference) * np.log(comparison / reference))
    )
    js = float(
        0.5 * np.sum(reference * np.log(reference / midpoint))
        + 0.5 * np.sum(comparison * np.log(comparison / midpoint))
    )
    total_variation = float(0.5 * np.sum(np.abs(reference - comparison)))
    return psi, js, total_variation


def numeric_divergences(
    reference: np.ndarray, comparison: np.ndarray
) -> tuple[float, float]:
    """Calculate PSI and JS using reference-quantile bins."""
    reference = np.asarray(reference, dtype=np.float64)
    comparison = np.asarray(comparison, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, DIVERGENCE_BINS + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) == 1:
        value = float(edges[0])
        edges = np.array(
            [
                -np.inf,
                np.nextafter(value, -np.inf),
                np.nextafter(value, np.inf),
                np.inf,
            ]
        )
    elif len(edges) == 2:
        midpoint = float(edges[0] + (edges[1] - edges[0]) / 2.0)
        edges = np.array([-np.inf, midpoint, np.inf])
    else:
        edges = edges.astype(np.float64)
        edges[0] = -np.inf
        edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    comparison_counts, _ = np.histogram(comparison, bins=edges)
    psi, js, _ = probability_divergences(
        reference_counts, comparison_counts
    )
    return psi, js


def score_summary(
    name: str, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Summarize a class-conditional score distribution."""
    scores = np.asarray(scores, dtype=np.float64)
    quantile_values = np.quantile(
        scores,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    return {
        "group": name,
        "count": len(scores),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "q01": float(quantile_values[0]),
        "q05": float(quantile_values[1]),
        "q10": float(quantile_values[2]),
        "q25": float(quantile_values[3]),
        "q50": float(quantile_values[4]),
        "q75": float(quantile_values[5]),
        "q90": float(quantile_values[6]),
        "q95": float(quantile_values[7]),
        "q99": float(quantile_values[8]),
        "max": float(np.max(scores)),
        "share_at_or_above_threshold": float(np.mean(scores >= threshold)),
    }


def generate_oof_scores(
    features: pd.DataFrame, target: pd.Series
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], float]:
    """Regenerate diagnostic OOF scores with fold-local preprocessing."""
    numeric_features, categorical_features = identify_feature_groups(features)
    target_array = target.to_numpy(dtype=np.int8, copy=True)
    scores = np.full(len(features), np.nan, dtype=np.float64)
    score_counts = np.zeros(len(features), dtype=np.uint8)
    fold_assignments = np.full(len(features), -1, dtype=np.int8)
    fold_reports: list[dict[str, Any]] = []

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    total_start = perf_counter()
    for fold, (train_positions, holdout_positions) in enumerate(
        cv.split(features, target_array), start=1
    ):
        fold_start = perf_counter()
        assert np.intersect1d(train_positions, holdout_positions).size == 0
        assert np.all(score_counts[holdout_positions] == 0)

        train_raw = features.iloc[train_positions]
        holdout_raw = features.iloc[holdout_positions]
        train_target = target.iloc[train_positions]
        preprocessor = build_preprocessor(
            numeric_features, categorical_features
        )

        preprocessing_start = perf_counter()
        train_matrix = preprocessor.fit_transform(train_raw)
        holdout_matrix = transform_network_flows(preprocessor, holdout_raw)
        preprocessing_seconds = perf_counter() - preprocessing_start
        scaler = preprocessor.named_transformers_["numeric"]
        assert int(scaler.n_samples_seen_) == len(train_positions)

        model = RandomForestClassifier(**FROZEN_MODEL_HYPERPARAMETERS)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            training_start = perf_counter()
            model.fit(train_matrix, train_target)
            training_seconds = perf_counter() - training_start
            inference_start = perf_counter()
            fold_scores = attack_probabilities(model, holdout_matrix)
            inference_seconds = perf_counter() - inference_start

        assert np.isfinite(fold_scores).all()
        scores[holdout_positions] = fold_scores
        score_counts[holdout_positions] += 1
        fold_assignments[holdout_positions] = fold
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": len(train_positions),
                "holdout_rows": len(holdout_positions),
                "encoded_features": train_matrix.shape[1],
                "preprocessing_seconds": preprocessing_seconds,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "total_seconds": perf_counter() - fold_start,
                "warnings": [
                    {
                        "category": warning.category.__name__,
                        "message": str(warning.message),
                    }
                    for warning in captured
                ],
            }
        )
        print(
            f"Diagnostic OOF fold {fold}/{CV_SPLITS}: "
            f"training={training_seconds:.3f}s, "
            f"inference={inference_seconds:.3f}s, "
            f"features={train_matrix.shape[1]}",
            flush=True,
        )
        del model, preprocessor, train_matrix, holdout_matrix, fold_scores

    total_seconds = perf_counter() - total_start
    assert np.all(score_counts == 1)
    assert np.isfinite(scores).all()
    assert set(np.unique(fold_assignments).tolist()) == {1, 2, 3}
    return scores, fold_assignments, fold_reports, total_seconds


def error_labels(target: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return TN/FP/FN/TP labels for each row."""
    target = np.asarray(target, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    labels = np.empty(len(target), dtype=object)
    labels[(target == 0) & (predictions == 0)] = "TN"
    labels[(target == 0) & (predictions == 1)] = "FP"
    labels[(target == 1) & (predictions == 0)] = "FN"
    labels[(target == 1) & (predictions == 1)] = "TP"
    return labels


def build_numeric_drift(
    training_normal: pd.DataFrame,
    test_normal: pd.DataFrame,
    oof_tn: pd.DataFrame,
    test_fp: pd.DataFrame,
    test_tn: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    """Quantify normal-population drift and FP/TN separation."""
    rows: list[dict[str, Any]] = []
    quantile_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    for feature in numeric_features:
        train_values = training_normal[feature].to_numpy(dtype=np.float64)
        test_values = test_normal[feature].to_numpy(dtype=np.float64)
        oof_tn_values = oof_tn[feature].to_numpy(dtype=np.float64)
        fp_values = test_fp[feature].to_numpy(dtype=np.float64)
        tn_values = test_tn[feature].to_numpy(dtype=np.float64)
        train_quantiles = np.quantile(train_values, quantile_levels)
        test_quantiles = np.quantile(test_values, quantile_levels)
        oof_tn_quantiles = np.quantile(oof_tn_values, quantile_levels)
        fp_quantiles = np.quantile(fp_values, quantile_levels)
        tn_quantiles = np.quantile(tn_values, quantile_levels)
        psi, js = numeric_divergences(train_values, test_values)
        rows.append(
            {
                "feature": feature,
                "normal_train_test_ks": empirical_ks(
                    train_values, test_values
                ),
                "normal_train_test_psi": psi,
                "normal_train_test_js": js,
                "oof_tn_test_fp_ks": empirical_ks(
                    oof_tn_values, fp_values
                ),
                "test_fp_tn_ks": empirical_ks(fp_values, tn_values),
                "train_normal_mean": float(np.mean(train_values)),
                "test_normal_mean": float(np.mean(test_values)),
                "oof_tn_mean": float(np.mean(oof_tn_values)),
                "test_fp_mean": float(np.mean(fp_values)),
                "test_tn_mean": float(np.mean(tn_values)),
                **{
                    f"train_normal_q{int(level * 100):02d}": float(value)
                    for level, value in zip(
                        quantile_levels, train_quantiles, strict=True
                    )
                },
                **{
                    f"test_normal_q{int(level * 100):02d}": float(value)
                    for level, value in zip(
                        quantile_levels, test_quantiles, strict=True
                    )
                },
                **{
                    f"oof_tn_q{int(level * 100):02d}": float(value)
                    for level, value in zip(
                        quantile_levels, oof_tn_quantiles, strict=True
                    )
                },
                **{
                    f"test_fp_q{int(level * 100):02d}": float(value)
                    for level, value in zip(
                        quantile_levels, fp_quantiles, strict=True
                    )
                },
                **{
                    f"test_tn_q{int(level * 100):02d}": float(value)
                    for level, value in zip(
                        quantile_levels, tn_quantiles, strict=True
                    )
                },
            }
        )
    return pd.DataFrame(rows).sort_values(
        "normal_train_test_ks", ascending=False
    )


def categorical_probabilities(
    values: pd.Series, categories: list[str]
) -> np.ndarray:
    """Return category probabilities in a fixed union vocabulary."""
    normalized = values.fillna("<NA>").astype(str)
    counts = normalized.value_counts()
    return np.array(
        [counts.get(category, 0) for category in categories],
        dtype=np.float64,
    )


def build_categorical_drift_and_profiles(
    training_normal: pd.DataFrame,
    test_normal: pd.DataFrame,
    oof_tn: pd.DataFrame,
    test_fp: pd.DataFrame,
    test_tn: pd.DataFrame,
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify categorical drift and enumerate false-positive profiles."""
    drift_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for feature in categorical_features:
        groups = {
            "train_normal": training_normal[feature].fillna("<NA>").astype(str),
            "test_normal": test_normal[feature].fillna("<NA>").astype(str),
            "oof_tn": oof_tn[feature].fillna("<NA>").astype(str),
            "test_fp": test_fp[feature].fillna("<NA>").astype(str),
            "test_tn": test_tn[feature].fillna("<NA>").astype(str),
        }
        categories = sorted(
            set().union(*(set(series.unique()) for series in groups.values()))
        )
        probabilities = {
            name: categorical_probabilities(series, categories)
            for name, series in groups.items()
        }
        normal_psi, normal_js, normal_tv = probability_divergences(
            probabilities["train_normal"], probabilities["test_normal"]
        )
        fp_tn_psi, fp_tn_js, fp_tn_tv = probability_divergences(
            probabilities["test_tn"], probabilities["test_fp"]
        )
        oof_tn_fp_psi, oof_tn_fp_js, oof_tn_fp_tv = (
            probability_divergences(
                probabilities["oof_tn"], probabilities["test_fp"]
            )
        )
        drift_rows.append(
            {
                "feature": feature,
                "category_count": len(categories),
                "normal_train_test_psi": normal_psi,
                "normal_train_test_js": normal_js,
                "normal_train_test_total_variation": normal_tv,
                "oof_tn_test_fp_psi": oof_tn_fp_psi,
                "oof_tn_test_fp_js": oof_tn_fp_js,
                "oof_tn_test_fp_total_variation": oof_tn_fp_tv,
                "test_fp_tn_psi": fp_tn_psi,
                "test_fp_tn_js": fp_tn_js,
                "test_fp_tn_total_variation": fp_tn_tv,
            }
        )

        counts = {
            name: series.value_counts()
            for name, series in groups.items()
        }
        for category in categories:
            train_count = int(counts["train_normal"].get(category, 0))
            test_count = int(counts["test_normal"].get(category, 0))
            oof_tn_count = int(counts["oof_tn"].get(category, 0))
            fp_count = int(counts["test_fp"].get(category, 0))
            tn_count = int(counts["test_tn"].get(category, 0))
            profile_rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "train_normal_count": train_count,
                    "train_normal_share": train_count / len(training_normal),
                    "test_normal_count": test_count,
                    "test_normal_share": test_count / len(test_normal),
                    "oof_tn_count": oof_tn_count,
                    "oof_tn_share": oof_tn_count / len(oof_tn),
                    "test_fp_count": fp_count,
                    "share_of_all_test_fp": fp_count / len(test_fp),
                    "test_tn_count": tn_count,
                    "false_alert_rate_within_category": (
                        fp_count / test_count if test_count else np.nan
                    ),
                }
            )

    drift = pd.DataFrame(drift_rows).sort_values(
        "normal_train_test_total_variation", ascending=False
    )
    profiles = pd.DataFrame(profile_rows).sort_values(
        ["feature", "test_fp_count"], ascending=[True, False]
    )
    return drift, profiles


def build_fp_clusters(
    test_normal: pd.DataFrame, normal_predictions: np.ndarray
) -> pd.DataFrame:
    """Group normal flows by proto/service/state and count false alerts."""
    cluster_data = test_normal[["proto", "service", "state"]].copy()
    cluster_data["is_false_positive"] = normal_predictions.astype(bool)
    clusters = (
        cluster_data.groupby(
            ["proto", "service", "state"],
            dropna=False,
            observed=False,
        )["is_false_positive"]
        .agg(normal_flows="size", false_positives="sum")
        .reset_index()
    )
    clusters["true_negatives"] = (
        clusters["normal_flows"] - clusters["false_positives"]
    )
    clusters["false_alert_rate"] = (
        clusters["false_positives"] / clusters["normal_flows"]
    )
    clusters["share_of_all_false_positives"] = (
        clusters["false_positives"] / clusters["false_positives"].sum()
    )
    return clusters.sort_values(
        ["false_positives", "normal_flows"], ascending=False
    )


def build_fuzzers_diagnostics(
    official_test: pd.DataFrame,
    test_features: pd.DataFrame,
    test_scores: np.ndarray,
    test_predictions: np.ndarray,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Explain the concentrated false negatives for Fuzzers."""
    fuzzers_mask = (
        (official_test["label"].to_numpy() == 1)
        & (official_test["attack_cat"].astype(str).to_numpy() == "Fuzzers")
    )
    detected_mask = fuzzers_mask & (test_predictions == 1)
    missed_mask = fuzzers_mask & (test_predictions == 0)
    assert int(np.sum(fuzzers_mask)) > 0
    assert int(np.sum(missed_mask)) > 0

    numeric_rows: list[dict[str, Any]] = []
    for feature in numeric_features:
        detected = test_features.loc[detected_mask, feature].to_numpy(
            dtype=np.float64
        )
        missed = test_features.loc[missed_mask, feature].to_numpy(
            dtype=np.float64
        )
        numeric_rows.append(
            {
                "feature": feature,
                "fn_vs_tp_ks": empirical_ks(missed, detected),
                "detected_mean": float(np.mean(detected)),
                "missed_mean": float(np.mean(missed)),
                "detected_q25": float(np.quantile(detected, 0.25)),
                "missed_q25": float(np.quantile(missed, 0.25)),
                "detected_q50": float(np.quantile(detected, 0.50)),
                "missed_q50": float(np.quantile(missed, 0.50)),
                "detected_q75": float(np.quantile(detected, 0.75)),
                "missed_q75": float(np.quantile(missed, 0.75)),
            }
        )
    numeric = pd.DataFrame(numeric_rows).sort_values(
        "fn_vs_tp_ks", ascending=False
    )

    categorical_rows: list[dict[str, Any]] = []
    for feature in categorical_features:
        detected_values = test_features.loc[detected_mask, feature].astype(str)
        missed_values = test_features.loc[missed_mask, feature].astype(str)
        categories = sorted(
            set(detected_values.unique()).union(missed_values.unique())
        )
        detected_counts = detected_values.value_counts()
        missed_counts = missed_values.value_counts()
        for category in categories:
            detected_count = int(detected_counts.get(category, 0))
            missed_count = int(missed_counts.get(category, 0))
            total = detected_count + missed_count
            categorical_rows.append(
                {
                    "feature": feature,
                    "category": category,
                    "fuzzer_flows": total,
                    "detected": detected_count,
                    "missed": missed_count,
                    "miss_rate": missed_count / total,
                    "share_of_all_fuzzer_misses": (
                        missed_count / int(np.sum(missed_mask))
                    ),
                }
            )
    categorical = pd.DataFrame(categorical_rows).sort_values(
        ["feature", "missed"], ascending=[True, False]
    )

    summary = {
        "flows": int(np.sum(fuzzers_mask)),
        "detected": int(np.sum(detected_mask)),
        "missed": int(np.sum(missed_mask)),
        "recall": float(np.mean(test_predictions[fuzzers_mask] == 1)),
        "share_of_all_test_false_negatives": float(
            np.sum(missed_mask)
            / np.sum(
                (official_test["label"].to_numpy() == 1)
                & (test_predictions == 0)
            )
        ),
        "detected_score_summary": score_summary(
            "fuzzers_detected", test_scores[detected_mask], LOCKED_THRESHOLD
        ),
        "missed_score_summary": score_summary(
            "fuzzers_missed", test_scores[missed_mask], LOCKED_THRESHOLD
        ),
        "top_numeric_fn_vs_tp_features": numeric.head(10).to_dict(
            orient="records"
        ),
    }
    return numeric, categorical, summary


def reliability_table(
    target: np.ndarray, scores: np.ndarray, bins: int
) -> pd.DataFrame:
    """Build equal-width reliability bins without fitting calibration."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, bins - 1)
    rows: list[dict[str, Any]] = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "count": int(np.sum(mask)),
                "mean_score": float(np.mean(scores[mask])),
                "observed_attack_rate": float(np.mean(target[mask])),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Calculate count-weighted absolute reliability error."""
    weights = table["count"] / table["count"].sum()
    return float(
        np.sum(
            weights
            * np.abs(table["mean_score"] - table["observed_attack_rate"])
        )
    )


def plot_score_distributions(
    oof_target: np.ndarray,
    oof_scores: np.ndarray,
    test_target: np.ndarray,
    test_scores: np.ndarray,
) -> None:
    """Plot class-conditional histograms and empirical CDFs."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 51)
    groups = (
        (
            axes[0, 0],
            oof_scores[oof_target == 0],
            test_scores[test_target == 0],
            "Normal-flow attack scores",
        ),
        (
            axes[0, 1],
            oof_scores[oof_target == 1],
            test_scores[test_target == 1],
            "Attack-flow attack scores",
        ),
    )
    for axis, oof_values, test_values, title in groups:
        axis.hist(
            oof_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="OOF",
        )
        axis.hist(
            test_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label="Official test",
        )
        axis.axvline(
            LOCKED_THRESHOLD,
            color="black",
            linestyle="--",
            label="Locked threshold 0.45",
        )
        axis.set(title=title, xlabel="Attack score", ylabel="Density")
        axis.legend()
        axis.grid(alpha=0.2)

    for axis, class_value, title in (
        (axes[1, 0], 0, "Normal-score empirical CDF"),
        (axes[1, 1], 1, "Attack-score empirical CDF"),
    ):
        for label, values in (
            ("OOF", oof_scores[oof_target == class_value]),
            ("Official test", test_scores[test_target == class_value]),
        ):
            sorted_values = np.sort(values)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            axis.plot(sorted_values, cdf, linewidth=2, label=label)
        axis.axvline(LOCKED_THRESHOLD, color="black", linestyle="--")
        axis.set(title=title, xlabel="Attack score", ylabel="Cumulative share")
        axis.legend()
        axis.grid(alpha=0.2)

    fig.suptitle(
        "UNSW-NB15 class-conditional score shift: OOF vs official test",
        fontsize=14,
    )
    fig.savefig(SCORE_DISTRIBUTIONS_PLOT, dpi=160)
    plt.close(fig)


def plot_reliability(
    oof_reliability: pd.DataFrame, test_reliability: pd.DataFrame
) -> None:
    """Plot diagnostic reliability curves without applying calibration."""
    fig, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axis.plot(
        oof_reliability["mean_score"],
        oof_reliability["observed_attack_rate"],
        marker="o",
        linewidth=2,
        label="OOF",
    )
    axis.plot(
        test_reliability["mean_score"],
        test_reliability["observed_attack_rate"],
        marker="o",
        linewidth=2,
        label="Official test",
    )
    axis.set(
        title="Random Forest attack-score reliability (diagnostic only)",
        xlabel="Mean attack score",
        ylabel="Observed attack rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(RELIABILITY_PLOT, dpi=160)
    plt.close(fig)


def json_safe(value: Any) -> Any:
    """Recursively replace numpy scalars and non-finite values for JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> None:
    """Run the locked post-hoc shift and error investigation."""
    ensure_new_outputs()
    stage_start = perf_counter()

    final_metrics_report = json.loads(
        FINAL_TEST_METRICS_PATH.read_text(encoding="utf-8")
    )
    assert final_metrics_report["locked_operational_threshold"] == LOCKED_THRESHOLD
    assert final_metrics_report["process_lock"]["tuning_after_test_access"] is False

    training_data = load_official_training()
    training_features, training_target = separate_features_and_target(
        training_data
    )
    assert_binary_target(training_target, "official training")
    numeric_features, categorical_features = identify_feature_groups(
        training_features
    )

    oof_checkpoint_reused = OOF_SCORES_PATH.is_file()
    if oof_checkpoint_reused:
        oof_scores, fold_assignments = load_oof_checkpoint(
            training_data, training_target
        )
        fold_reports: list[dict[str, Any]] = []
        oof_runtime: float | None = None
        print(
            f"Reusing verified OOF checkpoint: {OOF_SCORES_PATH}",
            flush=True,
        )
    else:
        oof_scores, fold_assignments, fold_reports, oof_runtime = (
            generate_oof_scores(training_features, training_target)
        )
    oof_predictions = (oof_scores >= LOCKED_THRESHOLD).astype(np.int8)
    oof_metrics, audit_oof_predictions = calculate_threshold_metrics(
        training_target,
        oof_scores,
        LOCKED_THRESHOLD,
        include_score_metrics=True,
    )
    assert np.array_equal(oof_predictions, audit_oof_predictions)
    expected_oof = final_metrics_report["oof_reference_at_threshold_0_45"]
    oof_reproduction_deltas: dict[str, float] = {}
    for report_key, reference_key in (
        ("precision_attack", "precision_attack"),
        ("recall_attack", "recall_attack"),
        ("f1_attack", "f1_attack"),
        ("roc_auc", "roc_auc"),
        ("average_precision", "average_precision"),
        ("false_positive_rate", "false_positive_rate"),
        ("false_negative_rate", "false_negative_rate"),
    ):
        oof_reproduction_deltas[report_key] = float(
            oof_metrics[report_key] - expected_oof[reference_key]
        )
    oof_max_absolute_metric_delta = max(
        abs(delta) for delta in oof_reproduction_deltas.values()
    )
    oof_metrics_reproduced_within_tolerance = bool(
        oof_max_absolute_metric_delta
        <= OOF_METRIC_REPRODUCTION_TOLERANCE
    )
    assert oof_metrics_reproduced_within_tolerance, (
        "Regenerated OOF metrics differ materially from the locked report: "
        f"{oof_reproduction_deltas}"
    )

    if not oof_checkpoint_reused:
        oof_score_table = pd.DataFrame(
            {
                "official_training_row_index": training_data.index,
                "label": training_target.to_numpy(),
                "fold": fold_assignments,
                "attack_score": oof_scores,
                "prediction_at_0_45": oof_predictions,
                "error_type": error_labels(
                    training_target.to_numpy(), oof_predictions
                ),
            }
        )
        oof_score_table.to_csv(OOF_SCORES_PATH, index=False)

    # Official test is now used strictly for post-hoc diagnostics. No model,
    # threshold, calibration mapping, or feature decision is changed here.
    official_test = load_official_test()
    test_features, test_target = separate_features_and_target(official_test)
    final_preprocessor = joblib.load(FINAL_PREPROCESSOR_PATH)
    final_model = joblib.load(FINAL_MODEL_PATH)
    transformed_test = transform_network_flows(
        final_preprocessor, test_features
    )
    test_scores = attack_probabilities(final_model, transformed_test)
    test_predictions = (test_scores >= LOCKED_THRESHOLD).astype(np.int8)
    test_metrics, audit_test_predictions = calculate_threshold_metrics(
        test_target,
        test_scores,
        LOCKED_THRESHOLD,
        include_score_metrics=True,
    )
    assert np.array_equal(test_predictions, audit_test_predictions)
    locked_test_metrics = final_metrics_report["official_test_metrics"]
    test_reproduction_deltas: dict[str, float] = {}
    for metric in (
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "average_precision",
        "false_positive_rate",
        "false_negative_rate",
        "false_positives",
        "false_negatives",
    ):
        test_reproduction_deltas[metric] = float(
            test_metrics[metric] - locked_test_metrics[metric]
        )
    test_max_absolute_metric_delta = max(
        abs(delta) for delta in test_reproduction_deltas.values()
    )
    test_metrics_reproduced_within_tolerance = bool(
        test_max_absolute_metric_delta
        <= TEST_SCORE_METRIC_REPRODUCTION_TOLERANCE
    )
    assert test_metrics_reproduced_within_tolerance, (
        "Reloaded final-test metrics differ beyond machine-scale tolerance: "
        f"{test_reproduction_deltas}"
    )

    test_score_table = pd.DataFrame(
        {
            "official_test_row_index": official_test.index,
            "id": official_test["id"].to_numpy(),
            "label": test_target.to_numpy(),
            "attack_cat": official_test["attack_cat"].astype(str).to_numpy(),
            "attack_score": test_scores,
            "prediction_at_0_45": test_predictions,
            "error_type": error_labels(test_target.to_numpy(), test_predictions),
        }
    )
    test_score_table.to_csv(TEST_SCORES_PATH, index=False)

    training_target_array = training_target.to_numpy(dtype=np.int8)
    test_target_array = test_target.to_numpy(dtype=np.int8)
    score_summaries = pd.DataFrame(
        [
            score_summary(
                "OOF normal",
                oof_scores[training_target_array == 0],
                LOCKED_THRESHOLD,
            ),
            score_summary(
                "OOF attack",
                oof_scores[training_target_array == 1],
                LOCKED_THRESHOLD,
            ),
            score_summary(
                "Official test normal",
                test_scores[test_target_array == 0],
                LOCKED_THRESHOLD,
            ),
            score_summary(
                "Official test attack",
                test_scores[test_target_array == 1],
                LOCKED_THRESHOLD,
            ),
        ]
    )
    score_summaries.to_csv(SCORE_SUMMARY_PATH, index=False)

    score_bins = np.linspace(0.0, 1.0, 21)
    class_score_drift: dict[str, dict[str, float]] = {}
    for class_name, class_value in (("normal", 0), ("attack", 1)):
        oof_class_scores = oof_scores[training_target_array == class_value]
        test_class_scores = test_scores[test_target_array == class_value]
        oof_counts, _ = np.histogram(oof_class_scores, bins=score_bins)
        test_counts, _ = np.histogram(test_class_scores, bins=score_bins)
        psi, js, total_variation = probability_divergences(
            oof_counts, test_counts
        )
        class_score_drift[class_name] = {
            "ks_statistic": empirical_ks(oof_class_scores, test_class_scores),
            "psi": psi,
            "jensen_shannon_divergence": js,
            "total_variation": total_variation,
            "mean_shift_test_minus_oof": float(
                np.mean(test_class_scores) - np.mean(oof_class_scores)
            ),
            "median_shift_test_minus_oof": float(
                np.median(test_class_scores) - np.median(oof_class_scores)
            ),
            "threshold_crossing_rate_oof": float(
                np.mean(oof_class_scores >= LOCKED_THRESHOLD)
            ),
            "threshold_crossing_rate_test": float(
                np.mean(test_class_scores >= LOCKED_THRESHOLD)
            ),
        }

    train_normal_mask = training_target_array == 0
    test_normal_mask = test_target_array == 0
    test_fp_mask = test_normal_mask & (test_predictions == 1)
    test_tn_mask = test_normal_mask & (test_predictions == 0)
    oof_tn_mask = train_normal_mask & (oof_predictions == 0)
    training_normal = training_features.loc[train_normal_mask]
    test_normal = test_features.loc[test_normal_mask]
    oof_tn = training_features.loc[oof_tn_mask]
    test_fp = test_features.loc[test_fp_mask]
    test_tn = test_features.loc[test_tn_mask]

    numeric_drift = build_numeric_drift(
        training_normal,
        test_normal,
        oof_tn,
        test_fp,
        test_tn,
        numeric_features,
    )
    numeric_drift.to_csv(NUMERIC_DRIFT_PATH, index=False)
    categorical_drift, categorical_profiles = (
        build_categorical_drift_and_profiles(
            training_normal,
            test_normal,
            oof_tn,
            test_fp,
            test_tn,
            categorical_features,
        )
    )
    categorical_drift.to_csv(CATEGORICAL_DRIFT_PATH, index=False)
    categorical_profiles.to_csv(CATEGORICAL_PROFILES_PATH, index=False)

    fp_clusters = build_fp_clusters(
        test_normal, test_predictions[test_normal_mask]
    )
    fp_clusters.to_csv(FP_CLUSTERS_PATH, index=False)

    fuzzers_numeric, fuzzers_categorical, fuzzers_summary = (
        build_fuzzers_diagnostics(
            official_test,
            test_features,
            test_scores,
            test_predictions,
            numeric_features,
            categorical_features,
        )
    )
    fuzzers_numeric.to_csv(FUZZERS_NUMERIC_PATH, index=False)
    fuzzers_categorical.to_csv(FUZZERS_CATEGORICAL_PATH, index=False)

    oof_reliability = reliability_table(
        training_target_array, oof_scores, CALIBRATION_BINS
    )
    test_reliability = reliability_table(
        test_target_array, test_scores, CALIBRATION_BINS
    )
    calibration = {
        "interpretation": (
            "Diagnostic only. No calibration model was fitted and the locked "
            "threshold/model were not changed."
        ),
        "oof": {
            "brier_score": float(
                brier_score_loss(training_target_array, oof_scores)
            ),
            "expected_calibration_error_10_bins": (
                expected_calibration_error(oof_reliability)
            ),
            "reliability_bins": oof_reliability.to_dict(orient="records"),
        },
        "official_test": {
            "brier_score": float(
                brier_score_loss(test_target_array, test_scores)
            ),
            "expected_calibration_error_10_bins": (
                expected_calibration_error(test_reliability)
            ),
            "reliability_bins": test_reliability.to_dict(orient="records"),
        },
    }

    plot_score_distributions(
        training_target_array,
        oof_scores,
        test_target_array,
        test_scores,
    )
    plot_reliability(oof_reliability, test_reliability)

    stage_runtime = perf_counter() - stage_start
    artifact_paths = (
        OOF_SCORES_PATH,
        TEST_SCORES_PATH,
        SCORE_SUMMARY_PATH,
        NUMERIC_DRIFT_PATH,
        CATEGORICAL_DRIFT_PATH,
        CATEGORICAL_PROFILES_PATH,
        FP_CLUSTERS_PATH,
        FUZZERS_NUMERIC_PATH,
        FUZZERS_CATEGORICAL_PATH,
        SCORE_DISTRIBUTIONS_PLOT,
        RELIABILITY_PLOT,
    )
    report = {
        "stage": "post_hoc_distribution_shift_and_error_investigation",
        "scope_lock": {
            "diagnostic_only": True,
            "model_changed": False,
            "hyperparameters_changed": False,
            "features_changed": False,
            "threshold_changed": False,
            "calibration_fitted": False,
            "retraining_decision_made": False,
            "official_test_is_no_longer_an_untouched_holdout": True,
            "future_changes_require_a_new_untouched_holdout": True,
        },
        "model_hyperparameters": FROZEN_MODEL_HYPERPARAMETERS,
        "locked_threshold": LOCKED_THRESHOLD,
        "data": {
            "official_training_file": str(
                OFFICIAL_TRAINING_FILE.relative_to(PROJECT_ROOT)
            ),
            "official_training_sha256": sha256_file(
                OFFICIAL_TRAINING_FILE
            ),
            "official_test_file": str(
                OFFICIAL_TEST_FILE.relative_to(PROJECT_ROOT)
            ),
            "official_test_sha256": sha256_file(OFFICIAL_TEST_FILE),
            "training_rows": len(training_data),
            "test_rows": len(official_test),
        },
        "oof_generation": {
            "cv": {
                "class": "StratifiedKFold",
                "n_splits": CV_SPLITS,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
            "preprocessing_fit_scope": "each fold's training rows only",
            "each_row_scored_exactly_once": True,
            "checkpoint_reused_after_completed_fold_generation": (
                oof_checkpoint_reused
            ),
            "metric_reproduction": {
                "tolerance": OOF_METRIC_REPRODUCTION_TOLERANCE,
                "deltas_regenerated_minus_locked": oof_reproduction_deltas,
                "maximum_absolute_delta": oof_max_absolute_metric_delta,
                "within_tolerance": oof_metrics_reproduced_within_tolerance,
                "note": (
                    "Parallel Random Forest execution may introduce tiny "
                    "floating-point/tie variability across separate runs."
                ),
            },
            "folds": fold_reports,
            "runtime_seconds": oof_runtime,
            "metrics_at_locked_threshold": oof_metrics,
        },
        "locked_test_metrics_reproduced": test_metrics,
        "locked_test_metric_reproduction": {
            "tolerance": TEST_SCORE_METRIC_REPRODUCTION_TOLERANCE,
            "deltas_regenerated_minus_locked": test_reproduction_deltas,
            "maximum_absolute_delta": test_max_absolute_metric_delta,
            "within_tolerance": test_metrics_reproduced_within_tolerance,
            "discrete_predictions_and_counts_identical": all(
                test_reproduction_deltas[metric] == 0.0
                for metric in (
                    "precision_attack",
                    "recall_attack",
                    "f1_attack",
                    "false_positive_rate",
                    "false_negative_rate",
                    "false_positives",
                    "false_negatives",
                )
            ),
        },
        "score_distribution_summary": score_summaries.to_dict(
            orient="records"
        ),
        "class_conditional_score_drift": class_score_drift,
        "calibration_diagnostics": calibration,
        "feature_drift": {
            "numeric_metric_definitions": {
                "ks": "two-sample empirical CDF maximum distance",
                "psi": "reference-quantile-bin population stability index",
                "js": "reference-quantile-bin Jensen-Shannon divergence",
            },
            "top_numeric_normal_train_vs_test": numeric_drift.head(15).to_dict(
                orient="records"
            ),
            "top_numeric_test_fp_vs_tn": numeric_drift.sort_values(
                "test_fp_tn_ks", ascending=False
            ).head(15).to_dict(orient="records"),
            "top_numeric_oof_tn_vs_test_fp": numeric_drift.sort_values(
                "oof_tn_test_fp_ks", ascending=False
            ).head(15).to_dict(orient="records"),
            "categorical_summary": categorical_drift.to_dict(
                orient="records"
            ),
            "top_false_positive_clusters": fp_clusters.head(20).to_dict(
                orient="records"
            ),
        },
        "fuzzers": fuzzers_summary,
        "runtime": {
            "oof_generation_seconds": oof_runtime,
            "oof_checkpoint_reused": oof_checkpoint_reused,
            "total_investigation_seconds": stage_runtime,
        },
        "artifacts": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in artifact_paths
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
    }
    REPORT_PATH.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nPOST-HOC SHIFT INVESTIGATION", flush=True)
    print("Model/threshold/calibration changes: none", flush=True)
    print("\nClass-conditional score drift:", flush=True)
    for class_name, metrics in class_score_drift.items():
        print(f"  {class_name}: {metrics}", flush=True)
    print("\nCalibration diagnostics:", flush=True)
    print(
        f"  OOF Brier={calibration['oof']['brier_score']:.6f}, "
        "ECE="
        f"{calibration['oof']['expected_calibration_error_10_bins']:.6f}",
        flush=True,
    )
    print(
        "  Test Brier="
        f"{calibration['official_test']['brier_score']:.6f}, "
        "ECE="
        f"{calibration['official_test']['expected_calibration_error_10_bins']:.6f}",
        flush=True,
    )
    print("\nTop numeric normal train-vs-test drift:", flush=True)
    print(
        numeric_drift[
            [
                "feature",
                "normal_train_test_ks",
                "normal_train_test_psi",
                "normal_train_test_js",
                "oof_tn_test_fp_ks",
                "test_fp_tn_ks",
            ]
        ].head(15).to_string(index=False),
        flush=True,
    )
    print("\nCategorical drift:", flush=True)
    print(categorical_drift.to_string(index=False), flush=True)
    print("\nTop false-positive clusters:", flush=True)
    print(fp_clusters.head(15).to_string(index=False), flush=True)
    print("\nFuzzers:", flush=True)
    print(
        {
            key: value
            for key, value in fuzzers_summary.items()
            if not key.endswith("summary")
            and key != "top_numeric_fn_vs_tp_features"
        },
        flush=True,
    )
    print("Top Fuzzers FN-vs-TP numeric features:", flush=True)
    print(
        fuzzers_numeric.head(10).to_string(index=False),
        flush=True,
    )
    if oof_runtime is None:
        print("\nOOF runtime: reused verified checkpoint", flush=True)
    else:
        print(f"\nOOF runtime: {oof_runtime:.6f} seconds", flush=True)
    print(f"Total runtime: {stage_runtime:.6f} seconds", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)
    print(f"Diagnostics directory: {DIAGNOSTICS_DIR}", flush=True)


if __name__ == "__main__":
    main()
