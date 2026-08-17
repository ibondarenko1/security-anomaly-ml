"""Package the locked v1 generalization and FPR investigation.

This script consumes previously generated per-row OOF and official-test
scores. It does not fit, tune, calibrate, or change the locked v1 model.
Official-test labels are used only for post-hoc diagnosis; a future v2 requires
a new untouched locked test for an unbiased final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

try:
    from .evaluate_final import (
        FINAL_MANIFEST_PATH,
        FINAL_MODEL_PATH,
        FINAL_PREPROCESSOR_PATH,
        FINAL_TEST_METRICS_PATH,
        LOCKED_THRESHOLD,
    )
    from .investigate_shift import (
        OOF_SCORES_PATH,
        TEST_SCORES_PATH,
        empirical_ks,
        json_safe,
        numeric_divergences,
        probability_divergences,
    )
    from .preprocess import (
        OFFICIAL_TEST_FILE,
        OFFICIAL_TRAINING_FILE,
        assert_binary_target,
        identify_feature_groups,
        load_official_test,
        load_official_training,
        separate_features_and_target,
    )
except ImportError:
    from evaluate_final import (
        FINAL_MANIFEST_PATH,
        FINAL_MODEL_PATH,
        FINAL_PREPROCESSOR_PATH,
        FINAL_TEST_METRICS_PATH,
        LOCKED_THRESHOLD,
    )
    from investigate_shift import (
        OOF_SCORES_PATH,
        TEST_SCORES_PATH,
        empirical_ks,
        json_safe,
        numeric_divergences,
        probability_divergences,
    )
    from preprocess import (
        OFFICIAL_TEST_FILE,
        OFFICIAL_TRAINING_FILE,
        assert_binary_target,
        identify_feature_groups,
        load_official_test,
        load_official_training,
        separate_features_and_target,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_DRIFT_PATH = MODELS_DIR / "final_feature_drift.csv"
CATEGORICAL_FREQUENCIES_PATH = (
    MODELS_DIR / "final_categorical_frequency_drift.csv"
)
SCORE_SUMMARY_PATH = MODELS_DIR / "final_score_distribution_summary.csv"
FALSE_POSITIVES_PATH = MODELS_DIR / "final_false_positives.csv"
FP_VS_TN_PATH = MODELS_DIR / "final_fp_vs_true_normal.csv"
FP_CATEGORICAL_DETAIL_PATH = (
    MODELS_DIR / "final_fp_vs_true_normal_categories.csv"
)
FP_CLUSTERS_PATH = MODELS_DIR / "final_fp_traffic_clusters.csv"
FUZZERS_COMPARISON_PATH = MODELS_DIR / "final_fuzzers_comparison.csv"
FUZZERS_CATEGORICAL_DETAIL_PATH = (
    MODELS_DIR / "final_fuzzers_category_comparison.csv"
)
FUZZERS_PREDICTIONS_PATH = MODELS_DIR / "final_fuzzers_predictions.csv"
REPORT_PATH = MODELS_DIR / "generalization_diagnosis.json"

REFERENCE_THRESHOLDS = (0.45, 0.50)
V2_RECALL_TARGET = 0.985
V2_VALIDATION_FPR_TARGET = 0.10

OUTPUT_PATHS = (
    FEATURE_DRIFT_PATH,
    CATEGORICAL_FREQUENCIES_PATH,
    SCORE_SUMMARY_PATH,
    FALSE_POSITIVES_PATH,
    FP_VS_TN_PATH,
    FP_CATEGORICAL_DETAIL_PATH,
    FP_CLUSTERS_PATH,
    FUZZERS_COMPARISON_PATH,
    FUZZERS_CATEGORICAL_DETAIL_PATH,
    FUZZERS_PREDICTIONS_PATH,
    REPORT_PATH,
)


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_inputs_and_new_outputs() -> None:
    """Require locked inputs and refuse to overwrite diagnostic outputs."""
    required_inputs = (
        FINAL_MANIFEST_PATH,
        FINAL_MODEL_PATH,
        FINAL_PREPROCESSOR_PATH,
        FINAL_TEST_METRICS_PATH,
        OOF_SCORES_PATH,
        TEST_SCORES_PATH,
        OFFICIAL_TRAINING_FILE,
        OFFICIAL_TEST_FILE,
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required diagnostic inputs missing: " + ", ".join(missing))
    existing = [str(path) for path in OUTPUT_PATHS if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite generalization outputs: "
            + ", ".join(existing)
        )


def normalized_categories(series: pd.Series) -> pd.Series:
    """Represent missing categorical values explicitly for diagnostics."""
    return series.fillna("<NA>").astype(str)


def category_counts(
    series: pd.Series, categories: list[str]
) -> np.ndarray:
    """Return counts in a fixed union vocabulary."""
    counts = normalized_categories(series).value_counts()
    return np.array(
        [counts.get(category, 0) for category in categories],
        dtype=np.float64,
    )


def distribution_dict(series: pd.Series) -> dict[str, float]:
    """Return category shares ordered by descending prevalence."""
    shares = normalized_categories(series).value_counts(normalize=True)
    return {
        str(category): float(share)
        for category, share in shares.items()
    }


def score_distribution_row(
    group: str, scores: np.ndarray
) -> dict[str, Any]:
    """Calculate the required score summary for one class/dataset group."""
    scores = np.asarray(scores, dtype=np.float64)
    percentiles = np.quantile(scores, [0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "group": group,
        "count": len(scores),
        "mean_score": float(np.mean(scores)),
        "median_score": float(percentiles[0]),
        "p75_score": float(percentiles[1]),
        "p90_score": float(percentiles[2]),
        "p95_score": float(percentiles[3]),
        "p99_score": float(percentiles[4]),
        "share_at_or_above_0_45": float(np.mean(scores >= 0.45)),
        "share_at_or_above_0_50": float(np.mean(scores >= 0.50)),
    }


def build_feature_drift(
    training_features: pd.DataFrame,
    training_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one sortable row per feature plus categorical frequencies."""
    train_normal = training_features.loc[training_target.to_numpy() == 0]
    test_normal = test_features.loc[test_target.to_numpy() == 0]
    rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    for feature in numeric_features:
        train_all = training_features[feature].to_numpy(dtype=np.float64)
        test_all = test_features[feature].to_numpy(dtype=np.float64)
        train_normal_values = train_normal[feature].to_numpy(dtype=np.float64)
        test_normal_values = test_normal[feature].to_numpy(dtype=np.float64)
        all_psi, _ = numeric_divergences(train_all, test_all)
        normal_psi, _ = numeric_divergences(
            train_normal_values, test_normal_values
        )
        all_ks = empirical_ks(train_all, test_all)
        normal_ks = empirical_ks(train_normal_values, test_normal_values)
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "drift_score": max(all_ks, normal_ks),
                "all_train_mean": float(np.mean(train_all)),
                "all_train_median": float(np.median(train_all)),
                "all_test_mean": float(np.mean(test_all)),
                "all_test_median": float(np.median(test_all)),
                "all_ks_statistic": all_ks,
                "all_psi": all_psi,
                "normal_train_mean": float(np.mean(train_normal_values)),
                "normal_train_median": float(np.median(train_normal_values)),
                "normal_test_mean": float(np.mean(test_normal_values)),
                "normal_test_median": float(np.median(test_normal_values)),
                "normal_ks_statistic": normal_ks,
                "normal_psi": normal_psi,
                "unseen_test_categories": "",
                "all_train_category_shares": "",
                "all_test_category_shares": "",
                "normal_train_category_shares": "",
                "normal_test_category_shares": "",
            }
        )

    for feature in categorical_features:
        populations = {
            "all": (training_features[feature], test_features[feature]),
            "normal": (train_normal[feature], test_normal[feature]),
        }
        metrics: dict[str, dict[str, float]] = {}
        for population, (train_values, test_values) in populations.items():
            categories = sorted(
                set(normalized_categories(train_values).unique()).union(
                    normalized_categories(test_values).unique()
                )
            )
            train_counts = category_counts(train_values, categories)
            test_counts = category_counts(test_values, categories)
            psi, js, total_variation = probability_divergences(
                train_counts, test_counts
            )
            metrics[population] = {
                "psi": psi,
                "js": js,
                "total_variation": total_variation,
            }
            train_count_series = normalized_categories(train_values).value_counts()
            test_count_series = normalized_categories(test_values).value_counts()
            for category in categories:
                train_count = int(train_count_series.get(category, 0))
                test_count = int(test_count_series.get(category, 0))
                category_rows.append(
                    {
                        "feature": feature,
                        "population": population,
                        "category": category,
                        "train_count": train_count,
                        "train_share": train_count / len(train_values),
                        "test_count": test_count,
                        "test_share": test_count / len(test_values),
                        "share_change_test_minus_train": (
                            test_count / len(test_values)
                            - train_count / len(train_values)
                        ),
                        "unseen_in_training": bool(
                            train_count == 0 and test_count > 0
                        ),
                    }
                )

        train_categories = set(
            normalized_categories(training_features[feature]).unique()
        )
        test_categories = set(
            normalized_categories(test_features[feature]).unique()
        )
        unseen = sorted(test_categories - train_categories)
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "drift_score": max(
                    metrics["all"]["total_variation"],
                    metrics["normal"]["total_variation"],
                ),
                "all_train_mean": np.nan,
                "all_train_median": np.nan,
                "all_test_mean": np.nan,
                "all_test_median": np.nan,
                "all_ks_statistic": np.nan,
                "all_psi": metrics["all"]["psi"],
                "normal_train_mean": np.nan,
                "normal_train_median": np.nan,
                "normal_test_mean": np.nan,
                "normal_test_median": np.nan,
                "normal_ks_statistic": np.nan,
                "normal_psi": metrics["normal"]["psi"],
                "all_total_variation": metrics["all"]["total_variation"],
                "normal_total_variation": metrics["normal"][
                    "total_variation"
                ],
                "all_jensen_shannon": metrics["all"]["js"],
                "normal_jensen_shannon": metrics["normal"]["js"],
                "unseen_test_categories": json.dumps(unseen),
                "all_train_category_shares": json.dumps(
                    distribution_dict(training_features[feature])
                ),
                "all_test_category_shares": json.dumps(
                    distribution_dict(test_features[feature])
                ),
                "normal_train_category_shares": json.dumps(
                    distribution_dict(train_normal[feature])
                ),
                "normal_test_category_shares": json.dumps(
                    distribution_dict(test_normal[feature])
                ),
            }
        )

    drift = pd.DataFrame(rows).sort_values(
        ["drift_score", "feature"], ascending=[False, True]
    )
    category_frequencies = pd.DataFrame(category_rows).sort_values(
        ["feature", "population", "test_share"],
        ascending=[True, True, False],
    )
    return drift, category_frequencies


def build_two_group_feature_comparison(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str,
    second_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare two labeled groups across every input feature."""
    rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for feature in numeric_features:
        first_values = first[feature].to_numpy(dtype=np.float64)
        second_values = second[feature].to_numpy(dtype=np.float64)
        psi, js = numeric_divergences(second_values, first_values)
        ks = empirical_ks(first_values, second_values)
        rows.append(
            {
                "feature": feature,
                "feature_type": "numeric",
                "difference_score": ks,
                f"{first_name}_mean": float(np.mean(first_values)),
                f"{first_name}_median": float(np.median(first_values)),
                f"{second_name}_mean": float(np.mean(second_values)),
                f"{second_name}_median": float(np.median(second_values)),
                "ks_statistic": ks,
                "psi": psi,
                "jensen_shannon": js,
                f"{first_name}_category_shares": "",
                f"{second_name}_category_shares": "",
            }
        )

    for feature in categorical_features:
        categories = sorted(
            set(normalized_categories(first[feature]).unique()).union(
                normalized_categories(second[feature]).unique()
            )
        )
        first_counts = category_counts(first[feature], categories)
        second_counts = category_counts(second[feature], categories)
        psi, js, total_variation = probability_divergences(
            second_counts, first_counts
        )
        first_distribution = distribution_dict(first[feature])
        second_distribution = distribution_dict(second[feature])
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "difference_score": total_variation,
                f"{first_name}_mean": np.nan,
                f"{first_name}_median": np.nan,
                f"{second_name}_mean": np.nan,
                f"{second_name}_median": np.nan,
                "ks_statistic": np.nan,
                "psi": psi,
                "jensen_shannon": js,
                "total_variation": total_variation,
                f"{first_name}_category_shares": json.dumps(
                    first_distribution
                ),
                f"{second_name}_category_shares": json.dumps(
                    second_distribution
                ),
            }
        )
        for category in categories:
            first_count = int(
                np.sum(normalized_categories(first[feature]) == category)
            )
            second_count = int(
                np.sum(normalized_categories(second[feature]) == category)
            )
            category_rows.append(
                {
                    "feature": feature,
                    "category": category,
                    f"{first_name}_count": first_count,
                    f"{first_name}_share": first_count / len(first),
                    f"{second_name}_count": second_count,
                    f"{second_name}_share": second_count / len(second),
                }
            )

    comparison = pd.DataFrame(rows).sort_values(
        ["difference_score", "feature"], ascending=[False, True]
    )
    category_detail = pd.DataFrame(category_rows).sort_values(
        ["feature", f"{first_name}_count"], ascending=[True, False]
    )
    return comparison, category_detail


def build_fp_clusters(
    test_data: pd.DataFrame, fp_mask: np.ndarray
) -> pd.DataFrame:
    """Measure false-alert concentration in normal traffic profiles."""
    normal_mask = test_data["label"].to_numpy() == 0
    normal = test_data.loc[normal_mask, ["proto", "service", "state"]].copy()
    normal["is_false_positive"] = fp_mask[normal_mask]
    clusters = (
        normal.groupby(
            ["proto", "service", "state"],
            dropna=False,
            observed=False,
        )["is_false_positive"]
        .agg(normal_flows="size", false_positives="sum")
        .reset_index()
    )
    clusters["true_normals"] = (
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


def main() -> None:
    """Create the requested immutable-v1 diagnostic package."""
    ensure_inputs_and_new_outputs()
    start = perf_counter()

    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    final_metrics = json.loads(
        FINAL_TEST_METRICS_PATH.read_text(encoding="utf-8")
    )
    assert manifest["operational_threshold"] == LOCKED_THRESHOLD == 0.45
    assert manifest["locked_evaluation"]["tuning_after_official_test"] is False
    assert sha256_file(FINAL_MODEL_PATH) == manifest["artifact_sha256"][
        "final_random_forest"
    ]
    assert sha256_file(FINAL_PREPROCESSOR_PATH) == manifest[
        "artifact_sha256"
    ]["final_preprocessor"]

    training_data = load_official_training()
    test_data = load_official_test()
    training_features, training_target = separate_features_and_target(
        training_data
    )
    test_features, test_target = separate_features_and_target(test_data)
    assert_binary_target(training_target, "official training")
    assert_binary_target(test_target, "official test")
    numeric_features, categorical_features = identify_feature_groups(
        training_features
    )
    assert training_features.columns.tolist() == test_features.columns.tolist()

    oof_rows = pd.read_csv(OOF_SCORES_PATH)
    test_rows = pd.read_csv(TEST_SCORES_PATH)
    assert len(oof_rows) == len(training_data) == 175_341
    assert len(test_rows) == len(test_data) == 82_332
    assert np.array_equal(oof_rows["label"].to_numpy(), training_target.to_numpy())
    assert np.array_equal(test_rows["label"].to_numpy(), test_target.to_numpy())
    assert np.array_equal(
        (test_rows["attack_score"].to_numpy() >= LOCKED_THRESHOLD).astype(int),
        test_rows["prediction_at_0_45"].to_numpy(),
    )

    oof_scores = oof_rows["attack_score"].to_numpy(dtype=np.float64)
    test_scores = test_rows["attack_score"].to_numpy(dtype=np.float64)
    training_labels = training_target.to_numpy(dtype=np.int8)
    test_labels = test_target.to_numpy(dtype=np.int8)
    score_summary = pd.DataFrame(
        [
            score_distribution_row("OOF normal", oof_scores[training_labels == 0]),
            score_distribution_row("OOF attack", oof_scores[training_labels == 1]),
            score_distribution_row(
                "Official-test normal", test_scores[test_labels == 0]
            ),
            score_distribution_row(
                "Official-test attack", test_scores[test_labels == 1]
            ),
        ]
    )
    score_summary.to_csv(SCORE_SUMMARY_PATH, index=False)

    feature_drift, categorical_frequencies = build_feature_drift(
        training_features,
        training_target,
        test_features,
        test_target,
        numeric_features,
        categorical_features,
    )
    feature_drift.to_csv(FEATURE_DRIFT_PATH, index=False)
    categorical_frequencies.to_csv(CATEGORICAL_FREQUENCIES_PATH, index=False)

    test_predictions = test_rows["prediction_at_0_45"].to_numpy(dtype=np.int8)
    fp_mask = (test_labels == 0) & (test_predictions == 1)
    tn_mask = (test_labels == 0) & (test_predictions == 0)
    fn_mask = (test_labels == 1) & (test_predictions == 0)
    assert int(np.sum(fp_mask)) == 11_179
    assert int(np.sum(fn_mask)) == 386
    assert int(np.sum(tn_mask)) == 25_821

    false_positives = test_data.loc[fp_mask].copy()
    false_positives.insert(
        0,
        "official_test_row_index",
        test_data.index.to_numpy()[fp_mask],
    )
    false_positives.insert(1, "attack_score", test_scores[fp_mask])
    false_positives.insert(2, "predicted_class", test_predictions[fp_mask])
    false_positives.to_csv(FALSE_POSITIVES_PATH, index=False)

    fp_vs_tn, fp_category_detail = build_two_group_feature_comparison(
        test_features.loc[fp_mask],
        test_features.loc[tn_mask],
        "false_positive",
        "true_normal",
        numeric_features,
        categorical_features,
    )
    fp_vs_tn.to_csv(FP_VS_TN_PATH, index=False)
    fp_category_detail.to_csv(FP_CATEGORICAL_DETAIL_PATH, index=False)

    fp_clusters = build_fp_clusters(test_data, fp_mask)
    fp_clusters.to_csv(FP_CLUSTERS_PATH, index=False)

    fuzzers_mask = (
        (test_labels == 1)
        & (test_data["attack_cat"].astype(str).to_numpy() == "Fuzzers")
    )
    fuzzers_detected_mask = fuzzers_mask & (test_predictions == 1)
    fuzzers_missed_mask = fuzzers_mask & (test_predictions == 0)
    assert int(np.sum(fuzzers_missed_mask)) == 349
    fuzzers_comparison, fuzzers_category_detail = (
        build_two_group_feature_comparison(
            test_features.loc[fuzzers_missed_mask],
            test_features.loc[fuzzers_detected_mask],
            "missed_fuzzer",
            "detected_fuzzer",
            numeric_features,
            categorical_features,
        )
    )
    fuzzers_comparison.to_csv(FUZZERS_COMPARISON_PATH, index=False)
    fuzzers_category_detail.to_csv(
        FUZZERS_CATEGORICAL_DETAIL_PATH, index=False
    )
    fuzzers_predictions = test_data.loc[fuzzers_mask].copy()
    fuzzers_predictions.insert(
        0,
        "official_test_row_index",
        test_data.index.to_numpy()[fuzzers_mask],
    )
    fuzzers_predictions.insert(1, "attack_score", test_scores[fuzzers_mask])
    fuzzers_predictions.insert(
        2, "predicted_class", test_predictions[fuzzers_mask]
    )
    fuzzers_predictions.insert(
        3,
        "detection_result",
        np.where(test_predictions[fuzzers_mask] == 1, "detected", "missed"),
    )
    fuzzers_predictions.to_csv(FUZZERS_PREDICTIONS_PATH, index=False)

    score_summary_indexed = score_summary.set_index("group")
    normal_oof = score_summary_indexed.loc["OOF normal"]
    normal_test = score_summary_indexed.loc["Official-test normal"]
    normal_score_shift = {
        "mean_shift_test_minus_oof": float(
            normal_test["mean_score"] - normal_oof["mean_score"]
        ),
        "median_shift_test_minus_oof": float(
            normal_test["median_score"] - normal_oof["median_score"]
        ),
        "p75_shift_test_minus_oof": float(
            normal_test["p75_score"] - normal_oof["p75_score"]
        ),
        "share_above_0_45_shift": float(
            normal_test["share_at_or_above_0_45"]
            - normal_oof["share_at_or_above_0_45"]
        ),
        "substantial_right_shift_observed": bool(
            normal_test["share_at_or_above_0_45"]
            - normal_oof["share_at_or_above_0_45"]
            >= 0.10
        ),
    }
    top_clusters = fp_clusters.head(4)
    top_four_fp_share = float(
        top_clusters["false_positives"].sum() / np.sum(fp_mask)
    )
    runtime_seconds = perf_counter() - start

    artifact_paths = (
        FEATURE_DRIFT_PATH,
        CATEGORICAL_FREQUENCIES_PATH,
        SCORE_SUMMARY_PATH,
        FALSE_POSITIVES_PATH,
        FP_VS_TN_PATH,
        FP_CATEGORICAL_DETAIL_PATH,
        FP_CLUSTERS_PATH,
        FUZZERS_COMPARISON_PATH,
        FUZZERS_CATEGORICAL_DETAIL_PATH,
        FUZZERS_PREDICTIONS_PATH,
    )
    report = {
        "stage": "generalization_fpr_investigation",
        "versioning": {
            "locked_baseline": "v1",
            "v1_model_retrained_or_incrementally_fitted": False,
            "v2_not_trained": True,
        },
        "scope_lock": {
            "model_changed": False,
            "threshold_changed": False,
            "hyperparameters_changed": False,
            "features_changed": False,
            "official_test_used_for_tuning": False,
            "official_test_role": "fixed post-hoc diagnostic benchmark",
        },
        "locked_v1_artifacts": {
            "model": str(FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "model_sha256": sha256_file(FINAL_MODEL_PATH),
            "preprocessor": str(
                FINAL_PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)
            ),
            "preprocessor_sha256": sha256_file(FINAL_PREPROCESSOR_PATH),
            "operational_threshold": LOCKED_THRESHOLD,
        },
        "score_distribution_summary": score_summary.to_dict(orient="records"),
        "normal_score_shift": normal_score_shift,
        "feature_drift": {
            "rows": len(feature_drift),
            "sort_column": "drift_score",
            "top_features": feature_drift.head(15).to_dict(orient="records"),
            "required_artifact": str(
                FEATURE_DRIFT_PATH.relative_to(PROJECT_ROOT)
            ),
        },
        "false_positives": {
            "count": int(np.sum(fp_mask)),
            "true_normal_count": int(np.sum(tn_mask)),
            "all_raw_rows_saved": True,
            "top_four_cluster_share": top_four_fp_share,
            "top_clusters": top_clusters.to_dict(orient="records"),
            "top_fp_vs_tn_features": fp_vs_tn.head(15).to_dict(
                orient="records"
            ),
        },
        "fuzzers": {
            "flows": int(np.sum(fuzzers_mask)),
            "detected": int(np.sum(fuzzers_detected_mask)),
            "missed": int(np.sum(fuzzers_missed_mask)),
            "recall": float(np.mean(test_predictions[fuzzers_mask] == 1)),
            "share_of_all_false_negatives": float(
                np.sum(fuzzers_missed_mask) / np.sum(fn_mask)
            ),
            "top_missed_vs_detected_features": (
                fuzzers_comparison.head(15).to_dict(orient="records")
            ),
        },
        "investigation_branch": {
            "evidence": (
                "Strong normal-score and raw-feature drift with false positives "
                "concentrated in a small number of traffic profiles."
            ),
            "next_experiment_class": (
                "dataset/feature-representation and contextual-feature design; "
                "calibration may be evaluated only inside training data"
            ),
            "no_remediation_selected_from_official_test": True,
        },
        "v2_acceptance_target": {
            "validation_recall_minimum": V2_RECALL_TARGET,
            "validation_fpr_strictly_below": V2_VALIDATION_FPR_TARGET,
            "requires_new_locked_test": True,
            "tier_1_temporal_context_features_deferred": True,
        },
        "data": {
            "official_training_rows": len(training_data),
            "official_test_rows": len(test_data),
            "official_training_sha256": sha256_file(OFFICIAL_TRAINING_FILE),
            "official_test_sha256": sha256_file(OFFICIAL_TEST_FILE),
        },
        "artifacts": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in artifact_paths
        },
        "runtime_seconds": runtime_seconds,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "locked_final_metrics_reference": {
            "recall": final_metrics["official_test_metrics"]["recall_attack"],
            "fpr": final_metrics["official_test_metrics"][
                "false_positive_rate"
            ],
            "false_positives": final_metrics["official_test_metrics"][
                "false_positives"
            ],
            "false_negatives": final_metrics["official_test_metrics"][
                "false_negatives"
            ],
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

    print("GENERALIZATION / FPR INVESTIGATION", flush=True)
    print("Locked v1 changed: no", flush=True)
    print("v2 trained: no", flush=True)
    print("\nScore distributions:", flush=True)
    print(score_summary.to_string(index=False), flush=True)
    print("\nTop feature drift:", flush=True)
    display_columns = [
        "feature",
        "feature_type",
        "drift_score",
        "all_ks_statistic",
        "all_psi",
        "normal_ks_statistic",
        "normal_psi",
        "unseen_test_categories",
    ]
    print(feature_drift[display_columns].head(15).to_string(index=False), flush=True)
    print("\nTop FP vs TN differences:", flush=True)
    print(
        fp_vs_tn[
            ["feature", "feature_type", "difference_score", "ks_statistic", "psi"]
        ].head(15).to_string(index=False),
        flush=True,
    )
    print("\nTop FP clusters:", flush=True)
    print(fp_clusters.head(10).to_string(index=False), flush=True)
    print("\nTop Fuzzers missed vs detected differences:", flush=True)
    print(
        fuzzers_comparison[
            ["feature", "feature_type", "difference_score", "ks_statistic", "psi"]
        ].head(15).to_string(index=False),
        flush=True,
    )
    print(f"\nFalse-positive rows saved: {int(np.sum(fp_mask)):,}", flush=True)
    print(f"Fuzzers misses: {int(np.sum(fuzzers_missed_mask)):,}", flush=True)
    print(f"Runtime: {runtime_seconds:.6f} seconds", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
