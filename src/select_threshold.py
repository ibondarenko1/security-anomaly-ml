"""Select an operational attack-score threshold from OOF predictions only.

The frozen tuned Random Forest and fold-local preprocessing are fitted in a
three-fold cross-validation loop over the complete official training set.
The official UNSW test set is never loaded, transformed, or evaluated.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from .evaluate import attack_probabilities
    from .preprocess import (
        EXCLUDED_COLUMNS,
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        TARGET_COLUMN,
        assert_binary_target,
        build_preprocessor,
        find_unknown_categories,
        get_learned_categories,
        identify_feature_groups,
        load_official_training,
        separate_features_and_target,
        transform_network_flows,
    )
except ImportError:
    from evaluate import attack_probabilities
    from preprocess import (
        EXCLUDED_COLUMNS,
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        TARGET_COLUMN,
        assert_binary_target,
        build_preprocessor,
        find_unknown_categories,
        get_learned_categories,
        identify_feature_groups,
        load_official_training,
        separate_features_and_target,
        transform_network_flows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
TUNED_RF_METRICS_PATH = MODELS_DIR / "tuned_random_forest_metrics.json"
THRESHOLD_RESULTS_PATH = MODELS_DIR / "threshold_results.csv"
SELECTED_THRESHOLD_PATH = MODELS_DIR / "selected_threshold.json"

CV_SPLITS = 3
RECALL_REQUIREMENT = 0.985
THRESHOLD_START = 0.10
THRESHOLD_STOP = 0.90
THRESHOLD_STEP = 0.01
REFERENCE_THRESHOLD = 0.50

FROZEN_MODEL_HYPERPARAMETERS = {
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

REPORT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 hash for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash a contiguous array, including its dtype and shape."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def threshold_metrics(
    target: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Calculate binary metrics for attack when score >= threshold."""
    predictions = scores >= threshold
    actual_attack = target == 1
    actual_normal = ~actual_attack

    true_positives = int(np.sum(predictions & actual_attack))
    false_positives = int(np.sum(predictions & actual_normal))
    false_negatives = int(np.sum(~predictions & actual_attack))
    true_negatives = int(np.sum(~predictions & actual_normal))

    attack_count = true_positives + false_negatives
    normal_count = true_negatives + false_positives
    predicted_attack_count = true_positives + false_positives
    assert attack_count > 0 and normal_count > 0

    precision = (
        true_positives / predicted_attack_count
        if predicted_attack_count
        else 0.0
    )
    recall = true_positives / attack_count
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    specificity = true_negatives / normal_count

    return {
        "threshold": float(threshold),
        "accuracy": float(
            (true_positives + true_negatives) / len(target)
        ),
        "precision_attack": float(precision),
        "recall_attack": float(recall),
        "f1_attack": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(false_positives / normal_count),
        "false_negative_rate": float(false_negatives / attack_count),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
    }


def select_threshold(
    results: pd.DataFrame,
) -> tuple[int, bool, str]:
    """Apply the recall constraint, FPR objective, and exact-count ties."""
    eligible = results[
        results["recall_attack"] >= RECALL_REQUIREMENT
    ].copy()
    recall_requirement_achieved = not eligible.empty

    if recall_requirement_achieved:
        minimum_false_positives = int(eligible["false_positives"].min())
        selection_pool = eligible[
            eligible["false_positives"] == minimum_false_positives
        ].copy()
        selection_reason = (
            "Recall requirement achieved; minimized FPR/false positives, "
            "then preferred precision, F1, and higher threshold."
        )
    else:
        maximum_true_positives = int(results["true_positives"].max())
        recall_pool = results[
            results["true_positives"] == maximum_true_positives
        ].copy()
        minimum_false_positives = int(
            recall_pool["false_positives"].min()
        )
        selection_pool = recall_pool[
            recall_pool["false_positives"] == minimum_false_positives
        ].copy()
        selection_reason = (
            "Recall requirement not achieved; selected the highest-recall "
            "candidate with the lowest FPR, then precision, F1, and threshold."
        )

    selection_pool = selection_pool.sort_values(
        by=["precision_attack", "f1_attack", "threshold"],
        ascending=[False, False, False],
    )
    selected_index = int(selection_pool.index[0])
    return selected_index, recall_requirement_achieved, selection_reason


def compact_metrics(row: pd.Series) -> dict[str, Any]:
    """Return the operational metrics needed in the JSON report."""
    fields = (
        "threshold",
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
        "false_positives",
        "false_negatives",
        "true_positives",
        "true_negatives",
    )
    record: dict[str, Any] = {}
    for field in fields:
        value = row[field]
        if field in (
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
        ):
            record[field] = int(value)
        else:
            record[field] = float(value)
    return record


def main() -> None:
    """Generate OOF scores and select one frozen-model threshold."""
    if not TUNED_RF_METRICS_PATH.is_file():
        raise FileNotFoundError(
            f"Required tuning report missing: {TUNED_RF_METRICS_PATH}"
        )

    tuning_report = json.loads(
        TUNED_RF_METRICS_PATH.read_text(encoding="utf-8")
    )
    assert tuning_report["official_test"]["used"] is False
    prior_configuration = tuning_report["final_model"]["configuration"]
    assert prior_configuration == FROZEN_MODEL_HYPERPARAMETERS, (
        "Frozen model hyperparameters differ from tuning result"
    )

    official_training = load_official_training()
    features, target = separate_features_and_target(official_training)
    assert len(features) == 175_341
    assert_binary_target(target, "official training")

    excluded = {TARGET_COLUMN, *EXCLUDED_COLUMNS}
    assert not excluded.intersection(features.columns)
    assert features.shape[1] == 42
    assert features.index.equals(official_training.index)

    numeric_features, categorical_features = identify_feature_groups(features)
    assert len(numeric_features) == 39
    assert len(categorical_features) == 3

    target_array = target.to_numpy(dtype=np.int8, copy=True)
    oof_scores = np.full(len(features), np.nan, dtype=np.float64)
    score_counts = np.zeros(len(features), dtype=np.uint8)
    fold_assignments = np.full(len(features), -1, dtype=np.int8)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_reports: list[dict[str, Any]] = []
    all_warnings: list[dict[str, Any]] = []

    cv_start = perf_counter()
    for fold_number, (train_positions, holdout_positions) in enumerate(
        cv.split(features, target_array), start=1
    ):
        fold_start = perf_counter()
        assert np.intersect1d(train_positions, holdout_positions).size == 0
        assert len(train_positions) + len(holdout_positions) == len(features)
        assert np.all(score_counts[holdout_positions] == 0), (
            "An official training row is assigned to multiple OOF folds"
        )

        fold_train_raw = features.iloc[train_positions]
        fold_holdout_raw = features.iloc[holdout_positions]
        fold_train_target = target.iloc[train_positions]

        preprocessor = build_preprocessor(
            numeric_features, categorical_features
        )
        preprocessing_start = perf_counter()
        fold_train_matrix = preprocessor.fit_transform(fold_train_raw)
        fold_holdout_matrix = transform_network_flows(
            preprocessor, fold_holdout_raw
        )
        preprocessing_seconds = perf_counter() - preprocessing_start

        scaler = preprocessor.named_transformers_["numeric"]
        assert int(scaler.n_samples_seen_) == len(train_positions)
        learned_categories = get_learned_categories(
            preprocessor, categorical_features
        )
        assert "-" in learned_categories["service"]
        unknown_categories = find_unknown_categories(
            fold_holdout_raw, learned_categories
        )

        model = RandomForestClassifier(**FROZEN_MODEL_HYPERPARAMETERS)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            training_start = perf_counter()
            model.fit(fold_train_matrix, fold_train_target)
            training_seconds = perf_counter() - training_start

            inference_start = perf_counter()
            fold_scores = attack_probabilities(model, fold_holdout_matrix)
            inference_seconds = perf_counter() - inference_start

        fold_warnings = [
            {
                "fold": fold_number,
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            for warning in captured
        ]
        all_warnings.extend(fold_warnings)

        assert len(fold_scores) == len(holdout_positions)
        assert np.isfinite(fold_scores).all()
        assert np.all((fold_scores >= 0.0) & (fold_scores <= 1.0))
        oof_scores[holdout_positions] = fold_scores
        score_counts[holdout_positions] += 1
        fold_assignments[holdout_positions] = fold_number

        fold_reports.append(
            {
                "fold": fold_number,
                "train_rows": len(train_positions),
                "holdout_rows": len(holdout_positions),
                "train_normal": int(
                    np.sum(target_array[train_positions] == 0)
                ),
                "train_attack": int(
                    np.sum(target_array[train_positions] == 1)
                ),
                "holdout_normal": int(
                    np.sum(target_array[holdout_positions] == 0)
                ),
                "holdout_attack": int(
                    np.sum(target_array[holdout_positions] == 1)
                ),
                "input_features": fold_train_raw.shape[1],
                "encoded_features": fold_train_matrix.shape[1],
                "unknown_holdout_categories": unknown_categories,
                "preprocessing_seconds": preprocessing_seconds,
                "training_seconds": training_seconds,
                "inference_seconds": inference_seconds,
                "total_seconds": perf_counter() - fold_start,
                "warnings": fold_warnings,
            }
        )
        print(
            f"Fold {fold_number}/{CV_SPLITS} complete: "
            f"train={len(train_positions):,}, "
            f"holdout={len(holdout_positions):,}, "
            f"features={fold_train_matrix.shape[1]}, "
            f"training={training_seconds:.3f}s, "
            f"inference={inference_seconds:.3f}s",
            flush=True,
        )

        del (
            model,
            preprocessor,
            fold_train_matrix,
            fold_holdout_matrix,
            fold_scores,
        )

    total_cv_seconds = perf_counter() - cv_start

    assert np.all(score_counts == 1), (
        "Every official training row must receive exactly one OOF score"
    )
    assert np.all(fold_assignments >= 1)
    assert np.isfinite(oof_scores).all()
    assert set(np.unique(fold_assignments).tolist()) == {1, 2, 3}

    roc_auc = float(roc_auc_score(target_array, oof_scores))
    average_precision = float(
        average_precision_score(target_array, oof_scores)
    )

    thresholds = np.round(
        np.arange(
            THRESHOLD_START,
            THRESHOLD_STOP + THRESHOLD_STEP / 2,
            THRESHOLD_STEP,
        ),
        2,
    )
    assert len(thresholds) == 81
    assert thresholds[0] == THRESHOLD_START
    assert thresholds[-1] == THRESHOLD_STOP

    threshold_results = pd.DataFrame(
        [
            threshold_metrics(target_array, oof_scores, threshold)
            for threshold in thresholds
        ]
    )
    selected_index, recall_achieved, selection_reason = select_threshold(
        threshold_results
    )
    threshold_results["meets_recall_requirement"] = (
        threshold_results["recall_attack"] >= RECALL_REQUIREMENT
    )
    threshold_results["selected"] = False
    threshold_results.loc[selected_index, "selected"] = True
    assert int(threshold_results["selected"].sum()) == 1

    selected_row = threshold_results.loc[selected_index]
    reference_matches = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            REFERENCE_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    assert len(reference_matches) == 1
    reference_row = reference_matches.iloc[0]

    threshold_results.to_csv(THRESHOLD_RESULTS_PATH, index=False)

    selected_metrics = compact_metrics(selected_row)
    reference_metrics = compact_metrics(reference_row)
    difference_vs_reference = {
        metric: selected_metrics[metric] - reference_metrics[metric]
        for metric in (
            "accuracy",
            "precision_attack",
            "recall_attack",
            "f1_attack",
            "specificity",
            "false_positive_rate",
            "false_negative_rate",
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
        )
    }

    report = {
        "stage": "operational_threshold_selection",
        "selected_threshold": float(selected_row["threshold"]),
        "selection_policy": {
            "primary_requirement": (
                "OOF attack recall must be at least 0.985"
            ),
            "secondary_objective": (
                "Among eligible thresholds, minimize OOF FPR"
            ),
            "tie_breakers": [
                "higher attack precision",
                "higher attack F1",
                "higher threshold",
            ],
            "recall_requirement": RECALL_REQUIREMENT,
            "recall_requirement_achieved": recall_achieved,
            "selection_reason": selection_reason,
            "threshold_rule": "predict attack when attack_score >= threshold",
            "accuracy_used_for_selection": False,
        },
        "oof_metrics_at_selected_threshold": selected_metrics,
        "oof_metrics_at_threshold_0_50": reference_metrics,
        "difference_selected_minus_0_50": difference_vs_reference,
        "oof_score_metrics": {
            "roc_auc": roc_auc,
            "average_precision_pr_auc": average_precision,
        },
        "score_interpretation": (
            "RandomForestClassifier.predict_proba class-1 output is used as "
            "an attack score. It is not described as a calibrated real-world "
            "probability; probability calibration has not been performed."
        ),
        "random_seed": RANDOM_STATE,
        "cv_configuration": {
            "class": "StratifiedKFold",
            "n_splits": CV_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "folds": fold_reports,
            "total_cross_validation_seconds": total_cv_seconds,
        },
        "model_hyperparameters": FROZEN_MODEL_HYPERPARAMETERS,
        "preprocessing": {
            "fit_scope": "each fold's training rows only",
            "numeric": "StandardScaler",
            "categorical": "OneHotEncoder(handle_unknown='ignore')",
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "excluded_columns": ["id", "attack_cat", "label"],
            "saved_fold_preprocessors": False,
        },
        "data": {
            "source": str(
                OFFICIAL_TRAINING_FILE.relative_to(PROJECT_ROOT)
            ),
            "source_sha256": sha256_file(OFFICIAL_TRAINING_FILE),
            "rows": len(features),
            "normal_rows": int(np.sum(target_array == 0)),
            "attack_rows": int(np.sum(target_array == 1)),
            "raw_input_features": features.shape[1],
        },
        "threshold_grid": {
            "start": THRESHOLD_START,
            "stop": THRESHOLD_STOP,
            "step": THRESHOLD_STEP,
            "candidate_count": len(thresholds),
        },
        "warnings": all_warnings,
        "official_test": {
            "used": False,
            "statement": (
                "Official UNSW testing data and labels were not loaded, "
                "transformed, evaluated, or used for threshold selection."
            ),
        },
        "final_model_training": {
            "performed": False,
            "fold_models_saved": False,
        },
        "reproducibility_checks": {
            "all_training_rows_scored_once": True,
            "score_count_min": int(score_counts.min()),
            "score_count_max": int(score_counts.max()),
            "no_row_scored_by_its_training_fold_model": True,
            "oof_scores_all_finite": True,
            "oof_scores_sha256": sha256_array(oof_scores),
            "fold_assignments_sha256": sha256_array(fold_assignments),
            "frozen_hyperparameters_match_tuning_report": True,
            "official_test_accessed": False,
        },
        "artifacts": {
            "threshold_results": str(
                THRESHOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)
            ),
            "threshold_results_sha256": sha256_file(
                THRESHOLD_RESULTS_PATH
            ),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    SELECTED_THRESHOLD_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    loaded_report = json.loads(
        SELECTED_THRESHOLD_PATH.read_text(encoding="utf-8")
    )
    assert loaded_report["official_test"]["used"] is False
    assert loaded_report["selected_threshold"] == float(
        selected_row["threshold"]
    )

    display_thresholds = sorted(
        set(REPORT_THRESHOLDS) | {float(selected_row["threshold"])}
    )
    display_rows = threshold_results[
        threshold_results["threshold"].isin(display_thresholds)
    ]

    print("\nUNSW-NB15 OOF THRESHOLD-SELECTION REPORT")
    print("Official test used: no")
    print(f"OOF rows scored exactly once: {len(oof_scores):,}")
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC / Average Precision: {average_precision:.6f}")
    print(f"Total cross-validation time: {total_cv_seconds:.6f} seconds")
    print("\nThreshold comparison:")
    for _, row in display_rows.iterrows():
        marker = " [SELECTED]" if bool(row["selected"]) else ""
        print(
            f"  threshold={row['threshold']:.2f}{marker} "
            f"precision={row['precision_attack']:.6f} "
            f"recall={row['recall_attack']:.6f} "
            f"FPR={row['false_positive_rate']:.6f} "
            f"FNR={row['false_negative_rate']:.6f} "
            f"FP={int(row['false_positives']):,} "
            f"FN={int(row['false_negatives']):,}"
        )

    print("\nSelection:")
    print(f"  selected_threshold: {selected_row['threshold']:.2f}")
    print(f"  recall_requirement_achieved: {recall_achieved}")
    print(f"  recall: {selected_row['recall_attack']:.6f}")
    print(f"  precision: {selected_row['precision_attack']:.6f}")
    print(f"  F1: {selected_row['f1_attack']:.6f}")
    print(f"  FPR: {selected_row['false_positive_rate']:.6f}")
    print(f"  FNR: {selected_row['false_negative_rate']:.6f}")
    print(f"  false_positives: {int(selected_row['false_positives']):,}")
    print(f"  false_negatives: {int(selected_row['false_negatives']):,}")
    print("\nDifference selected minus threshold 0.50:")
    for metric, difference in difference_vs_reference.items():
        if metric in (
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
        ):
            print(f"  {metric}: {int(difference):+d}")
        else:
            print(f"  {metric}: {difference:+.6f}")

    print("\nSaved artifacts:")
    print(f"  Threshold table: {THRESHOLD_RESULTS_PATH}")
    print(f"  Selection report: {SELECTED_THRESHOLD_PATH}")
    print("  Final production candidate trained: no")


if __name__ == "__main__":
    main()
