"""Run the one-time locked evaluation on the official UNSW test set.

This module is called only after final training and artifact persistence have
completed. It never fits preprocessing or a model, and it never changes the
locked operational threshold.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from .evaluate import attack_probabilities
    from .preprocess import (
        EXCLUDED_COLUMNS,
        OFFICIAL_TEST_FILE,
        TARGET_COLUMN,
        assert_binary_target,
        find_unknown_categories,
        get_learned_categories,
        load_official_test,
        separate_features_and_target,
        transform_network_flows,
    )
except ImportError:
    from evaluate import attack_probabilities
    from preprocess import (
        EXCLUDED_COLUMNS,
        OFFICIAL_TEST_FILE,
        TARGET_COLUMN,
        assert_binary_target,
        find_unknown_categories,
        get_learned_categories,
        load_official_test,
        separate_features_and_target,
        transform_network_flows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

FINAL_MODEL_PATH = MODELS_DIR / "final_random_forest.joblib"
FINAL_PREPROCESSOR_PATH = MODELS_DIR / "final_preprocessor.joblib"
FINAL_TEST_METRICS_PATH = MODELS_DIR / "final_test_metrics.json"
FINAL_CATEGORY_METRICS_PATH = (
    MODELS_DIR / "final_attack_category_metrics.csv"
)
FINAL_MANIFEST_PATH = MODELS_DIR / "final_model_manifest.json"
SELECTED_THRESHOLD_PATH = MODELS_DIR / "selected_threshold.json"

LOCKED_THRESHOLD = 0.45
DIAGNOSTIC_THRESHOLD = 0.50
RECALL_TARGET = 0.985
MATERIAL_FPR_INCREASE = 0.01
OOF_CONSISTENCY_TOLERANCE = 0.02

ATTACK_CATEGORY_ORDER = (
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
)

COMPARISON_METRICS = {
    "precision_attack": "Precision",
    "recall_attack": "Recall",
    "f1_attack": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
}


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_threshold_metrics(
    target: pd.Series,
    attack_scores: np.ndarray,
    threshold: float,
    include_score_metrics: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate locked binary decisions from continuous attack scores."""
    target_array = target.to_numpy(dtype=np.int8, copy=False)
    predictions = (attack_scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        target_array, predictions, labels=[0, 1]
    ).ravel()
    actual_normal = int(tn + fp)
    actual_attack = int(tp + fn)
    assert actual_normal > 0 and actual_attack > 0

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(target_array, predictions)),
        "precision_attack": float(
            precision_score(
                target_array,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_attack": float(
            recall_score(
                target_array,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_attack": float(
            f1_score(
                target_array,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "specificity": float(tn / actual_normal),
        "false_positive_rate": float(fp / actual_normal),
        "false_negative_rate": float(fn / actual_attack),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "actual_normal": actual_normal,
        "actual_attack": actual_attack,
        "confusion_matrix": {
            "true_normal_predicted_normal": int(tn),
            "true_normal_predicted_attack": int(fp),
            "true_attack_predicted_normal": int(fn),
            "true_attack_predicted_attack": int(tp),
        },
    }
    if include_score_metrics:
        metrics["roc_auc"] = float(
            roc_auc_score(target_array, attack_scores)
        )
        metrics["average_precision"] = float(
            average_precision_score(target_array, attack_scores)
        )
    return metrics, predictions


def build_attack_category_metrics(
    test_data: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Diagnose binary detection recall within each true attack category."""
    attack_mask = test_data[TARGET_COLUMN].to_numpy() == 1
    attack_categories = test_data.loc[attack_mask, "attack_cat"].astype(str)
    observed_categories = set(attack_categories.unique().tolist())
    expected_categories = set(ATTACK_CATEGORY_ORDER)
    assert observed_categories == expected_categories, (
        "Unexpected official-test attack categories: "
        f"observed={sorted(observed_categories)}"
    )

    attack_predictions = predictions[attack_mask]
    rows: list[dict[str, Any]] = []
    for category in ATTACK_CATEGORY_ORDER:
        category_mask = attack_categories.to_numpy() == category
        flows = int(np.sum(category_mask))
        true_positives = int(np.sum(attack_predictions[category_mask] == 1))
        false_negatives = flows - true_positives
        assert flows > 0
        rows.append(
            {
                "attack_category": category,
                "flows": flows,
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "detection_recall": float(true_positives / flows),
            }
        )

    category_metrics = pd.DataFrame(rows)
    assert int(category_metrics["flows"].sum()) == int(np.sum(attack_mask))
    assert int(category_metrics["true_positives"].sum()) == int(
        np.sum(predictions[attack_mask] == 1)
    )
    assert int(category_metrics["false_negatives"].sum()) == int(
        np.sum(predictions[attack_mask] == 0)
    )
    return category_metrics


def load_oof_reference() -> tuple[dict[str, float], dict[str, Any]]:
    """Load the threshold decision that was frozen before test access."""
    if not SELECTED_THRESHOLD_PATH.is_file():
        raise FileNotFoundError(
            f"Threshold-selection artifact missing: {SELECTED_THRESHOLD_PATH}"
        )
    report = json.loads(SELECTED_THRESHOLD_PATH.read_text(encoding="utf-8"))
    assert report["official_test"]["used"] is False
    assert report["selected_threshold"] == LOCKED_THRESHOLD
    assert report["selection_policy"]["recall_requirement"] == RECALL_TARGET

    selected = report["oof_metrics_at_selected_threshold"]
    score_metrics = report["oof_score_metrics"]
    reference = {
        "precision_attack": float(selected["precision_attack"]),
        "recall_attack": float(selected["recall_attack"]),
        "f1_attack": float(selected["f1_attack"]),
        "roc_auc": float(score_metrics["roc_auc"]),
        "average_precision": float(
            score_metrics["average_precision_pr_auc"]
        ),
        "false_positive_rate": float(selected["false_positive_rate"]),
        "false_negative_rate": float(selected["false_negative_rate"]),
    }
    return reference, report["selection_policy"]


def compare_to_oof(
    oof_reference: dict[str, float],
    test_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build locked OOF-versus-official-test comparisons."""
    return [
        {
            "metric": display_name,
            "metric_key": metric_key,
            "oof_estimate": oof_reference[metric_key],
            "official_test": test_metrics[metric_key],
            "difference_test_minus_oof": (
                test_metrics[metric_key] - oof_reference[metric_key]
            ),
        }
        for metric_key, display_name in COMPARISON_METRICS.items()
    ]


def run_final_evaluation(
    model: Any,
    preprocessor: Any,
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Load and evaluate the official test after final training is complete."""
    assert training_metadata["training_complete"] is True
    assert FINAL_MODEL_PATH.is_file()
    assert FINAL_PREPROCESSOR_PATH.is_file()

    oof_reference, threshold_selection_policy = load_oof_reference()

    # This is the one and only official-test CSV load in the final stage. No
    # object is fitted or modified after this point.
    official_test = load_official_test()
    test_features, test_target = separate_features_and_target(official_test)
    assert len(official_test) == 82_332
    assert_binary_target(test_target, "official test")
    assert test_features.columns.tolist() == list(
        preprocessor.feature_names_in_
    )
    excluded = {TARGET_COLUMN, *EXCLUDED_COLUMNS}
    assert not excluded.intersection(test_features.columns)

    learned_categories = get_learned_categories(
        preprocessor, training_metadata["categorical_features"]
    )
    unknown_test_categories = find_unknown_categories(
        test_features, learned_categories
    )

    inference_start = perf_counter()
    transform_start = perf_counter()
    transformed_test = transform_network_flows(preprocessor, test_features)
    test_transform_seconds = perf_counter() - transform_start
    assert transformed_test.shape == (
        len(test_features),
        training_metadata["encoded_feature_count"],
    )

    scoring_start = perf_counter()
    attack_scores = attack_probabilities(model, transformed_test)
    model_scoring_seconds = perf_counter() - scoring_start
    total_inference_seconds = perf_counter() - inference_start
    assert np.isfinite(attack_scores).all()
    assert np.all((attack_scores >= 0.0) & (attack_scores <= 1.0))

    locked_metrics, locked_predictions = calculate_threshold_metrics(
        test_target,
        attack_scores,
        LOCKED_THRESHOLD,
        include_score_metrics=True,
    )
    diagnostic_metrics, _ = calculate_threshold_metrics(
        test_target,
        attack_scores,
        DIAGNOSTIC_THRESHOLD,
        include_score_metrics=False,
    )

    category_metrics = build_attack_category_metrics(
        official_test, locked_predictions
    )
    category_metrics.to_csv(FINAL_CATEGORY_METRICS_PATH, index=False)

    oof_comparison = compare_to_oof(oof_reference, locked_metrics)
    fpr_difference = (
        locked_metrics["false_positive_rate"]
        - oof_reference["false_positive_rate"]
    )
    recall_target_survives = bool(
        locked_metrics["recall_attack"] >= RECALL_TARGET
    )
    fpr_materially_increases = bool(
        fpr_difference >= MATERIAL_FPR_INCREASE
    )
    test_consistent_with_oof = bool(
        all(
            abs(row["difference_test_minus_oof"])
            <= OOF_CONSISTENCY_TOLERANCE
            for row in oof_comparison
        )
    )

    threshold_diagnostic_rows = []
    for metrics in (locked_metrics, diagnostic_metrics):
        threshold_diagnostic_rows.append(
            {
                key: metrics[key]
                for key in (
                    "threshold",
                    "precision_attack",
                    "recall_attack",
                    "false_positive_rate",
                    "false_negative_rate",
                    "false_positives",
                    "false_negatives",
                )
            }
        )

    # Required persistence check: reload both artifacts and repeat only the
    # predictions on the already-loaded official test. Metrics are not used to
    # alter any locked decision.
    reloaded_preprocessor = joblib.load(FINAL_PREPROCESSOR_PATH)
    reloaded_model = joblib.load(FINAL_MODEL_PATH)
    reloaded_test = transform_network_flows(
        reloaded_preprocessor, test_features
    )
    reloaded_scores = attack_probabilities(reloaded_model, reloaded_test)
    reloaded_predictions = (
        reloaded_scores >= LOCKED_THRESHOLD
    ).astype(np.int8)
    scores_max_absolute_difference = float(
        np.max(np.abs(attack_scores - reloaded_scores))
    )
    predictions_identical = bool(
        np.array_equal(locked_predictions, reloaded_predictions)
    )
    scores_within_machine_precision = bool(
        np.allclose(
            attack_scores,
            reloaded_scores,
            rtol=1e-12,
            atol=1e-15,
        )
    )
    assert predictions_identical
    assert scores_within_machine_precision

    model_sha256 = sha256_file(FINAL_MODEL_PATH)
    preprocessor_sha256 = sha256_file(FINAL_PREPROCESSOR_PATH)
    category_metrics_sha256 = sha256_file(FINAL_CATEGORY_METRICS_PATH)

    metrics_report = {
        "stage": "final_locked_official_test_evaluation",
        "locked_model": {
            "hyperparameters": training_metadata["model_hyperparameters"],
            "hyperparameters_changed_after_tuning": False,
        },
        "locked_operational_threshold": LOCKED_THRESHOLD,
        "score_interpretation": (
            "RandomForestClassifier.predict_proba class-1 output is treated "
            "as an attack score, not a calibrated real-world probability."
        ),
        "official_test_metrics": locked_metrics,
        "official_test_threshold_0_50_diagnostic": diagnostic_metrics,
        "threshold_diagnostic_comparison": threshold_diagnostic_rows,
        "oof_reference_at_threshold_0_45": oof_reference,
        "oof_vs_official_test": oof_comparison,
        "operational_assessment": {
            "recall_target": RECALL_TARGET,
            "recall_target_survives": recall_target_survives,
            "fpr_difference_test_minus_oof": fpr_difference,
            "material_fpr_increase_definition": (
                "Official-test FPR is at least 0.01 above OOF FPR"
            ),
            "fpr_materially_increases": fpr_materially_increases,
            "oof_consistency_definition": (
                "Absolute official-test minus OOF difference is <= 0.02 for "
                "precision, recall, F1, ROC-AUC, PR-AUC, FPR, and FNR"
            ),
            "test_performance_consistent_with_oof": test_consistent_with_oof,
        },
        "attack_category_metrics": category_metrics.to_dict(
            orient="records"
        ),
        "data": {
            "official_test_file": str(
                OFFICIAL_TEST_FILE.relative_to(PROJECT_ROOT)
            ),
            "official_test_sha256": sha256_file(OFFICIAL_TEST_FILE),
            "test_rows": len(official_test),
            "actual_normal_flows": locked_metrics["actual_normal"],
            "actual_attack_flows": locked_metrics["actual_attack"],
            "raw_feature_count": test_features.shape[1],
            "encoded_feature_count": transformed_test.shape[1],
            "unknown_test_categories": unknown_test_categories,
        },
        "runtime": {
            "test_transform_seconds": test_transform_seconds,
            "model_scoring_seconds": model_scoring_seconds,
            "total_test_inference_seconds": total_inference_seconds,
            "training": training_metadata["runtime"],
        },
        "reproducibility": {
            "model_reloaded": True,
            "preprocessor_reloaded": True,
            "predictions_identical": predictions_identical,
            "attack_scores_within_machine_precision": (
                scores_within_machine_precision
            ),
            "attack_score_max_absolute_difference": (
                scores_max_absolute_difference
            ),
            "score_comparison_rtol": 1e-12,
            "score_comparison_atol": 1e-15,
        },
        "process_lock": {
            "model_selection_completed_before_test_access": True,
            "threshold_selection_completed_before_test_access": True,
            "official_test_evaluations": 1,
            "reload_prediction_checks": 1,
            "tuning_after_test_access": False,
            "final_model_retraining_after_test_access": False,
        },
        "artifacts": {
            "model": str(FINAL_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "model_sha256": model_sha256,
            "preprocessor": str(
                FINAL_PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)
            ),
            "preprocessor_sha256": preprocessor_sha256,
            "attack_category_metrics": str(
                FINAL_CATEGORY_METRICS_PATH.relative_to(PROJECT_ROOT)
            ),
            "attack_category_metrics_sha256": category_metrics_sha256,
            "manifest": str(FINAL_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    FINAL_TEST_METRICS_PATH.write_text(
        json.dumps(metrics_report, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "stage": "final_locked_candidate",
        "dataset_filenames": {
            "training": training_metadata["training_file"],
            "official_test": str(
                OFFICIAL_TEST_FILE.relative_to(PROJECT_ROOT)
            ),
        },
        "dataset_sha256": {
            "training": training_metadata["training_sha256"],
            "official_test": sha256_file(OFFICIAL_TEST_FILE),
        },
        "training_row_count": training_metadata["training_rows"],
        "test_row_count": len(official_test),
        "feature_count_before_encoding": training_metadata[
            "raw_feature_count"
        ],
        "feature_count_after_encoding": training_metadata[
            "encoded_feature_count"
        ],
        "excluded_columns": ["id", "attack_cat", "label"],
        "numeric_features": training_metadata["numeric_features"],
        "categorical_features": training_metadata["categorical_features"],
        "model_hyperparameters": training_metadata[
            "model_hyperparameters"
        ],
        "operational_threshold": LOCKED_THRESHOLD,
        "threshold_selection_policy": threshold_selection_policy,
        "score_interpretation": metrics_report["score_interpretation"],
        "software": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "training_runtime": training_metadata["runtime"],
        "test_inference_runtime": metrics_report["runtime"],
        "artifact_sha256": {
            "final_random_forest": model_sha256,
            "final_preprocessor": preprocessor_sha256,
            "final_test_metrics": sha256_file(FINAL_TEST_METRICS_PATH),
            "final_attack_category_metrics": category_metrics_sha256,
        },
        "locked_evaluation": {
            "model_selection_completed_before_official_test": True,
            "threshold_selection_completed_before_official_test": True,
            "official_test_used_once_for_final_evaluation": True,
            "tuning_after_official_test": False,
        },
        "reproducibility": metrics_report["reproducibility"],
    }
    FINAL_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    strict_metrics = json.loads(
        FINAL_TEST_METRICS_PATH.read_text(encoding="utf-8")
    )
    strict_manifest = json.loads(
        FINAL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert strict_metrics["process_lock"]["tuning_after_test_access"] is False
    assert strict_manifest["operational_threshold"] == LOCKED_THRESHOLD

    print("\nFINAL LOCKED OFFICIAL TEST EVALUATION")
    print(f"Official test rows: {len(official_test):,}")
    print(f"Operational threshold: {LOCKED_THRESHOLD:.2f}")
    print("\nOfficial test metrics:")
    for metric in (
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "average_precision",
        "specificity",
        "false_positive_rate",
        "false_negative_rate",
    ):
        print(f"  {metric}: {locked_metrics[metric]:.6f}")
    for metric in (
        "actual_normal",
        "actual_attack",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    ):
        print(f"  {metric}: {locked_metrics[metric]:,}")

    print("\nOOF vs official test:")
    for row in oof_comparison:
        print(row)

    print("\nThreshold 0.45 vs 0.50:")
    for row in threshold_diagnostic_rows:
        print(row)

    print("\nAttack categories:")
    print(category_metrics.to_string(index=False))

    print("\nOperational assessment:")
    print(f"  Recall target survives: {recall_target_survives}")
    print(f"  FPR materially increases: {fpr_materially_increases}")
    print(f"  Test consistent with OOF: {test_consistent_with_oof}")
    print(
        "  Total test inference time: "
        f"{total_inference_seconds:.6f} seconds"
    )
    print("  Tuning after official test: no")
    print("\nSaved final evaluation artifacts:")
    for path in (
        FINAL_TEST_METRICS_PATH,
        FINAL_CATEGORY_METRICS_PATH,
        FINAL_MANIFEST_PATH,
    ):
        print(f"  {path}")

    return metrics_report

