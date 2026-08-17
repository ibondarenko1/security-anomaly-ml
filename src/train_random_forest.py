"""Train one Random Forest candidate using development data only.

The saved training-fitted preprocessor is reused without refitting. The
official UNSW testing set is not loaded, transformed, or evaluated.
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
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

try:
    from .evaluate import attack_probabilities, evaluate_binary_classifier
    from .preprocess import (
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        prepare_development_split,
        transform_network_flows,
    )
except ImportError:
    from evaluate import attack_probabilities, evaluate_binary_classifier
    from preprocess import (
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        prepare_development_split,
        transform_network_flows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.joblib"
RANDOM_FOREST_METRICS_PATH = MODELS_DIR / "random_forest_metrics.json"

COMPARISON_METRICS = {
    "precision_attack": "Precision",
    "recall_attack": "Recall",
    "f1_attack": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
    "false_positive_rate": "FPR",
    "false_negative_rate": "FNR",
    "false_positives": "False positives",
    "false_negatives": "False negatives",
}
VERY_HIGH_RECALL_FLOOR = 0.97
MAX_ALLOWED_RECALL_DROP = 0.02


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 file hash for reproducibility metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_with_timing(
    model: RandomForestClassifier, features: Any, target: pd.Series
) -> tuple[float, list[dict[str, str]]]:
    """Fit the one candidate model while recording time and warnings."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        training_start = perf_counter()
        model.fit(features, target)
        training_seconds = perf_counter() - training_start

    warning_report = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in captured
    ]
    return training_seconds, warning_report


def top_feature_importances(
    model: RandomForestClassifier, feature_names: np.ndarray, limit: int = 20
) -> list[dict[str, Any]]:
    """Return ranked feature importances without performing selection."""
    assert len(feature_names) == len(model.feature_importances_)
    ranked_indices = np.argsort(model.feature_importances_)[::-1][:limit]
    return [
        {
            "rank": rank,
            "feature": str(feature_names[index]),
            "importance": float(model.feature_importances_[index]),
        }
        for rank, index in enumerate(ranked_indices, start=1)
    ]


def main() -> None:
    """Run the single-candidate Random Forest comparison stage."""
    for path in (BASELINE_METRICS_PATH, PREPROCESSOR_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required saved baseline artifact missing: {path}")

    baseline_document = json.loads(
        BASELINE_METRICS_PATH.read_text(encoding="utf-8")
    )
    assert baseline_document["official_test"]["used"] is False
    logistic_metrics = baseline_document["models"]["logistic_regression"][
        "metrics"
    ]

    split = prepare_development_split()
    assert not hasattr(split, "X_test"), (
        "Development split unexpectedly contains official test data"
    )
    assert len(split.X_train_raw) == 140_272
    assert len(split.X_validation_raw) == 35_069

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler = preprocessor.named_transformers_["numeric"]
    assert int(scaler.n_samples_seen_) == len(split.X_train_raw), (
        "Saved preprocessor was not fit on exactly the development train rows"
    )

    X_train = transform_network_flows(preprocessor, split.X_train_raw)
    X_validation = transform_network_flows(
        preprocessor, split.X_validation_raw
    )
    feature_names = preprocessor.get_feature_names_out()
    assert X_train.shape == (140_272, len(feature_names))
    assert X_validation.shape == (35_069, len(feature_names))

    random_forest = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    training_seconds, training_warnings = fit_with_timing(
        random_forest, X_train, split.y_train
    )

    training_metrics, _, _ = evaluate_binary_classifier(
        random_forest, X_train, split.y_train
    )
    validation_metrics, validation_predictions, validation_probabilities = (
        evaluate_binary_classifier(
            random_forest, X_validation, split.y_validation
        )
    )

    top_20 = top_feature_importances(random_forest, feature_names, limit=20)
    assert np.isclose(random_forest.feature_importances_.sum(), 1.0)

    comparison_rows = []
    for metric_key, display_name in COMPARISON_METRICS.items():
        logistic_value = logistic_metrics[metric_key]
        forest_value = validation_metrics[metric_key]
        comparison_rows.append(
            {
                "metric": display_name,
                "metric_key": metric_key,
                "logistic_regression": logistic_value,
                "random_forest": forest_value,
                "difference_random_forest_minus_logistic": (
                    forest_value - logistic_value
                ),
            }
        )

    false_positive_reduction = (
        logistic_metrics["false_positives"]
        - validation_metrics["false_positives"]
    )
    materially_reduces_false_positives = bool(
        validation_metrics["false_positive_rate"]
        <= logistic_metrics["false_positive_rate"] - 0.02
        and validation_metrics["false_positives"]
        <= logistic_metrics["false_positives"] * 0.80
    )
    preserves_very_high_recall = bool(
        validation_metrics["recall_attack"] >= VERY_HIGH_RECALL_FLOOR
        and validation_metrics["recall_attack"]
        >= logistic_metrics["recall_attack"] - MAX_ALLOWED_RECALL_DROP
    )
    primary_criterion_met = bool(
        materially_reduces_false_positives and preserves_very_high_recall
    )

    overfitting_gaps = {
        metric: training_metrics[metric] - validation_metrics[metric]
        for metric in (
            "accuracy",
            "precision_attack",
            "recall_attack",
            "f1_attack",
            "roc_auc",
        )
    }
    evidence_of_overfitting = bool(
        overfitting_gaps["accuracy"] > 0.02
        or overfitting_gaps["f1_attack"] > 0.02
        or overfitting_gaps["roc_auc"] > 0.02
    )

    current_leader = (
        "Random Forest" if primary_criterion_met else "Logistic Regression"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(random_forest, RANDOM_FOREST_PATH, compress=3)

    reloaded_forest = joblib.load(RANDOM_FOREST_PATH)
    assert np.array_equal(
        validation_predictions, reloaded_forest.predict(X_validation)
    )
    reloaded_validation_probabilities = attack_probabilities(
        reloaded_forest, X_validation
    )
    probability_max_absolute_difference = float(
        np.max(
            np.abs(
                validation_probabilities - reloaded_validation_probabilities
            )
        )
    )
    assert np.allclose(
        validation_probabilities,
        reloaded_validation_probabilities,
        rtol=1e-12,
        atol=1e-15,
    )

    metrics_document = {
        "stage": "random_forest_model_comparison",
        "task": "binary network security anomaly detection",
        "official_test": {
            "used": False,
            "statement": (
                "Official UNSW testing data and labels were not loaded, "
                "transformed, evaluated, or used for fitting in this stage."
            ),
        },
        "data": {
            "source_file": str(OFFICIAL_TRAINING_FILE.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_file(OFFICIAL_TRAINING_FILE),
            "development_train_rows": len(split.X_train_raw),
            "validation_rows": len(split.X_validation_raw),
            "X_train_shape": list(X_train.shape),
            "X_validation_shape": list(X_validation.shape),
        },
        "preprocessing": {
            "artifact": str(PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)),
            "artifact_sha256": sha256_file(PREPROCESSOR_PATH),
            "refit": False,
            "fit_source": "development training subset only",
            "transformed_feature_count": len(feature_names),
        },
        "model": {
            "name": "RandomForestClassifier",
            "configuration": {
                "n_estimators": 300,
                "random_state": 42,
                "n_jobs": -1,
                "classification_threshold": 0.5,
            },
            "training_seconds": training_seconds,
            "warnings": training_warnings,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
        },
        "comparison_with_logistic_regression": {
            "baseline_metrics_artifact": str(
                BASELINE_METRICS_PATH.relative_to(PROJECT_ROOT)
            ),
            "baseline_metrics_sha256": sha256_file(BASELINE_METRICS_PATH),
            "rows": comparison_rows,
            "false_positive_reduction_count": false_positive_reduction,
            "materially_reduces_false_positives": (
                materially_reduces_false_positives
            ),
            "preserves_very_high_recall": preserves_very_high_recall,
            "very_high_recall_floor": VERY_HIGH_RECALL_FLOOR,
            "max_allowed_recall_drop": MAX_ALLOWED_RECALL_DROP,
            "primary_criterion_met": primary_criterion_met,
            "current_leading_candidate": current_leader,
        },
        "overfitting_check": {
            "training_minus_validation": overfitting_gaps,
            "evidence_of_overfitting": evidence_of_overfitting,
        },
        "top_20_feature_importances": top_20,
        "artifacts": {
            "random_forest": str(RANDOM_FOREST_PATH.relative_to(PROJECT_ROOT)),
            "random_forest_sha256": sha256_file(RANDOM_FOREST_PATH),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "reproducibility_checks": {
            "saved_model_predictions_match": True,
            "saved_model_probabilities_match_within_tolerance": True,
            "saved_model_probability_max_absolute_difference": (
                probability_max_absolute_difference
            ),
            "probability_comparison_rtol": 1e-12,
            "probability_comparison_atol": 1e-15,
            "existing_preprocessor_reused_without_refit": True,
            "official_test_accessed": False,
            "new_candidate_model_count": 1,
            "hyperparameter_tuning": False,
            "threshold_optimization": False,
        },
    }

    RANDOM_FOREST_METRICS_PATH.write_text(
        json.dumps(metrics_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded_metrics = json.loads(
        RANDOM_FOREST_METRICS_PATH.read_text(encoding="utf-8")
    )
    assert loaded_metrics["official_test"]["used"] is False
    assert (
        loaded_metrics["reproducibility_checks"]["new_candidate_model_count"]
        == 1
    )

    print("UNSW-NB15 RANDOM FOREST VALIDATION REPORT")
    print("Official test used: no")
    print(f"Training time: {training_seconds:.6f} seconds")
    print(f"Training warnings: {training_warnings or 'none'}")
    print("\nTraining metrics:")
    for metric in (
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
    ):
        print(f"  {metric}: {training_metrics[metric]:.6f}")

    print("\nValidation metrics:")
    for metric in (
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "average_precision",
        "false_positive_rate",
        "false_negative_rate",
    ):
        print(f"  {metric}: {validation_metrics[metric]:.6f}")
    print(
        "  validation_inference_seconds: "
        f"{validation_metrics['validation_inference_seconds']:.6f}"
    )
    print("  confusion_matrix:")
    for label, value in validation_metrics["confusion_matrix"].items():
        print(f"    {label}: {value:,}")

    print("\nComparison table (Random Forest difference vs Logistic):")
    for row in comparison_rows:
        print(
            f"  {row['metric']}: Logistic={row['logistic_regression']:.6f} "
            f"RandomForest={row['random_forest']:.6f} "
            f"Difference={row['difference_random_forest_minus_logistic']:.6f}"
        )

    print("\nTop 20 feature importances:")
    for item in top_20:
        print(
            f"  {item['rank']:>2}. {item['feature']}: "
            f"{item['importance']:.8f}"
        )

    print("\nDecision checks:")
    print(f"  False positives reduced by: {false_positive_reduction:,}")
    print(
        "  Materially reduces false positives: "
        f"{materially_reduces_false_positives}"
    )
    print(f"  Preserves very high recall: {preserves_very_high_recall}")
    print(f"  Primary criterion met: {primary_criterion_met}")
    print(f"  Evidence of overfitting: {evidence_of_overfitting}")
    print(f"  Current leading candidate: {current_leader}")
    print("\nSaved artifacts:")
    print(f"  Model: {RANDOM_FOREST_PATH}")
    print(f"  Metrics: {RANDOM_FOREST_METRICS_PATH}")


if __name__ == "__main__":
    main()
