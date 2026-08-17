"""Train and validate exactly two UNSW-NB15 baseline classifiers.

This stage reads only the official training CSV. The official test CSV is not
loaded, transformed, evaluated, or used for model/preprocessor fitting.
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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

try:
    from .evaluate import attack_probabilities, evaluate_binary_classifier
    from .preprocess import (
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        VALIDATION_SIZE,
        prepare_development_data,
        transform_network_flows,
    )
except ImportError:
    from evaluate import attack_probabilities, evaluate_binary_classifier
    from preprocess import (
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        VALIDATION_SIZE,
        prepare_development_data,
        transform_network_flows,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
MODEL_PATH = MODELS_DIR / "baseline_logistic_regression.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def sha256_file(path: Path) -> str:
    """Calculate a reproducibility hash without loading a whole file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_with_timing(model: Any, features: Any, target: pd.Series) -> tuple[float, list[dict[str, str]]]:
    """Fit one model and capture elapsed time and all emitted warnings."""
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


def matrices_match(left: Any, right: Any) -> bool:
    """Compare sparse or dense matrices without densifying sparse data."""
    if hasattr(left, "tocsr") and hasattr(right, "tocsr"):
        difference = left.tocsr() - right.tocsr()
        return difference.nnz == 0
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def print_model_report(name: str, report: dict[str, Any]) -> None:
    """Print security-focused validation results for one model."""
    metrics = report["metrics"]
    confusion = metrics["confusion_matrix"]
    print(f"\n{name}")
    print(f"  Training time: {report['training_seconds']:.6f} seconds")
    print(
        "  Validation inference time: "
        f"{metrics['validation_inference_seconds']:.6f} seconds"
    )
    print(f"  Accuracy: {metrics['accuracy']:.6f}")
    print(f"  Precision (attack): {metrics['precision_attack']:.6f}")
    print(f"  Recall (attack): {metrics['recall_attack']:.6f}")
    print(f"  F1 (attack): {metrics['f1_attack']:.6f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"  PR-AUC / Average Precision: {metrics['average_precision']:.6f}")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.6f}")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.6f}")
    print("  Confusion matrix:")
    for label, value in confusion.items():
        print(f"    {label}: {value:,}")
    print(f"  Warnings: {report['warnings'] or 'none'}")


def main() -> None:
    """Run the reproducible development-only baseline experiment."""
    development = prepare_development_data()
    assert not hasattr(development, "X_test"), (
        "Development preprocessing unexpectedly contains official test data"
    )
    assert len(development.X_train_raw) == 140_272
    assert len(development.X_validation_raw) == 35_069

    dummy = DummyClassifier(strategy="most_frequent")
    logistic_regression = LogisticRegression(max_iter=1000, solver="lbfgs")

    dummy_training_seconds, dummy_warnings = fit_with_timing(
        dummy, development.X_train, development.y_train
    )
    dummy_metrics, _, _ = evaluate_binary_classifier(
        dummy, development.X_validation, development.y_validation
    )

    logistic_training_seconds, logistic_warnings = fit_with_timing(
        logistic_regression, development.X_train, development.y_train
    )
    logistic_metrics, logistic_predictions, logistic_probabilities = (
        evaluate_binary_classifier(
            logistic_regression,
            development.X_validation,
            development.y_validation,
        )
    )

    model_reports = {
        "dummy_classifier": {
            "configuration": {"strategy": "most_frequent"},
            "training_seconds": dummy_training_seconds,
            "warnings": dummy_warnings,
            "metrics": dummy_metrics,
        },
        "logistic_regression": {
            "configuration": {"max_iter": 1000, "solver": "lbfgs"},
            "training_seconds": logistic_training_seconds,
            "iterations": logistic_regression.n_iter_.astype(int).tolist(),
            "warnings": logistic_warnings,
            "metrics": logistic_metrics,
        },
    }

    comparison_metrics = (
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
        "average_precision",
        "false_positive_rate",
        "false_negative_rate",
    )
    metric_deltas = {
        metric: logistic_metrics[metric] - dummy_metrics[metric]
        for metric in comparison_metrics
    }
    meaningfully_beats_dummy = bool(
        logistic_metrics["roc_auc"] > dummy_metrics["roc_auc"] + 0.10
        and logistic_metrics["average_precision"]
        > dummy_metrics["average_precision"] + 0.05
        and logistic_metrics["false_positive_rate"]
        < dummy_metrics["false_positive_rate"]
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(logistic_regression, MODEL_PATH)
    joblib.dump(development.preprocessor, PREPROCESSOR_PATH)

    reloaded_model = joblib.load(MODEL_PATH)
    reloaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
    reloaded_validation = transform_network_flows(
        reloaded_preprocessor, development.X_validation_raw
    )
    assert matrices_match(development.X_validation, reloaded_validation)
    assert np.array_equal(
        logistic_predictions, reloaded_model.predict(reloaded_validation)
    )
    assert np.allclose(
        logistic_probabilities,
        attack_probabilities(reloaded_model, reloaded_validation),
        rtol=0.0,
        atol=0.0,
    )

    metrics_document = {
        "stage": "baseline_training_and_validation",
        "task": "binary network security anomaly detection",
        "target_mapping": {"0": "normal", "1": "attack/anomalous"},
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
            "development_train_rows": len(development.X_train_raw),
            "validation_rows": len(development.X_validation_raw),
            "raw_feature_count": development.X_train_raw.shape[1],
            "transformed_feature_count": development.X_train.shape[1],
            "X_train_shape": list(development.X_train.shape),
            "X_validation_shape": list(development.X_validation.shape),
        },
        "split": {
            "validation_size": VALIDATION_SIZE,
            "random_state": RANDOM_STATE,
            "stratified_by": "label",
            "overlapping_row_indexes": False,
        },
        "preprocessing": {
            "fit_rows": len(development.X_train_raw),
            "fit_source": "development training subset only",
            "numeric_transformer": "StandardScaler",
            "categorical_transformer": (
                "OneHotEncoder(handle_unknown='ignore')"
            ),
            "numeric_feature_count": len(development.numeric_features),
            "categorical_features": development.categorical_features,
            "excluded_columns": ["id", "attack_cat", "label"],
            "service_dash_retained": True,
            "validation_unknown_categories": (
                development.validation_unknown_categories
            ),
        },
        "models": model_reports,
        "comparison": {
            "logistic_minus_dummy": metric_deltas,
            "meaningfully_beats_dummy": meaningfully_beats_dummy,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": {
            "logistic_regression": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "preprocessor": str(PREPROCESSOR_PATH.relative_to(PROJECT_ROOT)),
            "logistic_regression_sha256": sha256_file(MODEL_PATH),
            "preprocessor_sha256": sha256_file(PREPROCESSOR_PATH),
        },
        "reproducibility_checks": {
            "saved_preprocessor_transform_matches": True,
            "saved_model_predictions_match": True,
            "saved_model_probabilities_match": True,
            "official_test_accessed": False,
            "model_count": 2,
            "hyperparameter_tuning": False,
        },
    }

    METRICS_PATH.write_text(
        json.dumps(metrics_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert loaded_metrics["reproducibility_checks"]["model_count"] == 2
    assert loaded_metrics["official_test"]["used"] is False

    print("UNSW-NB15 BASELINE VALIDATION REPORT")
    print("Official test used: no")
    print_model_report("DummyClassifier", model_reports["dummy_classifier"])
    print_model_report(
        "LogisticRegression", model_reports["logistic_regression"]
    )
    print("\nComparison")
    print(f"  Logistic Regression meaningfully beats Dummy: {meaningfully_beats_dummy}")
    print(f"  Metric deltas (Logistic - Dummy): {metric_deltas}")
    print("\nSaved artifacts")
    print(f"  Metrics: {METRICS_PATH}")
    print(f"  Logistic Regression: {MODEL_PATH}")
    print(f"  Preprocessor: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
