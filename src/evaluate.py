"""Validation metrics for binary network attack detection models."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


def attack_probabilities(model: Any, features: Any) -> np.ndarray:
    """Return the probability assigned to binary attack class 1."""
    classes = model.classes_.tolist()
    if 1 not in classes:
        raise ValueError(f"Model classes do not contain attack class 1: {classes}")
    attack_class_index = classes.index(1)
    return model.predict_proba(features)[:, attack_class_index]


def evaluate_binary_classifier(
    model: Any, features: Any, target: pd.Series
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate one fitted classifier at the default 0.5 threshold."""
    evaluation_start = perf_counter()
    predictions = model.predict(features)
    probabilities = attack_probabilities(model, features)
    inference_seconds = perf_counter() - evaluation_start

    # sklearn's binary predict uses argmax, so an exact 0.5 tie resolves to
    # class 0 when classes_ == [0, 1]. This is still the default 0.5 boundary.
    threshold_predictions = (
        probabilities > DEFAULT_CLASSIFICATION_THRESHOLD
    ).astype(int)
    assert np.array_equal(predictions, threshold_predictions), (
        "Model predictions do not match the required default 0.5 threshold"
    )

    tn, fp, fn, tp = confusion_matrix(
        target, predictions, labels=[0, 1]
    ).ravel()
    actual_normal = tn + fp
    actual_attack = tp + fn
    assert actual_normal > 0 and actual_attack > 0

    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision_attack": float(
            precision_score(target, predictions, pos_label=1, zero_division=0)
        ),
        "recall_attack": float(
            recall_score(target, predictions, pos_label=1, zero_division=0)
        ),
        "f1_attack": float(
            f1_score(target, predictions, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(
            average_precision_score(target, probabilities)
        ),
        "false_positive_rate": float(fp / actual_normal),
        "false_negative_rate": float(fn / actual_attack),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "actual_normal": int(actual_normal),
        "actual_attack": int(actual_attack),
        "classification_threshold": DEFAULT_CLASSIFICATION_THRESHOLD,
        "validation_inference_seconds": inference_seconds,
        "confusion_matrix": {
            "true_normal_predicted_normal": int(tn),
            "true_normal_predicted_attack": int(fp),
            "true_attack_predicted_normal": int(fn),
            "true_attack_predicted_attack": int(tp),
        },
    }
    return metrics, predictions, probabilities
