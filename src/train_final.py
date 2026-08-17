"""Train the locked final Random Forest, then evaluate official test once."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier

try:
    from .evaluate_final import (
        FINAL_CATEGORY_METRICS_PATH,
        FINAL_MANIFEST_PATH,
        FINAL_MODEL_PATH,
        FINAL_PREPROCESSOR_PATH,
        FINAL_TEST_METRICS_PATH,
        LOCKED_THRESHOLD,
        run_final_evaluation,
    )
    from .preprocess import (
        EXCLUDED_COLUMNS,
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        TARGET_COLUMN,
        assert_binary_target,
        assert_finite_matrix,
        build_preprocessor,
        get_learned_categories,
        identify_feature_groups,
        load_official_training,
        separate_features_and_target,
    )
except ImportError:
    from evaluate_final import (
        FINAL_CATEGORY_METRICS_PATH,
        FINAL_MANIFEST_PATH,
        FINAL_MODEL_PATH,
        FINAL_PREPROCESSOR_PATH,
        FINAL_TEST_METRICS_PATH,
        LOCKED_THRESHOLD,
        run_final_evaluation,
    )
    from preprocess import (
        EXCLUDED_COLUMNS,
        OFFICIAL_TRAINING_FILE,
        RANDOM_STATE,
        TARGET_COLUMN,
        assert_binary_target,
        assert_finite_matrix,
        build_preprocessor,
        get_learned_categories,
        identify_feature_groups,
        load_official_training,
        separate_features_and_target,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
TUNED_RF_METRICS_PATH = MODELS_DIR / "tuned_random_forest_metrics.json"
SELECTED_THRESHOLD_PATH = MODELS_DIR / "selected_threshold.json"

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

FINAL_OUTPUT_PATHS = (
    FINAL_MODEL_PATH,
    FINAL_PREPROCESSOR_PATH,
    FINAL_TEST_METRICS_PATH,
    FINAL_CATEGORY_METRICS_PATH,
    FINAL_MANIFEST_PATH,
)


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 hash for training-data provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_locked_inputs() -> None:
    """Confirm tuning and threshold decisions predate official-test access."""
    for path in (TUNED_RF_METRICS_PATH, SELECTED_THRESHOLD_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required locked artifact missing: {path}")

    tuning_report = json.loads(
        TUNED_RF_METRICS_PATH.read_text(encoding="utf-8")
    )
    threshold_report = json.loads(
        SELECTED_THRESHOLD_PATH.read_text(encoding="utf-8")
    )
    assert tuning_report["official_test"]["used"] is False
    assert threshold_report["official_test"]["used"] is False
    assert (
        tuning_report["final_model"]["configuration"]
        == FROZEN_MODEL_HYPERPARAMETERS
    )
    assert threshold_report["selected_threshold"] == LOCKED_THRESHOLD
    assert threshold_report["selected_threshold"] == 0.45


def ensure_new_output_paths() -> None:
    """Protect all prior and any already-created final artifacts."""
    existing = [str(path) for path in FINAL_OUTPUT_PATHS if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite final artifacts: " + ", ".join(existing)
        )


def fit_with_warnings(
    model: RandomForestClassifier, features: Any, target: Any
) -> tuple[float, list[dict[str, str]]]:
    """Fit the final forest once and capture any warnings."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        start = perf_counter()
        model.fit(features, target)
        seconds = perf_counter() - start
    report = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in captured
    ]
    return seconds, report


def main() -> None:
    """Fit all-training preprocessing/model, save them, then open test."""
    assert_locked_inputs()
    ensure_new_output_paths()

    training_data = load_official_training()
    training_features, training_target = separate_features_and_target(
        training_data
    )
    assert len(training_data) == 175_341
    assert_binary_target(training_target, "complete official training")
    excluded = {TARGET_COLUMN, *EXCLUDED_COLUMNS}
    assert not excluded.intersection(training_features.columns)
    assert training_features.shape[1] == 42

    numeric_features, categorical_features = identify_feature_groups(
        training_features
    )
    assert len(numeric_features) == 39
    assert categorical_features == ["proto", "service", "state"]

    total_training_start = perf_counter()
    preprocessor = build_preprocessor(
        numeric_features, categorical_features
    )
    preprocessing_start = perf_counter()
    transformed_training = preprocessor.fit_transform(training_features)
    preprocessing_seconds = perf_counter() - preprocessing_start
    assert_finite_matrix(transformed_training, "final transformed training")

    scaler = preprocessor.named_transformers_["numeric"]
    assert int(scaler.n_samples_seen_) == len(training_features)
    learned_categories = get_learned_categories(
        preprocessor, categorical_features
    )
    assert "-" in learned_categories["service"]
    encoded_feature_count = len(preprocessor.get_feature_names_out())
    assert transformed_training.shape == (
        len(training_features),
        encoded_feature_count,
    )

    model = RandomForestClassifier(**FROZEN_MODEL_HYPERPARAMETERS)
    model_training_seconds, training_warnings = fit_with_warnings(
        model, transformed_training, training_target
    )
    total_training_seconds = perf_counter() - total_training_start
    assert model.classes_.tolist() == [0, 1]

    # Final artifacts are persisted before the official test is loaded.
    joblib.dump(preprocessor, FINAL_PREPROCESSOR_PATH, compress=3)
    joblib.dump(model, FINAL_MODEL_PATH, compress=3)
    assert FINAL_PREPROCESSOR_PATH.is_file()
    assert FINAL_MODEL_PATH.is_file()

    training_metadata = {
        "training_complete": True,
        "training_file": str(
            OFFICIAL_TRAINING_FILE.relative_to(PROJECT_ROOT)
        ),
        "training_sha256": sha256_file(OFFICIAL_TRAINING_FILE),
        "training_rows": len(training_data),
        "training_normal_rows": int((training_target == 0).sum()),
        "training_attack_rows": int((training_target == 1).sum()),
        "raw_feature_count": training_features.shape[1],
        "encoded_feature_count": encoded_feature_count,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "learned_categories": learned_categories,
        "model_hyperparameters": FROZEN_MODEL_HYPERPARAMETERS,
        "runtime": {
            "preprocessor_fit_transform_seconds": preprocessing_seconds,
            "random_forest_fit_seconds": model_training_seconds,
            "total_training_seconds": total_training_seconds,
        },
        "training_warnings": training_warnings,
        "preprocessor_fit_rows": int(scaler.n_samples_seen_),
    }

    print("FINAL LOCKED TRAINING COMPLETE", flush=True)
    print(f"Training rows: {len(training_data):,}", flush=True)
    print(f"Raw features: {training_features.shape[1]}", flush=True)
    print(f"Encoded features: {encoded_feature_count}", flush=True)
    print(
        f"Preprocessing time: {preprocessing_seconds:.6f} seconds",
        flush=True,
    )
    print(
        f"Random Forest fit time: {model_training_seconds:.6f} seconds",
        flush=True,
    )
    print(
        f"Total training time: {total_training_seconds:.6f} seconds",
        flush=True,
    )
    print(f"Training warnings: {training_warnings or 'none'}", flush=True)
    print(f"Saved preprocessor: {FINAL_PREPROCESSOR_PATH}", flush=True)
    print(f"Saved model: {FINAL_MODEL_PATH}", flush=True)
    print(
        "Final training is complete; official test access begins now.",
        flush=True,
    )

    run_final_evaluation(model, preprocessor, training_metadata)


if __name__ == "__main__":
    main()
