"""Tune Random Forest structure using development-train cross-validation only.

The existing validation subset is used exactly once after hyperparameter
selection. The official UNSW test set is never loaded or evaluated.
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
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

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
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
UNTUNED_RF_METRICS_PATH = MODELS_DIR / "random_forest_metrics.json"
TUNED_RF_PATH = MODELS_DIR / "tuned_random_forest.joblib"
TUNED_RF_METRICS_PATH = MODELS_DIR / "tuned_random_forest_metrics.json"
TUNING_RESULTS_PATH = MODELS_DIR / "random_forest_tuning_results.csv"

CV_SPLITS = 3
SEARCH_CANDIDATES = 12
SEARCH_N_JOBS = 12
CV_RECALL_TARGET = 0.975
CV_FPR_TIE_TOLERANCE = 0.001
DEFAULT_THRESHOLD = 0.5

PARAMETER_DISTRIBUTIONS = {
    "max_depth": [12, 18, 24, 32, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", 0.5],
    "max_samples": [None, 0.8],
    "bootstrap": [True],
}

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


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 hash for reproducibility metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    """Convert numpy scalars and parameter values into JSON-safe values."""
    if value is None or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def fit_with_timing(
    model: RandomForestClassifier, features: Any, target: pd.Series
) -> tuple[float, list[dict[str, str]]]:
    """Fit the selected forest and capture time and warnings."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        start = perf_counter()
        model.fit(features, target)
        elapsed = perf_counter() - start

    warning_report = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in captured
    ]
    return elapsed, warning_report


def build_scoring() -> dict[str, Any]:
    """Build security-focused CV scorers without using accuracy."""
    return {
        "precision_attack": make_scorer(
            precision_score, pos_label=1, zero_division=0
        ),
        "recall_attack": make_scorer(
            recall_score, pos_label=1, zero_division=0
        ),
        "f1_attack": make_scorer(
            f1_score, pos_label=1, zero_division=0
        ),
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "specificity": make_scorer(
            recall_score, pos_label=0, zero_division=0
        ),
    }


def create_results_table(search: RandomizedSearchCV) -> pd.DataFrame:
    """Create a compact, auditable table from sklearn CV results."""
    raw = pd.DataFrame(search.cv_results_)
    results = pd.DataFrame(
        {
            "candidate_index": np.arange(len(raw), dtype=int),
            "mean_fit_time_seconds": raw["mean_fit_time"],
            "std_fit_time_seconds": raw["std_fit_time"],
            "mean_cv_precision": raw["mean_test_precision_attack"],
            "std_cv_precision": raw["std_test_precision_attack"],
            "mean_cv_recall": raw["mean_test_recall_attack"],
            "std_cv_recall": raw["std_test_recall_attack"],
            "mean_cv_f1": raw["mean_test_f1_attack"],
            "std_cv_f1": raw["std_test_f1_attack"],
            "mean_cv_roc_auc": raw["mean_test_roc_auc"],
            "std_cv_roc_auc": raw["std_test_roc_auc"],
            "mean_cv_average_precision": raw[
                "mean_test_average_precision"
            ],
            "std_cv_average_precision": raw[
                "std_test_average_precision"
            ],
            "mean_cv_specificity": raw["mean_test_specificity"],
            "std_cv_specificity": raw["std_test_specificity"],
            "mean_train_precision": raw["mean_train_precision_attack"],
            "mean_train_recall": raw["mean_train_recall_attack"],
            "mean_train_f1": raw["mean_train_f1_attack"],
            "mean_train_roc_auc": raw["mean_train_roc_auc"],
            "mean_train_average_precision": raw[
                "mean_train_average_precision"
            ],
            "mean_train_specificity": raw["mean_train_specificity"],
        }
    )
    results["mean_cv_fpr"] = 1.0 - results["mean_cv_specificity"]
    results["meets_recall_target"] = (
        results["mean_cv_recall"] >= CV_RECALL_TARGET
    )

    for parameter in PARAMETER_DISTRIBUTIONS:
        results[f"param_{parameter}"] = raw[f"param_{parameter}"].map(
            json_value
        )

    return results


def select_candidate(results: pd.DataFrame) -> tuple[int, bool, pd.DataFrame]:
    """Apply recall floor, then FPR, then PR-AUC/F1 tie-breaking."""
    eligible = results[results["meets_recall_target"]].copy()
    target_achieved = not eligible.empty

    if target_achieved:
        best_fpr = float(eligible["mean_cv_fpr"].min())
        selection_pool = eligible[
            eligible["mean_cv_fpr"] <= best_fpr + CV_FPR_TIE_TOLERANCE
        ].copy()
        selection_pool = selection_pool.sort_values(
            by=[
                "mean_cv_average_precision",
                "mean_cv_f1",
                "mean_cv_fpr",
                "mean_cv_recall",
            ],
            ascending=[False, False, True, False],
        )
    else:
        best_recall = float(results["mean_cv_recall"].max())
        selection_pool = results[
            results["mean_cv_recall"] >= best_recall - 0.002
        ].copy()
        selection_pool = selection_pool.sort_values(
            by=[
                "mean_cv_fpr",
                "mean_cv_average_precision",
                "mean_cv_f1",
                "mean_cv_recall",
            ],
            ascending=[True, False, False, False],
        )

    selected_index = int(selection_pool.iloc[0]["candidate_index"])
    ranked = results.sort_values(
        by=[
            "meets_recall_target",
            "mean_cv_fpr",
            "mean_cv_average_precision",
            "mean_cv_f1",
            "mean_cv_recall",
        ],
        ascending=[False, True, False, False, False],
    ).copy()
    ranked.insert(0, "selection_rank", np.arange(1, len(ranked) + 1))
    ranked["selected"] = ranked["candidate_index"] == selected_index
    return selected_index, target_achieved, ranked


def candidate_record(row: pd.Series) -> dict[str, Any]:
    """Convert one compact CV row into a JSON report record."""
    keys = (
        "selection_rank",
        "candidate_index",
        "selected",
        "meets_recall_target",
        "mean_cv_precision",
        "mean_cv_recall",
        "mean_cv_f1",
        "mean_cv_roc_auc",
        "mean_cv_average_precision",
        "mean_cv_specificity",
        "mean_cv_fpr",
        "mean_fit_time_seconds",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features",
        "param_max_samples",
        "param_bootstrap",
    )
    return {key: json_value(row[key]) for key in keys}


def main() -> None:
    """Search, select, retrain, and validate one tuned Random Forest."""
    required_artifacts = (
        PREPROCESSOR_PATH,
        BASELINE_METRICS_PATH,
        UNTUNED_RF_METRICS_PATH,
    )
    for path in required_artifacts:
        if not path.is_file():
            raise FileNotFoundError(f"Required prior artifact missing: {path}")

    logistic_document = json.loads(
        BASELINE_METRICS_PATH.read_text(encoding="utf-8")
    )
    untuned_document = json.loads(
        UNTUNED_RF_METRICS_PATH.read_text(encoding="utf-8")
    )
    assert logistic_document["official_test"]["used"] is False
    assert untuned_document["official_test"]["used"] is False
    logistic_metrics = logistic_document["models"]["logistic_regression"][
        "metrics"
    ]
    untuned_metrics = untuned_document["model"]["validation_metrics"]
    untuned_training_metrics = untuned_document["model"]["training_metrics"]

    split = prepare_development_split()
    assert not hasattr(split, "X_test")
    assert len(split.X_train_raw) == 140_272
    assert len(split.X_validation_raw) == 35_069

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    scaler = preprocessor.named_transformers_["numeric"]
    assert int(scaler.n_samples_seen_) == len(split.X_train_raw)
    X_train = transform_network_flows(preprocessor, split.X_train_raw)
    assert X_train.shape == (140_272, 192)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search_estimator = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
    )
    search = RandomizedSearchCV(
        estimator=search_estimator,
        param_distributions=PARAMETER_DISTRIBUTIONS,
        n_iter=SEARCH_CANDIDATES,
        scoring=build_scoring(),
        n_jobs=SEARCH_N_JOBS,
        cv=cv,
        refit=False,
        random_state=42,
        return_train_score=True,
        error_score="raise",
        verbose=2,
        pre_dispatch=SEARCH_N_JOBS,
    )

    # Only X_train/y_train are available to hyperparameter search. The raw
    # validation subset is deliberately not transformed until selection ends.
    with warnings.catch_warnings(record=True) as search_captured:
        warnings.simplefilter("always")
        search_start = perf_counter()
        search.fit(X_train, split.y_train)
        search_seconds = perf_counter() - search_start
    search_warnings = [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in search_captured
    ]

    results = create_results_table(search)
    selected_index, cv_recall_target_achieved, ranked_results = (
        select_candidate(results)
    )
    ranked_results.to_csv(TUNING_RESULTS_PATH, index=False)

    selected_params = {
        key: json_value(value)
        for key, value in search.cv_results_["params"][selected_index].items()
    }
    selected_cv_row = ranked_results[
        ranked_results["candidate_index"] == selected_index
    ].iloc[0]
    top_cv_candidates = [
        candidate_record(row)
        for _, row in ranked_results.head(5).iterrows()
    ]

    final_configuration = {
        "n_estimators": 300,
        "random_state": 42,
        "n_jobs": -1,
        **selected_params,
    }
    tuned_forest = RandomForestClassifier(**final_configuration)
    final_training_seconds, final_training_warnings = fit_with_timing(
        tuned_forest, X_train, split.y_train
    )

    training_metrics, _, _ = evaluate_binary_classifier(
        tuned_forest, X_train, split.y_train
    )

    # First and only validation evaluation for the selected candidate.
    X_validation = transform_network_flows(
        preprocessor, split.X_validation_raw
    )
    validation_metrics, validation_predictions, validation_probabilities = (
        evaluate_binary_classifier(
            tuned_forest, X_validation, split.y_validation
        )
    )

    model_metrics = {
        "logistic_regression": logistic_metrics,
        "untuned_random_forest": untuned_metrics,
        "tuned_random_forest": validation_metrics,
    }
    comparison_rows = []
    for metric_key, display_name in COMPARISON_METRICS.items():
        comparison_rows.append(
            {
                "metric": display_name,
                "metric_key": metric_key,
                **{
                    model_name: metrics[metric_key]
                    for model_name, metrics in model_metrics.items()
                },
            }
        )

    tuned_overfitting_gaps = {
        metric: training_metrics[metric] - validation_metrics[metric]
        for metric in (
            "accuracy",
            "precision_attack",
            "recall_attack",
            "f1_attack",
            "roc_auc",
        )
    }
    untuned_overfitting_gaps = {
        metric: untuned_training_metrics[metric] - untuned_metrics[metric]
        for metric in tuned_overfitting_gaps
    }
    tuning_reduced_overfitting = bool(
        tuned_overfitting_gaps["accuracy"]
        < untuned_overfitting_gaps["accuracy"]
        and tuned_overfitting_gaps["f1_attack"]
        < untuned_overfitting_gaps["f1_attack"]
    )

    fpr_change_vs_untuned = (
        validation_metrics["false_positive_rate"]
        - untuned_metrics["false_positive_rate"]
    )
    recall_change_vs_untuned = (
        validation_metrics["recall_attack"] - untuned_metrics["recall_attack"]
    )
    false_alerts_eliminated = (
        untuned_metrics["false_positives"]
        - validation_metrics["false_positives"]
    )
    attacks_missed_change = (
        validation_metrics["false_negatives"]
        - untuned_metrics["false_negatives"]
    )

    validation_candidates = {
        name: metrics
        for name, metrics in model_metrics.items()
        if metrics["recall_attack"] >= 0.97
    }
    current_leader = min(
        validation_candidates,
        key=lambda name: (
            validation_candidates[name]["false_positive_rate"],
            -validation_candidates[name]["average_precision"],
            -validation_candidates[name]["f1_attack"],
        ),
    )

    joblib.dump(tuned_forest, TUNED_RF_PATH, compress=3)
    reloaded_forest = joblib.load(TUNED_RF_PATH)
    assert np.array_equal(
        validation_predictions, reloaded_forest.predict(X_validation)
    )
    reloaded_probabilities = attack_probabilities(
        reloaded_forest, X_validation
    )
    probability_max_absolute_difference = float(
        np.max(np.abs(validation_probabilities - reloaded_probabilities))
    )
    assert np.allclose(
        validation_probabilities,
        reloaded_probabilities,
        rtol=1e-12,
        atol=1e-15,
    )

    metrics_document = {
        "stage": "random_forest_hyperparameter_tuning",
        "official_test": {
            "used": False,
            "statement": (
                "Official UNSW testing data and labels were not loaded, "
                "transformed, evaluated, or used in this stage."
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
        },
        "search": {
            "method": "RandomizedSearchCV",
            "candidate_count": SEARCH_CANDIDATES,
            "cv": {
                "class": "StratifiedKFold",
                "n_splits": CV_SPLITS,
                "shuffle": True,
                "random_state": RANDOM_STATE,
            },
            "base_estimator": {
                "n_estimators": 200,
                "random_state": 42,
                "n_jobs": 1,
            },
            "parameter_distributions": PARAMETER_DISTRIBUTIONS,
            "selection_policy": {
                "recall_target": CV_RECALL_TARGET,
                "fpr_tie_tolerance": CV_FPR_TIE_TOLERANCE,
                "accuracy_used_for_selection": False,
                "target_achieved": cv_recall_target_achieved,
            },
            "search_seconds": search_seconds,
            "warnings": search_warnings,
            "selected_candidate_index": selected_index,
            "selected_cv_metrics": candidate_record(selected_cv_row),
            "selected_structural_hyperparameters": selected_params,
            "top_cv_candidates": top_cv_candidates,
            "results_artifact": str(
                TUNING_RESULTS_PATH.relative_to(PROJECT_ROOT)
            ),
        },
        "final_model": {
            "configuration": final_configuration,
            "classification_threshold": DEFAULT_THRESHOLD,
            "training_seconds": final_training_seconds,
            "training_warnings": final_training_warnings,
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
        },
        "comparison": {
            "rows": comparison_rows,
            "untuned_training_minus_validation": untuned_overfitting_gaps,
            "tuned_training_minus_validation": tuned_overfitting_gaps,
            "tuning_reduced_overfitting": tuning_reduced_overfitting,
            "fpr_change_vs_untuned": fpr_change_vs_untuned,
            "recall_change_vs_untuned": recall_change_vs_untuned,
            "false_alerts_eliminated_vs_untuned": false_alerts_eliminated,
            "false_negatives_change_vs_untuned": attacks_missed_change,
            "current_leading_candidate": current_leader,
        },
        "artifacts": {
            "tuned_random_forest": str(TUNED_RF_PATH.relative_to(PROJECT_ROOT)),
            "tuned_random_forest_sha256": sha256_file(TUNED_RF_PATH),
            "tuning_results": str(TUNING_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "tuning_results_sha256": sha256_file(TUNING_RESULTS_PATH),
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
            "search_used_only_development_training": True,
            "validation_evaluations_after_selection": 1,
            "official_test_accessed": False,
            "threshold_optimization": False,
        },
    }

    TUNED_RF_METRICS_PATH.write_text(
        json.dumps(metrics_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded_metrics = json.loads(
        TUNED_RF_METRICS_PATH.read_text(encoding="utf-8")
    )
    assert loaded_metrics["official_test"]["used"] is False
    assert (
        loaded_metrics["reproducibility_checks"]
        ["validation_evaluations_after_selection"]
        == 1
    )

    print("UNSW-NB15 RANDOM FOREST TUNING REPORT")
    print("Official test used: no")
    print(f"Search candidates: {SEARCH_CANDIDATES}")
    print(f"CV folds: {CV_SPLITS}")
    print(f"Search time: {search_seconds:.6f} seconds")
    print(f"Search warnings: {search_warnings or 'none'}")
    print(f"CV recall target achieved: {cv_recall_target_achieved}")
    print(f"Selected hyperparameters: {selected_params}")
    print("\nTop CV candidates:")
    for record in top_cv_candidates:
        print(record)

    print("\nFinal tuned training metrics:")
    for metric in (
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_attack",
        "roc_auc",
    ):
        print(f"  {metric}: {training_metrics[metric]:.6f}")

    print("\nFinal tuned validation metrics:")
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
    print(f"  false_positives: {validation_metrics['false_positives']}")
    print(f"  false_negatives: {validation_metrics['false_negatives']}")
    print(
        "  validation_inference_seconds: "
        f"{validation_metrics['validation_inference_seconds']:.6f}"
    )
    print("  confusion_matrix:")
    for label, value in validation_metrics["confusion_matrix"].items():
        print(f"    {label}: {value:,}")

    print("\nThree-model comparison:")
    for row in comparison_rows:
        print(row)

    print("\nOperational answers:")
    print(f"  Tuning reduced overfitting: {tuning_reduced_overfitting}")
    print(f"  FPR change vs untuned RF: {fpr_change_vs_untuned:+.6f}")
    print(f"  Recall change vs untuned RF: {recall_change_vs_untuned:+.6f}")
    print(f"  False alerts eliminated: {false_alerts_eliminated:+d}")
    print(f"  False negatives change: {attacks_missed_change:+d}")
    print(f"  Current leading candidate: {current_leader}")
    print(f"  Final training time: {final_training_seconds:.6f} seconds")
    print(f"  Final training warnings: {final_training_warnings or 'none'}")
    print("\nSaved artifacts:")
    print(f"  Model: {TUNED_RF_PATH}")
    print(f"  Metrics: {TUNED_RF_METRICS_PATH}")
    print(f"  CV results: {TUNING_RESULTS_PATH}")


if __name__ == "__main__":
    main()
