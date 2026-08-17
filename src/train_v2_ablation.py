"""Run the CIC-UNSW-NB15 v2 baseline-versus-context ablation.

Both Random Forests use identical frozen hyperparameters and Jan 22 training
rows.  Thresholds are selected only from Feb 17 validation scores.  This module
has no path or query for the Feb 18 locked holdout dataset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2"
TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_PATH = FEATURE_DIR / "validation_features.parquet"
FEATURE_MANIFEST_PATH = FEATURE_DIR / "feature_manifest.json"
CACHE_DIR = FEATURE_DIR / "_ablation_cache"
MODELS_DIR = PROJECT_ROOT / "models"

BASELINE_MODEL_PATH = MODELS_DIR / "v2_baseline_random_forest.joblib"
CONTEXT_MODEL_PATH = MODELS_DIR / "v2_context_random_forest.joblib"
METRICS_PATH = MODELS_DIR / "v2_ablation_metrics.json"
THRESHOLD_RESULTS_PATH = MODELS_DIR / "v2_threshold_results.csv"
COMPARISON_PATH = MODELS_DIR / "v2_ablation_comparison.csv"
CATEGORY_METRICS_PATH = MODELS_DIR / "v2_attack_category_metrics.csv"
VALIDATION_SCORES_PATH = MODELS_DIR / "v2_validation_scores.parquet"

TRAIN_ROWS = 1_765_922
VALIDATION_ROWS = 498_890
RECALL_GATE = 0.985
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
RANDOM_FOREST_PARAMS: dict[str, Any] = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duckdb-vectors-per-chunk",
        type=int,
        default=100,
        help="DuckDB vectors per streamed pandas chunk (default: 100).",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep generated float32 memory-map matrices after completion.",
    )
    return parser.parse_args()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_feature_contract() -> tuple[list[str], list[str], dict[str, Any]]:
    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    contract = manifest["feature_contract"]
    baseline = list(contract["baseline_v2_features"])
    context = [
        *baseline,
        *contract["context_v2_static_behavioral_features"],
        *contract["context_v2_temporal_features"],
    ]
    if len(baseline) != 76 or len(context) != 128:
        raise AssertionError(
            f"Unexpected feature counts: baseline={len(baseline)}, context={len(context)}"
        )
    if context[: len(baseline)] != baseline:
        raise AssertionError("Baseline features are not the leading context subset")
    return baseline, context, manifest


def cache_paths(split: str) -> dict[str, Path]:
    return {
        "baseline": CACHE_DIR / f"{split}_baseline.float32.mmap",
        "context": CACHE_DIR / f"{split}_context.float32.mmap",
        "target": CACHE_DIR / f"{split}_target.uint8.mmap",
    }


def stream_parquet_to_memmaps(
    path: Path,
    split: str,
    expected_rows: int,
    baseline_features: list[str],
    context_features: list[str],
    vectors_per_chunk: int,
    include_categories: bool,
) -> tuple[np.memmap, np.memmap, np.memmap, np.ndarray | None, float]:
    started = time.perf_counter()
    paths = cache_paths(split)
    baseline_matrix = np.memmap(
        paths["baseline"],
        mode="w+",
        dtype=np.float32,
        shape=(expected_rows, len(baseline_features)),
    )
    context_matrix = np.memmap(
        paths["context"],
        mode="w+",
        dtype=np.float32,
        shape=(expected_rows, len(context_features)),
    )
    target = np.memmap(
        paths["target"],
        mode="w+",
        dtype=np.uint8,
        shape=(expected_rows,),
    )

    selected = [*context_features, "label"]
    if include_categories:
        selected.append("attack_cat")
    selected_sql = ", ".join(quote_identifier(column) for column in selected)
    connection = duckdb.connect()
    categories: list[str] | None = [] if include_categories else None
    row_offset = 0
    try:
        connection.execute(
            f"SELECT {selected_sql} FROM read_parquet({sql_string(path)})"
        )
        while True:
            chunk = connection.fetch_df_chunk(vectors_per_chunk)
            if chunk.empty:
                break
            end = row_offset + len(chunk)
            if end > expected_rows:
                raise AssertionError(f"{split}: streamed more rows than expected")
            values = chunk[context_features].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(values).all():
                raise AssertionError(f"{split}: non-finite model input encountered")
            context_matrix[row_offset:end] = values
            baseline_matrix[row_offset:end] = values[:, : len(baseline_features)]
            target[row_offset:end] = chunk["label"].to_numpy(dtype=np.uint8)
            if categories is not None:
                categories.extend(chunk["attack_cat"].astype(str).tolist())
            row_offset = end
            print(f"Loaded {split}: {row_offset:,}/{expected_rows:,}", flush=True)
    finally:
        connection.close()

    if row_offset != expected_rows:
        raise AssertionError(
            f"{split}: expected {expected_rows:,} rows, streamed {row_offset:,}"
        )
    if not np.isin(np.asarray(target), [0, 1]).all():
        raise AssertionError(f"{split}: target contains values outside 0/1")
    baseline_matrix.flush()
    context_matrix.flush()
    target.flush()
    category_array = np.asarray(categories, dtype=object) if categories is not None else None
    return (
        baseline_matrix,
        context_matrix,
        target,
        category_array,
        time.perf_counter() - started,
    )


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()
    actual_normal = tn + fp
    actual_attack = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / actual_attack if actual_attack else 0.0
    specificity = tn / actual_normal if actual_normal else 0.0
    fpr = fp / actual_normal if actual_normal else 0.0
    fnr = fn / actual_attack if actual_attack else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "fp_per_100k_benign": float(fpr * 100_000),
    }


def sweep_thresholds(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    rows = [
        {"model": model_name, **threshold_metrics(y_true, scores, threshold)}
        for threshold in THRESHOLDS
    ]
    frame = pd.DataFrame(rows)
    eligible = frame[frame["recall"] >= RECALL_GATE]
    gate_met = not eligible.empty
    if gate_met:
        selected_row = eligible.sort_values(
            by=["false_positives", "precision", "f1", "threshold"],
            ascending=[True, False, False, False],
        ).iloc[0]
    else:
        selected_row = frame.sort_values(
            by=["recall", "false_positives", "precision", "f1", "threshold"],
            ascending=[False, True, False, False, False],
        ).iloc[0]
    selected = {
        key: (
            int(selected_row[key])
            if key in {
                "false_positives",
                "false_negatives",
                "true_positives",
                "true_negatives",
            }
            else float(selected_row[key])
        )
        for key in frame.columns
        if key != "model"
    }
    return frame, selected, gate_met


def category_metrics(
    model_name: str,
    categories: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    predictions = scores >= threshold
    rows: list[dict[str, Any]] = []
    for category in sorted(np.unique(categories)):
        mask = categories == category
        total = int(mask.sum())
        if category == "Benign":
            false_positives = int(predictions[mask].sum())
            true_negatives = total - false_positives
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "attack_cat": category,
                    "total": total,
                    "detected_attacks": 0,
                    "missed_attacks": 0,
                    "attack_recall": np.nan,
                    "false_positives": false_positives,
                    "true_negatives": true_negatives,
                    "category_fpr": false_positives / total if total else 0.0,
                }
            )
        else:
            if not np.all(y_true[mask] == 1):
                raise AssertionError(f"Non-benign category {category} contains label 0")
            detected = int(predictions[mask].sum())
            missed = total - detected
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "attack_cat": category,
                    "total": total,
                    "detected_attacks": detected,
                    "missed_attacks": missed,
                    "attack_recall": detected / total if total else 0.0,
                    "false_positives": 0,
                    "true_negatives": 0,
                    "category_fpr": np.nan,
                }
            )
    return pd.DataFrame(rows)


def fit_and_score(
    model_name: str,
    model_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        started = time.perf_counter()
        model.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = model.predict_proba(x_validation)[:, 1]
        inference_seconds = time.perf_counter() - started
        captured_warnings = [str(record.message) for record in warning_records]

    joblib.dump(model, model_path, compress=3)
    if not np.isfinite(scores).all():
        raise AssertionError(f"{model_name}: validation scores are non-finite")
    continuous = {
        "roc_auc": float(roc_auc_score(y_validation, scores)),
        "pr_auc_average_precision": float(
            average_precision_score(y_validation, scores)
        ),
        "training_seconds": training_seconds,
        "validation_inference_seconds": inference_seconds,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "model_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
    }

    reloaded = joblib.load(model_path)
    check_size = min(10_000, len(x_validation))
    reloaded_scores = reloaded.predict_proba(x_validation[:check_size])[:, 1]
    max_difference = float(np.max(np.abs(scores[:check_size] - reloaded_scores)))
    if max_difference > 1e-12:
        raise AssertionError(f"{model_name}: saved model did not reproduce scores")
    continuous["reload_check_rows"] = check_size
    continuous["reload_max_score_difference"] = max_difference
    del reloaded, model
    gc.collect()
    return scores, continuous, captured_warnings


def save_validation_scores(
    categories: np.ndarray,
    y_true: np.ndarray,
    baseline_scores: np.ndarray,
    context_scores: np.ndarray,
    baseline_threshold: float,
    context_threshold: float,
) -> None:
    frame = pd.DataFrame(
        {
            "validation_row": np.arange(len(y_true), dtype=np.int64),
            "attack_cat": categories,
            "label": y_true.astype(np.uint8),
            "baseline_attack_score": baseline_scores,
            "context_attack_score": context_scores,
            "baseline_selected_prediction": (
                baseline_scores >= baseline_threshold
            ).astype(np.uint8),
            "context_selected_prediction": (
                context_scores >= context_threshold
            ).astype(np.uint8),
        }
    )
    connection = duckdb.connect()
    try:
        connection.register("validation_scores", frame)
        temporary_path = VALIDATION_SCORES_PATH.with_suffix(".tmp.parquet")
        if temporary_path.exists():
            temporary_path.unlink()
        connection.execute(
            f"""
            COPY validation_scores
            TO {sql_string(temporary_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        os.replace(temporary_path, VALIDATION_SCORES_PATH)
    finally:
        connection.close()


def clean_cache() -> None:
    resolved_cache = CACHE_DIR.resolve()
    resolved_feature_dir = FEATURE_DIR.resolve()
    if resolved_cache.parent != resolved_feature_dir:
        raise RuntimeError("Refusing to clean cache outside the v2 feature directory")
    if CACHE_DIR.is_dir():
        expected_files = {
            path.resolve()
            for split in ("train", "validation")
            for path in cache_paths(split).values()
        }
        unexpected = [
            path for path in CACHE_DIR.iterdir() if path.resolve() not in expected_files
        ]
        if unexpected:
            raise RuntimeError(f"Refusing to remove unexpected cache files: {unexpected}")
        for path in sorted(expected_files):
            if path.is_file():
                path.unlink()
        try:
            CACHE_DIR.rmdir()
        except OSError:
            # OneDrive can retain an empty read-only reparse-point directory.
            # All cache files have already been removed at this point.
            pass


def main() -> None:
    args = parse_args()
    if args.duckdb_vectors_per_chunk <= 0:
        raise ValueError("--duckdb-vectors-per-chunk must be positive")
    for required in (TRAIN_PATH, VALIDATION_PATH, FEATURE_MANIFEST_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)

    baseline_features, context_features, feature_manifest = load_feature_contract()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()

    print("Streaming Jan 22 train features into float32 memory maps...", flush=True)
    (
        x_train_baseline,
        x_train_context,
        y_train,
        _,
        train_load_seconds,
    ) = stream_parquet_to_memmaps(
        TRAIN_PATH,
        "train",
        TRAIN_ROWS,
        baseline_features,
        context_features,
        args.duckdb_vectors_per_chunk,
        include_categories=False,
    )
    print("Streaming Feb 17 validation features into float32 memory maps...", flush=True)
    (
        x_validation_baseline,
        x_validation_context,
        y_validation_memmap,
        validation_categories,
        validation_load_seconds,
    ) = stream_parquet_to_memmaps(
        VALIDATION_PATH,
        "validation",
        VALIDATION_ROWS,
        baseline_features,
        context_features,
        args.duckdb_vectors_per_chunk,
        include_categories=True,
    )
    if validation_categories is None:
        raise AssertionError("Validation attack categories were not loaded")
    y_validation = np.asarray(y_validation_memmap)

    print("Training 76-feature baseline Random Forest...", flush=True)
    baseline_scores, baseline_continuous, baseline_warnings = fit_and_score(
        "baseline_v2",
        BASELINE_MODEL_PATH,
        x_train_baseline,
        np.asarray(y_train),
        x_validation_baseline,
        y_validation,
    )
    print("Training 128-feature context Random Forest...", flush=True)
    context_scores, context_continuous, context_warnings = fit_and_score(
        "context_v2",
        CONTEXT_MODEL_PATH,
        x_train_context,
        np.asarray(y_train),
        x_validation_context,
        y_validation,
    )

    baseline_sweep, baseline_selected, baseline_gate = sweep_thresholds(
        "baseline_v2", y_validation, baseline_scores
    )
    context_sweep, context_selected, context_gate = sweep_thresholds(
        "context_v2", y_validation, context_scores
    )
    threshold_results = pd.concat(
        [baseline_sweep, context_sweep], ignore_index=True
    )
    threshold_results.to_csv(THRESHOLD_RESULTS_PATH, index=False)

    baseline_categories = category_metrics(
        "baseline_v2",
        validation_categories,
        y_validation,
        baseline_scores,
        baseline_selected["threshold"],
    )
    context_categories = category_metrics(
        "context_v2",
        validation_categories,
        y_validation,
        context_scores,
        context_selected["threshold"],
    )
    pd.concat([baseline_categories, context_categories], ignore_index=True).to_csv(
        CATEGORY_METRICS_PATH,
        index=False,
    )

    save_validation_scores(
        validation_categories,
        y_validation,
        baseline_scores,
        context_scores,
        baseline_selected["threshold"],
        context_selected["threshold"],
    )

    comparison_metrics = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
        "fnr",
        "false_positives",
        "false_negatives",
        "fp_per_100k_benign",
    ]
    comparison_rows = []
    for metric in comparison_metrics:
        baseline_value = baseline_selected[metric]
        context_value = context_selected[metric]
        comparison_rows.append(
            {
                "metric": metric,
                "baseline_v2": baseline_value,
                "context_v2": context_value,
                "context_minus_baseline": context_value - baseline_value,
            }
        )
    for metric, key in (
        ("roc_auc", "roc_auc"),
        ("pr_auc_average_precision", "pr_auc_average_precision"),
    ):
        comparison_rows.append(
            {
                "metric": metric,
                "baseline_v2": baseline_continuous[key],
                "context_v2": context_continuous[key],
                "context_minus_baseline": (
                    context_continuous[key] - baseline_continuous[key]
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_PATH, index=False)

    if baseline_gate and context_gate:
        fpr_delta = context_selected["fpr"] - baseline_selected["fpr"]
        if fpr_delta < -0.001:
            outcome = (
                "Temporal context hypothesis supported: context reduced validation "
                "FPR by more than 0.1 percentage point while meeting recall >= 98.5%."
            )
        elif context_continuous["roc_auc"] > baseline_continuous["roc_auc"]:
            outcome = (
                "Context improved ranking but did not materially reduce the gated "
                "validation FPR."
            )
        else:
            outcome = (
                "Temporal context did not improve the gated operating point and "
                "should not be retained without further training-side evidence."
            )
    elif context_gate and not baseline_gate:
        outcome = "Only context v2 achieved the recall gate; temporal context is supported."
    elif baseline_gate and not context_gate:
        outcome = "Context v2 failed the recall gate while baseline passed; context is rejected."
    else:
        outcome = "Neither model achieved the validation recall gate."

    fuzzers_rows = pd.concat(
        [baseline_categories, context_categories], ignore_index=True
    )
    fuzzers_rows = fuzzers_rows[fuzzers_rows["attack_cat"] == "Fuzzers"]
    metrics = {
        "stage": "v2 baseline versus temporal-context ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "train_capture_date": "2015-01-22",
            "validation_capture_date": "2015-02-17",
            "locked_holdout_loaded": False,
            "locked_holdout_evaluated": False,
            "threshold_source": "Feb 17 validation only",
        },
        "selection_policy": {
            "threshold_grid": "0.01 through 0.99, step 0.01",
            "recall_gate": RECALL_GATE,
            "objective_after_gate": "minimum FPR / false positives",
            "tie_breakers": ["higher precision", "higher F1", "higher threshold"],
        },
        "random_forest_params": RANDOM_FOREST_PARAMS,
        "data": {
            "train_rows": TRAIN_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "train_label_counts": {
                str(value): int(count)
                for value, count in zip(*np.unique(y_train, return_counts=True), strict=True)
            },
            "validation_label_counts": {
                str(value): int(count)
                for value, count in zip(
                    *np.unique(y_validation, return_counts=True), strict=True
                )
            },
            "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
            "train_parquet_sha256": feature_manifest["outputs"]["train"]["sha256"],
            "validation_parquet_sha256": feature_manifest["outputs"]["validation"][
                "sha256"
            ],
        },
        "models": {
            "baseline_v2": {
                "feature_count": len(baseline_features),
                "recall_gate_achieved": baseline_gate,
                "continuous_metrics": baseline_continuous,
                "selected_operating_point": baseline_selected,
                "threshold_0_50": threshold_metrics(
                    y_validation, baseline_scores, 0.50
                ),
                "warnings": baseline_warnings,
            },
            "context_v2": {
                "feature_count": len(context_features),
                "recall_gate_achieved": context_gate,
                "continuous_metrics": context_continuous,
                "selected_operating_point": context_selected,
                "threshold_0_50": threshold_metrics(
                    y_validation, context_scores, 0.50
                ),
                "warnings": context_warnings,
            },
        },
        "fuzzers_validation": fuzzers_rows.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "outcome": outcome,
        "runtime": {
            "train_data_load_seconds": train_load_seconds,
            "validation_data_load_seconds": validation_load_seconds,
            "total_seconds": time.perf_counter() - overall_started,
        },
        "software": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "duckdb": duckdb.__version__,
        },
        "artifacts": {
            "threshold_results": str(THRESHOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "comparison": str(COMPARISON_PATH.relative_to(PROJECT_ROOT)),
            "category_metrics": str(CATEGORY_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "validation_scores": str(VALIDATION_SCORES_PATH.relative_to(PROJECT_ROOT)),
            "validation_scores_sha256": sha256_file(VALIDATION_SCORES_PATH),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.keep_cache:
        del (
            x_train_baseline,
            x_train_context,
            y_train,
            x_validation_baseline,
            x_validation_context,
            y_validation,
            y_validation_memmap,
        )
        gc.collect()
        clean_cache()

    print(json.dumps({"outcome": outcome, "models": metrics["models"]}, indent=2))
    print(f"Metrics: {METRICS_PATH}")
    print("Feb 18 locked holdout was not loaded or evaluated.")


if __name__ == "__main__":
    main()
