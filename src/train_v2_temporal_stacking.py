"""Train small v2 stackers from purged temporal OOF base-model scores.

Jan 22 is divided into five chronological blocks. A 60,000-row warm-up and
three additional 60,000-row boundaries keep positive-class support in every
OOF fold; the final fold extends to the end of the day. Four expanding-window
folds generate OOF baseline/context scores. A row purge larger than the known
maximum same-second group prevents a timestamp group from crossing a
train/predict boundary after Timestamp was removed from the model-ready
Parquet. February 17 is used only for final stacker validation. This module has
no February 18 data path.
"""

from __future__ import annotations

import argparse
import gc
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .train_v2_ablation import (
        RANDOM_FOREST_PARAMS,
        RECALL_GATE,
        category_metrics,
        load_feature_contract,
        quote_identifier,
        sha256_file,
        sql_string,
        sweep_thresholds,
        threshold_metrics,
    )
except ImportError:  # Direct execution
    from train_v2_ablation import (
        RANDOM_FOREST_PARAMS,
        RECALL_GATE,
        category_metrics,
        load_feature_contract,
        quote_identifier,
        sha256_file,
        sql_string,
        sweep_thresholds,
        threshold_metrics,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2"
TRAIN_PATH = FEATURE_DIR / "train_features.parquet"
VALIDATION_PATH = FEATURE_DIR / "validation_features.parquet"
FEATURE_MANIFEST_PATH = FEATURE_DIR / "feature_manifest.json"
CACHE_DIR = FEATURE_DIR / "_temporal_stacking_cache"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "_v2_temporal_oof_checkpoints"

FROZEN_BASELINE_MODEL_PATH = MODELS_DIR / "v2_baseline_random_forest.joblib"
FROZEN_CONTEXT_MODEL_PATH = MODELS_DIR / "v2_context_random_forest.joblib"
FROZEN_ABLATION_METRICS_PATH = MODELS_DIR / "v2_ablation_metrics.json"
FROZEN_VALIDATION_SCORES_PATH = MODELS_DIR / "v2_validation_scores.parquet"

LOGISTIC_MODEL_PATH = MODELS_DIR / "v2_stacker_logistic_regression.joblib"
HIST_MODEL_PATH = MODELS_DIR / "v2_stacker_hist_gradient_boosting.joblib"
OOF_SCORES_PATH = MODELS_DIR / "v2_temporal_oof_scores.parquet"
OOF_MANIFEST_PATH = MODELS_DIR / "v2_temporal_oof_manifest.json"
METRICS_PATH = MODELS_DIR / "v2_stacking_metrics.json"
THRESHOLD_RESULTS_PATH = MODELS_DIR / "v2_stacking_threshold_results.csv"
COMPARISON_PATH = MODELS_DIR / "v2_stacking_comparison.csv"
CATEGORY_METRICS_PATH = MODELS_DIR / "v2_stacking_category_metrics.csv"
VALIDATION_SCORES_PATH = MODELS_DIR / "v2_stacking_validation_scores.parquet"

TRAIN_ROWS = 1_765_922
VALIDATION_ROWS = 498_890
BASELINE_FEATURE_COUNT = 76
CONTEXT_FEATURE_COUNT = 128
TEMPORAL_BOUNDARIES = (0, 60_000, 120_000, 180_000, 240_000, TRAIN_ROWS)
MAX_FLOWS_PER_TIMESTAMP_SECOND = 776
BOUNDARY_PURGE_ROWS = 1_024
EXPECTED_OOF_ROWS = 1_701_826
STRICT_FPR_TARGET = 0.02
DESIRED_FUZZER_FN_MAX_EXCLUSIVE = 200

CATEGORY_NAMES = (
    "Analysis",
    "Backdoor",
    "Benign",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
)
CATEGORY_TO_CODE = {category: code for code, category in enumerate(CATEGORY_NAMES)}

SCORE_META_FEATURES = (
    "baseline_score",
    "context_score",
    "baseline_minus_context_score",
    "max_score",
    "min_score",
)
CONTEXT_META_FEATURES = (
    "Protocol",
    "dst_unique_sport_60s",
    "src_dst_conn_60s",
    "src_dport_conn_60s",
    "src_conn_60s",
    "dst_conn_60s",
    "src_packets_60s",
    "dst_packets_60s",
    "src_mean_packets_per_flow_60s",
    "dst_mean_packets_per_flow_60s",
    "src_udp_ratio_60s",
    "dst_udp_ratio_60s",
)
META_FEATURES = (*SCORE_META_FEATURES, *CONTEXT_META_FEATURES)

LOGISTIC_PARAMS: dict[str, Any] = {
    "max_iter": 1_000,
    "solver": "lbfgs",
}
HIST_GRADIENT_BOOSTING_PARAMS: dict[str, Any] = {
    "max_iter": 100,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 42,
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
        "--restart-folds",
        action="store_true",
        help="Ignore valid fold score checkpoints and regenerate every OOF fold.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep float32 feature memory maps after completion.",
    )
    return parser.parse_args()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_joblib_dump(model: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, path)


def atomic_npz_dump(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def temporal_folds(
    total_rows: int = TRAIN_ROWS,
    purge_rows: int = BOUNDARY_PURGE_ROWS,
    max_same_time_rows: int = MAX_FLOWS_PER_TIMESTAMP_SECOND,
) -> list[dict[str, int]]:
    if total_rows != TRAIN_ROWS:
        raise ValueError("This frozen experiment requires the complete Jan 22 split")
    if purge_rows <= max_same_time_rows:
        raise ValueError("Boundary purge must exceed the largest same-time group")
    boundaries = list(TEMPORAL_BOUNDARIES)
    if boundaries[0] != 0 or boundaries[-1] != total_rows:
        raise AssertionError("Temporal boundaries do not span the Jan 22 split")
    folds: list[dict[str, int]] = []
    for fold_number in range(1, len(boundaries) - 1):
        train_end = boundaries[fold_number]
        prediction_start = train_end + purge_rows
        prediction_end = boundaries[fold_number + 1]
        if prediction_start >= prediction_end:
            raise ValueError("Temporal block is too small for the boundary purge")
        folds.append(
            {
                "fold": fold_number,
                "train_start": 0,
                "train_end_exclusive": train_end,
                "purge_start": train_end,
                "purge_end_exclusive": prediction_start,
                "prediction_start": prediction_start,
                "prediction_end_exclusive": prediction_end,
                "training_rows": train_end,
                "prediction_rows": prediction_end - prediction_start,
            }
        )
    return folds


def cache_paths(split: str) -> dict[str, Path]:
    return {
        "context": CACHE_DIR / f"{split}_context.float32.mmap",
        "target": CACHE_DIR / f"{split}_target.uint8.mmap",
        "category": CACHE_DIR / f"{split}_category.uint8.mmap",
    }


def stream_split(
    path: Path,
    split: str,
    expected_rows: int,
    context_features: list[str],
    vectors_per_chunk: int,
) -> tuple[np.memmap, np.memmap, np.memmap, dict[str, int], float]:
    started = time.perf_counter()
    paths = cache_paths(split)
    matrix = np.memmap(
        paths["context"],
        mode="w+",
        dtype=np.float32,
        shape=(expected_rows, len(context_features)),
    )
    target = np.memmap(
        paths["target"], mode="w+", dtype=np.uint8, shape=(expected_rows,)
    )
    category_code = np.memmap(
        paths["category"], mode="w+", dtype=np.uint8, shape=(expected_rows,)
    )
    selected = [*context_features, "label", "attack_cat"]
    selected_sql = ", ".join(quote_identifier(column) for column in selected)
    counts: dict[str, int] = {category: 0 for category in CATEGORY_NAMES}
    connection = duckdb.connect()
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
            labels = chunk["label"].to_numpy(dtype=np.uint8)
            categories = chunk["attack_cat"].astype(str)
            codes = categories.map(CATEGORY_TO_CODE)
            if codes.isna().any() or not np.isfinite(values).all():
                raise AssertionError(f"{split}: invalid feature/category value")
            matrix[row_offset:end] = values
            target[row_offset:end] = labels
            category_code[row_offset:end] = codes.to_numpy(dtype=np.uint8)
            for category, count in categories.value_counts().items():
                counts[category] += int(count)
            row_offset = end
            print(f"Loaded {split}: {row_offset:,}/{expected_rows:,}", flush=True)
    finally:
        connection.close()
    if row_offset != expected_rows:
        raise AssertionError(f"{split}: expected {expected_rows:,}, got {row_offset:,}")
    if not np.isin(np.asarray(target), [0, 1]).all():
        raise AssertionError(f"{split}: target contains values outside 0/1")
    benign_code = CATEGORY_TO_CODE["Benign"]
    if not np.array_equal(np.asarray(target) == 0, np.asarray(category_code) == benign_code):
        raise AssertionError(f"{split}: category and binary target disagree")
    matrix.flush()
    target.flush()
    category_code.flush()
    return matrix, target, category_code, counts, time.perf_counter() - started


def checkpoint_paths(fold_number: int) -> tuple[Path, Path]:
    return (
        CHECKPOINT_DIR / f"fold_{fold_number}_scores.npz",
        CHECKPOINT_DIR / f"fold_{fold_number}_metadata.json",
    )


def fold_fingerprint(
    fold: dict[str, int],
    feature_manifest_sha256: str,
    train_sha256: str,
) -> dict[str, Any]:
    return {
        "fold": fold,
        "random_forest_params": RANDOM_FOREST_PARAMS,
        "baseline_feature_count": BASELINE_FEATURE_COUNT,
        "context_feature_count": CONTEXT_FEATURE_COUNT,
        "feature_manifest_sha256": feature_manifest_sha256,
        "train_parquet_sha256": train_sha256,
        "scikit_learn": sklearn.__version__,
        "sample_weight": None,
    }


def load_fold_checkpoint(
    fold: dict[str, int], expected_fingerprint: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]] | None:
    score_path, metadata_path = checkpoint_paths(fold["fold"])
    if not score_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected_fingerprint:
        return None
    if metadata.get("score_sha256") != sha256_file(score_path):
        return None
    with np.load(score_path) as saved:
        baseline_scores = saved["baseline_scores"]
        context_scores = saved["context_scores"]
    expected_rows = fold["prediction_rows"]
    if baseline_scores.shape != (expected_rows,) or context_scores.shape != (
        expected_rows,
    ):
        return None
    if not np.isfinite(baseline_scores).all() or not np.isfinite(context_scores).all():
        return None
    return baseline_scores, context_scores, metadata


def fit_fold_base_models(
    fold: dict[str, int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    fingerprint: dict[str, Any],
    restart: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not restart:
        checkpoint = load_fold_checkpoint(fold, fingerprint)
        if checkpoint is not None:
            baseline_scores, context_scores, metadata = checkpoint
            print(f"Resumed verified temporal fold {fold['fold']}", flush=True)
            metadata["resumed_from_checkpoint"] = True
            return baseline_scores, context_scores, metadata

    train_end = fold["train_end_exclusive"]
    prediction_start = fold["prediction_start"]
    prediction_end = fold["prediction_end_exclusive"]
    if train_end > prediction_start or prediction_start >= prediction_end:
        raise AssertionError("Temporal train/prediction ranges overlap")
    if not np.array_equal(np.unique(y_train[:train_end]), np.array([0, 1])):
        raise AssertionError("A temporal training prefix does not contain both classes")
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        baseline = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        started = time.perf_counter()
        baseline.fit(x_train[:train_end, :BASELINE_FEATURE_COUNT], y_train[:train_end])
        baseline_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        baseline_scores = baseline.predict_proba(
            x_train[prediction_start:prediction_end, :BASELINE_FEATURE_COUNT]
        )[:, 1]
        baseline_inference_seconds = time.perf_counter() - started
        del baseline
        gc.collect()

        context = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
        started = time.perf_counter()
        context.fit(x_train[:train_end], y_train[:train_end])
        context_fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        context_scores = context.predict_proba(
            x_train[prediction_start:prediction_end]
        )[:, 1]
        context_inference_seconds = time.perf_counter() - started
        captured_warnings = [str(record.message) for record in warning_records]
        del context
        gc.collect()

    if not np.isfinite(baseline_scores).all() or not np.isfinite(context_scores).all():
        raise AssertionError("A temporal fold produced non-finite scores")
    score_path, metadata_path = checkpoint_paths(fold["fold"])
    atomic_npz_dump(
        score_path,
        baseline_scores=baseline_scores,
        context_scores=context_scores,
    )
    metadata = {
        "fingerprint": fingerprint,
        "score_sha256": sha256_file(score_path),
        "baseline_training_seconds": baseline_fit_seconds,
        "baseline_inference_seconds": baseline_inference_seconds,
        "context_training_seconds": context_fit_seconds,
        "context_inference_seconds": context_inference_seconds,
        "warnings": captured_warnings,
        "resumed_from_checkpoint": False,
    }
    atomic_json_dump(metadata, metadata_path)
    return baseline_scores, context_scores, metadata


def assemble_meta_matrix(
    context_matrix: np.ndarray,
    rows: np.ndarray,
    baseline_scores: np.ndarray,
    context_scores: np.ndarray,
    context_feature_indices: list[int],
) -> np.ndarray:
    if len(rows) != len(baseline_scores) or len(rows) != len(context_scores):
        raise AssertionError("Meta-feature inputs have inconsistent row counts")
    score_features = np.column_stack(
        [
            baseline_scores,
            context_scores,
            baseline_scores - context_scores,
            np.maximum(baseline_scores, context_scores),
            np.minimum(baseline_scores, context_scores),
        ]
    ).astype(np.float32, copy=False)
    selected_context = np.asarray(
        context_matrix[np.ix_(rows, context_feature_indices)], dtype=np.float32
    )
    meta = np.concatenate([score_features, selected_context], axis=1)
    if meta.shape != (len(rows), len(META_FEATURES)) or not np.isfinite(meta).all():
        raise AssertionError("Invalid stacker meta-feature matrix")
    return meta


def make_logistic_stacker() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(**LOGISTIC_PARAMS)),
        ]
    )


def make_hist_stacker() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HIST_GRADIENT_BOOSTING_PARAMS)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    connection = duckdb.connect()
    try:
        connection.register("output_frame", frame)
        connection.execute(
            f"""
            COPY output_frame TO {sql_string(temporary)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        os.replace(temporary, path)
    finally:
        connection.close()


def fit_stacker(
    name: str,
    estimator: Any,
    model_path: Path,
    x_oof: np.ndarray,
    y_oof: np.ndarray,
    x_validation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        started = time.perf_counter()
        estimator.fit(x_oof, y_oof)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = estimator.predict_proba(x_validation)[:, 1]
        inference_seconds = time.perf_counter() - started
        captured_warnings = [str(record.message) for record in warning_records]
    if not np.isfinite(scores).all():
        raise AssertionError(f"{name}: non-finite validation scores")
    atomic_joblib_dump(estimator, model_path)
    reloaded = joblib.load(model_path)
    check_rows = min(10_000, len(x_validation))
    reloaded_scores = reloaded.predict_proba(x_validation[:check_rows])[:, 1]
    max_difference = float(np.max(np.abs(scores[:check_rows] - reloaded_scores)))
    if max_difference > 1e-12:
        raise AssertionError(f"{name}: saved stacker did not reproduce scores")
    continuous = {
        "training_seconds": training_seconds,
        "validation_inference_seconds": inference_seconds,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "model_size_bytes": model_path.stat().st_size,
        "model_sha256": sha256_file(model_path),
        "reload_check_rows": check_rows,
        "reload_max_score_difference": max_difference,
    }
    del reloaded, estimator
    gc.collect()
    return scores, continuous, captured_warnings


def evaluate_stacker(
    name: str,
    scores: np.ndarray,
    y_validation: np.ndarray,
    validation_categories: np.ndarray,
    continuous: dict[str, Any],
    captured_warnings: list[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    continuous["roc_auc"] = float(roc_auc_score(y_validation, scores))
    continuous["pr_auc_average_precision"] = float(
        average_precision_score(y_validation, scores)
    )
    sweep, selected, gate_met = sweep_thresholds(name, y_validation, scores)
    categories = category_metrics(
        name,
        validation_categories,
        y_validation,
        scores,
        selected["threshold"],
    )
    fuzzer = categories[categories["attack_cat"] == "Fuzzers"].iloc[0]
    analysis = categories[categories["attack_cat"] == "Analysis"].iloc[0]
    result = {
        "recall_gate_achieved": bool(gate_met),
        "continuous_metrics": continuous,
        "selected_operating_point": selected,
        "threshold_0_50": threshold_metrics(y_validation, scores, 0.50),
        "fuzzer_validation": {
            "total": int(fuzzer["total"]),
            "false_negatives": int(fuzzer["missed_attacks"]),
            "recall": float(fuzzer["attack_recall"]),
        },
        "analysis_validation": {
            "total": int(analysis["total"]),
            "false_negatives": int(analysis["missed_attacks"]),
            "recall": float(analysis["attack_recall"]),
        },
        "success": {
            "recall_at_least_0_985": bool(selected["recall"] >= RECALL_GATE),
            "fpr_below_0_02": bool(selected["fpr"] < STRICT_FPR_TARGET),
            "fuzzer_fn_below_200": bool(
                int(fuzzer["missed_attacks"]) < DESIRED_FUZZER_FN_MAX_EXCLUSIVE
            ),
            "strict_operational_success": bool(
                selected["recall"] >= RECALL_GATE
                and selected["fpr"] < STRICT_FPR_TARGET
            ),
            "strong_success_including_fuzzers": bool(
                selected["recall"] >= RECALL_GATE
                and selected["fpr"] < STRICT_FPR_TARGET
                and int(fuzzer["missed_attacks"])
                < DESIRED_FUZZER_FN_MAX_EXCLUSIVE
            ),
        },
        "warnings": captured_warnings,
    }
    return result, sweep, categories


def load_frozen_validation_scores() -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        scores = connection.execute(
            f"SELECT * FROM read_parquet({sql_string(FROZEN_VALIDATION_SCORES_PATH)}) "
            "ORDER BY validation_row"
        ).fetchdf()
    finally:
        connection.close()
    if len(scores) != VALIDATION_ROWS:
        raise AssertionError("Frozen validation score row count changed")
    if scores["validation_row"].tolist() != list(range(VALIDATION_ROWS)):
        raise AssertionError("Frozen validation scores lost row order")
    return scores


def clean_cache() -> None:
    resolved_cache = CACHE_DIR.resolve()
    if resolved_cache.parent != FEATURE_DIR.resolve():
        raise RuntimeError("Refusing to clean cache outside v2 feature directory")
    if not CACHE_DIR.is_dir():
        return
    expected = {
        path.resolve()
        for split in ("train", "validation")
        for path in cache_paths(split).values()
    }
    unexpected = [path for path in CACHE_DIR.iterdir() if path.resolve() not in expected]
    if unexpected:
        raise RuntimeError(f"Refusing to remove unexpected cache files: {unexpected}")
    for path in sorted(expected):
        if path.is_file():
            path.unlink()
    try:
        CACHE_DIR.rmdir()
    except OSError:
        pass


def main() -> None:
    args = parse_args()
    if args.duckdb_vectors_per_chunk <= 0:
        raise ValueError("--duckdb-vectors-per-chunk must be positive")
    for required in (
        TRAIN_PATH,
        VALIDATION_PATH,
        FEATURE_MANIFEST_PATH,
        FROZEN_BASELINE_MODEL_PATH,
        FROZEN_CONTEXT_MODEL_PATH,
        FROZEN_ABLATION_METRICS_PATH,
        FROZEN_VALIDATION_SCORES_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    baseline_features, context_features, feature_manifest = load_feature_contract()
    if len(baseline_features) != BASELINE_FEATURE_COUNT or len(
        context_features
    ) != CONTEXT_FEATURE_COUNT:
        raise AssertionError("Frozen v2 feature counts changed")
    if feature_manifest["split_policy"]["sort_key"] != ["Timestamp"]:
        raise AssertionError("Jan 22 feature rows are not declared Timestamp-sorted")
    if not feature_manifest["split_policy"]["context_state_reset_at_each_split"]:
        raise AssertionError("Context state reset contract changed")
    if any(feature not in context_features for feature in CONTEXT_META_FEATURES):
        raise AssertionError("A frozen stacker meta-feature is absent")
    context_feature_indices = [
        context_features.index(feature) for feature in CONTEXT_META_FEATURES
    ]
    folds = temporal_folds()
    if sum(fold["prediction_rows"] for fold in folds) != EXPECTED_OOF_ROWS:
        raise AssertionError("Unexpected temporal OOF row count")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()
    frozen_hashes_before = {
        "baseline_model": sha256_file(FROZEN_BASELINE_MODEL_PATH),
        "context_model": sha256_file(FROZEN_CONTEXT_MODEL_PATH),
        "ablation_metrics": sha256_file(FROZEN_ABLATION_METRICS_PATH),
        "validation_scores": sha256_file(FROZEN_VALIDATION_SCORES_PATH),
    }

    print("Streaming Timestamp-sorted Jan 22 features...", flush=True)
    x_train, y_train_memmap, train_category_codes, train_counts, train_load_seconds = (
        stream_split(
            TRAIN_PATH,
            "train",
            TRAIN_ROWS,
            context_features,
            args.duckdb_vectors_per_chunk,
        )
    )
    print("Streaming Feb 17 validation features...", flush=True)
    (
        x_validation,
        y_validation_memmap,
        validation_category_codes,
        validation_counts,
        validation_load_seconds,
    ) = stream_split(
        VALIDATION_PATH,
        "validation",
        VALIDATION_ROWS,
        context_features,
        args.duckdb_vectors_per_chunk,
    )
    y_train = np.asarray(y_train_memmap)
    y_validation = np.asarray(y_validation_memmap)
    validation_categories = np.asarray(CATEGORY_NAMES, dtype=object)[
        np.asarray(validation_category_codes)
    ]

    feature_manifest_hash = sha256_file(FEATURE_MANIFEST_PATH)
    train_hash = feature_manifest["outputs"]["train"]["sha256"]
    validation_hash = feature_manifest["outputs"]["validation"]["sha256"]
    baseline_oof = np.full(TRAIN_ROWS, np.nan, dtype=np.float64)
    context_oof = np.full(TRAIN_ROWS, np.nan, dtype=np.float64)
    fold_assignment = np.zeros(TRAIN_ROWS, dtype=np.uint8)
    fold_metadata: list[dict[str, Any]] = []

    for fold in folds:
        print(
            f"Temporal fold {fold['fold']}: train [0:{fold['train_end_exclusive']:,}), "
            f"purge {fold['purge_start']:,}:{fold['purge_end_exclusive']:,}, "
            f"predict {fold['prediction_start']:,}:{fold['prediction_end_exclusive']:,}",
            flush=True,
        )
        fingerprint = fold_fingerprint(fold, feature_manifest_hash, train_hash)
        baseline_scores, context_scores, metadata = fit_fold_base_models(
            fold,
            x_train,
            y_train,
            fingerprint,
            args.restart_folds,
        )
        start = fold["prediction_start"]
        end = fold["prediction_end_exclusive"]
        if np.any(fold_assignment[start:end] != 0):
            raise AssertionError("An OOF row was assigned more than once")
        baseline_oof[start:end] = baseline_scores
        context_oof[start:end] = context_scores
        fold_assignment[start:end] = fold["fold"]
        fold_metadata.append(metadata)

    oof_rows = np.flatnonzero(fold_assignment > 0)
    if len(oof_rows) != EXPECTED_OOF_ROWS:
        raise AssertionError("Not every expected OOF row received one score pair")
    if not np.isfinite(baseline_oof[oof_rows]).all() or not np.isfinite(
        context_oof[oof_rows]
    ).all():
        raise AssertionError("OOF score coverage contains a gap")
    for fold in folds:
        mask = fold_assignment == fold["fold"]
        rows = np.flatnonzero(mask)
        if rows.min() < fold["prediction_start"] or rows.max() >= fold[
            "prediction_end_exclusive"
        ]:
            raise AssertionError("OOF row escaped its temporal fold")
        if fold["train_end_exclusive"] > rows.min():
            raise AssertionError("An OOF row was seen during its base-model fit")

    x_oof_meta = assemble_meta_matrix(
        x_train,
        oof_rows,
        baseline_oof[oof_rows],
        context_oof[oof_rows],
        context_feature_indices,
    )
    y_oof = y_train[oof_rows]
    oof_categories = np.asarray(CATEGORY_NAMES, dtype=object)[
        np.asarray(train_category_codes)[oof_rows]
    ]
    oof_fold_label_counts = [
        {
            "fold": fold["fold"],
            "normal": int(
                np.sum((fold_assignment == fold["fold"]) & (y_train == 0))
            ),
            "attack": int(
                np.sum((fold_assignment == fold["fold"]) & (y_train == 1))
            ),
        }
        for fold in folds
    ]
    oof_category_counts = {
        category: int(np.sum(oof_categories == category))
        for category in CATEGORY_NAMES
    }
    oof_frame = pd.DataFrame(
        {
            "train_row": oof_rows,
            "temporal_fold": fold_assignment[oof_rows],
            "attack_cat": oof_categories,
            "label": y_oof,
        }
    )
    for index, feature in enumerate(META_FEATURES):
        oof_frame[feature] = x_oof_meta[:, index]
    atomic_write_parquet(oof_frame, OOF_SCORES_PATH)
    del oof_frame
    gc.collect()

    frozen_validation = load_frozen_validation_scores()
    expected_validation_categories = np.asarray(
        frozen_validation["attack_cat"].astype(str), dtype=object
    )
    if not np.array_equal(
        frozen_validation["label"].to_numpy(dtype=np.uint8), y_validation
    ) or not np.array_equal(expected_validation_categories, validation_categories):
        raise AssertionError("Frozen validation scores do not align with feature rows")
    validation_rows = np.arange(VALIDATION_ROWS, dtype=np.int64)
    x_validation_meta = assemble_meta_matrix(
        x_validation,
        validation_rows,
        frozen_validation["baseline_attack_score"].to_numpy(dtype=float),
        frozen_validation["context_attack_score"].to_numpy(dtype=float),
        context_feature_indices,
    )

    stacker_results: dict[str, dict[str, Any]] = {}
    stacker_scores: dict[str, np.ndarray] = {}
    threshold_frames: list[pd.DataFrame] = []
    category_frames: list[pd.DataFrame] = []

    print("Training Logistic Regression stacker on Jan 22 temporal OOF rows...", flush=True)
    logistic_scores, logistic_continuous, logistic_warnings = fit_stacker(
        "logistic_stacker",
        make_logistic_stacker(),
        LOGISTIC_MODEL_PATH,
        x_oof_meta,
        y_oof,
        x_validation_meta,
    )
    logistic_result, logistic_sweep, logistic_categories = evaluate_stacker(
        "logistic_stacker",
        logistic_scores,
        y_validation,
        validation_categories,
        logistic_continuous,
        logistic_warnings,
    )
    stacker_results["logistic_stacker"] = logistic_result
    stacker_scores["logistic_stacker"] = logistic_scores
    threshold_frames.append(logistic_sweep)
    category_frames.append(logistic_categories)

    if not logistic_result["success"]["strict_operational_success"]:
        print(
            "Logistic stacker missed the strict target; training the predeclared "
            "shallow HistGradientBoosting stacker...",
            flush=True,
        )
        hist_scores, hist_continuous, hist_warnings = fit_stacker(
            "hist_gradient_boosting_stacker",
            make_hist_stacker(),
            HIST_MODEL_PATH,
            x_oof_meta,
            y_oof,
            x_validation_meta,
        )
        hist_result, hist_sweep, hist_categories = evaluate_stacker(
            "hist_gradient_boosting_stacker",
            hist_scores,
            y_validation,
            validation_categories,
            hist_continuous,
            hist_warnings,
        )
        stacker_results["hist_gradient_boosting_stacker"] = hist_result
        stacker_scores["hist_gradient_boosting_stacker"] = hist_scores
        threshold_frames.append(hist_sweep)
        category_frames.append(hist_categories)

    threshold_results = pd.concat(threshold_frames, ignore_index=True)
    all_categories = pd.concat(category_frames, ignore_index=True)
    comparison_rows: list[dict[str, Any]] = [
        {
            "model": "frozen_context",
            "threshold": 0.10,
            "recall": 0.9859078856918393,
            "precision": 0.6242693694814078,
            "f1": 0.7644774414620217,
            "roc_auc": 0.9946836002585427,
            "pr_auc_average_precision": 0.8477457428319373,
            "fpr": 0.02525474166394998,
            "false_positives": 12_085,
            "overall_false_negatives": 287,
            "fuzzer_false_negatives": 261,
            "analysis_false_negatives": 19,
            "fp_per_100k_benign": 2_525.474166394998,
            "recall_gate_achieved": True,
            "strict_operational_success": False,
            "strong_success_including_fuzzers": False,
        }
    ]
    for name, result in stacker_results.items():
        selected = result["selected_operating_point"]
        comparison_rows.append(
            {
                "model": name,
                "threshold": selected["threshold"],
                "recall": selected["recall"],
                "precision": selected["precision"],
                "f1": selected["f1"],
                "roc_auc": result["continuous_metrics"]["roc_auc"],
                "pr_auc_average_precision": result["continuous_metrics"][
                    "pr_auc_average_precision"
                ],
                "fpr": selected["fpr"],
                "false_positives": selected["false_positives"],
                "overall_false_negatives": selected["false_negatives"],
                "fuzzer_false_negatives": result["fuzzer_validation"][
                    "false_negatives"
                ],
                "analysis_false_negatives": result["analysis_validation"][
                    "false_negatives"
                ],
                "fp_per_100k_benign": selected["fp_per_100k_benign"],
                "recall_gate_achieved": result["recall_gate_achieved"],
                "strict_operational_success": result["success"][
                    "strict_operational_success"
                ],
                "strong_success_including_fuzzers": result["success"][
                    "strong_success_including_fuzzers"
                ],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    eligible_stackers = comparison[
        (comparison["model"] != "frozen_context")
        & comparison["recall_gate_achieved"]
    ]
    if eligible_stackers.empty:
        candidate_rows = comparison[comparison["model"] != "frozen_context"]
        selected_row = candidate_rows.sort_values(
            ["recall", "false_positives"], ascending=[False, True]
        ).iloc[0]
    else:
        selected_row = eligible_stackers.sort_values(
            ["false_positives", "pr_auc_average_precision", "f1"],
            ascending=[True, False, False],
        ).iloc[0]
    selected_name = str(selected_row["model"])

    threshold_results.to_csv(THRESHOLD_RESULTS_PATH, index=False)
    all_categories.to_csv(CATEGORY_METRICS_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    validation_score_frame = frozen_validation[
        ["validation_row", "attack_cat", "label", "baseline_attack_score", "context_attack_score"]
    ].copy()
    for name, scores in stacker_scores.items():
        threshold = stacker_results[name]["selected_operating_point"]["threshold"]
        validation_score_frame[f"{name}_score"] = scores
        validation_score_frame[f"{name}_selected_prediction"] = (
            scores >= threshold
        ).astype(np.uint8)
    atomic_write_parquet(validation_score_frame, VALIDATION_SCORES_PATH)

    oof_manifest = {
        "stage": "v2 purged expanding-window temporal OOF",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "2015-01-22 only",
        "row_order": "Timestamp ascending from frozen feature build",
        "timestamp_column_used_as_model_feature": False,
        "block_count": len(TEMPORAL_BOUNDARIES) - 1,
        "block_boundaries": list(TEMPORAL_BOUNDARIES),
        "boundary_design": (
            "Jan 22-only temporal boundaries retain attack support in all four "
            "OOF prediction intervals; the long final interval represents the "
            "remaining benign-heavy tail."
        ),
        "warmup_rows": folds[0]["train_end_exclusive"],
        "folds": folds,
        "boundary_purge_rows_per_fold": BOUNDARY_PURGE_ROWS,
        "known_maximum_rows_in_one_timestamp_second": MAX_FLOWS_PER_TIMESTAMP_SECOND,
        "purge_exceeds_maximum_timestamp_group": True,
        "total_purged_boundary_rows": BOUNDARY_PURGE_ROWS * len(folds),
        "oof_rows": len(oof_rows),
        "oof_label_counts": {
            "normal": int(np.sum(y_oof == 0)),
            "attack": int(np.sum(y_oof == 1)),
        },
        "oof_fold_label_counts": oof_fold_label_counts,
        "oof_category_counts": oof_category_counts,
        "warmup_label_counts": {
            "normal": int(np.sum(y_train[: TEMPORAL_BOUNDARIES[1]] == 0)),
            "attack": int(np.sum(y_train[: TEMPORAL_BOUNDARIES[1]] == 1)),
        },
        "rejected_equal_row_block_design": {
            "boundaries": [0, 353_184, 706_368, 1_059_553, 1_412_737, TRAIN_ROWS],
            "observed_post_warmup_attack_rows": 0,
            "reason": (
                "All Jan 22 attacks end by train row 283,648, so an equal-fifths "
                "warm-up absorbs the complete positive class and cannot train a "
                "binary stacker. This Jan 22-only design was rejected before any "
                "stacker could be fitted."
            ),
        },
        "every_saved_oof_row_scored_once": True,
        "future_rows_used_for_base_model_training": False,
        "random_forest_params": RANDOM_FOREST_PARAMS,
        "sample_weight": None,
        "fold_runtime_and_checkpoints": fold_metadata,
        "meta_features": list(META_FEATURES),
        "train_parquet_sha256": train_hash,
        "feature_manifest_sha256": feature_manifest_hash,
        "oof_scores_path": str(OOF_SCORES_PATH.relative_to(PROJECT_ROOT)),
        "oof_scores_sha256": sha256_file(OOF_SCORES_PATH),
        "locked_holdout_loaded": False,
        "locked_holdout_evaluated": False,
    }
    atomic_json_dump(oof_manifest, OOF_MANIFEST_PATH)

    frozen_hashes_after = {
        "baseline_model": sha256_file(FROZEN_BASELINE_MODEL_PATH),
        "context_model": sha256_file(FROZEN_CONTEXT_MODEL_PATH),
        "ablation_metrics": sha256_file(FROZEN_ABLATION_METRICS_PATH),
        "validation_scores": sha256_file(FROZEN_VALIDATION_SCORES_PATH),
    }
    if frozen_hashes_before != frozen_hashes_after:
        raise AssertionError("A frozen v2 artifact changed during stacking")

    base_oof_training_seconds = float(
        sum(
            fold["baseline_training_seconds"] + fold["context_training_seconds"]
            for fold in fold_metadata
        )
    )
    base_oof_inference_seconds = float(
        sum(
            fold["baseline_inference_seconds"] + fold["context_inference_seconds"]
            for fold in fold_metadata
        )
    )
    stacker_training_seconds = float(
        sum(
            result["continuous_metrics"]["training_seconds"]
            for result in stacker_results.values()
        )
    )
    stacker_inference_seconds = float(
        sum(
            result["continuous_metrics"]["validation_inference_seconds"]
            for result in stacker_results.values()
        )
    )

    metrics = {
        "stage": "v2 temporal OOF stacking",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "base_training_and_oof_source": "2015-01-22",
            "stacker_training_source": "2015-01-22 temporal OOF rows only",
            "validation_capture_date": "2015-02-17",
            "feb17_used_for_stacker_fit": False,
            "locked_holdout_loaded": False,
            "locked_holdout_evaluated": False,
        },
        "selection_policy": {
            "threshold_grid": "0.01 through 0.99, step 0.01",
            "recall_gate": RECALL_GATE,
            "objective_after_gate": "minimum FPR / false positives",
            "strict_success": "recall >= 0.985 and FPR < 0.02",
            "strong_success": "strict success and Fuzzer FN < 200",
        },
        "temporal_oof": oof_manifest,
        "meta_feature_count": len(META_FEATURES),
        "meta_features": list(META_FEATURES),
        "models": stacker_results,
        "selected_by_fixed_policy": json.loads(selected_row.to_json()),
        "strict_success_models": [
            name
            for name, result in stacker_results.items()
            if result["success"]["strict_operational_success"]
        ],
        "strong_success_models": [
            name
            for name, result in stacker_results.items()
            if result["success"]["strong_success_including_fuzzers"]
        ],
        "data": {
            "train_rows": TRAIN_ROWS,
            "oof_rows": len(oof_rows),
            "validation_rows": VALIDATION_ROWS,
            "train_category_counts": train_counts,
            "validation_category_counts": validation_counts,
            "train_parquet_sha256": train_hash,
            "validation_parquet_sha256": validation_hash,
        },
        "runtime": {
            "train_data_load_seconds": train_load_seconds,
            "validation_data_load_seconds": validation_load_seconds,
            "base_oof_training_seconds_from_checkpoints": base_oof_training_seconds,
            "base_oof_inference_seconds_from_checkpoints": base_oof_inference_seconds,
            "stacker_training_seconds": stacker_training_seconds,
            "stacker_validation_inference_seconds": stacker_inference_seconds,
            "model_compute_seconds": (
                base_oof_training_seconds
                + base_oof_inference_seconds
                + stacker_training_seconds
                + stacker_inference_seconds
            ),
            "current_execution_used_fold_checkpoints": bool(
                all(fold.get("resumed_from_checkpoint") for fold in fold_metadata)
            ),
            "current_execution_total_seconds": time.perf_counter() - overall_started,
        },
        "reproducibility": {
            "frozen_artifact_hashes_before": frozen_hashes_before,
            "frozen_artifact_hashes_after": frozen_hashes_after,
            "frozen_artifacts_unchanged": True,
            "software": {
                "python": os.sys.version,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "duckdb": duckdb.__version__,
            },
        },
        "artifacts": {
            "oof_manifest": str(OOF_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "oof_scores": str(OOF_SCORES_PATH.relative_to(PROJECT_ROOT)),
            "logistic_model": str(LOGISTIC_MODEL_PATH.relative_to(PROJECT_ROOT)),
            "hist_model": (
                str(HIST_MODEL_PATH.relative_to(PROJECT_ROOT))
                if HIST_MODEL_PATH.is_file()
                else None
            ),
            "threshold_results": str(THRESHOLD_RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "comparison": str(COMPARISON_PATH.relative_to(PROJECT_ROOT)),
            "category_metrics": str(CATEGORY_METRICS_PATH.relative_to(PROJECT_ROOT)),
            "validation_scores": str(VALIDATION_SCORES_PATH.relative_to(PROJECT_ROOT)),
            "validation_scores_sha256": sha256_file(VALIDATION_SCORES_PATH),
        },
    }
    atomic_json_dump(metrics, METRICS_PATH)

    if not args.keep_cache:
        del (
            x_train,
            x_validation,
            x_oof_meta,
            x_validation_meta,
            y_train,
            y_train_memmap,
            y_validation,
            y_validation_memmap,
            train_category_codes,
            validation_category_codes,
        )
        gc.collect()
        clean_cache()

    print(comparison.to_string(index=False), flush=True)
    print(f"Selected stacker by fixed policy: {selected_name}", flush=True)
    print(f"Metrics: {METRICS_PATH}", flush=True)
    print("Feb 17 was validation only. Feb 18 was not loaded or evaluated.", flush=True)


if __name__ == "__main__":
    main()
