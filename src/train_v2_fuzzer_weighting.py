"""Evaluate Jan 22 Fuzzer-only sample weights for the frozen Context v2 RF.

Every candidate uses the same 128 features and Random Forest hyperparameters.
Only Jan 22 ``sample_weight`` changes. Thresholds are selected on February 17
validation under the existing recall >= 98.5%, then minimum-FPR policy. The
module has no data path for the February 18 locked holdout.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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
except ImportError:  # Direct execution: python src/train_v2_fuzzer_weighting.py
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
CACHE_DIR = FEATURE_DIR / "_fuzzer_weight_cache"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "_v2_fuzzer_weight_checkpoints"

FROZEN_CONTEXT_MODEL_PATH = MODELS_DIR / "v2_context_random_forest.joblib"
FROZEN_ABLATION_METRICS_PATH = MODELS_DIR / "v2_ablation_metrics.json"
FROZEN_VALIDATION_SCORES_PATH = MODELS_DIR / "v2_validation_scores.parquet"

METRICS_PATH = MODELS_DIR / "v2_fuzzer_weight_metrics.json"
THRESHOLD_RESULTS_PATH = MODELS_DIR / "v2_fuzzer_weight_threshold_results.csv"
COMPARISON_PATH = MODELS_DIR / "v2_fuzzer_weight_comparison.csv"
CATEGORY_METRICS_PATH = MODELS_DIR / "v2_fuzzer_weight_category_metrics.csv"
VALIDATION_SCORES_PATH = MODELS_DIR / "v2_fuzzer_weight_validation_scores.parquet"

TRAIN_ROWS = 1_765_922
VALIDATION_ROWS = 498_890
EXPECTED_TRAIN_FUZZERS = 5_627
EXPECTED_VALIDATION_FUZZERS = 6_413
EXPECTED_VALIDATION_ANALYSIS = 52
FUZZER_WEIGHTS = (1.0, 1.25, 1.5, 2.0, 3.0)
CONTEXT_REFERENCE_FPR = 0.02525474166394998
CONTEXT_REFERENCE_FUZZER_FN = 261
CONTEXT_REFERENCE_ANALYSIS_FN = 19
RESEARCH_FPR_TARGET = 0.02
FUZZER_RECALL_IDEAL = 0.97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duckdb-vectors-per-chunk",
        type=int,
        default=100,
        help="DuckDB vectors per streamed pandas chunk (default: 100).",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=list(FUZZER_WEIGHTS),
        help="Frozen Fuzzer weight grid; a complete report requires all five values.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore valid per-weight checkpoints and retrain requested weights.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep float32 feature memory maps after completion.",
    )
    return parser.parse_args()


def weight_slug(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "p")


def model_name(weight: float) -> str:
    return f"context_fuzzer_weight_{weight:.2f}"


def model_path(weight: float) -> Path:
    return MODELS_DIR / f"v2_context_fuzzer_weight_{weight_slug(weight)}.joblib"


def checkpoint_paths(weight: float) -> tuple[Path, Path]:
    slug = weight_slug(weight)
    return (
        CHECKPOINT_DIR / f"weight_{slug}_scores.npy",
        CHECKPOINT_DIR / f"weight_{slug}_metadata.json",
    )


def cache_paths(split: str) -> dict[str, Path]:
    paths = {
        "context": CACHE_DIR / f"{split}_context.float32.mmap",
        "target": CACHE_DIR / f"{split}_target.uint8.mmap",
    }
    if split == "train":
        paths["is_fuzzer"] = CACHE_DIR / "train_is_fuzzer.uint8.mmap"
    return paths


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_numpy_save(values: np.ndarray, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def atomic_joblib_dump(model: RandomForestClassifier, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary, compress=3)
    os.replace(temporary, path)


def stream_context_split(
    path: Path,
    split: str,
    expected_rows: int,
    context_features: list[str],
    vectors_per_chunk: int,
) -> tuple[
    np.memmap,
    np.memmap,
    np.memmap | None,
    np.ndarray | None,
    dict[str, int],
    float,
]:
    """Stream only model features plus targets; identifiers are never selected."""

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
    is_fuzzer = (
        np.memmap(
            paths["is_fuzzer"],
            mode="w+",
            dtype=np.uint8,
            shape=(expected_rows,),
        )
        if split == "train"
        else None
    )
    validation_categories: list[str] | None = [] if split == "validation" else None
    category_counts: Counter[str] = Counter()

    selected = [*context_features, "label", "attack_cat"]
    selected_sql = ", ".join(quote_identifier(column) for column in selected)
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
            if not np.isfinite(values).all():
                raise AssertionError(f"{split}: non-finite model input")
            labels = chunk["label"].to_numpy(dtype=np.uint8)
            categories = chunk["attack_cat"].astype(str)
            matrix[row_offset:end] = values
            target[row_offset:end] = labels
            category_counts.update(categories.tolist())
            if is_fuzzer is not None:
                is_fuzzer[row_offset:end] = (categories == "Fuzzers").to_numpy(
                    dtype=np.uint8
                )
            if validation_categories is not None:
                validation_categories.extend(categories.tolist())
            row_offset = end
            print(f"Loaded {split}: {row_offset:,}/{expected_rows:,}", flush=True)
    finally:
        connection.close()

    if row_offset != expected_rows:
        raise AssertionError(f"{split}: expected {expected_rows:,}, got {row_offset:,}")
    if not np.isin(np.asarray(target), [0, 1]).all():
        raise AssertionError(f"{split}: target contains values outside 0/1")
    if category_counts.get("Benign", 0) != int((np.asarray(target) == 0).sum()):
        raise AssertionError(f"{split}: Benign/category label alignment changed")
    matrix.flush()
    target.flush()
    if is_fuzzer is not None:
        is_fuzzer.flush()
        if int(np.asarray(is_fuzzer).sum()) != EXPECTED_TRAIN_FUZZERS:
            raise AssertionError("Unexpected Jan 22 Fuzzer count")
    validation_array = (
        np.asarray(validation_categories, dtype=object)
        if validation_categories is not None
        else None
    )
    return (
        matrix,
        target,
        is_fuzzer,
        validation_array,
        dict(sorted(category_counts.items())),
        time.perf_counter() - started,
    )


def build_sample_weight(is_fuzzer: np.ndarray, fuzzer_weight: float) -> np.ndarray:
    if fuzzer_weight not in FUZZER_WEIGHTS:
        raise ValueError(f"Weight {fuzzer_weight} is outside the frozen grid")
    mask = np.asarray(is_fuzzer, dtype=bool)
    if int(mask.sum()) != EXPECTED_TRAIN_FUZZERS:
        raise AssertionError("Sample-weight mask has an unexpected Fuzzer count")
    sample_weight = np.ones(mask.shape[0], dtype=np.float32)
    sample_weight[mask] = np.float32(fuzzer_weight)
    if not np.all(sample_weight[~mask] == 1.0):
        raise AssertionError("A non-Fuzzer training row received a custom weight")
    return sample_weight


def checkpoint_fingerprint(
    weight: float, feature_manifest_sha256: str, train_sha256: str, validation_sha256: str
) -> dict[str, Any]:
    return {
        "fuzzer_weight": float(weight),
        "feature_count": 128,
        "train_rows": TRAIN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "random_forest_params": RANDOM_FOREST_PARAMS,
        "feature_manifest_sha256": feature_manifest_sha256,
        "train_parquet_sha256": train_sha256,
        "validation_parquet_sha256": validation_sha256,
        "scikit_learn": sklearn.__version__,
    }


def load_checkpoint(
    weight: float, expected_fingerprint: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any], list[str]] | None:
    score_path, metadata_path = checkpoint_paths(weight)
    fitted_model_path = model_path(weight)
    if not (score_path.is_file() and metadata_path.is_file() and fitted_model_path.is_file()):
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected_fingerprint:
        return None
    if metadata.get("model_sha256") != sha256_file(fitted_model_path):
        return None
    if metadata.get("score_sha256") != sha256_file(score_path):
        return None
    scores = np.load(score_path)
    if scores.shape != (VALIDATION_ROWS,) or not np.isfinite(scores).all():
        return None
    return scores, metadata["continuous_metrics"], metadata.get("warnings", [])


def fit_and_score_weight(
    weight: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_is_fuzzer: np.ndarray,
    x_validation: np.ndarray,
    fingerprint: dict[str, Any],
    restart: bool,
) -> tuple[np.ndarray, dict[str, Any], list[str], bool]:
    if not restart:
        checkpoint = load_checkpoint(weight, fingerprint)
        if checkpoint is not None:
            scores, continuous, captured_warnings = checkpoint
            print(f"Resumed verified checkpoint for Fuzzer weight {weight:.2f}", flush=True)
            return scores, continuous, captured_warnings, True

    name = model_name(weight)
    fitted_model_path = model_path(weight)
    score_path, metadata_path = checkpoint_paths(weight)
    sample_weight = build_sample_weight(train_is_fuzzer, weight)
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        started = time.perf_counter()
        model.fit(x_train, y_train, sample_weight=sample_weight)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        scores = model.predict_proba(x_validation)[:, 1]
        inference_seconds = time.perf_counter() - started
        captured_warnings = [str(record.message) for record in warning_records]
    del sample_weight

    if scores.shape != (VALIDATION_ROWS,) or not np.isfinite(scores).all():
        raise AssertionError(f"{name}: invalid validation score vector")
    atomic_joblib_dump(model, fitted_model_path)
    reloaded = joblib.load(fitted_model_path)
    check_rows = min(10_000, VALIDATION_ROWS)
    reloaded_scores = reloaded.predict_proba(x_validation[:check_rows])[:, 1]
    max_difference = float(np.max(np.abs(scores[:check_rows] - reloaded_scores)))
    if max_difference > 1e-12:
        raise AssertionError(f"{name}: saved model did not reproduce validation scores")

    continuous = {
        "roc_auc": None,
        "pr_auc_average_precision": None,
        "training_seconds": training_seconds,
        "validation_inference_seconds": inference_seconds,
        "model_path": str(fitted_model_path.relative_to(PROJECT_ROOT)),
        "model_size_bytes": fitted_model_path.stat().st_size,
        "model_sha256": sha256_file(fitted_model_path),
        "reload_check_rows": check_rows,
        "reload_max_score_difference": max_difference,
    }
    atomic_numpy_save(scores, score_path)
    checkpoint_metadata = {
        "fingerprint": fingerprint,
        "continuous_metrics": continuous,
        "warnings": captured_warnings,
        "model_sha256": continuous["model_sha256"],
        "score_sha256": sha256_file(score_path),
    }
    atomic_json_dump(checkpoint_metadata, metadata_path)
    del reloaded, model
    gc.collect()
    return scores, continuous, captured_warnings, False


def write_validation_scores(
    categories: np.ndarray,
    y_true: np.ndarray,
    scores_by_weight: dict[float, np.ndarray],
    selected_by_weight: dict[float, dict[str, Any]],
) -> None:
    frame = pd.DataFrame(
        {
            "validation_row": np.arange(VALIDATION_ROWS, dtype=np.int64),
            "attack_cat": categories,
            "label": y_true.astype(np.uint8),
        }
    )
    for weight, scores in scores_by_weight.items():
        slug = weight_slug(weight)
        frame[f"weight_{slug}_attack_score"] = scores
        frame[f"weight_{slug}_selected_prediction"] = (
            scores >= float(selected_by_weight[weight]["threshold"])
        ).astype(np.uint8)
    connection = duckdb.connect()
    temporary = VALIDATION_SCORES_PATH.with_suffix(".tmp.parquet")
    try:
        if temporary.exists():
            temporary.unlink()
        connection.register("weighted_scores", frame)
        connection.execute(
            f"""
            COPY weighted_scores TO {sql_string(temporary)}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
        )
        os.replace(temporary, VALIDATION_SCORES_PATH)
    finally:
        connection.close()


def reference_score_difference(scores: np.ndarray) -> float:
    if not FROZEN_VALIDATION_SCORES_PATH.is_file():
        raise FileNotFoundError(FROZEN_VALIDATION_SCORES_PATH)
    connection = duckdb.connect()
    try:
        reference = connection.execute(
            f"SELECT context_attack_score FROM read_parquet("
            f"{sql_string(FROZEN_VALIDATION_SCORES_PATH)}) ORDER BY validation_row"
        ).fetchnumpy()["context_attack_score"]
    finally:
        connection.close()
    if reference.shape != scores.shape:
        raise AssertionError("Frozen Context score shape changed")
    return float(np.max(np.abs(reference - scores)))


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
    requested_weights = tuple(float(weight) for weight in args.weights)
    if len(set(requested_weights)) != len(requested_weights):
        raise ValueError("Weights must not be duplicated")
    if set(requested_weights) != set(FUZZER_WEIGHTS):
        raise ValueError(
            "A complete report requires exactly the frozen grid: "
            f"{list(FUZZER_WEIGHTS)}"
        )
    for required in (
        TRAIN_PATH,
        VALIDATION_PATH,
        FEATURE_MANIFEST_PATH,
        FROZEN_CONTEXT_MODEL_PATH,
        FROZEN_ABLATION_METRICS_PATH,
        FROZEN_VALIDATION_SCORES_PATH,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    _, context_features, feature_manifest = load_feature_contract()
    if len(context_features) != 128:
        raise AssertionError("Context v2 feature count changed")
    excluded = set(feature_manifest["feature_contract"]["excluded_raw_identifiers"])
    if excluded.intersection(context_features) or {"attack_cat", "label"}.intersection(
        context_features
    ):
        raise AssertionError("An identifier or target leaked into Context features")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    overall_started = time.perf_counter()
    frozen_hashes_before = {
        "context_model": sha256_file(FROZEN_CONTEXT_MODEL_PATH),
        "ablation_metrics": sha256_file(FROZEN_ABLATION_METRICS_PATH),
        "validation_scores": sha256_file(FROZEN_VALIDATION_SCORES_PATH),
    }

    print("Streaming Jan 22 Context features and Fuzzer weight mask...", flush=True)
    (
        x_train,
        y_train_memmap,
        train_is_fuzzer,
        _,
        train_categories,
        train_load_seconds,
    ) = stream_context_split(
        TRAIN_PATH,
        "train",
        TRAIN_ROWS,
        context_features,
        args.duckdb_vectors_per_chunk,
    )
    if train_is_fuzzer is None:
        raise AssertionError("Jan 22 Fuzzer mask was not built")
    print("Streaming Feb 17 validation features...", flush=True)
    (
        x_validation,
        y_validation_memmap,
        _,
        validation_categories,
        validation_category_counts,
        validation_load_seconds,
    ) = stream_context_split(
        VALIDATION_PATH,
        "validation",
        VALIDATION_ROWS,
        context_features,
        args.duckdb_vectors_per_chunk,
    )
    if validation_categories is None:
        raise AssertionError("Feb 17 categories were not loaded")
    if validation_category_counts.get("Fuzzers") != EXPECTED_VALIDATION_FUZZERS:
        raise AssertionError("Unexpected Feb 17 Fuzzer count")
    if validation_category_counts.get("Analysis") != EXPECTED_VALIDATION_ANALYSIS:
        raise AssertionError("Unexpected Feb 17 Analysis count")

    y_train = np.asarray(y_train_memmap)
    y_validation = np.asarray(y_validation_memmap)
    feature_manifest_hash = sha256_file(FEATURE_MANIFEST_PATH)
    train_hash = feature_manifest["outputs"]["train"]["sha256"]
    validation_hash = feature_manifest["outputs"]["validation"]["sha256"]
    threshold_frames: list[pd.DataFrame] = []
    category_frames: list[pd.DataFrame] = []
    results: dict[float, dict[str, Any]] = {}
    scores_by_weight: dict[float, np.ndarray] = {}
    selected_by_weight: dict[float, dict[str, Any]] = {}

    for weight in requested_weights:
        print(f"Training Context RF with Fuzzer weight {weight:.2f}...", flush=True)
        fingerprint = checkpoint_fingerprint(
            weight, feature_manifest_hash, train_hash, validation_hash
        )
        scores, continuous, captured_warnings, resumed = fit_and_score_weight(
            weight,
            x_train,
            y_train,
            train_is_fuzzer,
            x_validation,
            fingerprint,
            args.restart,
        )
        continuous["roc_auc"] = float(roc_auc_score(y_validation, scores))
        continuous["pr_auc_average_precision"] = float(
            average_precision_score(y_validation, scores)
        )
        sweep, selected, gate_met = sweep_thresholds(
            model_name(weight), y_validation, scores
        )
        categories = category_metrics(
            model_name(weight),
            validation_categories,
            y_validation,
            scores,
            selected["threshold"],
        )
        fuzzer_row = categories[categories["attack_cat"] == "Fuzzers"].iloc[0]
        analysis_row = categories[categories["attack_cat"] == "Analysis"].iloc[0]
        fuzzer_fn = int(fuzzer_row["missed_attacks"])
        analysis_fn = int(analysis_row["missed_attacks"])
        results[weight] = {
            "fuzzer_weight": weight,
            "feature_count": len(context_features),
            "recall_gate_achieved": bool(gate_met),
            "continuous_metrics": continuous,
            "selected_operating_point": selected,
            "threshold_0_50": threshold_metrics(y_validation, scores, 0.50),
            "fuzzer_validation": {
                "total": int(fuzzer_row["total"]),
                "detected": int(fuzzer_row["detected_attacks"]),
                "false_negatives": fuzzer_fn,
                "recall": float(fuzzer_row["attack_recall"]),
            },
            "analysis_validation": {
                "total": int(analysis_row["total"]),
                "detected": int(analysis_row["detected_attacks"]),
                "false_negatives": analysis_fn,
                "recall": float(analysis_row["attack_recall"]),
            },
            "comparison_to_unweighted_context": {
                "fpr_difference": float(selected["fpr"] - CONTEXT_REFERENCE_FPR),
                "false_positive_difference": int(
                    selected["false_positives"]
                    - round(CONTEXT_REFERENCE_FPR * validation_category_counts["Benign"])
                ),
                "fuzzer_fn_difference": fuzzer_fn - CONTEXT_REFERENCE_FUZZER_FN,
                "analysis_fn_difference": analysis_fn - CONTEXT_REFERENCE_ANALYSIS_FN,
            },
            "success_checks": {
                "recall_at_least_0_985": bool(selected["recall"] >= RECALL_GATE),
                "fpr_better_than_unweighted_context": bool(
                    selected["fpr"] < CONTEXT_REFERENCE_FPR
                ),
                "fuzzer_fn_lower_than_261": bool(
                    fuzzer_fn < CONTEXT_REFERENCE_FUZZER_FN
                ),
                "ideal_fpr_below_0_02": bool(selected["fpr"] < RESEARCH_FPR_TARGET),
                "ideal_fuzzer_recall_at_least_0_97": bool(
                    float(fuzzer_row["attack_recall"]) >= FUZZER_RECALL_IDEAL
                ),
            },
            "sample_weight_counts": {
                "fuzzer_rows": EXPECTED_TRAIN_FUZZERS,
                "fuzzer_weight": weight,
                "all_other_rows": TRAIN_ROWS - EXPECTED_TRAIN_FUZZERS,
                "all_other_weight": 1.0,
            },
            "warnings": captured_warnings,
            "resumed_from_checkpoint": resumed,
        }
        threshold_frames.append(sweep.assign(fuzzer_weight=weight))
        category_frames.append(categories.assign(fuzzer_weight=weight))
        scores_by_weight[weight] = scores
        selected_by_weight[weight] = selected
        print(
            json.dumps(
                {
                    "fuzzer_weight": weight,
                    "threshold": selected["threshold"],
                    "recall": selected["recall"],
                    "fpr": selected["fpr"],
                    "false_positives": selected["false_positives"],
                    "overall_fn": selected["false_negatives"],
                    "fuzzer_fn": fuzzer_fn,
                    "analysis_fn": analysis_fn,
                    "pr_auc": continuous["pr_auc_average_precision"],
                },
                indent=2,
            ),
            flush=True,
        )

    threshold_results = pd.concat(threshold_frames, ignore_index=True)
    all_categories = pd.concat(category_frames, ignore_index=True)
    comparison_rows: list[dict[str, Any]] = []
    for weight in requested_weights:
        result = results[weight]
        selected = result["selected_operating_point"]
        comparison_rows.append(
            {
                "fuzzer_weight": weight,
                "threshold": selected["threshold"],
                "recall": selected["recall"],
                "precision": selected["precision"],
                "f1": selected["f1"],
                "roc_auc": result["continuous_metrics"]["roc_auc"],
                "pr_auc_average_precision": result["continuous_metrics"][
                    "pr_auc_average_precision"
                ],
                "fpr": selected["fpr"],
                "fnr": selected["fnr"],
                "false_positives": selected["false_positives"],
                "overall_false_negatives": selected["false_negatives"],
                "fuzzer_false_negatives": result["fuzzer_validation"][
                    "false_negatives"
                ],
                "fuzzer_recall": result["fuzzer_validation"]["recall"],
                "analysis_false_negatives": result["analysis_validation"][
                    "false_negatives"
                ],
                "analysis_recall": result["analysis_validation"]["recall"],
                "fp_per_100k_benign": selected["fp_per_100k_benign"],
                "training_seconds": result["continuous_metrics"]["training_seconds"],
                "validation_inference_seconds": result["continuous_metrics"][
                    "validation_inference_seconds"
                ],
                "recall_gate_achieved": result["recall_gate_achieved"],
                "fpr_better_than_context": result["success_checks"][
                    "fpr_better_than_unweighted_context"
                ],
                "fuzzer_fn_reduction_vs_context": (
                    CONTEXT_REFERENCE_FUZZER_FN
                    - result["fuzzer_validation"]["false_negatives"]
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values("fuzzer_weight")
    eligible = comparison[comparison["recall_gate_achieved"]]
    if eligible.empty:
        best_row = comparison.sort_values(
            ["recall", "fpr"], ascending=[False, True]
        ).iloc[0]
    else:
        best_row = eligible.sort_values(
            ["false_positives", "fuzzer_false_negatives", "pr_auc_average_precision"],
            ascending=[True, True, False],
        ).iloc[0]
    best_weight = float(best_row["fuzzer_weight"])

    threshold_results.to_csv(THRESHOLD_RESULTS_PATH, index=False)
    all_categories.to_csv(CATEGORY_METRICS_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)
    write_validation_scores(
        validation_categories, y_validation, scores_by_weight, selected_by_weight
    )
    weight_one_score_difference = reference_score_difference(scores_by_weight[1.0])

    frozen_hashes_after = {
        "context_model": sha256_file(FROZEN_CONTEXT_MODEL_PATH),
        "ablation_metrics": sha256_file(FROZEN_ABLATION_METRICS_PATH),
        "validation_scores": sha256_file(FROZEN_VALIDATION_SCORES_PATH),
    }
    if frozen_hashes_before != frozen_hashes_after:
        raise AssertionError("A frozen v2 artifact changed during weighting")

    metrics = {
        "stage": "v2 Context Fuzzer-only sample-weight experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "train_capture_date": "2015-01-22",
            "validation_capture_date": "2015-02-17",
            "locked_holdout_loaded": False,
            "locked_holdout_evaluated": False,
            "stacking_trained": False,
            "model_features_changed": False,
            "model_hyperparameters_changed": False,
            "only_training_change": "Jan 22 Fuzzers sample_weight",
        },
        "selection_policy": {
            "threshold_grid": "0.01 through 0.99, step 0.01",
            "recall_gate": RECALL_GATE,
            "objective_after_gate": "minimum FPR / false positives",
            "tie_breakers": ["higher precision", "higher F1", "higher threshold"],
        },
        "fuzzer_weight_grid": list(FUZZER_WEIGHTS),
        "random_forest_params": RANDOM_FOREST_PARAMS,
        "data": {
            "train_rows": TRAIN_ROWS,
            "validation_rows": VALIDATION_ROWS,
            "feature_count": len(context_features),
            "train_category_counts": train_categories,
            "validation_category_counts": validation_category_counts,
            "feature_manifest_sha256": feature_manifest_hash,
            "train_parquet_sha256": train_hash,
            "validation_parquet_sha256": validation_hash,
        },
        "reference_context": {
            "fpr": CONTEXT_REFERENCE_FPR,
            "false_positives": 12_085,
            "fuzzer_false_negatives": CONTEXT_REFERENCE_FUZZER_FN,
            "analysis_false_negatives": CONTEXT_REFERENCE_ANALYSIS_FN,
            "weight_1_score_max_absolute_difference": weight_one_score_difference,
            "all_ones_sample_weight_bootstrap_explanation": {
                "verified_scikit_learn_version": sklearn.__version__,
                "sample_weight_none": (
                    "RandomForest bootstrap indices are generated with randint."
                ),
                "sample_weight_provided": (
                    "RandomForest bootstrap indices are generated with choice using "
                    "normalized sample_weight probabilities, including an all-ones "
                    "weight vector."
                ),
                "consequence": (
                    "The same random_state does not require the all-ones weighted "
                    "model to match the historical sample_weight=None model. "
                    "Reproducibility is required within each specific training mode."
                ),
            },
        },
        "variants": {f"{weight:.2f}": results[weight] for weight in requested_weights},
        "selected_by_fixed_policy": json.loads(
            comparison[comparison["fuzzer_weight"] == best_weight]
            .iloc[0]
            .to_json()
        ),
        "runtime": {
            "train_data_load_seconds": train_load_seconds,
            "validation_data_load_seconds": validation_load_seconds,
            "total_seconds": time.perf_counter() - overall_started,
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
            "models": {
                f"{weight:.2f}": str(model_path(weight).relative_to(PROJECT_ROOT))
                for weight in requested_weights
            },
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
            y_train,
            y_train_memmap,
            y_validation,
            y_validation_memmap,
            train_is_fuzzer,
        )
        gc.collect()
        clean_cache()

    print(comparison.to_string(index=False), flush=True)
    print(f"Selected by fixed policy: Fuzzer weight {best_weight:.2f}", flush=True)
    print(f"Metrics: {METRICS_PATH}", flush=True)
    print("No stacking was trained. Feb 18 was not loaded or evaluated.", flush=True)


if __name__ == "__main__":
    main()
