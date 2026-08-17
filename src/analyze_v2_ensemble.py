"""Analyze v2 validation error overlap and score-level ensembles.

This stage reads only saved February 17 validation features and scores. It does
not fit a model and has no data path for the February 18 locked holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .train_v2_ablation import (
        RECALL_GATE,
        category_metrics,
        sweep_thresholds,
    )
except ImportError:  # Direct execution: python src/analyze_v2_ensemble.py
    from train_v2_ablation import (
        RECALL_GATE,
        category_metrics,
        sweep_thresholds,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "processed" / "cic_unsw_nb15_v2"
VALIDATION_FEATURES_PATH = FEATURE_DIR / "validation_features.parquet"
FEATURE_MANIFEST_PATH = FEATURE_DIR / "feature_manifest.json"
MODELS_DIR = PROJECT_ROOT / "models"
VALIDATION_SCORES_PATH = MODELS_DIR / "v2_validation_scores.parquet"
ABLATION_METRICS_PATH = MODELS_DIR / "v2_ablation_metrics.json"
CONTEXT_MODEL_PATH = MODELS_DIR / "v2_context_random_forest.joblib"

OVERLAP_SUMMARY_PATH = MODELS_DIR / "v2_error_overlap.json"
OVERLAP_ROWS_PATH = MODELS_DIR / "v2_error_overlap_rows.parquet"
ENSEMBLE_THRESHOLDS_PATH = MODELS_DIR / "v2_ensemble_threshold_results.csv"
ENSEMBLE_COMPARISON_PATH = MODELS_DIR / "v2_ensemble_comparison.csv"
ENSEMBLE_CATEGORY_PATH = MODELS_DIR / "v2_ensemble_attack_category_metrics.csv"
ENSEMBLE_SCORES_PATH = MODELS_DIR / "v2_ensemble_validation_scores.parquet"
SELECTED_ENSEMBLE_PATH = MODELS_DIR / "v2_selected_ensemble.json"
FUZZER_REGRESSIONS_PATH = MODELS_DIR / "v2_fuzzer_context_regressions.parquet"
FUZZER_DIAGNOSTICS_PATH = MODELS_DIR / "v2_fuzzer_context_feature_diagnostics.csv"

EXPECTED_VALIDATION_ROWS = 498_890
ALPHAS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
FPR_RESEARCH_TARGET = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-context-importance",
        action="store_true",
        help="Skip loading the saved context RF for global feature importances.",
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


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
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


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    for path in (
        VALIDATION_SCORES_PATH,
        ABLATION_METRICS_PATH,
        FEATURE_MANIFEST_PATH,
        VALIDATION_FEATURES_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    connection = duckdb.connect()
    try:
        scores = connection.execute(
            f"SELECT * FROM read_parquet({sql_string(VALIDATION_SCORES_PATH)})"
        ).fetchdf()
    finally:
        connection.close()
    if len(scores) != EXPECTED_VALIDATION_ROWS:
        raise AssertionError("Validation score row count changed")
    if scores["validation_row"].tolist() != list(range(EXPECTED_VALIDATION_ROWS)):
        raise AssertionError("Validation scores are not in frozen row order")
    for column in ("baseline_attack_score", "context_attack_score"):
        if not np.isfinite(scores[column]).all():
            raise AssertionError(f"Non-finite scores in {column}")
    metrics = json.loads(ABLATION_METRICS_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if metrics["scope"]["locked_holdout_loaded"]:
        raise AssertionError("Ablation metrics unexpectedly report holdout access")
    return scores, metrics, manifest


def add_overlap_labels(
    scores: pd.DataFrame,
    baseline_threshold: float,
    context_threshold: float,
) -> pd.DataFrame:
    result = scores.copy()
    baseline_prediction = result["baseline_attack_score"].to_numpy() >= baseline_threshold
    context_prediction = result["context_attack_score"].to_numpy() >= context_threshold
    labels = result["label"].to_numpy(dtype=np.uint8)
    attacks = labels == 1
    benign = ~attacks

    attack_overlap = np.full(len(result), "not_attack", dtype=object)
    attack_overlap[attacks & baseline_prediction & context_prediction] = "both_catch"
    attack_overlap[attacks & baseline_prediction & ~context_prediction] = (
        "baseline_catch_context_miss"
    )
    attack_overlap[attacks & ~baseline_prediction & context_prediction] = (
        "baseline_miss_context_catch"
    )
    attack_overlap[attacks & ~baseline_prediction & ~context_prediction] = "both_miss"

    benign_overlap = np.full(len(result), "not_benign", dtype=object)
    benign_overlap[benign & baseline_prediction & context_prediction] = "both_false_positive"
    benign_overlap[benign & baseline_prediction & ~context_prediction] = (
        "baseline_only_false_positive"
    )
    benign_overlap[benign & ~baseline_prediction & context_prediction] = (
        "context_only_false_positive"
    )
    benign_overlap[benign & ~baseline_prediction & ~context_prediction] = (
        "both_true_negative"
    )

    result["baseline_selected_prediction"] = baseline_prediction.astype(np.uint8)
    result["context_selected_prediction"] = context_prediction.astype(np.uint8)
    result["score_delta_context_minus_baseline"] = (
        result["context_attack_score"] - result["baseline_attack_score"]
    )
    result["attack_overlap"] = attack_overlap
    result["benign_overlap"] = benign_overlap
    return result


def overlap_summary(overlap: pd.DataFrame) -> dict[str, Any]:
    attacks = overlap[overlap["label"] == 1]
    benign = overlap[overlap["label"] == 0]
    attack_counts = attacks["attack_overlap"].value_counts().to_dict()
    benign_counts = benign["benign_overlap"].value_counts().to_dict()
    category_table = (
        attacks.groupby(["attack_cat", "attack_overlap"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    for column in (
        "both_catch",
        "baseline_catch_context_miss",
        "baseline_miss_context_catch",
        "both_miss",
    ):
        if column not in category_table:
            category_table[column] = 0
    category_table = category_table[
        [
            "both_catch",
            "baseline_catch_context_miss",
            "baseline_miss_context_catch",
            "both_miss",
        ]
    ]
    fuzzer = category_table.loc["Fuzzers"].to_dict()
    context_fuzzer_misses = fuzzer["baseline_catch_context_miss"] + fuzzer["both_miss"]
    if context_fuzzer_misses != 261:
        raise AssertionError(
            f"Expected 261 context Fuzzer misses, found {context_fuzzer_misses}"
        )
    return {
        "attack_overlap": {key: int(value) for key, value in attack_counts.items()},
        "benign_overlap": {key: int(value) for key, value in benign_counts.items()},
        "attack_overlap_by_category": {
            category: {key: int(value) for key, value in row.items()}
            for category, row in category_table.to_dict(orient="index").items()
        },
        "fuzzer_focus": {
            **{key: int(value) for key, value in fuzzer.items()},
            "context_fuzzer_misses": int(context_fuzzer_misses),
            "context_fuzzer_misses_caught_by_baseline": int(
                fuzzer["baseline_catch_context_miss"]
            ),
            "share_of_context_fuzzer_misses_caught_by_baseline": float(
                fuzzer["baseline_catch_context_miss"] / context_fuzzer_misses
            ),
        },
    }


def evaluate_ensembles(
    overlap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    labels = overlap["label"].to_numpy(dtype=np.uint8)
    baseline = overlap["baseline_attack_score"].to_numpy(dtype=float)
    context = overlap["context_attack_score"].to_numpy(dtype=float)
    sweep_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    candidate_scores: dict[str, np.ndarray] = {}

    for alpha in ALPHAS:
        name = f"blend_alpha_{alpha:.2f}"
        scores = alpha * context + (1.0 - alpha) * baseline
        sweep, selected, gate_met = sweep_thresholds(name, labels, scores)
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
        sweep_frames.append(sweep)
        summaries.append(
            {
                "candidate": name,
                "method": "weighted_blend",
                "alpha_context": float(alpha),
                "alpha_baseline": float(1.0 - alpha),
                "recall_gate_achieved": bool(gate_met),
                "roc_auc": roc_auc,
                "pr_auc_average_precision": pr_auc,
                **selected,
            }
        )
        candidate_scores[name] = scores

    max_scores = np.maximum(baseline, context)
    max_name = "max_score"
    max_sweep, max_selected, max_gate = sweep_thresholds(max_name, labels, max_scores)
    sweep_frames.append(max_sweep)
    summaries.append(
        {
            "candidate": max_name,
            "method": "maximum",
            "alpha_context": np.nan,
            "alpha_baseline": np.nan,
            "recall_gate_achieved": bool(max_gate),
            "roc_auc": float(roc_auc_score(labels, max_scores)),
            "pr_auc_average_precision": float(
                average_precision_score(labels, max_scores)
            ),
            **max_selected,
        }
    )
    candidate_scores[max_name] = max_scores

    thresholds = pd.concat(sweep_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    eligible = summary[summary["recall_gate_achieved"]]
    if eligible.empty:
        raise AssertionError("No ensemble candidate met the already-achieved recall gate")
    best_row = eligible.sort_values(
        by=[
            "false_positives",
            "pr_auc_average_precision",
            "precision",
            "f1",
            "threshold",
        ],
        ascending=[True, False, False, False, False],
    ).iloc[0]
    best = {
        key: (
            bool(best_row[key])
            if key == "recall_gate_achieved"
            else int(best_row[key])
            if key
            in {
                "false_positives",
                "false_negatives",
                "true_positives",
                "true_negatives",
            }
            else None
            if pd.isna(best_row[key])
            else float(best_row[key])
            if isinstance(best_row[key], (float, np.floating))
            else best_row[key]
        )
        for key in summary.columns
    }
    return thresholds, summary, best, candidate_scores


def load_fuzzer_features(
    added_features: list[str],
    overlap: pd.DataFrame,
) -> pd.DataFrame:
    feature_sql = ", ".join(quote_identifier(feature) for feature in added_features)
    connection = duckdb.connect()
    try:
        fuzzers = connection.execute(
            f"""
            WITH ordered AS (
                SELECT
                    row_number() OVER () - 1 AS validation_row,
                    attack_cat,
                    label,
                    {feature_sql}
                FROM read_parquet({sql_string(VALIDATION_FEATURES_PATH)})
            )
            SELECT * FROM ordered WHERE attack_cat = 'Fuzzers'
            ORDER BY validation_row
            """
        ).fetchdf()
    finally:
        connection.close()
    score_columns = [
        "validation_row",
        "attack_cat",
        "baseline_attack_score",
        "context_attack_score",
        "score_delta_context_minus_baseline",
        "baseline_selected_prediction",
        "context_selected_prediction",
        "attack_overlap",
    ]
    joined = fuzzers.merge(
        overlap[score_columns],
        on="validation_row",
        how="left",
        suffixes=("", "_score"),
    )
    if len(joined) != 6_413 or joined["attack_overlap"].isna().any():
        raise AssertionError("Fuzzer feature/score alignment failed")
    if not (joined["attack_cat"] == joined["attack_cat_score"]).all():
        raise AssertionError("Fuzzer row order does not align with saved validation scores")
    joined = joined.drop(columns=["attack_cat_score"])
    return joined


def fuzzer_feature_diagnostics(
    fuzzers: pd.DataFrame,
    added_features: list[str],
    context_importances: dict[str, float],
) -> pd.DataFrame:
    regressions = fuzzers[
        fuzzers["attack_overlap"] == "baseline_catch_context_miss"
    ]
    both_catch = fuzzers[fuzzers["attack_overlap"] == "both_catch"]
    recoveries = fuzzers[
        fuzzers["attack_overlap"] == "baseline_miss_context_catch"
    ]
    rows: list[dict[str, Any]] = []
    for feature in added_features:
        regression_values = regressions[feature].to_numpy(dtype=float)
        catch_values = both_catch[feature].to_numpy(dtype=float)
        recovery_values = recoveries[feature].to_numpy(dtype=float)
        combined_variance = (
            (regression_values.var(ddof=1) + catch_values.var(ddof=1)) / 2
            if len(regression_values) > 1 and len(catch_values) > 1
            else 0.0
        )
        pooled_std = float(np.sqrt(combined_variance))
        standardized_difference = (
            float((regression_values.mean() - catch_values.mean()) / pooled_std)
            if pooled_std > 0
            else 0.0
        )
        ks = ks_2samp(regression_values, catch_values, method="auto")
        all_feature_values = fuzzers[feature].to_numpy(dtype=float)
        all_score_deltas = fuzzers[
            "score_delta_context_minus_baseline"
        ].to_numpy(dtype=float)
        if np.unique(all_feature_values).size <= 1:
            spearman_statistic = 0.0
            spearman_pvalue = 1.0
        else:
            spearman = spearmanr(
                all_feature_values,
                all_score_deltas,
                nan_policy="omit",
            )
            spearman_statistic = (
                float(spearman.statistic)
                if np.isfinite(spearman.statistic)
                else 0.0
            )
            spearman_pvalue = (
                float(spearman.pvalue) if np.isfinite(spearman.pvalue) else 1.0
            )
        rows.append(
            {
                "feature": feature,
                "context_regression_count": len(regression_values),
                "both_catch_count": len(catch_values),
                "context_recovery_count": len(recovery_values),
                "regression_mean": float(regression_values.mean()),
                "both_catch_mean": float(catch_values.mean()),
                "mean_difference": float(
                    regression_values.mean() - catch_values.mean()
                ),
                "regression_median": float(np.median(regression_values)),
                "both_catch_median": float(np.median(catch_values)),
                "median_difference": float(
                    np.median(regression_values) - np.median(catch_values)
                ),
                "recovery_median": (
                    float(np.median(recovery_values))
                    if len(recovery_values)
                    else np.nan
                ),
                "standardized_mean_difference": standardized_difference,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "spearman_with_context_minus_baseline_score": spearman_statistic,
                "spearman_pvalue": spearman_pvalue,
                "context_rf_global_importance": context_importances.get(feature, np.nan),
            }
        )
    frame = pd.DataFrame(rows)
    frame["absolute_standardized_mean_difference"] = frame[
        "standardized_mean_difference"
    ].abs()
    return frame.sort_values(
        by=["ks_statistic", "absolute_standardized_mean_difference"],
        ascending=[False, False],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    scores, metrics, manifest = load_inputs()
    baseline_threshold = metrics["models"]["baseline_v2"][
        "selected_operating_point"
    ]["threshold"]
    context_threshold = metrics["models"]["context_v2"][
        "selected_operating_point"
    ]["threshold"]
    overlap = add_overlap_labels(scores, baseline_threshold, context_threshold)
    overlap_details = overlap_summary(overlap)
    write_parquet(overlap, OVERLAP_ROWS_PATH)

    thresholds, ensemble_summary, best, candidate_scores = evaluate_ensembles(overlap)
    thresholds.to_csv(ENSEMBLE_THRESHOLDS_PATH, index=False)
    ensemble_summary.to_csv(ENSEMBLE_COMPARISON_PATH, index=False)
    best_name = str(best["candidate"])
    best_scores = candidate_scores[best_name]
    max_scores = candidate_scores["max_score"]
    labels = overlap["label"].to_numpy(dtype=np.uint8)
    categories = overlap["attack_cat"].to_numpy(dtype=object)

    best_categories = category_metrics(
        best_name, categories, labels, best_scores, float(best["threshold"])
    )
    max_row = ensemble_summary[ensemble_summary["candidate"] == "max_score"].iloc[0]
    max_categories = category_metrics(
        "max_score", categories, labels, max_scores, float(max_row["threshold"])
    )
    pd.concat([best_categories, max_categories], ignore_index=True).to_csv(
        ENSEMBLE_CATEGORY_PATH, index=False
    )

    ensemble_scores = overlap[
        ["validation_row", "attack_cat", "label"]
    ].copy()
    ensemble_scores["selected_ensemble_score"] = best_scores
    ensemble_scores["selected_ensemble_prediction"] = (
        best_scores >= float(best["threshold"])
    ).astype(np.uint8)
    ensemble_scores["max_score"] = max_scores
    ensemble_scores["max_score_prediction"] = (
        max_scores >= float(max_row["threshold"])
    ).astype(np.uint8)
    write_parquet(ensemble_scores, ENSEMBLE_SCORES_PATH)

    contract = manifest["feature_contract"]
    added_features = [
        *contract["context_v2_static_behavioral_features"],
        *contract["context_v2_temporal_features"],
    ]
    if len(added_features) != 52:
        raise AssertionError(f"Expected 52 added features, found {len(added_features)}")
    context_importances: dict[str, float] = {}
    if not args.skip_context_importance:
        model = joblib.load(CONTEXT_MODEL_PATH)
        context_features = [*contract["baseline_v2_features"], *added_features]
        if model.n_features_in_ != len(context_features):
            raise AssertionError("Context RF feature count does not match manifest")
        context_importances = dict(
            zip(context_features, model.feature_importances_, strict=True)
        )

    fuzzers = load_fuzzer_features(added_features, overlap)
    regression_fuzzers = fuzzers[
        fuzzers["attack_overlap"] == "baseline_catch_context_miss"
    ].copy()
    write_parquet(regression_fuzzers, FUZZER_REGRESSIONS_PATH)
    diagnostics = fuzzer_feature_diagnostics(
        fuzzers, added_features, context_importances
    )
    diagnostics.to_csv(FUZZER_DIAGNOSTICS_PATH, index=False)

    baseline_selected = metrics["models"]["baseline_v2"]["selected_operating_point"]
    context_selected = metrics["models"]["context_v2"]["selected_operating_point"]
    best_fpr = float(best["fpr"])
    best_recall = float(best["recall"])
    best_fp = int(best["false_positives"])
    best_fn = int(best["false_negatives"])
    selected = {
        "stage": "v2 validation error overlap and score ensemble",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "validation_capture_date": "2015-02-17",
            "model_retrained": False,
            "locked_holdout_loaded": False,
            "locked_holdout_evaluated": False,
        },
        "overlap": overlap_details,
        "ensemble_search": {
            "weighted_formula": (
                "alpha * context_score + (1 - alpha) * baseline_score"
            ),
            "alphas": [float(alpha) for alpha in ALPHAS],
            "additional_candidate": "max(baseline_score, context_score)",
            "threshold_policy": (
                "0.01-0.99 step 0.01; recall >= 0.985 then minimum FPR"
            ),
            "selected": best,
            "research_target_fpr_below_0_02_achieved": bool(
                best_recall >= RECALL_GATE and best_fpr < FPR_RESEARCH_TARGET
            ),
        },
        "comparison": {
            "baseline": baseline_selected,
            "context": context_selected,
            "selected_ensemble": {
                "recall": best_recall,
                "fpr": best_fpr,
                "false_positives": best_fp,
                "false_negatives": best_fn,
                "fp_reduction_vs_context": int(
                    context_selected["false_positives"] - best_fp
                ),
                "fn_change_vs_context": int(
                    best_fn - context_selected["false_negatives"]
                ),
            },
        },
        "fuzzer_regression": {
            "row_count": len(regression_fuzzers),
            "top_10_shifted_added_features": diagnostics.head(10)[
                [
                    "feature",
                    "ks_statistic",
                    "standardized_mean_difference",
                    "median_difference",
                    "spearman_with_context_minus_baseline_score",
                    "context_rf_global_importance",
                ]
            ].to_dict(orient="records"),
            "interpretation_limit": (
                "These are distribution/correlation diagnostics, not causal SHAP "
                "attributions."
            ),
        },
        "input_hashes": {
            "validation_scores_sha256": sha256_file(VALIDATION_SCORES_PATH),
            "validation_features_sha256": manifest["outputs"]["validation"][
                "sha256"
            ],
        },
        "artifacts": {
            "overlap_rows": str(OVERLAP_ROWS_PATH.relative_to(PROJECT_ROOT)),
            "ensemble_thresholds": str(
                ENSEMBLE_THRESHOLDS_PATH.relative_to(PROJECT_ROOT)
            ),
            "ensemble_comparison": str(
                ENSEMBLE_COMPARISON_PATH.relative_to(PROJECT_ROOT)
            ),
            "ensemble_category_metrics": str(
                ENSEMBLE_CATEGORY_PATH.relative_to(PROJECT_ROOT)
            ),
            "ensemble_scores": str(ENSEMBLE_SCORES_PATH.relative_to(PROJECT_ROOT)),
            "fuzzer_regression_rows": str(
                FUZZER_REGRESSIONS_PATH.relative_to(PROJECT_ROOT)
            ),
            "fuzzer_feature_diagnostics": str(
                FUZZER_DIAGNOSTICS_PATH.relative_to(PROJECT_ROOT)
            ),
        },
        "reproducibility": {
            "validation_rows": len(overlap),
            "data_files_read": [
                str(VALIDATION_SCORES_PATH.relative_to(PROJECT_ROOT)),
                str(VALIDATION_FEATURES_PATH.relative_to(PROJECT_ROOT)),
            ],
        },
    }
    OVERLAP_SUMMARY_PATH.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    SELECTED_ENSEMBLE_PATH.write_text(
        json.dumps(
            {
                "candidate": best_name,
                "formula": (
                    "alpha * context_score + (1 - alpha) * baseline_score"
                    if best["method"] == "weighted_blend"
                    else "max(baseline_score, context_score)"
                ),
                "alpha_context": best["alpha_context"],
                "alpha_baseline": best["alpha_baseline"],
                "threshold": best["threshold"],
                "validation_metrics": best,
                "recall_gate": RECALL_GATE,
                "selection_data": "Feb 17 validation only",
                "attack_score_is_calibrated_probability": False,
                "locked_holdout_evaluated": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps(selected["overlap"], indent=2))
    print(json.dumps(selected["ensemble_search"], indent=2))
    print("No model was retrained. Feb 18 was not loaded or evaluated.")


if __name__ == "__main__":
    main()
