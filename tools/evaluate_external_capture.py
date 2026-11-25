"""Evaluate an independent labelled CICFlowMeter capture without schema fiction.

The frozen product model consumes 128 features. Forty-three of those features
are causal context built from source/destination IPs and timestamps, so a
public flow file is never made compatible by filling missing identity columns
with constants. This evaluator audits compatibility first. When a file is
compatible it can run the frozen detector; otherwise it reports the exact
reason and evaluates only a fixed, label-independent behavioural comparator.

Raw datasets are intentionally kept outside the repository; the JSON report
contains only provenance, fingerprints and aggregate measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Keep direct ``python tools/evaluate_external_capture.py`` execution aligned
# with the other repository tools; the source checkout is not required to be
# installed as a wheel for an evaluation run.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.security_anomaly.contracts import FeatureContract

DEFAULT_CONTRACT = PROJECT_ROOT / "contracts" / "feature-contract-cicflow-v2-128.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "contracts" / "model-manifest-context-rf-v2.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "context-rf-v2.joblib"
DEFAULT_SOURCE_URL = (
    "https://cse-cic-ids2018.s3.amazonaws.com/"
    "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
)
DEFAULT_SOURCE_SIZE_BYTES = 107_842_858
DEFAULT_SOURCE_SHA256 = (
    "b0534c5d7d8b41e03df71c6966c995d116a8ed28e61f377c8b14cdf5d28f4edf"
)

SOURCE_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
PRODUCT_TIMESTAMP_FORMAT = "%d/%m/%Y %I:%M:%S %p"
DEFAULT_LABELS = ("Benign", "Infilteration")
DEFAULT_ANALYST_BUDGET = 10_000
BEHAVIOR_THRESHOLD = 0.50

# Policy constants, not quantiles learned from the external file. Equal
# weighting makes the comparator an auditable ranking signal, not a probability.
BEHAVIOR_LIMITS: dict[str, float] = {
    "Flow Pkts/s": 1_000.0,
    "Flow Byts/s": 1_000_000.0,
    "SYN Flag Cnt": 3.0,
    "RST Flag Cnt": 3.0,
}


class ExternalEvaluationError(ValueError):
    """Raised when a benchmark input cannot be evaluated defensibly."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.name
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _read_csv_header(path: str | Path) -> list[str]:
    """Read the raw header so duplicate columns cannot be silently renamed."""

    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle, strict=True), None)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ExternalEvaluationError(f"Cannot read CSV header: {error}") from error
    if not header:
        raise ExternalEvaluationError("External CSV is empty or has no header")
    if any(not column for column in header):
        raise ExternalEvaluationError("External CSV header contains an empty column")
    duplicates = sorted({column for column in header if header.count(column) > 1})
    if duplicates:
        raise ExternalEvaluationError(
            f"External CSV header contains duplicate columns: {duplicates}"
        )
    return header


def _timestamp_audit(values: pd.Series) -> dict[str, Any]:
    source = pd.to_datetime(values, format=SOURCE_TIMESTAMP_FORMAT, errors="coerce")
    product = pd.to_datetime(values, format=PRODUCT_TIMESTAMP_FORMAT, errors="coerce")
    return {
        "source_format": SOURCE_TIMESTAMP_FORMAT,
        "product_format": PRODUCT_TIMESTAMP_FORMAT,
        "rows": len(values),
        "source_parseable_rows": int(source.notna().sum()),
        "product_parseable_rows": int(product.notna().sum()),
        "source_only_rows": int((source.notna() & product.isna()).sum()),
        "product_only_rows": int((product.notna() & source.isna()).sum()),
        "source_format_is_unambiguous": bool(len(values) == 0 or source.notna().all()),
        "product_format_is_unambiguous": bool(
            len(values) == 0 or product.notna().all()
        ),
    }


def _normalise_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _nearby_name_matches(
    expected: Iterable[str], actual: Iterable[str]
) -> dict[str, list[str]]:
    """Return spelling-near names; never use these as automatic aliases."""

    actual_list = list(actual)
    by_normalized: dict[str, list[str]] = {}
    for name in actual_list:
        by_normalized.setdefault(_normalise_name(name), []).append(name)
    return {
        name: by_normalized.get(_normalise_name(name), [])
        for name in expected
        if name not in actual_list and by_normalized.get(_normalise_name(name))
    }


def audit_frozen_rf_compatibility(
    frame: pd.DataFrame, contract: FeatureContract
) -> dict[str, Any]:
    """Audit whether ``frame`` can enter the product's frozen RF path.

    A name that merely looks similar is reported as a possible spelling match,
    never renamed. Unit and direction semantics need a source-specific audit.
    """

    # The evaluator keeps bookkeeping columns with a reserved leading
    # underscore. They are not source columns and must not affect the schema
    # decision or its provenance report.
    columns = [
        str(column)
        for column in frame.columns
        if not str(column).startswith("_")
    ]
    required_context = list(
        contract.document["input"]["required_identity_context_columns"]
    )
    required_model = list(contract.baseline_features)
    missing_context = [name for name in required_context if name not in columns]
    missing_model = [name for name in required_model if name not in columns]
    source_timestamp_values = (
        frame["Timestamp"].astype("string")
        if "Timestamp" in frame.columns
        else pd.Series([], dtype="string")
    )
    timestamps = _timestamp_audit(source_timestamp_values)

    issues: list[str] = []
    if missing_context:
        issues.append(
            "missing required identity context columns: " + ", ".join(missing_context)
        )
    if missing_model:
        issues.append(
            "missing exact frozen baseline feature names: " + ", ".join(missing_model)
        )
    if "Timestamp" not in frame.columns:
        issues.append("Timestamp column is absent")
    elif not timestamps["product_format_is_unambiguous"]:
        issues.append(
            "Timestamp does not use the product's exact CICFlowMeter format "
            f"{PRODUCT_TIMESTAMP_FORMAT!r}"
        )
    if frame.columns.has_duplicates:
        issues.append("duplicate columns are present")

    source_specific_differences = []
    if "CWE Flag Count" in columns and "CWR Flag Count" not in columns:
        source_specific_differences.append(
            {
                "source_column": "CWE Flag Count",
                "contract_column": "CWR Flag Count",
                "decision": (
                    "upstream CICFlowMeter semantic alias is known, but the "
                    "exact product column is not renamed automatically"
                ),
            }
        )
    for source_name, contract_name in (
        ("Flow Byts/s", "Flow Bytes/s"),
        ("Flow Pkts/s", "Flow Packets/s"),
        ("Tot Fwd Pkts", "Total Fwd Packet"),
        ("Tot Bwd Pkts", "Total Bwd packets"),
    ):
        if source_name in columns and contract_name not in columns:
            source_specific_differences.append(
                {
                    "source_column": source_name,
                    "contract_column": contract_name,
                    "decision": "not renamed; source-specific schema audit required",
                }
            )

    return {
        "compatible": not issues,
        "required_identity_context_columns": required_context,
        "required_baseline_feature_count": len(required_model),
        "missing_identity_context_columns": missing_context,
        "missing_baseline_feature_names": missing_model,
        "nearby_spelling_matches_not_renamed": _nearby_name_matches(
            [*required_context, *required_model], columns
        ),
        "source_columns": columns,
        "source_column_count": len(columns),
        "timestamp_audit": timestamps,
        "source_specific_name_differences": source_specific_differences,
        "units_and_direction": (
            "No unit or forward/backward-direction conversion was applied; the "
            "source's CICFlowMeter-V3 claim does not by itself prove the product's "
            "exact feature contract."
        ),
        "issues": issues,
        "no_zero_fill_or_implicit_rename": True,
    }


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ExternalEvaluationError(f"Comparator input is missing {column!r}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ExternalEvaluationError(
            f"Comparator input contains non-finite values in {column!r}"
        )
    return values


def behavioural_scores(frame: pd.DataFrame) -> np.ndarray:
    """Return a fixed, transparent high-rate/flag score in [0, 1]."""

    components = []
    for column, limit in BEHAVIOR_LIMITS.items():
        values = _numeric_values(frame, column)
        components.append(np.clip(values / limit, 0.0, 1.0))
    score = np.mean(np.stack(components, axis=1), axis=1)
    if not np.isfinite(score).all():
        raise ExternalEvaluationError("Comparator produced a non-finite score")
    return score.astype(float, copy=False)


def top_k_mask(scores: np.ndarray, budget: int) -> np.ndarray:
    """Select the same deterministic review budget for every method."""

    if budget <= 0:
        raise ValueError("analyst budget must be positive")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite one-dimensional array")
    take = min(int(budget), len(values))
    # ``lexsort`` uses the last key as primary: descending score, then source
    # row order as a stable, data-independent tie break.
    order = np.lexsort((np.arange(len(values), dtype=np.int64), -values))
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:take]] = True
    return selected


def _classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Mapping[int, str],
    *,
    score: np.ndarray,
    threshold: float | None,
    capture_hours: float | None,
    budget: int | None,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=bool)
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    attack = labels == 1
    tp = int(np.count_nonzero(predictions & attack))
    fp = int(np.count_nonzero(predictions & ~attack))
    fn = int(np.count_nonzero(~predictions & attack))
    tn = int(np.count_nonzero(~predictions & ~attack))
    predicted_positive = tp + fp
    actual_positive = tp + fn
    precision = tp / predicted_positive if predicted_positive else 0.0
    recall = tp / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    flow_count = len(labels)

    per_class: dict[str, dict[str, Any]] = {}
    for value, name in class_names.items():
        class_mask = labels == value
        alerts = predictions & class_mask
        count = int(class_mask.sum())
        alert_count = int(alerts.sum())
        per_class[name] = {
            "rows": count,
            "alerts_or_reviewed": alert_count,
            "rate": alert_count / count if count else 0.0,
            "rate_denominator": "rows in this labelled class",
        }

    alert_count = int(predictions.sum())
    return {
        "threshold": threshold,
        "analyst_budget": budget,
        "flows": flow_count,
        "alerts_or_reviewed": alert_count,
        "alert_or_review_rate": alert_count / flow_count if flow_count else 0.0,
        "workload_denominator": "all evaluated labelled rows",
        "alerts_or_reviewed_per_1000_flows": (
            alert_count / flow_count * 1000 if flow_count else 0.0
        ),
        "capture_hours": capture_hours,
        "alerts_or_reviewed_per_capture_hour": (
            alert_count / capture_hours
            if capture_hours is not None and capture_hours > 0
            else None
        ),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class,
        "score_summary": {
            "min": float(np.min(score)) if len(score) else None,
            "median": float(np.median(score)) if len(score) else None,
            "max": float(np.max(score)) if len(score) else None,
        },
    }


def evaluate_scores(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    analyst_budget: int,
    class_names: Mapping[int, str] | None = None,
    timestamps: pd.Series | None = None,
) -> dict[str, Any]:
    """Evaluate fixed-threshold and equal-budget views of one score vector."""

    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if class_names is None:
        class_names = {0: "Benign", 1: "Attack"}
    capture_hours: float | None = None
    timestamp_range: dict[str, Any] | None = None
    if timestamps is not None and len(timestamps):
        parsed = pd.to_datetime(timestamps, errors="coerce")
        if parsed.isna().any():
            raise ValueError("timestamps must be parseable for workload rates")
        span_seconds = float((parsed.max() - parsed.min()).total_seconds())
        capture_hours = span_seconds / 3600.0 if span_seconds > 0 else None
        timestamp_range = {
            "minimum": pd.Timestamp(parsed.min()).isoformat(),
            "maximum": pd.Timestamp(parsed.max()).isoformat(),
            "span_seconds": span_seconds,
            "span_hours": capture_hours,
            "denominator": "max(timestamp)-min(timestamp), no padding",
        }

    threshold_predictions = scores >= threshold
    budget_predictions = top_k_mask(scores, analyst_budget)
    return {
        "threshold_policy": _classification_metrics(
            labels,
            threshold_predictions,
            class_names,
            score=scores,
            threshold=threshold,
            capture_hours=capture_hours,
            budget=None,
        ),
        "fixed_analyst_budget_policy": _classification_metrics(
            labels,
            budget_predictions,
            class_names,
            score=scores,
            threshold=None,
            capture_hours=capture_hours,
            budget=min(analyst_budget, len(scores)),
        ),
        "timestamp_range": timestamp_range,
        "score_definition": {
            "components": {
                name: f"clip({name} / {limit:g}, 0, 1)"
                for name, limit in BEHAVIOR_LIMITS.items()
            },
            "aggregation": "arithmetic mean of four equally weighted components",
            "threshold": threshold,
            "threshold_operator": ">=",
            "score_is_probability": False,
        },
    }


def _source_label_counts(frame: pd.DataFrame) -> dict[str, int]:
    values = frame["Label"].astype("string").fillna("<NA>")
    counts = values.value_counts(dropna=False)
    return {str(label): int(count) for label, count in counts.items()}


def load_external_slice(
    csv_path: str | Path,
    *,
    labels: tuple[str, ...] = DEFAULT_LABELS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and deterministically clean one labelled external CSV."""

    path = Path(csv_path)
    header = _read_csv_header(path)
    if "Label" not in header:
        raise ExternalEvaluationError("The external source must contain Label")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ExternalEvaluationError(f"Cannot read external CSV: {error}") from error
    if raw.columns.tolist() != header:
        raise ExternalEvaluationError(
            "CSV columns changed during parsing; header is not losslessly representable"
        )

    raw_rows = len(raw)
    raw_counts = _source_label_counts(raw)
    repeated_header = raw["Label"].astype("string").eq("Label")
    known_label = raw["Label"].astype("string").isin(labels)
    selected = raw.loc[~repeated_header & known_label].copy()
    unrecognized = int((~repeated_header & ~known_label).sum())
    selected.insert(0, "_source_row_index", selected.index.astype(np.int64))

    if "Timestamp" not in selected:
        raise ExternalEvaluationError("The external source must contain Timestamp")
    parsed_timestamps = pd.to_datetime(
        selected["Timestamp"], format=SOURCE_TIMESTAMP_FORMAT, errors="coerce"
    )
    invalid_timestamp = parsed_timestamps.isna()
    if invalid_timestamp.any():
        raise ExternalEvaluationError(
            f"External source has {int(invalid_timestamp.sum())} invalid source timestamps"
        )

    behavior_columns = list(BEHAVIOR_LIMITS)
    for column in behavior_columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    behavior_matrix = selected[behavior_columns].to_numpy(dtype=float)
    finite = np.isfinite(behavior_matrix).all(axis=1)
    nonfinite_count = int((~finite).sum())
    selected = selected.loc[finite].copy()
    parsed_timestamps = parsed_timestamps.loc[selected.index]
    if selected.empty:
        raise ExternalEvaluationError("No finite labelled rows remain after cleaning")
    selected["_parsed_timestamp"] = parsed_timestamps.to_numpy()
    selected["_binary_label"] = (
        selected["Label"].astype("string") != labels[0]
    ).astype(np.int8)
    selected = selected.reset_index(drop=True)

    info = {
        "raw_rows": int(raw_rows),
        "raw_label_counts": raw_counts,
        "repeated_header_rows_excluded": int(repeated_header.sum()),
        "unrecognized_label_rows_excluded": unrecognized,
        "nonfinite_behavior_rows_excluded": nonfinite_count,
        "evaluated_rows": len(selected),
        "evaluated_label_counts": {
            str(label): int(count)
            for label, count in selected["Label"].value_counts().items()
        },
        "label_mapping": {
            "0": labels[0],
            "1": "all other declared labels: " + ", ".join(labels[1:]),
        },
        "row_selection": (
            "source file order; no random sampling; repeated headers, labels "
            "outside the declared two-class slice, and non-finite comparator rows excluded"
        ),
        "source_timestamp_format": SOURCE_TIMESTAMP_FORMAT,
    }
    return selected, info


def _load_manifest_metadata(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        return {"status": "manifest_not_found", "path": manifest_path.name}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {
        "status": "fixed_manifest_read",
        "model_version": manifest.get("model_version"),
        "model_sha256": manifest.get("artifact", {}).get("sha256"),
        "feature_contract": manifest.get("compatibility", {}).get("feature_contract"),
        "feature_builder_version": manifest.get("compatibility", {}).get(
            "feature_builder_version"
        ),
        "flow_threshold": manifest.get("operational_policy", {})
        .get("flow_alert", {})
        .get("threshold"),
        "promotion_threshold": manifest.get("operational_policy", {})
        .get("incident_promotion", {})
        .get("threshold"),
        "path": manifest_path.name,
    }


def _try_frozen_model_scores(
    frame: pd.DataFrame,
    compatibility: Mapping[str, Any],
    *,
    model_path: Path,
    manifest_path: Path,
    contract_path: Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not compatibility["compatible"]:
        return None, {
            "status": "skipped_incompatible_schema",
            "reason": "; ".join(compatibility["issues"]),
        }
    if not model_path.is_file():
        return None, {"status": "skipped_model_not_found", "path": model_path.name}
    try:
        from src.security_anomaly.detector import SecurityAnomalyDetector

        detector = SecurityAnomalyDetector.load(
            model_path=model_path,
            manifest_path=manifest_path,
            feature_contract_path=contract_path,
        )
        input_columns = list(detector.bundle.contract.required_input_columns)
        input_columns.append("Label")
        model_input = frame.loc[:, list(dict.fromkeys(input_columns))].copy()
        result = detector.analyze_batch(model_input)
        scores = np.full(len(frame), np.nan, dtype=float)
        for detection in result.flow_detections:
            scores[int(detection.source_row_id)] = float(detection.attack_score)
        if not np.isfinite(scores).all():
            raise ExternalEvaluationError("Frozen detector did not score every row")
        return scores, {
            "status": "evaluated",
            "model_version": detector.bundle.model_version,
            "flow_threshold": detector.policy.flow_threshold,
            "promotion_threshold": detector.policy.promotion_threshold,
        }
    except Exception as error:  # noqa: BLE001 - report exact audit failure
        return None, {
            "status": "failed_to_load_or_score",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run_external_benchmark(
    csv_path: str | Path,
    *,
    source_url: str = DEFAULT_SOURCE_URL,
    expected_sha256: str | None = DEFAULT_SOURCE_SHA256,
    expected_size_bytes: int | None = DEFAULT_SOURCE_SIZE_BYTES,
    analyst_budget: int = DEFAULT_ANALYST_BUDGET,
    contract_path: str | Path = DEFAULT_CONTRACT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    model_path: str | Path = DEFAULT_MODEL,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Run the external benchmark and return a raw-data-free report."""

    path = Path(csv_path)
    if not path.is_file():
        raise ExternalEvaluationError(f"External CSV not found: {path}")
    if analyst_budget <= 0:
        raise ValueError("analyst_budget must be positive")
    contract = FeatureContract.load(contract_path)
    frame, selection = load_external_slice(path)
    compatibility = audit_frozen_rf_compatibility(frame, contract)
    scores = behavioural_scores(frame)
    labels = frame["_binary_label"].to_numpy(dtype=np.int8)
    timestamps = frame["_parsed_timestamp"]
    behavioural = evaluate_scores(
        labels,
        scores,
        threshold=BEHAVIOR_THRESHOLD,
        analyst_budget=analyst_budget,
        class_names={0: DEFAULT_LABELS[0], 1: DEFAULT_LABELS[1]},
        timestamps=timestamps,
    )

    model_scores, model_status = _try_frozen_model_scores(
        frame,
        compatibility,
        model_path=Path(model_path),
        manifest_path=Path(manifest_path),
        contract_path=Path(contract_path),
    )
    manifest_metadata = _load_manifest_metadata(Path(manifest_path))
    model_metrics = (
        evaluate_scores(
            labels,
            model_scores,
            threshold=float(manifest_metadata.get("flow_threshold", 0.10)),
            analyst_budget=analyst_budget,
            class_names={0: DEFAULT_LABELS[0], 1: DEFAULT_LABELS[1]},
            timestamps=timestamps,
        )
        if model_scores is not None
        else None
    )

    actual_hash = sha256_file(path) if compute_sha256 else None
    if expected_sha256 and actual_hash and actual_hash.lower() != expected_sha256.lower():
        raise ExternalEvaluationError(
            "External source SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_hash}"
        )
    actual_size = path.stat().st_size
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise ExternalEvaluationError(
            "External source size mismatch: "
            f"expected {expected_size_bytes}, got {actual_size}"
        )

    return {
        "report_schema": "external-evaluation-v1",
        "source": {
            "official_dataset_page": "https://www.unb.ca/cic/datasets/ids-2018.html",
            "official_object_url": source_url,
            "filename": path.name,
            "object_size_bytes": actual_size,
            "sha256": actual_hash,
            "expected_sha256": expected_sha256,
            "expected_size_bytes": expected_size_bytes,
            "raw_data_committed": False,
            "capture_profile": "CSE-CIC-IDS2018 / Thursday-01-03-2018",
            "feature_extractor_claim": "CICFlowMeter-V3 per the official dataset page",
        },
        "selection": selection,
        "frozen_policy": {
            "model_manifest": manifest_metadata,
            "model_thresholds_fixed_before_results": True,
            "analyst_budget_flows": analyst_budget,
            "budget_selection": "descending score, source row index tie-break",
            "behavioural_threshold": BEHAVIOR_THRESHOLD,
            "behavioural_limits": BEHAVIOR_LIMITS,
            "labels_fixed_before_results": list(DEFAULT_LABELS),
        },
        "frozen_rf_compatibility": compatibility,
        "frozen_rf_run": model_status,
        "behavioural_comparator": behavioural,
        "frozen_rf_metrics": model_metrics,
        "limitations": [
            "This is one labelled day/profile, not a representative deployment prevalence estimate.",
            "CSE-CIC-IDS2018 traffic is generated/testbed traffic; results do not establish performance on an independent live enterprise network.",
            "CSE processed rows omit endpoint identity required for causal context, so frozen RF metrics are intentionally absent here.",
            "The behavioural comparator is a fixed workload baseline, not a calibrated probability or a replacement for the frozen model.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--expected-sha256", default=DEFAULT_SOURCE_SHA256)
    parser.add_argument("--expected-size-bytes", type=int, default=DEFAULT_SOURCE_SIZE_BYTES)
    parser.add_argument("--analyst-budget", type=int, default=DEFAULT_ANALYST_BUDGET)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip source hash/size checks only when a previously recorded hash is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_external_benchmark(
        args.csv,
        source_url=args.source_url,
        expected_sha256=None if args.skip_sha256 else args.expected_sha256,
        expected_size_bytes=None if args.skip_sha256 else args.expected_size_bytes,
        analyst_budget=args.analyst_budget,
        contract_path=args.contract,
        manifest_path=args.manifest,
        model_path=args.model,
        compute_sha256=not args.skip_sha256,
    )
    rendered = json.dumps(report, indent=2, default=_json_value, sort_keys=True)
    if args.output:
        from security_anomaly.paths import ensure_distinct
        from security_anomaly.serialization import atomic_text_output

        ensure_distinct(args.output, args.csv, args.model, args.manifest, args.contract)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with atomic_text_output(args.output) as destination:
            destination.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
