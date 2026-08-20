"""Command-line interface for label-free Security Anomaly ML v0.1 analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

from .contracts import ContractError
from .detector import SecurityAnomalyDetector
from .model_bundle import ArtifactIntegrityError, ModelCompatibilityError
from .package_resources import (
    FEATURE_CONTRACT_RESOURCE,
    MODEL_MANIFEST_RESOURCE,
    materialized_resource,
    resource_json,
)
from .serialization import (
    INCIDENT_SCHEMA_VERSION,
    IncidentSerializationError,
    OutputWriteError,
    promoted_incidents_to_v1,
    write_incidents_jsonl,
)
from .validation import InputContractError, read_and_validate_flow_csv


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_MODEL = 4
EXIT_OUTPUT = 5
EXIT_RUNTIME = 10
MODEL_ENVIRONMENT_VARIABLE = "SECURITY_ANOMALY_MODEL"


class ModelResolutionError(RuntimeError):
    """Raised when a local frozen model cannot be located or verified."""


def _json_stdout(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def resolve_model_path(explicit: Path | None) -> Path:
    """Resolve the model using CLI > environment > conventional local paths."""

    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if not candidate.is_file():
            raise ModelResolutionError(f"Model artifact not found: {candidate}")
        return candidate
    environment_value = os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if environment_value:
        candidate = Path(environment_value).expanduser().resolve()
        if not candidate.is_file():
            raise ModelResolutionError(
                f"{MODEL_ENVIRONMENT_VARIABLE} does not point to a file: {candidate}"
            )
        return candidate
    candidates = (
        Path.cwd() / "models" / "context-rf-v2.joblib",
        Path.cwd() / "models" / "v2_context_random_forest.joblib",
        Path.cwd() / "context-rf-v2.joblib",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ModelResolutionError(
        "Frozen model asset is missing. Supply --model PATH or set "
        f"{MODEL_ENVIRONMENT_VARIABLE}; no download is performed."
    )


def _load_detector(model_path: Path) -> SecurityAnomalyDetector:
    try:
        with materialized_resource(MODEL_MANIFEST_RESOURCE) as manifest_path:
            with materialized_resource(FEATURE_CONTRACT_RESOURCE) as contract_path:
                return SecurityAnomalyDetector.load(
                    model_path=model_path,
                    manifest_path=manifest_path,
                    feature_contract_path=contract_path,
                )
    except (
        ArtifactIntegrityError,
        ContractError,
        FileNotFoundError,
        ModelCompatibilityError,
        OSError,
    ) as error:
        raise ModelResolutionError(str(error)) from error


def _version_payload() -> dict[str, object]:
    manifest = resource_json(MODEL_MANIFEST_RESOURCE)
    contract = resource_json(FEATURE_CONTRACT_RESOURCE)
    return {
        "product_version": manifest["product_version"],
        "feature_contract": contract["feature_contract"],
        "feature_builder_version": contract["feature_builder_version"],
        "incident_contract": INCIDENT_SCHEMA_VERSION,
    }


def _command_version(_args: argparse.Namespace) -> int:
    _json_stdout(_version_payload())
    return EXIT_SUCCESS


def _command_validate(args: argparse.Namespace) -> int:
    batch = read_and_validate_flow_csv(args.input)
    payload: dict[str, object] = {
        "valid": True,
        "rows": batch.row_count,
        "input_contract": "CICFlowMeter-compatible-v0.1",
        "feature_contract": batch.contract_version,
    }
    if batch.timestamp_min is not None:
        payload["timestamp_min"] = batch.timestamp_min.to_pydatetime().isoformat(
            timespec="seconds"
        )
        payload["timestamp_max"] = batch.timestamp_max.to_pydatetime().isoformat(
            timespec="seconds"
        )
    _json_stdout(payload)
    return EXIT_SUCCESS


def _command_analyze(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    batch = read_and_validate_flow_csv(args.input)
    model_path = resolve_model_path(args.model)
    output = args.output.expanduser().resolve()
    if Path(args.input).expanduser().resolve() == output:
        raise OutputWriteError("Output path must differ from input CSV path")
    detector = _load_detector(model_path)
    result = detector.analyze_batch(batch.frame)
    public_incidents = promoted_incidents_to_v1(result)
    write_incidents_jsonl(output, public_incidents)
    summary = {
        "validation_status": "valid",
        "rejected_rows": 0,
        "flows_processed": result.flows_processed,
        "flow_alerts": result.flow_alert_count,
        "aggregated_incidents": result.incident_count,
        "promoted_incidents": result.promoted_incident_count,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "product_version": result.product_version,
        "model_version": result.model_version,
        "feature_contract": result.feature_contract,
        "incident_contract": INCIDENT_SCHEMA_VERSION,
        "output": str(output),
    }
    print(json.dumps(summary, allow_nan=False, separators=(",", ":")), file=sys.stderr)
    return EXIT_SUCCESS


def _command_model_info(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model)
    detector = _load_detector(model_path)
    metadata = detector.bundle.metadata()
    policy = detector.policy
    _json_stdout(
        {
            "product_version": metadata["product_version"],
            "model_version": metadata["model_version"],
            "model_sha256": metadata["model_sha256"],
            "model_path": str(model_path),
            "feature_contract": metadata["feature_contract"],
            "feature_builder_version": metadata["feature_builder_version"],
            "flow_threshold": policy.flow_threshold,
            "aggregation_policy": policy.incident_policy,
            "grouping_key": list(policy.grouping_key),
            "incident_window_seconds": policy.incident_window_seconds,
            "promotion_threshold": policy.promotion_threshold,
            "serialization": metadata["serialization"],
        }
    )
    return EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="security-anomaly",
        description=(
            "Validate and analyze unlabeled CICFlowMeter-compatible flow CSV files "
            "with the frozen context-rf-v2 detector."
        ),
        epilog=(
            "Example: security-anomaly analyze flows.csv --model context-rf-v2.joblib "
            "--output incidents.jsonl"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show a traceback for unexpected failures",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="validate, score, aggregate, and write promoted incidents"
    )
    analyze.add_argument("input", type=Path, help="unlabeled CICFlowMeter CSV")
    analyze.add_argument("--model", type=Path, help="frozen context-rf-v2 joblib path")
    analyze.add_argument("--output", required=True, type=Path, help="output JSONL path")
    analyze.set_defaults(handler=_command_analyze)

    validate = subparsers.add_parser(
        "validate", help="validate input without loading a model"
    )
    validate.add_argument("input", type=Path, help="CICFlowMeter CSV to validate")
    validate.set_defaults(handler=_command_validate)

    version = subparsers.add_parser(
        "version", help="show product and contract versions without loading a model"
    )
    version.set_defaults(handler=_command_version)

    model_info = subparsers.add_parser(
        "model-info", help="verify the selected frozen model and show its metadata"
    )
    model_info.add_argument("--model", type=Path, help="frozen context-rf-v2 joblib path")
    model_info.set_defaults(handler=_command_model_info)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except InputContractError as error:
        print(f"input validation failed: {error}", file=sys.stderr)
        return EXIT_INPUT
    except ModelResolutionError as error:
        print(f"model error: {error}", file=sys.stderr)
        return EXIT_MODEL
    except (IncidentSerializationError, OutputWriteError) as error:
        print(f"output error: {error}", file=sys.stderr)
        return EXIT_OUTPUT
    except Exception as error:  # pragma: no cover - final CLI safety boundary
        print(f"unexpected runtime failure: {error}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return EXIT_RUNTIME


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
