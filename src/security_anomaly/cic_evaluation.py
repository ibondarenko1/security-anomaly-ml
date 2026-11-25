"""Evaluate the frozen CICFlowMeter detector against incident-level review budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .aggregation import POLICY_B_GROUPING_KEY, group_policy_b_flows
from .evaluation import ReviewCase, compare_cases, operating_point, read_labels
from .paths import ensure_distinct
from .serialization import atomic_text_output, incident_id_v1
from .validation import InputContractError, read_and_validate_flow_csv_bytes


def _observation_ids(raw: bytes, digest: str, count: int, mode: str) -> tuple[str, ...]:
    if mode == "row":
        # Bind positional labels to the exact capture, including its row order.
        return tuple(f"{digest}:{n}" for n in range(1, count + 1))
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "Flow ID" not in (reader.fieldnames or []):
            raise InputContractError("evaluate cic needs a unique Flow ID or --uid-mode row")
        ids = tuple(row["Flow ID"] for row in reader)
    if len(ids) != count or any(not uid or uid != uid.strip() for uid in ids):
        raise InputContractError("Flow IDs must be non-empty, unpadded record IDs; use --uid-mode row otherwise")
    if len(set(ids)) != count:
        raise InputContractError("Flow IDs repeat; use --uid-mode row for tuple-based exporter IDs")
    return ids


def _command(args: argparse.Namespace) -> int:
    from .cli import _load_detector, resolve_model_path

    source = args.input.expanduser()
    model = resolve_model_path(args.model)
    output = args.output.expanduser()
    ensure_distinct(output, source, args.labels, model)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise InputContractError(f"Cannot read CICFlowMeter CSV: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    batch = read_and_validate_flow_csv_bytes(raw, source=source)
    ids = _observation_ids(raw, digest, batch.row_count, args.uid_mode)
    fwd = batch.frame["Total Fwd Packet"].to_numpy(dtype=float)
    bwd = batch.frame["Total Bwd packets"].to_numpy(dtype=float)
    with np.errstate(over="ignore"):
        packets = fwd + bwd
    if (fwd < 0).any() or (bwd < 0).any() or not np.isfinite(packets).all():
        raise InputContractError("Packet-count comparison needs nonnegative counters and finite totals")

    detector = _load_detector(model)
    result = detector.analyze_batch(batch.frame)
    policy = detector.policy
    # The fixed-budget comparison groups the entire population before any
    # method is ranked. The deployed RF groups only alerted flows, below.
    groups = group_policy_b_flows(
        result.flow_detections,
        window_seconds=policy.incident_window_seconds,
        promotion_threshold=policy.promotion_threshold,
    )
    cases = [
        ReviewCase(
            incident_id_v1(group),
            tuple(ids[i] for i in group.source_row_ids),
            {"frozen_rf": group.max_attack_score,
             "packet_count": max(float(packets[i]) for i in group.source_row_ids)},
        )
        for group in groups
    ]
    policy_cases = [
        ReviewCase(
            incident_id_v1(group),
            tuple(ids[i] for i in group.source_row_ids),
            {"frozen_rf": group.max_attack_score},
            {"frozen_rf": group.promoted},
        )
        for group in result.incidents
    ]
    labels = read_labels(args.labels, ids)
    report = {
        "schema_version": "cic-evaluation-v1",
        "product_version": __version__,
        "model_version": result.model_version,
        "model_sha256": detector.bundle.metadata()["model_sha256"],
        "input_sha256": digest,
        "parameters": {
            "feature_contract": result.feature_contract,
            "uid_mode": args.uid_mode,
            "group_key": list(POLICY_B_GROUPING_KEY),
            "gap_seconds": policy.incident_window_seconds,
            "flow_threshold": policy.flow_threshold,
            "promotion_threshold": policy.promotion_threshold,
        },
        "coverage": {
            "accepted_flows": result.flows_processed,
            "all_flow_groups": len(cases),
            "alert_flows": result.flow_alert_count,
            "alert_groups": result.incident_count,
            "promoted_groups": result.promoted_incident_count,
        },
        "comparison": compare_cases(
            cases, labels, args.budget, seed=args.seed,
            methods=("frozen_rf", "packet_count", "random"),
        ),
        "operating_point": operating_point(policy_cases, labels, methods=("frozen_rf",)),
    }
    with atomic_text_output(output) as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"command": "evaluate cic", "cases": len(cases), "output": str(output)}), file=sys.stderr)
    return 0


def register_parser(formats) -> None:
    parser = formats.add_parser("cic", help="compare the frozen RF and packet volume on a labelled capture")
    parser.add_argument("input", type=Path, help="CICFlowMeter CSV in the frozen input contract")
    parser.add_argument("--labels", type=Path, required=True, help="separate UID/episode JSONL")
    parser.add_argument("--model", type=Path, help="verified frozen RF artifact")
    parser.add_argument("--uid-mode", choices=("flow-id", "row"), default="flow-id",
                        help="unique original Flow ID or <input-sha256>:<1-based data row>")
    parser.add_argument("--budget", type=int, default=10, help="number of cases per review queue")
    parser.add_argument("--seed", default="security-anomaly-ml", help="seed for reproducible random selection")
    parser.add_argument("--output", type=Path, required=True, help="evaluation JSON")
    parser.set_defaults(handler=_command)
