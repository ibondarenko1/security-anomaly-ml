"""Run the frozen CICFlowMeter detector on four fixed ranges of CSE-CIC-IDS2018."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from security_anomaly.cli import _load_detector
from security_anomaly.paths import ensure_distinct
from security_anomaly.serialization import atomic_text_output
from security_anomaly.validation import TIMESTAMP_FORMAT

BENCHMARKS = Path(__file__).resolve().parents[1] / "docs" / "benchmarks"
# Explicit spelling changes for the 2018 exporter; the remaining fields match by name.
ALIASES = {
    "Tot Fwd Pkts": "Total Fwd Packet",
    "Tot Bwd Pkts": "Total Bwd packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packet",
    "TotLen Bwd Pkts": "Total Length of Bwd Packet",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Len Var": "Packet Length Variance",
    "CWE Flag Count": "CWR Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Fwd Segment Size Avg",
    "Bwd Seg Size Avg": "Bwd Segment Size Avg",
    "Fwd Byts/b Avg": "Fwd Bytes/Bulk Avg",
    "Fwd Pkts/b Avg": "Fwd Packet/Bulk Avg",
    "Fwd Blk Rate Avg": "Fwd Bulk Rate Avg",
    "Bwd Byts/b Avg": "Bwd Bytes/Bulk Avg",
    "Bwd Pkts/b Avg": "Bwd Packet/Bulk Avg",
    "Bwd Blk Rate Avg": "Bwd Bulk Rate Avg",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",
    "Init Fwd Win Byts": "FWD Init Win Bytes",
    "Init Bwd Win Byts": "Bwd Init Win Bytes",
}
for direction in ("Fwd ", "Bwd ", ""):
    for statistic in ("Min", "Max", "Mean", "Std"):
        ALIASES[f"{direction}Pkt Len {statistic}"] = (
            f"{direction}Packet Length {statistic}"
        )
for flag in ("FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE"):
    ALIASES[f"{flag} Flag Cnt"] = f"{flag} Flag Count"


def read_range(protocol: dict, start: int) -> bytes:
    # Read a leading guard byte so a row starting exactly at the boundary is retained.
    low = max(0, start - 1)
    high = start + protocol["bytes_per_slice"] - 1
    match = f'"{protocol["object_etag"]}"'
    request = Request(
        protocol["source"], headers={"Range": f"bytes={low}-{high}", "If-Match": match}
    )
    with urlopen(request, timeout=90) as response:
        if (
            response.status != 206
            or response.headers.get("ETag") != match
            or response.headers.get("Content-Range")
            != f"bytes {low}-{high}/{protocol['object_bytes']}"
        ):
            raise ValueError(
                "Source object or returned byte range differs from the frozen protocol"
            )
        payload = response.read(high - low + 2)
    if len(payload) != high - low + 1:
        raise ValueError("Incomplete source byte range")
    return payload


def complete_lines(payload: bytes, start: int) -> bytes:
    if start:
        payload = payload.partition(b"\n")[2]
    end = payload.rfind(b"\n")
    return payload[: end + 1]


def prepare(lines: bytes, header: list[str], contract):
    renamed = [ALIASES.get(name, name) for name in header]
    expected = {*contract.required_input_columns, "Flow ID", "Label"}
    if len(set(renamed)) != len(renamed) or set(renamed) != expected:
        raise ValueError(
            f"Source header differs from the feature contract: {expected.symmetric_difference(renamed)}"
        )
    raw = pd.read_csv(
        io.BytesIO(lines), header=None, names=renamed, dtype=str, keep_default_na=False
    )
    repeated = raw["Timestamp"].eq("Timestamp") & raw["Label"].eq("Label")
    frame = raw.loc[~repeated].copy()
    measurements = list(contract.baseline_features)
    numeric = frame[measurements].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    timestamps = pd.to_datetime(
        frame["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    usable = finite & timestamps.notna() & frame["Label"].ne("")
    labels = frame.loc[usable, "Label"].to_numpy()
    frame[measurements] = numeric
    frame["Timestamp"] = timestamps.dt.strftime(TIMESTAMP_FORMAT)
    counts = {
        "source_rows": len(raw),
        "repeated_headers": int(repeated.sum()),
        "unscorable": int((~usable).sum()),
        "scored": int(usable.sum()),
    }
    return (
        frame.loc[usable].drop(columns="Label").reset_index(drop=True),
        labels,
        counts,
    )


def top_budget(scores: np.ndarray, size: int) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(-scores, kind="stable")[:size]] = True
    return selected


def score(frame, labels, detector, protocol):
    result = detector.analyze_batch(frame)
    scores = np.empty(len(frame), dtype=float)
    for flow in result.flow_detections:
        scores[flow.source_row_id] = flow.attack_score
    volume = (frame["Total Fwd Packet"] + frame["Total Bwd packets"]).to_numpy(
        dtype=float
    )
    budget = int(len(frame) * protocol["budget_fraction"])
    selected_rf = top_budget(scores, budget)
    selected_volume = top_budget(volume, budget)
    alert = scores >= detector.policy.flow_threshold
    classes = {}
    for name in sorted(set(labels)):
        mask = labels == name
        classes[name] = {
            "rows": int(mask.sum()),
            "frozen_alerts": int(alert[mask].sum()),
            "top_rf": int(selected_rf[mask].sum()),
            "top_packet_count": int(selected_volume[mask].sum()),
        }
    benign_incidents = sum(
        all(labels[i] == "Benign" for i in inc.source_row_ids)
        for inc in result.promoted_incidents
    )
    return {
        "classes": classes,
        "budget_per_ranker": budget,
        "flow_alerts": result.flow_alert_count,
        "promoted_incidents": result.promoted_incident_count,
        "pure_benign_promoted": benign_incidents,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = BENCHMARKS / "cic2018-frozen-protocol.json"
    ensure_distinct(args.output, args.model, protocol_path)
    protocol = json.loads(protocol_path.read_text())
    detector = _load_detector(args.model)
    if (
        detector.bundle.metadata()["model_sha256"] != protocol["model_sha256"]
        or detector.policy.flow_threshold != protocol["flow_threshold"]
        or detector.policy.promotion_threshold != protocol["promotion_threshold"]
    ):
        raise ValueError("Model or policy differs from the benchmark protocol")
    slices = []
    header = None
    for start in protocol["slice_start_bytes"]:
        raw = read_range(protocol, start)
        lines = complete_lines(raw, start)
        if start == 0:
            first, _, lines = lines.partition(b"\n")
            header = next(csv.reader([first.decode("utf-8-sig").strip()]))
        frame, labels, coverage = prepare(lines, header, detector.bundle.contract)
        slices.append(
            {
                "start": start,
                "range_sha256_including_guard": hashlib.sha256(raw).hexdigest(),
                **coverage,
                **score(frame, labels, detector, protocol),
            }
        )
    report = {
        "protocol": protocol["protocol"],
        "model_sha256": protocol["model_sha256"],
        "slices": slices,
    }
    with atomic_text_output(args.output) as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
