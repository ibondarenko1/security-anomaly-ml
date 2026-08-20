"""Generate the small, redistributable v0.1 product regression fixture.

All values come from the explicit profiles and documentation-only IP ranges in
this file. No UNSW/CIC data is read or used by this generator.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from security_anomaly.contracts import FeatureContract  # noqa: E402


@dataclass(frozen=True)
class RowPlan:
    profile: str
    timestamp: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: int


ROW_PLANS = (
    RowPlan("zero", "20/08/2026 01:00:00 PM", "192.0.2.1", 40000, "198.51.100.1", 80, 6),
    RowPlan("quiet", "20/08/2026 01:00:00 PM", "192.0.2.2", 40001, "198.51.100.2", 443, 6),
    RowPlan("syn", "20/08/2026 01:00:00 PM", "192.0.2.3", 40002, "198.51.100.1", 53, 17),
    RowPlan("bulk", "20/08/2026 01:00:00 PM", "192.0.2.10", 40003, "198.51.100.20", 445, 6),
    RowPlan("bulk", "20/08/2026 01:00:00 PM", "192.0.2.10", 40004, "198.51.100.20", 445, 17),
    RowPlan("quiet", "20/08/2026 01:00:05 PM", "192.0.2.2", 40005, "198.51.100.2", 443, 6),
    RowPlan("bulk", "20/08/2026 01:00:05 PM", "192.0.2.10", 40006, "198.51.100.20", 445, 6),
    RowPlan("zero", "20/08/2026 01:00:05 PM", "192.0.2.1", 40007, "198.51.100.1", 80, 6),
    RowPlan("quiet", "20/08/2026 01:00:10 PM", "192.0.2.4", 40008, "198.51.100.3", 8443, 6),
    RowPlan("syn", "20/08/2026 01:00:10 PM", "192.0.2.3", 40009, "198.51.100.1", 53, 17),
    RowPlan("bulk", "20/08/2026 01:06:00 PM", "192.0.2.10", 40010, "198.51.100.20", 445, 6),
    RowPlan("quiet", "20/08/2026 01:06:00 PM", "192.0.2.5", 40011, "198.51.100.4", 22, 6),
)


def _profile_values(profile: str) -> dict[str, float]:
    if profile == "zero":
        return {}
    if profile == "quiet":
        return {
            "Flow Duration": 100000,
            "Total Fwd Packet": 2,
            "Total Bwd packets": 2,
            "Total Length of Fwd Packet": 120,
            "Total Length of Bwd Packet": 160,
            "Fwd Packet Length Max": 60,
            "Fwd Packet Length Min": 60,
            "Fwd Packet Length Mean": 60,
            "Bwd Packet Length Max": 80,
            "Bwd Packet Length Min": 80,
            "Bwd Packet Length Mean": 80,
            "Flow Bytes/s": 2800,
            "Flow Packets/s": 40,
            "ACK Flag Count": 1,
            "Average Packet Size": 70,
            "Packet Length Mean": 70,
            "Packet Length Min": 60,
            "Packet Length Max": 80,
        }
    if profile == "syn":
        return {
            "Flow Duration": 10,
            "Total Fwd Packet": 1,
            "Total Bwd packets": 0,
            "Total Length of Fwd Packet": 0,
            "Flow Packets/s": 100000,
            "Fwd Packets/s": 100000,
            "SYN Flag Count": 1,
            "FWD Init Win Bytes": 1024,
            "Fwd Header Length": 20,
            "Fwd Seg Size Min": 20,
        }
    if profile == "bulk":
        return {
            "Flow Duration": 5000000,
            "Total Fwd Packet": 100,
            "Total Bwd packets": 80,
            "Total Length of Fwd Packet": 100000,
            "Total Length of Bwd Packet": 80000,
            "Fwd Packet Length Max": 1500,
            "Fwd Packet Length Min": 40,
            "Fwd Packet Length Mean": 1000,
            "Fwd Packet Length Std": 300,
            "Bwd Packet Length Max": 1500,
            "Bwd Packet Length Min": 40,
            "Bwd Packet Length Mean": 1000,
            "Bwd Packet Length Std": 300,
            "Flow Bytes/s": 36000,
            "Flow Packets/s": 36,
            "ACK Flag Count": 100,
            "Average Packet Size": 1000,
            "Packet Length Mean": 1000,
            "Packet Length Min": 40,
            "Packet Length Max": 1500,
        }
    raise ValueError(f"Unknown synthetic profile: {profile}")


def build_rows(contract: FeatureContract) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, plan in enumerate(ROW_PLANS):
        row: dict[str, Any] = {
            "Flow ID": f"synthetic-v01-{index:02d}-{plan.profile}",
            "Src IP": plan.src_ip,
            "Src Port": plan.src_port,
            "Dst IP": plan.dst_ip,
            "Dst Port": plan.dst_port,
            "Protocol": plan.protocol,
            "Timestamp": plan.timestamp,
            **{feature: 0.0 for feature in contract.baseline_features},
        }
        row.update(_profile_values(plan.profile))
        rows.append(row)
    return rows


def write_fixture(path: Path) -> None:
    contract = FeatureContract.load()
    columns = [
        "Flow ID",
        "Src IP",
        "Src Port",
        "Dst IP",
        "Dst Port",
        "Protocol",
        "Timestamp",
        *contract.baseline_features,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(build_rows(contract))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite fixture: {args.output}")
    write_fixture(args.output)
    print(f"Wrote {len(ROW_PLANS)} fully synthetic rows to {args.output}")


if __name__ == "__main__":
    main()
