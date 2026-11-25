"""Check whether a fixed slice of labelled UWF Zeek data can measure outbound recall."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import ipaddress
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from urllib.request import urlopen

from security_anomaly import zeek
from security_anomaly.serialization import atomic_text_output


CONTRACT = Path(__file__).resolve().parents[1] / "docs/benchmarks/zeek-scale-uwf-protocol.json"


def measure(contract: dict) -> dict:
    records, sources = [], {}
    uids = set()
    for source in contract["sources"]:
        with urlopen(source["url"], timeout=60) as response:
            payload = response.read(source["bytes"] + 1)
        if len(payload) != source["bytes"] or hashlib.sha256(payload).hexdigest() != source["sha256"]:
            raise ValueError(f"Source bytes differ from the selection contract for {source['id']}")
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig"), newline=""))
        scanned = complete = 0
        for row in islice(reader, contract["rows_per_file"]):
            scanned += 1
            uid = row["uid"]
            if not uid or uid in uids:
                raise ValueError(f"Missing or repeated connection UID in {source['id']}, row {scanned}")
            uids.add(uid)
            try:
                record = zeek._parse_record({
                    "uid": uid, "ts": row["ts"], "proto": row["proto"],
                    "id.orig_h": row["src_ip_zeek"], "id.resp_h": row["dest_ip_zeek"],
                    "id.orig_p": int(row["src_port_zeek"]), "id.resp_p": int(row["dest_port_zeek"]),
                    **{name: (float(row[name]) if name == "duration" else int(row[name]))
                       for name in zeek.PROFILE_FEATURES if row[name]},
                })
            except (ValueError, zeek.ZeekInputError):
                continue
            if record.complete:
                complete += 1
                records.append((record.timestamp, uid, record.src_ip, record.dst_ip, source["label"]))
        sources[source["id"]] = {"rows_read": scanned, "complete": complete}

    records.sort()
    cut = len(records) * contract["reference_percent"] // 100
    reference, observed = records[:cut], records[cut:]
    if not reference or not observed or reference[-1][0] >= observed[0][0]:
        raise ValueError("Reference and observed periods must be nonempty and strictly chronological")
    net = ipaddress.ip_network(contract["comparison_prefix"])
    outbound = [r for r in observed if ipaddress.ip_address(r[2]) in net and ipaddress.ip_address(r[3]) not in net]

    def period(rows):
        counts = Counter(row[4] for row in rows)
        return {"rows": len(rows), "target": counts["target"], "benign": counts["benign"],
                "first_utc": rows[0][0].isoformat(), "last_utc": rows[-1][0].isoformat()}

    return {"dataset": contract["dataset"], "sources": sources, "distinct_input_uids": len(uids),
            "complete_rows": len(records), "reference": period(reference), "observed": period(observed),
            "outbound_rows_for_comparison_prefix": len(outbound),
            "outbound_target_rows_for_comparison_prefix": sum(row[4] == "target" for row in outbound)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="local JSON report")
    args = parser.parse_args()
    result = measure(json.loads(CONTRACT.read_bytes()))
    with atomic_text_output(args.output) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
