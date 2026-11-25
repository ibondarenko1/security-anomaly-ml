"""Replay a frozen episode-budget evaluation on public IoT-23 Zeek logs.

Use --prepare to inspect feature-only JSONL and the independent truth sidecar.
Use --evaluate to produce a portable aggregate through the source checkout CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from security_anomaly.serialization import atomic_text_output


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "docs" / "benchmarks" / "iot23-periodic-protocol.json"


def download(spec: dict, max_bytes: int) -> bytes:
    request = Request(spec["source_url"], headers={"User-Agent": "security-anomaly-ml/IoT23"})
    with urlopen(request, timeout=60) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes or len(payload) != spec["source_bytes"]:
        raise ValueError(f"{spec['id']}: unexpected source length {len(payload)}")
    if hashlib.sha256(payload).hexdigest() != spec["source_sha256"]:
        raise ValueError(f"{spec['id']}: source SHA-256 differs from the protocol")
    return payload


def convert(value: str, kind: str):
    if value in {"", "-", "(empty)"}:
        return None
    if kind in {"time", "interval", "double", "real"}:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"nonfinite Zeek field {value!r}")
        return number
    if kind in {"port", "count", "int"}:
        return int(value)
    if kind == "bool":
        return {"t": True, "f": False}[value.casefold()]
    if kind.startswith("set["):
        return value.split(",")
    return value


def parse(payload: bytes) -> tuple[list[dict], dict[str, tuple[str, str]]]:
    lines = payload.decode("utf-8").splitlines()
    fields = next(line.split()[1:] for line in lines if line.startswith("#fields"))
    types = next(line.split()[1:] for line in lines if line.startswith("#types"))
    if fields[-2] != "label" or fields[-1] not in {"detailed-label", "det_label"}:
        raise ValueError("expected publisher labels as the last two fields")
    core = fields[:-2]
    rows = []
    labels = {}
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        # Some files pack tunnel_parents and both labels in one final tab field.
        columns = line.split("\t", len(core) - 1)
        tail = columns[-1].split()
        if len(columns) != len(core) or len(tail) != 3:
            raise ValueError(f"malformed labelled Zeek row {number}")
        record = {}
        for field, kind, raw in zip(core, types[:len(core)], [*columns[:-1], tail[0]], strict=True):
            value = convert(raw, kind)
            if value is not None:
                record[field] = value
        uid = record["uid"]
        if not isinstance(uid, str) or not uid or uid in labels:
            raise ValueError(f"empty or duplicate UID on row {number}")
        primary, detail = tail[1].casefold(), tail[2].casefold()
        if primary == "benign":
            label = "benign"
        elif primary == "malicious":
            label = "target" if "c&c" in detail or "heartbeat" in detail else "other"
        else:
            raise ValueError(f"unexpected publisher label on row {number}")
        rows.append(record)
        labels[uid] = label, detail
    rows.sort(key=lambda row: (row["ts"], row["uid"]))
    return rows, labels


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cut = len(rows) * 7 // 10
    while cut and rows[cut - 1]["ts"] == rows[cut]["ts"]:
        cut -= 1
    if not cut:
        raise ValueError("the chronological split needs two distinct time windows")
    return rows[:cut], rows[cut:]


def assign_episodes(rows: list[dict], labels: dict, capture: str, gap_seconds: int) -> dict[str, str]:
    """Label-defined runs span the split and are independent of case grouping."""
    last_by_key = {}
    episodes = {}
    for row in rows:
        label, family = labels[row["uid"]]
        if label != "target":
            continue
        key = (row["id.orig_h"], family)
        previous, episode = last_by_key.get(key, (None, 0))
        if previous is not None and row["ts"] - previous > gap_seconds:
            episode += 1
        last_by_key[key] = row["ts"], episode
        episodes[row["uid"]] = f"{capture}:source={key[0]}:family={family}:episode={episode}"
    return episodes


def write_rows(path: Path, rows) -> str:
    digest = hashlib.sha256()
    with atomic_text_output(path) as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def evaluate(directory: Path, spec: dict, config: dict) -> dict:
    command = [
        sys.executable, "-m", "src.security_anomaly.cli", "evaluate", "zeek",
        str(directory / "holdout.jsonl"), "--reference", str(directory / "reference.jsonl"),
        "--labels", str(directory / "labels.jsonl"), "--sensor", spec["id"],
        "--budget", str(config["budget"]), "--seed", config["seed"],
        "--output", str(directory / "evaluation.json"),
    ]
    for network in spec["local_networks"]:
        command.extend(("--local-net", network))
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads((directory / "evaluation.json").read_bytes())


def public_methods(methods: dict, spillover: set[str]) -> dict:
    result = {}
    for name, metrics in methods.items():
        selected = set(metrics["selected_episode_ids"])
        result[name] = {
            key: value for key, value in metrics.items()
            if key not in {"selected_case_ids", "selected_episode_ids", "method"}
        }
        result[name]["selected_new_episodes"] = len(selected - spillover)
        result[name]["selected_spillover_episodes"] = len(selected & spillover)
    return result


def run_capture(spec: dict, config: dict, root: Path, score: bool) -> dict:
    payload = download(spec, config["max_source_bytes"])
    rows, labels = parse(payload)
    episodes = assign_episodes(rows, labels, spec["id"], config["episode_gap_seconds"])
    reference, holdout = split(rows)
    reference_ids = {episodes[row["uid"]] for row in reference if row["uid"] in episodes}
    holdout_ids = {episodes[row["uid"]] for row in holdout if row["uid"] in episodes}
    spillover = reference_ids & holdout_ids
    directory = root / spec["id"]
    directory.mkdir(parents=True, exist_ok=True)
    truth = []
    for row in holdout:
        uid = row["uid"]
        entry = {"uid": uid, "label": labels[uid][0]}
        if uid in episodes:
            entry["episode"] = episodes[uid]
        truth.append(entry)
    hashes = {
        "reference_sha256": write_rows(directory / "reference.jsonl", reference),
        "holdout_sha256": write_rows(directory / "holdout.jsonl", holdout),
        "labels_sha256": write_rows(directory / "labels.jsonl", truth),
    }

    def period(subset: list[dict], ids: set[str]) -> dict:
        counts = Counter(labels[row["uid"]][0] for row in subset)
        return {
            "rows": len(subset), "first": subset[0]["ts"], "last": subset[-1]["ts"],
            "target_rows": counts["target"], "benign_rows": counts["benign"],
            "other_rows": counts["other"], "target_episodes": len(ids),
        }

    capture = {
        "id": spec["id"], "source_sha256": spec["source_sha256"], "files": hashes,
        "reference": period(reference, reference_ids), "holdout": period(holdout, holdout_ids),
        "new_holdout_episodes": len(holdout_ids - reference_ids),
        "spillover_episodes": len(spillover),
    }
    if score:
        evaluation = evaluate(directory, spec, config)
        comparison = evaluation["comparison"]
        capture["product_version"] = evaluation["product_version"]
        capture["coverage"] = evaluation["coverage"]
        capture["comparison"] = {
            key: value for key, value in comparison.items()
            if key not in {"reviewed_cases", "methods"}
        }
        capture["comparison"]["methods"] = public_methods(comparison["methods"], spillover)
        point = evaluation["operating_point"]
        capture["operating_point"] = {
            "case_count": point["case_count"], "methods": public_methods(point["methods"], spillover)
        }
    return capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", default="all", help="protocol capture ID or all")
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="retain features and labels for inspection")
    mode.add_argument("--evaluate", action="store_true", help="write the aggregate only")
    args = parser.parse_args()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    scenarios = [s for s in protocol["scenarios"] if args.capture in {"all", s["id"]}]
    if not scenarios:
        parser.error(f"unknown capture {args.capture!r}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "protocol": protocol["protocol"], "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "status": "evaluated" if args.evaluate else "prepared",
        "budget": protocol["input_contract"]["budget"], "seed": protocol["input_contract"]["seed"],
    }
    work_dir = TemporaryDirectory() if args.evaluate else nullcontext(str(output))
    with work_dir as directory:
        aggregate["captures"] = [
            run_capture(spec, protocol["input_contract"], Path(directory), args.evaluate)
            for spec in scenarios
        ]
    destination = output / "aggregate.json"
    with atomic_text_output(destination) as handle:
        json.dump(aggregate, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(destination)


if __name__ == "__main__":
    main()