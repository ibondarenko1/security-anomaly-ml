"""Replay a bounded, published IoT-23 Zeek TSV slice through the local CLI.

The source is downloaded into a temporary directory, converted field-for-field
to JSONL, split chronologically, and removed on exit.  ``-`` stays omitted;
the converter never invents a missing counter.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_URL = (
    "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/"
    "CTU-Honeypot-Capture-4-1/bro/conn.log"
)
SOURCE_SHA256 = "ef21b8f322caf62a35fd7a8ebf4e6864b275574eadead425f702772470becca0"
FIT_ROWS = 320
PROFILE_FIELDS = (
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "missed_bytes",
)


def _convert(value: str, kind: str) -> Any:
    if value in {"-", "(empty)"}:
        return None
    if kind in {"time", "interval"}:
        return float(value)
    if kind in {"port", "count"}:
        return int(value)
    if kind == "bool":
        return value.lower() in {"t", "true", "1"}
    if kind.startswith("set["):
        return value.split(",") if value else []
    return value


def _parse_tsv(payload: bytes) -> list[dict[str, Any]]:
    lines = payload.decode("utf-8").splitlines()
    fields = next(line.split("\t")[1:] for line in lines if line.startswith("#fields\t"))
    kinds = next(line.split("\t")[1:] for line in lines if line.startswith("#types\t"))
    if len(fields) != len(kinds):
        raise ValueError("Zeek fields/types headers differ")
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        values = line.split("\t")
        if len(values) != len(fields):
            raise ValueError("Zeek data row does not match its header")
        record: dict[str, Any] = {}
        for field, kind, value in zip(fields, kinds, values):
            converted = _convert(value, kind)
            if converted is not None:
                record[field] = converted
        records.append(record)
    records.sort(key=lambda record: record["ts"])
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    data = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _summary(stderr: str) -> dict[str, Any]:
    for line in reversed(stderr.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"CLI summary missing: {stderr}")


def _run(*args: str) -> tuple[dict[str, Any], str]:
    process = subprocess.run(
        [sys.executable, "-m", "src.security_anomaly.cli", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(args)}\n{process.stderr}"
        )
    return _summary(process.stderr), process.stderr


def main() -> int:
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        source = response.read()
    source_sha256 = hashlib.sha256(source).hexdigest()
    if source_sha256 != SOURCE_SHA256:
        raise RuntimeError(
            f"published source changed: expected {SOURCE_SHA256}, got {source_sha256}"
        )
    records = _parse_tsv(source)
    fit = records[:FIT_ROWS]
    holdout = records[FIT_ROWS:]
    if not fit or not holdout or holdout[0]["ts"] <= fit[-1]["ts"]:
        raise RuntimeError("chronological fit/holdout split is not strictly ordered")
    incomplete = [record for record in holdout if any(field not in record for field in PROFILE_FIELDS)]
    if len(incomplete) != 1 or "orig_bytes" in incomplete[0] or "resp_bytes" in incomplete[0]:
        raise RuntimeError("published missing counters were not preserved as incomplete")
    with tempfile.TemporaryDirectory(prefix="zeek-iot23-check-") as directory:
        root = Path(directory)
        fit_path = root / "fit.jsonl"
        holdout_path = root / "holdout.jsonl"
        profile_path = root / "profile.json"
        output_path = root / "findings.jsonl"
        fit_sha256 = _write_jsonl(fit_path, fit)
        holdout_sha256 = _write_jsonl(holdout_path, holdout)
        fit_summary, _ = _run(
            "zeek-fit",
            str(fit_path),
            "--sensor",
            "iot23-capture-4",
            "--output",
            str(profile_path),
        )
        profile_before = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        score_summary, _ = _run(
            "zeek-score",
            str(holdout_path),
            "--profile",
            str(profile_path),
            "--sensor",
            "iot23-capture-4",
            "--output",
            str(output_path),
        )
        profile_after = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        if profile_before != profile_after:
            raise RuntimeError("profile changed during scoring")
    result = {
        "source_url": SOURCE_URL,
        "source_sha256": source_sha256,
        "source_bytes": len(source),
        "records_total": len(records),
        "fit_records_input": len(fit),
        "holdout_records_input": len(holdout),
        "holdout_incomplete_profile_records": len(incomplete),
        "fit_first_ts": fit[0]["ts"],
        "fit_last_ts": fit[-1]["ts"],
        "holdout_first_ts": holdout[0]["ts"],
        "holdout_last_ts": holdout[-1]["ts"],
        "fit_jsonl_sha256": fit_sha256,
        "holdout_jsonl_sha256": holdout_sha256,
        "profile_sha256_before_score": profile_before,
        "profile_sha256_after_score": profile_after,
        "fit_summary": fit_summary,
        "score_summary": score_summary,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
