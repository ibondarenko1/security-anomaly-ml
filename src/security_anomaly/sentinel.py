"""Prepare one Azure Monitor Logs Ingestion request body on local disk."""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

from .ocsf import EPOCH, OCSF_VERSION
from .paths import ensure_distinct
from .serialization import atomic_text_output
from .validation import InputContractError

# Leave headroom below the API's 1 MB maximum for compressed and uncompressed data.
MAX_BATCH_BYTES = 1_000_000


def _request_rows(path: Path):
    try:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                try:
                    event = json.loads(line)
                    if (
                        event["class_uid"] != 2004
                        or event["metadata"]["version"] != OCSF_VERSION
                        or type(event["time"]) is not int
                        or not event["finding_info"]["uid"]
                    ):
                        raise ValueError(
                            "Expected a timestamped OCSF 1.6 Detection Finding"
                        )
                    stamp = EPOCH + timedelta(milliseconds=event["time"])
                    row = {
                        "TimeGenerated": stamp.isoformat(timespec="milliseconds"),
                        "Finding": event,
                    }
                    yield json.dumps(
                        row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    )
                except (KeyError, ValueError, TypeError, OverflowError) as error:
                    raise InputContractError(
                        f"Invalid finding on line {number}: {error}"
                    ) from error
    except (OSError, UnicodeError) as error:
        raise InputContractError(f"Cannot read findings: {error}") from error


def _command_batch(args: argparse.Namespace) -> int:
    ensure_distinct(args.output, args.input)
    count = 0
    size = len(b"[]\n")
    with atomic_text_output(args.output) as output:
        output.write("[")
        for encoded in _request_rows(args.input):
            size += len(encoded.encode("utf-8")) + int(count > 0)
            if size > MAX_BATCH_BYTES:
                raise InputContractError(
                    "Request exceeds 1,000,000 bytes; split the JSONL at a line boundary"
                )
            if count:
                output.write(",")
            output.write(encoded)
            count += 1
        output.write("]\n")
    print(json.dumps({"records": count, "bytes": size}), file=sys.stderr)
    return 0


def register_parser(subparsers) -> None:
    command = subparsers.add_parser(
        "sentinel-batch", help="prepare a local Azure Monitor request body"
    )
    command.add_argument("input", type=Path, help="OCSF finding JSONL")
    command.add_argument("--output", type=Path, required=True, help="UTF-8 JSON array")
    command.set_defaults(handler=_command_batch)
