"""Record local analyst decisions against the exact Zeek case that was reviewed."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .paths import ensure_distinct
from .review import SCHEMA_VERSION, snapshot_fingerprint
from .serialization import OutputWriteError
from .validation import InputContractError


_APPLICATION_ID = 0x53415256  # SARV, the local review-verdict database
_VALUES = ("benign", "escalate", "inconclusive")


def _queue_rows(path: Path) -> Iterator[dict[str, Any]]:
    seen: set[str] = set()
    try:
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or row.get("schema_version") != SCHEMA_VERSION:
                        raise ValueError(f"expected {SCHEMA_VERSION}")
                    if type(row.get("rank")) is not int or row["rank"] != len(seen) + 1:
                        raise ValueError("expected consecutive ranks starting at 1")
                    case_id = row.get("case_id")
                    if not isinstance(case_id, str) or not case_id or case_id in seen:
                        raise ValueError("case ID must be a unique non-empty string")
                    if row.get("snapshot_id") != snapshot_fingerprint(row):
                        raise ValueError("snapshot fingerprint does not match the case evidence")
                    seen.add(case_id)
                except (TypeError, ValueError, KeyError) as error:
                    raise InputContractError(f"Invalid review queue at line {line_no}: {error}") from error
                yield row
    except (OSError, UnicodeError) as error:
        raise InputContractError(f"Cannot read review queue {path}: {error}") from error


@contextmanager
def _store(path: Path, *, write: bool) -> Iterator[sqlite3.Connection]:
    mode = "rwc" if write else "ro"
    uri = path.expanduser().resolve().as_uri() + f"?mode={mode}"
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=15)) as db, db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            app_id = db.execute("PRAGMA application_id").fetchone()[0]
            if write and app_id == 0 and not db.execute("SELECT 1 FROM sqlite_schema LIMIT 1").fetchone():
                db.execute("""CREATE TABLE verdicts (
                    snapshot_id TEXT NOT NULL, revision INTEGER NOT NULL, case_id TEXT NOT NULL,
                    verdict TEXT NOT NULL, note TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, revision)
                ) WITHOUT ROWID""")
                db.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
                db.execute("PRAGMA user_version = 1")
            elif app_id != _APPLICATION_ID or db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise InputContractError(f"Expected a version 1 analyst verdict database at {path}")
            yield db
    except sqlite3.Error as error:
        error_type = OutputWriteError if write else InputContractError
        raise error_type(f"Cannot {'update' if write else 'read'} verdict database {path}: {error}") from error


def _latest(db: sqlite3.Connection, snapshot_id: str) -> dict[str, Any]:
    row = db.execute(
        "SELECT revision, verdict, note, recorded_at FROM verdicts "
        "WHERE snapshot_id = ? ORDER BY revision DESC LIMIT 1", (snapshot_id,),
    ).fetchone()
    if row is None:
        return {"revision": 0, "verdict": "unreviewed", "note": "", "recorded_at": None}
    return dict(row)


def _record(args: argparse.Namespace) -> int:
    ensure_distinct(args.store, args.queue)
    if args.expected_revision < 0:
        raise InputContractError("expected revision must be non-negative")
    # Validate the entire queue before opening the database for writing.
    match = None
    for row in _queue_rows(args.queue):
        if row["snapshot_id"] == args.snapshot:
            match = row
    if match is None:
        raise InputContractError("snapshot does not occur in this queue")
    with _store(args.store, write=True) as db:
        current = _latest(db, args.snapshot)
        revision = current["revision"]
        identical = current["verdict"] == args.value and current["note"] == args.note
        if identical and revision in (args.expected_revision, args.expected_revision + 1):
            pass  # Replay of a saved decision.
        elif revision != args.expected_revision:
            raise InputContractError(f"Stored revision is {revision}; inspect review status before amending")
        else:
            db.execute(
                "INSERT INTO verdicts (snapshot_id, revision, case_id, verdict, note, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (args.snapshot, revision + 1, match["case_id"], args.value, args.note,
                 datetime.now(UTC).isoformat()),
            )
            current = _latest(db, args.snapshot)
    print(json.dumps({"case_id": match["case_id"], "snapshot_id": args.snapshot, **current}, ensure_ascii=False))
    return 0


def _status(args: argparse.Namespace) -> int:
    ensure_distinct(args.store, args.queue)
    # Materialize a validated view before writing any JSONL to stdout.
    with _store(args.store, write=False) as db:
        rows = [
            {"rank": case["rank"], "case_id": case["case_id"], "snapshot_id": case["snapshot_id"],
             **_latest(db, case["snapshot_id"])}
            for case in _queue_rows(args.queue)
        ]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return 0


def register_verdict_parser(commands: argparse._SubParsersAction) -> None:
    record = commands.add_parser("verdict", help="record or amend a decision for an exact case snapshot")
    status = commands.add_parser("status", help="print the latest verdict for each queued snapshot")
    for command in (record, status):
        command.add_argument("queue", type=Path, help="JSONL from review zeek")
        command.add_argument("--store", required=True, type=Path, help="local SQLite verdict database")
    record.add_argument("--snapshot", required=True, help="snapshot_id from the queue")
    record.add_argument("--value", required=True, choices=_VALUES)
    record.add_argument("--note", default="", help="analyst comment")
    record.add_argument("--expected-revision", type=int, default=0, help="current revision when amending")
    record.set_defaults(handler=_record)
    status.set_defaults(handler=_status)
