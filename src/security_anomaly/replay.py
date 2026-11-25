"""Transactional replay of a finite set of completed CICFlowMeter rotations."""

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import __version__
from .cli import _load_detector
from .cli import resolve_model_path as _resolve_model_path
from .model_bundle import sha256_file
from .ocsf import incident_finding
from .package_resources import (
    FEATURE_CONTRACT_RESOURCE,
    MODEL_MANIFEST_RESOURCE,
    resource_bytes,
)
from .paths import ensure_distinct
from .serialization import (
    OutputWriteError,
    atomic_text_output,
    jsonl_text,
    promoted_incidents_to_v1,
)
from .timeutils import source_utc, source_zone
from .validation import (
    TIMESTAMP_FORMAT,
    InputContractError,
    read_and_validate_flow_csv_bytes,
)

APPLICATION_ID = 0x53414D4C
LEGACY_FORMAT = "legacy-v1"
OCSF_FORMAT = "ocsf"
SIDECARS = ("-wal", "-shm", "-journal")


@dataclass(frozen=True)
class ReplaySummary:
    status: str
    rotations: int
    rotations_added: int
    rows: int
    rescored: bool


def _discover(directory, checkpoint, output, model):
    if not directory.is_dir():
        raise InputContractError(f"Rotation directory does not exist: {directory}")
    if checkpoint.name.lower().endswith(SIDECARS):
        raise OutputWriteError("Choose a checkpoint distinct from SQLite sidecar names")
    for destination in (checkpoint, output):
        if destination.parent == directory and destination.suffix.lower() == ".csv":
            raise OutputWriteError(
                "Output and checkpoint must use a different path from an input rotation"
            )
    reserved = [checkpoint, *(Path(str(checkpoint) + suffix) for suffix in SIDECARS)]
    ensure_distinct(output, model, *reserved)
    ensure_distinct(model, *reserved)
    paths = sorted(
        p for p in directory.iterdir() if p.suffix.lower() == ".csv" and p.is_file()
    )
    for path in paths:
        if path.is_symlink():
            raise InputContractError(
                f"Publish a completed file in the rotation directory: {path.name}"
            )
        ensure_distinct(path, output, model, *reserved)
    return paths


def _initialize(db, checkpoint):
    # Check ownership while holding the writer lock and before creating tables.
    app_id = db.execute("PRAGMA application_id").fetchone()[0]
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if app_id == APPLICATION_ID and version == 1:
        return
    if (
        app_id
        or version
        or checkpoint.stat().st_size
        or db.execute("SELECT 1 FROM sqlite_master").fetchone()
    ):
        raise InputContractError(
            f"File is not a supported replay checkpoint: {checkpoint}"
        )
    db.execute(
        "CREATE TABLE capture_parts (digest TEXT PRIMARY KEY, csv_bytes BLOB NOT NULL)"
    )
    db.execute(
        "CREATE TABLE replay_rotations (rotation_id TEXT PRIMARY KEY, digest TEXT NOT NULL REFERENCES capture_parts)"
    )
    db.execute(
        "CREATE TABLE snapshot (id INTEGER PRIMARY KEY CHECK (id = 1), binding TEXT NOT NULL,"
        " output TEXT NOT NULL, digest TEXT NOT NULL, row_count INTEGER NOT NULL)"
    )
    db.execute(f"PRAGMA application_id={APPLICATION_ID}")
    db.execute("PRAGMA user_version=1")


def _ingest(db, paths):
    known = dict(db.execute("SELECT rotation_id, digest FROM replay_rotations"))
    added = 0
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise InputContractError(f"Cannot read rotation {path}: {error}") from error
        digest = hashlib.sha256(raw).hexdigest()
        name = path.name.casefold()
        if name in known:
            if known[name] != digest:
                raise InputContractError(f"Accepted rotation was modified: {path.name}")
            continue
        # A renamed or copied file is another name for the same capture bytes.
        inserted = db.execute(
            "INSERT OR IGNORE INTO capture_parts VALUES (?, ?)", (digest, raw)
        )
        added += inserted.rowcount
        db.execute("INSERT INTO replay_rotations VALUES (?, ?)", (name, digest))
        known[name] = digest
    return added


def _score_capture(db, detector, zone, sensor, output_format):
    frames = []
    contract = detector.bundle.contract
    for digest, raw in db.execute(
        "SELECT digest, csv_bytes FROM capture_parts ORDER BY digest"
    ):
        if hashlib.sha256(raw).hexdigest() != digest:
            raise InputContractError(f"Checkpoint contains damaged rotation {digest}")
        batch = read_and_validate_flow_csv_bytes(raw, source=digest, contract=contract)
        frame = batch.frame
        if frame.empty:
            continue
        mapping = {
            stamp: source_utc(stamp.to_pydatetime(), zone)
            for stamp in batch.timestamps.unique()
        }
        utc = batch.timestamps.map(mapping)
        frame["Timestamp"] = utc.dt.strftime(TIMESTAMP_FORMAT)
        frame["_utc"] = utc
        frames.append(frame)
    frame = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=contract.required_input_columns)
    )
    if not frame.empty:
        # Fix peer order across different partition layouts, including optional Flow IDs.
        peers = [
            name for name in sorted(frame.columns) if name not in ("Timestamp", "_utc")
        ]
        frame = frame.sort_values(["_utc", *peers], ignore_index=True).drop(
            columns="_utc"
        )
    result = detector.analyze_batch(frame)
    records = promoted_incidents_to_v1(result)
    if output_format == LEGACY_FORMAT:
        return jsonl_text(records), result.flows_processed
    ids = {
        flow.source_row_id: flow.flow_id
        for flow in result.flow_detections
        if flow.flow_id is not None
    }
    findings = (
        incident_finding(
            record,
            sensor,
            ZoneInfo("UTC"),
            flow_ids=[ids[i] for i in inc.source_row_ids if i in ids],
        )
        for record, inc in zip(records, result.promoted_incidents, strict=True)
    )
    return jsonl_text(findings), result.flows_processed


def replay_directory(
    directory,
    *,
    model,
    sensor,
    timezone,
    checkpoint,
    output,
    output_format=LEGACY_FORMAT,
):
    if not sensor.strip() or sensor != sensor.strip():
        raise InputContractError(
            "Supply a stable non-empty sensor identifier without surrounding spaces"
        )
    if output_format not in (LEGACY_FORMAT, OCSF_FORMAT):
        raise InputContractError(f"Unknown replay output format: {output_format}")
    zone = source_zone(timezone)
    directory = Path(directory).expanduser().resolve()
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    model = _resolve_model_path(Path(model) if model is not None else None)
    paths = _discover(directory, checkpoint, output, model)
    binding = json.dumps(
        {
            "format": output_format,
            "sensor": sensor,
            "timezone": zone.key,
            "directory": str(directory),
            "output": str(output),
            "model_sha256": sha256_file(model),
            "runtime": __version__,
            "feature_contract": hashlib.sha256(
                resource_bytes(FEATURE_CONTRACT_RESOURCE)
            ).hexdigest(),
            "model_manifest": hashlib.sha256(
                resource_bytes(MODEL_MANIFEST_RESOURCE)
            ).hexdigest(),
        },
        sort_keys=True,
    )
    try:
        with closing(
            sqlite3.connect(checkpoint, timeout=30, isolation_level=None)
        ) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("BEGIN IMMEDIATE")
            _initialize(db, checkpoint)
            previous = db.execute("SELECT binding FROM snapshot WHERE id=1").fetchone()
            if previous is not None and previous[0] != binding:
                raise InputContractError(
                    "Replay checkpoint binding differs from the requested capture configuration"
                )
            added = _ingest(db, paths)
            rescored = previous is None or added > 0
            if rescored:
                text, rows = _score_capture(
                    db, _load_detector(model), zone, sensor, output_format
                )
                checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
                db.execute(
                    "INSERT OR REPLACE INTO snapshot VALUES (1, ?, ?, ?, ?)",
                    (binding, text, checksum, rows),
                )
            db.commit()
            # Another writer may have ingested data since this commit. Publish the
            # latest committed snapshot under the same DB's lock, never a stale one.
            db.execute("BEGIN IMMEDIATE")
            text, checksum, rows = db.execute(
                "SELECT output, digest, row_count FROM snapshot WHERE id=1"
            ).fetchone()
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != checksum:
                raise InputContractError("Replay checkpoint snapshot checksum mismatch")
            restore = not output.is_file() or sha256_file(output) != checksum
            if restore:
                with atomic_text_output(output) as destination:
                    destination.write(text)
            rotations = db.execute("SELECT count(*) FROM capture_parts").fetchone()[0]
            db.commit()
    except sqlite3.OperationalError as error:
        raise OutputWriteError(f"Cannot update replay checkpoint: {error}") from error
    except sqlite3.DatabaseError as error:
        raise InputContractError(f"Invalid replay checkpoint: {error}") from error
    status = "replayed" if rescored else ("recovered" if restore else "unchanged")
    return ReplaySummary(status, rotations, added, rows, rescored)


def run_replay(args: argparse.Namespace) -> int:
    result = replay_directory(
        args.directory,
        model=args.model,
        sensor=args.sensor,
        timezone=args.timezone,
        checkpoint=args.checkpoint,
        output=args.output,
        output_format=args.format,
    )
    print(
        json.dumps(
            {
                **asdict(result),
                "sensor": args.sensor,
                "timezone": args.timezone,
                "format": args.format,
            }
        ),
        file=sys.stderr,
    )
    return 0


def register_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "replay", help="replay completed CICFlowMeter rotations from a checkpoint"
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--timezone", required=True, help="IANA source timezone")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--format", choices=(LEGACY_FORMAT, OCSF_FORMAT), default=LEGACY_FORMAT
    )
    parser.set_defaults(handler=run_replay)
