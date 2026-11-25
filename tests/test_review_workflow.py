from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from src.security_anomaly.cli import main


def export(directory):
    return main(["demo", "zeek", "--output-dir", str(directory)])


def test_installed_demo_matches_both_fixtures_and_preserves_edits(tmp_path):
    directory = tmp_path / "sample"
    assert export(directory) == export(directory) == 0
    for name in ("fit.jsonl", "heldout.jsonl"):
        fixture = Path(__file__).parent / "fixtures" / "zeek" / name
        assert (directory / name).read_text() == fixture.read_text()
    edited = directory / "heldout.jsonl"
    edited.write_bytes(b"my capture\n")
    assert export(directory) == 5
    assert edited.read_bytes() == b"my capture\n"


@pytest.fixture
def review(tmp_path):
    demo = tmp_path / "sample"
    assert export(demo) == 0
    queue = tmp_path / "queue.jsonl"
    assert main(["review", "zeek", str(demo / "heldout.jsonl"), "--reference", str(demo / "fit.jsonl"),
                 "--sensor", "edge", "--local-net", "192.0.2.0/24", "--budget", "2",
                 "--output", str(queue)]) == 0
    rows = [json.loads(line) for line in queue.read_text().splitlines()]
    assert rows[0]["uids"] == ["H2"]
    store = tmp_path / "decisions.sqlite"
    record = ["review", "verdict", str(queue), "--store", str(store), "--snapshot", rows[0]["snapshot_id"]]
    return queue, rows, store, record


def test_verdict_revision_retry_and_unreviewed_status(review, capsys):
    queue, _, store, record = review
    assert main(record + ["--value", "escalate", "--note", "port 8443"]) == 0
    assert main(record + ["--value", "escalate", "--note", "port 8443"]) == 0
    assert main(record + ["--value", "benign"]) == 3
    assert main(record + ["--value", "benign", "--expected-revision", "1"]) == 0
    assert main(record + ["--value", "escalate", "--note", "port 8443"]) == 3
    capsys.readouterr()
    assert main(["review", "status", str(queue), "--store", str(store)]) == 0
    statuses = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [(s["verdict"], s["revision"]) for s in statuses] == [("benign", 2), ("unreviewed", 0)]
    with closing(sqlite3.connect(store)) as db:
        assert db.execute("SELECT note FROM verdicts ORDER BY revision").fetchall() == [("port 8443",), ("",)]


def test_modified_evidence_and_unrelated_database_are_preserved(review):
    queue, rows, store, record = review
    with closing(sqlite3.connect(store)) as db, db:
        db.execute("CREATE TABLE mydata (value TEXT)")
    assert main(record + ["--value", "benign"]) == 3
    with closing(sqlite3.connect(store)) as db:
        assert db.execute("SELECT name FROM sqlite_schema").fetchall() == [("mydata",)]
    store.unlink()
    rows[-1]["endpoints"]["destination_port"] = 1
    queue.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert main(record + ["--value", "benign"]) == 3
    assert not store.exists()


def test_same_verdict_from_concurrent_writers_is_recorded_once(review):
    _, _, store, record = review
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(main, [record + ["--value", "inconclusive"]] * 2)) == [0, 0]
    with closing(sqlite3.connect(store)) as db:
        assert db.execute("SELECT COUNT(*), MAX(revision) FROM verdicts").fetchone() == (1, 1)
