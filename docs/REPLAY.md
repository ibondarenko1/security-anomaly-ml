# Replay a finite sensor capture

Point `replay` at a directory of completed CICFlowMeter CSV rotations from one sensor. Give each rotation a unique filename and publish completed files by renaming them into the directory. Create the parent folders for the checkpoint and output first.

```bash
security-anomaly replay rotations/ --sensor edge-a --timezone America/New_York \
  --model models/context-rf-v2.joblib --checkpoint edge-a.sqlite --output incidents.jsonl
```

The default output is the full promoted incident snapshot in `incident-v1` JSONL. Timestamps in that snapshot represent UTC; the legacy schema prints them without an offset. To produce OCSF Detection Findings with sensor-scoped IDs and any supplied Flow IDs, use a dedicated checkpoint and output.

```bash
security-anomaly replay rotations/ --sensor edge-a --timezone America/New_York --format ocsf \
  --model models/context-rf-v2.joblib --checkpoint edge-a-ocsf.sqlite --output findings.jsonl
```

## Rotation and restart behavior

| Input or state change | Result |
|---|---|
| New CSV content, including an earlier timestamp | All accepted flows are scored together. The new snapshot includes the late data. |
| Identical bytes at the same or a new filename | SHA-256 deduplication. An alias becomes an accepted filename for that content. |
| An accepted filename contains different bytes | The run reports an input error and rolls back its pending rotations. Names are compared case-insensitively. |
| An accepted file disappears | Its checkpointed bytes remain part of the capture. |
| Missing or altered output file | The next run restores the latest committed snapshot. With no new content it skips model loading and scoring. |
| Changed sensor, timezone, paths, model hash, runtime or contract | Use a new checkpoint for the new configuration. |

Deduplication is per file. Two different rotations with overlapping rows both contribute those rows. Use disjoint rotations. Differently formatted CSV files representing the same observations have different byte hashes.

The source timezone is explicit. The parser rejects ambiguous or nonexistent wall-clock times during daylight-saving transitions. IP addresses are canonicalized. Temporal features are evaluated in UTC with a deterministic order for equal-time rows, and all peers at one instant are excluded from each other's history. Rebuilding the finite capture preserves the frozen detector's 10-second, 60-second and unbounded history.

The checkpoint stores the raw CSV bytes, including optional input columns, and the committed output. Treat it as sensitive capture data. SQLite serializes writers. After committing a snapshot, the process acquires the writer lock again and atomically publishes the latest committed version. A failed publication can be retried even if the source rotation has already been removed.

## Capacity

New content requires enough memory to score the complete retained capture. Checkpoint size grows with unique input bytes and the current snapshot. Start a separate checkpoint for a new retention period. This command is for finite captures; it reads completed files on each invocation.

One Windows run with Python 3.13.7 used the first 8,820 scorable flows of the [transfer benchmark](RF_TRANSFER.md) as a single rotation (30 source IPs and four destination IPs). The normalized CSV occupied 4,284,112 bytes and the resulting checkpoint 4,313,088 bytes. Cold replay took 31.30 s and reached a peak process working set of 389.92 MiB. An unchanged restart took 0.70 s and peaked at 96.16 MiB. Times include process startup and local I/O. These measurements describe this capture and machine.

The real-model regression compares a single file with split rotations, late arrivals and a process restart. It runs in the frozen-model CI job. [Replay tests](../tests/test_product_replay.py)
