# Security Anomaly ML CLI

| Command | Input and result | First stable release |
|---|---|---|
| `analyze`, `validate`, `version`, `model-info` | Validate CICFlowMeter CSV, apply the frozen model, or inspect its contract. | 0.1.0 |
| `demo zeek` | Export the two installed synthetic connection logs to a new directory. | 0.2.0 |
| `zeek-fit`, `zeek-score` | Fit a sensor profile and write OCSF findings for unusual connections. [Zeek guide](ZEEK.md) | 0.2.0 |
| `zeek-periodic` | Emit OCSF findings for recurring outbound sessions at the rule thresholds. [Rule](EPISODE_EVALUATION.md#what-the-zeek-rule-measures) | 0.2.0 |
| `review zeek`, `review verdict`, `review status` | Rank outbound cases on an unlabeled capture and record local decisions on individual snapshots. | 0.2.0 |
| `evaluate zeek`, `evaluate cic` | Compare detectors at a fixed review budget using an independent UID-to-label sidecar. [Evaluation guide](EPISODE_EVALUATION.md) | 0.2.0 |
| `replay` | Combine bounded CICFlowMeter rotations and resume from local SQLite state. [Replay guide](REPLAY.md) | 0.2.0 |
| `export-ocsf`, `sentinel-batch` | Convert findings to local interchange files. [Sentinel example](../examples/sentinel/README.md) | 0.2.0 |

## Install

Python 3.13.x is required by the frozen joblib manifest.

```bash
python -m pip install security-anomaly-ml==0.2.0
security-anomaly --help
security-anomaly version
```

The wheel, source distribution and checksums are available from the [0.2.0 release](https://github.com/ibondarenko1/security-anomaly-ml/releases/tag/v0.2.0). From a source checkout use `python -m pip install .`. Windows users can also invoke the installed CLI as `python -m security_anomaly.cli`.

The installed wheel contains the feature contract, model manifest, and public incident schema. The model binary is deliberately separate from Git and is never downloaded automatically.

Runtime package versions are pinned to those in `requirements-runtime.txt`.

## Rank an unlabeled Zeek capture

Export the example to a fresh directory. Rerunning the export checks that the two files still match the packaged example. Use a different directory to keep edited examples.

```bash
security-anomaly demo zeek --output-dir zeek-example
security-anomaly review zeek zeek-example/heldout.jsonl --reference zeek-example/fit.jsonl --sensor edge --local-net 192.0.2.0/24 --method zeek_mad --budget 2 --output queue.jsonl
```

Read `queue.jsonl` from top to bottom. Each row contains its rank, logical case ID, snapshot ID, source and destination, times, original Zeek UIDs, score, threshold decision and evidence. The summary on stderr gives the size of the outbound population, how many cases this method could score, how many scored zero and how many were selected. A below-threshold case can enter the queue when its rank fits the budget. Repeating the same input reproduces the order. Ties use the lexical case ID.

Repeat `--local-net` for each sensor-local prefix. Use `zeek_mad` for deviations from the fitted profile, `periodicity_only` for regular starts or `rare_periodic` to also weight destination rarity. A case enters the method's queue when it has a numeric score, including zero. The labelled evaluator uses the common pool of comparable cases across available methods. [Grouping and chronology](EPISODE_EVALUATION.md#what-the-zeek-rule-measures)

## Record a review decision

Copy the exact `snapshot_id` from a row you inspected and use it in place of `SNAPSHOT_ID` below. The local SQLite file stores a verdict and optional note. Values are `benign`, `escalate` and `inconclusive`.

```bash
security-anomaly review verdict queue.jsonl --store verdicts.sqlite --snapshot SNAPSHOT_ID --value escalate --note "Check the service on port 8443"
security-anomaly review status queue.jsonl --store verdicts.sqlite
```

`status` prints one row per case in queue order. A fresh snapshot appears as `unreviewed` with revision 0. The first decision receives revision 1. To amend it, use `--expected-revision 1` with the revised verdict or note. The store preserves prior revisions and an immediate retry of the same decision is idempotent. An extended case gets a new snapshot and needs its own review. A rank change alone preserves the snapshot. Analyst decisions stay local and are separate from the independent labels required by `evaluate`.

## CICFlowMeter input

Required identity/context columns are `Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Protocol`, and `Timestamp`, plus all 76 baseline fields listed by `cicflow-v2-128`. `Flow ID` is optional. `Label`, `label`, and `attack_cat` are ignored and removed; no label is required or consumed.

Validation rejects all missing columns together, duplicate raw headers, reserved/derived fields, incompatible extras, malformed timestamps, non-numeric or non-finite model values, non-integral/out-of-range ports, and protocols outside `[0, 255]`. A header-only file is a valid empty batch. A completely empty file is not.

The timestamp format is the CICFlowMeter source format, for example `17/02/2015 08:00:00 PM`. Its timezone meaning is defined by the source capture; v0.1 does not infer UTC.

## Commands

Validate without loading a model:

```bash
security-anomaly validate flows.csv
```

Analyze and atomically write promoted incidents:

```bash
security-anomaly analyze flows.csv \
  --model C:/models/context-rf-v2.joblib \
  --output incidents.jsonl
```

The run summary goes to stderr and cannot corrupt JSONL. A failed write never leaves a partial destination that appears successful. Header-only valid input produces an empty JSONL file with exit code 0.

Show public versions without a model:

```bash
security-anomaly version
```

Verify model checksum, runtime compatibility, contract, and frozen policy:

```bash
security-anomaly model-info --model C:/models/context-rf-v2.joblib
```

Model resolution order is:

1. `--model PATH`
2. `SECURITY_ANOMALY_MODEL`
3. `models/context-rf-v2.joblib`, `models/v2_context_random_forest.joblib`, or `context-rf-v2.joblib` below the current directory
4. a clear missing-model error

No network download occurs.

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success |
| 2 | Command or argument usage error |
| 3 | Input/schema validation failure |
| 4 | Model missing, integrity failure, or compatibility failure |
| 5 | Serialization or atomic output failure |
| 10 | Unexpected runtime failure |

Tracebacks are hidden by default. Put global `--debug` before the subcommand to show a traceback for an unexpected failure.

The exact public JSON object, deterministic ID algorithm, timestamp limitation, and Policy B semantics are documented in [`INCIDENT_V1.md`](INCIDENT_V1.md).

This v0.1 path is a research/evaluation prototype and is not production-ready.
