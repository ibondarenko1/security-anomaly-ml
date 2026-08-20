# Security Anomaly ML v0.1 CLI

The CLI accepts unlabeled CICFlowMeter-compatible CSV flows, runs the frozen label-free detector, and writes promoted `incident-v1` JSON Lines. It does not train, tune, download, or modify a model.

## Install

Python 3.13.x is required by the frozen joblib manifest.

```bash
python -m pip install .
security-anomaly --help
security-anomaly version
```

The installed wheel contains the feature contract, model manifest, and public incident schema. The model binary is deliberately separate from Git and is never downloaded automatically.

Runtime package versions are pinned to those in `requirements-runtime.txt`.

## Input

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
