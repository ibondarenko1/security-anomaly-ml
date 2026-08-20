# Privacy and data handling

Security Anomaly ML processes network-flow metadata. Even without packet payloads, this data can be sensitive because it may expose:

- source and destination IP addresses;
- source and destination ports;
- timestamps and communication timing;
- protocol, packet-count, byte-volume, flag, and duration characteristics;
- network topology, host relationships, services, and activity patterns.

## v0.1 processing behavior

Normal `validate`, `analyze`, `version`, and `model-info` execution is local:

- no telemetry is collected or transmitted;
- no input or output is uploaded to a cloud service;
- no model or dependency is downloaded during normal inference;
- no hidden outbound network request is made by the inference path;
- Docker inference is tested with `--network none`;
- the input CSV remains at the location selected by the user and is opened read-only;
- promoted `incident-v1` JSONL is written only to the requested output path;
- output uses a sibling temporary file and atomic replacement, so an existing destination may be replaced only after a complete successful write;
- the run summary is written to stderr and contains counts, versions, elapsed time, and the output path, not raw flow rows.

The repository helper `tools/fetch_frozen_model.py` is an explicit model-acquisition command and does use the network to retrieve the named public GitHub Release asset. It is not called automatically by the installed CLI or Docker inference runtime.

## Memory and temporal state

The CSV is loaded for batch processing. Temporal feature construction uses an in-memory DuckDB connection and in-process data structures. State begins empty for each batch, same-timestamp peers are processed together, and state is not carried into a later CLI run.

Normal inference does not create a persistent runtime database, state store, cache, or telemetry file. Memory and temporary resources disappear when the process exits. The only intended persistent product output is the JSONL destination explicitly supplied by the user.

## Identifiers and model input

Raw source/destination IP addresses and timestamps are not passed to the Random Forest as identity features. They are still used to construct causal context, group incidents, and populate analyst-facing output. Consequently, the output remains sensitive and can reveal internal relationships.

`Label`, `label`, and `attack_cat` are not needed for inference and are removed if present. Users should prefer unlabeled exports and avoid adding unrelated sensitive columns.

## Operator responsibilities

Operators should:

- restrict filesystem and container-volume access to input and output files;
- use encrypted storage and transport appropriate to their environment;
- apply retention/deletion policies to flow data and incidents;
- avoid placing real network metadata in shell history, CI logs, public issues, or support reports;
- sanitize incident examples before sharing them;
- verify the model SHA-256 and published container digest before use;
- understand that host filesystem ownership and permissions govern mounted Docker output.

The application does not implement access control, encryption at rest, secure deletion, data-loss prevention, or a retention service. Those controls remain the operator's responsibility.
