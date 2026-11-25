# Outbound Zeek cases and detector evaluation

`zeek-periodic` groups recurring connections to an uncommon destination into one OCSF Detection Finding. `evaluate` compares the cases an analyst would open at a fixed review budget. Both commands run locally with the source installation.

## Try a recurring update and a simulated callback

Run these commands from the repository root. When using your own traffic, select an earlier reference period representative of the sensor's usual activity. Include every local prefix with a separate `--local-net` argument.

```bash
f=tests/fixtures/periodic
security-anomaly zeek-periodic "$f/observed.jsonl" --reference "$f/reference.jsonl" \
  --sensor edge-a --local-net 192.0.2.0/24 --output findings.jsonl
security-anomaly evaluate zeek "$f/observed.jsonl" --reference "$f/reference.jsonl" \
  --sensor edge-a --local-net 192.0.2.0/24 --labels "$f/labels.jsonl" \
  --budget 2 --output evaluation.json
security-anomaly sentinel-batch findings.jsonl --output request.json
```

The fixture contains a known update server, a new update mirror, a simulated callback and a one-off connection. The detector emits two findings. Four `callback-*` UIDs form one and four `new-mirror-*` UIDs form the other. The latter is a benign control. A new update server can look as regular and unfamiliar as a callback. Look up the destination and original connection UIDs before assigning an incident verdict.

## What the Zeek rule measures

- The origin must fall inside a supplied local network and the destination outside every supplied prefix. The detector excludes multicast, unspecified, loopback, link-local and IPv4 limited-broadcast destinations.
- Cases group one source IP, destination IP, destination port and transport protocol. A gap exceeding 30 minutes starts a new case. Equal-time observations count individually as evidence and share a single start instant for interval calculations.
- With four or more distinct starts, regularity is `1 / (1 + mean absolute deviation from the median interval / median interval)`. Rarity is `1 - (reference connections from that origin to that destination IP / all outbound reference connections from that origin)`. Fewer than three eligible reference connections for the origin give rarity zero.
- A finding requires both scores to reach 0.80. The ranking score is their product. The evidence contains observed intervals, reference counts, endpoints and member UIDs. The logical case ID survives an extension at the end of the session. Earlier arrivals that change the first observation create a new ID.
- The last reference start must precede the first observed start. UIDs must be disjoint between captures. Missing byte or packet counters still allow timing analysis when the connection identity is valid. Parser and coverage counts are written to stderr.

## Give the evaluator independent truth

Use UTF-8 JSON Lines with one row per accepted observation in the holdout, including inbound and local traffic. Use the same independent episode ID for separate connections or review cases attributable to one incident. A labelled case can cover multiple independent episodes. A case with both target and benign connections counts as a target case. A case containing only benign observations counts toward benign workload.

```jsonl
{"uid":"callback-0","label":"target","episode":"simulated-callback"}
{"uid":"update-0","label":"benign"}
{"uid":"unrelated-observation","label":"other"}
```

`other` denotes known activity outside the evaluation target. A missing UID, duplicate UID or extra sidecar row fails the comparison. Fit data and labels stay in separate files. Thresholds and queue order depend only on the features, reference period, seed and case IDs.

## Read the JSON report

`comparison.methods` reports reviewed cases, distinct target episodes recovered, case precision (`selected_target_rate`), episode recall, benign cases, zero-score fills and the selected case IDs. Every available method ranks the same cases at `min(budget, common pool size)`. A smaller budget models a shorter queue. A zero score can occupy a slot in this ranking exercise.

The Zeek methods rank by rarity × regularity, regularity alone, the largest available connection-level deviation from the existing median/MAD profile, and a seeded uniform selection. The MAD score needs at least one member with complete measurements. The report lists exclusions from the common pool. A method with no usable scores has status `unavailable` and selects no cases. Changing the seed changes the random control. Numeric ties use the stable case ID.

`comparison.reviewed_cases` maps selected case IDs back to member UIDs, scores and retrospective truth. `comparison.uniform_expectation` gives the expected number of target cases and distinct target episodes for uniform sampling without replacement. `operating_point` shows what each real threshold actually emits, including empty queues. Episode recall uses all target episodes in the accepted holdout, including episodes outside the rankable pool. Undefined rates are JSON `null`. Run separate comparisons for separate sensors and review periods to avoid pooling a quiet device with a busy one.

## Evaluate a frozen Random Forest on CICFlowMeter CSV

```bash
security-anomaly evaluate cic flows.csv --model models/context-rf-v2.joblib \
  --labels truth.jsonl --budget 10 --output rf-evaluation.json
```

By default each sidecar UID must exactly match a unique textual `Flow ID`. For files with missing or repeated exporter IDs, use `--uid-mode row`. Its UID format is `<full SHA-256 of the input CSV bytes>:<1-based data row>`. This binds positional truth to the exact source file. The report records the CSV and model digests.

All flows are grouped by the frozen Policy B endpoint key and five-minute inactivity gap for the fixed-budget comparison. Methods rank each group by its maximum RF attack score, its maximum per-flow total of forward and backward packets, or the same seeded uniform control. The operational section reproduces actual RF promotion, including the original flow alert filter. Its target-case recall is conditional on RF alert groups. Independent episode recall still uses the full sidecar. Filtering before grouping can change group boundaries. Running on a single evaluation file starts the frozen causal context empty.

## Public labelled captures

The [frozen IoT-23 protocol](benchmarks/iot23-periodic-protocol.json) specifies the datasets, warmup, target labels, episode boundaries, thresholds and review budget used for the public comparison.
