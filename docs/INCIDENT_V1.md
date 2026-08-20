# Public incident-v1 contract

`incident-v1` is the default analyst-facing output of Security Anomaly ML v0.1. It contains promoted incidents only; raw flow alerts and non-promoted aggregates are not emitted by the public serializer.

The machine-readable contract is [`contracts/incident-v1.schema.json`](../contracts/incident-v1.schema.json). It is fail-closed with `additionalProperties: false`. Public output never contains `Label`, `label`, `attack_cat`, or an invented attack-family name.

## Policy B semantics

The frozen detector first alerts on flow attack score `>= 0.10`, then groups alert flows by:

`src_ip + dst_ip + dst_port`

Protocol is evidence, not part of that key. Consequently, `protocols` is a sorted, unique integer array. Consecutive alert flows remain in one incident when their gap is at most 300 seconds; only a gap greater than 300 seconds starts another incident. An incident is promoted when its maximum attack score is `>= 0.25`.

## Stable public ID

The serializer creates the identifier from this exact UTF-8 canonical string, with values substituted without added whitespace:

```text
incident-v1|src_ip=<src_ip>|dst_ip=<dst_ip>|dst_port=<integer>|first_seen=<ISO>|last_seen=<ISO>
```

It computes the full lowercase SHA-256 hex digest and prefixes it with `inc_`. Internal incident sequence numbers, source row positions, Python hashes, protocols, and evidence ordering do not participate. Equivalent reordered input therefore has the same public incident ID.

## Time and score semantics

CICFlowMeter timestamps in the frozen input have no timezone or UTC offset. v0.1 preserves that source-capture-defined interpretation and serializes seconds as `YYYY-MM-DDTHH:MM:SS`. It does not append `Z` or claim UTC.

`max_attack_score` and `mean_attack_score` are finite values in `[0, 1]`. The Random Forest score is an attack score, not a calibrated real-world probability.

This release is a research/evaluation prototype and is not production-ready.
