# Frozen product detector core

`SecurityAnomalyDetector` is the reusable, label-free Python facade for the
complete frozen v0.1 operational path. This is an internal Python contract for
issue #4; it is not the public incident-v1 JSON contract.

## Usage

```python
from src.security_anomaly import SecurityAnomalyDetector

detector = SecurityAnomalyDetector.load(
    model_path="models/context-rf-v2.joblib",
    manifest_path="contracts/model-manifest-context-rf-v2.json",
    feature_contract_path="contracts/feature-contract-cicflow-v2-128.json",
)
result = detector.analyze_batch(unlabeled_cicflowmeter_rows)
```

The facade composes the existing `CausalTemporalFeatureBuilder` and
`FrozenModelBundle`; it does not duplicate feature or model-loading logic.
Every batch starts with empty temporal state.

## Frozen operational path

```text
unlabeled CICFlowMeter-compatible flows
-> input and feature-contract validation
-> causal-temporal-v2 feature builder
-> context-rf-v2 attack score
-> flow alert when score >= 0.10
-> Policy B aggregation by src_ip + dst_ip + dst_port
-> rolling consecutive-alert inactivity gap of 300 seconds
-> promotion when max_attack_score >= 0.25
-> BatchAnalysisResult
```

Only alert flows enter aggregation. Non-alert flows do not create or bridge
incidents. A gap equal to 300 seconds remains in the same incident; only a gap
strictly greater than 300 seconds begins a new one. Protocol is retained as
evidence but is not part of the Policy B grouping key.

The detector refuses a manifest that changes any frozen threshold, operator,
grouping key, window, model version, feature contract, or feature-builder
version.

## Internal result objects

- `FlowDetection` retains source-row provenance, timestamp, optional Flow ID,
  endpoints, ports, protocol, attack score, and alert decision.
- `IncidentDetection` retains deterministic sequence, first/last time, Policy B
  key, observed protocols, score summary, promotion decision, and member source
  rows.
- `BatchAnalysisResult` contains counts, all flow detections, all incidents,
  promoted incidents, frozen versions, and state mode.

Incident ordering and same-timestamp membership use stable source-row
tie-breakers. Repeated analysis of identical input is deterministic. Reordering
input rows preserves the operational result while source-row identifiers still
refer to positions in the caller's submitted batch.

## Full Feb 17 parity

`tools/verify_product_incident_parity.py` supplies 498,890 isolated raw Feb 17
flows after removing `Label` and all evaluation fields. Ground truth never
enters the product detector. The verifier independently aligns raw source rows
to the frozen research artifacts and compares all operational layers.

| Gate | Result |
|---|---:|
| Flows processed | 498,890 |
| Flow alerts at 0.10 | 32,164 |
| Score differences above `1e-12` | 0 |
| Flow decision disagreements | 0 |
| Policy B / 5m incidents | 5,408 |
| Missing or extra incident memberships | 0 |
| Key/timestamp/flow-count disagreements | 0 |
| Max/mean score disagreements above `1e-12` | 0 |
| Promoted incidents at 0.25 | 5,274 |
| Promotion disagreements | 0 |

The maximum absolute flow-score, incident-max-score, and incident-mean-score
differences were each `2.22e-16`. The model was not retrained or retuned; no
threshold or policy changed, and Feb 18 was not accessed.
