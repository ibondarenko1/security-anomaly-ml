# Frozen model bundle

This document covers the frozen v2 Context Random Forest artifact and the first
label-free inference component. It does not define a CLI, API, container, or a
new training procedure.

## Frozen decisions

The production candidate remains unchanged:

- model: Context Random Forest v2;
- inputs: 76 CICFlowMeter fields, 9 static derived fields, and 43 causal temporal
  fields, in the exact 128-field order in the feature contract;
- flow threshold: `0.10`;
- incident key: source IP, destination IP, and destination port;
- incident window: 300 seconds;
- promotion threshold: maximum flow score of at least `0.25`.

The model output is an attack score. It must not be described as a calibrated
real-world probability.

## Release assets

The model binary is deliberately excluded from Git. A release bundle contains:

- `context-rf-v2.joblib`;
- `model-manifest-context-rf-v2.json`;
- `feature-contract-cicflow-v2-128.json`;
- `SHA256SUMS.json`.

The tracked manifest records the exact model SHA-256, byte size, estimator
parameters, runtime versions, contract hash, and frozen operating decisions.
The loader verifies the model checksum and size before deserialization, then
checks the runtime, estimator type, feature count, classes, and parameters.

Regenerate the tracked metadata from the frozen local research artifacts:

```powershell
.\.venv\Scripts\python.exe tools\export_frozen_bundle_contracts.py
```

Prepare files for a GitHub Release without retraining:

```powershell
.\.venv\Scripts\python.exe tools\package_frozen_model.py `
  --output-dir models\release-context-rf-v2
```

The output directory must not already exist. This prevents a previous release
bundle from being overwritten silently.

## Runtime contract

Use Python 3.13 and install the exact versions in `requirements-runtime.txt`.
The frozen artifact currently requires:

- NumPy 2.5.2;
- pandas 3.0.5;
- scikit-learn 1.9.0;
- joblib 1.5.3;
- DuckDB 1.5.5.

Inspect and validate a bundle before integrating it:

```powershell
.\.venv\Scripts\python.exe -m src.security_anomaly.model_bundle `
  --model models\v2_context_random_forest.joblib `
  --manifest contracts\model-manifest-context-rf-v2.json `
  --contract contracts\feature-contract-cicflow-v2-128.json
```

## Label-free temporal inference

`CausalTemporalFeatureBuilder` accepts CICFlowMeter-compatible rows without
`Label` or `Attack` fields. Each batch starts with empty state, is ordered by
`Timestamp`, and computes a timestamp group's features only from rows with an
earlier timestamp. Rows sharing the same timestamp cannot see one another.
Only after a complete timestamp group is scored may all rows in that group be
considered part of subsequent state.

Raw identifiers are retained only as output identity metadata. `Flow ID`, source
IP, destination IP, and timestamp never enter the 128-field model matrix.

The initial stable Python surface is:

```python
from src.security_anomaly import CausalTemporalFeatureBuilder, FrozenModelBundle

batch = CausalTemporalFeatureBuilder.from_contract_file(
    "contracts/feature-contract-cicflow-v2-128.json"
).build(flows)

bundle = FrozenModelBundle.load(
    model_path="models/v2_context_random_forest.joblib",
    manifest_path="contracts/model-manifest-context-rf-v2.json",
    feature_contract_path="contracts/feature-contract-cicflow-v2-128.json",
)
scores = batch.scores_in_source_order(bundle.predict_scores(batch.to_numpy()))
```

Input contract violations fail explicitly: missing required columns, invalid
timestamps, non-finite numeric values, invalid ports, unsupported protocols, or
feature-order mismatches are not silently repaired.

## Parity gates

Two separate checks cover different failure modes. Neither check reads the
locked Feb 18 holdout or fits, retrains, or retunes a model.

### Model-bundle serialization parity

`verify_frozen_score_parity.py` starts from the already-generated frozen Feb 17
research matrix. It proves that checksum/contract validation, joblib loading,
and scoring preserve the stored scores:

```text
frozen validation_features.parquet -> FrozenModelBundle -> scores
```

```powershell
.\.venv\Scripts\python.exe tools\verify_frozen_score_parity.py
```

All 498,890 rows agree: the maximum absolute score difference is
`2.22e-16`, no difference exceeds `1e-12`, and there are zero decision
disagreements at threshold `0.10`.

### Full product-pipeline parity

`verify_product_pipeline_parity.py` starts from an isolated raw Feb 17
CICFlowMeter CSV. It refuses a file containing any other date, removes all
ground-truth/evaluation fields before the product builder is called, and runs:

```text
raw unlabeled Feb 17 flows
-> CausalTemporalFeatureBuilder
-> ordered cicflow-v2-128 matrix
-> FrozenModelBundle
-> scores
```

The verification dataset is not distributed in Git. Supply the existing
Feb-17-only raw CSV; do not pass the combined multi-day corpus:

```powershell
.\.venv\Scripts\python.exe tools\verify_product_pipeline_parity.py `
  --raw-feb17 data\processed\cic_unsw_nb15_v2\_product_parity_cache\feb17_raw.csv
```

Row order inside a timestamp is not assumed. The tool assigns every isolated
raw row a stable source-row identifier, independently regenerates the frozen
research path, fingerprints all 128 ordered values with SHA-256, verifies all
128 joined values directly, and maps to the physical frozen Parquet row.
Identical complete-feature duplicates receive deterministic occurrence ranks;
because every one of their 128 inputs is identical, their ordering is feature-
and score-equivalent.

The full-scale Feb 17 result is exact at the contract's float32 model-matrix
boundary:

| Check | Result |
|---|---:|
| Raw product rows | 498,890 |
| Ground-truth fields passed to builder | 0 |
| Features per row | 128 |
| Mismatched rows | 0 |
| Mismatched cells | 0 |
| Maximum absolute feature difference | 0.0 |
| Temporal features matching | 43 / 43 |
| Feature order matches `cicflow-v2-128` | Yes |
| Rows scored | 498,890 |
| Maximum absolute score difference | `2.22e-16` |
| Score differences above `1e-12` | 0 |
| Threshold `0.10` disagreements | 0 |

The feature check uses zero absolute and relative tolerance after conversion to
the explicitly frozen float32 model-input dtype. Per-feature mismatch counts
and maximum differences are included in the tool's JSON report.
