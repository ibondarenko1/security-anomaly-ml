# Independent external-flow evaluation

The independent check uses a labelled day from CSE-CIC-IDS2018, not the
CIC-UNSW-NB15 capture used to train `context-rf-v2`. The official dataset page
describes the processed files as CICFlowMeter-V3 output and links the public
AWS bucket:

- [UNB CSE-CIC-IDS2018 page](https://www.unb.ca/cic/datasets/ids-2018.html)
- [selected public object](https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv)

The frozen source object is `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`,
107,842,858 bytes, SHA-256
`b0534c5d7d8b41e03df71c6966c995d116a8ed28e61f377c8b14cdf5d28f4edf`.
The raw CSV is kept in a temporary directory outside the repository and is
never committed.

## Why the frozen RF is not scored here

The product contract requires exact 76 baseline feature names plus
`Src IP`, `Src Port`, `Dst IP`, `Dst Port`, `Protocol`, and `Timestamp` so that
the 43 causal context features can be built. The selected external CSV has 80
columns, but it omits `Src IP`, `Src Port`, and `Dst IP`; it uses abbreviated
names such as `Tot Fwd Pkts`, `Flow Byts/s`, and `Flow Pkts/s`; it uses the
upstream `CWE Flag Count` spelling for the semantically equivalent CWR flag
(confirmed in CICFlowMeter's `BasicFlow.java`) while the product contract is
`CWR Flag Count`; and
its timestamps use 24-hour `%d/%m/%Y %H:%M:%S`, while the product validator
requires `%d/%m/%Y %I:%M:%S %p`.

The absent endpoint identities prevent the frozen detector from building its
temporal features. Renaming measurement columns or converting timestamp text
cannot recover those identities. The machine-readable audit lists the missing
fields in
[`docs/benchmarks/cic-cse-cic-ids2018-thursday-20261125.json`](benchmarks/cic-cse-cic-ids2018-thursday-20261125.json).

The endpoint-complete February 20 transfer experiment is maintained separately
in the [frozen RF transfer report](RF_TRANSFER.md). That file contains the
endpoint identities and uses verified exporter aliases, including the CWR counter.

## Reproduction

The following downloads the exact public object outside the checkout and runs
the fixed evaluator. No labels or thresholds are selected during the run.

```powershell
$tmp = Join-Path $env:TEMP "security-anomaly-ml-external"
New-Item -ItemType Directory -Force $tmp | Out-Null
$url = "https://cse-cic-ids2018.s3.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
$csv = Join-Path $tmp "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
curl.exe -L --fail $url -o $csv
.venv/Scripts/python.exe tools/evaluate_external_capture.py `
  --csv $csv `
  --output docs/benchmarks/cic-cse-cic-ids2018-thursday-20261125.json
```

The evaluator verifies the object size and SHA-256 by default. `--skip-sha256`
is available only for a rerun when the recorded fingerprint is already known.

## Fixed comparator and denominators

The real-data fallback is a transparent behavioural score, not a retrained
model or a probability:

```text
mean(
  clip(Flow Pkts/s / 1,000, 0, 1),
  clip(Flow Byts/s / 1,000,000, 0, 1),
  clip(SYN Flag Cnt / 3, 0, 1),
  clip(RST Flag Cnt / 3, 0, 1)
)
```

The limits, equal weights, `>= 0.50` alert threshold, and 10,000-flow analyst
budget are fixed constants. The budget view ranks only by this score and breaks
ties by source row order, so every compatible method can be compared using the
same number of reviewed flows. Labels are used only after scoring. Per-class
rates use the number of evaluated rows in that class. Workload rates use the
observed timestamp span, `max(timestamp)-min(timestamp)`, with no padding.

## Measured result

The file contained 331,125 rows. The evaluator removed 25 repeated header rows
and 2,919 rows with non-finite comparator inputs, leaving 328,181 labelled
flows: 235,778 benign and 92,403 `Infilteration` flows. The observed capture
span was 43,199 seconds (11.9997 hours).

| Policy | Benign alerts/reviews | Benign rate | Infilteration alerts/reviews | Infilteration rate | Precision | Recall | Workload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed threshold `>= 0.50` | 3,242 | 1.3750% | 923 | 0.9989% | 22.1609% | 0.9989% | 4,165 / 11.9997 h = 347.09 per hour |
| Same 10,000-flow analyst budget | 7,810 | 3.3124% | 2,190 | 2.3701% | 21.9000% | 2.3701% | 10,000 / 11.9997 h = 833.35 per hour |

The comparator is intentionally reported as a workload baseline. Its low
recall and precision on this day do not establish a claim about the frozen RF,
because the RF could not be fed the external schema honestly.

Synthetic regression coverage for score normalization, deterministic tie
breaking, fixed-budget confusion counts, repeated-header removal, and
non-finite-row exclusion is in
[`tests/test_external_evaluation.py`](../tests/test_external_evaluation.py).
