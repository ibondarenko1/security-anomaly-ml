# Frozen Random Forest on CSE-CIC-IDS2018

The published model flagged 495 of 8,754 attack-labelled flows and 14,710 of 23,302 benign flows in four samples from the February 20 capture. At its original flow threshold of 0.10, recall was **5.65%**, the benign false-positive rate was **63.13%**, and flow precision was **3.26%**. The 2015 holdout numbers do not transfer to these samples. [Source dataset](https://www.unb.ca/cic/datasets/ids-2018.html)

| Measured on four slices | Count |
|---|---:|
| Complete source rows | 32,403 |
| Rows meeting the measurement contract | 32,056 |
| Excluded rows | 347 |
| Flow alerts at 0.10 | 15,205 |
| Promoted incidents at 0.25 | 2,963 |
| Promoted incidents containing only benign-labelled alert flows | 2,958 |

For a fixed review queue, each ranker selected the top 1% of scorable flows within each slice, rounded down. Both queues held 319 flows. The frozen RF retrieved **86** attack-labelled flows and a baseline ordering by total packet count retrieved **66**. Ties follow the original row order. This comparison ranks flows. The incident workload above uses the separate frozen aggregation and promotion policy.

## Reproduce

Install the source checkout with Python 3.13. Run the commands from the repository root. The evaluation reads approximately 16 MiB from the publisher's public S3 object with anonymous HTTP range requests and writes only aggregate results.

```bash
python tools/fetch_frozen_model.py --tag model-context-rf-v2 --destination models/context-rf-v2.joblib
python tools/evaluate_rf_transfer.py --model models/context-rf-v2.joblib --output reproduced.json
python -c "import json,pathlib; load=lambda p: json.loads(pathlib.Path(p).read_bytes()); assert load('reproduced.json') == load('docs/benchmarks/cic2018-frozen-result.json')"
```

The selection and thresholds were fixed before scoring. Four 4 MiB ranges begin at byte fractions 0, ¼, ½ and ¾ of the same 4,054,925,350-byte object. The evaluator checks the HTTP range and object ETag, retains whole lines, and records the SHA-256 of each response. Context starts empty in each slice. The contract requires the two IPs, both ports, protocol, timestamp and all 76 measured flow features. Column aliases follow the exporter definitions. Its historical `CWE Flag Count` header corresponds to its TCP CWR counter. Raw values retain the exporter's units. The evaluator converts the printed day-first timestamps to the product format, leaves source time local, and excludes rows with unusable numeric measurements, timestamps or labels before constructing context. [Frozen protocol](benchmarks/cic2018-frozen-protocol.json) · [Per-slice counts and hashes](benchmarks/cic2018-frozen-result.json) · [Exporter field list](https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter/blob/acaf8bea8611fb4b996b4d33964dfd9155d9efdf/src/main/java/cic/cs/unb/ca/jnetpcap/FlowFeature.java) · [Counter in the exporter](https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter/blob/acaf8bea8611fb4b996b4d33964dfd9155d9efdf/src/main/java/cic/cs/unb/ca/jnetpcap/BasicFlow.java#L1182)

All sampled positive rows have the publisher label `DDoS attacks-LOIC-HTTP` and fall in the first range. The other three ranges contain only benign-labelled flows. File order is clustered, the sample covers one capture day, and earlier history outside each range is unavailable. Counts refer to these rows and the incidents they form. The gaps between byte ranges prevent a defensible incidents-per-hour estimate for the full network. The companion benchmark measures a separate period with a different source schema. [Independent workload benchmark](EXTERNAL_EVALUATION.md)
