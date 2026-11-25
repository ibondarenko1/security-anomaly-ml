# Outbound target coverage in UWF-ZeekData24

We checked three UWF-ZeekData24 CSV partitions to see whether they could support an independent comparison of Zeek scoring scales at review budgets of one, three and five. The later period contained **12,391 benign-labelled records and zero target-labelled records**. It cannot establish which scale retrieves more target episodes. The released detector retains its existing scale.

The selection used the first 20,000 rows of the named benign and credential-access partitions, and all 2,909 rows of the named reconnaissance partition. The labels come from the publisher's partitions. All 42,909 input UIDs were distinct. The checker mapped the publisher's address and port fields into the product's Zeek parser and retained 41,303 connections with all eight measurements. It sorted by start time and UID and split the population chronologically.

| Period | Complete connections | Target-labelled | Benign-labelled |
|---|---:|---:|---:|
| Earlier 70% | 28,912 | 22,242 | 6,670 |
| Later 30% | 12,391 | 0 | 12,391 |

Using `143.88.0.0/16` as a candidate local prefix yields nine complete later connections from inside the prefix to an outside destination. This is a count under an explicit prefix assumption. The files identify no monitored enterprise egress boundary. [University of West Florida data](https://datasets.uwf.edu/data/UWF-ZeekData24/csv/) · [Selected partitions and SHA-256](benchmarks/zeek-scale-uwf-protocol.json) · [Counts and UTC timestamps](benchmarks/zeek-scale-uwf-result.json)

From a source checkout with the project installed on Python 3.13, run the following command. It verifies the publisher files against all three recorded lengths and digests before parsing. The report contains aggregate counts and time ranges.

```bash
python tools/check_uwf_coverage.py --output uwf-coverage.json
```
