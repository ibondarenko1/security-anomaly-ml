# Zeek JSONL triage

`zeek-fit` and `zeek-score` provide a small, local baseline for Zeek JSON
`conn.log` records.  They are intentionally separate from the frozen
CICFlowMeter detector: the profile uses only Zeek-native connection fields and
does not impute missing values or reuse the CIC feature contract.

## Workflow

Fit the profile from a sensor-local reference period, then score a different
held-out period:

```bash
security-anomaly zeek-fit fit.jsonl \
  --sensor edge-a \
  --output edge-a-profile.json

security-anomaly zeek-score heldout.jsonl \
  --profile edge-a-profile.json \
  --sensor edge-a \
  --output edge-a-findings.jsonl
```

The profile is a JSON document containing the ordered Zeek feature list,
median centers, positive robust scales (`1.4826 * MAD`), fit-record count,
sensor name, and threshold.  A score is the largest absolute robust deviation
across the eight measured fields.  The default threshold is `6.0`; it can be
set during fitting or explicitly overridden while scoring.  This is a
transparent triage score, not a probability and not a verdict that traffic is
malicious.

The profile is sensor-scoped.  `zeek-score` rejects a requested sensor that
does not match the profile, preventing an unrelated sensor's baseline from
being silently reused.

## Input contract

Each non-empty, non-comment line must be a JSON object with Zeek connection
identity fields:

`ts`, `uid`, `id.orig_h`, `id.orig_p`, `id.resp_h`, `id.resp_p`, and `proto`.

The timestamp is a Unix epoch converted to an aware UTC `datetime`.  IPs are
validated with Python's `ipaddress` module and canonicalized; ports must be
integers in `[0, 65535]`; protocol names are normalized to lowercase.  The
profile fields are:

`duration`, `orig_bytes`, `resp_bytes`, `orig_pkts`, `resp_pkts`,
`orig_ip_bytes`, `resp_ip_bytes`, and `missed_bytes`.

Rows with invalid JSON or invalid identity/feature values are excluded.  Rows
with valid identity but missing profile fields are counted as incomplete and
are never scored.  No replacement value is inserted.  Duplicate `uid` values
within one input keep the first row and count subsequent rows as excluded
duplicates.  The command summary reports parsed, scored, excluded,
incomplete, invalid, and duplicate counts.

## Output contract

Only rows whose score reaches the threshold are emitted.  Each finding is
constructed through the shared `detection_finding` builder as OCSF Detection
Finding class `2004`.  The Zeek `uid` remains the observation identifier and
the builder scopes its stable identity by sensor. The OCSF start time comes
from the Zeek timestamp. The end time adds the observed connection duration.
Source and destination addresses, ports, detector metadata and package version
are included in each finding.

The evidence object carries the protocol, all measured Zeek fields, the
training median/MAD baseline, per-field deviations, score, threshold,
anomalous fields, and the explicit `imputation: "none"` marker.  Findings are
written as UTF-8 JSONL through the shared atomic serializer.  Input and
profile paths are protected from output collisions, including hard-link
aliases. A SHA-256 fingerprint identifies the exact fitted profile used for
the observation.

## Validation boundary

`tests/fixtures/zeek/fit.jsonl` and `heldout.jsonl` are synthetic contract
fixtures.  They verify parser validation, UTC conversion, IPv4/IPv6
canonicalization, duplicate/incomplete accounting, sensor separation, and
that a deliberately large held-out deviation is surfaced.  Passing these
tests demonstrates deterministic implementation behavior only; it does not
establish detection quality or generalization to a production network.

Before operational use, fit separately per sensor and time period, choose the
threshold from a clean validation period, measure alert volume and analyst
outcomes, and compare against an independently collected benign and incident
set.  Profile rotation, drift monitoring, suppression policy, and enrichment
are deliberately outside this bounded local module.

## Published-log smoke check

The measurements and profile file digest below were recorded on source version `0.2.0.dev0`.
The serialized profile includes the product version. A separate [episode-level comparison](EPISODE_EVALUATION.md) uses the publisher's labelled logs.

`tools/check_zeek_capture.py` performs one bounded, offline replay against the
54.4 KiB [`CTU-Honeypot-Capture-4-1/bro/conn.log`](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-Honeypot-Capture-4-1/bro/conn.log)
listed by the [official IoT-23 scenario index](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/).
It downloads into a temporary directory, converts the published Zeek TSV
field-for-field to JSONL, sorts by `ts`, fits before scoring, and removes all
temporary data.  The captured source was 55,690 bytes with SHA-256
`ef21b8f322caf62a35fd7a8ebf4e6864b275574eadead425f702772470becca0`; the
converter produced 452 records.  The
conversion preserves Zeek's `-` as an omitted field: the one source row with
missing `orig_bytes` and `resp_bytes` stayed incomplete and no value was
invented.

The run used the first 320 chronological records for fitting and the later
132 for holdout (`1540469197.400159` to `1540526323.472635` for fit, then
`1540526524.791389` to `1540547904.130404` for holdout).  Their JSONL SHA-256 values were
`ce9d14330051328f2f43100c9b4b1b58f5cf956e146059ae5c37fc1315eeff61` and
`8f41238b06f6c26e72fc5839c6be13ca54545958e479108a0e03915fe4233cc3`.
`zeek-fit` accepted 320 complete rows.  `zeek-score` parsed 132, scored 131,
and counted one incomplete row; it emitted 46 findings at threshold `6.0`.
The profile SHA-256 was identical before and after scoring:
`f2c7e8d7d4b4fcff7674638bee77c8432aeeda694b2cba505da6192b9f5e3656`.

This is a parser/profile smoke check on one unlabeled published capture, not
an accuracy, detection-rate, or generalization claim.  The TSV field meanings
follow Zeek's [conn.log documentation](https://docs.zeek.org/en/current/logs/conn.html).
