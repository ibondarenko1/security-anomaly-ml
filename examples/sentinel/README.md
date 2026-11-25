# Query local network findings in Microsoft Sentinel

The CLI emits OCSF 1.6 Detection Findings for both sensor types. A second local command wraps each finding as an Azure Monitor input row with a UTC `TimeGenerated` and the full event in a dynamic `Finding` column.

## Try the mapping offline

Install the source checkout and run the following from its root. The public incident fixture contains two promoted CICFlowMeter incidents. Its timestamps represent UTC in this example.

```bash
security-anomaly export-ocsf tests/fixtures/product-v01/expected-incidents.jsonl --sensor demo-edge --timezone UTC --output findings.jsonl
security-anomaly sentinel-batch findings.jsonl --output request.json
```

The first finding references three alert flows to destination port 445. Its highest score is 0.432590909091 and the promotion threshold is 0.25. For rotated captures, `replay --format ocsf` obtains the sensor and timezone from the replay configuration and includes available Flow IDs. For Zeek, pass the JSONL produced by `zeek-score` directly to `sentinel-batch`.

| OCSF field | Analyst use |
|---|---|
| `metadata.log_provider` | Identify the originating sensor. |
| `finding_info.uid` | Group snapshots of a finding within a sensor. CIC identity includes the Policy B key and session start. Earlier data that changes the start creates a different group. |
| `metadata.uid` | Recognize an identical snapshot on repeated export. |
| `finding_info.analytic` | Identify the detector and its version. |
| `evidences[0].uid` | Look up the original incident ID or Zeek connection UID. |
| `evidences[0].data` | Read the observed scores and policy, or the Zeek measurements and profile deviations. |

Each snapshot records the creation of an observation for triage. Severity is informational. Source timestamps for legacy incident files require an explicit IANA zone. Ambiguous or nonexistent wall times fail validation. Replay's normalized legacy output uses UTC. The exporter writes epoch milliseconds in the OCSF envelope. [OCSF Detection Finding 1.6](https://github.com/ocsf/ocsf-schema/blob/1.6.0/events/findings/detection_finding.json)

## Workspace integration

The Bicep example declares the `SecurityAnomalyFindings_CL` table and a direct-ingestion DCR in the existing workspace's resource group. Supply the workspace name and region. It can be compiled locally with `bicep build examples/sentinel/findings.bicep`. The sample is also suitable for the workspace used by the companion detection-engineering project. [Bicep template](findings.bicep) · [Azure Sentinel Detection Engineering](https://github.com/ibondarenko1/azure-sentinel-detection-engineering)

For an operator-managed deployment, use the DCR output `logsIngestion`, its `immutableId` and stream `Custom-SecurityAnomalyFindings` with the Azure Monitor Logs Ingestion API. Grant the sending identity the Monitoring Metrics Publisher role on that DCR. Submit the UTF-8 array in `request.json` as the request body. The batch command caps the body at 1,000,000 bytes and preserves an existing file on error. Split a larger JSONL between findings before preparing each request. [Azure Monitor ingestion API](https://learn.microsoft.com/en-us/azure/azure-monitor/logs/logs-ingestion-api-overview)

Run the supplied KQL against the table to collapse identical snapshots and inspect endpoints, detector evidence and the source UID. A revised finding stays visible under the same finding ID when a later replay or changed profile yields different evidence. The query uses the observation time inside OCSF so historical fixtures remain identifiable even if Azure adjusts the stored ingestion timestamp. This example was validated locally against the table contract. [Triage query](triage.kql)
