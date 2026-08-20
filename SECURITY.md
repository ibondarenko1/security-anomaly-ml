# Security policy

## Supported versions

| Version | Security support |
|---|---|
| 0.1.x | Supported |
| Earlier research snapshots | Not supported |

Security support means maintainers may investigate and address application-security defects in the supported release line. It is not a production-readiness statement or a guaranteed response SLA.

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/ibondarenko1/security-anomaly-ml/security/advisories/new) for suspected exploitable security defects. Please do not open a public issue or disclose exploit details publicly before coordinated handling.

Include, where possible:

- affected product version, commit, installation method, and operating system;
- whether the issue affects the Python CLI, Docker image, artifact verification, input validation, or output handling;
- a concise description and security impact;
- minimal reproduction steps or a proof of concept using synthetic/redacted data;
- relevant logs with credentials, internal addresses, flow records, and other sensitive values removed;
- any known mitigations or suggested remediation.

Reports will be handled on a best-effort basis. No acknowledgement, remediation, or disclosure timeline is promised.

## Security defect versus model-quality report

Private vulnerability reporting is for application-security problems such as unsafe artifact loading, validation bypass, command or path injection, unintended data disclosure, dependency compromise, or container isolation defects.

False positives, false negatives, score drift, weak attack-category performance, dataset bias, and generalization concerns are model-quality reports unless they demonstrate a concrete exploitable application-security impact. Model-quality reports may use [GitHub Issues](https://github.com/ibondarenko1/security-anomaly-ml/issues), but must contain only synthetic or safely aggregated evidence. Never post packet captures, raw internal flow exports, credentials, private IP inventories, or unredacted incident output publicly.

## Scope and expectations

Security Anomaly ML v0.1 is a research/evaluation-grade batch detector, not a production security boundary or SOC replacement. Operators remain responsible for protecting input/output files, verifying published artifact hashes and image digests, controlling access to network metadata, and validating suitability in their own environment.
