# Public product CI and regression gates

`.github/workflows/ci.yml` runs on every pull request and push to `main` with read-only repository permissions. Public CI needs only a clean checkout and the dedicated public `model-context-rf-v2` release; no private/research dataset is used.

## Jobs

### dependency-audit

Installs the pinned `pip-audit==2.10.1` scanner and audits the complete pinned container runtime closure in `requirements-container.txt`. Known vulnerability findings or dependency-resolution errors fail the job. This explicit CI security check uses network access to query vulnerability data; normal product inference does not.

### unit-and-contracts

Uses Python 3.13.7, installs the source/test dependencies, runs the complete committed suite without an external model, and runs `pip check`. Real-model-only tests skip explicitly. Contract tests verify repository and package-resource bytes, including the frozen feature-contract SHA.

### clean-install-cli

Builds wheel and sdist, installs the wheel into a separate clean environment, changes to a directory outside the checkout, and verifies:

- `security-anomaly --help`;
- `security-anomaly version`;
- label-free validation of the synthetic fixture;
- resources resolve from installed `site-packages`;
- `pip check`.

### frozen-model-e2e

Downloads `context-rf-v2.joblib` from tag `model-context-rf-v2` with `tools/fetch_frozen_model.py`, verifies SHA-256 before model loading, installs the built wheel, runs `model-info`, and executes the real-model fixture regression. It gates flow scores (absolute tolerance `1e-12`), alert/incident/promoted counts, incident membership, public IDs, repeated determinism, CLI/direct detector agreement, and exact golden JSONL.

No fake model may satisfy this job.

### docker-smoke

Downloads/verifies the same public model, supplies it as a named Docker build context, builds the non-root image, and runs `version`, `model-info`, `validate`, and `analyze` with `--network none`. Container output must match the same committed golden byte-for-byte. The job also reports image size/base digest and checks that research data/build directories are absent.

## Synthetic fixture and golden policy

The fixture under `tests/fixtures/product-v01/` is generated entirely from explicit synthetic profiles. It contains no UNSW/CIC rows or labels. Its compact manifest freezes:

- fixture and output SHA-256 values;
- frozen product/model/feature/builder/incident versions;
- model SHA and release tag;
- flow threshold, Policy B key/window, and promotion threshold;
- representative flow scores and tolerance;
- alert, incident, and promoted counts;
- incident membership and promotion;
- exact promoted `incident-v1` JSONL and public IDs.

Public incident scores are canonically serialized at 12 decimal places so platform-level numeric noise below the `1e-12` regression tolerance cannot change JSONL bytes. Threshold, aggregation, and promotion decisions use the original unrounded detector scores.

CI never updates goldens. A failure requires investigation; golden changes must be explicit and human-reviewed. Model, threshold, aggregation, promotion, feature contract, or serializer changes cannot be accepted as incidental snapshot updates.

Research reproduction tests that require non-redistributed datasets may remain skipped, but all product jobs are runnable from public inputs alone.

This CI demonstrates reproducibility and contract stability; it does not establish production readiness or production performance.
