# Reproducible Docker runtime

The v0.1 image packages the installed `security-anomaly-ml` wheel, exact Python 3.13-compatible dependency closure, package resources, and the verified `context-rf-v2` model. It does not use repository `PYTHONPATH`, run as root, or download anything during inference.

This remains a research/evaluation prototype and is not production-ready.

## Published v0.1.0 image

The normal user image includes the verified frozen model:

```bash
docker pull ghcr.io/ibondarenko1/security-anomaly-ml:0.1.0
docker run --rm --network none \
  ghcr.io/ibondarenko1/security-anomaly-ml:0.1.0 version
docker run --rm --network none \
  ghcr.io/ibondarenko1/security-anomaly-ml:0.1.0 model-info
```

Only the immutable release tag `0.1.0` is published. v0.1.0 does not publish a mutable `latest` tag. The GitHub Release records the image's immutable registry digest.

## Frozen model release

The dedicated prerelease is [`model-context-rf-v2`](https://github.com/ibondarenko1/security-anomaly-ml/releases/tag/model-context-rf-v2). It is an artifact-distribution release, not the final application `v0.1.0` release.

The release contains:

- `context-rf-v2.joblib`
- `model-manifest-context-rf-v2.json`
- `feature-contract-cicflow-v2-128.json`
- `SHA256SUMS.json`

Required model SHA-256:

```text
4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72
```

The release asset was downloaded again into a separate directory after publication and independently matched this hash. The CLI never downloads a model silently.

## Build from source

Download the exact asset outside the primary Docker context:

```bash
MODEL_DIR="$(mktemp -d)"
python tools/fetch_frozen_model.py \
  --tag model-context-rf-v2 \
  --destination "$MODEL_DIR/context-rf-v2.joblib"
```

The helper writes atomically, verifies SHA-256 before success, and refuses to replace an existing unexpected file.

Build with that verified directory as the explicit named model context:

```bash
docker build \
  --build-context model="$MODEL_DIR" \
  --tag security-anomaly-ml:0.1.0 .
```

The Dockerfile checks the model hash again before completing. A missing/wrong model fails the build. The base image is Python `3.13.7-slim-bookworm`, pinned to OCI index digest:

```text
sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d
```

The strict `.dockerignore` sends only wheel-build inputs. Research scripts, tests, datasets, models from the checkout, `.git`, virtual environments, caches, and generated evaluation artifacts are excluded.

## Use a locally built image

The image sets `SECURITY_ANOMALY_MODEL=/opt/security-anomaly/models/context-rf-v2.joblib`, so normal use needs no `--model` argument:

```bash
docker run --rm security-anomaly-ml:0.1.0 version
docker run --rm security-anomaly-ml:0.1.0 model-info
```

Analyze a bind-mounted CSV:

```bash
docker run --rm \
  --network none \
  -v "$PWD:/data" \
  security-anomaly-ml:0.1.0 \
  analyze /data/flows.csv --output /data/incidents.jsonl
```

Windows PowerShell:

```powershell
docker run --rm --network none `
  --mount "type=bind,source=$($PWD.Path),target=/data" `
  ghcr.io/ibondarenko1/security-anomaly-ml:0.1.0 `
  validate /data/flows.csv

docker run --rm --network none `
  --mount "type=bind,source=$($PWD.Path),target=/data" `
  ghcr.io/ibondarenko1/security-anomaly-ml:0.1.0 `
  analyze /data/flows.csv `
  --output /data/incidents.jsonl
```

`--network none` is supported for `version`, `model-info`, `validate`, and `analyze`. Normal inference is offline. The container runs as UID/GID `10001`, so the mounted output directory must be writable by that identity.

## Local smoke snapshot

On the 12-row synthetic fixture, one local verification produced:

- image size: `266,805,442` bytes (about 254.4 MiB);
- Python CLI wall time: about `4.84s` on a cold run;
- Docker CLI wall time: about `4.21s` on a cold run;
- Docker internal analysis time: about `1.63s`;
- promoted JSONL SHA-256: `4c6b8a5d93d7c005d7e3a9741562fb7ba08025ed5feed7dda7573736066df073`.

Peak memory was not collected reliably in this smoke. These numbers are informational workstation measurements, not production performance claims.
