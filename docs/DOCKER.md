# Reproducible Docker runtime

The image contains the installable `security-anomaly-ml` wheel, its pinned
runtime dependencies, package resources, and the verified `context-rf-v2`
model. The model is supplied through a separate named build context. The
checkout does not provide `PYTHONPATH`, the image runs as UID/GID `10001`, and
inference does not download data or model files.

## Published image

Release images use the same stable version as the Git tag and Python package.
There is no mutable `latest` tag.

```bash
docker pull ghcr.io/ibondarenko1/security-anomaly-ml:0.2.0
docker run --rm --network none \
  ghcr.io/ibondarenko1/security-anomaly-ml:0.2.0 version
docker run --rm --network none \
  ghcr.io/ibondarenko1/security-anomaly-ml:0.2.0 model-info
```

The `0.2.0` value is an example. Use the version in the release tag. The
GitHub Release records the immutable image digest after the image has passed
the offline regression.

## Build from a checked-out tag

The release workflows reject prerelease tags and require the tag, project
metadata, distribution metadata, and image label to agree. The same checks can
be run locally before opening a release.

```bash
VERSION=0.2.0
TAG="v${VERSION}"
python tools/validate_release.py --project-root . --tag "$TAG"
python tools/fetch_frozen_model.py \
  --tag model-context-rf-v2 \
  --destination "$RUNNER_TEMP/context-rf-v2.joblib" \
  --sha256 4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72

docker build \
  --build-context model="$RUNNER_TEMP" \
  --build-arg PACKAGE_VERSION="$VERSION" \
  --build-arg SOURCE_REVISION="$(git rev-parse "$TAG^{commit}")" \
  --tag "security-anomaly-ml:$VERSION" .
```

The model directory must contain `context-rf-v2.joblib` at its root. The
Dockerfile checks its SHA-256 again during the build. The base image is Python
`3.13.7-slim-bookworm`, pinned to OCI index digest
`sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d`.

## Offline smoke checks

The bundled Zeek demonstration is useful for checking a clean wheel or image
without a model download. It should emit exactly one finding for connection
`H2`.

```bash
security-anomaly demo zeek --output-dir "$TMPDIR/security-anomaly-zeek"
security-anomaly zeek-fit "$TMPDIR/security-anomaly-zeek/fit.jsonl" \
  --sensor edge --output "$TMPDIR/profile.json"
security-anomaly zeek-score "$TMPDIR/security-anomaly-zeek/heldout.jsonl" \
  --sensor edge --profile "$TMPDIR/profile.json" \
  --output "$TMPDIR/findings.jsonl"
```

For a container, bind a writable directory to `/data` and run the same
commands with `--network none`.

```bash
docker run --rm --network none \
  --volume "$TMPDIR/security-anomaly-zeek:/data" \
  "security-anomaly-ml:$VERSION" demo zeek --output-dir /data/demo
docker run --rm --network none \
  --volume "$TMPDIR/security-anomaly-zeek:/data" \
  "security-anomaly-ml:$VERSION" zeek-fit /data/demo/fit.jsonl \
  --sensor edge --output /data/profile.json
docker run --rm --network none \
  --volume "$TMPDIR/security-anomaly-zeek:/data" \
  "security-anomaly-ml:$VERSION" zeek-score /data/demo/heldout.jsonl \
  --sensor edge --profile /data/profile.json --output /data/findings.jsonl
```

The image also runs the existing non-root CICFlowMeter golden regression in CI.
For local use, mount a writable directory because UID/GID `10001` owns the
runtime output path.

```bash
docker run --rm --network none \
  --volume "$PWD:/data" \
  "security-anomaly-ml:$VERSION" analyze /data/flows.csv \
  --output /data/incidents.jsonl
```

## Release and retry procedure

Build distributions from the tag and write a checksum manifest before any
registry mutation.

```bash
rm -rf dist
python -m pip install --upgrade build
python -m build --sdist --wheel --outdir dist
python tools/validate_release.py \
  --project-root . --tag "$TAG" --dist-dir dist
```

`dist/SHA256SUMS` is the release asset to attach to the matching GitHub
Release. Upload the wheel, source archive, and manifest without replacing an
existing asset. If a retry finds an existing asset, download it and compare
the recorded SHA-256 before considering the step complete.

Create the release as a draft, upload these exact three files, and publish the
draft only after their hashes have been checked. The package workflow consumes
these bytes and never rebuilds timestamp-sensitive archives.

```bash
cp dist/SHA256SUMS "$TMPDIR/SHA256SUMS"
rm dist/SHA256SUMS
gh release create "$TAG" --verify-tag --draft --title "Security Anomaly ML $VERSION"
gh release upload "$TAG" dist/*.whl dist/*.tar.gz "$TMPDIR/SHA256SUMS"
gh release edit "$TAG" --draft=false
```

`gh release upload` deliberately omits `--clobber`. A retry that sees an
existing asset must compare its downloaded hash and keep the existing bytes.

The two release workflows accept a manual dry-run for an existing tag.

```bash
gh workflow run "Publish release to PyPI" \
  -f release_tag="$TAG" -f publish=false
gh workflow run "Publish release image" \
  -f release_tag="$TAG" -f publish=false
```

Publishing the GitHub Release for the exact stable tag runs the same checks and
enables publication. This event is used after the wheel, source archive, and
`SHA256SUMS` assets have been attached, so the package workflow cannot race
release-asset creation.
The PyPI workflow compares an existing version's file hashes and skips an
identical retry. A different hash fails closed. The GHCR workflow compares the
existing image content before deciding whether a push is needed. Neither
workflow changes an earlier stable version or creates `latest`.

## Frozen model release

The dedicated prerelease [`model-context-rf-v2`](https://github.com/ibondarenko1/security-anomaly-ml/releases/tag/model-context-rf-v2)
contains the model, manifest, feature contract, and `SHA256SUMS.json`. The
required model SHA-256 is:

```text
4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72
```

`tools/fetch_frozen_model.py` writes atomically, verifies that digest, and
refuses to replace an unexpected existing file. The CLI never downloads a
model silently.

The strict `.dockerignore` excludes research scripts, tests, datasets, local
models, `.git`, virtual environments, caches, and generated evaluation files
from the primary build context.
