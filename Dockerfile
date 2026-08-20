# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.13.7-slim-bookworm@sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d

FROM ${PYTHON_IMAGE} AS wheel-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build

COPY pyproject.toml README.md LICENSE requirements-container.txt ./
COPY src/security_anomaly/ ./src/security_anomaly/

RUN python -m pip wheel --wheel-dir /wheels -r requirements-container.txt \
    && python -m pip wheel --no-deps --wheel-dir /wheels .


FROM ${PYTHON_IMAGE} AS runtime

ARG MODEL_SHA256=4730a06506d8c5f2af93679c492e1544b3c2b11acd16fe74120d64d4dbfc5c72

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SECURITY_ANOMALY_MODEL=/opt/security-anomaly/models/context-rf-v2.joblib

RUN groupadd --gid 10001 anomaly \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin anomaly \
    && install -d -o 10001 -g 10001 /data /opt/security-anomaly/models

COPY --from=wheel-builder /wheels/ /tmp/wheels/
RUN python -m pip install --no-index --find-links=/tmp/wheels security-anomaly-ml==0.1.0 \
    && python -m pip check \
    && rm -rf /tmp/wheels

# The named build context must be supplied explicitly:
#   --build-context model=/path/to/verified-model-directory
COPY --from=model /context-rf-v2.joblib /opt/security-anomaly/models/context-rf-v2.joblib

RUN printf '%s  %s\n' "${MODEL_SHA256}" "${SECURITY_ANOMALY_MODEL}" | sha256sum -c - \
    && security-anomaly version \
    && security-anomaly model-info

USER 10001:10001
WORKDIR /data

ENTRYPOINT ["security-anomaly"]
CMD ["--help"]
