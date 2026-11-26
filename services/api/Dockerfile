# syntax=docker/dockerfile:1
#
# Image of the SELENE service, built from the root of the repository:
#
#   services/api/image/build.sh
#
# The build installs, without any index and without building anything:
# - the runtime dependencies of services/api/requirements.lock, from the wheels that
#   build.sh downloaded into dist/wheelhouse, checked by the hashes of the lock;
# - the engine wheel dist/civiqhub_selene-<version>-py3-none-any.whl built from the same
#   commit, checked against ENGINE_WHEEL_SHA256 and ENGINE_VERSION: a missing, ambiguous
#   or different wheel fails the build, and no other engine is ever fetched;
# - the service wheel dist/selene_api-<version>-py3-none-any.whl of the same commit,
#   checked against SERVICE_WHEEL_SHA256.
# The image runs as the non-root user 10001 on a read-only root file system; it holds no
# secret, no environment configuration and no data. The same image runs locally, in dev
# and in staging: only the SELENE_* variables and the mounted files change.

# Python 3.13, Debian slim, pinned by the digest of its manifest list.
ARG PYTHON_IMAGE=docker.io/library/python:3.13-slim@sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50

FROM ${PYTHON_IMAGE} AS build

ARG ENGINE_VERSION
ARG ENGINE_WHEEL_SHA256
ARG SERVICE_VERSION
ARG SERVICE_WHEEL_SHA256
ARG CONTRACT_VERSION
ARG COMMIT
ARG SOURCE=https://github.com/civiqhub/civiqhub-selene

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY services/api/requirements.lock /build/requirements.lock
COPY dist/ /build/dist/

RUN set -eu; \
    for name in ENGINE_VERSION ENGINE_WHEEL_SHA256 SERVICE_VERSION SERVICE_WHEEL_SHA256 \
                CONTRACT_VERSION COMMIT; do \
      eval "value=\${$name:-}"; \
      [ -n "$value" ] || { echo "build argument $name is required" >&2; exit 1; }; \
    done; \
    cd /build/dist; \
    engine="civiqhub_selene-${ENGINE_VERSION}-py3-none-any.whl"; \
    service="selene_api-${SERVICE_VERSION}-py3-none-any.whl"; \
    [ "$(ls -1 ./*.whl | wc -l)" -eq 2 ] || { echo "dist/ holds other wheels than the two of this commit" >&2; ls -1 >&2; exit 1; }; \
    [ -f "$engine" ] || { echo "the engine wheel $engine is missing" >&2; exit 1; }; \
    [ -f "$service" ] || { echo "the service wheel $service is missing" >&2; exit 1; }; \
    echo "$ENGINE_WHEEL_SHA256  $engine" | sha256sum -c -; \
    echo "$SERVICE_WHEEL_SHA256  $service" | sha256sum -c -; \
    python -m venv --without-pip /opt/selene; \
    python -m pip --python /opt/selene/bin/python install --no-index --no-deps \
      --find-links /build/dist/wheelhouse --require-hashes -r /build/requirements.lock; \
    python -m pip --python /opt/selene/bin/python install --no-index --no-deps \
      "/build/dist/$engine" "/build/dist/$service"; \
    python -m pip --python /opt/selene/bin/python check; \
    /opt/selene/bin/python -c "import sys, civiqhub_selene, importlib.metadata as m; \
v = civiqhub_selene.__version__; s = m.version('selene-api'); \
assert v == sys.argv[1], (v, sys.argv[1]); assert s == sys.argv[2], (s, sys.argv[2]); \
import selene_api.schemas as c; assert c.CONTRACT_VERSION == sys.argv[3]" \
      "$ENGINE_VERSION" "$SERVICE_VERSION" "$CONTRACT_VERSION"; \
    /opt/selene/bin/python -m compileall -q /opt/selene/lib; \
    mkdir -p /app; \
    printf '{"commit": "%s", "source": "%s", "engine_version": "%s", "engine_wheel_sha256": "%s", "service_version": "%s", "service_wheel_sha256": "%s", "contract_version": "%s"}\n' \
      "$COMMIT" "$SOURCE" "$ENGINE_VERSION" "$ENGINE_WHEEL_SHA256" "$SERVICE_VERSION" \
      "$SERVICE_WHEEL_SHA256" "$CONTRACT_VERSION" > /app/build-info.json


FROM ${PYTHON_IMAGE}

ARG ENGINE_VERSION
ARG ENGINE_WHEEL_SHA256
ARG SERVICE_VERSION
ARG CONTRACT_VERSION
ARG COMMIT
ARG SOURCE=https://github.com/civiqhub/civiqhub-selene
ARG PYTHON_IMAGE

LABEL org.opencontainers.image.title="selene-api" \
      org.opencontainers.image.description="SELENE computation service, contract v1" \
      org.opencontainers.image.source="${SOURCE}" \
      org.opencontainers.image.revision="${COMMIT}" \
      org.opencontainers.image.version="${SERVICE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="CiviqHub SAS" \
      org.opencontainers.image.base.name="${PYTHON_IMAGE}" \
      io.civiqhub.selene.contract-version="${CONTRACT_VERSION}" \
      io.civiqhub.selene.engine-version="${ENGINE_VERSION}" \
      io.civiqhub.selene.engine-wheel-sha256="${ENGINE_WHEEL_SHA256}"

# No package manager is needed at run time; the user owns nothing in the image.
RUN python -m pip uninstall --yes --quiet pip \
 && groupadd --system --gid 10001 selene \
 && useradd --system --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
      --shell /usr/sbin/nologin selene

COPY --from=build /opt/selene /opt/selene
COPY --from=build /app/build-info.json /app/build-info.json
# The contract, its client and its TEST ONLY fixtures, for the acceptance job.
COPY services/api/openapi.yaml /app/openapi.yaml
COPY services/api/contract/selene_contract.py /app/contract/selene_contract.py
COPY services/api/contract/fixtures/ /app/contract/fixtures/

ENV PATH=/opt/selene/bin:${PATH} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/tmp \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    SELENE_BUILD_INFO_FILE=/app/build-info.json

WORKDIR /app
USER 10001:10001
EXPOSE 8443 9464
STOPSIGNAL SIGTERM
CMD ["python", "-m", "selene_api"]
