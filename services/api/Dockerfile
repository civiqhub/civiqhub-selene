# syntax=docker/dockerfile:1
#
# Image of the SELENE service. Stub: nothing to build until the service has an
# implementation, so every instruction below is commented out.
#
# Target shape:
# - the release CI builds the engine wheel once; this image installs that exact
#   wheel and never rebuilds the engine from source;
# - the server wrapper and its locked dependencies are installed on top;
# - the base image is pinned by digest, and the service runs as a non-root user;
# - the image is published as ghcr.io/civiqhub/civiqhub-selene with the same version
#   as the package, signed, with provenance and an SBOM, and deployed by digest only.
#
# FROM python:3.13-slim@sha256:<digest>
#
# RUN useradd --create-home --uid 10001 selene
# WORKDIR /app
#
# COPY dist/civiqhub_selene-*.whl /tmp/wheels/
# COPY services/api/ /app/services/api/
# RUN python -m pip install --no-cache-dir /tmp/wheels/*.whl /app/services/api \
#  && rm -rf /tmp/wheels
#
# USER selene
# EXPOSE 8000
# CMD ["<server command, defined with the first implementation>"]
