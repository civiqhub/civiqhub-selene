#!/usr/bin/env bash
# Build the image of the service from this checkout, and describe it.
#
#   services/api/image/build.sh [TAG]          (default tag: selene-api:candidate)
#
# 1. Build the engine wheel and the service wheel of this commit into dist/.
# 2. Download the locked runtime wheels into dist/wheelhouse/, checked by their hashes.
# 3. Build the image offline from these files (services/api/Dockerfile).
# 4. Save the image as an OCI archive, dist/selene-image.oci.tar, and check that its index
#    is the image that was built: the publication pushes this archive, so the digest it
#    publishes is the digest the tests ran.
# 5. Write dist/image-metadata.json: commit, versions, wheel hashes, digests.
#
# It needs the containerd image store of Docker (the default since Docker 29), where the
# identifier of a built image is the digest of its index.
#
# PYTHON_IMAGE, when set, replaces the registry of the base image; its digest must stay
# the one of the Dockerfile.
set -euo pipefail

tag="${1:-selene-api:candidate}"
root="$(git rev-parse --show-toplevel)"
cd "$root"

if [ -d dist ]; then
  find dist -mindepth 1 -delete
fi
mkdir -p dist/wheelhouse

python -m build --wheel --outdir dist . >/dev/null
python -m build --wheel --outdir dist services/api >/dev/null
python -m pip download --quiet --no-deps --require-hashes --only-binary=:all: \
  --implementation cp --python-version 3.13 --abi cp313 \
  --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
  --platform manylinux2014_x86_64 --platform linux_x86_64 \
  --dest dist/wheelhouse -r services/api/requirements.lock

engine_wheel="$(ls dist/civiqhub_selene-*.whl)"
service_wheel="$(ls dist/selene_api-*.whl)"
engine_version="$(basename "$engine_wheel" | cut -d- -f2)"
service_version="$(basename "$service_wheel" | cut -d- -f2)"
contract_version="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["info"]["version"])' services/api/openapi.yaml)"
engine_sha256="$(sha256sum "$engine_wheel" | cut -d' ' -f1)"
service_sha256="$(sha256sum "$service_wheel" | cut -d' ' -f1)"
commit="$(git rev-parse HEAD)"

base_args=()
if [ -n "${PYTHON_IMAGE:-}" ]; then
  base_args=(--build-arg "PYTHON_IMAGE=${PYTHON_IMAGE}")
fi

docker build \
  --file services/api/Dockerfile \
  --tag "$tag" \
  "${base_args[@]}" \
  --build-arg "ENGINE_VERSION=${engine_version}" \
  --build-arg "ENGINE_WHEEL_SHA256=${engine_sha256}" \
  --build-arg "SERVICE_VERSION=${service_version}" \
  --build-arg "SERVICE_WHEEL_SHA256=${service_sha256}" \
  --build-arg "CONTRACT_VERSION=${contract_version}" \
  --build-arg "COMMIT=${commit}" \
  .

image_id="$(docker image inspect --format '{{.Id}}' "$tag")"
docker save --output dist/selene-image.oci.tar "$tag"
python - "$tag" "$image_id" "$commit" "$engine_version" "$engine_sha256" "$service_version" \
  "$service_sha256" "$contract_version" "${PYTHON_IMAGE:-}" > dist/image-metadata.json <<'PYTHON'
import hashlib
import json
import sys
import tarfile

tag, image_id, commit, engine, engine_sha, service, service_sha, contract, base = sys.argv[1:]
archive = "dist/selene-image.oci.tar"
with tarfile.open(archive) as oci:
    index = json.load(oci.extractfile("index.json"))
digests = [entry["digest"] for entry in index["manifests"]]
if digests != [image_id]:
    sys.exit(f"the OCI archive holds {digests}, not the built image {image_id}")
with open(archive, "rb") as data:
    archive_sha = hashlib.file_digest(data, "sha256").hexdigest()
print(json.dumps({
    "tag": tag,
    "image_digest": image_id,
    "archive": {"path": archive, "sha256": archive_sha},
    "commit": commit,
    "engine": {"version": engine, "wheel_sha256": engine_sha},
    "service": {"version": service, "wheel_sha256": service_sha},
    "contract": {"version": contract},
    "python_image": base or None,
}, indent=2))
PYTHON
cat dist/image-metadata.json
