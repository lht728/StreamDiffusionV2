#!/usr/bin/env bash
# Build the StreamDiffusionV2 image and push it to the configured registry.
#
# Usage:
#   ./docker-build-push.sh                  # tag = current git short sha + 'latest'
#   ./docker-build-push.sh v0512            # tag = v0512 + 'latest'
#   TAG=v0512 ./docker-build-push.sh
#   BUILD_FLASH_ATTN=1 ./docker-build-push.sh v0512-fa   # also build flash-attn
#   PUSH=0 ./docker-build-push.sh v0512                  # build only, no push
#
# Required: docker daemon running, and `docker login mirrors.tencent.com` done once.

set -euo pipefail

REGISTRY="${REGISTRY:-mirrors.tencent.com/gputest/streamdiffusionv2}"
BUILD_FLASH_ATTN="${BUILD_FLASH_ATTN:-0}"
PUSH="${PUSH:-1}"

# Resolve tag: positional arg > $TAG env > git short sha > timestamp.
if [ "${1:-}" != "" ]; then
  TAG="$1"
elif [ "${TAG:-}" != "" ]; then
  TAG="$TAG"
elif command -v git >/dev/null 2>&1 && git rev-parse --short HEAD >/dev/null 2>&1; then
  TAG="$(git rev-parse --short HEAD)"
else
  TAG="$(date +%Y%m%d_%H%M%S)"
fi

IMG_TAG="${REGISTRY}:${TAG}"
IMG_LATEST="${REGISTRY}:latest"

echo "==> Registry:        ${REGISTRY}"
echo "==> Tag:             ${TAG}"
echo "==> BUILD_FLASH_ATTN=${BUILD_FLASH_ATTN}"
echo "==> Will push:       ${PUSH}"
echo

# BuildKit gives us better caching + multi-stage parallelism.
export DOCKER_BUILDKIT=1

docker build \
  --build-arg BUILD_FLASH_ATTN="${BUILD_FLASH_ATTN}" \
  --tag "${IMG_TAG}" \
  --tag "${IMG_LATEST}" \
  .

echo
echo "==> Built ${IMG_TAG}"
docker images --filter=reference="${REGISTRY}" | head -n 5

if [ "${PUSH}" = "1" ] || [ "${PUSH}" = "true" ]; then
  echo
  echo "==> Pushing ${IMG_TAG}"
  docker push "${IMG_TAG}"
  echo "==> Pushing ${IMG_LATEST}"
  docker push "${IMG_LATEST}"
  echo
  echo "==> Done. Pull on another GPU host with:"
  echo "    docker pull ${IMG_TAG}"
else
  echo
  echo "==> PUSH=0, skipping push. To push later:"
  echo "    docker push ${IMG_TAG} && docker push ${IMG_LATEST}"
fi
