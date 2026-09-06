#!/usr/bin/env bash
# Safe legacy NVIDIA release entry point: CUDA 12.4, Pascal-compatible sm_61 PTX,
# both in-process GPU inference engines, and all four distribution formats.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${KERYX_BUILD_IMAGE:-keryx-build:offline}"
CUDA_DIR="${KERYX_HOST_CUDA_DIR:-/tmp/cuda124}"

[[ -x "$CUDA_DIR/bin/nvcc" ]] || {
  echo "ERROR: CUDA 12.4 toolkit not found at $CUDA_DIR." >&2
  echo "       Set KERYX_HOST_CUDA_DIR to an extracted CUDA 12.4 toolkit." >&2
  exit 2
}

BOFF_POM_CUDA_ARCH=compute_61 BOFF_CUDA_COMPUTE_CAP=61 \
  "$REPO/hiveos/build-offline.sh" "$IMAGE" dist-legacy legacy "$CUDA_DIR"
"$REPO/hiveos/package-line.sh" dist-legacy legacy

echo ">> Legacy NVIDIA release is complete in $REPO/hiveos/dist-legacy"
