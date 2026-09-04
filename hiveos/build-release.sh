#!/usr/bin/env bash
# Safe NVIDIA release entry point.
#
# This delegates to the reproducible, network-isolated modern-line builder and then packages
# every supported distribution flavor. In particular, build-offline.sh builds both mandatory
# CUDA llama.cpp inference engines and the complete CUDA runtime closure before package-line.sh
# is allowed to publish anything.
#
# Offline prerequisites (documented in build-offline.sh):
#   KERYX_LLAMA_SRC   clean llama.cpp b10015 checkout
#   KERYX_CMAKE_ROOT  portable CMake >= 3.18 tree
# Optional overrides:
#   KERYX_BUILD_IMAGE      build image (default: keryx-build:offline)
#   KERYX_HOST_CUDA_DIR    host CUDA toolkit mounted into the build container
#   KERYX_LLAMA_JOBS       parallel jobs for each llama.cpp engine build
#
# Output: all modern-line tarballs plus SHA256SUMS-modern.txt in hiveos/dist-modern/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${KERYX_BUILD_IMAGE:-keryx-build:offline}"
BUILD_ARGS=("$IMAGE" dist-modern modern)
if [[ -n "${KERYX_HOST_CUDA_DIR:-}" ]]; then
  BUILD_ARGS+=("$KERYX_HOST_CUDA_DIR")
fi

"$REPO/hiveos/build-offline.sh" "${BUILD_ARGS[@]}"
"$REPO/hiveos/package-line.sh" dist-modern modern

echo ">> Modern NVIDIA release is complete in $REPO/hiveos/dist-modern"
