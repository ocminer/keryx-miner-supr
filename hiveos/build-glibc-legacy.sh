#!/usr/bin/env bash
# Compatibility name retained for operators with old build commands. The former script could
# overwrite dist-legacy with an incomplete compute_70 build, so all release work now goes through
# the provenance-checked CUDA 12.4/sm_61 pipeline.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo ">> build-glibc-legacy.sh is retired; delegating to build-release-legacy.sh" >&2
exec "$REPO/hiveos/build-release-legacy.sh" "$@"
