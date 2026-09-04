#!/usr/bin/env bash
# Retired release path. The historical package omitted both mandatory H6/H10 CUDA llama
# inference engines and could therefore install successfully while the OPoI safety gate kept
# mining stopped. The maintained line packager validates the engines, CUDA worker, and complete
# runtime closure before creating an archive.
set -euo pipefail

echo "ERROR: hiveos/package.sh is retired because it cannot produce a complete OPoI package." >&2
echo "       Use hiveos/build-release.sh (modern) or build-offline.sh + package-line.sh." >&2
exit 2
