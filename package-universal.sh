#!/usr/bin/env bash
# Retired post-fork: a single host binary has exactly one compiled PoM backend.
#
# The old "universal" archive paired a pom-opencl host with both worker plugins.
# A CUDA worker could therefore start, but its post-fork possession walk was still
# routed through the process-wide OpenCL backend (or failed to stage). That is not
# a valid mixed-vendor miner and can misattribute work across physical devices.
#
# Keep mixed rigs as one vendor-specific process per backend:
#   AMD:    hiveos/build-amd-glibc.sh && package-amd.sh
#   NVIDIA: hiveos/build-release.sh (or the legacy/modern line packagers)
# Scope each process to its own devices with the corresponding device-selection
# option. A future universal package needs separate pom-opencl and pom-cuda host
# executables plus a launcher; merely combining plugin libraries is insufficient.
set -euo pipefail

echo "ERROR: package-universal.sh is retired: one miner process cannot safely mix pom-opencl and pom-cuda backends." >&2
echo "       Run separate AMD and NVIDIA packages/processes on mixed-vendor rigs (see this script's header)." >&2
exit 2
