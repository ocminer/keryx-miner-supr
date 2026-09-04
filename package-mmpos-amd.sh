#!/bin/bash
# Assemble the mmpOS external-miner package for keryx-miner-supr (AMD/OpenCL, PoM).
# Output: hiveos/dist-amd/keryx-miner-supr-amd-mmpos_<version>.tar.gz
#
# Unlike the NVIDIA mmpOS package (bundles candle's CUDA runtime), the AMD flavour
# ships the dynamic binary + the dlopen'd libkeryxopencl.so. The GPU driver
# provides libOpenCL.so.1 at runtime; OPoI inference uses packaged Vulkan GPU routes. Stats come from
# the miner's own /mmpos HTTP endpoint (mmp-launch starts it with --api-bind).
#
# Prereq: hiveos/build-amd-glibc.sh (produces hiveos/dist-amd/{binary,libkeryxopencl.so}).
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

NAME="keryx-miner-supr-amd"
# Use the same release label as the HiveOS AMD package (CUSTOM_VERSION) so both match — the
# HiveOS label can carry a 4th component (e.g. 0.6.3.1) that Cargo semver can't.
VERSION=$(grep -m1 '^CUSTOM_VERSION=' "$REPO/hiveos/pkg-amd/keryx-miner-supr-amd/h-manifest.conf" | cut -d= -f2)
PKG="${NAME}-mmpos_${VERSION}"
DIST="$REPO/hiveos/dist-amd"
BIN="$DIST/keryx-miner-supr"
PLUGIN="$DIST/libkeryxopencl.so"
SRC="$REPO/mmpos/keryx-miner-supr-amd"
# shellcheck source=hiveos/amd-inference-route.sh
source "$REPO/hiveos/amd-inference-route.sh"

[[ -x "$BIN" ]]    || { echo "ERROR: $BIN missing — run hiveos/build-amd-glibc.sh first"; exit 1; }
[[ -f "$PLUGIN" ]] || { echo "ERROR: $PLUGIN missing — run hiveos/build-amd-glibc.sh first"; exit 1; }
keryx_require_amd_inference_route "$DIST"

# Sanity: the PoM OpenCL kernel must be embedded, else AMD mines the dead algo post-fork.
# Match the kernel NAME (pom_mine_v4), not `__kernel void pom_mine`: the v4 grind kernel carries a
# reqd_work_group_size attribute between `__kernel` and `void`, so the old contiguous pattern no
# longer matches the current source (it would false-fail every v0.11.0+ build).
# (grep -c reads all input so `strings` doesn't take SIGPIPE under `set -o pipefail`.)
if [[ "$(strings "$BIN" | grep -c 'pom_mine_v4' || true)" -eq 0 ]]; then
    echo "ERROR: $BIN has no embedded PoM v4 kernel — was it built with --features pom-opencl?"; exit 1
fi

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$PKG"
mkdir -p "$DEST"

cp "$BIN" "$DEST/keryx-miner-supr"
cp "$PLUGIN" "$DEST/libkeryxopencl.so"
# Required GPU inference route + Vulkan loader; any optional server is copied only with its exact
# DT_NEEDED llama/ggml closure. Packaging fails before creating an unusable archive.
keryx_copy_amd_inference_route "$DIST" "$DEST"
cp "$SRC/mmp-external.conf" "$SRC/mmp-launch.sh" "$SRC/mmp-launcher.sh" "$SRC/mmp-stats.sh" "$SRC/mmp-release-notes.txt" "$DEST/"
chmod +x "$DEST/keryx-miner-supr" "$DEST"/*.sh

# Pin the version in the manifest.
sed -i "s/^EXTERNAL_VERSION=.*/EXTERNAL_VERSION=\"${VERSION}\"/" "$DEST/mmp-external.conf"

OUT="$DIST/${PKG}.tar.gz"
tar -czf "$OUT" -C "$STAGE" "$PKG"
echo ">> Wrote $OUT ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT"
sha256sum "$OUT" > "$OUT.sha256"
echo ">> sha256:"; cat "$OUT.sha256"
echo ">> checksum: $OUT.sha256"
