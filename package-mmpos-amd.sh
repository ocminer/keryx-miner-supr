#!/bin/bash
# Assemble the mmpOS external-miner package for keryx-miner-supr (AMD/OpenCL, PoM).
# Output: hiveos/dist-amd/keryx-miner-supr-amd-mmpos_<version>.tar.gz
#
# Unlike the NVIDIA mmpOS package (bundles candle's CUDA runtime), the AMD flavour
# ships the dynamic binary + the dlopen'd libkeryxopencl.so. The GPU driver
# provides libOpenCL.so.1 at runtime; OPoI inference runs on CPU. Stats come from
# the miner's own /mmpos HTTP endpoint (mmp-launch starts it with --api-bind).
#
# Prereq: hiveos/build-amd-glibc.sh (produces hiveos/dist-amd/{binary,libkeryxopencl.so}).
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

NAME="keryx-miner-supr-amd"
VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
PKG="${NAME}-mmpos_${VERSION}"
DIST="$REPO/hiveos/dist-amd"
BIN="$DIST/keryx-miner-supr"
PLUGIN="$DIST/libkeryxopencl.so"
SRC="$REPO/mmpos/keryx-miner-supr-amd"

[[ -x "$BIN" ]]    || { echo "ERROR: $BIN missing — run hiveos/build-amd-glibc.sh first"; exit 1; }
[[ -f "$PLUGIN" ]] || { echo "ERROR: $PLUGIN missing — run hiveos/build-amd-glibc.sh first"; exit 1; }

# Sanity: the PoM OpenCL kernel must be embedded, else AMD mines the dead algo post-fork.
# (grep -c reads all input so `strings` doesn't take SIGPIPE under `set -o pipefail`.)
if [[ "$(strings "$BIN" | grep -c '__kernel void pom_mine' || true)" -eq 0 ]]; then
    echo "ERROR: $BIN has no embedded PoM kernel — was it built with --features pom-opencl?"; exit 1
fi

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$PKG"
mkdir -p "$DEST"

cp "$BIN" "$DEST/keryx-miner-supr"
cp "$PLUGIN" "$DEST/libkeryxopencl.so"
cp "$SRC/mmp-external.conf" "$SRC/mmp-launch.sh" "$SRC/mmp-stats.sh" "$DEST/"
chmod +x "$DEST/keryx-miner-supr" "$DEST"/*.sh

# Pin the version in the manifest.
sed -i "s/^EXTERNAL_VERSION=.*/EXTERNAL_VERSION=\"${VERSION}\"/" "$DEST/mmp-external.conf"

OUT="$DIST/${PKG}.tar.gz"
tar -czf "$OUT" -C "$STAGE" "$PKG"
echo ">> Wrote $OUT ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT"
echo ">> sha256:"; sha256sum "$OUT"
