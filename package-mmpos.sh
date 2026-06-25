#!/bin/bash
# Assemble the mmpOS external-miner package for keryx-miner-supr (NVIDIA, PoM).
# Output: keryx-miner-supr-mmpos_<version>.tar.gz  (internal dir = same name).
#
# Like the HiveOS package, the full-parity PoM binary needs candle's CUDA runtime
# (libcublas/libcublasLt/libcurand) at load, so they are bundled into ./lib and
# mmp-launch.sh prepends ./lib to LD_LIBRARY_PATH. Stats come from the miner's own
# /mmpos HTTP endpoint (mmp-launch starts it with --api-bind).
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"

NAME="keryx-miner-supr"
VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
PKG="${NAME}-mmpos_${VERSION}"
BIN="$REPO/hiveos/dist/keryx-miner-supr"
SRC="$REPO/mmpos/keryx-miner-supr"
CUDA_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib

[[ -x "$BIN" ]] || { echo "ERROR: $BIN missing — run hiveos/build-glibc.sh first"; exit 1; }

STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$PKG"
mkdir -p "$DEST/lib"

cp "$BIN" "$DEST/keryx-miner-supr"
cp "$SRC/mmp-external.conf" "$SRC/mmp-launch.sh" "$SRC/mmp-stats.sh" "$DEST/"
chmod +x "$DEST/keryx-miner-supr" "$DEST"/*.sh

# Pin the version in the manifest.
sed -i "s/^EXTERNAL_VERSION=.*/EXTERNAL_VERSION=\"${VERSION}\"/" "$DEST/mmp-external.conf"

# Bundle candle's CUDA runtime libs.
for l in libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do
    [[ -f "$CUDA_LIB/$l" ]] || { echo "ERROR: $CUDA_LIB/$l missing — install the CUDA 12.8 runtime"; exit 1; }
    cp -L "$CUDA_LIB/$l" "$DEST/lib/"
done
echo ">> bundled CUDA runtime libs ($(du -sh "$DEST/lib" | cut -f1)): $(ls "$DEST/lib")"

OUT="$REPO/hiveos/dist/${PKG}.tar.gz"
tar -czf "$OUT" -C "$STAGE" "$PKG"
echo ">> Wrote $OUT ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT"
echo ">> sha256:"; sha256sum "$OUT"
