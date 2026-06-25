#!/usr/bin/env bash
# Assemble the HiveOS custom-miner tarball from the old-glibc binaries
# (hiveos/dist/, produced by build-glibc.sh) + the h-*.sh integration scripts
# (hiveos/pkg/). Output: hiveos/dist/keryx-miner-supr-<version>.tar.gz
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$REPO/hiveos/dist"
PKG="$REPO/hiveos/pkg/keryx-miner-supr"
NAME=keryx-miner-supr
VERSION=$(grep -m1 '^CUSTOM_VERSION=' "$PKG/h-manifest.conf" | cut -d= -f2)

# Single self-contained binary (static-cuda) — no .so plugins to ship.
[[ -f "$DIST/keryx-miner-supr" ]] || { echo "ERROR: $DIST/keryx-miner-supr missing — run hiveos/build-glibc.sh first"; exit 1; }
# Drop any stale .so from earlier (dynamic) builds so they don't get bundled.
rm -f "$DIST"/libkeryxcuda.so "$DIST"/libkeryxopencl.so

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$NAME"
mkdir -p "$DEST"

# integration scripts
cp "$PKG"/h-manifest.conf "$PKG"/h-config.sh "$PKG"/h-run.sh "$PKG"/h-stats.sh "$DEST/"
# single static binary
cp "$DIST"/keryx-miner-supr "$DEST/"
chmod +x "$DEST"/h-*.sh "$DEST"/keryx-miner-supr

# Bundle candle's CUDA runtime libs (the full-parity PoM build links cuBLAS/cuRAND
# for OPoI inference; HiveOS rigs ship only the driver libcuda). h-run.sh adds
# ./lib to LD_LIBRARY_PATH. cudart is statically linked, so it is not bundled.
CUDA_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib
mkdir -p "$DEST/lib"
for l in libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do
  [[ -f "$CUDA_LIB/$l" ]] || { echo "ERROR: $CUDA_LIB/$l missing — install the CUDA 12.8 runtime"; exit 1; }
  cp -L "$CUDA_LIB/$l" "$DEST/lib/"
done
echo ">> bundled CUDA runtime libs ($(du -sh "$DEST/lib" | cut -f1)): $(ls "$DEST/lib")"

TARBALL="$DIST/${NAME}-${VERSION}.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
echo ">> Wrote $TARBALL"
tar -tzf "$TARBALL"
echo ">> sha256:"; sha256sum "$TARBALL"
