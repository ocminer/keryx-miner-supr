#!/usr/bin/env bash
# Assemble the SMOS (SimpleMiningOS) custom-miner tarball for keryx-miner-supr.
# SMOS accepts HiveOS-style custom miners, so this mirrors hiveos/package.sh:
# the single static-cuda binary + bundled CUDA runtime (./lib) + the h-*.sh
# integration scripts (h-run.sh adds ./lib to LD_LIBRARY_PATH for OPoI cuBLAS).
# Output: hiveos/dist/keryx-miner-supr-smos_<version>.tar.gz
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$REPO/hiveos/dist"
PKG="$REPO/hiveos/pkg/keryx-miner-supr"
NAME=keryx-miner-supr
VERSION=$(grep -m1 '^CUSTOM_VERSION=' "$PKG/h-manifest.conf" | cut -d= -f2)
CUDA_LIB=/usr/local/cuda-12.8/targets/x86_64-linux/lib

[[ -f "$DIST/keryx-miner-supr" ]] || { echo "ERROR: $DIST/keryx-miner-supr missing — run hiveos/build-release.sh first"; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/${NAME}-smos_${VERSION}"
mkdir -p "$DEST/lib"

cp "$PKG"/h-manifest.conf "$PKG"/h-config.sh "$PKG"/h-run.sh "$PKG"/h-stats.sh "$DEST/"
cp "$DIST"/keryx-miner-supr "$DEST/"
chmod +x "$DEST"/h-*.sh "$DEST"/keryx-miner-supr

for l in libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do
  [[ -f "$CUDA_LIB/$l" ]] || { echo "ERROR: $CUDA_LIB/$l missing — install the CUDA 12.8 runtime"; exit 1; }
  cp -L "$CUDA_LIB/$l" "$DEST/lib/"
done
echo ">> bundled CUDA runtime libs ($(du -sh "$DEST/lib" | cut -f1)): $(ls "$DEST/lib")"

TARBALL="$DIST/${NAME}-smos_${VERSION}.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "${NAME}-smos_${VERSION}"
echo ">> Wrote $TARBALL"
tar -tzf "$TARBALL"
echo ">> sha256:"; sha256sum "$TARBALL"
