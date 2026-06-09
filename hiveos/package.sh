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

for f in keryx-miner-supr libkeryxcuda.so libkeryxopencl.so; do
  [[ -f "$DIST/$f" ]] || { echo "ERROR: $DIST/$f missing — run hiveos/build-glibc.sh first"; exit 1; }
done

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$NAME"
mkdir -p "$DEST"

# integration scripts
cp "$PKG"/h-manifest.conf "$PKG"/h-config.sh "$PKG"/h-run.sh "$PKG"/h-stats.sh "$DEST/"
# binaries
cp "$DIST"/keryx-miner-supr "$DIST"/libkeryxcuda.so "$DIST"/libkeryxopencl.so "$DEST/"
chmod +x "$DEST"/h-*.sh "$DEST"/keryx-miner-supr

TARBALL="$DIST/${NAME}-${VERSION}.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
echo ">> Wrote $TARBALL"
tar -tzf "$TARBALL"
echo ">> sha256:"; sha256sum "$TARBALL"
