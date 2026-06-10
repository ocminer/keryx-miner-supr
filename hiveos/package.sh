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

TARBALL="$DIST/${NAME}-${VERSION}.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
echo ">> Wrote $TARBALL"
tar -tzf "$TARBALL"
echo ">> sha256:"; sha256sum "$TARBALL"
