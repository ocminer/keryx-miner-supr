#!/usr/bin/env bash
# Assemble the HiveOS custom-miner tarball for the AMD/OpenCL build, from the
# old-glibc artifacts (hiveos/dist-amd/, produced by build-amd-glibc.sh) + the
# AMD h-*.sh integration scripts (hiveos/pkg-amd/).
#
# Unlike hiveos/package.sh (single static-cuda NVIDIA binary), the AMD payload
# is the dynamic binary PLUS libkeryxopencl.so dlopened next to it.
#
# Output: hiveos/dist-amd/keryx-miner-supr-amd-<version>-hiveos.tar.gz
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$REPO/hiveos/dist-amd"
PKG="$REPO/hiveos/pkg-amd/keryx-miner-supr-amd"
NAME=keryx-miner-supr-amd            # HiveOS custom-miner name (install dir)
BIN=keryx-miner-supr                 # the actual executable inside the package
VERSION=$(grep -m1 '^CUSTOM_VERSION=' "$PKG/h-manifest.conf" | cut -d= -f2)

[[ -f "$DIST/$BIN" ]]               || { echo "ERROR: $DIST/$BIN missing — run hiveos/build-amd-glibc.sh first"; exit 1; }
[[ -f "$DIST/libkeryxopencl.so" ]] || { echo "ERROR: $DIST/libkeryxopencl.so missing — run hiveos/build-amd-glibc.sh first"; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$NAME"
mkdir -p "$DEST"

# integration scripts + dynamic binary + OpenCL plugin
cp "$PKG"/h-manifest.conf "$PKG"/h-config.sh "$PKG"/h-run.sh "$PKG"/h-stats.sh "$DEST/"
cp "$DIST/$BIN" "$DEST/"
cp "$DIST/libkeryxopencl.so" "$DEST/"
chmod +x "$DEST"/h-*.sh "$DEST/$BIN"

TARBALL="$DIST/${NAME}-${VERSION}-hiveos.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
echo ">> Wrote $TARBALL"
tar -tzf "$TARBALL"
echo ">> sha256:"; sha256sum "$TARBALL"
