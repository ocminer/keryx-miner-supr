#!/usr/bin/env bash
# Assemble the LEGACY (CUDA 12.4, driver floor 550) HiveOS/Linux tarball from
# hiveos/dist-legacy/ (produced by build-glibc-legacy.sh). Bundles the FULL 12.4
# runtime incl. libcudart so the floor really is 550 regardless of host CUDA.
# Output: hiveos/dist-legacy/keryx-miner-supr-<version>-cuda124.tar.gz
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$REPO/hiveos/dist-legacy"
PKG="$REPO/hiveos/pkg/keryx-miner-supr"
NAME=keryx-miner-supr
VERSION=$(grep -m1 '^CUSTOM_VERSION=' "$PKG/h-manifest.conf" | cut -d= -f2)

[[ -f "$DIST/keryx-miner-supr" ]] || { echo "ERROR: $DIST/keryx-miner-supr missing — run hiveos/build-glibc-legacy.sh first"; exit 1; }

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/$NAME"
mkdir -p "$DEST/lib"

cp "$PKG"/h-manifest.conf "$PKG"/h-config.sh "$PKG"/h-run.sh "$PKG"/h-stats.sh "$DEST/"
cp "$DIST"/keryx-miner-supr "$DEST/"
chmod +x "$DEST"/h-*.sh "$DEST"/keryx-miner-supr

# Bundle the 12.4 runtime extracted from the build container (incl. libcudart).
for l in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do
  [[ -f "$DIST/lib/$l" ]] || { echo "ERROR: $DIST/lib/$l missing — re-run build-glibc-legacy.sh"; exit 1; }
  cp -L "$DIST/lib/$l" "$DEST/lib/"
done
echo ">> bundled 12.4 runtime ($(du -sh "$DEST/lib" | cut -f1)): $(ls "$DEST/lib")"

TARBALL="$DIST/${NAME}-${VERSION}-cuda124.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
echo ">> Wrote $TARBALL"
tar -tzf "$TARBALL"
echo ">> sha256:"; sha256sum "$TARBALL"
