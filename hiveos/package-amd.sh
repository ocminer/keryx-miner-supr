#!/usr/bin/env bash
# Assemble the HiveOS custom-miner tarball for the AMD/OpenCL build, from the
# old-glibc artifacts (hiveos/dist-amd/, produced by build-amd-glibc.sh) + the
# AMD h-*.sh integration scripts (hiveos/pkg-amd/).
#
# Unlike hiveos/package.sh (single static-cuda NVIDIA binary), the AMD payload
# is the dynamic binary PLUS libkeryxopencl.so dlopened next to it.
#
# Output: hiveos/dist-amd/keryx-miner-supr-amd-<version>.tar.gz
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
# Vulkan GPU inference (optional): bundle llama-server + its ggml/llama .so. The miner spawns it
# for OPoI inference on the AMD GPU; if absent (or no Vulkan ICD on the rig), it falls back to CPU.
if [[ -f "$DIST/llama-server" ]]; then
  cp -P "$DIST/llama-server" "$DEST/"
  cp -P "$DIST"/lib{ggml,llama,mtmd}*.so* "$DEST/" 2>/dev/null || true
  chmod +x "$DEST/llama-server"
  echo ">> bundled Vulkan GPU inference (llama-server + $(ls "$DEST"/lib{ggml,llama,mtmd}*.so* 2>/dev/null | wc -l) libs)"
fi
chmod +x "$DEST"/h-*.sh "$DEST/$BIN"

# HiveOS custom-get derives the miner NAME by stripping the LAST hyphen-delimited
# field as the "version": basename | awk -F- '{print $NF}'. So the archive MUST be
# named "<NAME>-<VERSION>.tar.gz" with NO extra hyphenated suffix and NO hyphen in
# the version, or it mis-parses the name and the install dir won't match the tar's
# internal "$NAME/" folder (chown/sed fail -> "Miner screen is not running").
# A "-hiveos" suffix here is exactly that bug — DO NOT add it. (NVIDIA build-release.sh
# documents the same rule.) e.g. keryx-miner-supr-amd-0.5.4.tar.gz -> NAME=keryx-miner-supr-amd.
TARBALL="$DIST/${NAME}-${VERSION}.tar.gz"
tar -czf "$TARBALL" -C "$STAGE" "$NAME"
echo ">> Wrote $TARBALL"
tar -tzf "$TARBALL"
echo ">> sha256:"; sha256sum "$TARBALL"
