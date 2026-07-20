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
# Use the same release label as the HiveOS AMD package (CUSTOM_VERSION) so both match — the
# HiveOS label can carry a 4th component (e.g. 0.6.3.1) that Cargo semver can't.
VERSION=$(grep -m1 '^CUSTOM_VERSION=' "$REPO/hiveos/pkg-amd/keryx-miner-supr-amd/h-manifest.conf" | cut -d= -f2)
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
# Zero-dup in-process engine (optional): libkeryx-llama-vk.so — llama.cpp hosts the model in VRAM,
# serves OPoI inference in-process, and (on RDNA cards) hosts the PoM walk over the same resident
# weights (no OpenCL blob for that card). Absent = the llama-server subprocess fallback below.
if [[ -f "$DIST/libkeryx-llama-vk.so" ]]; then
  cp -P "$DIST/libkeryx-llama-vk.so" "$DEST/"
  echo ">> bundled zero-dup in-process engine (libkeryx-llama-vk.so)"
fi
# Baseline CPU-ISA build for pre-AVX rig CPUs — auto-selected by the miner when the host CPU lacks
# the AVX2/FMA/F16C/BMI2 set baked into the default .so (v0.7.2 SIGILL fix).
if [[ -f "$DIST/libkeryx-llama-vk-noavx.so" ]]; then
  cp -P "$DIST/libkeryx-llama-vk-noavx.so" "$DEST/"
  echo ">> bundled no-AVX in-process engine (libkeryx-llama-vk-noavx.so)"
fi
# Vulkan loader (libvulkan.so.1): our vk .so + ggml-vulkan + llama-server all NEED it, but it's a
# SYSTEM library (libvulkan1) absent on stock HiveOS/mmpOS where users can't apt-install → the miner
# died with "libvulkan.so.1: cannot open shared object file". Bundle the glibc-2.29 loader from the
# ubuntu-20.04 build image; the launcher's LD_LIBRARY_PATH resolves it. Reaching the GPU still needs
# an AMD ICD (mesa RADV); if absent the miner falls back to CPU inference and keeps mining.
if [[ -f "$DIST/libvulkan.so.1" ]]; then
  cp -P "$DIST"/libvulkan.so.1* "$DEST/"
  echo ">> bundled Vulkan loader (libvulkan.so.1 -> $(readlink "$DIST/libvulkan.so.1"))"
fi
# Vulkan GPU inference (fallback): bundle llama-server + its ggml/llama .so (miner spawns it for
# OPoI inference only when the in-process engine is unavailable; absent / no Vulkan ICD → CPU).
if [[ -f "$DIST/llama-server" ]]; then
  cp -P "$DIST/llama-server" "$DEST/"
  cp -P "$DIST"/lib{ggml,llama,mtmd}*.so* "$DEST/" 2>/dev/null || true
  chmod +x "$DEST/llama-server"
  echo ">> bundled Vulkan GPU inference (llama-server + ggml/llama libs)"
fi
cp "$SRC/mmp-external.conf" "$SRC/mmp-launch.sh" "$SRC/mmp-launcher.sh" "$SRC/mmp-stats.sh" "$SRC/mmp-release-notes.txt" "$DEST/"
chmod +x "$DEST/keryx-miner-supr" "$DEST"/*.sh

# Pin the version in the manifest.
sed -i "s/^EXTERNAL_VERSION=.*/EXTERNAL_VERSION=\"${VERSION}\"/" "$DEST/mmp-external.conf"

OUT="$DIST/${PKG}.tar.gz"
tar -czf "$OUT" -C "$STAGE" "$PKG"
echo ">> Wrote $OUT ($(du -h "$OUT" | cut -f1))"
tar -tzf "$OUT"
echo ">> sha256:"; sha256sum "$OUT"
