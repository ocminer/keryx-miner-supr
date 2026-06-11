#!/bin/bash
#
# Assemble a single "both worlds" keryx-miner-supr package that auto-detects the
# GPU vendor at runtime: one binary + both plugin .so files.
#
#   keryx-miner-supr     vendor-agnostic host (dynamic build, CUDA-free; the
#                         binary itself needs no CUDA or OpenCL toolkit).
#   libkeryxopencl.so    AMD path  — enables on an AMD OpenCL platform.
#   libkeryxcuda.so      NVIDIA path — enables on a CUDA device.
#
# At startup the binary dlopens whichever .so sit next to it; each plugin
# self-enables ONLY for its vendor's GPUs. So the same folder runs on a pure-AMD
# rig, a pure-NVIDIA rig, or a mixed rig (each plugin claims its own cards). A
# .so whose runtime is absent (e.g. libkeryxcuda.so on an AMD-only box, which
# can't find libcuda.so.1) is logged and skipped — not fatal.
#
# Build logistics: the binary + libkeryxopencl.so build on an AMD/OpenCL host
# (./build-amd.sh, no CUDA toolkit needed). libkeryxcuda.so must be built on an
# NVIDIA/CUDA host. This script combines the two — build the CUDA side on the
# NVIDIA box, copy libkeryxcuda.so here, then run this.
#
# Usage:
#   ./package-universal.sh [path/to/libkeryxcuda.so]
#
# The CUDA plugin is resolved from, in order: $1, $LIBKERYXCUDA,
# ./libkeryxcuda.so, dist-amd/libkeryxcuda.so. If none is found the package is
# assembled AMD-only and prints how to add NVIDIA support later (just drop
# libkeryxcuda.so into the output dir — no rebuild needed).
#
set -e
cd "$(dirname "$0")"
REPO="$(pwd)"

VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
NAME="keryx-miner-supr"
OUT="$REPO/dist-universal"
DEST="$OUT/$NAME"

# --- AMD side: ensure binary + libkeryxopencl.so exist -------------------------
if [ ! -x "$REPO/dist-amd/$NAME" ] || [ ! -f "$REPO/dist-amd/libkeryxopencl.so" ]; then
    echo "[universal] AMD artifacts missing — running ./build-amd.sh ..."
    ./build-amd.sh
fi

# --- NVIDIA side: locate a prebuilt libkeryxcuda.so ----------------------------
CUDA_SO=""
for cand in "$1" "$LIBKERYXCUDA" "$REPO/libkeryxcuda.so" "$REPO/dist-amd/libkeryxcuda.so" \
            "$REPO/target/release/libkeryxcuda.so"; do
    if [ -n "$cand" ] && [ -f "$cand" ]; then CUDA_SO="$cand"; break; fi
done

# --- assemble ------------------------------------------------------------------
rm -rf "$OUT"
mkdir -p "$DEST"
cp "$REPO/dist-amd/$NAME" "$DEST/"
cp "$REPO/dist-amd/libkeryxopencl.so" "$DEST/"
chmod +x "$DEST/$NAME"

if [ -n "$CUDA_SO" ]; then
    cp "$CUDA_SO" "$DEST/libkeryxcuda.so"
    PLUGINS="libkeryxopencl.so + libkeryxcuda.so (AMD + NVIDIA)"
else
    PLUGINS="libkeryxopencl.so (AMD only — see RUN.txt to add NVIDIA)"
fi

cat > "$DEST/RUN.txt" <<TXT
keryx-miner-supr — universal package (v${VERSION})

Auto-detects the GPU vendor at runtime. The binary loads the plugin .so files
from its own directory; each enables ONLY for its vendor's GPUs:
  libkeryxopencl.so -> AMD   (enables on an AMD OpenCL platform)
  libkeryxcuda.so   -> NVIDIA (enables on a CUDA device; needs the NVIDIA driver)

This build ships: ${PLUGINS}

Run:
  ./keryx-miner-supr \\
      -a keryx:<your_address>.<worker> \\
      -s stratum+tcp://krx.suprnova.cc:4401 \\
      --light
(--light = TinyLlama only; drop it for higher LLM tiers / more VRAM.)

AMD notes: v_dot8 is the no-flag default on MI50/MI60 (gfx906) and RDNA 3/4.
NVIDIA notes: needs libcuda.so.1 from the NVIDIA driver (CUDA toolkit NOT needed
at runtime). To add NVIDIA support to an AMD-only package, build libkeryxcuda.so
on an NVIDIA/CUDA host (cargo build --release -p keryxcuda) and drop it into this
folder next to the binary — no rebuild of the miner needed.
TXT

TARBALL="$REPO/${NAME}-${VERSION}-universal.tar.gz"
tar -czf "$TARBALL" -C "$OUT" "$NAME"

echo ""
echo "[universal] package ready:"
ls -la "$DEST"
echo ""
echo "  archive: $TARBALL"
[ -z "$CUDA_SO" ] && echo "  NOTE: AMD-only (no libkeryxcuda.so found). See $DEST/RUN.txt to add NVIDIA." || true
