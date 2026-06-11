#!/bin/bash
#
# Assemble the distributable AMD/OpenCL package: the dynamic keryx-miner-supr
# binary + libkeryxopencl.so + a RUN.txt, as keryx-miner-supr-amd-<version>.tar.gz.
#
# By default it ships the portable OLD-glibc artifacts from hiveos/dist-amd/
# (built by hiveos/build-amd-glibc.sh — runs on HiveOS, Ubuntu 20.04+, etc.).
# Pass --native to package the local dist-amd/ build instead (glibc of this host;
# only for distros as new as the build box).
#
set -e
cd "$(dirname "$0")"
REPO="$(pwd)"

SRC="$REPO/hiveos/dist-amd"          # portable old-glibc build (default)
LABEL="portable (glibc 2.31)"
if [ "${1:-}" = "--native" ]; then
    SRC="$REPO/dist-amd"
    LABEL="native ($(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1) glibc)"
fi

VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
NAME="keryx-miner-supr"
PKGNAME="keryx-miner-supr-amd-${VERSION}"
OUT="$REPO/dist-pkg"
DEST="$OUT/$PKGNAME"

if [ ! -x "$SRC/$NAME" ] || [ ! -f "$SRC/libkeryxopencl.so" ]; then
    echo "ERROR: AMD artifacts missing in $SRC" >&2
    echo "  build them first:  ${1:-} hiveos/build-amd-glibc.sh   (or ./build-amd.sh for --native)" >&2
    exit 1
fi

rm -rf "$DEST"; mkdir -p "$DEST"
cp "$SRC/$NAME" "$DEST/"
cp "$SRC/libkeryxopencl.so" "$DEST/"
chmod +x "$DEST/$NAME"

cat > "$DEST/RUN.txt" <<TXT
keryx-miner-supr — AMD/OpenCL package (v${VERSION})

A KeryxHash miner for AMD GPUs (OpenCL). The binary dlopens libkeryxopencl.so
from its own directory; it auto-detects the AMD OpenCL platform and mines on all
AMD GPUs it finds. Needs an AMD OpenCL runtime (libOpenCL.so.1 from the AMD/ROCm
driver) — no CUDA, no OpenCL SDK.

Run:
  ./keryx-miner-supr \\
      -a keryx:<your_address>.<worker> \\
      -s stratum+tcp://krx.suprnova.cc:4401 \\
      --light
(--light = TinyLlama only; drop it for higher LLM tiers / more VRAM.)

Useful flags:
  --opencl-device 0,1     mine on specific GPU indices (default: all)
  --opencl-workload N     nonces-per-dispatch ratio. AUTO by default — picks a
                          capability-driven value per arch (gfx906 MI50 -> 2048,
                          gfx1102/RDNA3 -> 4096). Override only to tune.
  --disable-gpu           CPU-only mining (testing / GPU-less boxes).

Performance is the v_dot8 packed-4-bit matmul + per-arch workload defaults: e.g.
RX 7600 XT ~332 MH/s, MI50 ~421 MH/s peak (thermally capped), 0 rejects.
TXT

mkdir -p "$OUT"
TARBALL="$OUT/${PKGNAME}.tar.gz"
tar -czf "$TARBALL" -C "$OUT" "$PKGNAME"

echo ""
echo "[package-amd] $LABEL package ready:"
ls -la "$DEST"
echo ""
echo "  archive: $TARBALL"
echo "  sha256:  $(sha256sum "$TARBALL" | cut -d' ' -f1)"
