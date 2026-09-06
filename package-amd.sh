#!/bin/bash
#
# Assemble the distributable AMD/OpenCL package: the dynamic keryx-miner-supr
# binary + libkeryxopencl.so + a RUN.txt, as keryx-miner-supr-amd-<version>.tar.gz.
#
# By default it ships the portable OLD-glibc artifacts from hiveos/dist-amd/
# (built by hiveos/build-amd-glibc.sh — runs on HiveOS, Ubuntu 20.04+, etc.).
# Native packaging is deliberately retired: `build-amd.sh` does not produce the
# mandatory, source-bound Vulkan inference route. Use the portable builder only.
#
set -e
cd "$(dirname "$0")"
REPO="$(pwd)"
# shellcheck source=hiveos/amd-inference-route.sh
source "$REPO/hiveos/amd-inference-route.sh"

SRC="$REPO/hiveos/dist-amd"          # portable old-glibc build (default)
LABEL="portable (glibc 2.31)"
if [ "${1:-}" = "--native" ]; then
    echo "ERROR: --native was retired because it cannot produce the mandatory GPU inference route." >&2
    echo "Use: hiveos/build-amd-glibc.sh && ./package-amd.sh" >&2
    exit 2
elif [ -n "${1:-}" ]; then
    echo "ERROR: unknown argument: $1" >&2
    exit 2
fi

VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
NAME="keryx-miner-supr"
PKGNAME="keryx-miner-supr-amd-${VERSION}"
OUT="$REPO/dist-pkg"
DEST="$OUT/$PKGNAME"

if [ ! -x "$SRC/$NAME" ] || [ ! -f "$SRC/libkeryxopencl.so" ]; then
    echo "ERROR: AMD artifacts missing in $SRC" >&2
    echo "  build them first: hiveos/build-amd-glibc.sh" >&2
    exit 1
fi
keryx_require_amd_inference_route "$SRC"

rm -rf "$DEST"; mkdir -p "$DEST"
cp "$SRC/$NAME" "$DEST/"
cp "$SRC/libkeryxopencl.so" "$DEST/"
keryx_copy_amd_inference_route "$SRC" "$DEST"
cat > "$DEST/run.sh" <<'SH'
#!/usr/bin/env bash
set -e
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export LD_LIBRARY_PATH="$HERE${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/keryx-miner-supr" "$@"
SH
chmod +x "$DEST/$NAME" "$DEST/run.sh"

cat > "$DEST/RUN.txt" <<TXT
keryx-miner-supr — AMD/OpenCL package (v${VERSION})

A KeryxHash miner for AMD GPUs (OpenCL). The binary dlopens libkeryxopencl.so
for mining and the bundled libkeryx-llama-vk.so for required OPoI GPU inference.
The portable Vulkan loader is included; the AMD driver must still provide both
the OpenCL runtime (libOpenCL.so.1) and a Vulkan ICD. No CUDA toolkit is needed.

Run:
  ./run.sh \\
      -a keryx:<your_address>.<worker> \\
      -s stratum+tcp://krx.suprnova.cc:4401 \\
      --very-light
(--very-light pins H6 tier 0, Qwen3.5-9B; omit it for per-card AUTO tier selection.)

Useful flags:
  --opencl-device 0,1     mine on specific GPU indices (default: all)
  --opencl-workload N     nonces-per-dispatch ratio. AUTO by default — picks a
                          capability-driven value per arch (gfx906 MI50 -> 2048,
                          gfx1102/RDNA3 -> 4096). Override only to tune.
  --enable-cpu-inference  deprecated emergency-only inference fallback; normally omit.

PoM v4 reference pool measurements (tier-0 model): RX 7900 XTX gfx1100
~1.781 MH/s with its exact-tested one-state WMMA path; MI50 gfx906 ~0.521 MH/s
with DP4A. gfx1102/gfx12 retain the exact-tested multi-state WMMA path.
TXT

mkdir -p "$OUT"

# --- Standalone (manual / docker) tarball: <name>-<ver>-linux-x86_64.tar.gz (run.sh + RUN.txt) ---
STANDALONE="$OUT/${PKGNAME}-linux-x86_64.tar.gz"
tar -czf "$STANDALONE" -C "$OUT" "$PKGNAME"

# --- HiveOS custom-miner package: bare <name>-<ver>.tar.gz ---
# HiveOS custom-get splits the filename on the LAST '-' for the version and extracts the single
# top-level directory (which must equal CUSTOM_NAME) into /hive/miners/custom/. So the bare name is
# the HiveOS package: it bundles the h-* integration files from hiveos/pkg-amd next to the same
# binary + libkeryxopencl.so + Vulkan inference route (NOT run.sh/RUN.txt — those are standalone).
HPKG="$REPO/hiveos/pkg-amd/keryx-miner-supr-amd"
HVN="keryx-miner-supr-amd"
HSTAGE="$OUT/hv-amd"
HDEST="$HSTAGE/$HVN"
rm -rf "$HSTAGE"; mkdir -p "$HDEST"
cp "$HPKG"/h-manifest.conf "$HPKG"/h-config.sh "$HPKG"/h-run.sh "$HPKG"/h-stats.sh "$HDEST/"
cp "$SRC/$NAME" "$SRC/libkeryxopencl.so" "$HDEST/"
keryx_copy_amd_inference_route "$SRC" "$HDEST"
chmod +x "$HDEST/$NAME" "$HDEST/h-run.sh" "$HDEST/h-stats.sh" "$HDEST/h-config.sh"
TARBALL="$OUT/${PKGNAME}.tar.gz"
tar -czf "$TARBALL" -C "$HSTAGE" "$HVN"

echo ""
echo "[package-amd] $LABEL packages ready:"
echo "  HiveOS:     $TARBALL  (top-dir $HVN/, h-* integration)"
echo "  standalone: $STANDALONE  (run.sh / RUN.txt)"
for t in "$TARBALL" "$STANDALONE"; do
    sha256sum "$t" > "$t.sha256"
    echo "  sha256 $(basename "$t"): $(cut -d' ' -f1 < "$t.sha256")"
done
