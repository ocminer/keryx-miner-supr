#!/bin/bash
#
# Build the AMD/OpenCL flavour of keryx-miner-supr.
#
# Produces, in ./dist-amd/:
#   keryx-miner-supr      - the miner binary (dynamic plugin build, CUDA-free,
#                           candle falls back to CPU; OPoI per-share tag still works)
#   libkeryxopencl.so     - the OpenCL worker plugin (dlopened at runtime)
#
# The miner auto-detects AMD: on a box with an AMD OpenCL platform the OpenCL
# plugin self-enables (plugins/opencl/src/lib.rs); on NVIDIA it stays dormant and
# libkeryxcuda.so handles the GPU. Drop both .so next to the binary for a mixed rig.
#
# This does NOT build plugins/cuda (that needs the CUDA toolkit). The CUDA path is
# unchanged — build it on the NVIDIA host as before (cargo build --release, or
# --features static-cuda for the single-file HiveOS binary).
#
set -e
cd "$(dirname "$0")"

# --- protoc (build-time only, for the gRPC protos in build.rs) ----------------
# Prefer a system protoc; otherwise fetch a standalone binary into ~/.local (no root).
if command -v protoc >/dev/null 2>&1; then
    export PROTOC="$(command -v protoc)"
else
    PROTOC_DIR="$HOME/.local/protoc"
    if [ ! -x "$PROTOC_DIR/bin/protoc" ]; then
        echo "[build-amd] fetching standalone protoc into $PROTOC_DIR ..."
        PB_VER=25.1
        tmp="$(mktemp -d)"
        curl -fsSL -o "$tmp/protoc.zip" \
            "https://github.com/protocolbuffers/protobuf/releases/download/v${PB_VER}/protoc-${PB_VER}-linux-x86_64.zip"
        mkdir -p "$PROTOC_DIR"
        ( cd "$PROTOC_DIR" && unzip -o "$tmp/protoc.zip" >/dev/null )
        rm -rf "$tmp"
    fi
    export PROTOC="$PROTOC_DIR/bin/protoc"
fi
echo "[build-amd] PROTOC=$PROTOC ($($PROTOC --version))"

# --- libOpenCL.so link target -------------------------------------------------
# The opencl3/cl3 crate links -lOpenCL, but ROCm/distro packages often ship only
# libOpenCL.so.1 (no dev symlink). Create a private libOpenCL.so and add it to the
# linker search path, without touching system dirs.
if ! ldconfig -p | grep -q 'libOpenCL\.so$' && ! ls /usr/lib/x86_64-linux-gnu/libOpenCL.so >/dev/null 2>&1; then
    OCL_REAL="$(ldconfig -p | awk '/libOpenCL\.so\.1 /{print $NF; exit}')"
    [ -z "$OCL_REAL" ] && OCL_REAL="/lib/x86_64-linux-gnu/libOpenCL.so.1"
    if [ ! -e "$OCL_REAL" ]; then
        echo "[build-amd] ERROR: libOpenCL.so.1 not found. Install an OpenCL ICD loader (e.g. ocl-icd-libopencl1)." >&2
        exit 1
    fi
    OCL_LINK_DIR="$HOME/.local/ocllib"
    mkdir -p "$OCL_LINK_DIR"
    ln -sf "$OCL_REAL" "$OCL_LINK_DIR/libOpenCL.so"
    export LIBRARY_PATH="$OCL_LINK_DIR:$LIBRARY_PATH"
    echo "[build-amd] libOpenCL.so -> $OCL_REAL  (via $OCL_LINK_DIR)"
fi

# --- build --------------------------------------------------------------------
echo "[build-amd] building libkeryxopencl.so ..."
cargo build -p keryxopencl --release

echo "[build-amd] building keryx-miner-supr (dynamic, CUDA-free) ..."
cargo build --release --bin keryx-miner-supr

mkdir -p dist-amd
cp -f target/release/keryx-miner-supr dist-amd/
cp -f target/release/libkeryxopencl.so dist-amd/
echo ""
echo "[build-amd] done. dist-amd/:"
ls -la dist-amd/
echo ""
echo "Run e.g.:"
echo "  cd dist-amd && ./keryx-miner-supr \\"
echo "    -s stratum+tcp://krx.suprnova.cc:4401 \\"
echo "    -a keryx:<your_address>.<worker> --light"
