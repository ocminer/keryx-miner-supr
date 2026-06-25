#!/usr/bin/env bash
# Build keryx-miner-supr against an OLD glibc (Ubuntu 20.04 = glibc 2.31) so the
# binary + plugins run on HiveOS, which ships an older glibc than the dev rig
# (Ubuntu 24.04 = glibc 2.39). The PoW PTX is pre-built and `include_str!`'d, so
# no nvcc-in-container PTX regen happens — the CUDA 13.0 / PTX 9.0 kernels ride
# along unchanged. `cust` links the CUDA *driver* API (libcuda), provided on the
# HiveOS host at runtime; here we link against the toolkit stub.
#
# Usage:  hiveos/build-glibc.sh
# Output: hiveos/dist/{keryx-miner-supr,libkeryxcuda.so,libkeryxopencl.so}
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="nvidia/cuda:12.8.0-devel-ubuntu20.04"   # glibc 2.31
OUT="$REPO/hiveos/dist"
mkdir -p "$OUT"

echo ">> Building in $IMAGE (glibc 2.31) ..."
docker run --rm \
  -v "$REPO":/src \
  -w /src \
  -e DEBIAN_FRONTEND=noninteractive \
  "$IMAGE" bash -euo pipefail -c '
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates build-essential pkg-config \
        protobuf-compiler cmake libssl-dev ocl-icd-opencl-dev >/dev/null
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null
    . "$HOME/.cargo/env"

    export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
    export CUDA_COMPUTE_CAP=120
    export PATH=/usr/local/cuda/bin:$PATH
    # Link cust against the toolkit libcuda stub (driver provides it at runtime).
    export RUSTFLAGS="-L /usr/local/cuda/lib64/stubs"
    export CARGO_TARGET_DIR=/src/target-hiveos

    # static-cuda: the CUDA worker is linked into the binary, so the HiveOS
    # payload is ONE executable (no libkeryx*.so to ship).
    # pom-cuda: the Proof-of-Model CUDA search driver (post-fork algo). WITHOUT it
    # the miner mines the dead kHeavyHash algo after the PoM hardfork.
    # NOTE: full-parity pom-cuda uses candle 0.9 / cudarc 0.19 and builds on the
    # 12.8 toolkit here. It links the candle CUDA RUNTIME libcudart + libcublas, so
    # unlike the old kHeavyHash static-cuda build the payload now needs those .so
    # files at runtime; hiveos/package.sh bundles them next to the binary.
    cargo build --release --features static-cuda,pom-cuda
    # Make artifacts readable by the host (uid 1000) after a root build.
    cp target-hiveos/release/keryx-miner-supr /src/hiveos/dist/
    chmod -R a+rX /src/hiveos/dist
'

echo ">> Done. Verifying glibc symbol ceiling (must be <= 2.31):"
max=$(objdump -T "$OUT/keryx-miner-supr" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)
printf '   %-22s max %s\n' "keryx-miner-supr" "$max"
echo ">> Self-contained check (no .so plugin deps expected):"
ldd "$OUT/keryx-miner-supr" 2>/dev/null | grep -iE 'keryxcuda|keryxopencl' && echo "   WARNING: still links a plugin .so" || echo "   OK — single binary, only the driver libcuda is dynamic"
ls -la "$OUT"
