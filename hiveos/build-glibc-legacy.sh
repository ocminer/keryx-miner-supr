#!/usr/bin/env bash
# LEGACY build: same as build-glibc.sh but against the CUDA 12.4 toolkit so the
# bundled CUDA runtime (cuBLAS/cuRAND/cudart) requires only NVIDIA driver >= 550
# (CUDA 12.4 floor) instead of >= 575 (12.9) / >= 570 (12.8). This unblocks
# miners on CUDA 12.4/12.6 boxes (Turing 20xx, Ampere 3060/3070, Ada) whose
# older drivers can't satisfy the 12.8/12.9 cuBLAS the default release ships.
#
# The PoW walk uses the CUDA *driver* API (works on any R525+), so it was never
# the problem — only OPoI inference (CUDA *runtime* API -> cuBLAS) hit the floor.
#
# Output: hiveos/dist-legacy/{keryx-miner-supr, lib/{libcudart,libcublas,libcublasLt,libcurand}.so.*}
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="nvidia/cuda:12.4.1-devel-ubuntu20.04"   # glibc 2.31, CUDA 12.4 -> driver floor 550
OUT="$REPO/hiveos/dist-legacy"
rm -rf "$OUT"; mkdir -p "$OUT/lib"

echo ">> LEGACY build in $IMAGE (CUDA 12.4, glibc 2.31) ..."
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
    export CUDA_COMPUTE_CAP=70                  # sm_70 PTX JITs across the fleet (Volta..Blackwell)
    export PATH=/usr/local/cuda/bin:$PATH
    export RUSTFLAGS="-L /usr/local/cuda/lib64/stubs"
    export CARGO_TARGET_DIR=/src/target-hiveos-legacy

    # candle-kernels stale-PTX guard (CUDA_COMPUTE_CAP change -> must re-emit PTX)
    rm -rf target-hiveos-legacy/release/build/candle-kernels-* \
           target-hiveos-legacy/release/.fingerprint/candle-kernels-* \
           target-hiveos-legacy/release/deps/*candle_kernels* 2>/dev/null || true

    # Dynamic build first (binary + CUDA/OpenCL plugins) — needed for the generic-Linux package.
    echo "=== dynamic build (binary + plugins) ==="
    cargo build --release --features pom-cuda
    cp target-hiveos-legacy/release/keryx-miner-supr /src/hiveos/dist-legacy/keryx-miner-supr-dynamic
    cp target-hiveos-legacy/release/libkeryxcuda.so target-hiveos-legacy/release/libkeryxopencl.so /src/hiveos/dist-legacy/

    # Static build (single self-contained binary) — used for HiveOS/SMOS/mmpOS packages.
    echo "=== static build (single binary) ==="
    cargo build --release --features static-cuda,pom-cuda

    cp target-hiveos-legacy/release/keryx-miner-supr /src/hiveos/dist-legacy/

    # Extract the CONTAINER (12.4) CUDA runtime libs — these set the driver floor.
    # libcudart is bundled here (unlike the 12.8 release) so the bundled 12.4 cuBLAS
    # does not fall back to a newer system cudart that would re-raise the floor.
    L=/usr/local/cuda/targets/x86_64-linux/lib
    for f in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do
      cp -L "$L/$f" /src/hiveos/dist-legacy/lib/
    done
    echo ">> CUDA 12.4 runtime version (cudart):"; cat /usr/local/cuda/version.json 2>/dev/null | grep -A2 cuda_cudart || true
    chmod -R a+rX /src/hiveos/dist-legacy
'

echo ">> Done. glibc symbol ceiling (must be <= 2.31):"
objdump -T "$OUT/keryx-miner-supr" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1
echo ">> bundled 12.4 runtime libs:"; ls -la "$OUT/lib"
