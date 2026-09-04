#!/usr/bin/env bash
# PASCAL build: a third build line for GTX 10-series (Pascal, sm_61 — e.g. the
# GTX 1080 Ti, 1080, 1070, 1060). Forked from build-glibc-legacy.sh.
#
# WHY a separate line: the default + legacy builds emit PTX at compute_70 (candle inference)
# and compute_75 (the PoM walk). PTX JIT is FORWARD-only, so compute_70/75 run on Turing+ but
# CANNOT JIT down to Pascal sm_61 -> a 1080 Ti fails to load the kernels. Pascal needs PTX
# emitted at sm_61. Tesla P100/sm_60 is not supported because the v4 walk uses __dp4a.
#
#   - POM_CUDA_ARCH=compute_61  -> the PoM walk kernel emits Pascal-compatible PTX.
#   - CUDA_COMPUTE_CAP=61       -> candle-kernels emit sm_61 PTX for the OPoI inference path.
#
# CUDA 12.4 still supports Pascal (compute 6.x); CUDA 13 DROPS it -> stay on the 12.x toolkit.
# ⚠️ Pascal has NO tensor cores and FP16 throughput is ~1/64 of FP32 -> the OPoI inference
#    challenge must run the cuBLAS FP32 (SGEMM) path; CPU inference is only the deprecated,
#    explicit --enable-cpu-inference emergency fallback. The PoW
#    walk is memory-bound and runs full-speed (1080 Ti = 484 GB/s GDDR5X, > a 3070's 448).
#    VALIDATE inference on a real 1080 Ti before shipping — it's the only unknown.
#
# Output: hiveos/dist-pascal/{keryx-miner-supr, lib/{libcudart,libcublas,libcublasLt,libcurand}.so.*}
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="nvidia/cuda:12.4.1-devel-ubuntu20.04"   # glibc 2.31, CUDA 12.4 -> driver floor 550, supports Pascal
OUT="$REPO/hiveos/dist-pascal"
rm -rf "$OUT"; mkdir -p "$OUT/lib"

echo ">> PASCAL build in $IMAGE (CUDA 12.4, sm_61 PTX for GTX 10-series) ..."
docker run --rm --network host \
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
    export CUDA_COMPUTE_CAP=61                  # candle-kernels PTX for GTX 10-series
    export POM_CUDA_ARCH=compute_61             # PoM v4 needs __dp4a, which starts at sm_61
    export PATH=/usr/local/cuda/bin:$PATH
    export RUSTFLAGS="-L /usr/local/cuda/lib64/stubs"
    export CARGO_TARGET_DIR=/src/target-hiveos-pascal

    # candle-kernels stale-PTX guard (CUDA_COMPUTE_CAP change -> must re-emit PTX)
    rm -rf target-hiveos-pascal/release/build/candle-kernels-* \
           target-hiveos-pascal/release/.fingerprint/candle-kernels-* \
           target-hiveos-pascal/release/deps/*candle_kernels* 2>/dev/null || true

    # NVIDIA dynamic build first — the compile-time host PoM backend is CUDA, so never pair it
    # with the OpenCL worker plugin in the generic-Linux package.
    echo "=== NVIDIA dynamic build (pom-cuda host + CUDA plugin) ==="
    cargo build --locked --release -p keryxcuda
    cargo build --locked --release -p keryx-miner-supr --features pom-cuda
    cp target-hiveos-pascal/release/keryx-miner-supr /src/hiveos/dist-pascal/keryx-miner-supr-dynamic
    cp target-hiveos-pascal/release/libkeryxcuda.so /src/hiveos/dist-pascal/

    # Static build (single self-contained binary) — used for HiveOS/SMOS/mmpOS packages.
    echo "=== static build (single binary) ==="
    cargo build --locked --release -p keryx-miner-supr --features static-cuda,pom-cuda

    cp target-hiveos-pascal/release/keryx-miner-supr /src/hiveos/dist-pascal/

    L=/usr/local/cuda/targets/x86_64-linux/lib
    for f in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do
      cp -L "$L/$f" /src/hiveos/dist-pascal/lib/
    done
    echo ">> CUDA 12.4 runtime version (cudart):"; cat /usr/local/cuda/version.json 2>/dev/null | grep -A2 cuda_cudart || true
    chmod -R a+rX /src/hiveos/dist-pascal

    # Prove the PoM walk PTX targets sm_61 (must load on GTX 10-series Pascal).
    echo ">> PoM walk PTX target line:"; head -8 target-hiveos-pascal/release/build/*/out/pom_mine.ptx 2>/dev/null | grep -iE ".target|.version" | head -3 || true
'

echo ">> Done. glibc symbol ceiling (must be <= 2.31):"
objdump -T "$OUT/keryx-miner-supr" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1
echo ">> bundled 12.4 runtime libs:"; ls -la "$OUT/lib"
echo ">> verify the binary embeds sm_61 PoM PTX:"
strings "$OUT/keryx-miner-supr" 2>/dev/null | grep -m1 -E "\.target sm_61" || echo "   (PTX embedded as include_str; checked in-container above)"
