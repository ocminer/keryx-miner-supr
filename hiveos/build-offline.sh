#!/usr/bin/env bash
# Offline keryx build inside a prebuilt image (NO network: host crate cache mounted,
# --network none). Produces dynamic (binary+plugins) + static (single binary) +
# the image's CUDA runtime libs, into hiveos/<OUTDIR>/.
#
# Usage: build-offline.sh <IMAGE> <OUTDIR-name> <SUFFIX> [HOST_CUDA_DIR]
#   e.g. build-offline.sh keryx-build:offline dist-modern modern
#   HOST_CUDA_DIR: optional host path to a CUDA toolkit to mount at /opt/cuda and
#   build against (e.g. an extracted 12.2 toolkit) instead of the image's CUDA.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="$1"; OUTDIR="$2"; SUF="$3"; CUDADIR="${4:-}"
OUT="$REPO/hiveos/$OUTDIR"
SCRATCH=/tmp/koffcargo-$SUF
TGT="target-offline-$SUF"

CUDAMOUNT=(); KCUDA=/usr/local/cuda
if [ -n "$CUDADIR" ]; then CUDAMOUNT=(-v "$CUDADIR":/opt/cuda:ro); KCUDA=/opt/cuda; fi

# host crate cache -> scratch (so the container, running as root, never pollutes ~/.cargo)
if [ ! -d "$SCRATCH/registry" ]; then echo ">> copying crate cache to scratch..."; rm -rf "$SCRATCH"; cp -a "$HOME/.cargo" "$SCRATCH"; fi

rm -rf "$OUT"; mkdir -p "$OUT/lib"
echo ">> offline build in $IMAGE (suffix=$SUF) $(date +%H:%M:%S)"
docker run --rm --network none \
  -v "$REPO":/src -w /src \
  -v "$SCRATCH":/root/.cargo "${CUDAMOUNT[@]}" \
  -e CARGO_HOME=/root/.cargo -e CARGO_NET_OFFLINE=true -e RUSTUP_HOME=/usr/local/rustup \
  -e KCUDA="$KCUDA" \
  "$IMAGE" bash -euo pipefail -c '
    export CUDA_HOME=$KCUDA CUDA_PATH=$KCUDA CUDA_COMPUTE_CAP=70
    export PATH=$KCUDA/bin:/usr/local/cargo/bin:/root/.cargo/bin:$PATH
    export RUSTFLAGS="-L $KCUDA/lib64/stubs"
    export CARGO_TARGET_DIR=/src/'"$TGT"'
    O=/src/hiveos/'"$OUTDIR"'
    echo "building against CUDA: $(nvcc --version | grep -oE "release [0-9.]+")"
    rm -rf '"$TGT"'/release/build/candle-kernels-* '"$TGT"'/release/.fingerprint/candle-kernels-* '"$TGT"'/release/deps/*candle_kernels* 2>/dev/null || true
    echo "=== dynamic build (binary + plugins) ==="
    cargo build --offline --release --features pom-cuda
    cp '"$TGT"'/release/keryx-miner-supr "$O/keryx-miner-supr-dynamic"
    cp '"$TGT"'/release/libkeryxcuda.so '"$TGT"'/release/libkeryxopencl.so "$O/"
    echo "=== static build (single binary) ==="
    cargo build --offline --release --features static-cuda,pom-cuda
    cp '"$TGT"'/release/keryx-miner-supr "$O/keryx-miner-supr"
    L=$KCUDA/targets/x86_64-linux/lib
    for f in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do cp -L "$L/$f" "$O/lib/"; done
    chmod -R a+rX "$O"
  '
echo ">> done. static fallback=$(strings "$OUT/keryx-miner-supr" 2>/dev/null | grep -c KERYX_FORCE_GPU_INFER_FAIL) glibc=$(objdump -T "$OUT/keryx-miner-supr" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)"
echo ">> bundled libs ($(du -sh "$OUT/lib"|cut -f1)): $(ls "$OUT/lib")"
