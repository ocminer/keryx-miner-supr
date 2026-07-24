#!/usr/bin/env bash
# Offline keryx build inside a prebuilt image (NO network: host crate cache mounted,
# --network none). Produces dynamic (binary+plugins) + static (single binary) +
# the image's CUDA runtime libs, into hiveos/<OUTDIR>/.
#
# Usage: build-offline.sh <IMAGE> <OUTDIR-name> <SUFFIX> [HOST_CUDA_DIR]
#   e.g. build-offline.sh keryx-build:offline dist-modern modern
#   HOST_CUDA_DIR: optional host path to a CUDA toolkit to mount at /opt/cuda and
#   build against (e.g. an extracted 12.4 toolkit) instead of the image's CUDA.
# Per-line arch overrides (env): BOFF_POM_CUDA_ARCH (walk PTX; default compute_75 via
# build.rs; legacy=compute_70, pascal=compute_60) and BOFF_CUDA_COMPUTE_CAP (candle
# inference PTX; default 70; pascal=60).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="$1"; OUTDIR="$2"; SUF="$3"; CUDADIR="${4:-}"
POMARCH="${BOFF_POM_CUDA_ARCH:-}"; CCAP="${BOFF_CUDA_COMPUTE_CAP:-70}"
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
  -e KCUDA="$KCUDA" -e POMARCH="$POMARCH" -e CCAP="$CCAP" \
  "$IMAGE" bash -euo pipefail -c '
    export CUDA_HOME=$KCUDA CUDA_PATH=$KCUDA CUDA_COMPUTE_CAP=$CCAP
    if [ -n "$POMARCH" ]; then export POM_CUDA_ARCH="$POMARCH"; fi
    export PATH=$KCUDA/bin:/usr/local/cargo/bin:/root/.cargo/bin:$PATH
    # -rpath $ORIGIN/lib so the binary finds the bundled CUDA runtime (libcurand/libcublas/…) next
    # to itself on a clean rig (NVIDIA driver only, no system CUDA), without needing a launcher to
    # set LD_LIBRARY_PATH. \$ORIGIN is kept literal for the linker (escaped past the container shell).
    export RUSTFLAGS="-L $KCUDA/lib64/stubs -C link-arg=-Wl,-rpath,\$ORIGIN/lib"
    export CARGO_TARGET_DIR=/src/'"$TGT"'
    O=/src/hiveos/'"$OUTDIR"'
    echo "building against CUDA: $(nvcc --version | grep -oE "release [0-9.]+") walk-arch=${POMARCH:-default} candle-cap=$CCAP"
    rm -rf '"$TGT"'/release/build/candle-kernels-* '"$TGT"'/release/.fingerprint/candle-kernels-* '"$TGT"'/release/deps/*candle_kernels* 2>/dev/null || true
    touch /src/src/pom_mine.cu 2>/dev/null || true
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
