#!/usr/bin/env bash
# Shippable llama.cpp CUDA `llama-server` for the NVIDIA packages (Phase 1 candle-independence).
# Built inside the glibc-2.31 build container so it runs on HiveOS; links against the SAME
# CUDA line as the package it ships in (its cublas/cudart come from the package's lib/).
#
# Usage: build-llama-server.sh <modern|legacy|pascal> [JOBS]
#   modern: container CUDA 12.9, archs 75;80;86;89;90;120  (Turing+ .. Blackwell)
#   legacy: /tmp/cuda124 (12.4), archs 70;75;80;86;89;90   (Volta+ .. Hopper)
#   pascal: /tmp/cuda124 (12.4), archs 60;61               (GTX 10-series / P100)
# Output: hiveos/dist-<line>/llama-server  (package-line.sh bundles it when present)
#
# llama.cpp is PINNED to release tag b10015 (2026-07). Bump deliberately, then re-run all lines.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINE="$1"; JOBS="${2:-16}"
TAG=b10015
case "$LINE" in
  modern) ARCHS="75;80;86;89;90;120"; CUDAMOUNT=(); KCUDA=/usr/local/cuda ;;
  legacy) ARCHS="70;75;80;86;89;90";  CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro); KCUDA=/opt/cuda ;;
  pascal) ARCHS="60;61";              CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro); KCUDA=/opt/cuda ;;
  *) echo "usage: $0 <modern|legacy|pascal> [JOBS]"; exit 1 ;;
esac
OUT="$REPO/hiveos/dist-$LINE"
mkdir -p "$OUT"
SRC=/tmp/llama-src-$TAG
if [ ! -d "$SRC" ]; then
  git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi

docker run --rm --network host \
  -v "$SRC":/llama -v "$OUT":/out "${CUDAMOUNT[@]}" \
  -e KCUDA="$KCUDA" -e ARCHS="$ARCHS" -e JOBS="$JOBS" \
  keryx-build:offline bash -euo pipefail -c '
    # llama.cpp needs a newer cmake than 20.04 ships — fetch a prebuilt one.
    if [ ! -x /tmp/cmk/bin/cmake ]; then
      curl -sL https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz \
        | tar xz -C /tmp && mv /tmp/cmake-3.28.6-linux-x86_64 /tmp/cmk
    fi
    export PATH=/tmp/cmk/bin:$KCUDA/bin:$PATH CUDA_HOME=$KCUDA
    B=/tmp/llama-build
    /tmp/cmk/bin/cmake -S /llama -B $B -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=$KCUDA/bin/nvcc
    /tmp/cmk/bin/cmake --build $B --target llama-server -j "$JOBS"
    cp $B/bin/llama-server /out/llama-server
    chmod a+rx /out/llama-server
  '
echo ">> $LINE llama-server: $(ls -la "$OUT/llama-server" | awk '{print $5}') bytes, glibc=$(objdump -T "$OUT/llama-server" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)"
