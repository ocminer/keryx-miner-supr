#!/usr/bin/env bash
# Shippable `libkeryx-llama.so` — the miner's in-process llama.cpp engine (Phase 2
# candle-independence: hosts the model for the PoM walk zero-dup + serves OPoI inference).
# Built inside the glibc-2.31 container; links the SAME CUDA line as the package it ships in
# (cublas/cudart resolve from the package's lib/ at runtime).
#
# Usage: build-keryx-llama.sh <modern|legacy|pascal> [JOBS] [noavx]
#   modern: container CUDA 12.9, archs 75;80;86;89;90;120
#   legacy: /tmp/cuda124 (12.4), archs 70;75;80;86;89;90
#   pascal: /tmp/cuda124 (12.4), archs 60;61
# Output: hiveos/dist-<line>/libkeryx-llama.so  (package-line.sh bundles it when present)
#
# "noavx" 3rd arg: baseline-ISA CPU backend (GGML_AVX/AVX2/FMA/F16C/BMI2 OFF, SSE4.2 kept) →
# hiveos/dist-<line>/libkeryx-llama-noavx.so. The default build bakes AVX2-level SIMD into the
# ggml CPU path with NO runtime dispatch — pre-AVX rig CPUs (Celeron/Pentium, pre-Sandy-Bridge
# Xeon) SIGILL on it (v0.7.2 fix); the miner auto-selects the -noavx file on such CPUs.
#
# llama.cpp PINNED to b10015 — the SAME pin as build-llama-server.sh and the byte-identity
# proof in tools/llama_zerodup_spike. Bump all together, then re-verify the spike.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINE="$1"; JOBS="${2:-16}"; VARIANT="${3:-avx}"
TAG=b10015
case "$VARIANT" in
  avx)   OUTNAME=libkeryx-llama.so;       SIMD_FLAGS="" ;;
  noavx) OUTNAME=libkeryx-llama-noavx.so; SIMD_FLAGS="-DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF" ;;
  *) echo "usage: $0 <modern|legacy|pascal> [JOBS] [noavx]"; exit 1 ;;
esac
case "$LINE" in
  modern) ARCHS="75;80;86;89;90;120"; CUDAMOUNT=(); KCUDA=/usr/local/cuda ;;
  legacy) ARCHS="70;75;80;86;89;90";  CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro); KCUDA=/opt/cuda ;;
  pascal) ARCHS="60;61";              CUDAMOUNT=(-v /tmp/cuda124:/opt/cuda:ro); KCUDA=/opt/cuda ;;
  *) echo "usage: $0 <modern|legacy|pascal> [JOBS]"; exit 1 ;;
esac
OUT="${BKL_OUT:-$REPO/hiveos/dist-$LINE}"   # BKL_OUT: stage elsewhere (avoids racing a concurrent line build's dist rm -rf)
mkdir -p "$OUT"
SRC=/tmp/llama-src-$TAG
if [ ! -d "$SRC" ]; then
  git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi

docker run --rm --network host \
  -v "$SRC":/llama -v "$REPO":/repo:ro -v "$OUT":/out "${CUDAMOUNT[@]}" \
  -e KCUDA="$KCUDA" -e ARCHS="$ARCHS" -e JOBS="$JOBS" -e SIMD_FLAGS="$SIMD_FLAGS" -e OUTNAME="$OUTNAME" \
  keryx-build:offline bash -euo pipefail -c '
    if [ ! -x /tmp/cmk/bin/cmake ]; then
      curl -sL https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz \
        | tar xz -C /tmp && mv /tmp/cmake-3.28.6-linux-x86_64 /tmp/cmk
    fi
    export PATH=/tmp/cmk/bin:$KCUDA/bin:$PATH CUDA_HOME=$KCUDA
    B=/tmp/llama-pic-build
    /tmp/cmk/bin/cmake -S /llama -B $B -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=$KCUDA/bin/nvcc $SIMD_FLAGS
    /tmp/cmk/bin/cmake --build $B --target llama -j "$JOBS"
    g++ -O2 -std=c++17 -shared -fPIC -fopenmp /repo/tools/keryx-llama/keryx_llama.cpp \
      -I /llama/include -I /llama/ggml/include -I /llama/src -I /llama/common \
      -I $KCUDA/include \
      -Wl,--start-group $B/src/libllama.a $B/ggml/src/ggml-cuda/libggml-cuda.a \
        $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml.a $B/ggml/src/libggml-base.a \
      -Wl,--end-group \
      -L$KCUDA/lib64 -L$KCUDA/targets/x86_64-linux/lib -lcudart -lcublas -lcublasLt \
      -L$KCUDA/lib64/stubs -L$KCUDA/targets/x86_64-linux/lib/stubs -lcuda \
      -lpthread -ldl -o /out/$OUTNAME
    chmod a+rx /out/$OUTNAME
  '
echo ">> $LINE $OUTNAME: $(ls -la "$OUT/$OUTNAME" | awk '{print $5}') bytes, glibc=$(objdump -T "$OUT/$OUTNAME" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1), syms=$(nm -D "$OUT/$OUTNAME" | grep -c keryx_llama)"
if [ "$VARIANT" = noavx ]; then
  VEX=$(objdump -d --disassemble="ggml_vec_dot_q8_0_q8_0" "$OUT/$OUTNAME" 2>/dev/null | grep -cE '\bv[a-z]' || true)
  echo ">> $LINE $OUTNAME VEX-instr count in ggml_vec_dot_q8_0_q8_0 (MUST be 0): $VEX"
  [ "$VEX" = 0 ] || { echo "ERROR: noavx build still contains AVX code"; exit 1; }
fi
