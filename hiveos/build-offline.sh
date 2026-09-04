#!/usr/bin/env bash
# Offline keryx build inside a prebuilt image (NO network: host crate cache, pinned llama.cpp
# source, and a portable CMake are mounted; Docker uses --network none). Produces the NVIDIA
# dynamic binary/plugin, static binary, both required CUDA llama.cpp engines, and the image's
# CUDA runtime libs in hiveos/<OUTDIR>/.
#
# Usage: build-offline.sh <IMAGE> <OUTDIR-name> <SUFFIX> [HOST_CUDA_DIR]
#   e.g. build-offline.sh keryx-build:offline dist-modern modern
#   HOST_CUDA_DIR: optional host path to a CUDA toolkit to mount at /opt/cuda and
#   build against (e.g. an extracted 12.4 toolkit) instead of the image's CUDA.
# Offline inference-build prerequisites (override these defaults with environment variables):
#   KERYX_LLAMA_SRC=/tmp/llama-src-b10015
#     Clean, exact checkout of https://github.com/ggml-org/llama.cpp tag b10015.
#   KERYX_CMAKE_ROOT=/tmp/cmake-3.28.6-linux-x86_64
#     Extracted portable Kitware CMake tree; bin/cmake must be version >= 3.18.
#   KERYX_LLAMA_JOBS=16
# One-time online preparation example:
#   git clone --depth 1 --branch b10015 https://github.com/ggml-org/llama.cpp /tmp/llama-src-b10015
#   curl -fsSLO https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.tar.gz
#   tar -xzf cmake-3.28.6-linux-x86_64.tar.gz -C /tmp
# Per-line arch overrides (env): BOFF_POM_CUDA_ARCH (walk PTX; default compute_75 via
# build.rs; legacy=compute_70, pascal=compute_61) and BOFF_CUDA_COMPUTE_CAP (candle
# inference PTX; default 70; pascal=61).
#
# PASCAL MUST BE compute_61, NOT compute_60: the v4 walk uses __dp4a (src/pom_mine.cu),
# which is sm_61+. compute_60 (P100) fails to compile outright:
#   src/pom_mine.cu(303): error: identifier "__dp4a" is undefined
# So the pascal line covers GTX 10-series (1080 Ti = sm_61), not P100. It also needs a
# CUDA 12.x toolkit mounted via HOST_CUDA_DIR — CUDA 13 dropped Pascal entirely
# ("nvcc fatal: Unsupported gpu architecture 'compute_61'").
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="$1"; OUTDIR="$2"; SUF="$3"; CUDADIR="${4:-}"
[[ "$OUTDIR" =~ ^dist-[A-Za-z0-9._-]+$ ]] || {
  echo "ERROR: OUTDIR must be a simple dist-* directory name below hiveos/: $OUTDIR" >&2
  exit 2
}
POMARCH="${BOFF_POM_CUDA_ARCH:-}"; CCAP="${BOFF_CUDA_COMPUTE_CAP:-70}"
OUT="$REPO/hiveos/$OUTDIR"
SCRATCH=/tmp/koffcargo-$SUF
TGT="target-offline-$SUF"
LLAMA_TAG=b10015
LLAMA_SRC="${KERYX_LLAMA_SRC:-/tmp/llama-src-$LLAMA_TAG}"
CMAKE_ROOT="${KERYX_CMAKE_ROOT:-/tmp/cmake-3.28.6-linux-x86_64}"
LLAMA_JOBS="${KERYX_LLAMA_JOBS:-16}"

case "$SUF" in
  modern) LLAMA_ARCHS="75;80;86;89;90;120" ;;
  legacy) LLAMA_ARCHS="61;70;75;80;86;89;90" ;;
  # One-off only: the v4 walk itself still requires sm_61, but retain the established
  # Pascal inference-engine images for both sm_60 and sm_61.
  pascal) LLAMA_ARCHS="60;61" ;;
  *) echo "ERROR: unsupported NVIDIA line '$SUF' (expected modern, legacy, or pascal)" >&2; exit 2 ;;
esac
[[ "$LLAMA_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: KERYX_LLAMA_JOBS must be a positive integer" >&2; exit 2; }
[[ -d "$LLAMA_SRC/.git" && -f "$LLAMA_SRC/CMakeLists.txt" ]] || {
  echo "ERROR: pinned llama.cpp source missing at $LLAMA_SRC" >&2
  echo "       Set KERYX_LLAMA_SRC to a clean checkout of tag $LLAMA_TAG (see script header)." >&2
  exit 2
}
[[ "$(git -C "$LLAMA_SRC" describe --tags --exact-match 2>/dev/null || true)" == "$LLAMA_TAG" ]] || {
  echo "ERROR: KERYX_LLAMA_SRC must be checked out at exact tag $LLAMA_TAG: $LLAMA_SRC" >&2
  exit 2
}
[[ -z "$(git -C "$LLAMA_SRC" status --porcelain=v1 --untracked-files=all --ignore-submodules)" ]] || {
  echo "ERROR: KERYX_LLAMA_SRC must be clean for a reproducible offline build: $LLAMA_SRC" >&2
  exit 2
}
[[ -x "$CMAKE_ROOT/bin/cmake" ]] || {
  echo "ERROR: portable CMake missing at $CMAKE_ROOT/bin/cmake" >&2
  echo "       Set KERYX_CMAKE_ROOT to an extracted CMake >= 3.18 tree (see script header)." >&2
  exit 2
}
CMAKE_VERSION=$("$CMAKE_ROOT/bin/cmake" --version | awk 'NR == 1 { print $3 }')
[[ -n "$CMAKE_VERSION" && "$(printf '%s\n' 3.18 "$CMAKE_VERSION" | sort -V | head -1)" == 3.18 ]] || {
  echo "ERROR: CMake >= 3.18 required, found '${CMAKE_VERSION:-unknown}' in $CMAKE_ROOT" >&2
  exit 2
}

CUDAMOUNT=(); KCUDA=/usr/local/cuda
if [ -n "$CUDADIR" ]; then
  [[ -x "$CUDADIR/bin/nvcc" ]] || { echo "ERROR: HOST_CUDA_DIR has no executable bin/nvcc: $CUDADIR" >&2; exit 2; }
  CUDAMOUNT=(-v "$CUDADIR":/opt/cuda:ro)
  KCUDA=/opt/cuda
fi

# host crate cache -> scratch (so the container, running as root, never pollutes ~/.cargo)
if [ ! -d "$SCRATCH/registry" ]; then echo ">> copying crate cache to scratch..."; rm -rf "$SCRATCH"; cp -a "$HOME/.cargo" "$SCRATCH"; fi

rm -rf "$OUT"; mkdir -p "$OUT/lib"
echo ">> offline build in $IMAGE (suffix=$SUF) $(date +%H:%M:%S)"
docker run --rm --network none \
  -v "$REPO":/src -w /src \
  -v "$SCRATCH":/root/.cargo "${CUDAMOUNT[@]}" \
  -v "$LLAMA_SRC":/llama:ro -v "$CMAKE_ROOT":/opt/keryx-cmake:ro \
  -e CARGO_HOME=/root/.cargo -e CARGO_NET_OFFLINE=true -e RUSTUP_HOME=/usr/local/rustup \
  -e KCUDA="$KCUDA" -e POMARCH="$POMARCH" -e CCAP="$CCAP" \
  -e KERYX_TARGET_DIR="$TGT" -e KERYX_OUTDIR="$OUTDIR" \
  -e LLAMA_ARCHS="$LLAMA_ARCHS" -e LLAMA_JOBS="$LLAMA_JOBS" \
  "$IMAGE" bash -euo pipefail -c '
    export CUDA_HOME=$KCUDA CUDA_PATH=$KCUDA CUDA_COMPUTE_CAP=$CCAP
    if [ -n "$POMARCH" ]; then export POM_CUDA_ARCH="$POMARCH"; fi
    export PATH=$KCUDA/bin:/usr/local/cargo/bin:/root/.cargo/bin:$PATH
    # -rpath $ORIGIN/lib so the binary finds the bundled CUDA runtime (libcurand/libcublas/…) next
    # to itself on a clean rig (NVIDIA driver only, no system CUDA), without needing a launcher to
    # set LD_LIBRARY_PATH. \$ORIGIN is kept literal for the linker (escaped past the container shell).
    export RUSTFLAGS="-L $KCUDA/lib64/stubs -C link-arg=-Wl,-rpath,\$ORIGIN/lib"
    export CARGO_TARGET_DIR="/src/$KERYX_TARGET_DIR"
    O="/src/hiveos/$KERYX_OUTDIR"
    echo "building against CUDA: $(nvcc --version | grep -oE "release [0-9.]+") walk-arch=${POMARCH:-default} candle-cap=$CCAP"
    rm -rf "$KERYX_TARGET_DIR"/release/build/candle-kernels-* \
           "$KERYX_TARGET_DIR"/release/.fingerprint/candle-kernels-* \
           "$KERYX_TARGET_DIR"/release/deps/*candle_kernels* 2>/dev/null || true
    touch /src/src/pom_mine.cu 2>/dev/null || true
    echo "=== NVIDIA dynamic build (pom-cuda host + CUDA plugin) ==="
    cargo build --locked --offline --release -p keryxcuda
    cargo build --locked --offline --release -p keryx-miner-supr --features pom-cuda
    cp "$KERYX_TARGET_DIR/release/keryx-miner-supr" "$O/keryx-miner-supr-dynamic"
    cp "$KERYX_TARGET_DIR/release/libkeryxcuda.so" "$O/"
    echo "=== static build (single binary) ==="
    cargo build --locked --offline --release -p keryx-miner-supr --features static-cuda,pom-cuda
    cp "$KERYX_TARGET_DIR/release/keryx-miner-supr" "$O/keryx-miner-supr"

    echo "=== CUDA llama.cpp engines (AVX2 + no-AVX) ==="
    git config --global --add safe.directory /llama
    build_llama_engine() {
      variant="$1"
      outname="$2"
      build_dir="/tmp/keryx-llama-$variant"
      simd_flags=()
      if [ "$variant" = noavx ]; then
        simd_flags=(
          -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF
          -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF
        )
      fi
      /opt/keryx-cmake/bin/cmake -S /llama -B "$build_dir" \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$LLAMA_ARCHS" \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF -DGGML_CUDA_NCCL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$KCUDA/bin/nvcc" \
        "${simd_flags[@]}"
      /opt/keryx-cmake/bin/cmake --build "$build_dir" --target llama -j "$LLAMA_JOBS"
      g++ -O2 -std=c++17 -shared -fPIC -fopenmp /src/tools/keryx-llama/keryx_llama.cpp \
        -I /llama/include -I /llama/ggml/include -I /llama/src -I /llama/common \
        -I "$KCUDA/include" \
        -Wl,--start-group "$build_dir/src/libllama.a" \
          "$build_dir/ggml/src/ggml-cuda/libggml-cuda.a" \
          "$build_dir/ggml/src/libggml-cpu.a" "$build_dir/ggml/src/libggml.a" \
          "$build_dir/ggml/src/libggml-base.a" \
        -Wl,--end-group \
        -L"$KCUDA/lib64" -L"$KCUDA/targets/x86_64-linux/lib" \
        -lcudart -lcublas -lcublasLt \
        -L"$KCUDA/lib64/stubs" -L"$KCUDA/targets/x86_64-linux/lib/stubs" -lcuda \
        -lpthread -ldl -o "$O/$outname"
      chmod a+rx "$O/$outname"
    }
    build_llama_engine avx libkeryx-llama.so
    build_llama_engine noavx libkeryx-llama-noavx.so

    L=$KCUDA/targets/x86_64-linux/lib
    for f in libcudart.so.12 libcublas.so.12 libcublasLt.so.12 libcurand.so.10; do cp -L "$L/$f" "$O/lib/"; done
    chmod -R a+rX "$O"
  '
echo ">> done. static fallback=$(strings "$OUT/keryx-miner-supr" 2>/dev/null | grep -c KERYX_FORCE_GPU_INFER_FAIL) glibc=$(objdump -T "$OUT/keryx-miner-supr" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)"
echo ">> bundled libs ($(du -sh "$OUT/lib"|cut -f1)): $(ls "$OUT/lib")"
for engine in libkeryx-llama.so libkeryx-llama-noavx.so; do
  [[ -s "$OUT/$engine" ]] || { echo "ERROR: required inference engine was not produced: $OUT/$engine" >&2; exit 1; }
  ENGINE_GLIBC=$(objdump -T "$OUT/$engine" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1 || true)
  ENGINE_SYMS=$(nm -D "$OUT/$engine" 2>/dev/null | grep -c keryx_llama || true)
  [[ -n "$ENGINE_GLIBC" && "$(printf '%s\n' GLIBC_2.31 "$ENGINE_GLIBC" | sort -V | tail -1)" == GLIBC_2.31 ]] || {
    echo "ERROR: $engine requires ${ENGINE_GLIBC:-an unknown glibc}, above/without the GLIBC_2.31 release ceiling" >&2
    exit 1
  }
  [[ "$ENGINE_SYMS" -gt 0 ]] || { echo "ERROR: $engine exports no keryx_llama ABI symbols" >&2; exit 1; }
  echo ">> $engine: $(stat -c %s "$OUT/$engine") bytes, glibc=$ENGINE_GLIBC, syms=$ENGINE_SYMS"
done
VEX=$(objdump -d --disassemble="ggml_vec_dot_q8_0_q8_0" "$OUT/libkeryx-llama-noavx.so" 2>/dev/null \
  | awk -F'\t' 'NF >= 3 { print $3 }' | grep -cE '^v[a-z]' || true)
[[ "$VEX" == 0 ]] || { echo "ERROR: no-AVX engine still contains AVX instructions in ggml_vec_dot_q8_0_q8_0" >&2; exit 1; }
echo ">> no-AVX engine verification: VEX instruction count=$VEX"
