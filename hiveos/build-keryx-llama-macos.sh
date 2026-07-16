#!/usr/bin/env bash
# Shippable `libkeryx-llama.dylib` — the miner's in-process llama.cpp engine on Apple Silicon
# (Metal backend). Byte-identical counterpart to hiveos/build-keryx-llama.sh (CUDA / Linux).
#
# Usage: build-keryx-llama-macos.sh [OUT_DIR] [JOBS]
#   OUT_DIR — where to drop libkeryx-llama.dylib (default: <repo>/hiveos/dist-macos-arm64)
#   JOBS    — parallel build jobs (default: sysctl -n hw.ncpu)
#
# llama.cpp PINNED to b10015 — the SAME pin the CUDA line + tools/llama_zerodup_spike use.
# Bump all together, then re-verify byte-exactness.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$REPO/hiveos/dist-macos-arm64}"
JOBS="${2:-$(sysctl -n hw.ncpu)}"
TAG=b10015
SRC=/tmp/llama-src-$TAG
BUILD=/tmp/llama-metal-build

command -v cmake >/dev/null || { echo "cmake missing — brew install cmake"; exit 1; }

if [ ! -d "$SRC" ]; then
  echo ">> cloning llama.cpp @ $TAG"
  git clone --quiet --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$SRC"
fi

# Static-lib llama.cpp build with Metal backend + embedded Metal shader source (no runtime
# ggml-metal.metal lookup — the .dylib is fully self-contained). GGML_NATIVE=ON picks up Apple
# Silicon dotprod + i8mm.
if [ ! -f "$BUILD/src/libllama.a" ]; then
  echo ">> configuring llama.cpp (Metal)"
  rm -rf "$BUILD"
  cmake -S "$SRC" -B "$BUILD" \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_CUDA=OFF \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DGGML_ACCELERATE=ON -DGGML_BLAS=OFF \
    -DCMAKE_BUILD_TYPE=Release
  echo ">> building llama (static, -j $JOBS)"
  cmake --build "$BUILD" --target llama -j "$JOBS"
fi

mkdir -p "$OUT"

# Link the wrapper + all ggml/llama static archives into a single Mach-O dylib. macOS `ld`
# uses `-force_load` for whole-archive semantics; `-Wl,-all_load` covers everything at once.
# `-framework Metal / Foundation / MetalKit / MetalPerformanceShaders / Accelerate` are the
# system frameworks ggml-metal needs at link time.
echo ">> linking libkeryx-llama.dylib"
clang++ -O2 -std=c++17 -shared -fPIC \
  -install_name @rpath/libkeryx-llama.dylib \
  "$REPO/tools/keryx-llama/keryx_llama.cpp" \
  -I "$SRC/include" -I "$SRC/ggml/include" -I "$SRC/src" -I "$SRC/common" \
  -Wl,-force_load,"$BUILD/src/libllama.a" \
  -Wl,-force_load,"$BUILD/ggml/src/libggml.a" \
  -Wl,-force_load,"$BUILD/ggml/src/libggml-base.a" \
  -Wl,-force_load,"$BUILD/ggml/src/libggml-cpu.a" \
  -Wl,-force_load,"$BUILD/ggml/src/ggml-metal/libggml-metal.a" \
  -framework Metal -framework Foundation -framework MetalKit \
  -framework MetalPerformanceShaders -framework Accelerate \
  -lpthread -o "$OUT/libkeryx-llama.dylib"

# Ad-hoc code-sign so macOS won't refuse to dlopen from Gatekeeper-quarantined tarballs.
codesign --force --sign - --timestamp=none "$OUT/libkeryx-llama.dylib"
echo ">> $(ls -la "$OUT/libkeryx-llama.dylib" | awk '{print $5}') B  $(nm -gU "$OUT/libkeryx-llama.dylib" | grep -c keryx_llama) exported syms"
