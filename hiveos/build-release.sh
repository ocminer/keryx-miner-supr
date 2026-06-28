#!/usr/bin/env bash
# Build BOTH release archives against old glibc (Ubuntu 20.04 = 2.31, artifacts
# land at GLIBC_2.30) in a single container pass:
#
#   1. keryx-miner-supr-<ver>-linux-x86_64.tar.gz  — "normal" dynamic build
#        (binary + libkeryxcuda.so + libkeryxopencl.so + RUN.txt) for general
#        Linux. Run from its own dir; the binary loads the .so plugins next to
#        it and libcuda from the NVIDIA driver.
#   2. keryx-miner-supr-<ver>-hiveos.tar.gz        — HiveOS custom miner
#        (single static-cuda binary + h-*.sh). No .so to ship.
#
# Output + SHA256SUMS in hiveos/dist/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="nvidia/cuda:12.8.0-devel-ubuntu20.04"   # glibc 2.31
OUT="$REPO/hiveos/dist"
PKG="$REPO/hiveos/pkg/keryx-miner-supr"
VERSION=$(grep -m1 '^CUSTOM_VERSION=' "$PKG/h-manifest.conf" | cut -d= -f2)
mkdir -p "$OUT/linux-stage"

echo ">> Building dynamic + static in $IMAGE (glibc 2.31) ..."
docker run --rm --network host --dns 1.1.1.1 --dns 8.8.8.8 -v "$REPO":/src -w /src -e DEBIAN_FRONTEND=noninteractive "$IMAGE" bash -euo pipefail -c '
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates build-essential pkg-config \
        protobuf-compiler cmake libssl-dev ocl-icd-opencl-dev >/dev/null
    # Install rust only if absent. Re-running rustup over an existing /root/.rustup
    # (e.g. a cached layer) can leave `cargo` off PATH after `. cargo/env`, so the
    # later `cargo build` silently no-ops and packaging fails with "No such file".
    if ! command -v cargo >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null
    fi
    . "$HOME/.cargo/env"
    command -v cargo >/dev/null || { echo "FATAL: cargo not on PATH after rust setup"; exit 3; }
    # CUDA_COMPUTE_CAP=70: lowest arch candle-kernels 0.9.2 compiles for; its
    # `.target sm_70` PTX forward-JITs to the whole fleet sm_70→sm_120 (Volta,
    # Turing, 3070/Ampere, Ada, Hopper, 5090). Was 80 (re-broke Volta/Turing).
    # Pascal sm_61 fails to compile (reduce.cu half atomicAdd) → --cpu-inference.
    export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CUDA_COMPUTE_CAP=70
    export PATH=/usr/local/cuda/bin:$PATH
    export RUSTFLAGS="-L /usr/local/cuda/lib64/stubs"
    export CARGO_TARGET_DIR=/src/target-hiveos

    # candle-kernels stale-PTX guard: bindgen_cuda skips nvcc when its OUT_DIR .ptx
    # is newer than the .cu source, so `cargo clean -p candle-kernels` ALONE does
    # NOT regenerate the PTX after CUDA_COMPUTE_CAP changes — the old-arch .ptx
    # survives in build/candle-kernels-*/out and gets reused. Nuke the whole
    # candle-kernels build + fingerprint so the new sm_70 PTX is emitted fresh.
    rm -rf target-hiveos/release/build/candle-kernels-* \
           target-hiveos/release/.fingerprint/candle-kernels-* \
           target-hiveos/release/deps/*candle_kernels* 2>/dev/null || true

    # pom-cuda = the Proof-of-Model CUDA search driver (post-fork algo); without it
    # the binary mines the dead kHeavyHash algo after the PoM hardfork.
    # NOTE: pom-cuda's cudarc 0.13.9 requires the build image's CUDA toolkit be
    # <= 12.8 (12.9+ panics "Unsupported cuda toolkit version").

    # 1) dynamic ("normal") — binary + both plugin .so
    cargo build --release --features pom-cuda
    cp target-hiveos/release/keryx-miner-supr \
       target-hiveos/release/libkeryxcuda.so \
       target-hiveos/release/libkeryxopencl.so \
       /src/hiveos/dist/linux-stage/

    # 2) static-cuda (HiveOS) — single binary (overwrites the dynamic one)
    cargo build --release --features static-cuda,pom-cuda
    cp target-hiveos/release/keryx-miner-supr /src/hiveos/dist/keryx-miner-supr
    chmod -R a+rX /src/hiveos/dist
'

# ---- assemble the "normal" linux archive --------------------------------------
echo ">> Packaging normal (dynamic) archive ..."
LINUX_NAME="keryx-miner-supr"
LINUX_STAGE=$(mktemp -d); LDEST="$LINUX_STAGE/$LINUX_NAME"
mkdir -p "$LDEST"
cp "$OUT"/linux-stage/keryx-miner-supr "$OUT"/linux-stage/libkeryxcuda.so "$OUT"/linux-stage/libkeryxopencl.so "$LDEST/"
chmod +x "$LDEST/keryx-miner-supr"
cat > "$LDEST/RUN.txt" <<'TXT'
keryx-miner-supr — Linux x86_64 (dynamic build, glibc >= 2.30)

Files: keryx-miner-supr + libkeryxcuda.so + libkeryxopencl.so (keep together).
Needs the NVIDIA driver installed (provides libcuda.so.1). CUDA toolkit NOT
required at runtime.

Run (the binary loads the .so plugins from its own directory):

  ./keryx-miner-supr \
      -a keryx:<your_address>.<worker> \
      -s stratum+tcp://krx.suprnova.cc:4401 \
      --light --cuda-device 0

--light = TinyLlama only. Drop it for higher LLM tiers (more VRAM + model
download). No devfund tax; the miner always mines to your address.
TXT
LINUX_TARBALL="$OUT/keryx-miner-supr-${VERSION}-linux-x86_64.tar.gz"
tar -czf "$LINUX_TARBALL" -C "$LINUX_STAGE" "$LINUX_NAME"
rm -rf "$LINUX_STAGE" "$OUT/linux-stage"

# ---- assemble the HiveOS archive (static binary + h-*.sh) ----------------------
echo ">> Packaging HiveOS (static) archive ..."
HIVE_STAGE=$(mktemp -d); HDEST="$HIVE_STAGE/keryx-miner-supr"
mkdir -p "$HDEST"
cp "$PKG"/h-manifest.conf "$PKG"/h-config.sh "$PKG"/h-run.sh "$PKG"/h-stats.sh "$HDEST/"
cp "$OUT"/keryx-miner-supr "$HDEST/"
chmod +x "$HDEST"/h-*.sh "$HDEST"/keryx-miner-supr
# HiveOS requires `<minername>-<version>.tar.gz` with no '-' in the version, so
# this one is NOT suffixed (its plain version is what HiveOS parses).
HIVE_TARBALL="$OUT/keryx-miner-supr-${VERSION}.tar.gz"
tar -czf "$HIVE_TARBALL" -C "$HIVE_STAGE" keryx-miner-supr
rm -rf "$HIVE_STAGE"

# ---- checksums + report -------------------------------------------------------
( cd "$OUT" && sha256sum "keryx-miner-supr-${VERSION}-linux-x86_64.tar.gz" \
                          "keryx-miner-supr-${VERSION}.tar.gz" > SHA256SUMS.txt )
echo ">> Done:"
ls -la "$LINUX_TARBALL" "$HIVE_TARBALL"
echo ">> glibc ceilings:"
for f in "$OUT/keryx-miner-supr"; do printf '   static binary  %s\n' "$(objdump -T "$f" | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)"; done
echo ">> SHA256SUMS:"; cat "$OUT/SHA256SUMS.txt"
