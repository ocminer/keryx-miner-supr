#!/usr/bin/env bash
# Build the AMD/OpenCL keryx-miner-supr against an OLD glibc (Ubuntu 20.04 =
# glibc 2.31) so the binary + libkeryxopencl.so run on HiveOS and older distros
# (the dev rig is Ubuntu 24.04 = glibc 2.39, whose binaries won't load there).
#
# Unlike hiveos/build-glibc.sh (which builds the static-cuda NVIDIA single
# binary), this builds the DYNAMIC, CUDA-free AMD flavour: the host binary plus
# the dlopen'd libkeryxopencl.so. No CUDA toolkit involved — only an OpenCL ICD
# loader for link time (the AMD driver provides libOpenCL.so.1 at runtime).
#
# Usage:  hiveos/build-amd-glibc.sh
# Output: hiveos/dist-amd/{keryx-miner-supr, libkeryxopencl.so}
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="ubuntu:20.04"                 # glibc 2.31
OUT="$REPO/hiveos/dist-amd"
mkdir -p "$OUT"

echo ">> Building AMD flavour in $IMAGE (glibc 2.31) ..."
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

    # protoc for the gRPC build.rs; ocl-icd-opencl-dev provides libOpenCL.so.
    export PROTOC=/usr/bin/protoc
    # Separate target dir so it never clobbers the host target/ (LLM weight cache)
    # or the NVIDIA hiveos target-hiveos/.
    export CARGO_TARGET_DIR=/src/target-hiveos-amd

    echo ">> building libkeryxopencl.so ..."
    cargo build -p keryxopencl --release
    echo ">> building keryx-miner-supr (dynamic, CUDA-free) ..."
    cargo build --release --bin keryx-miner-supr

    cp target-hiveos-amd/release/keryx-miner-supr     /src/hiveos/dist-amd/
    cp target-hiveos-amd/release/libkeryxopencl.so    /src/hiveos/dist-amd/
    chmod -R a+rX /src/hiveos/dist-amd
'

echo ">> Done. Verifying glibc symbol ceiling (must be <= 2.31):"
for f in keryx-miner-supr libkeryxopencl.so; do
    max=$(objdump -T "$OUT/$f" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)
    printf '   %-22s max %s\n' "$f" "$max"
done
ls -la "$OUT"
