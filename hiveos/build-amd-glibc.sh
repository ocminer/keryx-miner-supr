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
# Usage:  hiveos/build-amd-glibc.sh [--miner-only]
# Output: hiveos/dist-amd/{keryx-miner-supr, libkeryxopencl.so,
#                         libkeryx-llama-vk*.so, libvulkan.so.1*}
#
# The default is a RELEASE-COMPLETE build: after the miner/OpenCL plugin it rebuilds both Vulkan
# engine ISA variants. `--miner-only` intentionally stops after the mining artifacts for local
# development; release packagers reject that incomplete directory.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="ubuntu:20.04"                 # glibc 2.31
OUT="$REPO/hiveos/dist-amd"
MINER_ONLY=0
case "${1:-}" in
  "") ;;
  --miner-only) MINER_ONLY=1 ;;
  *) echo "usage: $0 [--miner-only]" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

echo ">> Building AMD flavour in $IMAGE (glibc 2.31) ..."
docker run --rm \
  --network host \
  -v "$REPO":/src \
  -w /src \
  -e DEBIAN_FRONTEND=noninteractive \
  "$IMAGE" bash -euo pipefail -c '
    # archive.ubuntu.com resolves to IPv6-only inside the container but there is no IPv6
    # route -> apt-get update fetches nothing -> "Unable to locate package". Force apt IPv4.
    echo "Acquire::ForceIPv4 \"true\";" > /etc/apt/apt.conf.d/99force-ipv4
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates build-essential pkg-config \
        protobuf-compiler cmake libssl-dev ocl-icd-opencl-dev libvulkan1 >/dev/null
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal >/dev/null
    . "$HOME/.cargo/env"

    # protoc for the gRPC build.rs; ocl-icd-opencl-dev provides libOpenCL.so.
    export PROTOC=/usr/bin/protoc
    # Separate target dir so it never clobbers the host target/ (LLM weight cache)
    # or the NVIDIA hiveos target-hiveos/.
    export CARGO_TARGET_DIR=/src/target-hiveos-amd

    echo ">> building libkeryxopencl.so ..."
    cargo build --locked -p keryxopencl --release
    echo ">> building keryx-miner-supr (dynamic, CUDA-free, PoM/OpenCL) ..."
    # pom-opencl REQUIRED post-fork: without it the PoM mining path is compiled out
    # and AMD mines the dead kHeavyHash algo -> zero valid blocks.
    cargo build --locked --release --bin keryx-miner-supr --features pom-opencl

    cp target-hiveos-amd/release/keryx-miner-supr     /src/hiveos/dist-amd/
    cp target-hiveos-amd/release/libkeryxopencl.so    /src/hiveos/dist-amd/
    # Stock HiveOS may have an AMD Vulkan ICD but no loader package. Bundle the loader built for
    # this same glibc floor; the vendor ICD itself remains a required host-driver component.
    cp -a /usr/lib/x86_64-linux-gnu/libvulkan.so.1*    /src/hiveos/dist-amd/
    chmod -R a+rX /src/hiveos/dist-amd
'

echo ">> Done. Verifying glibc symbol ceiling (must be <= 2.31):"
for f in keryx-miner-supr libkeryxopencl.so libvulkan.so.1; do
    max=$(objdump -T "$OUT/$f" 2>/dev/null | grep -oE 'GLIBC_[0-9.]+' | sort -V | tail -1)
    printf '   %-22s max %s\n' "$f" "$max"
    if [[ "$(printf '%s\n' GLIBC_2.31 "$max" | sort -V | tail -1)" != "GLIBC_2.31" ]]; then
        echo "ERROR: $f requires $max, above the promised HiveOS GLIBC_2.31 ceiling" >&2
        exit 1
    fi
done

if [[ "$MINER_ONLY" == 0 ]]; then
    echo ">> Building required AMD/Vulkan inference engine (AVX + baseline ISA) ..."
    "$REPO/hiveos/build-keryx-llama-vk.sh"
    "$REPO/hiveos/build-keryx-llama-vk.sh" "$(nproc)" noavx
    # shellcheck source=hiveos/amd-inference-route.sh
    source "$REPO/hiveos/amd-inference-route.sh"
    keryx_require_amd_inference_route "$OUT"
else
    # A successful miner-only refresh must not inherit a previously packageable sidecar route from
    # this shared output directory. Invalidate only the provenance manifests (leave large engines
    # cached for the next full rebuild); every AMD packager requires these manifests and therefore
    # fails closed exactly as the option promises.
    rm -f "$OUT/libkeryx-llama-vk.so.manifest" "$OUT/libkeryx-llama-vk-noavx.so.manifest"
    echo ">> --miner-only: skipping Vulkan inference engines (intentionally not packageable)."
fi
ls -la "$OUT"
