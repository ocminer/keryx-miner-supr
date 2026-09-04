#!/usr/bin/env bash
set -euo pipefail

# Rebuild every PTX embedded by the legacy CUDA plugin from the one canonical source.
#
# CUDA 12.2 is intentional for sm_61..sm_90: it preserves the PTX ISA 8.2 driver floor of the
# original Pascal/Turing/Ampere images. CUDA 12.8 is the first release with consumer Blackwell
# sm_120 and also emits sm_100. Image digests make compiler inputs reproducible instead of depending
# on whichever /usr/local/cuda happens to be selected.

readonly CUDA122_IMAGE="${CUDA122_IMAGE:-nvidia/cuda@sha256:b7074ef6f9aa30c27fe747f3a7e10402ec442f001290718c73e0972d1ee61342}"
readonly CUDA128_IMAGE="${CUDA128_IMAGE:-nvidia/cuda@sha256:cd0f8e1c41628ff2513e9f7c42dd9bd8740fe87fd6615b9c4a6f392995a589c0}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
source_rel="plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu"
input_rels=(
    "${source_rel}"
    "plugins/cuda/kaspa-cuda-native/src/keccak-tiny.c"
    "plugins/cuda/kaspa-cuda-native/src/xoshiro256starstar.c"
)
resources_rel="plugins/cuda/resources"
resources_dir="${repo_root}/${resources_rel}"
stage_dir="$(mktemp -d "${resources_dir}/.ptx-build.XXXXXX")"
stage_rel="${stage_dir#"${repo_root}/"}"

cleanup() {
    rm -rf -- "${stage_dir}"
}
trap cleanup EXIT

# Do not publish a generation assembled from two revisions if an editor changes the canonical
# source or either textual include while nvcc is running.
hash_local_inputs() {
    local rel digest
    for rel in "${input_rels[@]}"; do
        digest=$(sha256sum "${repo_root}/${rel}" | awk '{print $1}')
        # Feed the aggregate canonical repo-relative names, not sha256sum's absolute paths. This
        # makes inputs_set_sha256 identical in every checkout while still binding file identity and
        # ordering as well as content.
        printf '%s  %s\n' "${digest}" "${rel}"
    done | sha256sum | awk '{print $1}'
}
inputs_sha_before=$(hash_local_inputs)

run_nvcc() {
    local image="$1"
    local arch="$2"
    docker run --rm --network=none \
        --user "$(id -u):$(id -g)" \
        --volume "${repo_root}:/work" \
        --workdir /work \
        "${image}" \
        nvcc "/work/${source_rel}" \
            -std=c++11 -O3 --restrict --ptx \
            --gpu-architecture="compute_${arch}" --gpu-code="sm_${arch}" \
            -Xptxas -O3 -Xcompiler -O3 \
            -o "/work/${stage_rel}/keryx-cuda-sm${arch}.ptx"
}

for arch in 61 75 80 86 89 90; do
    run_nvcc "${CUDA122_IMAGE}" "${arch}"
done
for arch in 100 120; do
    run_nvcc "${CUDA128_IMAGE}" "${arch}"
done

# Refuse to publish a partial or stale generation. The winner must be atomicMin, no generated image
# may retain the old atomicCAS path, and every virtual target must agree with its filename.
for arch in 61 75 80 86 89 90 100 120; do
    ptx="${stage_dir}/keryx-cuda-sm${arch}.ptx"
    grep -Fq ".target sm_${arch}" "${ptx}"
    grep -Fq ".global.min.u64" "${ptx}"
    if grep -Fq ".global.cas.b64" "${ptx}"; then
        echo "error: ${ptx} still contains atomicCAS" >&2
        exit 1
    fi
done

inputs_sha_after=$(hash_local_inputs)
if [[ "${inputs_sha_before}" != "${inputs_sha_after}" ]]; then
    echo "error: a local CUDA compile input changed during generation; refusing to publish mixed PTX" >&2
    exit 1
fi

for arch in 61 75 80 86 89 90 100 120; do
    mv -- "${stage_dir}/keryx-cuda-sm${arch}.ptx" "${resources_dir}/keryx-cuda-sm${arch}.ptx"
done

manifest_tmp="${stage_dir}/PTX_MANIFEST.txt"
{
    echo "Keryx legacy CUDA PTX manifest"
    echo "source=${source_rel}"
    for rel in "${input_rels[@]}"; do
        echo "input_sha256=$(sha256sum "${repo_root}/${rel}" | awk '{print $1}') ${rel}"
    done
    echo "inputs_set_sha256=${inputs_sha_before}"
    echo "flags=-std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_N --gpu-code=sm_N -Xptxas -O3 -Xcompiler -O3"
    echo "cuda_12_2_image=${CUDA122_IMAGE} (sm_61,sm_75,sm_80,sm_86,sm_89,sm_90)"
    echo "cuda_12_8_image=${CUDA128_IMAGE} (sm_100,sm_120)"
    echo
    echo "sha256  bytes  file"
    for arch in 61 75 80 86 89 90 100 120; do
        file="${resources_dir}/keryx-cuda-sm${arch}.ptx"
        printf '%s  %s  %s\n' \
            "$(sha256sum "${file}" | awk '{print $1}')" \
            "$(wc -c < "${file}" | tr -d ' ')" \
            "keryx-cuda-sm${arch}.ptx"
    done
} > "${manifest_tmp}"
mv -- "${manifest_tmp}" "${resources_dir}/PTX_MANIFEST.txt"

echo "Regenerated and verified PTX resources."
