#!/usr/bin/env bash
set -euo pipefail

# Rebuild the committed modern PoM walk image with the architecture inventory expected by
# build.rs. The output defaults to /tmp so replacing the production artifact is always an explicit
# review step. When a baseline is supplied, every established sm_80 entry point must retain
# instruction-identical SASS; this protects the already-fast GA100 path while new optional kernels
# are added. Blackwell coverage is checked by the architecture inventory below, not this sm_80 SASS
# comparison.

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-/tmp/pom-all.fatbin}
baseline=${2:-}
nvcc_bin=${NVCC:-nvcc}
cuobjdump_bin=${CUOBJDUMP:-cuobjdump}

required_sms=(75 80 86 87 88 89 90 100 103 110 120 121)
available=$($nvcc_bin --list-gpu-code)
for sm in "${required_sms[@]}"; do
    if ! grep -qw "sm_${sm}" <<<"$available"; then
        echo "error: $nvcc_bin does not support required sm_${sm}" >&2
        exit 1
    fi
done

args=(
    -O3 -fatbin
    -gencode=arch=compute_75,code=compute_75
    -gencode=arch=compute_80,code=compute_80
)
for sm in "${required_sms[@]}"; do
    args+=("-gencode=arch=compute_${sm},code=sm_${sm}")
done
args+=(-Xptxas=-v -o "$out" "$repo_dir/src/pom_mine.cu")

"$nvcc_bin" "${args[@]}"

elf_inventory=$($cuobjdump_bin -lelf "$out")
ptx_inventory=$($cuobjdump_bin -lptx "$out")
for sm in "${required_sms[@]}"; do
    grep -q "sm_${sm}\.cubin" <<<"$elf_inventory" || {
        echo "error: generated fatbin is missing native sm_${sm}" >&2
        exit 1
    }
done
for sm in 75 80; do
    grep -q "sm_${sm}\.ptx" <<<"$ptx_inventory" || {
        echo "error: generated fatbin is missing compute_${sm} PTX" >&2
        exit 1
    }
done

if [[ -n "$baseline" ]]; then
    [[ -f "$baseline" ]] || { echo "error: baseline not found: $baseline" >&2; exit 1; }
    scratch=$(mktemp -d)
    trap 'rm -rf -- "$scratch"' EXIT
    established=(
        pom_mine_v4_seeded
        pom_mine_v4_chase_seeded
        pom_mine_v4_tc_seeded
        pom_mine_v4_ncf_seeded
        pom_seed_h10_batch
    )
    for symbol in "${established[@]}"; do
        # `-Xptxas=-v` is deliberately present on the reproducible build command. cuobjdump appends
        # that container option after the SASS, even though it cannot affect an instruction. Compare
        # only through the cubin disassembly terminator so metadata spelling cannot create a false
        # kernel-regression failure.
        "$cuobjdump_bin" -sass -arch sm_80 -fun "$symbol" "$baseline" \
            | sed '/^Fatbin ptx code:/,$d' >"$scratch/old-$symbol.sass"
        "$cuobjdump_bin" -sass -arch sm_80 -fun "$symbol" "$out" \
            | sed '/^Fatbin ptx code:/,$d' >"$scratch/new-$symbol.sass"
        cmp "$scratch/old-$symbol.sass" "$scratch/new-$symbol.sass" || {
            echo "error: established sm_80 SASS changed: $symbol" >&2
            exit 1
        }
        echo "sm_80 SASS unchanged: $symbol"
    done
fi

sha256sum "$repo_dir/src/pom_mine.cu" "$out"
stat -c '%n %s bytes' "$out"
printf '%s\n' "$elf_inventory" "$ptx_inventory"
