#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=hiveos/amd-inference-route.sh
source "$REPO/hiveos/amd-inference-route.sh"

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
SRC="$TMP/source"
DST="$TMP/staged"
EMPTY="$TMP/empty"
mkdir -p "$SRC" "$DST" "$EMPTY"

expect_rejected() {
    local dir="$1" label="$2"
    if keryx_require_amd_inference_route "$dir" >/dev/null 2>&1; then
        echo "ERROR: $label was accepted" >&2
        exit 1
    fi
}

# A tiny ELF supplies the production export names without loading Vulkan or touching a GPU. The
# manifest still binds it to the real checked-out wrapper/picker/shader inputs, exercising the same
# stale-artifact checks used by release packaging.
cat > "$TMP/engine.c" <<'C'
#define F(name) void name(void) {}
F(keryx_llama_abi)
F(keryx_llama_vk_abi)
F(keryx_llama_load)
F(keryx_llama_free)
F(keryx_llama_generate)
F(keryx_llama_pom_ready)
F(keryx_llama_pom_n_chunks)
F(keryx_llama_pom_supl_bytes)
F(keryx_llama_pom_fetch)
F(keryx_llama_pom_mine)
F(keryx_llama_pom_pci)
F(keryx_vk_picker_abi)
F(keryx_vk_pick_discrete_device)
F(keryx_vk_device_pci)
F(vkGetInstanceProcAddr)
C
cc -shared -fPIC -Wl,-rpath,'$ORIGIN' "$TMP/engine.c" -o "$SRC/libkeryx-llama-vk.so"
cc -shared -fPIC -Wl,-rpath,'$ORIGIN' "$TMP/engine.c" -o "$SRC/libkeryx-llama-vk-noavx.so"
cc -shared -fPIC -Wl,-soname,libvulkan.so.1 "$TMP/engine.c" -o "$SRC/libvulkan.so.1"
keryx_write_amd_engine_manifest "$SRC/libkeryx-llama-vk.so" avx
keryx_write_amd_engine_manifest "$SRC/libkeryx-llama-vk-noavx.so" noavx

keryx_require_amd_inference_route "$SRC"
keryx_copy_amd_inference_route "$SRC" "$DST"
[[ -f "$DST/libkeryx-llama-vk.so" ]]
[[ -f "$DST/libkeryx-llama-vk.so.manifest" ]]
[[ -f "$DST/libkeryx-llama-vk-noavx.so" ]]
[[ -f "$DST/libkeryx-llama-vk-noavx.so.manifest" ]]
[[ -f "$DST/libvulkan.so.1" ]]

expect_rejected "$EMPTY" "missing inference route"

MISSING_BASE="$TMP/missing-baseline"
cp -a "$SRC" "$MISSING_BASE"
rm "$MISSING_BASE/libkeryx-llama-vk-noavx.so" "$MISSING_BASE/libkeryx-llama-vk-noavx.so.manifest"
expect_rejected "$MISSING_BASE" "route without its baseline-ISA engine"

WRONG_VARIANT="$TMP/wrong-variant"
cp -a "$SRC" "$WRONG_VARIANT"
sed -i 's/^variant=avx$/variant=noavx/' "$WRONG_VARIANT/libkeryx-llama-vk.so.manifest"
expect_rejected "$WRONG_VARIANT" "renamed/swapped engine variant"

NO_RPATH="$TMP/no-rpath"
cp -a "$SRC" "$NO_RPATH"
cc -shared -fPIC "$TMP/engine.c" -o "$NO_RPATH/libkeryx-llama-vk.so"
keryx_write_amd_engine_manifest "$NO_RPATH/libkeryx-llama-vk.so" avx
expect_rejected "$NO_RPATH" "engine without a colocated-library runpath"

BAD_LOADER="$TMP/bad-loader"
cp -a "$SRC" "$BAD_LOADER"
printf 'void not_vulkan(void) {}\n' > "$TMP/not-vulkan.c"
cc -shared -fPIC -Wl,-soname,libvulkan.so.1 "$TMP/not-vulkan.c" -o "$BAD_LOADER/libvulkan.so.1"
expect_rejected "$BAD_LOADER" "ELF impostor Vulkan loader"

printf x >> "$SRC/libkeryx-llama-vk.so"
expect_rejected "$SRC" "artifact/manifest hash mismatch"

# Restore a valid engine, then prove an advertised but non-ELF server is rejected instead of being
# copied with a swallowed dependency error.
cc -shared -fPIC -Wl,-rpath,'$ORIGIN' "$TMP/engine.c" -o "$SRC/libkeryx-llama-vk.so"
keryx_write_amd_engine_manifest "$SRC/libkeryx-llama-vk.so" avx
printf '#!/bin/sh\nexit 0\n' > "$SRC/llama-server"
chmod +x "$SRC/llama-server"
expect_rejected "$SRC" "invalid optional llama-server"

echo "AMD inference package-route checks: PASS"
