#!/usr/bin/env bash
# Shared validation/copy helpers for shippable AMD/OpenCL archives.
#
# Source this file; do not execute it directly.  A pom-opencl miner cannot safely use a bare
# llama-server: before spawning the child it maps ggml's device index back to the selected OpenCL
# worker's full PCI identity through exports in libkeryx-llama-vk.so.  The sidecar is therefore
# required even when llama-server is also bundled (normal inference uses the sidecar directly;
# the server is a fallback).

KERYX_AMD_ROUTE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERYX_AMD_ROUTE_REPO="$(cd "$KERYX_AMD_ROUTE_DIR/.." && pwd)"
KERYX_LLAMA_VK_TAG="b10015"

_keryx_sha256() {
    sha256sum "$1" | awk '{print $1}'
}

_keryx_manifest_value() {
    local manifest="$1" key="$2"
    awk -F= -v key="$key" '$1 == key { sub(/^[^=]*=/, ""); print; found=1; exit } END { if (!found) exit 1 }' "$manifest"
}

_keryx_engine_sources() {
    printf '%s\n' \
        "tools/keryx-llama/keryx_llama_vk.cpp" \
        "tools/keryx-llama/keryx_vk_exports.inc.cpp" \
        "tools/keryx-llama/pom_walk_vk.comp" \
        "tools/keryx-llama/pom_fetch_vk.comp" \
        "hiveos/build-keryx-llama-vk.sh"
}

# Write a provenance manifest next to a freshly built engine.  Packaging validates every input
# hash, which prevents an older sidecar (with stale PCI-selection or walk code) being silently
# paired with a newly built miner under the same release version.
keryx_write_amd_engine_manifest() {
    local engine="$1" variant="$2" manifest tmp rel key
    manifest="${engine}.manifest"
    [[ -f "$engine" ]] || { echo "ERROR: cannot manifest missing Vulkan engine: $engine" >&2; return 1; }
    command -v sha256sum >/dev/null 2>&1 || { echo "ERROR: sha256sum is required" >&2; return 1; }
    tmp=$(mktemp "${manifest}.tmp.XXXXXX") || return 1
    {
        echo "format=1"
        echo "variant=$variant"
        echo "llama_tag=$KERYX_LLAMA_VK_TAG"
        echo "artifact_sha256=$(_keryx_sha256 "$engine")"
        while IFS= read -r rel; do
            key=$(printf '%s' "$rel" | tr '/.-' '___')
            echo "source_${key}_sha256=$(_keryx_sha256 "$KERYX_AMD_ROUTE_REPO/$rel")"
        done < <(_keryx_engine_sources)
    } > "$tmp"
    mv -f "$tmp" "$manifest"
    chmod a+r "$manifest"
}

_keryx_defined_symbols() {
    local engine="$1"
    if command -v nm >/dev/null 2>&1; then
        nm -D --defined-only "$engine" 2>/dev/null | awk '{print $NF}'
    elif command -v objdump >/dev/null 2>&1; then
        objdump -T "$engine" 2>/dev/null | awk '$4 != "*UND*" {print $NF}'
    else
        echo "ERROR: nm or objdump is required to validate $engine" >&2
        return 1
    fi
}

keryx_validate_amd_engine() {
    local engine="$1" manifest="${1}.manifest" symbols symbol rel key expected actual
    local expected_variant variant dynamic
    [[ -r "$engine" ]] || { echo "ERROR: Vulkan engine missing/unreadable: $engine" >&2; return 1; }
    [[ -r "$manifest" ]] || {
        echo "ERROR: $manifest missing; rebuild the engine with hiveos/build-keryx-llama-vk.sh" >&2
        return 1
    }
    command -v sha256sum >/dev/null 2>&1 || { echo "ERROR: sha256sum is required" >&2; return 1; }
    command -v readelf >/dev/null 2>&1 || { echo "ERROR: readelf is required" >&2; return 1; }

    case "$(basename "$engine")" in
        libkeryx-llama-vk.so)       expected_variant=avx ;;
        libkeryx-llama-vk-noavx.so) expected_variant=noavx ;;
        *) echo "ERROR: unexpected AMD Vulkan-engine filename: $engine" >&2; return 1 ;;
    esac

    [[ "$(_keryx_manifest_value "$manifest" format 2>/dev/null)" == "1" ]] || {
        echo "ERROR: unsupported/malformed Vulkan-engine manifest: $manifest" >&2; return 1;
    }
    variant=$(_keryx_manifest_value "$manifest" variant 2>/dev/null) || {
        echo "ERROR: build variant missing from $manifest" >&2; return 1;
    }
    [[ "$variant" == "$expected_variant" ]] || {
        echo "ERROR: $engine is variant '$variant', expected '$expected_variant' for this filename" >&2
        return 1
    }
    [[ "$(_keryx_manifest_value "$manifest" llama_tag 2>/dev/null)" == "$KERYX_LLAMA_VK_TAG" ]] || {
        echo "ERROR: Vulkan engine was built from the wrong llama.cpp pin: $manifest" >&2; return 1;
    }
    expected=$(_keryx_manifest_value "$manifest" artifact_sha256 2>/dev/null) || {
        echo "ERROR: artifact hash missing from $manifest" >&2; return 1;
    }
    actual=$(_keryx_sha256 "$engine")
    [[ "$actual" == "$expected" ]] || {
        echo "ERROR: Vulkan engine hash does not match $manifest" >&2; return 1;
    }

    readelf -h "$engine" >/dev/null 2>&1 || {
        echo "ERROR: $engine is not a readable ELF library" >&2; return 1
    }
    dynamic=$(readelf -d "$engine" 2>/dev/null) || {
        echo "ERROR: cannot inspect dynamic dependencies for $engine" >&2; return 1
    }
    grep -Eq '\((RPATH|RUNPATH)\).*\[\$ORIGIN([/:][^]]*)?\]' <<< "$dynamic" || {
        echo "ERROR: $engine has no \$ORIGIN RPATH/RUNPATH for its bundled Vulkan loader" >&2
        return 1
    }

    while IFS= read -r rel; do
        [[ -r "$KERYX_AMD_ROUTE_REPO/$rel" ]] || {
            echo "ERROR: Vulkan-engine source input missing: $rel" >&2; return 1;
        }
        key=$(printf '%s' "$rel" | tr '/.-' '___')
        expected=$(_keryx_manifest_value "$manifest" "source_${key}_sha256" 2>/dev/null) || {
            echo "ERROR: source hash for $rel missing from $manifest" >&2; return 1;
        }
        actual=$(_keryx_sha256 "$KERYX_AMD_ROUTE_REPO/$rel")
        [[ "$actual" == "$expected" ]] || {
            echo "ERROR: $engine is stale relative to $rel; rebuild it before packaging" >&2
            return 1
        }
    done < <(_keryx_engine_sources)

    symbols=$(_keryx_defined_symbols "$engine") || return 1
    for symbol in \
        keryx_llama_abi keryx_llama_vk_abi keryx_llama_load keryx_llama_free \
        keryx_llama_generate keryx_llama_pom_ready keryx_llama_pom_n_chunks \
        keryx_llama_pom_supl_bytes keryx_llama_pom_fetch keryx_llama_pom_mine \
        keryx_llama_pom_pci keryx_vk_picker_abi keryx_vk_pick_discrete_device \
        keryx_vk_device_pci; do
        grep -qx "$symbol" <<< "$symbols" || {
            echo "ERROR: $engine is missing required export $symbol" >&2
            return 1
        }
    done
}

_keryx_server_local_dependencies() {
    local server="$1"
    command -v readelf >/dev/null 2>&1 || {
        echo "ERROR: readelf is required to validate $server" >&2
        return 1
    }
    readelf -h "$server" >/dev/null 2>&1 || {
        echo "ERROR: $server is not a readable ELF executable" >&2
        return 1
    }
    readelf -d "$server" 2>/dev/null \
        | sed -n 's/.*Shared library: \[\(lib\(ggml\|llama\|mtmd\)[^]]*\)\].*/\1/p' \
        | sort -u
}

keryx_validate_amd_server() {
    local dir="$1" server="$1/llama-server" dep deps
    [[ -x "$server" ]] || { echo "ERROR: Vulkan llama-server missing/not executable: $server" >&2; return 1; }
    deps=$(_keryx_server_local_dependencies "$server") || return 1
    while IFS= read -r dep; do
        [[ -n "$dep" ]] || continue
        [[ -e "$dir/$dep" ]] || {
            echo "ERROR: $server requires missing local dependency $dep" >&2
            return 1
        }
    done <<< "$deps"
}

# Validate a source artifact directory before it is staged into any release archive.
keryx_require_amd_inference_route() {
    local dir="$1" vulkan_soname vulkan_symbols
    keryx_validate_amd_engine "$dir/libkeryx-llama-vk.so" || {
        echo "ERROR: no current, safely scoped AMD Vulkan inference route in $dir." >&2
        echo "       Run hiveos/build-keryx-llama-vk.sh after hiveos/build-amd-glibc.sh." >&2
        return 1
    }
    keryx_validate_amd_engine "$dir/libkeryx-llama-vk-noavx.so" || {
        echo "ERROR: baseline-ISA AMD Vulkan engine missing/stale in $dir." >&2
        echo "       Portable releases require both AVX and no-AVX variants." >&2
        return 1
    }
    [[ -e "$dir/libvulkan.so.1" ]] || {
        echo "ERROR: $dir/libvulkan.so.1 missing; the portable AMD route is incomplete" >&2
        return 1
    }
    command -v readelf >/dev/null 2>&1 || {
        echo "ERROR: readelf is required to validate $dir/libvulkan.so.1" >&2
        return 1
    }
    readelf -h "$dir/libvulkan.so.1" >/dev/null 2>&1 || {
        echo "ERROR: $dir/libvulkan.so.1 is not a readable ELF library" >&2
        return 1
    }
    vulkan_soname=$(readelf -d "$dir/libvulkan.so.1" 2>/dev/null \
        | sed -n 's/.*(SONAME).*\[\([^]]*\)\].*/\1/p' | head -1)
    [[ "$vulkan_soname" == "libvulkan.so.1" ]] || {
        echo "ERROR: $dir/libvulkan.so.1 has unexpected/missing SONAME '$vulkan_soname'" >&2
        return 1
    }
    vulkan_symbols=$(_keryx_defined_symbols "$dir/libvulkan.so.1") || return 1
    grep -qx vkGetInstanceProcAddr <<< "$vulkan_symbols" || {
        echo "ERROR: $dir/libvulkan.so.1 is missing the Vulkan loader entry point vkGetInstanceProcAddr" >&2
        return 1
    }

    # An optional subprocess fallback must also be internally complete; never ship a binary which
    # can be selected but cannot resolve its colocated llama/ggml libraries.
    if [[ -e "$dir/llama-server" ]]; then
        keryx_validate_amd_server "$dir" || return 1
    fi
}

keryx_copy_amd_inference_route() {
    local src="$1" dest="$2" dep deps binary ldd_output
    keryx_require_amd_inference_route "$src" || return 1

    cp -P "$src/libkeryx-llama-vk.so" "$src/libkeryx-llama-vk.so.manifest" "$dest/"
    cp -P "$src/libkeryx-llama-vk-noavx.so" "$src/libkeryx-llama-vk-noavx.so.manifest" "$dest/"
    cp -P "$src"/libvulkan.so.1* "$dest/"

    if [[ -e "$src/llama-server" ]]; then
        cp -P "$src/llama-server" "$dest/"
        # Copy each exact DT_NEEDED filename as a regular file. This avoids both a broad, swallowed
        # glob failure and a dangling symlink whose versioned target was omitted from the archive.
        deps=$(_keryx_server_local_dependencies "$src/llama-server") || return 1
        while IFS= read -r dep; do
            [[ -n "$dep" ]] || continue
            cp -L "$src/$dep" "$dest/$dep"
        done <<< "$deps"
        chmod +x "$dest/llama-server"
    fi
    chmod a+r "$dest"/libkeryx-llama-vk*.so "$dest"/libkeryx-llama-vk*.manifest

    # Catch any other unresolved dynamic dependency after staging. Base glibc/libstdc++ may resolve
    # from the build host; Vulkan and llama-family libraries were explicitly required above so the
    # host cannot mask their accidental omission.
    if command -v ldd >/dev/null 2>&1; then
        for binary in "$dest"/libkeryx-llama-vk*.so; do
            if ! ldd_output=$(LD_LIBRARY_PATH="$dest${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$binary" 2>&1); then
                echo "ERROR: ldd could not validate staged AMD inference artifact: $binary" >&2
                echo "$ldd_output" >&2
                return 1
            fi
            if grep -q 'not found' <<< "$ldd_output"; then
                echo "ERROR: staged AMD inference artifact has unresolved dependencies: $binary" >&2
                echo "$ldd_output" >&2
                return 1
            fi
        done
        if [[ -e "$dest/llama-server" ]]; then
            if ! ldd_output=$(LD_LIBRARY_PATH="$dest${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$dest/llama-server" 2>&1); then
                echo "ERROR: ldd could not validate staged llama-server" >&2
                echo "$ldd_output" >&2
                return 1
            fi
            if grep -q 'not found' <<< "$ldd_output"; then
                echo "ERROR: staged llama-server has unresolved dependencies" >&2
                echo "$ldd_output" >&2
                return 1
            fi
        fi
    fi
    echo ">> bundled required AMD Vulkan inference route (libkeryx-llama-vk.so + libvulkan.so.1)"
}
