#!/usr/bin/env bash
# Launch keryx-miner-supr (AMD/OpenCL) for HiveOS. Runs in the FOREGROUND and
# tees output to $CUSTOM_LOG_BASENAME.log — h-stats.sh parses that log, and the
# HiveOS agent does NOT capture our stdout otherwise.
cd "$(dirname "$(realpath "$0")")"

. h-manifest.conf

# The HiveOS agent runs h-config.sh (with the flight-sheet CUSTOM_* vars in scope,
# %WAL%/%WORKER_NAME% already substituted) BEFORE us, and writes
# $CUSTOM_CONFIG_FILENAME. Do NOT unconditionally regenerate it here: our process
# may not carry the CUSTOM_* env vars, so re-running h-config.sh would emit an
# empty `-a`/`-s` and the miner aborts ("--mining-address requires a value").
# Read the config the agent already generated; only (re)generate as a fallback.
[[ -f "$CUSTOM_CONFIG_FILENAME" ]] && . "$CUSTOM_CONFIG_FILENAME"   # -> $CLI_ARGS
if [[ -z "$CLI_ARGS" ]]; then . h-config.sh; . "$CUSTOM_CONFIG_FILENAME"; fi

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
LOG="$CUSTOM_LOG_BASENAME.log"
: > "$LOG"   # fresh log for this run; preflight + miner output both append below

# Dynamic, CUDA-free AMD build: the binary dlopens ./libkeryxopencl.so (shipped
# next to it). libOpenCL.so.1 comes from the AMD/ROCm driver on the HiveOS rig.
export LD_LIBRARY_PATH="$(pwd):/opt/rocm/lib:/opt/rocm/lib64:/opt/amdgpu-pro/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- OpenCL preflight -------------------------------------------------------
# This dynamic AMD build mines via libkeryxopencl.so, which needs libOpenCL.so.1
# (the ICD loader) PLUS a registered AMD OpenCL ICD on the rig. If either is
# missing, or no GPU is visible, the miner creates zero workers and exits with
# "No workers specified" — which on HiveOS looks like a black screen with NO
# error. Emit an actionable diagnostic here (to the log AND the screen) so the
# failure is visible instead of silent.
preflight() { echo "[keryx-amd] $*" | tee -a "$LOG"; }
if ! ldconfig -p 2>/dev/null | grep -q 'libOpenCL\.so' \
   && ! ls /opt/rocm*/lib*/libOpenCL.so.1 /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 >/dev/null 2>&1; then
  preflight "ERROR: libOpenCL.so.1 not found — the AMD OpenCL runtime is missing."
  preflight "       Install it: 'apt-get install -y ocl-icd-libopencl1' plus the AMD GPU OpenCL driver."
  preflight "       Without it the miner creates no GPU workers and will exit."
fi
if command -v clinfo >/dev/null 2>&1; then
  ndev=$(timeout 15 clinfo 2>/dev/null | grep -c 'Device Name')
  preflight "OpenCL preflight: clinfo reports ${ndev:-0} device(s)."
  if [[ "${ndev:-0}" -eq 0 ]]; then
    preflight "WARNING: no OpenCL devices visible — the miner will exit with 'No workers specified'."
    preflight "         Check the AMD GPU driver and the OpenCL ICD registration in /etc/OpenCL/vendors/."
  fi
else
  preflight "note: 'clinfo' not installed — cannot preflight OpenCL device visibility ('apt-get install clinfo')."
fi
# ---------------------------------------------------------------------------

# --- Vulkan GPU-inference preflight ----------------------------------------
# OPoI inference runs on the AMD GPU via the bundled ./llama-server (Vulkan). That
# needs libvulkan.so.1 + a registered AMD Vulkan ICD (RADV/Mesa or AMDVLK) on the
# rig. If anything is missing the miner AUTO-FALLS BACK to CPU inference (slow but
# works) — NOT fatal. Report which path this rig will take so it's visible per-rig.
if [[ -x ./llama-server ]]; then
  vk_ok=1
  if ! ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1' \
     && ! ls /usr/lib/x86_64-linux-gnu/libvulkan.so.1 >/dev/null 2>&1; then
    preflight "Vulkan preflight: libvulkan.so.1 not found — OPoI inference will run on the CPU (slow)."
    preflight "                  For GPU inference: 'apt-get install -y libvulkan1 mesa-vulkan-drivers'."
    vk_ok=0
  fi
  if command -v vulkaninfo >/dev/null 2>&1; then
    amdvk=$(timeout 15 vulkaninfo --summary 2>/dev/null | grep -iE 'deviceName.*(AMD|Radeon|RADV|Instinct)' | head -1 | sed 's/^[[:space:]]*//')
    if [[ -n "$amdvk" ]]; then
      preflight "Vulkan preflight: AMD GPU visible to Vulkan ($amdvk)."
    else
      preflight "WARNING: no AMD GPU visible to Vulkan — OPoI inference falls back to CPU (slow)."
      vk_ok=0
    fi
  elif [[ "$vk_ok" == 1 ]]; then
    preflight "note: 'vulkaninfo' not installed — can't confirm AMD Vulkan visibility ('apt-get install vulkan-tools')."
  fi
  [[ "$vk_ok" == 1 ]] && preflight "Vulkan GPU inference: AVAILABLE — OPoI inference will run on the AMD GPU (fast)."
else
  preflight "note: no bundled llama-server — OPoI inference runs on the CPU (CPU-only build)."
fi
# ---------------------------------------------------------------------------

echo "[keryx-miner-supr-amd] launching: ./keryx-miner-supr $CLI_ARGS" | tee -a "$LOG"
# tee -a (not exec) so the log keeps the preflight lines + miner output for h-stats.sh.
./keryx-miner-supr $CLI_ARGS 2>&1 | tee -a "$LOG"
