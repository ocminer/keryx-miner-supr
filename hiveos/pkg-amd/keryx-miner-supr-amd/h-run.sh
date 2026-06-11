#!/usr/bin/env bash
# Launch keryx-miner-supr (AMD/OpenCL) for HiveOS. Runs in the FOREGROUND and
# tees output to $CUSTOM_LOG_BASENAME.log — h-stats.sh parses that log, and the
# HiveOS agent does NOT capture our stdout otherwise.
cd "$(dirname "$(realpath "$0")")"

. h-manifest.conf
. h-config.sh                      # regenerate keryx.conf from the flight sheet
. "$CUSTOM_CONFIG_FILENAME"        # -> $CLI_ARGS

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

# Dynamic, CUDA-free AMD build: the binary dlopens ./libkeryxopencl.so (shipped
# next to it). libOpenCL.so.1 comes from the AMD/ROCm driver on the HiveOS rig.
export LD_LIBRARY_PATH="$(pwd):/opt/rocm/lib:/opt/rocm/lib64:/opt/amdgpu-pro/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[keryx-miner-supr-amd] launching: ./keryx-miner-supr $CLI_ARGS"
# tee (not exec) so the log file exists for h-stats.sh.
./keryx-miner-supr $CLI_ARGS 2>&1 | tee "$CUSTOM_LOG_BASENAME.log"
