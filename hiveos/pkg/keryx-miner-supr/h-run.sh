#!/usr/bin/env bash
# Launch keryx-miner-supr for HiveOS. The agent runs this in a screen session
# and captures stdout/stderr into $CUSTOM_LOG_BASENAME.log, which h-stats.sh
# parses. Must run the miner in the FOREGROUND (exec).
cd "$(dirname "$(realpath "$0")")"

. h-manifest.conf
. h-config.sh                      # regenerate keryx.conf from the flight sheet
. "$CUSTOM_CONFIG_FILENAME"        # -> $CLI_ARGS

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"

# Plugins (libkeryxcuda.so / libkeryxopencl.so) load from the binary's dir;
# libcuda.so.1 comes from the installed NVIDIA driver.
export LD_LIBRARY_PATH="$(pwd):/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[keryx-miner-supr] launching: ./keryx-miner-supr $CLI_ARGS"
exec ./keryx-miner-supr $CLI_ARGS
