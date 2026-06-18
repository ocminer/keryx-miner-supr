#!/usr/bin/env bash
# Launch keryx-miner-supr for HiveOS. Runs the miner in the FOREGROUND and tees
# its output to $CUSTOM_LOG_BASENAME.log — h-stats.sh parses that log for
# hashrate/shares, and the HiveOS agent does NOT capture our stdout otherwise.
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

# This is the single static-cuda binary (CUDA worker linked in) — no plugin
# .so to find. libcuda.so.1 comes from the installed NVIDIA driver on the rig.
export LD_LIBRARY_PATH="$(pwd):/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[keryx-miner-supr] launching: ./keryx-miner-supr $CLI_ARGS"
# tee (not exec) so the log file exists for h-stats.sh.
./keryx-miner-supr $CLI_ARGS 2>&1 | tee "$CUSTOM_LOG_BASENAME.log"
