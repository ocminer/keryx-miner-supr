#!/usr/bin/env bash
# Translate the HiveOS flight-sheet fields into a keryx-miner-supr command line
# (AMD/OpenCL build). Sourced by h-run.sh; writes $CUSTOM_CONFIG_FILENAME with
# the CLI_ARGS string.
#
# Flight-sheet mapping:
#   Pool URL  ($CUSTOM_URL)        -> -s stratum+tcp://host:port  (scheme required)
#   Wallet    ($CUSTOM_TEMPLATE)   -> -a keryx:addr.worker
#   Password  ($CUSTOM_PASS)       -> -p <pass>   (only if set; keryx pools usually ignore)
#   Extra args($CUSTOM_USER_CONFIG)-> appended verbatim
#                                     (e.g. "--light --opencl-device 0,1")

[[ -t 1 ]] || exec 2>/dev/null   # quiet when run by the agent

. /hive/miners/custom/keryx-miner-supr-amd/h-manifest.conf 2>/dev/null

# Ensure the stratum scheme is present — without it the miner falls back to gRPC.
url="$CUSTOM_URL"
[[ "$url" == *://* ]] || url="stratum+tcp://$url"

args="-a ${CUSTOM_TEMPLATE} -s ${url}"
[[ -n "$CUSTOM_PASS" ]] && args="$args -p ${CUSTOM_PASS}"

# Default tier is --light (TinyLlama only) unless the user overrides in extra args.
extra="$CUSTOM_USER_CONFIG"
[[ "$extra" == *"--light"* || "$extra" == *"--high"* || "$extra" == *"--very-high"* ]] || extra="--light $extra"

args="$args ${extra}"

echo "CLI_ARGS=\"${args}\"" > "$CUSTOM_CONFIG_FILENAME"
