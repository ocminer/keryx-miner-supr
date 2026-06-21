#!/usr/bin/env bash
# Translate the HiveOS flight-sheet fields into a keryx-miner-supr command line
# (AMD/OpenCL build). Sourced by h-run.sh; writes $CUSTOM_CONFIG_FILENAME with
# the CLI_ARGS string.
#
# Flight-sheet mapping:
#   Pool URL  ($CUSTOM_URL)        -> -s stratum+tcp://host:port  (scheme required)
#   Wallet    ($CUSTOM_TEMPLATE)   -> -a keryx:addr.worker
#   Password  ($CUSTOM_PASS)       -> IGNORED. keryx-miner-supr has NO password flag
#                                     (wallet-only auth); its `-p` is --port (a NUMBER),
#                                     so passing the flight-sheet password as `-p` made
#                                     the miner abort: 'Invalid value "x" for --port'.
#                                     HiveOS pool configs often default the password to
#                                     "x", so this aborted instantly (error only in the
#                                     log -> looked like a black screen). Do NOT pass -p.
#   Extra args($CUSTOM_USER_CONFIG)-> appended verbatim
#                                     (e.g. "--light --opencl-device 0,1")

[[ -t 1 ]] || exec 2>/dev/null   # quiet when run by the agent

. /hive/miners/custom/keryx-miner-supr-amd/h-manifest.conf 2>/dev/null

# Ensure the stratum scheme is present — without it the miner falls back to gRPC.
url="$CUSTOM_URL"
[[ "$url" == *://* ]] || url="stratum+tcp://$url"

args="-a ${CUSTOM_TEMPLATE} -s ${url}"
# NOTE: deliberately NOT passing $CUSTOM_PASS — keryx has no password flag and `-p`
# is --port (see header). Passing it aborts the miner.

# Default tier is --light (TinyLlama only) unless the user overrides in extra args.
extra="$CUSTOM_USER_CONFIG"
[[ "$extra" == *"--light"* || "$extra" == *"--high"* || "$extra" == *"--very-high"* ]] || extra="--light $extra"

args="$args ${extra}"

echo "CLI_ARGS=\"${args}\"" > "$CUSTOM_CONFIG_FILENAME"
