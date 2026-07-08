#!/usr/bin/env bash
# Translate the HiveOS flight-sheet fields into a keryx-miner-supr command line
# (AMD/OpenCL build). Sourced by h-run.sh; writes $CUSTOM_CONFIG_FILENAME with
# the CLI_ARGS string.
#
# Flight-sheet mapping:
#   Pool URL  ($CUSTOM_URL)        -> -s stratum+tcp://host:port  (scheme required)
#   Wallet    ($CUSTOM_TEMPLATE)   -> -a keryx:addr.worker
#   Password  ($CUSTOM_PASS)       -> --pool-password (sent in stratum mining.authorize).
#                                     Most keryx pools ignore it; on suprnova set e.g.
#                                     Pass=d=16 for static difficulty 16. NEVER map it to
#                                     -p: keryx's `-p` is --port (a NUMBER), so `-p x` aborts
#                                     the miner ('Invalid value "x" for --port' = the old
#                                     black screen / instant exit).
#   Extra args($CUSTOM_USER_CONFIG)-> appended verbatim
#                                     (e.g. "--light --opencl-device 0,1")

[[ -t 1 ]] || exec 2>/dev/null   # quiet when run by the agent

. /hive/miners/custom/keryx-miner-supr-amd/h-manifest.conf 2>/dev/null

# Ensure the stratum scheme is present — without it the miner falls back to gRPC.
url="$CUSTOM_URL"
[[ "$url" == *://* ]] || url="stratum+tcp://$url"

args="-a ${CUSTOM_TEMPLATE} -s ${url}"
# Pass the flight-sheet password to the POOL via --pool-password (NOT -p, which is --port).
[[ -n "$CUSTOM_PASS" ]] && args="$args --pool-password ${CUSTOM_PASS}"

# Model selection: honour whatever the user put in extra args — a tier flag (--very-light/--light/
# --high/--very-high), the --tier <name> form, OR --force-model <csv>. Only if NONE is present do we
# prepend the default. As of v0.6.9.3 the AMD/OpenCL build HONORS these overrides (was hardcoded Light
# — issue #7): a user override is applied PROCESS-WIDE (one resident tier for all cards; there is no
# per-card model map on AMD, unlike CUDA), and --force-model bypasses the VRAM gate. The default stays
# --light (Gemma-3-4B) — the OOM-safe choice, since AMD loads the whole tier blob into VRAM and a bigger
# tier would OOM a small card. So we differ from the NVIDIA launcher (per-card --tier auto): AMD pins
# --light and lets the operator opt into a heavier tier explicitly (--tier high / --force-model qwen3-32b).
extra="$CUSTOM_USER_CONFIG"
case " $extra " in
  *" --very-light "*|*" --light "*|*" --high "*|*" --very-high "*|*" --tier "*|*" --tier="*|*" --force-model "*|*" --force-model="*) : ;;  # model chosen
  *) extra="--light $extra" ;;   # default: --light (Gemma-3-4B) — OOM-safe; override with --tier/--force-model
esac

args="$args ${extra}"

echo "CLI_ARGS=\"${args}\"" > "$CUSTOM_CONFIG_FILENAME"
