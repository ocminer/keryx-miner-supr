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
# per-card model map on AMD, unlike CUDA), and --force-model bypasses the VRAM gate. The default is
# --very-light (EXAONE-4.0-1.2B) — the OOM-safe choice for the H4 lineup: AMD loads the whole tier blob
# AND the model's inference context into VRAM, and post-H4 the light tier is Mistral-7B (needs 8 GB, OOMs
# an 8 GB card), so we default to very-light which fits ANY card (min_vram_mb=0, ~883 MiB blob). This is
# the direct H4 analog of the pre-H4 --light=Gemma default (Gemma WAS the min-VRAM model). AMD has no
# real --tier auto (the pom-opencl path falls back to Light, i.e. Mistral, on "auto"), so we pin the
# safe floor explicitly; operators with bigger cards opt up: --light (Mistral, 8 GB) / --high (Qwen3.6,
# 24 GB) / --force-model qwen3.6-27b. A future enhancement is real AMD auto-tier (biggest tier that fits).
extra="$CUSTOM_USER_CONFIG"
case " $extra " in
  *" --very-light "*|*" --light "*|*" --high "*|*" --very-high "*|*" --tier "*|*" --tier="*|*" --force-model "*|*" --force-model="*) : ;;  # model chosen
  *) extra="--very-light $extra" ;;   # default: --very-light (EXAONE) — fits any card; OOM-safe on H4. Override with --light/--high.
esac

args="$args ${extra}"

echo "CLI_ARGS=\"${args}\"" > "$CUSTOM_CONFIG_FILENAME"
