#!/usr/bin/env bash
# Translate the HiveOS flight-sheet fields into a keryx-miner-supr command line.
# Sourced by h-run.sh; writes $CUSTOM_CONFIG_FILENAME with the CLI_ARGS string.
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
#   Extra args($CUSTOM_USER_CONFIG)-> appended verbatim (e.g. "--light --cuda-device 0")

[[ -t 1 ]] || exec 2>/dev/null   # quiet when run by the agent

. /hive/miners/custom/keryx-miner-supr/h-manifest.conf 2>/dev/null

# Ensure the stratum scheme is present — without it the miner falls back to gRPC.
url="$CUSTOM_URL"
[[ "$url" == *://* ]] || url="stratum+tcp://$url"

args="-a ${CUSTOM_TEMPLATE} -s ${url}"
# Pass the flight-sheet password to the POOL via --pool-password (NOT -p, which is --port).
[[ -n "$CUSTOM_PASS" ]] && args="$args --pool-password ${CUSTOM_PASS}"

# Model selection: honour whatever the user put in extra args — a tier flag
# (--very-light/--light/--high/--very-high), the --tier <name> form, OR --force-model <csv>.
# If NONE is present, default to per-card AUTO: each GPU loads the heaviest model its own VRAM
# can hold (a mixed rig gets a different model per card) = max reward. Passing a tier flag still
# pins every card to that one tier; --tier auto is the explicit form of the default.
# (Recognising --force-model here avoids a spurious "--light" that would clash with it.)
extra="$CUSTOM_USER_CONFIG"
case " $extra " in
  *" --very-light "*|*" --light "*|*" --high "*|*" --very-high "*|*" --tier "*|*" --tier="*|*" --force-model "*|*" --force-model="*) : ;;  # model chosen
  *) extra="--tier auto $extra" ;;   # default: per-card AUTO best-fit
esac

args="$args ${extra}"

echo "CLI_ARGS=\"${args}\"" > "$CUSTOM_CONFIG_FILENAME"
