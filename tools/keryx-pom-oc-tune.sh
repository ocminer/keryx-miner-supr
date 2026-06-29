#!/usr/bin/env bash
# keryx-pom-oc-tune.sh — overclock tuner for keryx Proof-of-Model (PoM) mining.
# Applies a MEMORY-clock + CORE-clock VF offset via suprtuner, then stays resident to re-apply it
# (driver reloads / watchdogs reset offsets). Run it AFTER the keryx miner is mining (so the GPU is
# at its boost clock).
#
# WHY BOTH CLOCKS (measured 2026-06-29 on a 5090, microbench of the real pom_mine walk):
#   The PoM walk is a LATENCY-BOUND pointer-chase (gather -> mix64 -> 64-bit-mod -> next gather) and
#   is NOT power-bound (69 MH/s @ only 106 W vs a 600 W cap). Cripple-clock A/B:
#     mem 13801->810  : 69.2 -> 3.6 MH/s  (-95%)  => MEMORY clock dominates
#     core 2842->990  : 69.2 -> 26.3 MH/s (-62%)  => CORE clock is a strong secondary
#       (core gates the memory-request ISSUE RATE in the serial dependent chain)
#   Because it's far from the power cap, raising BOTH clocks helps with no power tradeoff — unlike
#   PearlHash/kHeavyHash (power-bound -> core-offset only). suprtuner uses NVML GpcClk/MemClk VF
#   offsets (headless, no X).
#
# ⚠️ SAFETY — read before pushing the mem offset hard:
#   * PoM is MEMORY-HARD. A memory error from too-aggressive mem OC does NOT crash — it silently
#     corrupts the walk -> wrong proof -> the POOL REJECTS the share. So watch your pool reject rate,
#     not just "did it hang". RAMP the mem offset up in small steps and back off the moment rejects
#     appear. The defaults below are a conservative STARTING point, not a target.
#   * CORE offset wedge limit (from the pearl tuning): keep core <= +250. +400 wedged a 5090
#     (GSP/WPR2) — recover headless with `rmmod nvidia*; echo 1 > /sys/bus/pci/devices/<bdf>/reset;
#     modprobe nvidia` (no reboot). This script refuses core > 250.
#   * HBM cards (CMP 170HX / A100 / H100) return NOT_SUPPORTED for the mem offset — that's expected;
#     suprtuner skips mem on them and still applies the core offset.
#
# RAMP PROCEDURE (per rig / per card type):
#   1. Start miner -> let it reach steady hashrate. Note the baseline MH/s + reject count.
#   2. Run this with a modest mem offset (default +600). Confirm MH/s up, rejects flat for ~10 min.
#   3. Raise KERYX_MEM_OFFSET in +250 steps, re-running, until MH/s plateaus OR rejects appear.
#      Set the final value ~250 BELOW the first offset that produced rejects (headroom for heat/voltage drift).
#   4. Core: +200 is usually safe; nudge toward +250 if stable. Core rarely changes rejects (it's not
#      the memory); if it wedges, lower it.
#
# USAGE:
#   tools/keryx-pom-oc-tune.sh                 # apply defaults to all GPUs, stay resident
#   KERYX_MEM_OFFSET=1000 KERYX_CORE_OFFSET=200 tools/keryx-pom-oc-tune.sh
#   KERYX_OC_DEVICES=0,2 tools/keryx-pom-oc-tune.sh    # only GPUs 0 and 2 (mixed-card rigs: run one per card group)
#   tools/keryx-pom-oc-tune.sh --dry-run       # preview current clocks + planned offsets, then exit
#   tools/keryx-pom-oc-tune.sh --once          # apply once and exit (no resident re-apply)
set -u

ST="${SUPRTUNER:-/usr/local/bin/suprtuner}"; [ -x "$ST" ] || ST="$(command -v suprtuner 2>/dev/null || echo /tmp/suprtuner)"
[ -x "$ST" ] || { echo "keryx-pom-oc-tune: suprtuner not found (set SUPRTUNER=/path/to/suprtuner)"; exit 1; }

MEM_OFFSET="${KERYX_MEM_OFFSET:-600}"      # the dominant lever — conservative start; RAMP per card (see above)
CORE_OFFSET="${KERYX_CORE_OFFSET:-200}"    # strong secondary; keep <= 250
INTERVAL="${KERYX_OC_INTERVAL:-60}"        # resident re-apply cadence (s)
DEVICES="${KERYX_OC_DEVICES:-}"            # e.g. 0,2 ; empty = all GPUs

# --- safety caps ---
[ "$CORE_OFFSET" -le 250 ] || { echo "keryx-pom-oc-tune: refusing core offset > 250 (wedge risk; see header)"; exit 1; }
[ "$MEM_OFFSET" -ge 0 ]    || { echo "keryx-pom-oc-tune: mem offset must be >= 0"; exit 1; }
if [ "$MEM_OFFSET" -gt 2000 ]; then
  echo "keryx-pom-oc-tune: WARNING — mem offset $MEM_OFFSET > 2000 is aggressive; expect rejects from memory errors. Continuing in 5s (Ctrl-C to abort)."
  sleep 5
fi

# --- argument passthrough ---
MODE="resident"
for a in "$@"; do
  case "$a" in
    --dry-run) MODE="dry" ;;
    --once)    MODE="once" ;;
    *) echo "keryx-pom-oc-tune: unknown arg '$a' (use --dry-run or --once)"; exit 1 ;;
  esac
done

# Build the suprtuner device flag.
devflag=(); [ -n "$DEVICES" ] && devflag=(--devices "$DEVICES")

echo "keryx-pom-oc-tune: PoM is memory-bound + not power-bound -> applying mem=+${MEM_OFFSET} core=+${CORE_OFFSET}" \
     "MHz to ${DEVICES:-all GPUs} (re-apply every ${INTERVAL}s). HBM cards skip mem automatically."

case "$MODE" in
  dry)  exec sudo "$ST" --mem-offset "$MEM_OFFSET" --core-offset "$CORE_OFFSET" "${devflag[@]}" --dry-run ;;
  once) exec sudo "$ST" --mem-offset "$MEM_OFFSET" --core-offset "$CORE_OFFSET" "${devflag[@]}" ;;
  *)    exec sudo "$ST" --mem-offset "$MEM_OFFSET" --core-offset "$CORE_OFFSET" "${devflag[@]}" --interval "$INTERVAL" ;;
esac
