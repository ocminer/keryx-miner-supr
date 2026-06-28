#!/usr/bin/env bash
# HiveOS stats reporter for keryx-miner-supr (AMD/OpenCL build).
# Sourced by the HiveOS agent; must export `khs` (total kH/s) and `stats` (JSON).
# Hashrate + shares come from the miner log; temps/fans come from the agent's
# $gpu_stats JSON.

. /hive/miners/custom/keryx-miner-supr-amd/h-manifest.conf 2>/dev/null
log="${CUSTOM_LOG_BASENAME}.log"

khs=0
stats=""

if [[ -f "$log" ]]; then
  tailbuf=$(tail -n 400 "$log" 2>/dev/null)

  # --- per-device hashrates (-> kH/s) -------------------------------------
  # Lines look like: "Device #0 (AMD Radeon RX 7600 XT): 332.19 Mhash/s"
  hs_json=$(awk '
    /Device #[0-9]+ .*: [0-9.]+ [GMk]?hash\/s/ {
      for (i=1;i<=NF;i++) {
        if ($i ~ /^[0-9.]+$/ && $(i+1) ~ /hash\/s/) { val=$i; unit=$(i+1) }
      }
      mult = 1
      if (unit ~ /^Ghash/) mult = 1000000
      else if (unit ~ /^Mhash/) mult = 1000
      else if (unit ~ /^khash/) mult = 1
      else if (unit ~ /^hash/)  mult = 0.001
      match($0, /Device #([0-9]+)/, m)
      dev[m[1]] = val * mult         # kH/s, keep latest per device
    }
    END {
      n=0; for (d in dev) n++
      printf "["
      for (i=0;i<n;i++) { printf "%s%.0f", (i?",":""), dev[i]+0 }
      printf "]"
    }' <<< "$tailbuf")
  [[ -z "$hs_json" || "$hs_json" == "[]" ]] && hs_json="[]"

  # --- total hashrate (-> kH/s) -------------------------------------------
  read tval tunit < <(grep -oE 'Current hashrate is [0-9.]+ [GMk]?hash/s' <<< "$tailbuf" \
                        | tail -n1 | grep -oE '[0-9.]+ [GMk]?hash/s')
  case "$tunit" in
    Ghash/s) khs=$(awk "BEGIN{printf \"%.0f\", $tval*1000000}") ;;
    Mhash/s) khs=$(awk "BEGIN{printf \"%.0f\", $tval*1000}") ;;
    khash/s) khs=$(awk "BEGIN{printf \"%.0f\", $tval}") ;;
    hash/s)  khs=$(awk "BEGIN{printf \"%.0f\", $tval/1000}") ;;
    *)       khs=0 ;;
  esac
  # Fall back to summing per-device if the total line wasn't found.
  if [[ "$khs" == "0" && "$hs_json" != "[]" ]]; then
    khs=$(echo "$hs_json" | jq 'add // 0' 2>/dev/null)
  fi

  # --- accepted / rejected shares -----------------------------------------
  acc=$(grep -oE 'Accepted: [0-9]+' <<< "$tailbuf" | tail -n1 | grep -oE '[0-9]+')
  [[ -z "$acc" ]] && acc=$(grep -c 'Share accepted' <<< "$tailbuf")
  rej=$(grep -ciE 'reject|invalid share' <<< "$tailbuf")
  [[ -z "$acc" ]] && acc=0
  [[ -z "$rej" ]] && rej=0
else
  hs_json="[]"; acc=0; rej=0
fi

# --- temps / fans / bus from the agent ------------------------------------
temp=$(jq -c '.temp' <<< "$gpu_stats" 2>/dev/null); [[ -z "$temp" || "$temp" == "null" ]] && temp="[]"
fan=$(jq -c '.fan'  <<< "$gpu_stats" 2>/dev/null); [[ -z "$fan"  || "$fan"  == "null" ]] && fan="[]"
busn=$(jq -c '.busids' <<< "$gpu_stats" 2>/dev/null); [[ -z "$busn" || "$busn" == "null" ]] && busn="[]"

uptime=$(( $(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s) ))
[[ $uptime -lt 0 ]] && uptime=0

stats=$(jq -nc \
  --argjson hs "$hs_json" \
  --argjson temp "$temp" \
  --argjson fan "$fan" \
  --argjson bus "$busn" \
  --argjson acc "${acc:-0}" \
  --argjson rej "${rej:-0}" \
  --argjson up "$uptime" \
  --arg ver "${CUSTOM_VERSION:-0.6.3.1}" \
  '{hs:$hs, hs_units:"khs", temp:$temp, fan:$fan, uptime:$up,
    ver:$ver, ar:[$acc,$rej], algo:"keryxhash", bus_numbers:$bus}' 2>/dev/null)

[[ -z "$stats" ]] && stats="{\"hs\":[],\"hs_units\":\"khs\",\"ver\":\"${CUSTOM_VERSION:-0.6.3.1}\",\"algo\":\"keryxhash\"}"
