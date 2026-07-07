#!/usr/bin/env bash
# HiveOS stats reporter for keryx-miner-supr (AMD/OpenCL build).
# Reads the agent's $GPU_STATS_JSON for per-GPU busids/brand/temp/fan, brand-filters
# to the miner's GPUs, and emits aligned hs[]/temp[]/fan[]/bus_numbers[] arrays plus
# ar[]=[accepted,rejected] shares. Hashrate + shares are parsed from the miner log.
#
# Ported from upstream keryx-miner integration pkg 0.3.31 (commits c6f7948/599fb99/9b568a0),
# then hardened for the HiveOS UI (issue #4 "UI stats in hive (AMD)"):
#   * env_logger ISO timestamp parser (the miner logs "[2026-06-26T06:31:08Z INFO ...]")
#   * robust float->khs conversion via awk (rate*multiplier), decimal-count independent
#     (Thash *1e9, Ghash *1e6, Mhash *1e3, khash *1) — replaces fragile string slicing
#   * per-GPU line matches BOTH NVIDIA ("Device #N (name): R unit") and AMD/Vulkan
#     ("Device Vulkan #N: R unit") — the old NVIDIA-only grep left AMD hs[] at 0
#   * ar[] shares: cumulative "Shares: Accepted: N ..." + a count of "Share rejected by pool"
#   * uptime emitted as a JSON number (HiveOS expects numeric), algo=keryxhash, ver from manifest
#   * iGPU off-by-one guard (separate miner_dev counter that advances only for mining-brand cards)

. /hive/miners/custom/keryx-miner-supr-amd/h-manifest.conf

# Read the tail of the log ONCE and derive everything below from this in-memory copy, instead of
# re-reading the whole log file once per GPU (cheap on big rigs). tr -d '\000' guards stray NULs.
log=`tail -n 4000 "$CUSTOM_LOG_BASENAME.log" 2>/dev/null | tr -d '\000'`

stats_raw=`grep "Current hashrate is" <<< "$log" | tail -n 1`

maxDelay=120
time_now=`date +%s`

# The miner logs with env_logger, whose default line starts "[2026-06-26T06:31:08Z INFO ...]"
# (ISO-8601 UTC, leading '['). Older builds logged "2026-06-24 19:11:32.000+02:00 [INFO ]".
# Pull the timestamp anywhere on the line (bracket/position independent) and let GNU date parse it.
ts_field=`echo "$stats_raw" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+-][0-9]{2}:?[0-9]{2})?' | head -1`
time_rep=`date -d "$ts_field" +%s 2>/dev/null || echo 0`
diffTime=`echo $((time_now-time_rep)) | tr -d '-'`

# Convert a miner log line ("... R <unit>hash/s") to kilohashes/s. Robust to any decimal count:
# takes the value field (NF-1) and multiplies by the unit's khs factor. khash=1, Mhash=1e3,
# Ghash=1e6, Thash=1e9; default Mhash if no unit token is present.
to_khs() {
        awk '{
                rate = $(NF-1); m = 1000;
                if      ($0 ~ /Thash/) m = 1000000000;
                else if ($0 ~ /Ghash/) m = 1000000;
                else if ($0 ~ /Mhash/) m = 1000;
                else if ($0 ~ /khash/) m = 1;
                printf "%.0f\n", rate * m;
        }'
}

if [ "$diffTime" -lt "$maxDelay" ]; then
        total_hashrate=`echo "$stats_raw" | to_khs`
        [[ -z $total_hashrate ]] && total_hashrate=0

        # GPU status — from the HiveOS agent's gpu-stats (temps/fans/busids/brand).
        readarray -t gpu_stats < <( jq --slurp -r -c '.[] | .busids, .brand, .temp, .fan | join(" ")' $GPU_STATS_JSON 2>/dev/null)
        busids=(${gpu_stats[0]})
        brands=(${gpu_stats[1]})
        temps=(${gpu_stats[2]})
        fans=(${gpu_stats[3]})
        gpu_count=${#busids[@]}

        hash_arr=()
        busid_arr=()
        fan_arr=()
        temp_arr=()

        # This is the AMD/OpenCL package — the miner enumerates AMD GPUs, so brand-filter to AMD
        # unconditionally (the NVIDIA package detects NVIDIA-first; on a mixed rig that would pick
        # the wrong brand for THIS build). AMD rigs are AMD-only in practice anyway.
        BRAND_MINER="amd"

        # The miner numbers its workers "Device #0..#K-1" over the GPUs IT enumerates (mining brand
        # only, PCI-bus order). HiveOS's busid list can also contain an onboard iGPU the miner never
        # sees; using the raw loop index `i` then desyncs once such a device is skipped. Keep a
        # SEPARATE counter that advances only for mining-brand cards. No iGPU -> miner_dev == i.
        miner_dev=0
        for(( i=0; i < gpu_count; i++ )); do
                [[ "${brands[i]}" != $BRAND_MINER ]] && continue
                [[ "${busids[i]}" =~ ^([A-Fa-f0-9]+): ]]
                busid_arr+=($((16#${BASH_REMATCH[1]})))
                temp_arr+=(${temps[i]})
                fan_arr+=(${fans[i]})
                # Per-device line, NVIDIA: "... Device #N (<name>): 5.23 Ghash/s"
                #                  AMD:    "... Device Vulkan #N: 5.23 Ghash/s"
                # Match "Device [Vulkan ]#N" followed by space or colon so "#1" never hits "#10".
                gpu_raw=`grep -E "Device (Vulkan )?#$miner_dev[ :]" <<< "$log" | tail -n 1`
                if [[ -n "$gpu_raw" ]]; then
                        hashrate=`echo "$gpu_raw" | to_khs`
                else
                        hashrate=0
                fi
                [[ -z "$hashrate" ]] && hashrate=0
                hash_arr+=($hashrate)
                miner_dev=$((miner_dev+1))
        done

        # Shares. The miner logs a cumulative summary line
        #   "Shares: Accepted: N [Stale: S ][Low difficulty: L ][Duplicate: D ]Pending: P"
        # (each sub-count is emitted only when >0), plus per-event "Share rejected by pool ..." WARN
        # lines (there is no cumulative reject counter). ar=[accepted,rejected] feeds the HiveOS
        # shares column: accepted from the last summary line, rejected = count of reject events seen.
        ac=`grep "Shares:" <<< "$log" | tail -n 1 | grep -oE 'Accepted: [0-9]+' | grep -oE '[0-9]+'`
        rj=`grep -c "Share rejected" <<< "$log"`
        [[ -z $ac ]] && ac=0
        [[ -z $rj ]] && rj=0

        hash_json=`printf '%s\n' "${hash_arr[@]}" | jq -cs '.'`
        bus_numbers=`printf '%s\n' "${busid_arr[@]}" | jq -cs '.'`
        fan_json=`printf '%s\n' "${fan_arr[@]}" | jq -cs '.'`
        temp_json=`printf '%s\n' "${temp_arr[@]}" | jq -cs '.'`

        uptime=$(( `date +%s` - `stat -c %Y $CUSTOM_CONFIG_FILENAME 2>/dev/null || date +%s` ))
        [[ $uptime -lt 0 ]] && uptime=0

        stats=$(jq -nc \
                --argjson hs "$hash_json" \
                --arg ver "$CUSTOM_VERSION" \
                --argjson bus_numbers "$bus_numbers" \
                --argjson fan "$fan_json" \
                --argjson temp "$temp_json" \
                --arg uptime "$uptime" \
                --argjson ar "[${ac:-0}, ${rj:-0}]" \
                '{ hs: $hs, hs_units: "khs", algo: "keryxhash", ver: $ver, uptime: ($uptime|tonumber), bus_numbers: $bus_numbers, temp: $temp, fan: $fan, ar: $ar }')
        khs=$total_hashrate
else
        khs=0
        stats="null"
fi

echo "Log file : $CUSTOM_LOG_BASENAME.log"
echo "Time since last log entry : $diffTime"
echo "Raw stats : $stats_raw"
echo "KHS : $khs"
echo "Output : $stats"

[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"
