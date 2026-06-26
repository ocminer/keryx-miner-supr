#!/usr/bin/env bash
# HiveOS stats reporter for keryx-miner-supr.
# Reads the agent's $GPU_STATS_JSON for per-GPU busids/brand/temp/fan, brand-filters
# to the miner's GPUs, and emits aligned hs[]/temp[]/fan[]/bus_numbers[] arrays.
# Hashrate is parsed from the miner log.
#
# Ported from upstream keryx-miner integration pkg 0.3.31 (commits c6f7948/599fb99/9b568a0):
#   * env_logger ISO timestamp parser (the miner logs "[2026-06-26T06:31:08Z INFO ...]")
#   * khs scaling fixed (rate*1000 base; Ghash *1e3, NOT *1e6 — the old code was 1000x too high)
#   * octal guard (sub-1.0 rates yield a leading zero -> bash rejects as octal)
#   * iGPU off-by-one (separate miner_dev counter that advances only for mining-brand cards)
# Our log lines carry the GPU name: "... [INFO ] Device #0 (NVIDIA GeForce RTX 5090): 3.28 Ghash/s"
# so the per-device grep matches "Device #N " (space) rather than "Device #N:".

. /hive/miners/custom/keryx-miner-supr/h-manifest.conf

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

if [ "$diffTime" -lt "$maxDelay" ]; then
        # Value is second-to-last field (before unit), unit is last field. The miner logs the rate
        # with 2 decimals; dropping the dot then appending one 0 yields rate*1000 (3.48 -> "3480").
        # Use `tr -d '.'` (not `cut --output-delimiter=''`, which emits a NUL byte).
        total_hashrate=`echo $stats_raw | awk 'NF>=2{print $(NF-1)}' | tr -d '.' | sed 's/$/0/'`
        # Force base 10: a sub-1.0 rate yields a leading zero (0.48 -> "0480") that bash parses as octal.
        total_hashrate=$((10#${total_hashrate:-0}))
        # HiveOS expects khs. base = rate*1000, so: Mhash needs nothing, Ghash *1e3, Thash *1e6.
        if [[ $stats_raw == *"Thash"* ]]; then
                total_hashrate=$(($total_hashrate*1000000))
        elif [[ $stats_raw == *"Ghash"* ]]; then
                total_hashrate=$(($total_hashrate*1000))
        elif [[ $stats_raw == *"Mhash"* ]]; then
                : # Mhash/s = rate*1e3 khs = rate*1000 already, no multiplier needed
        fi

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

        if [ $(gpu-detect NVIDIA) -gt 0 ]; then
                BRAND_MINER="nvidia"
        elif [ $(gpu-detect AMD) -gt 0 ]; then
                BRAND_MINER="amd"
        fi

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
                # Per-device line: "... [INFO ] Device #N (<name>): 5.23 Ghash/s" — match "#N "
                # (space) so "#1 " never matches "#10 ".
                gpu_raw=`grep "Device #$miner_dev " <<< "$log" | tail -n 1`
                if [[ -n "$gpu_raw" ]]; then
                        hashrate=`echo $gpu_raw | awk 'NF>=2{print $(NF-1)}' | tr -d '.' | sed 's/$/0/'`
                        hashrate=$((10#${hashrate:-0}))
                        if [[ $gpu_raw == *"Thash"* ]]; then
                                hashrate=$(($hashrate*1000000))
                        elif [[ $gpu_raw == *"Ghash"* ]]; then
                                hashrate=$(($hashrate*1000))
                        elif [[ $gpu_raw == *"Mhash"* ]]; then
                                : # Mhash/s = rate*1000 already
                        fi
                else
                        hashrate=0
                fi
                [[ -z "$hashrate" ]] && hashrate=0
                hash_arr+=($hashrate)
                miner_dev=$((miner_dev+1))
        done

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
                '{ hs: $hs, hs_units: "khs", algo: "keryxhash", ver: $ver, $uptime, $bus_numbers, $temp, $fan }')
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
