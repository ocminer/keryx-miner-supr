#!/usr/bin/env bash
# mmpOS stats reporter for keryx-miner-supr.
#
# Called by the mmpos agent as: mmp-stats.sh <DEVICE_NUM> <LOG_FILE>
# Echoes ONE line of mmpOS custom-miner JSON:
#   {busid, hash, units, air, miner_name, miner_version}
#
# Strategy: prefer the miner's OWN /mmpos HTTP endpoint (mmp-launch.sh starts the
# miner with --api-bind 127.0.0.1:<port>) — that JSON is authoritative and
# format-independent (works for both the kHeavyHash pre-fork path and PoM). Fall
# back to log-parsing only if the endpoint is unreachable.

DEVICE_NUM="${1:-0}"
LOG_FILE="${2:-}"
NAME="keryx-miner-supr"

DIR="$(dirname "$(realpath "$0")")"
PORT=4067
[[ -f "$DIR/mmp-external.conf" ]] && PORT=$(grep -m1 '^CUSTOM_API_PORT=' "$DIR/mmp-external.conf" | cut -d= -f2)
[[ -z "$PORT" ]] && PORT=4067

# 1) Authoritative: the miner's built-in /mmpos endpoint.
json=$(curl -s --max-time 3 "http://127.0.0.1:$PORT/mmpos" 2>/dev/null)
if [[ "$json" == *'"hash"'* ]]; then
    echo "$json"
    exit 0
fi

# 2) Fallback: parse the miner log.
busid="[]"; hash="[]"; acc=0; rej=0; ver=""
if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
    tailbuf=$(tail -n 400 "$LOG_FILE" 2>/dev/null)
    ver=$(grep -oE 'Keryx-Miner GPU [0-9.]+' <<< "$tailbuf" | tail -n1 | grep -oE '[0-9.]+')

    read -r busid hash < <(awk '
        match($0, /Device #([0-9]+) .*: ([0-9.]+) ([GMk]?)hash\/s/, m) {
            idx=m[1]+0; val=m[2]+0; u=m[3];
            mult=1; if(u=="G")mult=1e9; else if(u=="M")mult=1e6; else if(u=="k")mult=1e3;
            dev[idx]=val*mult;
        }
        END {
            n=asorti(dev, idxs, "@ind_num_asc");
            nb="["; nh="[";
            for (i=1;i<=n;i++) { k=idxs[i]; sep=(i>1?",":""); nb=nb sep k; nh=nh sep dev[k] }
            print nb"] "nh"]";
        }' <<< "$tailbuf")
    [[ -z "$busid" ]] && busid="[]"
    [[ -z "$hash"  ]] && hash="[]"

    acc=$(grep -oE 'Accepted: [0-9]+' <<< "$tailbuf" | tail -n1 | grep -oE '[0-9]+')
    [[ -z "$acc" ]] && acc=$(grep -c 'Share accepted' <<< "$tailbuf")
    rej=$(grep -ciE 'reject|invalid share' <<< "$tailbuf")
    [[ -z "$acc" ]] && acc=0
    [[ -z "$rej" ]] && rej=0
fi

echo "{\"busid\":$busid,\"hash\":$hash,\"units\":\"hs\",\"air\":[\"$acc\",\"0\",\"$rej\"],\"miner_name\":\"$NAME\",\"miner_version\":\"${ver:-0.6.0}\"}"
