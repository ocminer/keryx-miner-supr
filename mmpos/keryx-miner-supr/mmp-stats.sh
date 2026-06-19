#!/usr/bin/env bash
# mmpOS stats reporter for keryx-miner-supr.
#
# The mmpos agent calls this with two args:
#   $1 = DEVICE_NUM  (number of devices the agent wants stats for)
#   $2 = LOG_FILE    (miner log path)
# It must echo ONE line of mmpOS custom-miner JSON:
#   {busid, hash, units, air, miner_name, miner_version}
#
# This parses the miner log (no miner change needed). Alternatively, run the
# miner with `--api-bind 127.0.0.1:4067` and point mmpos at the built-in HTTP
# endpoint `http://127.0.0.1:4067/mmpos` (same JSON) — then this script is
# unnecessary.

DEVICE_NUM="${1:-0}"
LOG_FILE="${2:-}"
NAME="keryx-miner-supr"

busid="[]"; hash="[]"; acc=0; rej=0; ver=""

if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
  tailbuf=$(tail -n 400 "$LOG_FILE" 2>/dev/null)

  # miner version, e.g. "Keryx-Miner GPU 0.5.1"
  ver=$(grep -oE 'Keryx-Miner GPU [0-9.]+' <<< "$tailbuf" | tail -n1 | grep -oE '[0-9.]+')

  # per-device hashrate (-> hashes/sec) from
  #   "Device #N (NVIDIA GeForce RTX 5090): 3.28 Ghash/s"
  # busid = CUDA device index N (stable ordinal; mmpos maps hash[] by order).
  read -r busid hash < <(awk '
    match($0, /Device #([0-9]+) .*: ([0-9.]+) ([GMk]?)hash\/s/, m) {
      idx=m[1]+0; val=m[2]+0; u=m[3];
      mult=1; if(u=="G")mult=1e9; else if(u=="M")mult=1e6; else if(u=="k")mult=1e3;
      dev[idx]=val*mult;          # keep latest per device
    }
    END {
      n=asorti(dev, idxs, "@ind_num_asc");
      nb="["; nh="[";
      for (i=1;i<=n;i++) { k=idxs[i]; sep=(i>1?",":""); nb=nb sep k; nh=nh sep dev[k] }
      print nb"] "nh"]";
    }' <<< "$tailbuf")
  [[ -z "$busid" ]] && busid="[]"
  [[ -z "$hash"  ]] && hash="[]"

  # accepted / rejected shares
  acc=$(grep -oE 'Accepted: [0-9]+' <<< "$tailbuf" | tail -n1 | grep -oE '[0-9]+')
  [[ -z "$acc" ]] && acc=$(grep -c 'Share accepted' <<< "$tailbuf")
  rej=$(grep -ciE 'reject|invalid share' <<< "$tailbuf")
  [[ -z "$acc" ]] && acc=0
  [[ -z "$rej" ]] && rej=0
fi

# mmpOS JSON: air = [accepted, invalid, rejected] (strings)
echo "{\"busid\":$busid,\"hash\":$hash,\"units\":\"hs\",\"air\":[\"$acc\",\"0\",\"$rej\"],\"miner_name\":\"$NAME\",\"miner_version\":\"${ver:-0.5.1}\"}"
