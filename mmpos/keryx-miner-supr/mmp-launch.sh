#!/usr/bin/env bash
# mmpOS launcher for keryx-miner-supr. Maps the mmpos generic flags
# (--pool/--user/--password/--api-port) onto keryx's CLI and enables the
# built-in /mmpos stats endpoint. Bundled candle CUDA runtime libs live in ./lib.
cd "$(dirname "$(realpath "$0")")"

# candle needs cuBLAS/cuBLASLt/cuRAND at load; they are bundled in ./lib. The
# NVIDIA driver provides libcuda at runtime.
export LD_LIBRARY_PATH="$(pwd)/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXEC="./keryx-miner-supr"
CONF_FILE="mmp-external.conf"

ARGS=("$@")
FINAL_ARGS=()
POOL_VAL=""; USER_VAL=""; PASS_VAL=""; API_PORT=4067

i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    case "${ARGS[$i]}" in
        --pool)     POOL_VAL="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        --user)     USER_VAL="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        --password) PASS_VAL="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        --algo)     ((i+=2)) ;;                       # keryx is single-algo; ignore
        --api-port) API_PORT="${ARGS[$((i+1))]}"; ((i+=2)) ;;
        *)          FINAL_ARGS+=("${ARGS[$i]}"); ((i+=1)) ;;
    esac
done

[[ -z "$API_PORT" || "$API_PORT" == "0" ]] && API_PORT=4067

# Pool needs a stratum scheme, else keryx falls back to gRPC to a node.
[[ -n "$POOL_VAL" && "$POOL_VAL" != *://* ]] && POOL_VAL="stratum+tcp://$POOL_VAL"
# keryx mining addresses carry a 'keryx:' prefix (keryx:addr.worker).
[[ -n "$USER_VAL" && "$USER_VAL" != keryx:* ]] && USER_VAL="keryx:$USER_VAL"

# Persist the API port so mmp-stats.sh queries the right endpoint.
if grep -q '^CUSTOM_API_PORT=' "$CONF_FILE" 2>/dev/null; then
    sed -i "s/^CUSTOM_API_PORT=.*/CUSTOM_API_PORT=$API_PORT/" "$CONF_FILE"
else
    echo "CUSTOM_API_PORT=$API_PORT" >> "$CONF_FILE"
fi

CMD=( "$EXEC" )
[[ -n "$USER_VAL" ]] && CMD+=( -a "$USER_VAL" )
[[ -n "$POOL_VAL" ]] && CMD+=( -s "$POOL_VAL" )
# Map the flight-sheet password to the pool (NOT -p, which is keryx's --port number).
[[ -n "$PASS_VAL" && "$PASS_VAL" != "x" ]] && CMD+=( --pool-password "$PASS_VAL" )
CMD+=( --api-bind "127.0.0.1:$API_PORT" )
CMD+=( "${FINAL_ARGS[@]}" )

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
