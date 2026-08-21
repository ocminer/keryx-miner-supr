#!/bin/bash
# keryx-miner-supr entrypoint — env-driven, zero-intervention keryx PoM v4 mining.
#
# Env contract (all optional except KERYX_WALLET; sane defaults):
#   KERYX_WALLET     keryx:... payout address (REQUIRED). Falls back to WALLET.
#   KERYX_WORKER     worker name, appended as WALLET.WORKER (default: hostname)
#   KERYX_POOL       stratum URL       (default: stratum+tcp://krx.suprnova.cc:4404)
#   KERYX_BACKUP     backup stratum URL (default: stratum+tcp://krx.suprnova.cc:4401)
#   KERYX_TIER       auto|very-light|light|default|high|very-high  (default: auto)
#   KERYX_MODEL_DIR  model directory    (default: /models — mount a volume or NFS here;
#                    absent models are fetched from the Keryx IPFS gateway on first run)
#   KERYX_DEVICES    CUDA device list "-> --cuda-device"  (default: all)
#   KERYX_INFERENCE_GPU  pin OPoI inference to this CUDA ordinal (default: biggest-VRAM card)
#   KERYX_EXTRA_ARGS  appended verbatim to the miner command line
#   ROOT_PASSWORD    ssh root password  (default: keryx)
set -u
cd "$(dirname "$(realpath "$0")")"
export LD_LIBRARY_PATH="$PWD:$PWD/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# SSH for ops/debug (matches the octa suprminer-base contract).
echo "root:${ROOT_PASSWORD:-keryx}" | chpasswd 2>/dev/null || true
mkdir -p /run/sshd && /usr/sbin/sshd 2>/dev/null || true
echo "[entrypoint] SSH on :22 (user root)"

WALLET="${KERYX_WALLET:-${WALLET:-}}"
if [ -z "$WALLET" ]; then
    echo "[entrypoint] ERROR: set KERYX_WALLET to your keryx:... payout address. Idling for SSH."
    sleep infinity
fi
WORKER="${KERYX_WORKER:-$(hostname)}"
[ -n "$WORKER" ] && WALLET="${WALLET}.${WORKER}"
POOL="${KERYX_POOL:-stratum+tcp://krx.suprnova.cc:4404}"
MODEL_DIR="${KERYX_MODEL_DIR:-/models}"
mkdir -p "$MODEL_DIR"

CMD=(./keryx-miner-supr -a "$WALLET" -s "$POOL" --tier "${KERYX_TIER:-auto}" --model-dir "$MODEL_DIR")
[ -n "${KERYX_BACKUP:-stratum+tcp://krx.suprnova.cc:4401}" ] && CMD+=(--backup-pool "${KERYX_BACKUP:-stratum+tcp://krx.suprnova.cc:4401}")
[ -n "${KERYX_DEVICES:-}" ] && CMD+=(--cuda-device "$KERYX_DEVICES")
[ -n "${KERYX_INFERENCE_GPU:-}" ] && export KERYX_INFERENCE_GPU
# shellcheck disable=SC2206
[ -n "${KERYX_EXTRA_ARGS:-}" ] && CMD+=($KERYX_EXTRA_ARGS)

# docker stop (SIGTERM) → forward as SIGINT (keryx clean shutdown; NEVER -9).
MINER_PID=""; STOPPING=0
forward() { STOPPING=1; [ -n "$MINER_PID" ] && kill -INT "$MINER_PID" 2>/dev/null; }
trap forward TERM INT

echo "[entrypoint] launching: ${CMD[*]}"
while true; do
    "${CMD[@]}" & MINER_PID=$!
    wait "$MINER_PID"; RC=$?; MINER_PID=""
    [ "$STOPPING" = 1 ] && { echo "[entrypoint] stopped (rc=$RC)"; exit 0; }
    echo "[entrypoint] miner exited rc=$RC — restarting in 10s (docker stop to quit)"
    sleep 10; [ "$STOPPING" = 1 ] && exit 0
done
