# keryx-miner-supr

Suprnova fork of the [keryx-labs/keryx-miner](https://github.com/keryx-labs/keryx-miner) GPU miner.

## What's different from upstream

| Area | Upstream | `-supr` |
|---|---|---|
| Dev tax | 2 % hardcoded, minimum-clamped | 0 % default, no clamp — pass `--devfund-percent N` explicitly to opt in |
| Devfund address | `keryx:qrxpcusy…najuhte` (Keryx Labs) | pointed at the operator's own pool wallet by default; pass `--devfund-percent 0` to skip entirely |
| NVIDIA Blackwell consumer (sm_120) | Loads sm_100 PTX → "unknown error" → falls back to sm_86 JIT (~50 % of native perf on RTX 5090) | Ships native `keryx-cuda-sm120.ptx` compiled with CUDA 13.0 nvcc (`-gencode=arch=compute_120,code=compute_120 --use_fast_math -Xptxas -O3`); `plugins/cuda/src/worker.rs` dispatches `major >= 12 → PTX_120`. Unrolled Keccak round loop → **3.28 GH/s** on RTX 5090 (see Performance) |
| Datacenter Ampere (sm_80 — A100 / CMP 170HX) | Falls through to sm_75 PTX | Ships native `keryx-cuda-sm80.ptx` with an arch-gated `__launch_bounds__(512, 2)` for 2 blocks/SM → **188 MH/s** on a CMP 170HX |
| CUDA toolkit support | `cudarc 0.13.9` rejects CUDA 13.x | Tracking newer `cudarc` to enable CUDA 13.0/13.2 native (in progress; see `Cargo.toml`) |
| Model weight hosting | IPFS gateway via `keryx-labs.com/ipfs/...` (intermittent 504s) | Same gateway by default, plus a configurable fallback URL the operator can host themselves (in progress) |

## Performance

Measured on the Suprnova rig (driver 580, CUDA 13.0, `--light` tier, **stock
power — no overclock, no power-limit tuning**). Verified by pool share
acceptance, 0 rejects.

| GPU | Hashrate | Bound by |
|---|---|---|
| RTX 5090 (sm_120) | **3.28 GH/s** | power (pinned at 575 W TDP cap) |
| CMP 170HX (sm_80) | **188 MH/s** | occupancy (1410 MHz, ~120 W of 250 W) |
| Both together | **3.46 GH/s** | clean sum — no PCIe/DMA contention |

The dominant win was **unrolling the Keccak-f1600 round loop**. Rolled, the
25-lane state stays an addressable local array, pinning the kernel at 229
registers → 1 block/SM (33 % occupancy) with no cross-round ILP. Unrolled, the
permutation becomes pure register renaming: 64 registers, 0 spill, 2 blocks/SM
— taking the 5090 from **2.57 → 3.28 GH/s (+28 %)**. sm_80 (170HX) lands at 72
registers from the unroll alone, so it gets an arch-gated
`__launch_bounds__(512, 2)` to force the same 64-reg / 2-blocks-per-SM layout
(**154 → 188 MH/s, +22 %**). The 170HX is occupancy-capped there (the ~50-reg
Keccak state forbids a 3rd block) and its undersized blower is the wall for
sustained runs — see `HANDOFF_OPTIMIZATION.md` for the thermal notes.

## Build

```bash
# CUDA 12.8 is currently the only supported toolkit (cudarc 0.13.9 won't accept 13.x yet).
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8 CUDA_COMPUTE_CAP=120

# Workspace build — this also produces libkeryxcuda.so + libkeryxopencl.so.
# Using `--bin keryx-miner-supr` would skip the plugins and the binary would
# refuse to start with "No workers specified".
cargo build --release
```

The binary lands at `target/release/keryx-miner-supr` (~26 MB). Copy alongside `target/release/libkeryxcuda.so` + `target/release/libkeryxopencl.so` into a single run directory.

## Run

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
  ./keryx-miner-supr \
    -a keryx:<your_mining_address>.<worker_name> \
    -s stratum+tcp://krx.suprnova.cc:4401 \
    --light --cuda-device 0
```

`-s` must be a full URL with the `stratum+tcp://` scheme — without it the miner silently picks gRPC. Port MUST be embedded in the URL; the standalone `-p` flag is ignored when a scheme is present.

Tier flags:
- `--light` — TinyLlama only (any GPU ≥ 6 GB).
- (default) TinyLlama + DeepSeek-R1-8B (RTX 3060 12 GB / 3070 / 3080).
- `--high` — + DeepSeek-R1-32B (RTX 3090 / 4090, 24 GB+).
- `--very-high` — + LLaMA-3.3-70B (RTX 5090, 32 GB+).

The miner blocks on model prefetch until every file in the chosen tier is local. Per-share OPoI `tag_fixed` MLP is baked into the binary; the LLM tier only affects optional AI-request task eligibility.

## Roadmap

1. Bump `candle` + `cudarc` to a CUDA 13.x-compatible pair (currently pinned to candle 0.8 / cudarc 0.13.9 by upstream).
2. Self-host the model weights with fallback to keryx-labs IPFS gateway.
3. ~~Squeeze the KeryxHash kernel on sm_120 — occupancy + register pressure.~~ **Done:** unrolling the Keccak round loop took the 5090 from 2.57 → 3.28 GH/s (229 → 64 regs, 1 → 2 blocks/SM). The card is now power-bound at the 575 W TDP cap, so further kernel work has to cut energy-per-hash; the matmul is only ~10 % of instructions and already rides the uniform datapath, so the remaining headroom is small.
4. Inline `tag_fixed` so it lives in the same launch as the heavy-hash kernel — saves one CPU↔GPU roundtrip per nonce window.
5. Tag a v0.4.0 release with a prebuilt static binary.

## Credit

This fork is derived from [keryx-labs/keryx-miner](https://github.com/keryx-labs/keryx-miner) v0.3.2 (commit `317fcab` "New release v0.3.2: SALT v4 + escrow/channel perf"). Original copyright belongs to the Keryx Labs team. License is dual MIT/Apache-2.0, same as upstream.
