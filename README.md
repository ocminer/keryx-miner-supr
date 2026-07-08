# keryx-miner-supr

Suprnova fork of the [keryx-labs/keryx-miner](https://github.com/keryx-labs/keryx-miner) GPU miner.

## What's different from upstream

| Area | Upstream | `-supr` |
|---|---|---|
| NVIDIA Blackwell consumer (sm_120) | Loads sm_100 PTX → "unknown error" → falls back to sm_86 JIT (~50 % of native perf on RTX 5090) | Ships native `keryx-cuda-sm120.ptx` compiled with CUDA 13.0 nvcc (`-gencode=arch=compute_120,code=compute_120 --use_fast_math -Xptxas -O3`); `plugins/cuda/src/worker.rs` dispatches `major >= 12 → PTX_120` |
| Datacenter Ampere (sm_80 — A100 / CMP 170HX) | Falls through to sm_75 PTX | Ships native `keryx-cuda-sm80.ptx` with an arch-gated `__launch_bounds__(512, 2)` for 2 blocks/SM |
| CUDA toolkit (PoW miner) | 12.x | Builds against **CUDA 13.0**. The PoW runtime is `cust 0.3` (binds to the driver, not the toolkit) — `cudarc` is not in the miner's dependency tree, so there's no 13.x pin to clear. Avoid 13.2: driver 580 caps PTX at ISA 9.0. |
| Model weight hosting | IPFS gateway via `keryx-labs.com/ipfs/...` (intermittent 504s) | Same gateway by default, plus a configurable fallback URL the operator can host themselves (in progress) |

## Performance

Since the PoM hardfork the PoW is **Proof-of-Model** — a memory-bound random
walk over the model weights — so hashrates are in **MH/s** and scale with a
card's memory bandwidth, not its compute. (The pre-fork kHeavyHash GH/s numbers
that used to live here are obsolete: kHeavyHash is no longer mined.)

Measured live on the pool (v0.6.9.3, AUTO tier — each card loads the heaviest
model that fits — **stock clocks and power limits**). Verified by pool share
acceptance, 0 rejects. Re-benchmarked 2026-07-08:

| GPU | Tier (AUTO) | Hashrate | Power |
|---|---|---|---|
| RTX 5090 | very-high (Llama-3.3-70B-Q2) | **~68 MH/s** | ~382 W of 600 W PL |
| RTX 5080 | default (Dolphin-Llama3-8B) | **~34.8 MH/s** | ~190–200 W |
| RTX 5070 Ti | default (Dolphin-Llama3-8B) | **~34 MH/s** | ~179 W |
| CMP 170HX | light (Gemma-3-4B) | **~20 MH/s** | ~115 W |
| RTX 3070 | light (Gemma-3-4B) | **~18.5 MH/s** | ~173 W |

Full per-card data (H100/H200, AMD, power sweeps) and tuning guidance:
[BENCHMARKS.md](BENCHMARKS.md) — PRs with your own cards welcome.

## Build

```bash
# ⚠️ TOOLKIT CHOICE = which GPUs your build can run on. All PoM/inference PTX
# is compiled at build time and JIT'd by the driver at runtime; PTX only JITs
# FORWARD — a sm_75 PTX never loads on a sm_70 card (CUDA_ERROR_INVALID_PTX).
#   - CUDA 13.x CANNOT compile for Volta or Pascal (compute_50–72 removed).
#     A CUDA 13 build is Turing (sm_75)+ only. For a Tesla V100 / CMP 100-210
#     (sm_70) build you MUST use a CUDA 12.x toolkit (12.4–12.9) — or just run
#     the prebuilt `legacy` release tarball (all its PTX targets sm_70).
#   - Do NOT use CUDA 13.2: driver 580 caps PTX at ISA 9.0, and 13.2 emits 9.2
#     PTX that fails to load at runtime with "unknown error".
export PATH=/usr/local/cuda-13.0/bin:$PATH   # Turing+ build; cuda-12.x for Volta
# CUDA_COMPUTE_CAP sets the arch for candle's OPoI inference PTX (bindgen_cuda
# emits a single `.target sm_NN`, no fatbin). POM_CUDA_ARCH sets the PoM walk
# kernel's arch (default compute_75). Set BOTH to your OLDEST card:
#   Turing+ fleet (CUDA 13.0): CUDA_COMPUTE_CAP=75              (walk default 75)
#   incl. Volta   (CUDA 12.x): CUDA_COMPUTE_CAP=70 POM_CUDA_ARCH=compute_70
#   Pascal        (CUDA 12.4): CUDA_COMPUTE_CAP=60 POM_CUDA_ARCH=compute_60
#     (Pascal can't run candle GPU inference — pair with --cpu-inference.)
export CUDA_HOME=/usr/local/cuda-13.0 CUDA_PATH=/usr/local/cuda-13.0 CUDA_COMPUTE_CAP=75

# Workspace build — this also produces libkeryxcuda.so + libkeryxopencl.so.
# Using `--bin keryx-miner-supr` would skip the plugins and the binary would
# refuse to start with "No workers specified".
cargo build --release
```

The binary lands at `target/release/keryx-miner-supr` (~26 MB). Copy alongside `target/release/libkeryxcuda.so` + `target/release/libkeryxopencl.so` into a single run directory.

Or skip building entirely — the release page ships three prebuilt lines: **modern** (sm_75+, native
sm_120 for RTX 50xx; driver 575+), **legacy** (sm_70+ — Tesla V100/Volta, CMP 100-210, Turing and
newer; driver 550+), **pascal** (sm_60/61 — GTX 10-series; driver 550+, runs `--cpu-inference`).

## Run

```bash
# No tier flag = AUTO: every GPU loads the heaviest model its VRAM can hold.
LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  ./keryx-miner-supr \
    -a keryx:<your_mining_address>.<worker_name> \
    -s stratum+tcp://krx.suprnova.cc:4401
```

Add `--cuda-device N` to run a single GPU, a tier flag (`--very-high`, …) to pin all cards to one tier, or `--force-model` to set a model per card (see [Model tiers](#model-tiers) below).

`-s` must be a full URL with the `stratum+tcp://` scheme — without it the miner silently picks gRPC. Port MUST be embedded in the URL; the standalone `-p` flag is ignored when a scheme is present.

### Model tiers

Heavier tiers earn more (higher tier-reward + higher OPoI inference-reward floor), so each card should
run the heaviest model its VRAM can hold. The tiers (lightest → heaviest):

| Tier flag      | Model            | GPU VRAM |
|----------------|------------------|----------|
| `--very-light` | Qwen3-1.7B       | ≥ 4 GB   |
| `--light`      | Gemma-3-4B       | ≥ 6 GB   |
| *(default)*    | Dolphin-Llama3-8B| ≥ 8 GB   |
| `--high`       | Qwen3-32B        | ≥ 24 GB  |
| `--very-high`  | Llama-3.3-70B-Q2 | ≥ 32 GB  |

You can also pin one with `--tier <auto|very-light|light|default|high|very-high>`.

### Automatic per-card selection (default — recommended)

With **no tier flag**, the miner runs in **AUTO** mode: it queries each GPU's VRAM independently and loads the
**heaviest tier that fits that card**. In a mixed rig every card gets its own best-fit model — e.g. on a box
with a 5090 + two 3070s, the 5090 loads Llama-70B while each 3070 loads Gemma, all in one process. A single
tier flag (`--very-high`, `--tier high`, …) instead pins **every** card to that one tier.

### Forcing a model per card — `--force-model`

To override AUTO and assign specific models to specific cards, pass a **comma-separated list mapped to CUDA
device order**:

```bash
# GPU0 → very-high, GPU1 → light, GPU2 → default, GPU3 → high
--force-model very-high,light,default,high

# force every card to Gemma
--force-model light,light,light,light

# pin only GPU0; every other card falls back to AUTO best-fit
--force-model very-high
```

Names are the tier names without the `--` (`very-light`, `light`, `default`, `high`, `very-high`).
`--force-model`:
- is **positional** — entry *N* applies to CUDA device *N*;
- **auto-fills** any cards beyond the list with normal per-card AUTO best-fit;
- **overrides** `--tier` / `--light` / `--very-high` / AUTO; and
- **bypasses the VRAM-fit and download-availability checks** — it's a power-user knob, so forcing a model
  larger than a card's VRAM will OOM that card.

The miner blocks on model prefetch until every file in the chosen tier is local. Per-share OPoI `tag_fixed` MLP is baked into the binary; the LLM tier only affects optional AI-request task eligibility.

## Roadmap

1. Verify the `candle` LLM-inference path on CUDA 13.0 (the PoW miner already builds and runs against 13.0 via `cust 0.3`; only the optional inference backend still rides upstream's candle 0.8 pin).
2. Self-host the model weights with fallback to keryx-labs IPFS gateway.
3. ~~Squeeze the KeryxHash kernel on sm_120 — occupancy + register pressure.~~ **Done, then obsoleted:** the Keccak round-loop unroll (229 → 64 regs, 1 → 2 blocks/SM) shipped pre-fork; the PoM hardfork replaced kHeavyHash, and the PoM walk is memory-bound — see [BENCHMARKS.md](BENCHMARKS.md).
4. Inline `tag_fixed` so it lives in the same launch as the heavy-hash kernel — saves one CPU↔GPU roundtrip per nonce window.
5. Tag a v0.4.0 release with a prebuilt static binary.

## Credit

This fork is derived from [keryx-labs/keryx-miner](https://github.com/keryx-labs/keryx-miner) v0.3.2 (commit `317fcab` "New release v0.3.2: SALT v4 + escrow/channel perf"). Original copyright belongs to the Keryx Labs team. License is dual MIT/Apache-2.0, same as upstream.
