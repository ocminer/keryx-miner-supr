# keryx-miner-supr — GPU benchmarks (Proof-of-Model)

Reference hashrates, clocks and power for the keryx miner's **Proof-of-Model (PoM)** algorithm, measured
on our own hardware. **This file is open — please submit a PR** to add your card, correct a number, or fill
in a blank. One row per card; keep it sorted roughly by hashrate.

## How to read this (and how to tune)

PoM is a **memory-latency / bandwidth-bound** algorithm — a data-dependent random walk over the model
weights (a pointer-chase of 32-byte reads). What that means for you:

- **Memory bandwidth is king.** HBM cards (H100, A100, CMP 170HX) punch far above their price for PoM;
  the walk is limited by random-access memory throughput, not compute.
- **Core clock and power barely matter.** The walk hides all compute behind memory latency, so a high
  core clock is wasted. On most cards the GPU **auto-downclocks** and already sits well under its power
  limit — you generally can't save much by capping it further, and you lose nothing by not overclocking.
- **The exception is very-high-TDP datacenter cards.** On an H100, the 700 W ceiling lets the OPoI
  inference bursts run wild; **capping the power limit to ~400 W keeps full hashrate and saves ~30% power**
  (see the H100 sweep below). Consumer cards already self-limit, so there's little to do.
- **VRAM sets the tier.** `--light` (Gemma-3-4B, ~3 GB) runs on any 6 GB+ card. `--high` (Qwen3-32B) needs
  24 GB; `--very-high` (Llama-70B) needs 32 GB+. Heavier tiers pay a higher block-reward bracket at
  ~the same walk hashrate — see the miner's `--help`.

Numbers below are the **`--light` (Gemma-3-4B)** tier unless noted.

## Per-card table (keryx PoM, `--light`)

| GPU | Architecture | VRAM | Mem clock | Core clock | Power (mining) | Hashrate | Efficiency | Notes |
|-----|--------------|------|-----------|------------|----------------|----------|------------|-------|
| **NVIDIA H100 80GB** | Hopper (2022, datacenter) | 80 GB HBM3 | 2619 MHz (fixed) | ~1650 MHz | **400 W** (cap; see below) | **~123 MH/s** (ceiling 125.6) | 0.31 MH/W | best PoM card; ~2× a 5090. Cap PL to 400 W = 0 loss, −30 % power |
| NVIDIA RTX 5090 | Blackwell (2025) | 32 GB GDDR7 | 13801 MHz | up to 3090 MHz | ~385 W (of 600 W PL) | ~65–68 MH/s | 0.18 MH/W | memory-bound; core-clock-insensitive, no gain from OC |
| NVIDIA RTX 5080 | Blackwell (2025) | 16 GB GDDR7 | — | — | ~360 W PL (self-limits) | ~34.6 MH/s | — | |
| NVIDIA RTX 5070 Ti | Blackwell (2025) | 16 GB GDDR7 | — | — | ~285 W PL (self-limits) | ~33.7 MH/s | — | |
| NVIDIA CMP 170HX | Ampere GA100 (2021, mining) | 8 GB HBM2 | 1458 MHz | ~465 MHz (auto) | ~109 W | ~19.5 MH/s | ~0.18 MH/W | HBM2 → strong for PoM; auto-parks low, already efficient |
| NVIDIA RTX 3070 | Ampere (2020) | 8 GB GDDR6 | — | — | ~220 W cap | ~18.35 MH/s | — | |

_"—" = not measured yet; PRs welcome. Efficiency = MH/s per watt (higher is better)._

## H100 power/hashrate sweep (why 400 W is the sweet spot)

Isolated walk vs. the live miner (walk + OPoI inference). The walk itself only draws ~287 W; the extra
draw at 700 W is inference bursts, which the power cap trims with negligible hashrate loss:

| Power limit | Live hashrate | Draw | Verdict |
|-------------|---------------|------|---------|
| 700 W | 123.3 MH/s | 575 W | default — inference runs wide open |
| 500 W | 123.7 MH/s | 500 W | full |
| 450 W | 123.7 MH/s | 450 W | full |
| **400 W** | **123.3 MH/s** | **400 W** | **sweet spot — full hashrate, −30 % power** |
| 375 W | 119.1 MH/s (−3 %) | 375 W | starting to starve the core |
| 350 W | 111.1 MH/s (−10 %) | 349 W | too aggressive |

`nvidia-smi -pl 400` (per GPU) on the 8× H100 box: same ~986 MH/s aggregate for ~1.2 kW less.

## Method

Measured live on the pool (`krx.suprnova.cc`) with `--light` (Gemma-3-4B, 77.6 M chunks / 2.48 GB
possession blob), plus an isolated CUDA microbench of the walk kernel (`tools/h100/bench_pom.cu`) for the
clean per-power-limit numbers. HBM cards were run at stock; consumer cards at stock power limits (they
self-limit for the memory-bound walk).
