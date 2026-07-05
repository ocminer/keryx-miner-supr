# keryx-miner-supr — GPU benchmarks (Proof-of-Model)

Reference hashrates, clocks and power for the keryx miner's **Proof-of-Model (PoM)** algorithm, measured
on our own hardware. **This file is open — please submit a PR** to add your card, correct a number, or fill
in a blank. One row per card; keep it sorted roughly by hashrate.

## How to read this (and how to tune)

PoM is a **memory-latency / bandwidth-bound** algorithm — a data-dependent random walk over the model
weights (a pointer-chase of 32-byte reads). What that means for you:

- **Memory bandwidth is king.** HBM cards (H100, A100, CMP 170HX, AMD Instinct MI50/MI60) punch far above
  their price for PoM; the walk is limited by random-access memory throughput, not compute. On AMD the HBM2
  MI50/MI60 beat the GDDR6 RX 7600 XT ~1.4× for the same reason.
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

## Per-card table — NVIDIA (keryx PoM, `--light`)

| GPU | Architecture | VRAM | Mem clock | Core clock | Power (mining) | Hashrate | Efficiency | Notes |
|-----|--------------|------|-----------|------------|----------------|----------|------------|-------|
| **NVIDIA H200 141GB** | Hopper (2023, datacenter) | 141 GB HBM3e | 3201 MHz | ~1980 MHz | ~410 W (of 700; cap-able) | **~170 MH/s** (bench ceiling; v0.6.8 128-bit loads) | ~0.41 MH/W | **fastest PoM card** — HBM3e ~4.8 TB/s, +35 % over H100. Memory-bound → cap power, no core-clock gain |
| **NVIDIA H100 80GB** | Hopper (2022, datacenter) | 80 GB HBM3 | 2619 MHz (fixed) | ~1650 MHz | **400 W** (cap; see below) | **~123 MH/s** (ceiling 125.6) | 0.31 MH/W | ~2× a 5090. Cap PL to 400 W = 0 loss, −30 % power |
| NVIDIA RTX 5090 | Blackwell (2025) | 32 GB GDDR7 | 13801 MHz | up to 3090 MHz | ~385 W (of 600 W PL) | ~65–68 MH/s | 0.18 MH/W | memory-bound; core-clock-insensitive, no gain from OC |
| NVIDIA RTX 5080 | Blackwell (2025) | 16 GB GDDR7 | — | — | ~360 W PL (self-limits) | ~34.6 MH/s | — | |
| NVIDIA RTX 5070 Ti | Blackwell (2025) | 16 GB GDDR7 | — | — | ~285 W PL (self-limits) | ~33.7 MH/s | — | |
| NVIDIA CMP 170HX | Ampere GA100 (2021, mining) | 8 GB HBM2 | 1458 MHz | ~465 MHz (auto) | ~109 W | ~19.5 MH/s | ~0.18 MH/W | HBM2 → strong for PoM; auto-parks low, already efficient |
| NVIDIA RTX 3070 | Ampere (2020) | 8 GB GDDR6 | — | — | ~220 W cap | ~18.35 MH/s | — | |

_"—" = not measured yet; PRs welcome. Efficiency = MH/s per watt (higher is better)._

## Per-card table — AMD (keryx PoM, `--light`)

AMD cards mine PoM via the OpenCL worker (`libkeryxopencl.so` / `keryxopencl.dll`). Clocks are as reported by
`rocm-smi` — note the **mem clock is the actual HBM/GDDR clock, not the effective data rate**, so it is not
directly comparable to the nvidia-smi numbers above (e.g. this 1000 MHz HBM2 ≈ a very wide, high-bandwidth bus).

| GPU | Architecture | VRAM | Mem clock | Core clock | Power (mining) | Hashrate | Efficiency | Notes |
|-----|--------------|------|-----------|------------|----------------|----------|------------|-------|
| **AMD Instinct MI50 / MI60** | Vega 20 (2018–19, datacenter) | 16 GB HBM2 | ~1000 MHz | ~1725 MHz | ~135 W | **~10.9 MH/s** | ~0.08 MH/W | HBM2 (~1 TB/s) → best AMD PoM card; passively cooled (needs chassis airflow), thermally sensitive — a hot card throttles core → hashrate |
| AMD Radeon RX 7600 XT | RDNA 3, Navi 33 (2024) | 16 GB GDDR6 | ~1124 MHz | ~2771 MHz | ~166 W (at 165 W cap) | ~7.76 MH/s | ~0.047 MH/W | GDDR6 (~288 GB/s); RDNA3's high clocks + Infinity Cache offset some of the bandwidth gap |

_Measured on a 3-GPU box (1× RX 7600 XT + 2× MI50/MI60): **~29.5 MH/s aggregate**, each card mining + submitting
its own shares independently (per-GPU PoM residency). "—" = not measured yet; PRs welcome._

**AMD notes.** The HBM2 MI50/MI60 beat the GDDR6 RX 7600 XT by **~1.4×** — exactly what the memory-bound
principle predicts. It's ~1.4× (not the ~3.5× raw-bandwidth ratio) because the walk is **latency-bound on
data-dependent reads** plus per-step hash compute, and RDNA3's high clocks + Infinity Cache narrow the gap.
So on AMD the Instinct HBM2 cards are the strongest PoM silicon. OPoI inference on AMD runs on the GPU via a
bundled **llama.cpp Vulkan** server (RADV/Mesa; ~68 tok/s Gemma-3-4B on the RX 7600 XT), auto-falling back to
CPU (candle, ~2.7 tok/s) if no Vulkan ICD is present — inference load does not reduce the PoM hashrate.

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
self-limit for the memory-bound walk). **AMD** numbers are the OpenCL worker (`keryxopencl`) measured live
on the same pool, with per-card hashrate/clocks/power read from `rocm-smi`, on a 3-GPU Ubuntu box (RADV/Mesa
Vulkan for OPoI inference).
