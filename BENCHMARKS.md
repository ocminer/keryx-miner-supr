# keryx-miner-supr — GPU benchmarks (Proof-of-Model)

## v4 relaunch walk — measured 2026-08-25 on v0.11.13

These are the numbers for the **current** PoM (the `pom_mine_v4` re-walk introduced with the keryxd
v1.5.1 relaunch). They are NOT comparable to the pre-relaunch figures further down: that was a
different walk over a different table, and its hashrates were an order of magnitude higher.

Method — deliberately strict, because earlier numbers on this rig were quietly wrong:

- **One card at a time.** Every other miner on the box was stopped and verified stopped
  (`nvidia-smi --query-compute-apps` empty) before each run. Two cards benchmarking at once contend
  for the host and skew both.
- Each card ran ~7 minutes: model staging, then the v0.11.13 autotune, then steady state. Hashrate,
  power and temperature are taken from the miner's own per-device line at steady state.
- **Pool-verified**: every row accepted shares with **0 rejects and 0 GPU faults**. A hashrate that
  does not produce shares is not a hashrate.
- Stock clocks and power limits except where noted.

| GPU | Arch | SMs | VRAM | Tier (AUTO) | Tuned batch | Hashrate | Power | Efficiency |
|---|---|---|---|---|---|---|---|---|
| RTX 5090 | Blackwell | 170 | 32 GB | Kimi-Linear-48B | 130,560 | **5.47 MH/s** | 601 W | 9.1 kH/s/W |
| RTX 5090 (2nd card) | Blackwell | 170 | 32 GB | Kimi-Linear-48B | 130,560 | 5.29 MH/s | 575 W | 9.2 kH/s/W |
| RTX 5080 | Blackwell | 84 | 16 GB | Gemma-4-12B | 64,512 | **3.00 MH/s** | 376 W | 8.0 kH/s/W |
| RTX 5080 (2nd card) | Blackwell | 84 | 16 GB | Gemma-4-12B | 64,512 | 2.99 MH/s | 400 W | 7.5 kH/s/W |
| RTX 5070 Ti | Blackwell | 70 | 16 GB | Gemma-4-12B | 53,760 | **2.69 MH/s** | 341 W | 7.9 kH/s/W |
| RTX 4070 Ti SUPER | Ada | 66 | 16 GB | Gemma-4-12B | 50,688 | **2.15 MH/s** | 284 W | 7.6 kH/s/W |
| CMP 170HX | Ampere | 70 | 64 GB | Kimi-Linear-48B | 53,760 | 1.76 MH/s | 213 W | 8.3 kH/s/W |
| RTX 3070 | Ampere | 46 | 8 GB | Qwen3.5-9B | 17,664 | **1.35 MH/s** | 249 W | 5.4 kH/s/W |

Caveats, so the numbers are not read for more than they are:

- **The tier matters.** The walk is over the model, so a card on Kimi-Linear-48B is not doing the same
  work as one on Gemma-4-12B. Compare cards within a tier, not across.
- **Two rows are thermally limited, not compute limited.** The CMP 170HX sat at 85 °C and clocked down
  to 1200 MHz; its own tuner measured **1.95 MH/s** on the same card before it heat-soaked. The second
  RTX 5090 ran at 79 °C against the first card's 69 °C — same batch, ~3% less hashrate, and the better
  efficiency of the two because it drew less power to do it.
- The two RTX 3070s carry a **+250 MHz core offset**; everything else is stock.
- Efficiency is steady-state hashrate over the miner's reported board power. The 3070 is the outlier:
  8 GB forces it onto the smallest tier, and it burns ~249 W doing it.

### What the autotune picked, and why it matters

v0.11.13 benchmarks each card on its first grind and caches the result in `~/.keryx/v4tune.json`.
The batch column above is measured, not derived — and it is the reason the table exists:

| GPU | batch/2 | SM-derived (384/SM) | batch×2 | picked |
|---|---|---|---|---|
| RTX 5090 | 4.95 | 5.32 | **5.47** | 130,560 |
| RTX 5080 | 2.74 | 2.90 | **3.00** | 64,512 |
| RTX 5070 Ti | 2.57 | 2.69 | **2.70** | 53,760 |
| RTX 4070 Ti SUPER | 2.09 | 2.12 | **2.15** | 50,688 |
| CMP 170HX | 1.84 | 1.92 | **1.95** | 53,760 |
| RTX 3070 | 1.30 | **1.35** | 1.32 | 17,664 |

Every card from ~66 SMs up wants **more** than 384 nonces/SM; only the 46-SM RTX 3070 peaks at the
SM-derived value and loses throughput above it. That is why the batch is measured per card instead of
being a formula — a fixed 64K would starve the 3070, and 384/SM leaves ~4% on the table on an RTX 5080.

Also measured: the **tensor-core walk is the right choice on every card tested**, not just Blackwell —
an RTX 3070 does 1.35 MH/s on it against 0.84 MH/s on the classic dp4a walk (+61%).

---

## RTX 5090 core-clock and memory-clock efficiency sweep — 2026-10-01, rig11r GPU1 (v0.13.3, PoM v4, Qwen3.5-9B tier)

Question from the operator: is there an underclock "sweet spot" where the card mines more efficiently and cooler?
Method: one card (rig11r GPU1, the thermally limited second RTX 5090, fan 95–100 % at stock), production miner
left running and submitting to the pool, power cap untouched at 575 W (the fleet guard asserts it). Each step:
set `nvidia-smi -lgc 0,<MHz>` (core) or `-lmc 0,<MHz>` (memory), 60 s settle, then 10 samples 15 s apart of the
miner's own per-device hashrate (`--api-bind` JSON) plus `nvidia-smi` power/temperature/clocks. Locks released
afterwards. Raw CSVs: `rig11r:/home/marcel/gpu1-sweep.csv`, `gpu1-memsweep.csv`.

| setting | MH/s | W | kH/s/W | °C | SM MHz | mem MHz |
|---|---:|---:|---:|---:|---:|---:|
| stock, unlocked (self-throttles at 575 W / 85 °C) | **5.27** | 572 | **9.21** | 83 | 1689 | 13801 |
| core locked 1600 | 4.08 | 464 | 8.78 | 80 | 1590 | 13801 |
| core locked 1400 | 3.58 | 410 | 8.72 | 79 | 1387 | 13801 |
| core locked 1250 | 3.20 | 372 | 8.60 | 78 | 1237 | 13801 |
| core locked 1100 | 2.82 | 336 | 8.41 | 75 | 1087 | 13801 |
| core locked 950 | 2.54 | 300 | 8.48 | 72 | 937 | 13801 |
| memory locked 12000 → driver snaps to 7001 | 3.16 | 416 | 7.60 | 81 | 2476 | 7001 |
| memory locked 10000 → 7001 | 3.17 | 417 | 7.60 | 82 | 2483 | 7001 |
| memory locked 8000 → 7001 | 3.17 | 417 | 7.59 | 82 | 2483 | 7001 |
| memory locked 6000 → 810 | 0.35 | 157 | 2.25 | 62 | 2845 | 810 |

Result — **there is no underclock sweet spot for the v4 walk on the 5090**:

- Hashrate falls almost linearly with the core clock and efficiency falls with it at every step
  (9.21 → 8.48 kH/s/W from 1689 to 937 MHz). The board's fixed power (GDDR7 at 13.8 GHz, fans, VRMs)
  does not shrink with the core, so slower = less work per watt. Cooler, yes; more efficient, no.
- Halving the memory clock cuts the hashrate 40 % while the core races to 2.48 GHz and cannot make
  it up: the walk is memory-bandwidth bound as well. Efficiency drops to 7.6 kH/s/W.
- The most efficient configuration measured is **stock clocks at the full power cap**. The first 5090
  on the same rig (GPU0, 69 °C, 2.4 GHz, 600 W) does 5.97 MH/s = **9.95 kH/s/W**, 8 % better than
  GPU1 — the difference is cooling, not settings: GPU1 is thermally throttled to 1.69 GHz.
- Actionable: keep 5090s at stock clocks and the 575–600 W cap; spend effort on airflow for GPU1
  (or accept the 88 °C guard limit set 2026-10-01). Do not deploy core or memory underclocks on 5090s.

## RTX 3070 core-clock and memory-clock efficiency sweep — 2026-10-01, rig08 GPU0 (v0.13.3, PoM v4, Qwen3.5-9B tier)

Same method as the 5090 sweep above (production miner left running, 240 W cap untouched, `-lgc`/`-lmc`,
60 s settle, 10 samples 15 s apart). Raw CSV: `rig08:/home/marcel/gpu0-sweep.csv`. Note: the first run on
this box produced garbage because bash `printf` under the German locale parsed "1.441" as 1 — the script
now forces `LC_ALL=C`.

| setting | MH/s | W | kH/s/W | °C | SM MHz | mem MHz |
|---|---:|---:|---:|---:|---:|---:|
| stock, unlocked (+250 MHz offset, 240 W cap) | 1.440 | 239 | 6.02 | 67 | 1704 | 6801 |
| core locked 1500 | 1.424 | 188 | 7.56 | 62 | 1500 | 6801 |
| **core locked 1300** | **1.333** | **154** | **8.66** | **59** | 1290 | 6801 |
| core locked 1100 | 1.162 | 140 | 8.33 | 58 | 1095 | 6801 |
| core locked 900 | 0.976 | 122 | 7.98 | 56 | 900 | 6801 |
| memory locked 6000 → 5001 | 1.073 | 238 | 4.51 | 67 | 1870 | 5001 |
| memory locked 5000 → 810 | 0.146 | 79 | 1.86 | 52 | 1905 | 810 |
| memory locked 4000 → 810 | 0.146 | 78 | 1.87 | 51 | 1905 | 810 |

Result — **the 3070 has a real sweet spot, and it is the opposite of the 5090**:

- At stock the 3070 burns its whole 240 W cap for 1.44 MH/s (6.0 kH/s/W). Locking the core at
  **1300 MHz** keeps 93 % of the hashrate for 64 % of the power: **8.66 kH/s/W, +44 % efficiency**,
  8 °C cooler. 1500 MHz is the "almost free" point (−1 % hashrate, −21 % power) if hashrate matters
  more than efficiency; below 1300 the efficiency curve turns down again.
- Memory must stay at stock: any lock makes the driver drop to 5001 or 810 MHz and the walk collapses.
- **Deployed 2026-10-01 17:40Z on both rig08 3070s:** `gpu_clocks: {uuid: {"lgc": [0, 1300]}}` in
  `rig08.json` / `rig08-launch.json` (applied by the fleet guard at every launch, released on exit),
  plus applied live with `nvidia-smi -lgc 0,1300`. Expected per rig: ~2.67 MH/s at ~310 W instead of
  2.88 MH/s at ~480 W.
- Why the two cards differ: on the 5090 the fixed board power (GDDR7 at 13.8 GHz) dominates and the
  walk is bandwidth bound, so a slower core only wastes the memory power; on the 3070 the stock boost
  sits far up the V/F curve and the +250 MHz offset pushes it further, so the first few hundred MHz
  off the top cost almost nothing in hashrate and a lot in watts. Measure each architecture; do not
  copy settings across generations.

## CMP 170HX core-clock efficiency sweep — 2026-10-01, rigtr12 GPU1 (v0.13.3, PoM v4, Qwen3.5-9B tier, 250 W cap)

Same method as the two sweeps above (production miner left running and submitting, cap untouched at
250 W, `nvidia-smi -lgc <MHz>,<MHz>`, 60 s settle, 12 samples 15 s apart of the miner API hashrate plus
`nvidia-smi` power/temperature/clocks, locks released afterwards). The CMP 170HX is a GA100 die with
8 GB HBM2e at a fixed 1458 MHz (memory locks are not offered); at stock it is power-capped at 250 W and
runs ~1270 MHz. Raw results: `rigtr12:/mnt/development/workspaces/germany-keryx-switch-20260930/cmp-sweep-20261001/results.json`.

| setting | MH/s | W | kH/s/W | °C core / mem | SM MHz | mem MHz |
|---|---:|---:|---:|---:|---:|---:|
| stock, unlocked (power-capped at 250 W) | 2.385 | 248 | 9.64 | 79 / 82 | 1269 | 1458 |
| core locked 1200 | 2.258 | 212 | 10.64 | 73 / 78 | 1200 | 1458 |
| core locked 1100 | 2.093 | 178 | 11.76 | 66 / 73 | 1110 | 1458 |
| **core locked 1000** | **1.913** | **154** | **12.42** | **62 / 71** | 1005 | 1458 |
| core locked 900 | 1.719 | 141 | 12.21 | 60 / 69 | 900 | 1458 |
| core locked 800 | 1.547 | 131 | 11.82 | 58 / 68 | 810 | 1458 |
| core locked 700 | 1.351 | 117 | 11.50 | 56 / 66 | 705 | 1458 |

Result — **the CMP 170HX behaves like the 3070, with a broad optimum around 1000 MHz**:

- Efficiency rises monotonically from stock down to 1000 MHz (**9.64 → 12.42 kH/s/W, +29 %**) and
  only then turns down; 1100 MHz is the balanced point (88 % of the hashrate for 72 % of the power,
  +22 %), 1000 MHz the efficiency maximum (80 % of the hashrate for 62 % of the power, 17 °C cooler
  on the core and 11 °C on the HBM).
- HBM2e power is small and fixed, so unlike the 5090 the core is where the watts go; the walk on this
  card is not bandwidth-starved until the core drops below ~900 MHz.
- **Deployed 2026-10-01 18:57Z on rigtr12 GPU1:** `gpu_clocks: {uuid: {"lgc": [0, 1000]}}` in
  `rigtr12-cmp.json` (applied by the fleet guard at the next launch) and applied live with
  `nvidia-smi -lgc 1000,1000`. Expected: ~1.9 MH/s at ~155–165 W instead of 2.39 MH/s at 248 W.
  The second CMP 170HX fleet (us-rig-02, five cards at 160–250 W caps) is the next candidate.

# Pre-relaunch PoM (historical)

Everything below predates the keryxd v1.5.1 relaunch walk and is kept for reference only. The
hashrates are on the old walk and cannot be compared with the table above.


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

Numbers below are at each card's **AUTO tier** (noted per row; heavier tiers pay more at ~the same
walk rate — the walk is near-flat, ~5 %, across tiers). Datacenter rows were measured at `--light`.

## Per-card table — NVIDIA (keryx PoM)

Consumer/fleet rows **re-verified 2026-07-08 on v0.6.9.3** (≥5 min live per card, AUTO tier,
stock clocks/power, 0 rejects).

| GPU | Architecture | VRAM | Mem clock | Core clock | Power (mining) | Hashrate | Efficiency | Notes |
|-----|--------------|------|-----------|------------|----------------|----------|------------|-------|
| **NVIDIA H200 141GB** | Hopper (2023, datacenter) | 141 GB HBM3e | 3201 MHz | ~1980 MHz | ~628 W (of 700; cap-able) | **~166 MH/s** live (bench ceiling ~170; v0.6.8 128-bit loads) | ~0.26 MH/W | **fastest PoM card** — HBM3e ~4.8 TB/s, +32 % over H100. Memory-bound (98 % util) → cap power, no core-clock gain. `--light` |
| **NVIDIA H100 80GB** | Hopper (2022, datacenter) | 80 GB HBM3 | 2619 MHz (fixed) | ~1650 MHz | **400 W** (cap; see below) | **~123 MH/s** (ceiling 125.6) | 0.31 MH/W | ~2× a 5090. Cap PL to 400 W = 0 loss, −30 % power. `--light` |
| NVIDIA RTX 5090 | Blackwell (2025) | 32 GB GDDR7 | 13801 MHz | ~2850 MHz | ~382 W (of 600 W PL) | **~68 MH/s** | 0.18 MH/W | AUTO → very-high (Llama-70B-Q2); memory-bound, core-clock-insensitive, no gain from OC |
| NVIDIA RTX 5080 | Blackwell (2025) | 16 GB GDDR7 | 14801 MHz | ~2835–2900 MHz | ~190–202 W (of 360 W PL) | **~34.8 MH/s** | ~0.18 MH/W | AUTO → default (Dolphin-8B); self-limits well under PL |
| NVIDIA RTX 5070 Ti | Blackwell (2025) | 16 GB GDDR7 | 13801 MHz | ~2812 MHz | ~179 W (of 285 W PL) | **~34 MH/s** | ~0.19 MH/W | AUTO → default (Dolphin-8B); self-limits well under PL |
| NVIDIA CMP 170HX | Ampere GA100 (2021, mining) | 8 GB HBM2 | 1458 MHz | ~510–570 MHz (auto) | ~115 W | **~20 MH/s** | ~0.17 MH/W | AUTO → light (Gemma); HBM2 → strong for PoM; auto-parks low, already efficient |
| NVIDIA RTX 3070 | Ampere (2020) | 8 GB GDDR6 | 6801 MHz | ~1900 MHz | ~173 W (at 220 W cap) | **~18.5 MH/s** | ~0.11 MH/W | AUTO → light (Gemma) |

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

Measured live on the pool (`krx.suprnova.cc`), verified by share acceptance (0 rejects), plus an isolated
CUDA microbench of the walk kernel (`tools/h100/bench_pom.cu`) for the clean per-power-limit numbers.
Consumer/fleet rows re-measured 2026-07-08 on **v0.6.9.3** at AUTO tier (5090 → Llama-70B-Q2,
5080/5070 Ti → Dolphin-8B, 170HX/3070 → Gemma-3-4B), ≥5 min live per card, stock clocks and power limits;
datacenter rows are `--light` (Gemma-3-4B, 77.6 M chunks / 2.48 GB possession blob). HBM cards were run at stock; consumer cards at stock power limits (they
self-limit for the memory-bound walk). **AMD** numbers are the OpenCL worker (`keryxopencl`) measured live
on the same pool, with per-card hashrate/clocks/power read from `rocm-smi`, on a 3-GPU Ubuntu box (RADV/Mesa
Vulkan for OPoI inference).
