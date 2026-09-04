# keryx-miner-supr — HiveOS Setup Guide

**keryx-miner-supr** installs on HiveOS as a **Custom miner**. NVIDIA rigs use the
CUDA build; AMD rigs use the OpenCL build (separate package). Each release ships a
ready-made HiveOS tarball.

## 1. Requirements
- **NVIDIA driver ≥ 570** for RTX 50 (sm_120) / H100 (sm_90). Older GPUs
  (RTX 30/20…) auto-fall-back to the bundled sm_86/sm_75 kernels and work on
  older drivers.
- **Internet + enough disk for the selected H6 model.** keryx is *Optimistic Proof of Inference*:
  **no model = no mining.** On first start the miner auto-downloads the pinned GGUF
  weights (`--very-light` = Qwen3.5-9B, roughly 6.5 GB) and won't submit shares until they are
  ready. This is by design — the gate enforces "no inference, no mining".
- A keryx wallet address (`keryx:…`).

## 2. Get the package URL
Use the **HiveOS** asset for the NVIDIA line your oldest card supports: `modern`
for sm_75+ or `legacy` for sm_61+. Do **not** use the `-linux-x86_64` general-Linux
archive or the AMD package on an NVIDIA rig:

```
https://github.com/ocminer/keryx-miner-supr/releases/download/v<version>/keryx-miner-supr-modern-<version>.tar.gz
```

AMD rigs: use `keryx-miner-supr-amd-<version>.tar.gz` instead (same steps,
algorithm field still `keryxhash`).

> The line name is part of the HiveOS miner name, before the final version field
> (for example `keryx-miner-supr-modern-0.13.1.tar.gz`). Don't rename it.

## 3. Create the Flight Sheet
HiveOS → **Flight Sheets** → **Add Flight Sheet**. Pick any coin/wallet (keryx
uses the template below), set **Miner = Custom**, then open **Setup Miner Config**:

| Field | Value |
|---|---|
| **Installation URL** | the tarball URL from step 2 |
| **Miner name** | auto-filled from the URL (`keryx-miner-supr-modern`, `-legacy`, or `-amd`) |
| **Hash algorithm** | `keryxhash` |
| **Wallet and worker template** | `keryx:YOUR_KERYX_ADDRESS.%WORKER_NAME%` |
| **Pool URL** | `stratum+tcp://krx.suprnova.cc:4401` |
| **Pass** | *(optional — sent to the pool; on suprnova use `d=16` for static difficulty 16, otherwise leave blank)* |
| **Extra config arguments** | *optional* — blank = per-card AUTO; e.g. `--very-high` pins a tier |

Keep the `stratum+tcp://` scheme on the Pool URL — without it the miner falls
back to gRPC. Apply the Flight Sheet to your rig(s).

## 4. Extra config arguments — reference

**Model tier** (heavier = more reward; each needs more VRAM + a bigger first-run download):
- `--very-light` — Qwen3.5-9B (≥ 7 GB) · `--light` — GLM-4-9B (≥ 11 GB) ·
  *(default)* Gemma-4-12B (≥ 15 GB) · `--high` — Qwen3.6-27B (≥ 22 GB) ·
  `--very-high` — Kimi-Linear-48B (≥ 28 GB).
- **`--tier auto`** — **per-card AUTO** (this is also the default with blank extra args): each GPU
  loads the heaviest model its own VRAM can hold, so a mixed rig gets a different model per card = max reward.
- **`--force-model <csv>`** — force a model per card, in CUDA device order, e.g.
  `--force-model very-high,light,default` → GPU0=Kimi-48B, GPU1=GLM-9B, GPU2=Gemma-12B. Unlisted cards fall
  back to AUTO. Bypasses the VRAM-fit check (a card too small will OOM). Names: `very-light,light,default,high,very-high`.
- ℹ️ **Blank extra args = `--tier auto`** (per-card best-fit — the launcher adds it for you). Pass a single
  tier flag (`--light`, `--very-high`, …) to instead pin **every** card to that one tier.
- `--cpu-inference` — **deprecated emergency-only override** that forces OPoI inference onto the
  CPU. It is extremely slow and planned for removal; normal rigs must use the packaged GPU
  inference engine. PoW pauses safely while a GPU inference challenge runs.
- `--cuda-device 0` — which GPU(s). Comma-separated for several: `--cuda-device 0,1`.
  **Omit entirely to use all GPUs.** (For PCI-bus order, the miner respects
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.)

## 5. Verify
The HiveOS dashboard shows hashrate and accepted/rejected per GPU (reported by
`h-stats.sh`). For detail, open the rig's miner log or:

```
cat /var/log/miner/keryx-miner-supr/keryx-miner-supr.log
```

Expected first-run sequence: model prefetch → GPU inference residency → optimized PoM kernel selection →
`Share accepted`.

## Troubleshooting
- **`GLIBC_2.3x not found`** — wrong asset. Use the
  `keryx-miner-supr-<line>-<version>.tar.gz` HiveOS package (built against glibc <= 2.31),
  not the `-linux-x86_64` one.
- **`OPoI: no models ready — mining suspended`** — the weights are still
  downloading, or the rig has no internet / not enough disk. Mining begins once
  the model files are present.
- **No shares / falls back to gRPC** — make sure the Pool URL keeps the
  `stratum+tcp://` prefix.
- **Worker on the wrong difficulty** — krx.suprnova.cc offers several stratum
  ports; pick the one for your difficulty per the pool site.
