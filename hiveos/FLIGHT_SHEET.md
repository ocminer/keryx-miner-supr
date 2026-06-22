# keryx-miner-supr — HiveOS Setup Guide

**keryx-miner-supr** installs on HiveOS as a **Custom miner**. NVIDIA rigs use the
CUDA build; AMD rigs use the OpenCL build (separate package). Each release ships a
ready-made HiveOS tarball.

## 1. Requirements
- **NVIDIA driver ≥ 570** for RTX 50 (sm_120) / H100 (sm_90). Older GPUs
  (RTX 30/20…) auto-fall-back to the bundled sm_86/sm_75 kernels and work on
  older drivers.
- **Internet + a few GB of free disk.** keryx is *Optimistic Proof of Inference*:
  **no model = no mining.** On first start the miner auto-downloads the LLM
  weights (`--light` = TinyLlama, ~2 GB) and won't submit shares until they are
  ready. This is by design — the gate enforces "no inference, no mining".
- A keryx wallet address (`keryx:…`).

## 2. Get the package URL
Use the **HiveOS** asset from the GitHub release — the one named
`keryx-miner-supr-<version>.tar.gz` (**not** the `-linux-x86_64` general-Linux
build, and **not** the `-amd-` build, on an NVIDIA rig):

```
https://github.com/ocminer/keryx-miner-supr/releases/download/v<version>/keryx-miner-supr-<version>.tar.gz
```

AMD rigs: use `keryx-miner-supr-amd-<version>-hiveos.tar.gz` instead (same steps,
algorithm field still `keryxhash`).

> The HiveOS tarball is named `keryx-miner-supr-<version>.tar.gz` with **no dash
> in the version** — required by HiveOS. Don't rename it.

## 3. Create the Flight Sheet
HiveOS → **Flight Sheets** → **Add Flight Sheet**. Pick any coin/wallet (keryx
uses the template below), set **Miner = Custom**, then open **Setup Miner Config**:

| Field | Value |
|---|---|
| **Installation URL** | the tarball URL from step 2 |
| **Miner name** | `keryx-miner-supr` (auto-fills from the URL) |
| **Hash algorithm** | `keryxhash` |
| **Wallet and worker template** | `keryx:YOUR_KERYX_ADDRESS.%WORKER_NAME%` |
| **Pool URL** | `stratum+tcp://krx.suprnova.cc:4401` |
| **Pass** | *(optional — sent to the pool; on suprnova use `d=16` for static difficulty 16, otherwise leave blank)* |
| **Extra config arguments** | `--light --cpu-inference --cuda-device 0` |

Keep the `stratum+tcp://` scheme on the Pool URL — without it the miner falls
back to gRPC. Apply the Flight Sheet to your rig(s).

## 4. Extra config arguments — reference
- `--light` — TinyLlama only (smallest weights; recommended for mining rigs).
  Omit for bigger tiers: `--high` (DeepSeek-R1-32B, 24 GB+),
  `--very-high` (LLaMA-3.3-70B, 32 GB+) — those download much larger weights.
- `--cpu-inference` — run the OPoI inference on the **CPU** so the GPU stays 100%
  on hashing (recommended for dedicated rigs). Drop it to run inference on the
  GPU (needs spare VRAM; PoW pauses briefly during an inference challenge).
- `--cuda-device 0` — which GPU(s). Comma-separated for several: `--cuda-device 0,1`.
  **Omit entirely to use all GPUs.** (For PCI-bus order, the miner respects
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`.)
- If you pass no tier flag, the launcher adds `--light` automatically.

## 5. Verify
The HiveOS dashboard shows hashrate and accepted/rejected per GPU (reported by
`h-stats.sh`). For detail, open the rig's miner log or:

```
cat /var/log/miner/keryx-miner-supr/keryx-miner-supr.log
```

Expected first-run sequence: model prefetch → `using optimised sm_XX PTX` →
`Share accepted`.

## Troubleshooting
- **`GLIBC_2.3x not found`** — wrong asset. Use the
  `keryx-miner-supr-<version>.tar.gz` HiveOS package (built against glibc 2.30),
  not the `-linux-x86_64` one.
- **`OPoI: no models ready — mining suspended`** — the weights are still
  downloading, or the rig has no internet / not enough disk. Mining begins once
  the model files are present.
- **No shares / falls back to gRPC** — make sure the Pool URL keeps the
  `stratum+tcp://` prefix.
- **Worker on the wrong difficulty** — krx.suprnova.cc offers several stratum
  ports; pick the one for your difficulty per the pool site.
