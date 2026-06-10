# HiveOS custom miner — keryx-miner-supr

HiveOS rigs run an **older glibc** (Ubuntu 20.04 = glibc 2.31) than the dev rig
(Ubuntu 24.04 = glibc 2.39). A binary built natively on the dev rig needs up to
`GLIBC_2.39` and fails on HiveOS with `version 'GLIBC_2.3x' not found`. This
directory builds a glibc-2.31 binary in a container and wraps it in the HiveOS
custom-miner format.

## Build the package

```bash
hiveos/build-glibc.sh     # builds binary + plugins in nvidia/cuda:12.8.0-devel-ubuntu20.04 (glibc 2.31)
hiveos/package.sh         # -> hiveos/dist/keryx-miner-supr-<version>.tar.gz
```

`build-glibc.sh` prints the max `GLIBC_*` symbol of each artifact — it must be
`<= 2.31`. The CUDA PoW kernels (`keryx-cuda-sm120.ptx` etc.) are pre-built and
embedded, so the container does **not** regenerate PTX; the committed CUDA 13.0 /
PTX 9.0 kernels ship as-is. `cust` links the CUDA *driver* API (`libcuda.so.1`),
which the installed NVIDIA driver provides on the rig.

> **Driver note:** the native `sm_120` PTX is PTX ISA 9.0 and needs an NVIDIA
> driver **>= 570** to JIT (required for the RTX 5090 anyway). Older cards on
> older HiveOS drivers fall back to the upstream sm_86/sm_75 PTX automatically.

## Package layout

The CUDA worker is linked into the binary (`--features static-cuda`), so the
payload is a **single executable** — no `libkeryx*.so` to ship. The only `.so`
it needs at runtime is the rig's NVIDIA driver `libcuda.so.1`.

```
keryx-miner-supr/
├── h-manifest.conf      # name / version / log + config paths
├── h-config.sh          # flight sheet -> CLI args
├── h-run.sh             # launches the miner (foreground)
├── h-stats.sh           # hashrate/shares from log + temps/fans from agent
└── keryx-miner-supr     # single static-cuda binary (glibc 2.30)
```

## Install on a rig (Flight Sheet)

1. Host the `keryx-miner-supr-<version>.tar.gz` at a URL the rig can reach
   (e.g. a GitHub release asset).
2. HiveOS → **Flight Sheet** → Miner = **Custom**:
   - **Installation URL**: the tarball URL
   - **Miner name**: `keryx-miner-supr`
   - **Hash algorithm**: `keryxhash`
   - **Wallet and worker template**: `keryx:<addr>.%WORKER_NAME%`
   - **Pool URL**: `stratum+tcp://krx.suprnova.cc:4401` (the scheme is required)
   - **Pass**: as needed (keryx pools usually ignore)
   - **Extra config arguments**: e.g. `--light --cuda-device 0`
     (`--light` = TinyLlama only; omit for higher model tiers, which need the
     weights pre-fetched and more VRAM)

Stats (hashrate, accepted/rejected, per-GPU) report back to the HiveOS dashboard
via `h-stats.sh`.
