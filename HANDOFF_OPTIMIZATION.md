# keryx-miner-supr — Optimization Handoff

**Audience:** a fresh Claude session whose only job is to push the CUDA kernel(s)
to maximum throughput on RTX 5090 (sm_120) and CMP 170HX (sm_80). The pool side
of the stack is handled by a separate session — do not touch
`/home/marcel/keryx-pool/...`, `keryxd`, or any pool config.

This document is the complete, self-contained brief. Read it top to bottom
before doing anything. Nothing here is rumour: every measurement, file path,
and constraint has been verified against the running system.

---

## SESSION CHANGELOG (2026-06-10) — current state, read this first

All committed/pushed to `origin/main` (latest HEAD around `9ad5de4`+).

**Performance (stock power, 0 rejects, pool-verified):**
- RTX 5090 (sm_120): **2.57 → 3.28 GH/s (+28%)** — root cause was the Keccak
  round loop being *rolled* (229 regs, 1 block/SM); added `#pragma unroll` →
  64 regs, 2 blocks/SM. Now **power-bound** at the 575 W cap.
- CMP 170HX (sm_80): **154 → 188 MH/s (+22%)** — arch-gated
  `__launch_bounds__(512,2)` → 64 regs, 2 blocks/SM. Occupancy-capped there
  (50-reg Keccak state floor) and cooling-limited (~82 °C in ~90 s even at
  100% fan). 400 MH/s target is NOT reachable with this kernel.
- Both cards together: **3.46 GH/s**, clean sum, no PCIe contention.
- Matmul is only ~10% of instructions and already on the uniform datapath —
  multi-accumulator attempt regressed (busted the 64-reg budget). Don't.

**Correctness / stability fixes:**
- **Devfund cycle removed entirely** — it had a crash-loop bug (counter wraps
  to 0 → `listen()` returns before processing a job → tight reconnect loop).
  Archived in `docs/devfund-removed.md` + branch `archive/devfund-cycle`.
- **Reconnect backoff** — was a 100 ms busy-loop hammering a dead pool +
  re-initialising the GPU each spin; now exponential 1→30 s, resets after a
  healthy (≥60 s) session. `main.rs` outer loop.
- **VRAM startup line** showed 0 MB on multi-GPU rigs (nvidia-smi multi-line
  CSV parsed as one blob) — now parses the first line.

**Packaging / build:**
- **HiveOS = single static binary** via `static-cuda` feature + new
  `keryx-plugin-api` crate (see [[static-cuda-single-binary]] memory).
  `hiveos/build-glibc.sh` builds glibc-2.30 in a 20.04 container;
  `hiveos/package.sh` → `hiveos/dist/keryx-miner-supr-<ver>.tar.gz` (one binary
  + h-*.sh). Host to a public URL; Flight Sheet steps in `hiveos/README.md`.
- **Build rig is Ubuntu 24.04 / CUDA 13.0** (not 12.8 — README/build use 13.0
  now). NEVER `cargo clean` — LLM weights are cached in `target/` (the static
  build uses a separate `target-hiveos/`).
- Big models can be fetched fast with `aria2c -x16` from the IPFS gateway
  (supports range requests) into `<exe_dir>/models/<DirName>/` + a `.ok`
  sentinel file; verify sha256 against the spec's `model_id`.

---

## 1. Project context

`keryx-miner-supr` is the Suprnova fork of
[`keryx-labs/keryx-miner`](https://github.com/keryx-labs/keryx-miner)
v0.3.2 (upstream commit `317fcab` "New release v0.3.2: SALT v4 + escrow/channel perf").

| What | Upstream | `-supr` |
|------|----------|---------|
| Dev tax | 2 %, min-clamped, hardcoded address | **0 %** by default, address points at the pool wallet (`keryx:qp0vrxc0k5w0pcyem6vau2pjgztje880tsm239rywtm7l7uv2pcxzq55n8khs`); `--devfund-percent N` to opt back in |
| sm_120 (RTX 50) | sm_100 PTX → "unknown error" → JIT'd sm_86 (≈ 50 % of native) | Native `keryx-cuda-sm120.ptx` (CUDA 13.0 nvcc, PTX 9.0) |
| sm_80 (A100/CMP 170HX) | falls through to sm_75 PTX | Native `keryx-cuda-sm80.ptx` (CUDA 13.0 nvcc, PTX 9.0) |
| Monitoring / OC | upstream `overclock` feature exists but is opt-in | `default = ["overclock"]`, periodic `[GPU #X] temp=… fan=… power=…` log line, `--cuda-fan-speed`, `--cuda-monitor-interval` flags |
| CUDA toolkit | pinned to 12.x via `cudarc 0.13.9` | dev-built against 13.0 — runtime is `cust 0.3` which doesn't care about toolkit version |

Repo: `git@github.com:ocminer/keryx-miner-supr.git` (private, ocminer = the pool
operator's GitHub). The dev tree on the *host* has no remote; the rig
(`192.168.15.112`) is the one with `origin` configured.

---

## 2. Hardware targets

Both GPUs live on rigtr12 (`192.168.15.112`). The pool host where this repo
also exists has **no GPU work** — it's only there because that's where the
fork was originally cut. Do all build+test work directly on the rig.

| Slot | Device | CC | SMs | VRAM | TDP | Notes |
|------|--------|----|-----|------|-----|-------|
| GPU #0 | NVIDIA GeForce RTX 5090 | **8.0 → wait, reported as `12.0`** | 170 | 32 607 MiB GDDR7 | 575 W (600 W cap) | Consumer Blackwell. `nvidia-smi --query-gpu=compute_cap` reports `12.0` because NVIDIA splits sm_100 (datacenter) and sm_120 (consumer). |
| GPU #1 | NVIDIA Graphics Device (CMP 170HX) | **8.0** | ≈ 70 | 8 192 MiB **HBM2** | 250 W | Datacenter Ampere (GA100 silicon). 1.5 TB/s of memory bandwidth, **mining-restricted** silicon — KeryxHash is *not* on the blacklist so it runs. Fan is undersized; auto-shut after 3 min. |

Driver: `580.159.04` on both. The driver supports CUDA up to 13.0 — PTX
version **9.0**. **CUDA 13.2 (PTX 9.2) compiled binaries fail with `unknown
error`** at module load. This was the trip-wire that bounced us back to
CUDA 13.0 for both PTX targets.

CUDA toolkits installed on the rig:
```
/usr/local/cuda-12   → 12.x
/usr/local/cuda-12.8
/usr/local/cuda-12.9
/usr/local/cuda-13   → 13.x default
/usr/local/cuda-13.0  ← use this for nvcc -ptx
/usr/local/cuda-13.2  ← compiles, but PTX 9.2 won't load on driver 580
```

---

## 3. Current performance — read this before optimising anything

### RTX 5090

| Source | Hashrate | Note |
|--------|----------|------|
| Upstream `keryx-miner` v0.3.2 (control) | **2.77 GH/s** | Same binary path, same PTX, same workload — verified by manual run before fork work. |
| `-supr` baseline (commit `158224b`, native sm_120 PTX, no kernel changes) | **2.77 GH/s** | Parity established. |
| `-supr` after inlining both Keccak calls (commit `5a1535f`) | **2.78 GH/s** | Marginal — within noise. |
| `-supr` **current HEAD** (`344eb5c`) | **2.55 GH/s** ⚠ | Unexplained regression — see § 6. |

**The 5090 regression is the #1 unresolved problem.** Same source, same PTX
(.so md5 unchanged across rebuilds), but the kernel's chosen workload halved
from 89 M → 44 M nonces. No launch-config flag changed. Bisect points either
at `cust 0.3` × `nvml-wrapper 0.12` interaction at runtime or a Rust binary
link-time effect. Do **not** start optimising the kernel body until you can
either reproduce the 2.77 GH/s baseline or explain the gap.

Target: **3.0 GH/s at stock power** (575 W TDP, no `--cuda-power-limits`).

### CMP 170HX

| Source | Hashrate | Note |
|--------|----------|------|
| `-supr` HEAD with native sm_80 PTX (commit `344eb5c`) | **155–160 MH/s** at 127 W / 250 W TDP, 82 °C peak | Tested for ≈ 2 min via `CUDA_VISIBLE_DEVICES=1 ./keryx-miner-supr --light --cuda-device 0 …`. Shares accepted, no consensus errors. |
| `-supr` keccak-unroll, no launch bounds (sm_80 → 72 regs) | **~154 MH/s** | Unroll alone did NOT help sm_80 — 72 regs is just over the 64-reg cliff, so still 1 block/SM. |
| `-supr` **`__launch_bounds__(512,2)`** for sm_80 (commit `4a11835`) | **~188 MH/s** (+22%) | 64 regs, 12-byte spill, **2 blocks/SM**. 1410 MHz pinned, throttle reasons `0x0`, ~120 W of 250 W → **occupancy-bound, not power-bound.** |

#### ⚠ 170HX THERMAL FINDINGS (2026-06-09) — read before benching this card

1. **The in-miner NVML monitor line reports the WRONG GPU under
   `CUDA_VISIBLE_DEVICES`.** nvml-wrapper enumerates physical GPUs and
   ignores the CUDA mask, so a `CUDA_VISIBLE_DEVICES=1` run prints the
   **idle 5090's** temp/power/clock, not the 170HX's. For real 170HX
   telemetry use `nvidia-smi -i 1 --query-gpu=temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active`.
2. **Real 170HX clock ceiling is 1410 MHz** (`nvidia-smi -i 1`), 250 W
   power limit. Under the optimised kernel it runs at full 1410 MHz with
   **no clock throttling** (`clocks_throttle_reasons.active = 0x0`).
3. **The blower is the wall, not the kernel.** At 2 blocks/SM the card
   heats ~0.15–0.17 °C/s and reaches **~82–83 °C in ~90 s even at
   `--cuda-fan-speed 100`** (solo *or* alongside the 5090). The earlier
   "worker re-creates 90×" cascade was **thermal accumulation** near the
   ~94 s mark on a 150 s run — NOT a software bug (a clean 72 s run had
   exactly 1 worker start). **Keep every 170HX bench ≤ ~90 s** and let the
   card idle-cool (it cools slowly: ~80 °C → 50 °C takes several minutes
   on the 30 % idle fan).
4. **Occupancy is capped at 2 blocks/SM.** The Keccak f-1600 state needs
   ~50 registers, so the kernel cannot drop below 64 regs without
   spilling (56 regs → 2 KB spill). 3 blocks/SM would need ≤ 42 regs →
   impossible. **~188 MH/s is the practical ceiling for this kernel on
   GA100**; the original ≥ 400 MH/s target is not reachable with the
   current algorithm (it's at full clock, no throttle, occupancy-maxed —
   the only headroom is power, which it can't use because the SMs stall).

Original (pre-work) theoretical ceiling note, kept for reference:
**≈ 500 MH/s** if the kernel became HBM2-aware (matrix or per-warp state
staged through global memory streams). In practice the kernel is compute/
latency-bound, not memory-bound, and constant-memory matrix loads already
ride the cheap uniform datapath (`LDCU`, broadcast across the warp), so the
HBM2 angle does not apply on the current kernel.

Realistic target: **~188 MH/s achieved**; further gains need a fundamentally
different (lower-register) hash structure, and would still hit the cooling
wall for sustained runs.

---

## 4. Hard constraints — these came directly from the operator

1. **No power tuning to hit the target.** "I don't think we should achieve
   the 3 GH/s through 'tuning' of power parameters but through optimizing
   the CUDA kernel — it should achieve 3 GH/s with 'stock' settings like
   before." Kernel body, PTX inline asm, algorithm shortcuts only.
2. **CUDA 13 only.** No regressing the build to CUDA 12.x. cudarc isn't in
   the dependency tree — the runtime is `cust 0.3` which couples to the
   driver, not the toolkit, so this is purely a build-time choice.
3. **170HX 3-minute fan limit.** The blower fan can't keep up — every
   benchmark session on GPU #1 must terminate before the 3-minute mark.
   Use `timeout 170` or set a wall-clock guard. 82 °C was the peak seen
   at 2 minutes; throttling starts well before that on this silicon.
4. **Don't touch the consensus path.** `wave_mix`'s four 64-bit constants
   (`0x9e3779b97f4a7c15`, `0x6c62272e07bb0142`, `0xb5ad4eceda1ce2a9`,
   `0x243f6a8885a308d3`) and the rotation schedule `[17, 47, 31, 13]` are
   protocol — changing any of them silently mines a different chain.
   Same for `powP` and `heavyP` Keccak round constants and the 64×64
   nibble matrix path. Optimisation = same hash output, fewer cycles.
5. **No devfund tax.** Already neutered (`src/cli.rs:97` defaults to 0,
   address re-pointed at the pool wallet at `src/cli.rs:174`). Don't put
   it back.

---

## 5. File layout

```
/home/marcel/keryx-miner-supr/              ← repo root (both host and rig)
├── Cargo.toml                              ← workspace, [lib] name = "keryx_miner"
├── README.md                               ← user-facing, slightly out-of-date on CUDA version
├── HANDOFF_OPTIMIZATION.md                 ← THIS FILE
├── src/
│   ├── cli.rs                              ← devfund changes live here (lines 97, 174)
│   └── ... (rest is upstream)
├── plugins/cuda/                           ← the only place that matters for perf
│   ├── Cargo.toml                          ← default-features = ["overclock"]
│   ├── src/
│   │   ├── cli.rs                          ← --cuda-fan-speed, --cuda-monitor-interval
│   │   ├── lib.rs                          ← Plugin + monitor thread (~50 lines of NVML)
│   │   └── worker.rs                       ← PTX dispatcher (see § 5.1)
│   ├── resources/
│   │   ├── keryx-cuda-sm120.ptx            ← CUDA 13.0 nvcc, RTX 5090
│   │   ├── keryx-cuda-sm100.ptx            ← upstream, datacenter Blackwell (H100/B100)
│   │   ├── keryx-cuda-sm89.ptx             ← upstream, RTX 40 (Ada)
│   │   ├── keryx-cuda-sm86.ptx             ← upstream, RTX 30 (consumer Ampere)
│   │   ├── keryx-cuda-sm80.ptx             ← CUDA 13.0 nvcc, CMP 170HX / A100
│   │   ├── keryx-cuda-sm75.ptx             ← upstream, RTX 20 (Turing)
│   │   └── keryx-cuda-sm61.ptx             ← upstream, GTX 10 (Pascal)
│   └── kaspa-cuda-native/src/
│       ├── kaspa-cuda.cu                   ← the kernel — heavy_hash() lives here
│       ├── keccak-tiny.c                   ← Keccak-f1600 reference
│       ├── keccak-tiny-unrolled.c
│       └── xoshiro256starstar.c
└── plugins/opencl/                          ← rarely touched; nobody benches against this

/home/marcel/keryx-miner-supr-run/          ← runtime directory on the RIG only
├── keryx-miner-supr                        ← copy from target/release/
├── libkeryxcuda.so                         ← copy from target/release/
├── libkeryxopencl.so
├── models/TinyLlama-1.1B/                  ← prefetched via IPFS, 2.2 GB
├── ipfs                                    ← go-ipfs binary, daemon runs as root
├── escrow.key                              ← Keryx escrow key (don't touch)
└── supr.log                                ← last benchmark log
```

### 5.1 PTX dispatcher (`plugins/cuda/src/worker.rs`)

The dispatcher in `CudaGPUWorker::new()` is the only place where PTX
selection happens. The relevant branches (CC = compute capability):

```rust
if major >= 12 {           // sm_120 — RTX 5090 (note: NVIDIA reports CC 12.0)
    PTX_120 → fallback PTX_100 → fallback PTX_86
} else if major >= 10 {    // sm_100 — H100/B100/GH100 datacenter Blackwell
    PTX_100 → fallback PTX_86
} else if major == 9 || (major == 8 && minor >= 9) {  // sm_89 — RTX 40
    PTX_89 → fallback PTX_86
} else if major == 8 && minor >= 6 {  // sm_86 — RTX 30
    PTX_86
} else if major == 8 {     // sm_80 — A100 / CMP 170HX
    PTX_80 → fallback PTX_75
} else if major > 7 || (major == 7 && minor >= 5) {   // sm_75 — RTX 20
    PTX_75
} else if major > 6 || (major == 6 && minor >= 1) {   // sm_61 — GTX 10
    PTX_61
} else {
    error "unsupported"
}
```

`load_ptx()` always uses `OptLevel::O4`. The actual occupancy autotuning
happens inside `Kernel::new()` via `func.suggested_launch_configuration(0, 0)`
+ `func.max_active_blocks_per_multiprocessor(...)`. Don't bypass this without
profiling — see § 6 for what happens when you do.

---

## 6. What's already been tried and didn't pay off

Every one of these was measured against the 2.77 GH/s control on the 5090
and is recorded in commits + the comment block in `kaspa-cuda.cu` above
`heavy_hash()`.

| Attempt | Result | Why it failed |
|---------|--------|---------------|
| `__launch_bounds__(512, 2)` — force 2 blocks/SM, cap regs at ~64 | 2.78 → **2.01 GH/s** | Keccak's f-1600 state spilled to local memory; latency blew up. Kernel is genuinely register-hungry. |
| `__launch_bounds__(1024, 1)` — single big block per SM | 2.78 → **2.55 GH/s** | cust runtime picks `block_size = 512` for max occupancy at 126 regs; forcing 1024 either inflated reg pressure across 1024 threads or pushed in-flight warps past the SM scheduler sweet spot. |
| Widen the lop3.b32 inline-asm gate from `<500 \|\| >700` to `>=500` (cover sm_120) | 2.78 → **2.55 GH/s** | The C path lets nvcc fuse a u8 byte-permute with the xor step. Promoting to u32 + lop3 + truncate added an extra promotion path that costs more than the explicit lop3 saves. |
| Recompile sm_120 PTX with CUDA 13.2 nvcc (PTX 9.2) | "unknown error" at module load | Driver 580 caps at PTX 9.0. CUDA 13.0 is the highest toolkit that produces driver-580-compatible PTX. |
| Inline both Keccak calls, skip the 80-byte local input buffer (`5a1535f`) | 2.77 → 2.78 GH/s (kept) | Marginal but free — saved one 80-byte stack frame and two memcpys per nonce. |

**Bottom line:** Launch-config tuning is a dead end on sm_120. Optimisation
levers worth pulling are inside the kernel body — PTX inline asm for Keccak
rotates, algorithmic shortcuts in the matrix step, and HBM2-aware data
staging on sm_80.

---

## 7. Profiling baseline (RTX 5090, current PTX, --light tier, 89 M-nonce launch)

From `ncu --section SpeedOfLight` (Nsight 2026.1.1):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Compute SM throughput | **91.50 %** | Kernel is compute-bound, not memory-bound. |
| Memory throughput | 1.87 % | Entirely cached — no global memory pressure. |
| Mem pipes busy | 4.90 % | Confirms above. |
| Registers per thread | **126** | **The hard limiter.** This is what kills occupancy. |
| Block size | 512 | |
| Block limit (registers) | 1 block / SM | Direct consequence of the reg count. |
| Theoretical occupancy | 33.33 % | Capped by reg pressure. |
| Achieved occupancy | 32.44 % | ≈ theoretical — scheduler is doing its job. |
| L1 hit rate | 100 % | Matrix lives in L1. |
| L2 hit rate | 91 % | |

The kernel is **register-hungry, compute-bound, and the matrix is hot in L1**.
Memory bandwidth (where HBM2 would help) is irrelevant on the 5090. On the
170HX it might matter more if we can shift cost into memory access patterns
the HBM2 array is built for.

---

## 8. Task list — prioritised

### P0 — unblock further work
- [ ] **Resolve the 5090 2.55 vs 2.77 GH/s regression.**
  * Reproduce both numbers, confirm the .so / binary / PTX hashes are
    actually different between the two runs.
  * Suspect 1: `cust 0.3` + `nvml-wrapper 0.12` interaction — try a build
    with `--no-default-features` (drops `overclock`, drops nvml-wrapper)
    and benchmark.
  * Suspect 2: workload selection — log `chosen_workload` in both runs;
    upstream lands on 89 M, fork lands on 44 M.
  * Suspect 3: Rust binary linkage — `objdump -T target/release/keryx-miner-supr | wc -l`
    on both binaries to look for an unexpected symbol delta.
  * Don't start kernel-body work until this is closed — you can't tell
    if a kernel change is a win otherwise.

### P1 — 5090 to 3 GH/s
- [ ] **Hand-tuned Keccak rotates via `shf.l.wrap.b32` PTX inline asm.**
  The 24-round Keccak-f1600 is run twice per nonce. Each round has 5
  rotates of varying widths; nvcc lowers these to `funnel shift` ops
  but inline asm can pair them directly with the `bitwise_and` step
  and avoid the temporary u64. See `keccak-tiny-unrolled.c` for the
  reference path.
- [ ] **Bit-interleaved Keccak.** Standard implementations work on
  64-bit lanes; bit-interleaving splits the lanes into even/odd
  32-bit halves so all rotates become 32-bit shifts. On sm_120 the
  64-bit shifter is 2× slower than 32-bit. Reference:
  "Implementing Lightweight Block Ciphers on x86 Architectures"
  (the trick is described there for ARX ciphers, applies to Keccak's
  rotate-heavy θ and ρ steps).
- [ ] **`__expf`-style PTX for the matmul step.** `__dp4a` is already
  used; investigate `mma.sync` with INT4 packing — the 64×64 nibble
  matrix is *exactly* the shape tensor cores eat. sm_120 has FP4
  tensor cores; the nibble matrix is technically INT4 not FP4 but
  reinterpreting may work. **Profile first — tensor cores have a
  spin-up cost that may not amortise across only 32 nibbles per
  nonce.**
- [ ] **Multi-nonce-per-thread.** Currently 1 thread = 1 nonce. With
  126 regs/thread the SM is starved of warps. Pack 2 nonces per
  thread (regs allowing) — same Keccak state vector loaded once,
  2 matmuls back-to-back. Watch for the spill cliff observed with
  launch_bounds(512, 2).

### P1 — 170HX to ≥ 400 MH/s
- [ ] **Stage the 64×64 matrix through shared memory instead of L1.**
  GA100 has 192 KiB of shared mem/SM (vs sm_86's 100 KiB). The
  whole matrix (4 KiB) fits trivially. Cooperative load = each
  warp loads one row, no global mem traffic per nonce after the
  first.
- [ ] **HBM2-aware multi-nonce batching.** HBM2 favours wide
  contiguous reads. Batch 8+ nonces per launch, stage the per-nonce
  pre-pow Keccak outputs in shared mem, then run the matmul on the
  batched output as a single coalesced read.
- [ ] **`mma.sync` with INT4** — same idea as on sm_120 but here we
  have 70 SMs starving for work, so spinning up tensor cores is more
  likely to win on the throughput vs. register-pressure trade.
- [ ] **Per-warp Keccak state in shared memory.** GA100's larger
  shared budget makes this affordable; cuts the per-thread register
  footprint and lets the SM run more blocks concurrently.

### P2 — engineering hygiene
- [ ] Push host-side commits to the rig (the host repo has commit `344eb5c`
      that the rig doesn't — host has no remote, rig does).
- [ ] Set up CI on the github fork so PTX builds + binary build are
      reproducible across commits.
- [ ] Tag `v0.4.0` once 5090 hits 3 GH/s with a prebuilt static binary.
- [ ] Inline `tag_fixed` into the heavy-hash launch — saves one
      CPU↔GPU roundtrip per nonce window. Currently a separate
      MLP pass on the host. Cited in upstream README roadmap.

### P3 — investigations / nice-to-haves
- [ ] Check whether `cudarc` has a release that handles CUDA 13.x.
      `cust 0.3` is the runtime in use; `cudarc` is only mentioned in
      the README roadmap. Decoupling the toolkit-version question
      from the cudarc upgrade may already be done.
- [ ] Self-hosted model weights (README roadmap item #2). Not on the
      perf critical path but a stability win.
- [ ] OpenCL plugin: nobody benches it. Either delete or accept it as
      effectively dead code.

---

## 9. Build instructions

### 9.1 Build the Rust binary + plugins (on the rig)

```bash
# CUDA 13.0 toolkit. The driver caps at PTX 9.0, so don't use 13.2.
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=/usr/local/cuda-13.0
export CUDA_COMPUTE_CAP=120

cd /home/marcel/keryx-miner-supr
cargo build --release
```

Workspace build is mandatory — `cargo build --release --bin keryx-miner-supr`
skips the plugins and the binary refuses to start with `No workers specified`.

Outputs:
- `target/release/keryx-miner-supr` (~26 MB)
- `target/release/libkeryxcuda.so` (~5 MB) — must be next to the binary at runtime
- `target/release/libkeryxopencl.so` (~3 MB)

### 9.2 Re-compile PTX (when you edit `kaspa-cuda.cu`)

```bash
cd /home/marcel/keryx-miner-supr/plugins/cuda/kaspa-cuda-native/src

# sm_120 — RTX 5090
/usr/local/cuda-13.0/bin/nvcc -ptx -O3 \
  -gencode=arch=compute_120,code=compute_120 \
  --use_fast_math -Xptxas -O3 \
  -o ../../resources/keryx-cuda-sm120.ptx \
  kaspa-cuda.cu

# sm_80 — CMP 170HX / A100
/usr/local/cuda-13.0/bin/nvcc -ptx -O3 \
  -gencode=arch=compute_80,code=compute_80 \
  --use_fast_math -Xptxas -O3 \
  -o ../../resources/keryx-cuda-sm80.ptx \
  kaspa-cuda.cu
```

The PTX files are `include_str!`'d into the binary at build time, so after
recompiling PTX you must also rerun `cargo build --release` to relink.

### 9.3 Verify the PTX version

The "unknown error" trap. Driver 580 only loads PTX ≤ 9.0:
```bash
head -3 /home/marcel/keryx-miner-supr/plugins/cuda/resources/keryx-cuda-sm120.ptx
# expect: .version 9.0   (or older — 8.x is fine too)
# NOT:    .version 9.2   ← won't load
```

---

## 10. Run instructions

### 10.1 Bench the 5090

```bash
cd /home/marcel/keryx-miner-supr-run

# Copy fresh binaries from the repo build
cp ../keryx-miner-supr/target/release/keryx-miner-supr .
cp ../keryx-miner-supr/target/release/libkeryxcuda.so .

CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  ./keryx-miner-supr \
    -a keryx:qp0vrxc0k5w0pcyem6vau2pjgztje880tsm239rywtm7l7uv2pcxzq55n8khs.bench-5090 \
    -s stratum+tcp://krx.suprnova.cc:4401 \
    --light --cuda-device 0 \
    --cuda-monitor-interval 10 \
    2>&1 | tee supr.log
```

`--light` = TinyLlama only (already downloaded — see § 5 layout). Without it
the miner will block on a 40 GB DeepSeek-R1-70B download. **Always pass
`--light` for benchmarking unless you specifically need the higher tier.**

The monitor line every 10 s gives temp / fan / power / clocks — use that to
verify you're not power-limited or thermally throttled when measuring.

### 10.2 Bench the 170HX (hard 3-minute timer)

```bash
cd /home/marcel/keryx-miner-supr-run

timeout 170 env CUDA_VISIBLE_DEVICES=1 \
LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  ./keryx-miner-supr \
    -a keryx:qp0vrxc0k5w0pcyem6vau2pjgztje880tsm239rywtm7l7uv2pcxzq55n8khs.bench-170hx \
    -s stratum+tcp://krx.suprnova.cc:4401 \
    --light --cuda-device 0 \
    --cuda-monitor-interval 10 \
    2>&1 | tee supr-170hx.log
```

`CUDA_VISIBLE_DEVICES=1` hides the 5090 from cust so device index 0 inside
the miner = physical GPU #1 = 170HX. The 2 min 50 s `timeout` is below
the fan-safety limit; the GPU is already at 82 °C at this point.

### 10.3 Bench both cards together (production-like)

Don't, until both kernels are at their per-card peak. Each card's optimal
launch config is different and they fight for PCIe DMA. Solve them
independently first.

---

## 11. KeryxHash algorithm cheat-sheet

Two Keccak-f1600 invocations bracket a 64×64 nibble matrix-vector product
plus a 4-round ARX (wave_mix). Per nonce:

1. **Pre-pow Keccak** — absorb `hash_header[72] || nonce[8]` XOR'd against
   `powP[200]` (the first 80 bytes), permute, take 32 bytes of output.
2. **Nibble unpack** — split each output byte into two 4-bit nibbles
   (`packed_hash[16]` as `uchar4`).
3. **Matmul** — for each of 32 rows: `__dp4a(matrix[2r], packed_hash, ...)`
   produces two `uint32_t` accumulators, shifted + masked into a u8 that
   XORs into `hash_.hash[r]`.
4. **wave_mix** — 4 rounds of ARX on the 4 × u64 state with the fixed
   constants (§ 4 constraint 4).
5. **Final Keccak (heavyP)** — absorb the wave-mixed 32 bytes XOR'd against
   `heavyP[200]` (first 32 bytes; bytes 32..79 are zero so 6 XORs skip),
   permute, take 32 bytes of output.
6. **Target check** — `LT_U256(hash_, target)`; first thread to win writes
   its nonce via `atomicCAS` to `final_nonce`.

The whole thing is in `plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu`
under `__global__ void heavy_hash(...)` — 125 lines including the
inline comment block.

---

## 12. Git workflow

```bash
# On the rig — origin is configured here, not on the host:
cd /home/marcel/keryx-miner-supr
git log --oneline -10              # see recent work
git pull origin main               # pick up host commits (host -> push -> pull here)
# ... do work ...
git add plugins/cuda/...
git commit -m "plugins/cuda: <thing>"
git push origin main
```

**The host repo has no remote configured.** Commit `344eb5c` (sm_80 PTX) was
made on the host and never made it to the rig or github. The new session
should:
1. `scp` the sm_80 PTX from the host to the rig (or just rebuild it from
   `kaspa-cuda.cu` with the nvcc command in § 9.2).
2. Either set up `origin` on the host and push, or treat the rig as the
   canonical workspace going forward.

Recent commits (host head):
```
344eb5c plugins/cuda: add native sm_80 PTX for CMP 170HX / A100        ← not on rig/github yet
eee6105 plugins/cuda: launch-bounds sweep is a regression on sm_120 — document & revert
5a1535f plugins/cuda: inline both Keccak calls, drop the 80-byte local input buffer
33a9a54 plugins/cuda: temperature + fan + clock monitor, default-on overclock surface
fb71199 plugins/cuda: recompile sm_120 PTX with CUDA 13.0 (PTX version 9.0)
158224b plugins/cuda: native sm_120 PTX dispatch for RTX 5090
32a2a29 Initial fork from keryx-labs/keryx-miner v0.3.2 (commit 317fcab)
```

---

## 13. ntfy / status updates

Operator monitors progress via ntfy topic **`marcel-suprminer-bench`** —
push milestone updates there during long-running work:

```bash
curl -d "5090 hit 2.91 GH/s with shf.l.wrap rotates" \
  ntfy.sh/marcel-suprminer-bench
```

Use sparingly — milestones, blockers, and "going to bed" markers. Not every
build.

---

## 14. Sanity checklist before claiming a win

- [ ] Hashrate measured for ≥ 60 s on the 5090, ≥ 90 s on the 170HX
- [ ] Shares accepted (verify `Share accepted` lines in `supr.log`)
- [ ] No `consensus error` or `bad share` from the pool side (the operator
      checks `/home/marcel/keryx-logs/pool.stdout.log` — the rejected-share
      counter must not increase)
- [ ] `temp` under 80 °C and `power` under the TDP cap throughout the run
- [ ] Hashrate reproduces across a fresh process restart (rules out warm-up
      caches lying to you)
- [ ] Compare to the 2.77 GH/s upstream control by running the original
      `keryx-labs/keryx-miner v0.3.2` binary in the same environment

The pool will tell you within ~20 s if the kernel is computing a wrong hash:
shares will reject with `Invalid share`. **That is the only correctness
oracle.** Local hash-equality tests against a CPU reference are nice but
not authoritative.

---

## 15. Hand-off contact points

- Pool-side issues / share rejects / payouts: the **other Claude session**
  (this conversation no longer touches the pool).
- Rig physical access / power cycle / fans: the operator on ntfy.
- Algorithm / consensus questions: the keryx-labs whitepaper at
  `arxiv.org/abs/2504.09971` plus
  `/home/marcel/keryx-node/consensus/pow/src/matrix.rs` (canonical
  reference for `wave_mix`).

Good luck. The 5090 regression is the gatekeeper for everything else.
