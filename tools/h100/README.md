# H100 PoM walk — optimization notes (branch `h100-pom-opt`)

Optimization study of the Proof-of-Model mining walk (`src/pom_mine.cu`, driven by `src/pom_gpu.rs`)
on NVIDIA H100 80GB HBM3 (sm_90, CUDA 12.8). Goal: extract the most nonces/s per card while keeping
the walk **byte-exact** (a nonce found on GPU must build a `PomProof` the node accepts — the seed /
gather / pow folds cannot change).

Microbench + validation harness: [`bench_pom.cu`](bench_pom.cu).
Build: `nvcc -O3 -arch=sm_90 -o bench_pom tools/h100/bench_pom.cu`.
Run:   `./bench_pom <variant> <nonces_log2> <K> <G> <T>` (variant 0 = the production segmented kernel).
It fills a 2.48 GB synthetic blob with a deterministic per-chunk function and validates every variant
byte-exact against a CPU reference walk before timing it.

## TL;DR

The walk is a **serial data-dependent pointer chase** — each of the K=256 steps reads one random 32 B
chunk whose address depends on the previous read. That makes it **memory-latency / bandwidth bound**,
not compute bound. On an H100 the kernel already runs at the hardware memory ceiling (**~125.6 MH/s
per card ≈ 2× an RTX 5090**), and every micro-optimization tried below is inert. The one real,
free win is **batch size**: `POM_BATCH` 2^20 → 2^22 (~+2% on the kernel, more live once the per-launch
host round-trip is amortized).

## Measurements (single H100, clean, 2^24 nonces, K=256, byte-exact)

| variant | what | MH/s | note |
|---|---|---|---|
| v0 | segmented + binary search (**production**) | 125.2 | baseline |
| v1 | contiguous T=1, scalar loads | 115.4 | *slower* — scalar 4×u64 |
| v2 | contiguous T=1, 128-bit vector loads | 125.6 | ≈ v0 |
| v4 | contiguous + compile-time-const N (fast div) | 125.6 | modulo already hidden |
| v5 | mask instead of modulo (non-exact probe) | 125.9 | **compute is 100% hidden** |
| v3 | contiguous + vec + ILP G=4 | 114.1 | ILP halves occupancy, no MLP gain |
| v6 | `__ldcs` streaming loads | 125.5 | over-fetch unchanged |
| v7/v8 | `__ldcg` / `__ldlu` | 95.7 | *worse* — lose the L2 reuse |

Nsight Compute on v2 @ 2^24: occupancy **98.6%** (already maxed), DRAM throughput **60.95%** of HBM3
peak, L2 sector hit rate 12.7%, 22 registers/thread. DRAM bytes read = 272.9 GB for 137 GB useful =
**2 sectors (64 B) fetched per 32 B gather** → a fixed ~2× hardware over-fetch (a random 32 B read
pulls a 64 B DRAM unit; the paired 32 B is a different random chunk and is wasted).

## Why the usual levers do nothing here

- **Collapse the T-segment binary search (T→1 contiguous blob).** No gain (v0 ≈ v2). The ~log2(T)
  branchy `prefix[]` lookups per step are fully hidden under memory latency — same reason the 64-bit
  `% n` is free (v4/v5). Not worth the extra 2.48 GB VRAM copy; **keep the segmented production kernel.**
- **ILP (G independent walks/thread).** DRAM stays at ~61% while occupancy drops (G=4 → 49%, 54
  regs/thread). At max occupancy the warps already provide all the memory-level parallelism the DRAM
  subsystem will accept for this access pattern; adding per-thread ILP just moves the parallelism, it
  doesn't raise the ceiling.
- **Cache hints.** `__ldcs` doesn't reduce the 64 B fetch; `__ldcg`/`__ldlu` bypass the L2 reuse that
  the 12.7% hit rate is quietly buying, so they're *slower*.
- **Faster modulo.** Removing it entirely (mask probe) gives nothing — pure confirmation it's hidden.

The 2× over-fetch is HBM3 hardware behaviour and can't be hinted down to 32 B; the walk must read
exactly the 32 B of chunk `off` (consensus), so the paired sector can't be made useful either.

## The one real win — batch size

The walk needs enough concurrent nonces to fill the memory pipeline *and* amortize the driver's
per-batch host round-trip (`clone_htod(winner)` → launch → `synchronize` → `clone_dtoh`). Sweep:

```
2^18: 119.0   2^20: 122.6   2^21: 124.4   2^22: 124.8   2^23: 125.1   2^24: 125.2  MH/s
```

`POM_BATCH` bumped **2^20 → 2^22** in `src/miner.rs`: ~all of the kernel throughput, ~33 ms/launch on
an H100 (job switching stays responsive), and the fixed host round-trip is now a smaller fraction of
each launch (so the *live* gain is larger than the +2% the isolated kernel shows).

## Bottom line

The H100 is an excellent PoM card — ~2× a 5090 — because PoM rewards HBM bandwidth/latency, which is
exactly the H100's strength. But the kernel is already at the memory ceiling: there is no kernel
transformation that beats HBM3 random-access physics here. Effort is better spent on batch/launch
efficiency (done) than on the kernel arithmetic.
