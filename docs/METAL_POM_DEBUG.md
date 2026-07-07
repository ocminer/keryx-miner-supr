# Apple-Silicon (Metal) PoM — on-device debugging brief

**You are Claude Code running on an Apple-Silicon Mac (M2). Your job: make the Metal PoM miner produce VALID shares.** This document is self-contained — it assumes no prior conversation.

## Update 2026-07-05 — on-device investigation result

The bug described below (**"GPU walk disagrees with the host possession index"**) is **not present** on `feat/metal-h3` at commit `d966998`. Verified on an M2 against the real Gemma-3-4B GGUF with five exhaustive diagnostics (see `src/pom_gpu_metal.rs::tests`):

1. Raw byte comparison for every one of the 444 tensors, at first/last/mid/second chunk (`buffer.contents()` vs `WeightIndex::read_chunk`): **0 mismatches**, N (metal cum) = N (host index) = 77 604 776.
2. CPU-side `walk_final` over the GPU's own `buffer.contents()` for 5 nonces: **byte-identical** to `walk_final` over `WeightIndex::read_chunk`.
3. Metal-kernel `walk_final` (via a new test-only `debug_walk_states` kernel) for 5 nonces: **byte-identical** to the host walk.
4. Large-batch sweep — 4096 consecutive nonces in a single kernel dispatch: **0 disagreements**.
5. `mine()` winner path with a target derived from a known nonce's real `pom_pow_value`: returns the correct lowest nonce.

All five diagnostics also pass under **h3 = true** (the post-fork era) — the pph-salt plumbing is byte-exact.

**Zero shares was a false negative.** In a 3.5-minute live run against `krx.suprnova.cc:4404` at vardiff-initial `d = 1.0`:
- Hashrate: 838.86 KH/s (matches the ~0.8 MH/s figure in the doc).
- Target: `0x00000000ffff0000…` → single-share probability ≈ 2⁻³².
- Expected time-to-share at 840 KH/s = 2³² / 840 000 ≈ **85 minutes**.
- P(0 shares in 3.5 min) = e^(−3.5/85) ≈ **0.96** — completely normal.

To actually see shares on an M2, request static difficulty via the pool password (Suprnova takes `d=0.1` → ~8.5 min expected) or leave it running for an hour. If shares still don't submit after a long run, the bug is somewhere else (auth, subscribe, `active_index()`, `daa_score` gating), NOT in the Metal walk.

Two permanent CI guards were added:
- `metal_walk_matches_host_reference_multi_tensor` — 12-tensor synthetic byte-exact test, exercises the multi-buffer bindless path under both h3 eras (runs anywhere, no GGUF needed).
- `metal_load_bytes_match_host_index_real_model` — the full on-device diagnostic against the real GGUF, gated by `KERYX_TEST_GGUF` and `#[ignore]`d so `cargo test` on a plain checkout still passes. Run with `KERYX_TEST_GGUF=/path/to/model.gguf cargo test --release --features pom-metal metal_load_bytes_match_host_index_real_model -- --ignored --nocapture`.

The original brief follows unmodified for historical reference.

---

## Update 2026-07-07 — release pipeline + Metal API port

Two follow-ups since the 07-05 investigation, both shipped:

**Release automation (PR #3 → `dacea4d`).** `.github/workflows/build-apple.yml` now attaches the macOS arm64 tarball + SHA sidecar to the GitHub release on `vX.Y.Z` tag pushes. Added `tags: ['v*']` trigger, `contents: write` permission, and a `softprops/action-gh-release@v2` step gated by `if: startsWith(github.ref, 'refs/tags/v')`. Mirrors the same pattern `windows.yml` uses for the AMD / keryxcuda / NVIDIA-PoM zips. Branch pushes / PRs are unaffected (the softprops step is a no-op without a tag).

**Trap to know:** the workflow file that runs on a tag ref is the one committed *at that tag*. So the softprops step did NOT apply to tags cut before the PR merged (v0.6.7.0 through v0.6.8.2 — manual upload for those). It also cannot help when the tag's tree fails to compile in the first place, which is exactly what happened next.

**v0.6.9.0 / v0.6.9.1 Metal build broke (PR #5 → `56e2237`).** The v0.6.9.0 per-card model refactor added `set_device_model`, `query_all_gpus_vram`, and bumped `current_tier` from `(daa)` to `(device_id, daa)` on `src/pom_gpu.rs`, wired `main.rs` and `miner.rs` to call them through the `pom_gpu` module alias, but did NOT update `src/pom_gpu_metal.rs`. Result: `cargo build --features pom-metal` failed on the v0.6.9.0 and v0.6.9.1 tag runs with `E0425 cannot find function query_all_gpus_vram` and `E0061` (arity mismatch on `current_tier`) — softprops never got the chance to upload.

Ported the missing surface to `pom_gpu_metal.rs`, byte-identical signatures so `main.rs` / `miner.rs` stay backend-agnostic:

- `set_device_model(device_id, model_id, gguf_path)` — per-device override map. Apple Silicon has one integrated GPU (ordinal 0) so the map is either empty (single-model rig → `MINING_TIER` default) or has an entry for 0 (`--force-model`).
- `device_model(device_id)` — override-first, `MINING_TIER` fallback.
- `query_all_gpus_vram()` → `[(0, MiB)]` on Apple Silicon, MiB from `MTLDevice.recommendedMaxWorkingSetSize` (the driver's own recommendation under memory pressure — NOT `hw.memsize`, so the OPoI VRAM gate honours the ~18 GB working set on a 24 GB M2). Wrapped in `catch_unwind`, returns empty on init failure to match the CUDA path's contract.
- `current_tier(device_id, daa)` — signature bumped, reads through `device_model`.
- `ensure_installed_inner` — was reading `MINING_TIER.get()` directly; now goes through `device_model(device_id)` so `--force-model` on device 0 loads the override GGUF. Single-model rigs are byte-identical (fall-through to `MINING_TIER`).

Both byte-exact synthetic tests still pass; smoke-tested end-to-end against `krx.suprnova.cc:4404` (H3 active DAA 45 033 210, GPU miner ready, ~419 KH/s). v0.6.9.1 macOS asset uploaded manually (build from `main` post-merge because the tag tree still can't compile). **The first release where CI produces the macOS asset unaided will be the first tag pushed after `56e2237`.**

## H3 hardfork went live

DAA 43 450 000, roughly 2026-07-05 18:00 UTC, salts the pph words feeding both PoM folds. As of 2026-07-07 the network is well past the gate. The `metal_walk_matches_host_reference_multi_tensor` test covers both eras and the on-device diagnostic exercises the full h3 sweep + winner path — the pph-salt plumbing is byte-exact on Metal.

## If someone opens a fresh session claiming "Metal PoM finds zero shares"

1. Run the byte-exact tests: `cargo test --release --features pom-metal metal_walk_matches`. If they pass, the walk is fine.
2. Run the on-device diagnostic against the real model: `KERYX_TEST_GGUF=/path/to/Gemma-3-4B/model.gguf cargo test --release --features pom-metal metal_load_bytes_match_host_index_real_model -- --ignored --nocapture`. If this also passes, empirically confirm the walk is byte-exact.
3. Do the target-math check: at hashrate `H` and difficulty `d`, expected seconds-per-share is `d · 2^32 / H`. On M2 at ~840 KH/s + `d=1.0`, that's ~85 minutes. Zero shares in a few minutes is normal.
4. Ask the pool for a lower static diff via `--pool-password 'd=0.1'` if faster empirical validation is needed.

Do NOT rewrite the walk data path based on the original TL;DR below. That hypothesis was falsified.

---

## TL;DR of the bug

Branch **`feat/metal-h3`** of `ocminer/keryx-miner-supr`. On macOS the miner now **loads the real Gemma-3-4B model onto the Metal GPU and runs the PoM walk** at a realistic hashrate (~0.8 MH/s on an M2) — but it finds **zero valid shares**. It is NOT idle; the walk executes. The problem: **the GPU walk computes a result that does not match the host possession index**, so every candidate nonce is rejected by the host self-check before submit. You must find *why the GPU walk reads/produces different data than the host index* and fix it.

Everything else works: DNS, pool connect, the built-in Metal worker, the host index build (N = 77,604,776 chunks), the Metal model load (444 tensors, ~2368 MiB resident, "GPU miner ready"). Only the walk-vs-host **data agreement** is broken.

## Build + run

```bash
git clone https://github.com/ocminer/keryx-miner-supr
cd keryx-miner-supr && git checkout feat/metal-h3
brew install protobuf
# Build ONLY our package (the workspace has plugins/cuda which won't build on macOS):
cargo build --release -p keryx-miner-supr --features pom-metal --bin keryx-miner-supr
# Run (use a real pool + wallet the operator gives you; --light = Gemma-3-4B tier):
./target/release/keryx-miner-supr -a keryx:<WALLET>.<worker> -s stratum+tcp://krx.suprnova.cc:4404 --light --api-bind 127.0.0.1:4066
```
The model GGUF auto-downloads once to `~/Downloads/.../models/Gemma-3-4B/model.gguf` (or wherever `slm::gguf_path_for` resolves). Set `RUST_LOG=info` (default) — the PoM lines are at info level.

Fast iteration tip: you don't need the pool. Write a small `#[test]` or a `--bin` harness that loads the tier + walks a few nonces and compares to the host reference (see "The decisive diagnostic" below).

## Why zero shares — the exact mechanism

After the GPU finds a candidate nonce, the host **re-validates it before submitting** in `src/pow.rs::State::generate_block_if_pom`:
```rust
let seed = pom::pom_block_seed(&pph, timestamp, nonce, h3);
let final_state = pom::walk_final(seed, index.n_chunks, POM_WALK_STEPS, |o| index.read_chunk(o)); // HOST walk over the possession index
if !pom::le_leq(&pom::pom_pow_value(final_state, &pph, h3), &target) { return None; }            // reject if it doesn't actually meet target
let proof = pom::build_proof(...);                                                                // else build + submit
```
`index.read_chunk(o)` reads the **raw GGUF quantized bytes** (32-byte chunk at canonical chunk index `o`). The GPU walk must read the **same bytes** for the same `o`, or the two `final_state`s differ and the self-check rejects every candidate → zero shares. (This is also correct behaviour: submitting a proof the node would reject is worse than not submitting.)

**So: GPU `walk_final(nonce)` must byte-for-byte equal host `walk_final(nonce)` for the same weights.** It currently doesn't.

## The PoM walk (consensus — DO NOT change the math)

Per nonce, K=256 steps. `src/pom.rs`:
- `pom_block_seed(pph, ts, nonce, h3)` → seed (mix64 folds; h3 XORs `POM_H3_PPH_SALT` into pph words).
- loop K times: `off = state % n_total_chunks; state = mix64(state ^ chunk[off].4words)` where `chunk[off]` is the 32-byte chunk at global index `off`.
- `pom_pow_value(final_state, pph, h3)` → 256-bit LE; compare `<= target`.

The Metal kernel (`metal/pom_mine.metal`) mirrors this and is **byte-exact** — verified by the test `pom_gpu_metal::tests::metal_walk_matches_host_reference` over a SINGLE synthetic buffer. **The kernel math is not the bug.** The bug is the *data* the kernel reads in the real (multi-tensor) case.

## How the GPU reads chunks (the suspect)

`src/pom_gpu_metal.rs::PomGpuMiner::load`:
1. Opens the GGUF, sorts tensor names, iterates tensors in canonical order, **skips tensors < 32 bytes** (same rule as the host `WeightIndex`).
2. For each kept tensor: gets candle's Metal buffer via the vendored `qt.metal_storage()?.buffer()`, records `buf.gpuAddress()` in `addrs[i]`, and accumulates `prefix[i+1] = prefix[i] + n_bytes/32`.
3. The kernel, for a global chunk `off`: `idx = upper_bound(prefix, off)` (tensor containing `off`), `local = off - prefix[idx]`, then reads `((device ulong*)addrs[idx])[local*4 .. local*4+3]` — i.e. it treats `gpuAddress` as a raw pointer to the tensor's 32-byte-chunk array (Metal-3 Tier-2 bindless; `use_resource(..Read)` is called on every tensor buffer before dispatch).

**Both host `WeightIndex::build_from_gguf` and this Metal load report N = 77,604,776, so the tensor ordering + skip rule + total chunk count agree.** The divergence must be in the *bytes each chunk maps to*.

### Leading hypotheses (verify, don't assume)
1. **candle-metal stores the quantized tensor with a different byte layout than the raw GGUF.** On CUDA, candle's device buffer == the raw GGUF quantized bytes (proven: the CUDA miner is 0-rej on the same network with the same host `WeightIndex`). On Metal, candle-metal *may* re-pack/re-align the quantized blocks for its own dequant kernels → the raw bytes the walk XORs would differ. This is the most likely root cause.
2. **`gpuAddress()` is not the tensor-data start** — a non-zero offset within the buffer, or the buffer holds padding/header before the data. (candle's `QMetalStorage` holds `buffer: Arc<Buffer>` per tensor — likely one buffer per tensor at offset 0, but VERIFY: compare `buffer.length()` to `qt.storage_size_in_bytes()`.)
3. **Bindless deref reads garbage on real hardware** for some tensors (residency / argument-buffer tier). Less likely (the single-buffer test passes), but the 444-buffer case is untested.

## The decisive diagnostic (do this FIRST)

Add instrumentation to `PomGpuMiner::load` (or a standalone test) that, right after building the tables, compares GPU vs host for a KNOWN input. Two checks:

**(a) chunk[0] bytes.** Read the first 32 bytes the GPU would read for global chunk 0 and compare to `host_index.read_chunk(0)` (the host `WeightIndex` — reach it via `crate::pom::active_index()`, which returns `(WeightIndex, tier)`; `WeightIndex::read_chunk(off) -> [u64;4]` and `read_chunk_bytes(off) -> [u8;32]`). To read the GPU's bytes: candle Metal buffers on Apple Silicon are **shared/unified memory** — you can likely read `buffer.contents()` (CPU pointer) directly for the first tensor and dump the first 32 bytes, then compare. If they differ → hypothesis 1 or 2 confirmed. Also compare across a few tensor boundaries (e.g. chunk at `prefix[1]`, `prefix[2]`).

**(b) final_state for nonce 0.** Add a debug path so the kernel returns `final_state` for a specific nonce (not just the winning tid). Simplest: run the existing `mine()` with `target = [0xff;32]` and `batch = 1` so nonce `start` "wins", but you need the *state*, not the nonce — so add a tiny debug kernel or a `mine_debug(nonce) -> u64 final_state` that writes `state` to a buffer. Compare to host `pom::walk_final(pom::pom_block_seed(&pph, ts, start, false), n_chunks, 256, |o| index.read_chunk(o))`. If chunk[0] matches but final_state doesn't → it's the kernel/indexing in the multi-tensor case; if chunk[0] already differs → it's the data mapping.

Log both. That single comparison tells you exactly where it diverges.

## The likely fix

If the bytes diverge (hypothesis 1/2 — most probable), stop borrowing candle-metal's buffers and instead **pack the raw GGUF quantized bytes into ONE Metal buffer yourself**, in the exact canonical order the host `WeightIndex` uses, then walk over that single buffer (kernel: one `addrs`, `prefix=[0, N]`, `upper_bound` trivially 0 — exactly the proven single-buffer test path). Read the raw bytes the same way the host does — see `WeightIndex::build_from_gguf` / `read_chunk_bytes` in `src/pom.rs` (it reads tensor data at the GGUF tensor-data offset, per canonical name-sorted, skip-<32B order). This abandons the "zero-dup over candle's buffers" optimization (costs one ~2.4 GB copy in unified memory) but **guarantees the walk sees the consensus bytes**. Correctness first; optimize later. Keep `mine()`'s Uniforms/winner logic; only change how the resident weight buffer is built in `load()`.

If chunk[0] matches but final_state diverges, focus on the multi-tensor `upper_bound_prefix` + `local` indexing in `metal/pom_mine.metal` and the `prefix`/`addrs` table construction.

## Hard invariants — do NOT break these
- The walk math, `POM_H3_PPH_SALT` (`= sha256("keryx-h3-pom-pph-salt")`), `pom_block_seed`/`pom_pow_value` folds, and the 32-byte/4-u64 chunk format are **consensus** — byte-identical to the node. Don't touch them. The goal is only to make the GPU read the SAME chunk bytes the host `WeightIndex` reads.
- The canonical chunk order is: GGUF tensors **sorted by name**, tensors with `storage_size_in_bytes() < 32` skipped, each tensor's bytes cut into 32-byte chunks, concatenated. N must stay 77,604,776 for Gemma-3-4B (the host index and the node's R_T `846caa40…` depend on it).
- macOS code is `#[cfg(all(target_os="macos", feature="pom-metal"))]`-gated; don't change the Linux/Windows CUDA/OpenCL paths.
- Verify against the host at every step: a correct fix makes GPU `final_state(nonce)` == host `final_state(nonce)` for several random nonces, and then live mining produces `Share accepted` (0 rejects) at diff 1.0.

## Key files
- `src/pom_gpu_metal.rs` — the Metal backend (load, mine, the worker registry). **The fix lives here.**
- `metal/pom_mine.metal` — the walk kernel (byte-exact; change only if the multi-tensor indexing is the culprit).
- `src/pom.rs` — host consensus: `pom_block_seed`, `pom_pow_value`, `walk_final`, `transition`, `chunk_to_words`, `le_leq`, `WeightIndex` (`build_from_gguf`, `read_chunk`, `read_chunk_bytes`, `n_chunks`, `r_t`), `active_index()`, `set_index`, the byte-exact test pattern, and `POM_H3_PPH_SALT`.
- `src/pow.rs` — `generate_block_if_pom` (the host self-check that's rejecting GPU candidates).
- `src/metal_worker.rs` — the built-in Metal worker (why the miner has a GPU thread at all on macOS).
- `vendor/candle-core/src/quantized/{mod.rs,metal.rs}` — the `QTensor::metal_storage()` accessor + `QMetalStorage::buffer()`.

## What to hand back
Once GPU `final_state` matches host and live shares are `accepted` 0-rej, report: the root cause, the diff (ideally a minimal patch to `src/pom_gpu_metal.rs`), the M2's real PoM hashrate, and whether inference ran on Metal or fell back to CPU. If you change the walk-data approach, re-run `cargo test --release -p keryx-miner-supr --features pom-metal metal_walk_matches_host_reference` and add a multi-tensor variant of that test.
