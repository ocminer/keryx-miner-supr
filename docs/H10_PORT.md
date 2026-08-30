# H10 one-way seed — finalize checklist (v0.12.0)

Status: **SCAFFOLD ONLY — seed formula is an UNVERIFIED PLACEHOLDER.** The miner refuses to grind
the H10 era while `pom::H10_SPEC_VERIFIED == false`, so this branch cannot mine rejected blocks.

## What is already wired (validated against the placeholder)
- DAA gate `pom::POM_H10_SEED_ACTIVATION_DAA = 87_440_000` (confirm vs node when released).
- Era dispatch: `pom::pom_block_seed_v4_era(pph, ts, nonce, h10)` (host) and
  `pom_mine.cu::pom_seed_fold_era(h10, ...)` (GPU), threaded through all four v4 kernels
  (`pom_mine_v4`, `_chase`, `_tc`, `_ncf`) via a new `h10` launch arg.
- `daa` threaded miner.rs → `pom_gpu::mine_v4` → per-launch `h10` flag.
- Hard guard: `mine_v4` refuses to launch H10-era jobs (returns None, logs once) until VERIFIED.
- Host↔GPU lockstep test `v4_h10_seed_host_gpu_lockstep` PASSES on the placeholder (wiring proven).

## To FINALIZE when the node ships the spec/binaries this afternoon
1. Read the node's new `pom_block_seed_v4` (consensus/core/src/pom.rs) — the exact one-way fold:
   hash primitive, input byte order/endianness, which pph words (raw vs v4-salted), truncation.
2. Replace the body of BOTH mirrors, byte-identically:
   - `src/pom.rs::pom_seed_fold_v4_h10`
   - `src/pom_mine.cu::pom_seed_fold_h10`
3. Confirm the gate constant `POM_H10_SEED_ACTIVATION_DAA` == the node's.
4. Add the node's official golden vectors to `pom::tests::h10_seed_golden` and make it pass.
5. Regenerate `cuda/pom_mine.fatbin` (all 6 arches) and run `v4_h10_seed_host_gpu_lockstep`.
6. Flip `pom::H10_SPEC_VERIFIED = true`.
7. Live-verify: mine one H10-era block accepted by the pool BEFORE the fleet-wide roll.
8. OpenCL (AMD claude) + Metal mirrors get the same fold + gate.
9. Version bump to 0.12.0 (operator: consensus fork = middle-digit bump, the one exception to
   last-digit-only); release train.
