# H10 one-way seed — DONE (v0.12.0)

Status: **SHIPPED & GOLDEN-VERIFIED.** H10 seed = leading 64 bits of PowHash(pre_pow_hash,
timestamp, nonce) = cSHAKE256("ProofOfWorkHash"), ported byte-for-byte from node v1.5.7 +
miner-upstream v0.5.3. Gate DAA 87_360_000 (the tagged node value). H10_SPEC_VERIFIED = true.

Verified: host golden vectors (0x1fadaf72b089e024 / 0xcec7e2d9fce5bda6 / 0x60977326f8e922ab);
GPU standalone keccak (cuda/tests/seed_h10_check.cu) reproduces all three; host↔GPU lockstep
through the production kernels; pre-H10 walk still bit-exact. Uses RAW pph words (not v4-salted).

Files: src/keccak.rs (host f1600, links existing asm), src/pom.rs (POW_HASH_INITIAL_STATE,
pom_seed_h10_state, pom_block_seed_h10, era dispatch, gate, golden test), src/pom_mine.cu
(keccak_f1600 + pom_seed_fold_h10 device), src/pom_gpu.rs (raw pph words when h10_era).
OpenCL (AMD claude) + Metal still need the same fold + gate.
