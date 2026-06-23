// Keryx Proof-of-Model (PoM) mining kernel — OpenCL port of cuda/pom_mine.cu.
//
// Per nonce: seed-fold -> K=256 data-dependent 32B reads over the resident weight
// blob -> pow-fold -> target check. Only mix64 + memory reads (light -> high
// hashrate; the memory-hardness is the K non-prefetchable reads).
//
// BYTE-IDENTICAL to the consensus verifier `keryx-node::consensus/core/src/pom.rs`
// (mix64, pom_block_seed, transition, pom_pow_value) and to `cuda/pom_mine.cu`.
// Verified against upstream source 2026-06-23. The seed/pow folds MUST match the
// host proof builder (src/pom.rs build_proof) exactly or blocks are rejected.
//
// AMD difference vs CUDA: CUDA gathers through an array of per-tensor device
// pointers (bases[]+prefix[] binary search) because candle hands it per-tensor
// VRAM. OpenCL can't deref device pointers across buffers, so we load the whole
// tier into ONE contiguous buffer (`weights`, N*4 little-endian u64 in canonical
// chunk order). AMD CL_DEVICE_MAX_MEM_ALLOC_SIZE is 13.59 GiB here -> covers
// tier 0 (2.48 GiB) and tier 1 (4.9 GiB). The kernel is thus simpler than CUDA's.

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

typedef ulong u64;

// SplitMix64 finalizer (pom.rs:117-124). All `*` are wrapping (ulong overflow wraps).
inline u64 pom_mix64(u64 x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

// state[0] = pom_block_seed (pom.rs:143-152). NB: used DIRECTLY as the walk start —
// production passes this raw to the verifier (body_validation_in_isolation.rs:157);
// the `pom_seed_state`/POM_SEED_SALT in pom.rs is test-only, do NOT apply it.
inline u64 pom_seed_fold(u64 nonce, u64 time_, u64 p0, u64 p1, u64 p2, u64 p3) {
    u64 s = pom_mix64(nonce ^ 0x4B65727978531UL);
    s = pom_mix64(s ^ time_);
    s = pom_mix64(s ^ p0); s = pom_mix64(s ^ p1); s = pom_mix64(s ^ p2); s = pom_mix64(s ^ p3);
    return s;
}

// pow_value = pom_pow_value (pom.rs:157-169) — the 4-round golden-salt fold the node
// passes as `final_hash` (body_validation_in_isolation.rs:170). NOT kHeavyHash.
inline void pom_pow_fold(u64 fin, u64 p0, u64 p1, u64 p2, u64 p3, u64 out[4]) {
    out[0] = pom_mix64(fin    ^ p0 ^ 0x9E3779B97F4A7C15UL);
    out[1] = pom_mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FUL);
    out[2] = pom_mix64(out[1] ^ p2 ^ 0x165667B19E3779F9UL);
    out[3] = pom_mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93UL);
}

// 256-bit little-endian `a <= b` (pom.rs le_leq): word 3 is the most-significant.
inline bool pom_le_leq(const u64 a[4], u64 b0, u64 b1, u64 b2, u64 b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

// One nonce per work-item. `weights` = the tier blob (N chunks * 4 LE u64), contiguous
// in canonical chunk order (R_T is built over exactly this order). `winner` is a single
// u64 pre-set to U64_MAX by the host; lowest passing nonce wins (host re-verifies).
__kernel void pom_mine(
    __global const u64* restrict weights,
    const u64 n_total_chunks,
    const uint K,
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,   // pre_pow_hash as 4 LE u64
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,   // target as 4 LE u64
    const u64 nonce_base, const u64 n_nonces,
    volatile __global u64* winner)
{
    u64 tid = get_global_id(0);
    if (tid >= n_nonces) return;
    u64 nonce = nonce_base + tid;

    u64 state = pom_seed_fold(nonce, time_, p0, p1, p2, p3);
    u64 off = state % n_total_chunks;
    for (uint i = 0; i < K; i++) {
        u64 base = off * 4UL;
        u64 h = state ^ weights[base] ^ weights[base + 1] ^ weights[base + 2] ^ weights[base + 3];
        state = pom_mix64(h);
        off = state % n_total_chunks;
    }

    u64 pv[4];
    pom_pow_fold(state, p0, p1, p2, p3, pv);
    if (pom_le_leq(pv, t0, t1, t2, t3)) {
        // atomic-min via CAS loop — needs only cl_khr_int64_base_atomics (atom_min for
        // 64-bit is the *extended* ext, not always present on AMD). winner starts U64_MAX.
        u64 old = *winner;
        while (nonce < old) {
            u64 prev = atom_cmpxchg(winner, old, nonce);
            if (prev == old) break;
            old = prev;
        }
    }
}
