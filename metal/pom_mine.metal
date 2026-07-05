// Keryx Proof-of-Model mining kernel (Metal, Apple Silicon port).
//
// Port of cuda/pom_mine.cu. Per nonce: seed-fold + data-dependent gather walk over the
// resident (zero-dup) weight blob + pow-fold + target check. The seed/pow folds are
// byte-identical to `pom_mine.cu::pom_seed_fold`/`pom_pow_fold` and the host
// `pom::pom_block_seed`/`pom::pom_pow_value`, so nonces mined here build proofs the
// node accepts.
//
// Layout — bindless walk over candle's own MTLBuffers (Metal 3 / MSL 3, Apple Silicon):
//
//   * `prefix[i]` is the cumulative chunk count of tensors [0..i); it has length
//     n_tensors + 1 so the terminating sentinel is `prefix[n_tensors] == n_total_chunks`.
//     A chunk is 32 bytes (four ulongs) — the same slicing the CUDA path uses.
//   * `tensor_addrs[i]` is the raw GPU address of tensor i's MTLBuffer, i.e.
//     `[buffer gpuAddress]`. In Metal 3 argument-buffer Tier-2 (guaranteed on Apple
//     Silicon) these are plain 64-bit pointers we can reinterpret to
//     `device const ulong*`. The host calls `use_resource` on each tensor buffer at
//     dispatch time so the driver keeps them resident even though nothing else binds them.
//   * `winner` is an `atomic_uint` holding the winning tid (0..n_nonces); Metal does not
//     guarantee `atomic_ulong`. Host reconstructs the nonce as `nonce_base + tid`. Since
//     POM_BATCH is 1<<20 and always < 2^32, this is byte-identical to CUDA's
//     `atomicMin(winner, nonce)`.

#include <metal_stdlib>
using namespace metal;

struct PomUniforms {
    ulong  n_total_chunks;
    uint   k_steps;
    uint   n_tensors;
    ulong  p0; ulong p1; ulong p2; ulong p3;
    ulong  time_;
    ulong  t0; ulong t1; ulong t2; ulong t3;
    ulong  nonce_base;
    uint   n_nonces;
    uint   _pad;
};

inline ulong mix64(ulong x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

inline ulong pom_seed_fold(ulong nonce, ulong time_,
                           ulong p0, ulong p1, ulong p2, ulong p3) {
    ulong s = mix64(nonce ^ 0x4B65727978531UL);
    s = mix64(s ^ time_);
    s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}

inline void pom_pow_fold(ulong fin, ulong p0, ulong p1, ulong p2, ulong p3,
                         thread ulong* out) {
    out[0] = mix64(fin    ^ p0 ^ 0x9E3779B97F4A7C15UL);
    out[1] = mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FUL);
    out[2] = mix64(out[1] ^ p2 ^ 0x165667B19E3779F9UL);
    out[3] = mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93UL);
}

inline bool pom_le_leq(thread const ulong* a,
                       ulong b0, ulong b1, ulong b2, ulong b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

// Binary search: largest i in [0, n_tensors] such that prefix[i] <= off.
// Mirrors the `bases[]` upper_bound in cuda/pom_mine.cu.
inline uint upper_bound_prefix(device const ulong* prefix, uint n_tensors, ulong off) {
    uint lo = 0;
    uint hi = n_tensors; // sentinel prefix[n_tensors] == n_total_chunks > off
    while (lo + 1 < hi) {
        uint mid = (lo + hi) >> 1;
        if (prefix[mid] <= off) { lo = mid; } else { hi = mid; }
    }
    return lo;
}

kernel void pom_mine(
    device   const ulong*        prefix       [[buffer(0)]],  // n_tensors + 1 entries, in chunks
    device   const ulong*        tensor_addrs [[buffer(1)]],  // n_tensors gpu addresses
    constant const PomUniforms&  u            [[buffer(2)]],
    device   atomic_uint*        winner       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= u.n_nonces) return;
    ulong nonce = u.nonce_base + (ulong)tid;

    ulong state = pom_seed_fold(nonce, u.time_, u.p0, u.p1, u.p2, u.p3);
    ulong off = state % u.n_total_chunks;
    for (uint i = 0; i < u.k_steps; i++) {
        uint idx = upper_bound_prefix(prefix, u.n_tensors, off);
        ulong local = off - prefix[idx];
        // Bindless deref (MSL 3 / Apple Silicon Tier-2 argbufs): the u64 in
        // tensor_addrs[] is a raw GPU pointer; reinterpret it as a device pointer.
        device const ulong* ptr =
            reinterpret_cast<device const ulong*>(tensor_addrs[idx]);
        ulong base = local * 4UL;
        ulong h = state;
        h ^= ptr[base + 0];
        h ^= ptr[base + 1];
        h ^= ptr[base + 2];
        h ^= ptr[base + 3];
        state = mix64(h);
        off = state % u.n_total_chunks;
    }
    ulong pv[4];
    pom_pow_fold(state, u.p0, u.p1, u.p2, u.p3, pv);
    if (pom_le_leq(pv, u.t0, u.t1, u.t2, u.t3)) {
        atomic_fetch_min_explicit(winner, tid, memory_order_relaxed);
    }
}
