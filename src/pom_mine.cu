// Keryx Proof-of-Model mining kernel (design A), loaded into candle's CUDA context.
// Per nonce: seed-fold + data-dependent gather walk over the resident weight blob +
// pow-fold + target check. Only mix64 + memory reads (light → high hashrate). Mirrors
// pom-q4-probe::pom_mine. The seed/pow folds MUST match the host (src/pom.rs build_proof).

#include <cstdint>

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ __forceinline__ unsigned long long pom_seed_fold(
    unsigned long long nonce, unsigned long long time_,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3) {
    unsigned long long s = mix64(nonce ^ 0x4B65727978531ULL);
    s = mix64(s ^ time_);
    s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}

__device__ __forceinline__ void pom_pow_fold(
    unsigned long long fin, unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
    unsigned long long out[4]) {
    out[0] = mix64(fin ^ p0 ^ 0x9E3779B97F4A7C15ULL);
    out[1] = mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FULL);
    out[2] = mix64(out[1] ^ p2 ^ 0x165667B19E3779F9ULL);
    out[3] = mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93ULL);
}

__device__ __forceinline__ bool pom_le_leq(const unsigned long long a[4],
                                           unsigned long long b0, unsigned long long b1,
                                           unsigned long long b2, unsigned long long b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

extern "C" __global__ void pom_mine(const unsigned long long* bases, const unsigned long long* prefix,
                                    unsigned int T, unsigned long long n_total_chunks, unsigned int K,
                                    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
                                    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
                                    unsigned long long time_,
                                    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
                                    unsigned long long nonce_base, unsigned long long n_nonces,
                                    unsigned long long* winner, unsigned int walk_v2) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nonces) return;
    unsigned long long nonce = nonce_base + tid;

    // H5.1: the SEED fold reads the (host-salted) seed words s0..s3; the pow fold below keeps p0..p3.
    // Pre-H5.1 the host passes s == p, so this stays byte-identical to the H5 build.
    unsigned long long state = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    unsigned long long off = state % n_total_chunks;
    for (unsigned int i = 0; i < K; i++) {
        unsigned int lo = 0, hi = T;
        while (lo + 1 < hi) {
            unsigned int mid = (lo + hi) >> 1;
            if (prefix[mid] <= off) lo = mid; else hi = mid;
        }
        unsigned long long local = off - prefix[lo];
        // 128-bit vector loads: read the 32-byte chunk as 2x ulonglong2 instead of 4x u64. Same
        // bytes XOR'd (a.x^a.y^c.x^c.y == the 4 u64), so BYTE-EXACT, but half the load instructions
        // and better coalescing → +1.8% on H200/Hopper (byte-exact validated), neutral+ elsewhere.
        const ulonglong2* p = (const ulonglong2*)bases[lo];
        unsigned long long base2 = local * 2ULL;
        ulonglong2 a = p[base2], c = p[base2 + 1];
        unsigned long long h = state;
        if (walk_v2) {
            // H5 non-foldable walk: chain mix64 through each of the 4 chunk words (order a.x,a.y,c.x,c.y
            // = w0..w3) so all 32 bytes are load-bearing. walk_v2 is uniform -> no warp divergence.
            h = mix64(h ^ a.x); h = mix64(h ^ a.y); h = mix64(h ^ c.x); h = mix64(h ^ c.y);
            state = h;
        } else {
            // Pre-H5 fold (frozen — validates all blocks below H5_ACTIVATION_DAA).
            h ^= a.x; h ^= a.y; h ^= c.x; h ^= c.y;
            state = mix64(h);
        }
        off = state % n_total_chunks;
    }
    unsigned long long pv[4];
    pom_pow_fold(state, p0, p1, p2, p3, pv);
    if (pom_le_leq(pv, t0, t1, t2, t3)) {
        atomicMin(winner, nonce);
    }
}

// ILP x2 variant: each thread grinds TWO nonces with their walk steps interleaved, so the second
// walk's search/mix work overlaps the first walk's DRAM fetch (both chunk loads issue back-to-back
// -> their latencies overlap). BYTE-EXACT per nonce — each walk's math is untouched, only the
// scheduling interleaves. Helps latency-bound parts that DON'T already saturate their outstanding-
// miss slots at 1 nonce/thread (e.g. GDDR6X 3090). NEUTRAL-to-NEGATIVE where the slots are already
// full (measured: 5090/GDDR7 -6%, 3070/GDDR6 ~0). The miner autotunes ILP1-vs-ILP2 per device
// (pom_gpu::autotune_block) and only uses this where it actually wins — never a forced default.
// Host launches ceil(batch/2) threads (grid in pom_gpu.rs). Identical signature to pom_mine.
extern "C" __global__ void pom_mine_ilp2(const unsigned long long* bases, const unsigned long long* prefix,
                                    unsigned int T, unsigned long long n_total_chunks, unsigned int K,
                                    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
                                    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
                                    unsigned long long time_,
                                    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
                                    unsigned long long nonce_base, unsigned long long n_nonces,
                                    unsigned long long* winner, unsigned int walk_v2) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long i0 = tid * 2ULL;
    if (i0 >= n_nonces) return;
    unsigned long long i1 = i0 + 1ULL;
    bool has1 = i1 < n_nonces;
    unsigned long long nonce0 = nonce_base + i0;
    // Odd-batch boundary thread: walk nonce0 twice (the duplicate result is dropped below) rather
    // than branching the whole body on has1 — keeps both lanes busy, byte-exact for nonce0.
    unsigned long long nonce1 = nonce_base + (has1 ? i1 : i0);

    unsigned long long state0 = pom_seed_fold(nonce0, time_, s0, s1, s2, s3);
    unsigned long long state1 = pom_seed_fold(nonce1, time_, s0, s1, s2, s3);
    unsigned long long off0 = state0 % n_total_chunks;
    unsigned long long off1 = state1 % n_total_chunks;
    for (unsigned int i = 0; i < K; i++) {
        unsigned int lo0 = 0, hi0 = T;
        while (lo0 + 1 < hi0) { unsigned int mid = (lo0 + hi0) >> 1; if (prefix[mid] <= off0) lo0 = mid; else hi0 = mid; }
        unsigned int lo1 = 0, hi1 = T;
        while (lo1 + 1 < hi1) { unsigned int mid = (lo1 + hi1) >> 1; if (prefix[mid] <= off1) lo1 = mid; else hi1 = mid; }
        const ulonglong2* q0 = (const ulonglong2*)bases[lo0];
        const ulonglong2* q1 = (const ulonglong2*)bases[lo1];
        unsigned long long b0 = (off0 - prefix[lo0]) * 2ULL;
        unsigned long long b1 = (off1 - prefix[lo1]) * 2ULL;
        ulonglong2 a0 = q0[b0], c0 = q0[b0 + 1]; // both fetches issue back-to-back ->
        ulonglong2 a1 = q1[b1], c1 = q1[b1 + 1]; // their DRAM latencies overlap
        unsigned long long h0 = state0, h1 = state1;
        if (walk_v2) {
            h0 = mix64(h0 ^ a0.x); h1 = mix64(h1 ^ a1.x);
            h0 = mix64(h0 ^ a0.y); h1 = mix64(h1 ^ a1.y);
            h0 = mix64(h0 ^ c0.x); h1 = mix64(h1 ^ c1.x);
            h0 = mix64(h0 ^ c0.y); h1 = mix64(h1 ^ c1.y);
            state0 = h0; state1 = h1;
        } else {
            h0 ^= a0.x; h0 ^= a0.y; h0 ^= c0.x; h0 ^= c0.y; state0 = mix64(h0);
            h1 ^= a1.x; h1 ^= a1.y; h1 ^= c1.x; h1 ^= c1.y; state1 = mix64(h1);
        }
        off0 = state0 % n_total_chunks;
        off1 = state1 % n_total_chunks;
    }
    unsigned long long pv0[4]; pom_pow_fold(state0, p0, p1, p2, p3, pv0);
    if (pom_le_leq(pv0, t0, t1, t2, t3)) atomicMin(winner, nonce0);
    if (has1) {
        unsigned long long pv1[4]; pom_pow_fold(state1, p0, p1, p2, p3, pv1);
        if (pom_le_leq(pv1, t0, t1, t2, t3)) atomicMin(winner, nonce1);
    }
}
