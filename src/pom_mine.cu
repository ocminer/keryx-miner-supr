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
                                    unsigned long long time_,
                                    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
                                    unsigned long long nonce_base, unsigned long long n_nonces,
                                    unsigned long long* winner) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nonces) return;
    unsigned long long nonce = nonce_base + tid;

    unsigned long long state = pom_seed_fold(nonce, time_, p0, p1, p2, p3);
    unsigned long long off = state % n_total_chunks;
    for (unsigned int i = 0; i < K; i++) {
        unsigned int lo = 0, hi = T;
        while (lo + 1 < hi) {
            unsigned int mid = (lo + hi) >> 1;
            if (prefix[mid] <= off) lo = mid; else hi = mid;
        }
        unsigned long long local = off - prefix[lo];
        const unsigned long long* p = (const unsigned long long*)bases[lo];
        unsigned long long base = local * 4ULL;
        unsigned long long h = state;
        h ^= p[base]; h ^= p[base + 1]; h ^= p[base + 2]; h ^= p[base + 3];
        state = mix64(h);
        off = state % n_total_chunks;
    }
    unsigned long long pv[4];
    pom_pow_fold(state, p0, p1, p2, p3, pv);
    if (pom_le_leq(pv, t0, t1, t2, t3)) {
        atomicMin(winner, nonce);
    }
}
