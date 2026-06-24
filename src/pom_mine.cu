// Keryx Proof-of-Model mining kernel — contiguous-blob variant (NVIDIA / cudarc).
// Per nonce: seed-fold + K data-dependent 32-byte reads over the resident weight blob +
// pow-fold + target check. The weights are one contiguous buffer of N*4 little-endian u64 in
// canonical chunk order (built host-side from pom::WeightIndex::read_chunk — the SAME bytes the
// proof side uses), so chunk `off` is weights[off*4 .. off*4+4]. mix64 / seed-fold / pow-fold /
// le_leq are byte-identical to src/pom.rs (consensus) and to the AMD pom_mine.cl.

#include <cstdint>

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

extern "C" __global__ void pom_mine(
    const unsigned long long* weights, unsigned long long n_chunks, unsigned int K,
    const unsigned long long* pph, unsigned long long time_,
    const unsigned long long* target, unsigned long long nonce_base, unsigned long long n_nonces,
    unsigned long long* winner) {
    unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nonces) return;
    unsigned long long nonce = nonce_base + tid;

    unsigned long long p0 = pph[0], p1 = pph[1], p2 = pph[2], p3 = pph[3];

    // seed fold (pom_block_seed): mix64(nonce ^ 0x4B65727978531) then fold time + pph words
    unsigned long long state = mix64(nonce ^ 0x4B65727978531ULL);
    state = mix64(state ^ time_);
    state = mix64(state ^ p0); state = mix64(state ^ p1);
    state = mix64(state ^ p2); state = mix64(state ^ p3);

    // K-step data-dependent walk over the contiguous weight blob
    unsigned long long off = state % n_chunks;
    for (unsigned int i = 0; i < K; i++) {
        unsigned long long base = off * 4ULL;
        unsigned long long h = state;
        h ^= weights[base]; h ^= weights[base + 1]; h ^= weights[base + 2]; h ^= weights[base + 3];
        state = mix64(h);
        off = state % n_chunks;
    }

    // pow fold (pom_pow_value): 4 salted mix64 rounds -> 256-bit LE value
    unsigned long long pv0 = mix64(state ^ p0 ^ 0x9E3779B97F4A7C15ULL);
    unsigned long long pv1 = mix64(pv0   ^ p1 ^ 0xC2B2AE3D27D4EB4FULL);
    unsigned long long pv2 = mix64(pv1   ^ p2 ^ 0x165667B19E3779F9ULL);
    unsigned long long pv3 = mix64(pv2   ^ p3 ^ 0xD6E8FEB86659FD93ULL);

    // le_leq(pv, target): 256-bit little-endian compare, word3 is the most significant
    unsigned long long t0 = target[0], t1 = target[1], t2 = target[2], t3 = target[3];
    bool le;
    if (pv3 != t3)      le = pv3 < t3;
    else if (pv2 != t2) le = pv2 < t2;
    else if (pv1 != t1) le = pv1 < t1;
    else                le = pv0 <= t0;

    if (le) atomicMin(winner, nonce);
}
