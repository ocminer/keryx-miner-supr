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

// The per-step `state % n_chunks` is a 64-bit division with a RUNTIME divisor — a slow ALU library
// call (hundreds of cycles), issued 256×/nonce. The host JIT-compiles this source per tier, so it
// bakes the tier's chunk count in as -D POM_NC=<n>UL: the compiler then emits the exact
// Granlund-Montgomery multiply-high sequence for the constant divisor (byte-exact by construction —
// it's the compiler's own proven strength reduction, same result for every input). The kernel arg
// stays in the signature; without the define we fall back to the runtime divisor.
#ifdef POM_NC
#define POM_N(nc_arg) ((u64)(POM_NC))
#else
#define POM_N(nc_arg) (nc_arg)
#endif

// Slab-split fetch: Polaris-class OpenCL stacks (ORCA) enforce a hard per-buffer limit (~4 GiB)
// REGARDLESS of the reported CL_DEVICE_MAX_MEM_ALLOC_SIZE / GPU_SINGLE_ALLOC_PERCENT — a single
// 4.6 GiB post-H5 blob fails with CL_MEM_OBJECT_ALLOCATION_FAILURE (RX 580 8 GB field report).
// The host therefore splits the blob into up to 4 slabs of 2^slab_shift chunks each and passes
// them as c0..c3. Chunk off lives in slab off>>slab_shift at index off&mask. Single-slab rigs
// (the common case) pass slab_shift=63 -> s==0 always, i==off: the address math degenerates to
// the old one-buffer layout. BYTE-EXACT either way — only WHERE a chunk lives changes, never its
// bytes, and the walk consumes chunks by canonical index.
inline ulong4 pom_fetch(const __global ulong4* restrict c0, const __global ulong4* restrict c1,
                        const __global ulong4* restrict c2, const __global ulong4* restrict c3,
                        u64 off, uint slab_shift) {
    u64 s = off >> slab_shift;
    u64 i = off & ((1UL << slab_shift) - 1UL);
    const __global ulong4* b = (s == 0UL) ? c0 : (s == 1UL) ? c1 : (s == 2UL) ? c2 : c3;
    return b[i];
}


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
    __global const ulong4* restrict w0,   // blob slab 0 (single-slab rigs: the whole blob; unused slabs = slab 0 repeated)
    __global const ulong4* restrict w1,
    __global const ulong4* restrict w2,
    __global const ulong4* restrict w3,
    const u64 n_total_chunks,
    const uint K,
    const uint slab_shift,                 // chunks per slab = 2^slab_shift; 63 = single-slab layout
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,   // POW-fold pph words (H3-salted), 4 LE u64
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,   // SEED-fold pph words (H5.1-salted at/after gate)
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,   // target as 4 LE u64
    const u64 nonce_base, const u64 n_nonces,
    volatile __global u64* winner,
    const uint walk_v2)   // H5 era flag: 0 = frozen v1 fold, 1 = non-foldable mix64-chain
{
    u64 tid = get_global_id(0);
    if (tid >= n_nonces) return;
    u64 nonce = nonce_base + tid;

    // H5.1: the SEED fold reads the (host-salted) seed words s0..s3; the pow fold below keeps p0..p3.
    // Pre-H5.1 the host passes s == p, so this is byte-identical to the H5 build.
    u64 state = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    u64 off = state % POM_N(n_total_chunks);
    for (uint i = 0; i < K; i++) {
        ulong4 w = pom_fetch(w0, w1, w2, w3, off, slab_shift);
        if (walk_v2) {
            // H5 non-foldable walk (at/after H5_ACTIVATION_DAA): chain mix64 through each of the 4
            // chunk words (w0..w3) so all 32 bytes are load-bearing and order-dependent — byte-exact
            // with pom.rs transition_v2 and pom_mine.cu. walk_v2 is uniform across work-items -> no
            // divergence.
            u64 h = pom_mix64(state ^ w.s0);
            h = pom_mix64(h ^ w.s1);
            h = pom_mix64(h ^ w.s2);
            h = pom_mix64(h ^ w.s3);
            state = h;
        } else {
            // Pre-H5 fold (frozen — validates every block below H5_ACTIVATION_DAA).
            u64 h = state ^ w.s0 ^ w.s1 ^ w.s2 ^ w.s3;
            state = pom_mix64(h);
        }
        off = state % POM_N(n_total_chunks);
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

// Atomic-min helper shared by the ILP variants (same CAS loop as above).
inline void pom_submit(volatile __global u64* winner, u64 nonce) {
    u64 old = *winner;
    while (nonce < old) {
        u64 prev = atom_cmpxchg(winner, old, nonce);
        if (prev == old) break;
        old = prev;
    }
}

// ILP x2: each work-item grinds TWO nonces with their walk steps interleaved — both chunk loads
// issue back-to-back so their DRAM latencies overlap. BYTE-EXACT per nonce (each walk's math is
// untouched; only the scheduling interleaves — mirror of cuda pom_mine_ilp2). Wins where a card
// doesn't already saturate its outstanding-miss slots at 1 nonce/lane (our MI50/MI60 runs at ~17%
// of HBM2 bandwidth = latency-bound with idle slots). The host autotunes ILP1/2/4 per device and
// only uses what actually measures faster. Host launches ceil(batch/2) items.
__kernel void pom_mine_ilp2(
    __global const ulong4* restrict w0,   // blob slab 0 (single-slab rigs: the whole blob; unused slabs = slab 0 repeated)
    __global const ulong4* restrict w1,
    __global const ulong4* restrict w2,
    __global const ulong4* restrict w3,
    const u64 n_total_chunks,
    const uint K,
    const uint slab_shift,                 // chunks per slab = 2^slab_shift; 63 = single-slab layout
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,
    const u64 nonce_base, const u64 n_nonces,
    volatile __global u64* winner,
    const uint walk_v2)
{
    u64 tid = get_global_id(0);
    u64 i0 = tid * 2UL;
    if (i0 >= n_nonces) return;
    u64 i1 = i0 + 1UL;
    bool has1 = i1 < n_nonces;
    u64 nonce0 = nonce_base + i0;
    // Odd-batch boundary item: walk nonce0 twice (duplicate result dropped below) instead of
    // branching the whole body — keeps the lane busy, byte-exact for nonce0 (CUDA-identical trick).
    u64 nonce1 = nonce_base + (has1 ? i1 : i0);

    u64 state0 = pom_seed_fold(nonce0, time_, s0, s1, s2, s3);
    u64 state1 = pom_seed_fold(nonce1, time_, s0, s1, s2, s3);
    u64 off0 = state0 % POM_N(n_total_chunks);
    u64 off1 = state1 % POM_N(n_total_chunks);
    for (uint i = 0; i < K; i++) {
        ulong4 a = pom_fetch(w0, w1, w2, w3, off0, slab_shift);   // both loads issue back-to-back ->
        ulong4 b = pom_fetch(w0, w1, w2, w3, off1, slab_shift);   // their DRAM latencies overlap
        if (walk_v2) {
            u64 h0 = pom_mix64(state0 ^ a.s0), h1 = pom_mix64(state1 ^ b.s0);
            h0 = pom_mix64(h0 ^ a.s1); h1 = pom_mix64(h1 ^ b.s1);
            h0 = pom_mix64(h0 ^ a.s2); h1 = pom_mix64(h1 ^ b.s2);
            h0 = pom_mix64(h0 ^ a.s3); h1 = pom_mix64(h1 ^ b.s3);
            state0 = h0; state1 = h1;
        } else {
            state0 = pom_mix64(state0 ^ a.s0 ^ a.s1 ^ a.s2 ^ a.s3);
            state1 = pom_mix64(state1 ^ b.s0 ^ b.s1 ^ b.s2 ^ b.s3);
        }
        off0 = state0 % POM_N(n_total_chunks);
        off1 = state1 % POM_N(n_total_chunks);
    }

    u64 pv[4];
    pom_pow_fold(state0, p0, p1, p2, p3, pv);
    if (pom_le_leq(pv, t0, t1, t2, t3)) pom_submit(winner, nonce0);
    if (has1) {
        pom_pow_fold(state1, p0, p1, p2, p3, pv);
        if (pom_le_leq(pv, t0, t1, t2, t3)) pom_submit(winner, nonce1);
    }
}

// ILP x4: four interleaved walks per work-item — four outstanding chunk loads per lane. More VGPRs
// per lane (fewer waves in flight), but on a latency-bound card with idle bandwidth the added
// memory-level parallelism can more than compensate. Autotuned; never a forced default.
// Host launches ceil(batch/4) items.
__kernel void pom_mine_ilp4(
    __global const ulong4* restrict w0,   // blob slab 0 (single-slab rigs: the whole blob; unused slabs = slab 0 repeated)
    __global const ulong4* restrict w1,
    __global const ulong4* restrict w2,
    __global const ulong4* restrict w3,
    const u64 n_total_chunks,
    const uint K,
    const uint slab_shift,                 // chunks per slab = 2^slab_shift; 63 = single-slab layout
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,
    const u64 nonce_base, const u64 n_nonces,
    volatile __global u64* winner,
    const uint walk_v2)
{
    u64 tid = get_global_id(0);
    u64 i0 = tid * 4UL;
    if (i0 >= n_nonces) return;

    // Boundary items re-walk nonce i0 in the missing lanes (duplicates dropped at submit).
    u64 idx1 = i0 + 1UL < n_nonces ? i0 + 1UL : i0;
    u64 idx2 = i0 + 2UL < n_nonces ? i0 + 2UL : i0;
    u64 idx3 = i0 + 3UL < n_nonces ? i0 + 3UL : i0;
    u64 n0 = nonce_base + i0, n1 = nonce_base + idx1, n2 = nonce_base + idx2, n3 = nonce_base + idx3;

    u64 st0 = pom_seed_fold(n0, time_, s0, s1, s2, s3);
    u64 st1 = pom_seed_fold(n1, time_, s0, s1, s2, s3);
    u64 st2 = pom_seed_fold(n2, time_, s0, s1, s2, s3);
    u64 st3 = pom_seed_fold(n3, time_, s0, s1, s2, s3);
    u64 of0 = st0 % POM_N(n_total_chunks), of1 = st1 % POM_N(n_total_chunks);
    u64 of2 = st2 % POM_N(n_total_chunks), of3 = st3 % POM_N(n_total_chunks);
    for (uint i = 0; i < K; i++) {
        ulong4 a = pom_fetch(w0, w1, w2, w3, of0, slab_shift);
        ulong4 b = pom_fetch(w0, w1, w2, w3, of1, slab_shift);
        ulong4 c = pom_fetch(w0, w1, w2, w3, of2, slab_shift);
        ulong4 d = pom_fetch(w0, w1, w2, w3, of3, slab_shift);
        if (walk_v2) {
            u64 h0 = pom_mix64(st0 ^ a.s0), h1 = pom_mix64(st1 ^ b.s0), h2 = pom_mix64(st2 ^ c.s0), h3 = pom_mix64(st3 ^ d.s0);
            h0 = pom_mix64(h0 ^ a.s1); h1 = pom_mix64(h1 ^ b.s1); h2 = pom_mix64(h2 ^ c.s1); h3 = pom_mix64(h3 ^ d.s1);
            h0 = pom_mix64(h0 ^ a.s2); h1 = pom_mix64(h1 ^ b.s2); h2 = pom_mix64(h2 ^ c.s2); h3 = pom_mix64(h3 ^ d.s2);
            h0 = pom_mix64(h0 ^ a.s3); h1 = pom_mix64(h1 ^ b.s3); h2 = pom_mix64(h2 ^ c.s3); h3 = pom_mix64(h3 ^ d.s3);
            st0 = h0; st1 = h1; st2 = h2; st3 = h3;
        } else {
            st0 = pom_mix64(st0 ^ a.s0 ^ a.s1 ^ a.s2 ^ a.s3);
            st1 = pom_mix64(st1 ^ b.s0 ^ b.s1 ^ b.s2 ^ b.s3);
            st2 = pom_mix64(st2 ^ c.s0 ^ c.s1 ^ c.s2 ^ c.s3);
            st3 = pom_mix64(st3 ^ d.s0 ^ d.s1 ^ d.s2 ^ d.s3);
        }
        of0 = st0 % POM_N(n_total_chunks); of1 = st1 % POM_N(n_total_chunks);
        of2 = st2 % POM_N(n_total_chunks); of3 = st3 % POM_N(n_total_chunks);
    }

    u64 pv[4];
    pom_pow_fold(st0, p0, p1, p2, p3, pv);
    if (pom_le_leq(pv, t0, t1, t2, t3)) pom_submit(winner, n0);
    if (idx1 != i0) { pom_pow_fold(st1, p0, p1, p2, p3, pv); if (pom_le_leq(pv, t0, t1, t2, t3)) pom_submit(winner, n1); }
    if (idx2 != i0) { pom_pow_fold(st2, p0, p1, p2, p3, pv); if (pom_le_leq(pv, t0, t1, t2, t3)) pom_submit(winner, n2); }
    if (idx3 != i0) { pom_pow_fold(st3, p0, p1, p2, p3, pv); if (pom_le_leq(pv, t0, t1, t2, t3)) pom_submit(winner, n3); }
}

// ============================ PoM v3 (H6 matrix-state walk) ============================
// Byte-exact OpenCL mirror of src/pom_v3.rs / src/pom_mine.cu (CUDA pom_mine_v3). One WORK-GROUP
// per nonce; local_size = 256 = D; work-item x owns state row x (256 int8 = 64 packed uint). The
// 64 KB tile lives in __local (LDS). Blob = single contiguous uint buffer (chunk idx = 8 u32 at
// blob32[idx*8]). Consensus-critical: MUST match the host walk / node verifier bit-for-bit.

#define V3_D    256
#define V3_D4   64          // D/4 (packed uints per row)
#define V3_TILE_CHUNKS 2048
#define V3_S0_ROW_SALT       0x6B61F28F3CC48744UL
#define V3_OFFSET_FIRST_SALT 0x3F1F886D659E316AUL
#define V3_OFFSET_STEP_SALT  0xD4C194F3ADB3B1C7UL

// signed int8×int8 dot of 4 packed bytes (byte-exact with CUDA __dp4a.s32).
// USE_AMD_DOT4: RDNA3+/CDNA native `v_dot4_i32_i8` (__builtin_amdgcn_sudot4, signed×signed, no
// saturation) — 1 instruction vs the 4-mul unpack, so the matmul (the walk's hot loop) runs several
// times faster. Byte-EXACT with the scalar path: signed int8 terms can't overflow i32 (max per
// accumulator over the 64 dp4 calls ≈ 4.2M ≪ 2^31), so clamp=false wraparound never triggers. The
// driver only enables it on gfx11/gfx12 (dot9-insts); every other device keeps the scalar unpack
// (identical result), and the plain rebuild-without-defines fallback covers a JIT that rejects it.
inline int v3_dp4(uint a, uint b, int acc) {
#ifdef USE_AMD_DOT4
    return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, acc, false);
#else
    acc += (int)((char)(a       & 0xff)) * (int)((char)(b       & 0xff));
    acc += (int)((char)((a >> 8) & 0xff)) * (int)((char)((b >> 8) & 0xff));
    acc += (int)((char)((a >>16) & 0xff)) * (int)((char)((b >>16) & 0xff));
    acc += (int)((char)((a >>24) & 0xff)) * (int)((char)((b >>24) & 0xff));
    return acc;
#endif
}

// v3_rho8 (pom_v3.rs rho8): finalizer -> low byte.
inline uint v3_rho8(int acc, uint tweak) {
    uint z = (uint)acc ^ tweak;
    z *= 0x9E3779B9u; z ^= z >> 16;
    z *= 0x85EBCA6Bu; z ^= z >> 13;
    return z & 0xffu;
}

// ---- blake3 (single-chunk path: inputs <= 1024 B, counter 0) ----
__constant uint B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u };
__constant uchar B3_SCHED[7][16] = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8},
    {3,4,10,12,13,2,7,14,6,5,9,0,11,15,8,1},
    {10,7,12,9,14,3,13,15,4,0,11,2,5,8,1,6},
    {12,13,9,11,15,10,14,8,7,2,5,3,0,1,6,4},
    {9,14,11,5,8,12,15,1,13,3,0,10,2,6,4,7},
    {11,15,5,0,1,9,8,6,14,10,2,12,3,4,7,13} };
#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_ROOT        8u
inline uint b3_rotr(uint x, int n) { return (x >> n) | (x << (32 - n)); }
inline void b3_g(uint* v, int a, int b, int c, int d, uint mx, uint my) {
    v[a] += v[b] + mx; v[d] = b3_rotr(v[d] ^ v[a], 16);
    v[c] += v[d];      v[b] = b3_rotr(v[b] ^ v[c], 12);
    v[a] += v[b] + my; v[d] = b3_rotr(v[d] ^ v[a], 8);
    v[c] += v[d];      v[b] = b3_rotr(v[b] ^ v[c], 7);
}
inline void b3_compress(uint cv[8], const uint m[16], uint block_len, uint flags) {
    uint v[16];
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    v[8]=B3_IV[0]; v[9]=B3_IV[1]; v[10]=B3_IV[2]; v[11]=B3_IV[3];
    v[12]=0; v[13]=0; v[14]=block_len; v[15]=flags;
    for (int r = 0; r < 7; r++) {
        __constant uchar* s = B3_SCHED[r];
        b3_g(v,0,4,8,12,  m[s[0]],  m[s[1]]);
        b3_g(v,1,5,9,13,  m[s[2]],  m[s[3]]);
        b3_g(v,2,6,10,14, m[s[4]],  m[s[5]]);
        b3_g(v,3,7,11,15, m[s[6]],  m[s[7]]);
        b3_g(v,0,5,10,15, m[s[8]],  m[s[9]]);
        b3_g(v,1,6,11,12, m[s[10]], m[s[11]]);
        b3_g(v,2,7,8,13,  m[s[12]], m[s[13]]);
        b3_g(v,3,4,9,14,  m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) cv[i] = v[i] ^ v[i + 8];
}
// blake3 of one 256 B state row (64 packed words, PRIVATE) -> 8 words in LDS.
inline void b3_hash_row(const uint row[64], __local uint* out) {
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    for (int b = 0; b < 4; b++) {
        uint flags = (b == 0 ? B3_CHUNK_START : 0u) | (b == 3 ? (B3_CHUNK_END | B3_ROOT) : 0u);
        b3_compress(cv, row + b * 16, 64, flags);
    }
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}
// blake3 of a 64 B concat (two child hashes, both in LDS) -> 8 words in LDS. The message is
// staged into a PRIVATE buffer first because b3_compress takes a __private message pointer.
inline void b3_hash_pair(__local const uint* m, __local uint* out) {
    uint mp[16];
    for (int i = 0; i < 16; i++) mp[i] = m[i];
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    b3_compress(cv, mp, 64, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// v3 grind: 1 work-group per nonce, local_size = 256. Winner = lowest nonce whose pow<=target.
__kernel void pom_mine_v3(
    __global const uint* restrict blob32,   // whole tier blob as u32 (chunk idx -> blob32[idx*8])
    const u64 n_tiles,                       // n_chunks / 2048
    const uint K,                            // 256
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,   // POW-fold pph words
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,   // SEED-fold pph words
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,   // target (4 LE u64)
    const u64 nonce_base, const u64 n_nonces,
    volatile __global u64* winner,
    __local uint* s_merkle)                  // 3072 u32 = 12 KB scratch for the final Merkle tree only
{
    const uint x  = get_local_id(0);         // state row index 0..255
    const u64  gid = get_group_id(0);
    if (gid >= n_nonces) return;
    const u64 nonce = nonce_base + gid;
    const u64 seed  = pom_seed_fold(nonce, time_, s0, s1, s2, s3);

    // S_0 row x: mix64 keystream.
    uint row4[V3_D4];
    {
        u64 h = pom_mix64(seed ^ (V3_S0_ROW_SALT + (u64)x));
        for (int k4 = 0; k4 < V3_D4; k4++) { h = pom_mix64(h); row4[k4] = (uint)h; }
    }
    u64 off = pom_mix64(seed ^ V3_OFFSET_FIRST_SALT) % n_tiles;

    // The tile is read DIRECTLY from the resident blob (VRAM/L2), NOT staged into LDS: freeing the
    // 64 KB LDS lets far more wavefronts stay resident (latency hiding) and the 64 KB tile is small
    // enough to stay L2/Infinity-Cache-hot across a step's 256×256 reuse — measured ~1.6x faster
    // than the LDS-staged version. The state update is row-independent and reads no shared memory,
    // so NO per-step barrier is needed. Byte-identical (same bytes, different memory space).
    for (uint step = 1; step <= K; step++) {
        const u64 chunk0 = off * (u64)V3_TILE_CHUNKS;
        __global const uint* gtile = blob32 + chunk0 * 8UL;   // tile row j = gtile + j*64

        // Next offset from THIS tile's snippet (first 32 B = 8 u32), derived by every WI.
        {
            u64 sf = 0;
            for (int w = 0; w < 8; w++) sf = pom_mix64(sf ^ (u64)gtile[w]);
            off = pom_mix64(seed ^ (u64)(step + 1) * V3_OFFSET_STEP_SALT ^ sf) % n_tiles;
        }

        // next[x][j] = rho8(dot_i8(row_x, col_j), tweak(step,x,j)), packed 4 bytes/uint.
        const uint step_tweak = step * 0x9E3779B9u + x * 0xC2B2AE35u;
        uint new4[V3_D4];
        for (int j4 = 0; j4 < V3_D4; j4++) {
            uint packed = 0;
            for (int jj = 0; jj < 4; jj++) {
                const int j = j4 * 4 + jj;
                __global const uint* col = gtile + j * V3_D4;    // tile row j = 64 u32
                int a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                for (int k16 = 0; k16 < V3_D4 / 4; k16++) {
                    a0 = v3_dp4(row4[k16*4+0], col[k16*4+0], a0);
                    a1 = v3_dp4(row4[k16*4+1], col[k16*4+1], a1);
                    a2 = v3_dp4(row4[k16*4+2], col[k16*4+2], a2);
                    a3 = v3_dp4(row4[k16*4+3], col[k16*4+3], a3);
                }
                packed |= v3_rho8((a0 + a1) + (a2 + a3),
                                  step_tweak + (uint)j * 0x85EBCA6Bu) << (8 * jj);
            }
            new4[j4] = packed;
        }
        for (int k4 = 0; k4 < V3_D4; k4++) row4[k4] = new4[k4];
    }

    // root_K: blake3 row leaves + depth-8 tree in the small LDS scratch.
    b3_hash_row(row4, s_merkle + x * 8);
    barrier(CLK_LOCAL_MEM_FENCE);
    __local uint* src = s_merkle;
    __local uint* dst = s_merkle + V3_D * 8;
    for (uint n = V3_D; n > 1; n >>= 1) {
        if (x < n / 2) b3_hash_pair(src + x * 16, dst + x * 8);
        barrier(CLK_LOCAL_MEM_FENCE);
        __local uint* tmp = src; src = dst; dst = tmp;
    }
    if (x == 0) {
        const u64 fin = (u64)src[0] | ((u64)src[1] << 32);
        u64 pv[4];
        pom_pow_fold(fin, p0, p1, p2, p3, pv);
        if (pom_le_leq(pv, t0, t1, t2, t3)) {
            u64 old = *winner;
            while (nonce < old) {
                u64 prev = atom_cmpxchg(winner, old, nonce);
                if (prev == old) break;
                old = prev;
            }
        }
    }
}
