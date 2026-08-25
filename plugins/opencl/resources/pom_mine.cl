// Keryx Proof-of-Model v4 mining kernel — OpenCL port of cuda/pom_mine.cu (v0.11.0, keryxd v1.5.1).
//
// PoM v4: state S = 32×32 int8 (D=32), K=256 steps. Per step: read a 1 KB tile (32 canonical 32 B
// chunks) at a data-dependent offset from the resident weight blob, then
//   next[x][j] = rho8(dot_i8(row_x, tile_col_j), rho_tweak(step, x, j))
// Final: fold64(merkle_root(blake3 row leaves of S_K)) -> pow fold -> target check.
//
// BYTE-IDENTICAL to `src/pom_v4.rs` (the host proof builder / consensus mirror) and to
// `cuda/pom_mine.cu::pom_mine_v4`. The seed/pow folds MUST match the host exactly or blocks
// are rejected. v1/v2/v3 kernels are REMOVED — this build mines v4 only (upstream 1fd307a).
//
// AMD layout differences vs CUDA:
// - CUDA gathers via per-tensor device pointers (bases[]+prefix[] binary search); OpenCL loads
//   the blob into 1..4 contiguous slabs (Polaris per-buffer caps — see pom_opencl.rs). Slabs are
//   TILE-ALIGNED so a tile never straddles a slab; `v4_slab` picks the slab per step.
// - CUDA runs one 32-thread block per nonce. 32-thread workgroups underfill AMD CUs, so we pack
//   V4_NPG = 8 nonces per 256-thread workgroup: sub-nonce = local_id/32, lane = local_id&31, each
//   sub-nonce owns a 2 KB LDS strip (1 KB tile + 1 KB blake3/merkle ping-pong). barrier() is
//   workgroup-wide but UNIFORM: every sub-nonce runs the same K-step loop in lockstep.

#ifdef cl_khr_int64_base_atomics
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#endif

typedef ulong u64;

#define V4_D 32
#define V4_D4 8                    // uints per row/column (32 bytes)
#define V4_TILE_CHUNKS 32
#define V4_TILE_U32 256            // 1 KB tile as u32 words
#define V4_STRIP_U32 512           // per-sub-nonce LDS strip: tile (256) + merkle scratch (256)
#define V4_NPG 8                   // sub-nonces per 256-thread workgroup

#define V4_S0_ROW_SALT       0x03421325594C3C51UL
#define V4_OFFSET_FIRST_SALT 0x6D1CCF96AC4D76F9UL
#define V4_OFFSET_STEP_SALT  0x89050E78D34609EFUL

// Baked-divisor defines (JIT sets these; runtime args are the fallback): POM_NT = n_tiles,
// POM_SLABT = tiles per slab. Constant divisors strength-reduce to multiply-highs (byte-exact —
// the compiler's own constant-division transform). Single-slab rigs bake POM_SLABT = n_tiles so
// the slab select folds away entirely.
#ifdef POM_NT
#define V4_NT(arg) ((u64)(POM_NT))
#else
#define V4_NT(arg) (arg)
#endif
#ifdef POM_SLABT
#define V4_ST(arg) ((u64)(POM_SLABT))
#else
#define V4_ST(arg) (arg)
#endif

// SplitMix64 finalizer — identical to pom.rs mix64.
inline u64 pom_mix64(u64 x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

// Block seed fold — identical to pom.rs pom_block_seed_v4 (host passes the v4-salted pph words).
inline u64 pom_seed_fold(u64 nonce, u64 time_, u64 p0, u64 p1, u64 p2, u64 p3) {
    u64 s = pom_mix64(nonce ^ 0x4B65727978531UL);
    s = pom_mix64(s ^ time_);
    s = pom_mix64(s ^ p0); s = pom_mix64(s ^ p1); s = pom_mix64(s ^ p2); s = pom_mix64(s ^ p3);
    return s;
}

// pow_value fold — identical to pom.rs pom_pow_value (host passes the H3-salted pph words:
// "v4 pow uses the h3 fold").
inline void pom_pow_fold(u64 fin, u64 p0, u64 p1, u64 p2, u64 p3, u64 out[4]) {
    out[0] = pom_mix64(fin    ^ p0 ^ 0x9E3779B97F4A7C15UL);
    out[1] = pom_mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FUL);
    out[2] = pom_mix64(out[1] ^ p2 ^ 0x165667B19E3779F9UL);
    out[3] = pom_mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93UL);
}

// 256-bit little-endian a <= b (word 3 most-significant).
inline bool pom_le_leq(const u64 a[4], u64 b0, u64 b1, u64 b2, u64 b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

// Signed int8×int8 dot of 4 packed bytes — byte-exact with CUDA __dp4a.s32 and pom_v4::dot_i8.
// USE_AMD_DOT4: RDNA3+/gfx11-12 native v_dot4_i32_i8 (sudot4, dot9-insts).
// USE_AMD_SDOT4: GCN/CDNA gfx906/908/90a native dot (sdot4, dot1-insts) — ~6x on MI50.
// Fallback: scalar unpack (Polaris/RDNA1-2/Windows Adrenalin). All three are byte-identical
// (int8 dots cannot overflow i32 at these depths, so clamp/saturation never triggers).
inline int v4_dp4(uint a, uint b, int acc) {
#if defined(USE_AMD_DOT4)
    return __builtin_amdgcn_sudot4(true, (int)a, true, (int)b, acc, false);
#elif defined(USE_AMD_SDOT4)
    return __builtin_amdgcn_sdot4((int)a, (int)b, acc, false);
#else
    acc += (int)((char)(a       & 0xff)) * (int)((char)(b       & 0xff));
    acc += (int)((char)((a >> 8) & 0xff)) * (int)((char)((b >> 8) & 0xff));
    acc += (int)((char)((a >>16) & 0xff)) * (int)((char)((b >>16) & 0xff));
    acc += (int)((char)((a >>24) & 0xff)) * (int)((char)((b >>24) & 0xff));
    return acc;
#endif
}

// rho8 finalizer -> low byte (pom_v4.rs rho8).
inline uint v4_rho8(int acc, uint tweak) {
    uint z = (uint)acc ^ tweak;
    z *= 0x9E3779B9u; z ^= z >> 16;
    z *= 0x85EBCA6Bu; z ^= z >> 13;
    return z & 0xffu;
}

// Slab select for tile `off`: slabs are tile-aligned so the whole 1 KB tile lives in one slab.
inline __global const uint* v4_slab(const __global uint* restrict b0, const __global uint* restrict b1,
                                    const __global uint* restrict b2, const __global uint* restrict b3,
                                    u64 off, u64 slab_tiles, u64* tile_in_slab) {
    u64 s = off / V4_ST(slab_tiles);
    *tile_in_slab = off - s * V4_ST(slab_tiles);
    return (s == 0UL) ? b0 : (s == 1UL) ? b1 : (s == 2UL) ? b2 : b3;
}

// ---- blake3 (single-block paths; byte-exact with pom.rs blake / blake3 crate) ----
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
// blake3 of one 32 B state row (single partial block) -> 8 words into LDS.
inline void b3_hash_row32(const uint row[V4_D4], __local uint* out) {
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    uint m[16];
    for (int i = 0; i < 8; i++) m[i] = row[i];
    for (int i = 8; i < 16; i++) m[i] = 0u;
    b3_compress(cv, m, 32, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}
// blake3 of a 64 B parent (two child hashes, both in LDS) -> 8 words into LDS. The message is
// staged into a PRIVATE buffer because b3_compress takes a __private message pointer.
inline void b3_hash_pair(__local const uint* m, __local uint* out) {
    uint mp[16];
    for (int i = 0; i < 16; i++) mp[i] = m[i];
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    b3_compress(cv, mp, 64, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

#define V4_UNROLL __attribute__((opencl_unroll_hint))

// v4 grind: 256-thread workgroups of V4_NPG(8) sub-nonces × 32 lanes; lane x owns state row x.
// Winner = lowest passing nonce via CAS-min (host re-walks + re-checks, so a kernel false
// positive is dropped there, never submitted).
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void pom_mine_v4(
    __global const uint* restrict b0,        // blob slab 0 as u32 (single-slab rigs: the whole blob)
    __global const uint* restrict b1,        // absent slabs = slab 0 repeated (never selected)
    __global const uint* restrict b2,
    __global const uint* restrict b3,
    const u64 n_tiles,                       // n_chunks / 32
    const u64 slab_tiles,                    // tiles per slab (single-slab layout: == n_tiles)
    const uint K,                            // 256
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,   // POW-fold pph words (H3-salted)
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,   // SEED-fold pph words (v4-salted)
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,   // target (4 LE u64)
    const u64 nonce_base, const u64 n_nonces,
    volatile __global u64* winner,
    __local uint* scratch)                   // V4_NPG * V4_STRIP_U32 u32 = 16 KB (host sets size)
{
    const uint lid  = get_local_id(0);
    const uint sub  = lid >> 5;              // sub-nonce 0..7 within the group
    const uint lane = lid & 31u;             // state row 0..31 within the sub-nonce
    const u64  gsub = (u64)get_group_id(0) * V4_NPG + sub;
    // Sub-nonces past the batch still execute the walk with a dummy nonce (uniform barriers —
    // every lane must reach every workgroup barrier); they just never submit.
    const bool live  = gsub < n_nonces;
    const u64  nonce = nonce_base + (live ? gsub : 0UL);
    const u64  seed  = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    __local uint* strip = scratch + sub * V4_STRIP_U32;   // this sub-nonce's tile + merkle scratch

    // S_0 row `lane`: mix64 keystream (identical to pom_v4::v4_initial_state).
    uint row4[V4_D4];
    {
        u64 h = pom_mix64(seed ^ (V4_S0_ROW_SALT + (u64)lane));
        V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) { h = pom_mix64(h); row4[k4] = (uint)h; }
    }
    u64 off = pom_mix64(seed ^ V4_OFFSET_FIRST_SALT) % V4_NT(n_tiles);

    for (uint step = 1; step <= K; step++) {
        // Load this sub-nonce's 1 KB tile: lane loads chunk (tile*32 + lane) = 32 B via 2×uint4
        // (128-bit). 32-B-aligned src + LDS dst → fewer memory ops / better MLP on the latency-
        // bound random gather (measured +13% on gfx1102 WMMA). Byte-exact (same bytes, wider loads).
        {
            u64 tin;
            const __global uint* sb = v4_slab(b0, b1, b2, b3, off, slab_tiles, &tin);
            const __global uint4* src4 = (const __global uint4*)(sb + (tin * (u64)V4_TILE_CHUNKS + lane) * 8UL);
            __local uint4* dst4 = (__local uint4*)(strip + lane * 8);
            dst4[0] = src4[0]; dst4[1] = src4[1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next offset from THIS tile's snippet (first 32 B = 8 u32), derived by every lane.
        {
            u64 sf = 0;
            V4_UNROLL for (int w = 0; w < 8; w++) sf = pom_mix64(sf ^ (u64)strip[w]);
            off = pom_mix64(seed ^ (u64)(step + 1) * V4_OFFSET_STEP_SALT ^ sf) % V4_NT(n_tiles);
        }

        // next[lane][j] = rho8(dot_i8(row, col_j), tweak(step, lane, j)), packed 4 bytes/uint.
        const uint step_tweak = step * 0x9E3779B9u + lane * 0xC2B2AE35u;
        uint new4[V4_D4];
        V4_UNROLL for (int j4 = 0; j4 < V4_D4; j4++) {
            uint packed = 0;
            V4_UNROLL for (int jj = 0; jj < 4; jj++) {
                const int j = j4 * 4 + jj;
                __local const uint* col = strip + j * V4_D4;   // tile column j = 32 bytes
                int acc = 0;
                V4_UNROLL for (int k = 0; k < V4_D4; k++) acc = v4_dp4(row4[k], col[k], acc);
                packed |= v4_rho8(acc, step_tweak + (uint)j * 0x85EBCA6Bu) << (8 * jj);
            }
            new4[j4] = packed;
        }
        V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = new4[k4];
        barrier(CLK_LOCAL_MEM_FENCE);   // all lanes done with the tile before step+1 overwrites it
    }

    // root_K: 32 blake3 row leaves + complete depth-5 tree (scratch reuses the tile strip).
    b3_hash_row32(row4, strip + lane * 8);
    barrier(CLK_LOCAL_MEM_FENCE);
    __local uint* src = strip;
    __local uint* dst = strip + V4_D * 8;
    for (uint n = V4_D; n > 1; n >>= 1) {
        if (lane < n / 2) b3_hash_pair(src + lane * 16, dst + lane * 8);
        barrier(CLK_LOCAL_MEM_FENCE);
        __local uint* tmp = src; src = dst; dst = tmp;
    }

    if (lane == 0 && live) {
        const u64 fin = (u64)src[0] | ((u64)src[1] << 32);
        u64 pv[4];
        pom_pow_fold(fin, p0, p1, p2, p3, pv);
        if (pom_le_leq(pv, t0, t1, t2, t3)) {
            // CAS-min: needs only cl_khr_int64_base_atomics (64-bit atom_min is the extended ext).
            u64 old = *winner;
            while (nonce < old) {
                u64 prev = atom_cmpxchg(winner, old, nonce);
                if (prev == old) break;
                old = prev;
            }
        }
    }
}

// ============================================================================
// Two-phase v4 solver (port of cuda/pom_mine.cu pom_mine_v4_chase + the phase-2
// pipeline of pom_mine_v4_tc; the tensor-core inner loop is CUDA-only — here the
// matmul stays the same dp4a/scalar dot as pom_mine_v4). +20% expected dp4a-only.
//
// KEY INSIGHT: the v4 offset chain depends ONLY on tile snippets (chunk 0 of each
// tile), NEVER on the walk state. So the whole 256-tile offset sequence can be
// resolved BEFORE the matmul chain runs. In the single-phase kernel each step
// serializes load->snippet-fold->(next offset)->load, so the matmul overlaps no
// memory. Splitting it lets phase 2 prefetch tile t+1 while computing matmul t.
// ============================================================================

// PHASE 1 — chase: one work-item per nonce follows the snippet chain and records
// all K tile offsets (u32 each; n_tiles < 2^32 for any real model). Reads only
// chunk 0 (32 B) of each tile — a latency-bound pointer chase the GPU hides with
// parallelism (~10% of total time, +3% memory traffic). Byte-exact with the
// single-phase snippet fold: same 8 u32 words (LE, zero-extended), same order.
__kernel void pom_mine_v4_chase(
    __global const uint* restrict b0,
    __global const uint* restrict b1,
    __global const uint* restrict b2,
    __global const uint* restrict b3,
    const u64 n_tiles,
    const u64 slab_tiles,
    const uint K,
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,   // SEED-fold pph words (v4-salted)
    const u64 time_,
    const u64 nonce_base, const u64 n_nonces,
    __global uint* restrict offsets)                          // [n_nonces][K]
{
    const u64 i = (u64)get_global_id(0);
    if (i >= n_nonces) return;
    const u64 nonce = nonce_base + i;
    const u64 seed = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    __global uint* my = offsets + i * (u64)K;
    u64 off = pom_mix64(seed ^ V4_OFFSET_FIRST_SALT) % V4_NT(n_tiles);
    for (uint step = 1; step <= K; step++) {
        my[step - 1] = (uint)off;
        // chunk 0 of tile `off` = 8 u32 (the snippet).
        u64 tin;
        const __global uint* sb = v4_slab(b0, b1, b2, b3, off, slab_tiles, &tin);
        const __global uint* c0 = sb + tin * (u64)V4_TILE_U32;   // tin*256 = chunk 0 of this tile
        u64 sf = 0;
        V4_UNROLL for (int w = 0; w < 8; w++) sf = pom_mix64(sf ^ (u64)c0[w]);
        off = pom_mix64(seed ^ (u64)(step + 1) * V4_OFFSET_STEP_SALT ^ sf) % V4_NT(n_tiles);
    }
}

// PHASE 2 — pipelined walk: identical math to pom_mine_v4, but the offsets come
// from phase 1 (no snippet fold in the hot loop) and each step DOUBLE-BUFFERS the
// tile: prefetch tile[step] into registers while the matmul for tile[step-1] runs,
// then stage the registers into the other LDS buffer. The matmul never waits on
// DRAM. Same V4_NPG(8)-sub-nonce × 32-lane layout and 16 KB LDS as pom_mine_v4
// (strip = 2×256 u32 tile double-buffer; merkle reuses it).
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void pom_mine_v4_tp(
    __global const uint* restrict b0,
    __global const uint* restrict b1,
    __global const uint* restrict b2,
    __global const uint* restrict b3,
    const u64 n_tiles,
    const u64 slab_tiles,
    const uint K,
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,   // POW-fold pph words (H3-salted)
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,   // SEED-fold pph words (v4-salted)
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,
    const u64 nonce_base, const u64 n_nonces,
    __global const uint* restrict offsets,                    // [n_nonces][K] from phase 1
    volatile __global u64* winner,
    __local uint* scratch)                                    // V4_NPG * V4_STRIP_U32 u32 = 16 KB
{
    const uint lid  = get_local_id(0);
    const uint sub  = lid >> 5;
    const uint lane = lid & 31u;
    const u64  gsub = (u64)get_group_id(0) * V4_NPG + sub;
    const bool live  = gsub < n_nonces;
    const u64  idx   = live ? gsub : 0UL;      // dummy sub-nonces read nonce 0's offsets (in-bounds)
    const u64  nonce = nonce_base + idx;
    const u64  seed  = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    __local uint* strip = scratch + sub * V4_STRIP_U32;
    __local uint* buf0 = strip;                // tile double-buffer A
    __local uint* buf1 = strip + V4_TILE_U32;  // tile double-buffer B
    __global const uint* my = offsets + idx * (u64)K;

    // S_0 row `lane`.
    uint row4[V4_D4];
    {
        u64 h = pom_mix64(seed ^ (V4_S0_ROW_SALT + (u64)lane));
        V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) { h = pom_mix64(h); row4[k4] = (uint)h; }
    }

    // Prologue: load tile my[0] into buf0.
    {
        u64 tin;
        const __global uint* sb = v4_slab(b0, b1, b2, b3, (u64)my[0], slab_tiles, &tin);
        const __global uint* src = sb + (tin * (u64)V4_TILE_CHUNKS + lane) * 8UL;
        __local uint* dst = buf0 + lane * 8;
        V4_UNROLL for (int w = 0; w < 8; w++) dst[w] = src[w];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = 1; step <= K; step++) {
        __local uint* cur = ((step - 1u) & 1u) ? buf1 : buf0;   // step1->buf0, step2->buf1, ...
        __local uint* nxt = ((step - 1u) & 1u) ? buf0 : buf1;

        // Prefetch NEXT tile (my[step], 0-indexed) into registers — issued BEFORE the matmul so
        // its DRAM latency overlaps the compute. `has_next` is uniform across the workgroup.
        const bool has_next = (step < K);
        uint pref[8];
        if (has_next) {
            u64 tin;
            const __global uint* sb = v4_slab(b0, b1, b2, b3, (u64)my[step], slab_tiles, &tin);
            const __global uint* src = sb + (tin * (u64)V4_TILE_CHUNKS + lane) * 8UL;
            V4_UNROLL for (int w = 0; w < 8; w++) pref[w] = src[w];
        }

        // Matmul on `cur` — identical to pom_mine_v4.
        const uint step_tweak = step * 0x9E3779B9u + lane * 0xC2B2AE35u;
        uint new4[V4_D4];
        V4_UNROLL for (int j4 = 0; j4 < V4_D4; j4++) {
            uint packed = 0;
            V4_UNROLL for (int jj = 0; jj < 4; jj++) {
                const int j = j4 * 4 + jj;
                __local const uint* col = cur + j * V4_D4;
                int acc = 0;
                V4_UNROLL for (int k = 0; k < V4_D4; k++) acc = v4_dp4(row4[k], col[k], acc);
                packed |= v4_rho8(acc, step_tweak + (uint)j * 0x85EBCA6Bu) << (8 * jj);
            }
            new4[j4] = packed;
        }
        V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = new4[k4];

        // Stage the prefetched next tile into the other buffer, then one barrier: it guarantees
        // (a) `nxt` fully written before step+1 reads it and (b) all lanes done reading `cur`
        // before step+2 overwrites it (double-buffer).
        if (has_next) {
            __local uint* dst = nxt + lane * 8;
            V4_UNROLL for (int w = 0; w < 8; w++) dst[w] = pref[w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // root_K: identical tail to pom_mine_v4 (merkle reuses the strip).
    b3_hash_row32(row4, strip + lane * 8);
    barrier(CLK_LOCAL_MEM_FENCE);
    __local uint* src = strip;
    __local uint* dst = strip + V4_D * 8;
    for (uint n = V4_D; n > 1; n >>= 1) {
        if (lane < n / 2) b3_hash_pair(src + lane * 16, dst + lane * 8);
        barrier(CLK_LOCAL_MEM_FENCE);
        __local uint* tmp = src; src = dst; dst = tmp;
    }

    if (lane == 0 && live) {
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

// ============================================================================
// PoM v4 WMMA walk (RDNA3 gfx11 / gfx12 only — needs V_WMMA_I32_16X16X16_IU8).
// Two-phase like pom_mine_v4_tp (offsets from pom_mine_v4_chase, double-buffered
// tile prefetch), but the 32x32x32 int8 transition runs on the matrix cores: one
// wave32 per nonce, 2x2 output blocks x 2 k-steps = 8 WMMA/step. This makes the
// matmul near-free so the tile-load latency becomes the bottleneck — the regime
// the two-phase pipeline was built for. The state lives in LDS (WMMA A-fragments
// need any lane to read any row); rho8 is applied to the int32 accumulators with
// destination-computed (x,j), byte-exact with pom_v4::v4_transition (validated in
// scratchpad/clcheck/wmma_v4.cl and the v4_byte_exact test).
#ifdef USE_AMD_WMMA
typedef int int4v __attribute__((ext_vector_type(4)));
typedef int int8v __attribute__((ext_vector_type(8)));

// Per sub-nonce LDS strip (u32): S ping-pong (2x256) + tile double-buffer (2x256) = 1024 u32 = 4 KB.
#define V4W_STRIP_U32 1024

__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void pom_mine_v4_wmma(
    __global const uint* restrict b0,
    __global const uint* restrict b1,
    __global const uint* restrict b2,
    __global const uint* restrict b3,
    const u64 n_tiles,
    const u64 slab_tiles,
    const uint K,
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,
    const u64 nonce_base, const u64 n_nonces,
    __global const uint* restrict offsets,
    volatile __global u64* winner,
    __local uint* scratch)                     // V4_NPG * V4W_STRIP_U32 u32 = 32 KB
{
    const uint lid  = get_local_id(0);
    const uint sub  = lid >> 5;
    const uint lane = lid & 31u;               // wave lane 0..31
    const u64  gsub = (u64)get_group_id(0) * V4_NPG + sub;
    const bool live  = gsub < n_nonces;
    const u64  idx   = live ? gsub : 0UL;
    const u64  nonce = nonce_base + idx;
    const u64  seed  = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    __local uint* strip = scratch + sub * V4W_STRIP_U32;
    __local uint* sA = strip;                  // state buffer A (256 u32 = 32x32 int8)
    __local uint* sB = strip + 256;            // state buffer B
    __local uint* tA = strip + 512;            // tile buffer A
    __local uint* tB = strip + 768;            // tile buffer B
    __global const uint* my = offsets + idx * (u64)K;

    // S_0: lane `lane` writes state row `lane` (all 32 rows filled across the wave).
    {
        u64 h = pom_mix64(seed ^ (V4_S0_ROW_SALT + (u64)lane));
        V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) { h = pom_mix64(h); sA[lane * 8 + k4] = (uint)h; }
    }
    // Prologue: load tile my[0] into tA (lane loads chunk `lane`).
    {
        u64 tin;
        const __global uint* sb = v4_slab(b0, b1, b2, b3, (u64)my[0], slab_tiles, &tin);
        const __global uint* src = sb + (tin * (u64)V4_TILE_CHUNKS + lane) * 8UL;
        V4_UNROLL for (int w = 0; w < 8; w++) tA[lane * 8 + w] = src[w];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint step = 1; step <= K; step++) {
        __local uint* scur = (step & 1u) ? sA : sB;   // step1 reads sA, writes sB
        __local uint* snxt = (step & 1u) ? sB : sA;
        __local uint* tcur = ((step - 1u) & 1u) ? tB : tA;
        __local uint* tnxt = ((step - 1u) & 1u) ? tA : tB;

        // Prefetch NEXT tile into registers (overlaps the WMMA compute).
        const bool has_next = (step < K);
        uint pref[8];
        if (has_next) {
            u64 tin;
            const __global uint* sb = v4_slab(b0, b1, b2, b3, (u64)my[step], slab_tiles, &tin);
            const __global uint* src = sb + (tin * (u64)V4_TILE_CHUNKS + lane) * 8UL;
            V4_UNROLL for (int w = 0; w < 8; w++) pref[w] = src[w];
        }

        // WMMA transition: next = rho8(S · T^T, tweak). 2x2 blocks x 2 k-steps.
        __local const char* Sc = (__local const char*)scur;
        __local const char* Tc = (__local const char*)tcur;
        const uint xi = lane & 15u;
        const uint ji = lane & 15u;
        const uint step_base = step * 0x9E3779B9u;
        V4_UNROLL for (uint xb = 0; xb < 2; xb++) {
            V4_UNROLL for (uint jb = 0; jb < 2; jb++) {
                int8v acc = (int8v)(0);
                V4_UNROLL for (uint kb = 0; kb < 2; kb++) {
                    int4v a, b; char* ap = (char*)&a; char* bp = (char*)&b;
                    V4_UNROLL for (int ki = 0; ki < 16; ki++) {
                        ap[ki] = Sc[(16u*xb + xi) * 32 + 16u*kb + ki];
                        bp[ki] = Tc[(16u*jb + ji) * 32 + 16u*kb + ki];   // T^T (col-major B)
                    }
                    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc, false);
                }
                V4_UNROLL for (int vv = 0; vv < 8; vv++) {
                    const uint x = 16u*xb + 2u*(uint)vv + (lane >> 4);
                    const uint j = 16u*jb + (lane & 15u);
                    const uint tw = step_base + x * 0xC2B2AE35u + j * 0x85EBCA6Bu;
                    ((__local char*)snxt)[x * 32 + j] = (char)v4_rho8(acc[vv], tw);
                }
            }
        }

        // Stage the prefetched tile into the other buffer, then barrier (snxt visible for step+1;
        // all lanes done reading scur/tcur before step+2 reuses them).
        if (has_next) {
            V4_UNROLL for (int w = 0; w < 8; w++) tnxt[lane * 8 + w] = pref[w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Final state is in the buffer written at step K: (K & 1) ? sB : sA.
    __local uint* sfin = (K & 1u) ? sB : sA;
    uint row4[V4_D4];
    V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = sfin[lane * 8 + k4];
    barrier(CLK_LOCAL_MEM_FENCE);   // done reading sfin before merkle reuses the strip

    // root_K: reuse tile region (strip+512.. = 512 u32) for the merkle tree.
    __local uint* ms = strip + 512;
    b3_hash_row32(row4, ms + lane * 8);
    barrier(CLK_LOCAL_MEM_FENCE);
    __local uint* src = ms;
    __local uint* dst = ms + V4_D * 8;
    for (uint n = V4_D; n > 1; n >>= 1) {
        if (lane < n / 2) b3_hash_pair(src + lane * 16, dst + lane * 8);
        barrier(CLK_LOCAL_MEM_FENCE);
        __local uint* tmp = src; src = dst; dst = tmp;
    }

    if (lane == 0 && live) {
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
#endif // USE_AMD_WMMA

#ifdef USE_AMD_WMMA
// PoM v4 SINGLE-PHASE WMMA walk (RDNA3+): no offset chase — computes the next offset inline from
// the snippet like pom_mine_v4, but runs the 32x32x32 transition on the matrix cores. This is the
// combination that fits AMD: the single-phase kernel is already occupancy-latency-hidden (so the
// two-phase chase is dead weight on AMD), while WMMA still speeds the matmul itself. State lives in
// LDS (ping-pong); byte-exact with pom_v4::v4_transition (same layout as pom_mine_v4_wmma).
// Per sub-nonce LDS: 2 state (2x256) + 1 tile (256) = 768 u32 = 3 KB.
#define V4WSP_STRIP_U32 768
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void pom_mine_v4_wmma_sp(
    __global const uint* restrict b0,
    __global const uint* restrict b1,
    __global const uint* restrict b2,
    __global const uint* restrict b3,
    const u64 n_tiles,
    const u64 slab_tiles,
    const uint K,
    const u64 p0, const u64 p1, const u64 p2, const u64 p3,
    const u64 s0, const u64 s1, const u64 s2, const u64 s3,
    const u64 time_,
    const u64 t0, const u64 t1, const u64 t2, const u64 t3,
    const u64 nonce_base, const u64 n_nonces,
    volatile __global u64* winner,
    __local uint* scratch)                     // V4_NPG * V4WSP_STRIP_U32 u32 = 24 KB
{
    const uint lid  = get_local_id(0);
    const uint sub  = lid >> 5;
    const uint lane = lid & 31u;
    const u64  gsub = (u64)get_group_id(0) * V4_NPG + sub;
    const bool live  = gsub < n_nonces;
    const u64  nonce = nonce_base + (live ? gsub : 0UL);
    const u64  seed  = pom_seed_fold(nonce, time_, s0, s1, s2, s3);
    __local uint* strip = scratch + sub * V4WSP_STRIP_U32;
    __local uint* sA = strip;
    __local uint* sB = strip + 256;
    __local uint* tile = strip + 512;

    // S_0: lane writes state row `lane`.
    {
        u64 h = pom_mix64(seed ^ (V4_S0_ROW_SALT + (u64)lane));
        V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) { h = pom_mix64(h); sA[lane * 8 + k4] = (uint)h; }
    }
    u64 off = pom_mix64(seed ^ V4_OFFSET_FIRST_SALT) % V4_NT(n_tiles);

    for (uint step = 1; step <= K; step++) {
        __local uint* scur = (step & 1u) ? sA : sB;
        __local uint* snxt = (step & 1u) ? sB : sA;

        // Load this step's tile (lane loads chunk `lane` = 32 B). 128-bit (uint4) transactions: the
        // per-lane address is 32-B aligned (chunk stride) and the LDS dst is 32-B aligned, so 2×uint4
        // replaces 8×u32 — fewer memory ops / better MLP on the latency-bound random gather. Byte-exact.
        {
            u64 tin;
            const __global uint* sb = v4_slab(b0, b1, b2, b3, off, slab_tiles, &tin);
            const __global uint4* src4 = (const __global uint4*)(sb + (tin * (u64)V4_TILE_CHUNKS + lane) * 8UL);
            __local uint4* dst4 = (__local uint4*)(tile + lane * 8);
            dst4[0] = src4[0]; dst4[1] = src4[1];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Next offset from this tile's snippet (chunk 0 = tile[0..8]).
        {
            u64 sf = 0;
            V4_UNROLL for (int w = 0; w < 8; w++) sf = pom_mix64(sf ^ (u64)tile[w]);
            off = pom_mix64(seed ^ (u64)(step + 1) * V4_OFFSET_STEP_SALT ^ sf) % V4_NT(n_tiles);
        }

        // WMMA transition (same as pom_mine_v4_wmma).
        __local const char* Sc = (__local const char*)scur;
        __local const char* Tc = (__local const char*)tile;
        const uint xi = lane & 15u, ji = lane & 15u;
        const uint step_base = step * 0x9E3779B9u;
        V4_UNROLL for (uint xb = 0; xb < 2; xb++) {
            V4_UNROLL for (uint jb = 0; jb < 2; jb++) {
                int8v acc = (int8v)(0);
                V4_UNROLL for (uint kb = 0; kb < 2; kb++) {
                    int4v a, b; char* ap = (char*)&a; char* bp = (char*)&b;
                    V4_UNROLL for (int ki = 0; ki < 16; ki++) {
                        ap[ki] = Sc[(16u*xb + xi) * 32 + 16u*kb + ki];
                        bp[ki] = Tc[(16u*jb + ji) * 32 + 16u*kb + ki];
                    }
                    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc, false);
                }
                V4_UNROLL for (int vv = 0; vv < 8; vv++) {
                    const uint x = 16u*xb + 2u*(uint)vv + (lane >> 4);
                    const uint j = 16u*jb + (lane & 15u);
                    const uint tw = step_base + x * 0xC2B2AE35u + j * 0x85EBCA6Bu;
                    ((__local char*)snxt)[x * 32 + j] = (char)v4_rho8(acc[vv], tw);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __local uint* sfin = (K & 1u) ? sB : sA;
    uint row4[V4_D4];
    V4_UNROLL for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = sfin[lane * 8 + k4];
    barrier(CLK_LOCAL_MEM_FENCE);
    __local uint* ms = strip;                  // reuse state region for merkle (512 u32)
    b3_hash_row32(row4, ms + lane * 8);
    barrier(CLK_LOCAL_MEM_FENCE);
    __local uint* src = ms;
    __local uint* dst = ms + V4_D * 8;
    for (uint n = V4_D; n > 1; n >>= 1) {
        if (lane < n / 2) b3_hash_pair(src + lane * 16, dst + lane * 8);
        barrier(CLK_LOCAL_MEM_FENCE);
        __local uint* tmp = src; src = dst; dst = tmp;
    }
    if (lane == 0 && live) {
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
#endif // USE_AMD_WMMA
