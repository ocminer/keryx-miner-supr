// Keryx Proof-of-Model v4 mining kernel (Metal, Apple Silicon port).
//
// Byte-exact mirror of src/pom_mine.cu::pom_mine_v4 and the host src/pom_v4.rs
// (= keryxd v1.5.1 consensus/core/src/pom_v4.rs). PoM v4 is a D=32 int8 matrix
// re-walk: 256 steps, each reading a 1 KB tile (32 canonical 32-byte chunks) from
// the resident packed blob, computing state' = rho8(state · tile) with a per-cell
// tweak, chaining the tile offset from the tile snippet, then folding a blake3
// Merkle root of the final 32x32 state into the u64 final_state. The seed/pow
// folds match pom::pom_block_seed_v4 / pom::pom_pow_value.
//
// ONE THREADGROUP = ONE NONCE; 32 threads/threadgroup, thread x owns state row x
// as 8 packed uints (= 32 int8). Threadgroup memory `s_tile[512]` (2 KB) holds the
// 1 KB tile (256 uints) and doubles as the Merkle-fold scratch (host mirror uses
// the same 2 KB). Blob layout is the single packed MTLBuffer (prefix=[0,N],
// addrs=[buf.gpuAddress()]); v4_load_tile still does a per-lane prefix search so
// the multi-tensor byte-exact test path is exercised too.

#include <metal_stdlib>
using namespace metal;

struct PomV4Uniforms {
    ulong  n_tiles;       // n_total_chunks / 32
    uint   k_steps;       // POM_V4_K = 256
    uint   n_tensors;     // packed blob = 1
    ulong  p0; ulong p1; ulong p2; ulong p3;   // POW-fold words (H3-salted pph)
    ulong  s0; ulong s1; ulong s2; ulong s3;   // SEED-fold words (v4-salted pph)
    ulong  time_;
    ulong  t0; ulong t1; ulong t2; ulong t3;   // target (LE u64x4)
    ulong  nonce_base;
    uint   n_nonces;
    uint   _pad;
};

// ---- scalar folds (identical to the pre-v4 kernel / pom_mine.cu) ----
inline ulong mix64(ulong x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9UL;
    x ^= x >> 27; x *= 0x94d049bb133111ebUL;
    x ^= x >> 31;
    return x;
}

inline ulong pom_seed_fold(ulong nonce, ulong time_, ulong p0, ulong p1, ulong p2, ulong p3) {
    ulong s = mix64(nonce ^ 0x4B65727978531UL);
    s = mix64(s ^ time_);
    s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}

inline void pom_pow_fold(ulong fin, ulong p0, ulong p1, ulong p2, ulong p3, thread ulong* out) {
    out[0] = mix64(fin    ^ p0 ^ 0x9E3779B97F4A7C15UL);
    out[1] = mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FUL);
    out[2] = mix64(out[1] ^ p2 ^ 0x165667B19E3779F9UL);
    out[3] = mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93UL);
}

inline bool pom_le_leq(thread const ulong* a, ulong b0, ulong b1, ulong b2, ulong b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

// ---- blake3 (single-chunk path; inputs <= 64 B, counter 0) — mirror of pom_mine.cu ----
constant uint B3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u,
};
constant uchar B3_SCHED[7][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8},
    {3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1},
    {10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6},
    {12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    {9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7},
    {11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13},
};
#define B3_CHUNK_START 1u
#define B3_CHUNK_END   2u
#define B3_ROOT        8u

inline uint b3_rotr(uint x, uint n) { return (x >> n) | (x << (32u - n)); }

inline void b3_g(thread uint* v, int a, int b, int c, int d, uint mx, uint my) {
    v[a] += v[b] + mx; v[d] = b3_rotr(v[d] ^ v[a], 16u);
    v[c] += v[d];      v[b] = b3_rotr(v[b] ^ v[c], 12u);
    v[a] += v[b] + my; v[d] = b3_rotr(v[d] ^ v[a], 8u);
    v[c] += v[d];      v[b] = b3_rotr(v[b] ^ v[c], 7u);
}

inline void b3_compress(thread uint* cv, thread const uint* m, uint block_len, uint flags) {
    uint v[16];
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    v[8] = B3_IV[0]; v[9] = B3_IV[1]; v[10] = B3_IV[2]; v[11] = B3_IV[3];
    v[12] = 0u; v[13] = 0u; v[14] = block_len; v[15] = flags;
    for (int r = 0; r < 7; r++) {
        constant uchar* s = B3_SCHED[r];
        b3_g(v, 0, 4, 8,  12, m[s[0]],  m[s[1]]);
        b3_g(v, 1, 5, 9,  13, m[s[2]],  m[s[3]]);
        b3_g(v, 2, 6, 10, 14, m[s[4]],  m[s[5]]);
        b3_g(v, 3, 7, 11, 15, m[s[6]],  m[s[7]]);
        b3_g(v, 0, 5, 10, 15, m[s[8]],  m[s[9]]);
        b3_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        b3_g(v, 2, 7, 8,  13, m[s[12]], m[s[13]]);
        b3_g(v, 3, 4, 9,  14, m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) cv[i] = v[i] ^ v[i + 8];
}

// blake3 of a 32-byte state row (single partial block, block_len=32).
inline void b3_hash_row32(thread const uint* row, threadgroup uint* out) {
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    uint m[16];
    for (int i = 0; i < 8; i++) m[i] = row[i];
    for (int i = 8; i < 16; i++) m[i] = 0u;
    b3_compress(cv, m, 32u, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// blake3 of a 64-byte concat of two child hashes (single full block).
inline void b3_hash_pair(threadgroup const uint* in16, threadgroup uint* out) {
    uint cv[8];
    for (int i = 0; i < 8; i++) cv[i] = B3_IV[i];
    uint m[16];
    for (int i = 0; i < 16; i++) m[i] = in16[i];
    b3_compress(cv, m, 64u, B3_CHUNK_START | B3_CHUNK_END | B3_ROOT);
    for (int i = 0; i < 8; i++) out[i] = cv[i];
}

// ---- v4 walk primitives ----
#define V4_D4 8   // 8 uints per 32-byte row/column

inline uint v4_rho8(int acc, uint tweak) {
    uint z = (uint)acc ^ tweak;
    z *= 0x9E3779B9u; z ^= z >> 16;
    z *= 0x85EBCA6Bu; z ^= z >> 13;
    return z & 0xffu;
}

// signed int8 4-way dot-product accumulate (dp4a): a,b each pack 4 int8.
inline int v4_dp4a(uint a, uint b, int acc) {
    for (int i = 0; i < 4; i++) {
        int ai = (int)(char)((a >> (8 * i)) & 0xffu);
        int bi = (int)(char)((b >> (8 * i)) & 0xffu);
        acc += ai * bi;
    }
    return acc;
}

// Per-lane tile load: lane x fetches canonical chunk (tile_index*32 + x) into s_tile[x*8].
inline void v4_load_tile(device const ulong* prefix, device const ulong* tensor_addrs,
                         uint n_tensors, ulong tile_index, threadgroup uint* s_tile, uint lane) {
    ulong idx = tile_index * 32UL + lane;
    uint lo = 0, hi = n_tensors;
    while (lo + 1 < hi) { uint mid = (lo + hi) >> 1; if (prefix[mid] <= idx) lo = mid; else hi = mid; }
    device const uint* base = (device const uint*)tensor_addrs[lo];
    ulong local = (idx - prefix[lo]) * 8UL;   // 8 uints per 32-byte chunk
    threadgroup uint* dst = s_tile + lane * 8;
    for (int i = 0; i < 8; i++) dst[i] = base[local + i];
}

kernel void pom_mine_v4(
    device   const ulong*        prefix       [[buffer(0)]],  // n_tensors+1 cumulative chunk counts
    device   const ulong*        tensor_addrs [[buffer(1)]],  // n_tensors gpu addresses
    constant const PomV4Uniforms& u           [[buffer(2)]],
    device   atomic_uint*        winner       [[buffer(3)]],  // winning tid (nonce = base + tid)
    uint  gid  [[threadgroup_position_in_grid]],
    uint  x    [[thread_position_in_threadgroup]])
{
    threadgroup uint s_tile[512];   // 2 KB: tile (256 uints) + Merkle-fold scratch (256 uints)
    if ((ulong)gid >= u.n_nonces) return;
    ulong nonce = u.nonce_base + (ulong)gid;
    ulong seed = pom_seed_fold(nonce, u.time_, u.s0, u.s1, u.s2, u.s3);

    // S_0: mix64 keystream per row (thread x owns row x).
    uint row4[V4_D4];
    {
        ulong h = mix64(seed ^ (0x03421325594C3C51UL + (ulong)x));   // V4_S0_ROW_SALT
        for (int k4 = 0; k4 < V4_D4; k4++) { h = mix64(h); row4[k4] = (uint)h; }
    }

    ulong off = mix64(seed ^ 0x6D1CCF96AC4D76F9UL) % u.n_tiles;      // V4_OFFSET_FIRST_SALT

    for (uint step = 1; step <= u.k_steps; step++) {
        v4_load_tile(prefix, tensor_addrs, u.n_tensors, off, s_tile, x);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Next offset from the CURRENT tile's snippet (first 32 bytes = 8 words).
        {
            ulong sf = 0;
            for (int w = 0; w < 8; w++) sf = mix64(sf ^ (ulong)s_tile[w]);
            off = mix64(seed ^ ((ulong)(step + 1) * 0x89050E78D34609EFUL) ^ sf) % u.n_tiles; // V4_OFFSET_STEP_SALT
        }

        // Transition: new row x = rho8(row_x . tile_col_j) for j in 0..32.
        uint step_tweak = step * 0x9E3779B9u + x * 0xC2B2AE35u;
        uint new4[V4_D4];
        for (int j4 = 0; j4 < V4_D4; j4++) {
            uint packed = 0;
            for (int jj = 0; jj < 4; jj++) {
                int j = j4 * 4 + jj;
                threadgroup const uint* col = &s_tile[j * V4_D4];   // tile row j used as column j
                int acc = 0;
                for (int k = 0; k < V4_D4; k++) acc = v4_dp4a(row4[k], col[k], acc);
                packed |= v4_rho8(acc, step_tweak + (uint)j * 0x85EBCA6Bu) << (8 * jj);
            }
            new4[j4] = packed;
        }
        for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = new4[k4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // root_K: 32 blake3 row leaves + complete depth-5 tree (scratch reuses the tile region).
    b3_hash_row32(row4, s_tile + x * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* src = s_tile;
    threadgroup uint* dst = s_tile + 32 * 8;
    for (uint n = 32; n > 1; n >>= 1) {
        if (x < n / 2) b3_hash_pair(src + x * 16, dst + x * 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint* tmp = src; src = dst; dst = tmp;
    }

    if (x == 0) {
        ulong fin = (ulong)src[0] | ((ulong)src[1] << 32);
        ulong pv[4];
        pom_pow_fold(fin, u.p0, u.p1, u.p2, u.p3, pv);
        if (pom_le_leq(pv, u.t0, u.t1, u.t2, u.t3)) {
            atomic_fetch_min_explicit(winner, gid, memory_order_relaxed);
        }
    }
}

// Debug variant: writes fold64(v4_state_root(S_K)) per nonce to states[gid] (no pow fold / target),
// so the host can compare against pom_v4::build_proof_v4's final_state.
kernel void pom_walk_states_v4(
    device   const ulong*        prefix       [[buffer(0)]],
    device   const ulong*        tensor_addrs [[buffer(1)]],
    constant const PomV4Uniforms& u           [[buffer(2)]],
    device   ulong*              states       [[buffer(3)]],
    uint  gid  [[threadgroup_position_in_grid]],
    uint  x    [[thread_position_in_threadgroup]])
{
    threadgroup uint s_tile[512];
    if ((ulong)gid >= u.n_nonces) return;
    ulong nonce = u.nonce_base + (ulong)gid;
    ulong seed = pom_seed_fold(nonce, u.time_, u.s0, u.s1, u.s2, u.s3);

    uint row4[V4_D4];
    {
        ulong h = mix64(seed ^ (0x03421325594C3C51UL + (ulong)x));
        for (int k4 = 0; k4 < V4_D4; k4++) { h = mix64(h); row4[k4] = (uint)h; }
    }
    ulong off = mix64(seed ^ 0x6D1CCF96AC4D76F9UL) % u.n_tiles;

    for (uint step = 1; step <= u.k_steps; step++) {
        v4_load_tile(prefix, tensor_addrs, u.n_tensors, off, s_tile, x);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            ulong sf = 0;
            for (int w = 0; w < 8; w++) sf = mix64(sf ^ (ulong)s_tile[w]);
            off = mix64(seed ^ ((ulong)(step + 1) * 0x89050E78D34609EFUL) ^ sf) % u.n_tiles;
        }
        uint step_tweak = step * 0x9E3779B9u + x * 0xC2B2AE35u;
        uint new4[V4_D4];
        for (int j4 = 0; j4 < V4_D4; j4++) {
            uint packed = 0;
            for (int jj = 0; jj < 4; jj++) {
                int j = j4 * 4 + jj;
                threadgroup const uint* col = &s_tile[j * V4_D4];
                int acc = 0;
                for (int k = 0; k < V4_D4; k++) acc = v4_dp4a(row4[k], col[k], acc);
                packed |= v4_rho8(acc, step_tweak + (uint)j * 0x85EBCA6Bu) << (8 * jj);
            }
            new4[j4] = packed;
        }
        for (int k4 = 0; k4 < V4_D4; k4++) row4[k4] = new4[k4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    b3_hash_row32(row4, s_tile + x * 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup uint* src = s_tile;
    threadgroup uint* dst = s_tile + 32 * 8;
    for (uint n = 32; n > 1; n >>= 1) {
        if (x < n / 2) b3_hash_pair(src + x * 16, dst + x * 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint* tmp = src; src = dst; dst = tmp;
    }
    if (x == 0) states[gid] = (ulong)src[0] | ((ulong)src[1] << 32);
}
