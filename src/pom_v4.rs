//! PoM v4 (D=32 re-walk) — wire structs, constants, host proof-build. Byte-exact mirror of the
//! node's `consensus/core/src/pom_v4.rs`. Field order and salts MUST stay bit-identical.

use crate::pom::{blake, hash_pair, mix64, verify_merkle, WeightIndex};
use anyhow::{anyhow, Result};
use borsh::{BorshDeserialize, BorshSerialize};

// --- walk primitives (byte-exact mirrors of the node's `consensus/core/src/pom_v4.rs` /
// --- `pom_v3.rs`; the GPU kernel is the walk, these serve the host proof-build + self-check) ---

/// fold64: the low 8 bytes of a 32-byte Merkle root, little-endian, as the u64 `final_state`.
#[inline]
pub fn fold64(root: &[u8; 32]) -> u64 {
    u64::from_le_bytes(root[..8].try_into().unwrap())
}

/// rho8: the entrywise nonlinearity applied to each int8 dot-product accumulator.
#[inline]
pub fn rho8(acc: i32, tweak: u32) -> u8 {
    let mut z = (acc as u32) ^ tweak;
    z = z.wrapping_mul(0x9E3779B9);
    z ^= z >> 16;
    z = z.wrapping_mul(0x85EBCA6B);
    z ^= z >> 13;
    (z & 0xff) as u8
}

/// rho_tweak: per-(step,row,col) tweak folded into rho8.
#[inline]
pub fn rho_tweak(step: u32, row: u32, col: u32) -> u32 {
    step.wrapping_mul(0x9E3779B9).wrapping_add(row.wrapping_mul(0xC2B2AE35)).wrapping_add(col.wrapping_mul(0x85EBCA6B))
}

/// Signed int8 dot product of a state row and a tile column (both D bytes).
///
/// SIMD-dispatched (ported from keryx-node 38001c23): AVX2 → SSE4.1 → NEON → scalar. Result is
/// bit-identical on every path — the products are exact in i32 and addition is associative here,
/// and the GPU kernel does the same wrapping i8×i8 → i32 accumulation. Worth real time: a proof
/// re-walk is 256 steps × 1024 dots = 262k of these, on the per-share witness path.
#[inline]
pub fn dot_i8(row: &[u8], col: &[u8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx2(row, col) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { dot_i8_sse41(row, col) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_i8_neon(row, col) };
    }
    #[allow(unreachable_code)]
    dot_i8_scalar(row, col)
}

/// Reference reduction: the fallback, and the oracle the SIMD paths are tested against.
#[inline]
pub fn dot_i8_scalar(row: &[u8], col: &[u8]) -> i32 {
    debug_assert_eq!(row.len(), POM_V4_D);
    debug_assert_eq!(col.len(), POM_V4_D);
    let mut acc = 0i32;
    let mut i = 0;
    while i < POM_V4_D {
        acc += (row[i] as i8 as i32) * (col[i] as i8 as i32);
        acc += (row[i + 1] as i8 as i32) * (col[i + 1] as i8 as i32);
        acc += (row[i + 2] as i8 as i32) * (col[i + 2] as i8 as i32);
        acc += (row[i + 3] as i8 as i32) * (col[i + 3] as i8 as i32);
        acc += (row[i + 4] as i8 as i32) * (col[i + 4] as i8 as i32);
        acc += (row[i + 5] as i8 as i32) * (col[i + 5] as i8 as i32);
        acc += (row[i + 6] as i8 as i32) * (col[i + 6] as i8 as i32);
        acc += (row[i + 7] as i8 as i32) * (col[i + 7] as i8 as i32);
        i += 8;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(row: &[u8], col: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        // 16 i8 → 16 i16, madd pairwise → 8 i32; twice for the 32-byte row.
        let r0 = _mm_loadu_si128(row.as_ptr() as *const __m128i);
        let c0 = _mm_loadu_si128(col.as_ptr() as *const __m128i);
        let p0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(r0), _mm256_cvtepi8_epi16(c0));
        let r1 = _mm_loadu_si128(row.as_ptr().add(16) as *const __m128i);
        let c1 = _mm_loadu_si128(col.as_ptr().add(16) as *const __m128i);
        let p1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(r1), _mm256_cvtepi8_epi16(c1));
        let sum = _mm256_add_epi32(p0, p1);
        let hi = _mm256_extracti128_si256(sum, 1);
        let lo = _mm256_castsi256_si128(sum);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
        let s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
        _mm_cvtsi128_si32(s)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn dot_i8_sse41(row: &[u8], col: &[u8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = _mm_setzero_si128();
        let mut off = 0usize;
        while off < POM_V4_D {
            let r = _mm_loadl_epi64(row.as_ptr().add(off) as *const __m128i);
            let c = _mm_loadl_epi64(col.as_ptr().add(off) as *const __m128i);
            acc = _mm_add_epi32(acc, _mm_madd_epi16(_mm_cvtepi8_epi16(r), _mm_cvtepi8_epi16(c)));
            off += 8;
        }
        let acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
        let acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
        _mm_cvtsi128_si32(acc)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(row: &[u8], col: &[u8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        unsafe fn half(row: *const i8, col: *const i8) -> int32x4_t {
            let a = vld1q_s8(row);
            let b = vld1q_s8(col);
            let a_lo = vmovl_s8(vget_low_s8(a));
            let a_hi = vmovl_s8(vget_high_s8(a));
            let b_lo = vmovl_s8(vget_low_s8(b));
            let b_hi = vmovl_s8(vget_high_s8(b));
            let p0 = vmull_s16(vget_low_s16(a_lo), vget_low_s16(b_lo));
            let p1 = vmull_high_s16(a_lo, b_lo);
            let p2 = vmull_s16(vget_low_s16(a_hi), vget_low_s16(b_hi));
            let p3 = vmull_high_s16(a_hi, b_hi);
            vaddq_s32(vaddq_s32(p0, p1), vaddq_s32(p2, p3))
        }
        let s0 = half(row.as_ptr() as *const i8, col.as_ptr() as *const i8);
        let s1 = half(row.as_ptr().add(16) as *const i8, col.as_ptr().add(16) as *const i8);
        vaddvq_s32(vaddq_s32(s0, s1))
    }
}

/// snippet_fold: fold the 32-byte tile snippet (8 LE u32 words) into the offset chain.
#[inline]
pub fn snippet_fold(snippet: &[u8; POM_V4_SNIPPET_BYTES]) -> u64 {
    let mut sf = 0u64;
    for w in 0..8 {
        let word = u32::from_le_bytes(snippet[w * 4..w * 4 + 4].try_into().unwrap()) as u64;
        sf = mix64(sf ^ word);
    }
    sf
}

pub const POM_V4_D: usize = 32;
pub const POM_V4_K: usize = 256;
pub const POM_V4_CHUNK_BYTES: usize = 32;
pub const POM_V4_TILE_BYTES: usize = POM_V4_D * POM_V4_D; // 1 KB
pub const POM_V4_TILE_CHUNKS: u64 = (POM_V4_TILE_BYTES / POM_V4_CHUNK_BYTES) as u64;
pub const POM_V4_SNIPPET_BYTES: usize = 32;
pub const POM_V4_TILE_SUBTREE_DEPTH: u32 = 5; // log2(POM_V4_TILE_CHUNKS)

/// sha256("keryx-v4-s0-row-salt")
pub const POM_V4_S0_ROW_SALT: u64 = 0x03421325594C3C51;
/// sha256("keryx-v4-offset-first-salt")
pub const POM_V4_OFFSET_FIRST_SALT: u64 = 0x6D1CCF96AC4D76F9;
/// sha256("keryx-v4-offset-step-salt")
pub const POM_V4_OFFSET_STEP_SALT: u64 = 0x89050E78D34609EF;

/// Merkle range proof for one tile (path from the tile's aligned subtree root up to R_T).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomV4RangeProof {
    pub path: Vec<[u8; 32]>,
}

/// v4 walk witness — mirror of the node's `PomProofV4` (borsh field order).
#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProofV4 {
    pub tier: u8,
    pub tiles: Vec<Vec<u8>>,
    pub merkle: Vec<PomV4RangeProof>,
}

#[inline]
pub fn v4_first_offset(seed: u64, n_tiles: u64) -> u64 {
    mix64(seed ^ POM_V4_OFFSET_FIRST_SALT) % n_tiles
}

#[inline]
pub fn v4_next_offset(seed: u64, step: u64, snippet: &[u8; POM_V4_SNIPPET_BYTES], n_tiles: u64) -> u64 {
    mix64(seed ^ (step + 1).wrapping_mul(POM_V4_OFFSET_STEP_SALT) ^ snippet_fold(snippet)) % n_tiles
}

pub fn v4_initial_state(seed: u64) -> Vec<u8> {
    let mut s = vec![0u8; POM_V4_D * POM_V4_D];
    for r in 0..POM_V4_D {
        let mut h = mix64(seed ^ POM_V4_S0_ROW_SALT.wrapping_add(r as u64));
        for k4 in 0..POM_V4_D / 4 {
            h = mix64(h);
            s[r * POM_V4_D + k4 * 4..r * POM_V4_D + k4 * 4 + 4].copy_from_slice(&(h as u32).to_le_bytes());
        }
    }
    s
}

pub fn v4_transition(state: &[u8], tile: &[u8], step: u32) -> Vec<u8> {
    let mut next = vec![0u8; POM_V4_D * POM_V4_D];
    v4_transition_into(&mut next, state, tile, step);
    next
}

/// In-place transition `dst = rho(src × tile)` — no allocation, and the SIMD feature check is
/// hoisted OUT of the 1024-cell loop (once per step instead of once per dot). `dst` must not
/// alias `src`. Bit-identical to the allocating form.
pub fn v4_transition_into(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
    debug_assert_eq!(dst.len(), POM_V4_D * POM_V4_D);
    debug_assert_eq!(src.len(), POM_V4_D * POM_V4_D);
    debug_assert_eq!(tile.len(), POM_V4_TILE_BYTES);
    #[cfg(target_arch = "x86_64")]
    {
        // Dispatch the WHOLE 32x32 transition once. Calling a #[target_feature] dot-product helper
        // from the generic loop leaves an out-of-line call in every one of the 1,024 cells on rustc,
        // i.e. 262,144 calls per proof walk (twice that with the independent verifier). Once the
        // outer function carries the same feature, LLVM can inline the dot into the cell loop.
        if is_x86_feature_detected!("avx2") {
            unsafe { v4_transition_into_avx2(dst, src, tile, step) };
            return;
        }
        if is_x86_feature_detected!("sse4.1") {
            unsafe { v4_transition_into_sse41(dst, src, tile, step) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { v4_transition_into_neon(dst, src, tile, step) };
        return;
    }
    v4_transition_into_scalar(dst, src, tile, step);
}

#[inline]
fn v4_transition_into_scalar(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
    for x in 0..POM_V4_D {
        let row = &src[x * POM_V4_D..(x + 1) * POM_V4_D];
        for j in 0..POM_V4_D {
            let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
            let dot = dot_i8_scalar(row, col);
            dst[x * POM_V4_D + j] = rho8(dot, rho_tweak(step, x as u32, j as u32));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn v4_transition_into_avx2(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
    for x in 0..POM_V4_D {
        let row = &src[x * POM_V4_D..(x + 1) * POM_V4_D];
        for j in 0..POM_V4_D {
            let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
            let dot = unsafe { dot_i8_avx2(row, col) };
            dst[x * POM_V4_D + j] = rho8(dot, rho_tweak(step, x as u32, j as u32));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn v4_transition_into_sse41(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
    for x in 0..POM_V4_D {
        let row = &src[x * POM_V4_D..(x + 1) * POM_V4_D];
        for j in 0..POM_V4_D {
            let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
            let dot = unsafe { dot_i8_sse41(row, col) };
            dst[x * POM_V4_D + j] = rho8(dot, rho_tweak(step, x as u32, j as u32));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn v4_transition_into_neon(dst: &mut [u8], src: &[u8], tile: &[u8], step: u32) {
    for x in 0..POM_V4_D {
        let row = &src[x * POM_V4_D..(x + 1) * POM_V4_D];
        for j in 0..POM_V4_D {
            let col = &tile[j * POM_V4_D..(j + 1) * POM_V4_D];
            let dot = unsafe { dot_i8_neon(row, col) };
            dst[x * POM_V4_D + j] = rho8(dot, rho_tweak(step, x as u32, j as u32));
        }
    }
}

fn merkle_root_32(mut nodes: [[u8; 32]; POM_V4_D]) -> [u8; 32] {
    let mut len = POM_V4_D;
    while len > 1 {
        for i in 0..len / 2 {
            let left = nodes[i * 2];
            let right = nodes[i * 2 + 1];
            nodes[i] = hash_pair(&left, &right);
        }
        len /= 2;
    }
    nodes[0]
}

pub fn v4_state_root(state: &[u8]) -> [u8; 32] {
    debug_assert_eq!(state.len(), POM_V4_D * POM_V4_D);
    let mut leaves = [[0u8; 32]; POM_V4_D];
    for (r, leaf) in leaves.iter_mut().enumerate() {
        *leaf = blake(&state[r * POM_V4_D..(r + 1) * POM_V4_D]);
    }
    merkle_root_32(leaves)
}

fn v4_tile_subtree_root(tile: &[u8]) -> [u8; 32] {
    debug_assert_eq!(tile.len(), POM_V4_TILE_BYTES);
    let mut leaves = [[0u8; 32]; POM_V4_D];
    for (leaf, chunk) in leaves.iter_mut().zip(tile.chunks_exact(POM_V4_CHUNK_BYTES)) {
        *leaf = blake(chunk);
    }
    merkle_root_32(leaves)
}

/// Re-walk `seed` reading tiles from `index`, returning the proof and the derived `final_state`.
pub fn build_proof_v4(tier: u8, seed: u64, index: &WeightIndex) -> Result<(PomProofV4, u64)> {
    let n_tiles = index.n_chunks / POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small for the v4 walk"));
    }
    // Double-buffered walk: the state ping-pongs between two fixed buffers instead of allocating a
    // fresh 1 KB Vec per step (256 allocs per proof gone). The tiles themselves must still be kept
    // (they are the proof payload).
    let mut state = v4_initial_state(seed);
    let mut scratch = vec![0u8; POM_V4_D * POM_V4_D];
    let mut off = v4_first_offset(seed, n_tiles);
    let mut tiles = Vec::with_capacity(POM_V4_K);
    let mut merkle = Vec::with_capacity(POM_V4_K);
    for step in 1..=POM_V4_K as u64 {
        // One contiguous read per tensor overlap instead of 32 separate 32-byte preads. A proof
        // carries 256 tiles, so this removes up to 7,936 tiny syscalls from every winning share.
        let mut tile = vec![0u8; POM_V4_TILE_BYTES];
        index.read_chunks_into(off * POM_V4_TILE_CHUNKS, &mut tile);
        let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
        let path = index.merkle_path_from_level(off * POM_V4_TILE_CHUNKS, POM_V4_TILE_SUBTREE_DEPTH);
        merkle.push(PomV4RangeProof { path });
        v4_transition_into(&mut scratch, &state, &tile, step as u32);
        std::mem::swap(&mut state, &mut scratch);
        tiles.push(tile);
        if step < POM_V4_K as u64 {
            off = v4_next_offset(seed, step, &snippet, n_tiles);
        }
    }
    let final_state = fold64(&v4_state_root(&state));
    Ok((PomProofV4 { tier, tiles, merkle }, final_state))
}

/// Pre-submit self-check: re-walk the proof against `r_t` and return the derived `final_state`.
pub fn verify_proof_v4(seed: u64, proof: &PomProofV4, r_t: &[u8; 32], n_chunks: u64) -> Result<u64> {
    if proof.tiles.len() != POM_V4_K || proof.merkle.len() != POM_V4_K {
        return Err(anyhow!("v4 proof wrong shape"));
    }
    let n_tiles = n_chunks / POM_V4_TILE_CHUNKS;
    if n_tiles == 0 {
        return Err(anyhow!("blob too small"));
    }
    let mut state = v4_initial_state(seed);
    let mut scratch = vec![0u8; POM_V4_D * POM_V4_D];
    let mut off = v4_first_offset(seed, n_tiles);
    for step in 1..=POM_V4_K {
        let tile = &proof.tiles[step - 1];
        if tile.len() != POM_V4_TILE_BYTES {
            return Err(anyhow!("v4 tile wrong shape"));
        }
        if !verify_merkle(v4_tile_subtree_root(tile), off, &proof.merkle[step - 1].path, r_t) {
            return Err(anyhow!("v4 tile fails range proof at step {step}"));
        }
        v4_transition_into(&mut scratch, &state, tile, step as u32);
        std::mem::swap(&mut state, &mut scratch);
        if step < POM_V4_K {
            let snippet: [u8; 32] = tile[..POM_V4_SNIPPET_BYTES].try_into().unwrap();
            off = v4_next_offset(seed, step as u64, &snippet, n_tiles);
        }
    }
    Ok(fold64(&v4_state_root(&state)))
}

#[cfg(test)]
mod simd_dot_tests {
    use super::*;

    /// The SIMD paths MUST agree with the scalar oracle bit-for-bit: this dot feeds rho8 → the
    /// state → the final_state → the pow. One wrong lane = an invalid block.
    #[test]
    fn simd_dot_matches_scalar_oracle() {
        let mut h = 0x9E3779B97F4A7C15u64;
        let mut next = || {
            h = mix64(h);
            h
        };
        for _ in 0..2000 {
            let row: Vec<u8> = (0..POM_V4_D).map(|_| next() as u8).collect();
            let col: Vec<u8> = (0..POM_V4_D).map(|_| next() as u8).collect();
            assert_eq!(dot_i8(&row, &col), dot_i8_scalar(&row, &col), "dispatch != scalar");
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    assert_eq!(unsafe { dot_i8_avx2(&row, &col) }, dot_i8_scalar(&row, &col), "avx2 != scalar");
                }
                if is_x86_feature_detected!("sse4.1") {
                    assert_eq!(unsafe { dot_i8_sse41(&row, &col) }, dot_i8_scalar(&row, &col), "sse41 != scalar");
                }
            }
        }
        // extremes: all -128 / all +127 (max |accumulation|)
        let lo = vec![0x80u8; POM_V4_D];
        let hi = vec![0x7fu8; POM_V4_D];
        for (a, b) in [(&lo, &lo), (&lo, &hi), (&hi, &hi)] {
            assert_eq!(dot_i8(a, b), dot_i8_scalar(a, b));
        }
    }

    /// The zero-alloc transition must equal the allocating one.
    #[test]
    fn transition_into_matches_allocating() {
        let mut h = 0xDEADBEEFCAFEBABEu64;
        let mut next = || {
            h = mix64(h);
            h
        };
        let state: Vec<u8> = (0..POM_V4_D * POM_V4_D).map(|_| next() as u8).collect();
        let tile: Vec<u8> = (0..POM_V4_TILE_BYTES).map(|_| next() as u8).collect();
        for step in [1u32, 7, 255] {
            let a = v4_transition(&state, &tile, step);
            let mut b = vec![0u8; POM_V4_D * POM_V4_D];
            let mut oracle = vec![0u8; POM_V4_D * POM_V4_D];
            v4_transition_into(&mut b, &state, &tile, step);
            v4_transition_into_scalar(&mut oracle, &state, &tile, step);
            assert_eq!(a, b, "transition_into != v4_transition at step {step}");
            assert_eq!(b, oracle, "SIMD transition != scalar oracle at step {step}");
        }
    }

    #[test]
    fn fixed_merkle_reducers_match_generic_oracle() {
        let mut h = 0x1234_5678_9abc_def0u64;
        let mut bytes = vec![0u8; POM_V4_TILE_BYTES];
        for b in &mut bytes {
            h = mix64(h);
            *b = h as u8;
        }
        let leaves: Vec<[u8; 32]> = bytes.chunks_exact(POM_V4_CHUNK_BYTES).map(blake).collect();
        assert_eq!(v4_tile_subtree_root(&bytes), crate::pom::merkle_root(&leaves));
        assert_eq!(v4_state_root(&bytes), crate::pom::merkle_root(&leaves));
    }
}
