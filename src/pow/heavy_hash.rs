use crate::pow::{hasher::HeavyHasher, xoshiro::XoShiRo256PlusPlus};
use crate::Hash;
use std::mem::MaybeUninit;

/// Domain-separation salts — must match `KERYX_MATRIX_SALT_V1/V2/V4` in
/// `consensus/pow/src/matrix.rs` exactly or the miner will derive a different
/// matrix than the node and every submitted block will be rejected.
/// v3 is intentionally skipped: it belongs to the abandoned diff-spiral chain.
const KERYX_MATRIX_SALT_V1: [u8; 32] = *b"KERYX:KeryxHash-v1:2026-04-12:xx";
const KERYX_MATRIX_SALT_V2: [u8; 32] = *b"KERYX:KeryxHash-v2:2026-05-29:xx";
const KERYX_MATRIX_SALT_V4: [u8; 32] = *b"KERYX:KeryxHash-v4:2026-06-07:xx";

/// DAA score at which the miner switches to SALT v2 — must match `pow_salt_v2_activation`
/// in network params. Miners compiled before this update will keep using v1 and their
/// blocks will be rejected after activation — that is the forced-update mechanism.
///
/// Mainnet: 17_275_000 (2026-05-30 ~15:00 UTC emergency activation)
/// Testnet: 6_000
pub const POW_SALT_V2_ACTIVATION_DAA: u64 = 17_275_000;

/// DAA score at which the miner switches to SALT v4 (chain relaunch on stock difficulty) —
/// must match `pow_salt_v4_activation` in network params. The matrix is generated host-side
/// here (the CUDA kernel receives the precomputed matrix), so no kernel/PTX change is needed.
///
/// Mainnet: 21_932_751 (same DAA as the old v3 gate; forks cleanly off the broken chain)
pub const POW_SALT_V4_ACTIVATION_DAA: u64 = 21_932_751;

/// Returns the active matrix-salt version (1, 2 or 4) for a block at `daa_score`.
/// Must mirror `active_salt_version` in `consensus/pow/src/lib.rs` (compared with `>=`).
#[inline(always)]
pub fn active_salt_version(daa_score: u64) -> u8 {
    if daa_score >= POW_SALT_V4_ACTIVATION_DAA {
        4
    } else if daa_score >= POW_SALT_V2_ACTIVATION_DAA {
        2
    } else {
        1
    }
}

/// Round constants for wave_mix — same as `WAVE_MIX_KEYS` in matrix.rs.
const WAVE_MIX_KEYS: [u64; 4] = [
    0x9e3779b97f4a7c15,
    0x6c62272e07bb0142,
    0xb5ad4eceda1ce2a9,
    0x243f6a8885a308d3,
];

/// Rotation amounts — same as `WAVE_MIX_ROTATIONS` in matrix.rs.
const WAVE_MIX_ROTATIONS: [u32; 4] = [17, 31, 47, 13];

/// 4-round ARX post-processing — must be bit-for-bit identical to
/// `fn wave_mix()` in `consensus/pow/src/matrix.rs`.
#[inline(always)]
fn wave_mix(bytes: [u8; 32]) -> [u8; 32] {
    let mut w = [
        u64::from_le_bytes(bytes[0..8].try_into().unwrap()),
        u64::from_le_bytes(bytes[8..16].try_into().unwrap()),
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        u64::from_le_bytes(bytes[24..32].try_into().unwrap()),
    ];
    for r in 0..4usize {
        w[0] = w[0].wrapping_add(w[1]).rotate_left(WAVE_MIX_ROTATIONS[0]) ^ WAVE_MIX_KEYS[r & 3];
        w[2] = w[2].wrapping_add(w[3]).rotate_left(WAVE_MIX_ROTATIONS[2]) ^ WAVE_MIX_KEYS[(r + 2) & 3];
        w[1] = w[1].wrapping_add(w[2]).rotate_left(WAVE_MIX_ROTATIONS[1]) ^ WAVE_MIX_KEYS[(r + 1) & 3];
        w[3] = w[3].wrapping_add(w[0]).rotate_left(WAVE_MIX_ROTATIONS[3]) ^ WAVE_MIX_KEYS[(r + 3) & 3];
    }
    let mut out = [0u8; 32];
    out[0..8].copy_from_slice(&w[0].to_le_bytes());
    out[8..16].copy_from_slice(&w[1].to_le_bytes());
    out[16..24].copy_from_slice(&w[2].to_le_bytes());
    out[24..32].copy_from_slice(&w[3].to_le_bytes());
    out
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Matrix(pub [[u16; 64]; 64]);

impl Matrix {
    #[inline(always)]
    pub fn generate(hash: Hash, salt_version: u8) -> Self {
        // XOR the block-hash seed with the active Keryx domain salt before the PRNG.
        // Must match Matrix::generate() in consensus/pow/src/matrix.rs.
        let salt: &[u8; 32] = match salt_version {
            1 => &KERYX_MATRIX_SALT_V1,
            2 => &KERYX_MATRIX_SALT_V2,
            _ => &KERYX_MATRIX_SALT_V4,
        };
        let salted = {
            let mut bytes = hash.to_le_bytes();
            bytes.iter_mut().zip(salt.iter()).for_each(|(b, s)| *b ^= s);
            Hash::from_le_bytes(bytes)
        };
        let mut generator = XoShiRo256PlusPlus::new(salted);
        loop {
            let mat = Self::rand_matrix_no_rank_check(&mut generator);
            if mat.compute_rank() == 64 {
                return mat;
            }
        }
    }

    #[inline(always)]
    fn rand_matrix_no_rank_check(generator: &mut XoShiRo256PlusPlus) -> Self {
        Self(array_from_fn(|_| {
            let mut val = 0;
            array_from_fn(|j| {
                let shift = j % 16;
                if shift == 0 {
                    val = generator.u64();
                }
                (val >> (4 * shift) & 0x0F) as u16
            })
        }))
    }

    #[inline(always)]
    fn convert_to_float(&self) -> [[f64; 64]; 64] {
        // SAFETY: An uninitialized MaybrUninit is always safe.
        let mut out: [[MaybeUninit<f64>; 64]; 64] = unsafe { MaybeUninit::uninit().assume_init() };

        out.iter_mut().zip(self.0.iter()).for_each(|(out_row, mat_row)| {
            out_row.iter_mut().zip(mat_row).for_each(|(out_element, &element)| {
                out_element.write(f64::from(element));
            })
        });
        // SAFETY: The loop above wrote into all indexes.
        unsafe { std::mem::transmute(out) }
    }

    pub fn compute_rank(&self) -> usize {
        const EPS: f64 = 1e-9;
        let mut mat_float = self.convert_to_float();
        let mut rank = 0;
        let mut row_selected = [false; 64];
        for i in 0..64 {
            if i >= 64 {
                // Required for optimization, See https://github.com/rust-lang/rust/issues/90794
                unreachable!()
            }
            let mut j = 0;
            while j < 64 {
                if !row_selected[j] && mat_float[j][i].abs() > EPS {
                    break;
                }
                j += 1;
            }
            if j != 64 {
                rank += 1;
                row_selected[j] = true;
                for p in (i + 1)..64 {
                    mat_float[j][p] /= mat_float[j][i];
                }
                for k in 0..64 {
                    if k != j && mat_float[k][i].abs() > EPS {
                        for p in (i + 1)..64 {
                            mat_float[k][p] -= mat_float[j][p] * mat_float[k][i];
                        }
                    }
                }
            }
        }
        rank
    }

    pub fn heavy_hash(&self, hash: Hash) -> Hash {
        let hash = hash.to_le_bytes();
        // SAFETY: An uninitialized MaybrUninit is always safe.
        let mut vec: [MaybeUninit<u8>; 64] = unsafe { MaybeUninit::uninit().assume_init() };
        for i in 0..32 {
            vec[2 * i].write(hash[i] >> 4);
            vec[2 * i + 1].write(hash[i] & 0x0F);
        }
        // SAFETY: The loop above wrote into all indexes.
        let vec: [u8; 64] = unsafe { std::mem::transmute(vec) };

        // Matrix-vector multiplication, convert to 4 bits, and then combine back to 8 bits.
        let mut product: [u8; 32] = array_from_fn(|i| {
            let mut sum1 = 0;
            let mut sum2 = 0;
            for (j, &elem) in vec.iter().enumerate() {
                sum1 += self.0[2 * i][j] * (elem as u16);
                sum2 += self.0[2 * i + 1][j] * (elem as u16);
            }
            ((sum1 >> 10) << 4) as u8 | (sum2 >> 10) as u8
        });

        // Concatenate 4 LSBs back to 8 bit xor with sum1
        product.iter_mut().zip(hash).for_each(|(p, h)| *p ^= h);

        // Keryx wave-mix: ARX post-processing — must match the node's matrix.rs.
        let product = wave_mix(product);

        HeavyHasher::hash(Hash::from_le_bytes(product))
    }
}

pub fn array_from_fn<F, T, const N: usize>(mut cb: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    let mut idx = 0;
    [(); N].map(|_| {
        let res = cb(idx);
        idx += 1;
        res
    })
}

#[cfg(test)]
mod tests {
    use crate::pow::heavy_hash::Matrix;
    use crate::pow::xoshiro::XoShiRo256PlusPlus;
    use crate::Hash;

    #[test]
    fn test_compute_rank() {
        let zero = Matrix([[0; 64]; 64]);
        assert_eq!(zero.compute_rank(), 0);
        let mut matrix = zero;
        let mut gen = XoShiRo256PlusPlus::new(Hash::from_le_bytes([42; 32]));
        matrix.0.iter_mut().for_each(|row| {
            row.iter_mut().for_each(|val| {
                *val = gen.u64() as u16;
            })
        });
        assert_eq!(matrix.compute_rank(), 64);

        matrix.0[0] = matrix.0[1];
        assert_eq!(matrix.compute_rank(), 63);
    }

}

#[cfg(all(test, feature = "bench"))]
mod benches {
    extern crate test;

    use self::test::{black_box, Bencher};
    use super::{Matrix, XoShiRo256PlusPlus};
    use crate::Hash;
    use rand::{thread_rng, Rng};

    #[bench]
    pub fn bench_compute_rank(bh: &mut Bencher) {
        let mut generator = XoShiRo256PlusPlus::new(Hash::from_le_bytes([42; 32]));
        let mut matrix = Matrix::rand_matrix_no_rank_check(&mut generator);
        bh.iter(|| {
            for _ in 0..10 {
                black_box(&mut matrix);
                black_box(matrix.compute_rank());
            }
        });
    }

    #[bench]
    pub fn bench_heavy_hash(bh: &mut Bencher) {
        let mut generator = XoShiRo256PlusPlus::new(Hash::from_le_bytes([42; 32]));
        let mut input = Hash::new(thread_rng().gen());
        let mut matrix = Matrix::rand_matrix_no_rank_check(&mut generator);
        bh.iter(|| {
            for _ in 0..10 {
                black_box(&mut matrix);
                black_box(&mut input);
                black_box(matrix.heavy_hash(input));
            }
        });
    }
}
