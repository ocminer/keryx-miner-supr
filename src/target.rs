use core::cmp::Ordering;
use std::fmt;

pub fn u256_from_compact_target(bits: u32) -> Uint256 {
    // This is a floating-point "compact" encoding originally used by
    // OpenSSL, which satoshi put into consensus code, so we're stuck
    // with it. The exponent needs to have 3 subtracted from it, hence
    // this goofy decoding code:
    let (mant, expt) = {
        let unshifted_expt = bits >> 24;
        if unshifted_expt <= 3 {
            ((bits & 0xFFFFFF) >> (8 * (3 - unshifted_expt as usize)), 0)
        } else {
            (bits & 0xFFFFFF, 8 * ((bits >> 24) - 3))
        }
    };

    // The mantissa is signed but may not be negative
    if mant > 0x7FFFFF {
        Default::default()
    } else {
        Uint256::from_u64(mant as u64) << (expt as usize)
    }
}

const MAX_DIFFICULTY_TARGET: Uint256 = Uint256([u64::MAX, u64::MAX, u64::MAX, 0x7fff_ffff_ffff_ffff]);

/// Decode a compact consensus target only when it is a valid Keryx network target.
///
/// The raw compact decoder intentionally mirrors the historical consensus representation and
/// silently truncates values that overflow 256 bits. Network telemetry and pool block-candidate
/// accounting must instead reject negative, zero, overflowing, and above-maximum targets.
pub fn network_target_from_compact_target(bits: u32) -> Option<Uint256> {
    let size = bits >> 24;
    let mantissa = bits & 0x007f_ffff;
    if bits & 0x0080_0000 != 0
        || mantissa == 0
        || size > 34
        || (size > 33 && mantissa > 0xff)
        || (size > 32 && mantissa > 0xffff)
    {
        return None;
    }

    let target = u256_from_compact_target(bits);
    (target != Uint256::default() && target <= MAX_DIFFICULTY_TARGET).then_some(target)
}

/// Convert a consensus compact block target into the network difficulty reported by keryxd.
///
/// Keryx network difficulty is `MAX_DIFFICULTY_TARGET / target`, where the all-network maximum is
/// `2^255 - 1`. This is deliberately distinct from Stratum share difficulty, whose conventional
/// difficulty-one target is `0xffff * 2^208`.
pub fn network_difficulty_from_compact_target(bits: u32) -> Option<f64> {
    const MAX_DIFFICULTY_TARGET_AS_F64: f64 = 5.789_604_461_865_81e76;

    let target = network_target_from_compact_target(bits)?;
    let target_f64 = target
        .0
        .iter()
        .enumerate()
        .map(|(word, value)| *value as f64 * 2f64.powi((word * 64) as i32))
        .sum::<f64>();
    let difficulty = MAX_DIFFICULTY_TARGET_AS_F64 / target_f64;
    (difficulty.is_finite() && difficulty >= 1.0).then_some(difficulty)
}

/// Little-endian large integer type
#[derive(Copy, Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct Uint256(pub [u64; 4]);

impl Uint256 {
    #[inline(always)]
    pub fn new(v: [u64; 4]) -> Self {
        Self(v)
    }
    /// Create an object from a given unsigned 64-bit integer
    #[inline]
    pub fn from_u64(init: u64) -> Uint256 {
        let mut ret = [0; 4];
        ret[0] = init;
        Uint256(ret)
    }

    /// Creates big integer value from a byte slice using
    /// little-endian encoding
    #[inline(always)]
    pub fn from_le_bytes(bytes: [u8; 32]) -> Uint256 {
        let mut out = [0u64; 4];
        // This should optimize to basically a transmute.
        out.iter_mut()
            .zip(bytes.chunks_exact(8))
            .for_each(|(word, bytes)| *word = u64::from_le_bytes(bytes.try_into().unwrap()));
        Self(out)
    }

    #[inline(always)]
    pub fn to_le_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        // This should optimize to basically a transmute.
        out.chunks_exact_mut(8).zip(self.0).for_each(|(bytes, word)| bytes.copy_from_slice(&word.to_le_bytes()));
        out
    }

    #[inline(always)]
    pub fn to_be_bytes(self) -> [u8; 32] {
        let mut out = [0u8; 32];
        // This should optimize to basically a transmute.
        out.chunks_exact_mut(8)
            .zip(self.0.iter().rev())
            .for_each(|(bytes, word)| bytes.copy_from_slice(&word.to_be_bytes()));
        out
    }
}

impl fmt::LowerHex for Uint256 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_le_bytes().iter().try_for_each(|&c| write!(f, "{:02x}", c))
    }
}

impl PartialOrd for Uint256 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Uint256) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Uint256 {
    #[inline(always)]
    fn cmp(&self, other: &Uint256) -> Ordering {
        // We need to manually implement ordering because we use little-endian
        // and the auto derive is a lexicographic ordering(i.e. memcmp)
        // which with numbers is equivalent to big-endian
        Iterator::cmp(self.0.iter().rev(), other.0.iter().rev())
    }
}

impl core::ops::Shl<usize> for Uint256 {
    type Output = Uint256;

    fn shl(self, shift: usize) -> Uint256 {
        let Uint256(ref original) = self;
        let mut ret = [0u64; 4];
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        for i in 0..4 {
            // Shift
            if bit_shift < 64 && i + word_shift < 4 {
                ret[i + word_shift] += original[i] << bit_shift;
            }
            // Carry
            if bit_shift > 0 && i + word_shift + 1 < 4 {
                ret[i + word_shift + 1] += original[i] >> (64 - bit_shift);
            }
        }
        Uint256(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::{network_difficulty_from_compact_target, network_target_from_compact_target};

    fn assert_near(actual: f64, expected: f64) {
        let relative_error = (actual - expected).abs() / expected;
        assert!(relative_error < 1e-12, "actual={actual}, expected={expected}, relative_error={relative_error}");
    }

    #[test]
    fn compact_target_uses_keryx_network_difficulty_convention() {
        // 0x1e7fffff is the reset/genesis-scale target used throughout the miner tests.
        assert_near(network_difficulty_from_compact_target(0x1e7f_ffff).unwrap(), 65_536.007_812_500_93);
        // Known stratum-v3 fixture from statum_codec.rs.
        assert_near(network_difficulty_from_compact_target(490_707_704).unwrap(), 33_762_627.830_873_9);
    }

    #[test]
    fn compact_target_rejects_zero_negative_and_above_network_maximum() {
        assert_eq!(network_difficulty_from_compact_target(0), None);
        assert_eq!(network_difficulty_from_compact_target(0x2080_0000), None);
        assert_eq!(network_difficulty_from_compact_target(0x2100_ffff), None);
        // Exponent 34 with a mantissa wider than one byte overflows 256 bits; the historical raw
        // decoder truncates it, while the checked network decoder must reject it.
        assert_eq!(network_target_from_compact_target(0x2200_0101), None);
    }
}
