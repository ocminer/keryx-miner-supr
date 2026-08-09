//! Per-GPU health sampling (NVML) for the hashrate reporter and the stats API.
//!
//! Field request: combine the plugin's `[GPU #N] temp/fan/power/clocks/vram` health line
//! with the per-device hashrate and an efficiency figure (MH/s per watt) so tuning for
//! efficiency doesn't require correlating two log lines by eye — and expose the same
//! numbers via the stats API.
//!
//! NVML is loaded lazily and EVERY failure degrades to `None`: on AMD/CPU rigs (no
//! libnvidia-ml), under broken drivers, or when the CUDA-ordinal→NVML-index mapping can't
//! be trusted, the reporter simply prints the classic hashrate-only line. We never guess:
//! the NVML device's name must match the worker label's device name, otherwise the sample
//! is discarded (CUDA enumerates fastest-first by default while NVML follows PCI order, so
//! a blind index lookup can attribute the wrong card's wattage — CUDA_DEVICE_ORDER=
//! PCI_BUS_ID aligns them, but we refuse to rely on it silently).

// NVML (`nvml-wrapper`) is a dependency only for cfg(not(target_os = "macos")) (see Cargo.toml —
// there is no NVIDIA on Apple Silicon), so the NVML sampler is compiled out on macOS. Everything
// below the sampler (GpuHealth, to_log_fragment, ordinal_from_label, efficiency_mhs_per_w) is
// NVML-independent and shared by every backend, including the Metal build.
#[cfg(not(target_os = "macos"))]
use std::sync::OnceLock;

#[cfg(not(target_os = "macos"))]
use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
#[cfg(not(target_os = "macos"))]
use nvml_wrapper::Nvml;

#[cfg(not(target_os = "macos"))]
static NVML: OnceLock<Option<Nvml>> = OnceLock::new();

#[cfg(not(target_os = "macos"))]
fn nvml() -> Option<&'static Nvml> {
    NVML.get_or_init(|| Nvml::init().ok()).as_ref()
}

/// One NVML sample for a device. All fields are best-effort; `power_w` is the one the
/// efficiency figure needs, the rest mirror the plugin's health line so a single log line
/// carries everything.
#[derive(Default, Clone, Debug)]
pub struct GpuHealth {
    pub temp_c: Option<u32>,
    pub fan_pct: Option<u32>,
    pub power_w: Option<f64>,
    pub core_mhz: Option<u32>,
    pub mem_mhz: Option<u32>,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
}

impl GpuHealth {
    /// Compact human form mirroring the keryxcuda health line:
    /// `62°C fan=99% 114.8W core=1530MHz mem=8001MHz vram=8084/16376MB`.
    pub fn to_log_fragment(&self) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(6);
        if let Some(t) = self.temp_c {
            parts.push(format!("{}°C", t));
        }
        if let Some(f) = self.fan_pct {
            parts.push(format!("fan={}%", f));
        }
        if let Some(p) = self.power_w {
            parts.push(format!("{:.1}W", p));
        }
        if let Some(c) = self.core_mhz {
            parts.push(format!("core={}MHz", c));
        }
        if let Some(m) = self.mem_mhz {
            parts.push(format!("mem={}MHz", m));
        }
        if let (Some(u), Some(t)) = (self.vram_used_mb, self.vram_total_mb) {
            parts.push(format!("vram={}/{}MB", u, t));
        }
        parts.join(" ")
    }
}

/// Sample the GPU at NVML index `ordinal`, but ONLY if its NVML name appears in
/// `expect_label` (the worker label, e.g. `#0 (NVIDIA RTX A4000)`). A name mismatch means
/// the CUDA and NVML orderings differ on this rig — return `None` rather than attribute
/// another card's power draw.
#[cfg(not(target_os = "macos"))]
pub fn sample(ordinal: u32, expect_label: &str) -> Option<GpuHealth> {
    let nvml = nvml()?;
    let dev = nvml.device_by_index(ordinal).ok()?;
    let name = dev.name().ok()?;
    if !expect_label.contains(&name) {
        return None;
    }
    let mem = dev.memory_info().ok();
    Some(GpuHealth {
        temp_c: dev.temperature(TemperatureSensor::Gpu).ok(),
        fan_pct: dev.fan_speed(0).ok(),
        power_w: dev.power_usage().ok().map(|mw| mw as f64 / 1000.0),
        core_mhz: dev.clock_info(Clock::Graphics).ok(),
        mem_mhz: dev.clock_info(Clock::Memory).ok(),
        vram_used_mb: mem.as_ref().map(|m| m.used / 1024 / 1024),
        vram_total_mb: mem.as_ref().map(|m| m.total / 1024 / 1024),
    })
}

/// macOS (Metal) has no NVML — the Apple GPU exposes no per-device power/clock telemetry through
/// this path, so health sampling degrades to `None` and the reporter prints the hashrate-only
/// line (identical to an AMD/CPU rig with no libnvidia-ml). Keeps the `miner.rs` call site
/// backend-agnostic.
#[cfg(target_os = "macos")]
pub fn sample(_ordinal: u32, _expect_label: &str) -> Option<GpuHealth> {
    None
}

/// Parse the CUDA ordinal out of a worker label like `#0 (NVIDIA RTX A4000)`.
pub fn ordinal_from_label(label: &str) -> Option<u32> {
    label.strip_prefix('#')?.split_whitespace().next()?.parse().ok()
}

/// Efficiency in MH/s per watt; `None` when either side is unknown or zero.
pub fn efficiency_mhs_per_w(hashrate_hs: f64, power_w: Option<f64>) -> Option<f64> {
    match power_w {
        Some(w) if w > 1.0 && hashrate_hs > 0.0 => Some(hashrate_hs / 1_000_000.0 / w),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_ordinal_parses() {
        assert_eq!(ordinal_from_label("#0 (NVIDIA RTX A4000)"), Some(0));
        assert_eq!(ordinal_from_label("#12 (NVIDIA GeForce RTX 5090)"), Some(12));
        assert_eq!(ordinal_from_label("CPU"), None);
    }

    #[test]
    fn efficiency_math_matches_field_example() {
        // The user's example: 18.87 Mhash/s at 114.8 W -> 0.164 MH/s/W.
        let eff = efficiency_mhs_per_w(18_870_000.0, Some(114.8)).unwrap();
        assert!((eff - 0.1644).abs() < 0.001, "got {eff}");
        assert_eq!(efficiency_mhs_per_w(18_870_000.0, None), None);
        assert_eq!(efficiency_mhs_per_w(0.0, Some(100.0)), None);
    }

    #[test]
    fn log_fragment_shape() {
        let h = GpuHealth {
            temp_c: Some(62),
            fan_pct: Some(99),
            power_w: Some(114.8),
            core_mhz: Some(1530),
            mem_mhz: Some(8001),
            vram_used_mb: Some(8084),
            vram_total_mb: Some(16376),
        };
        assert_eq!(h.to_log_fragment(), "62°C fan=99% 114.8W core=1530MHz mem=8001MHz vram=8084/16376MB");
        // Partial data still yields a clean fragment (no dangling separators).
        let sparse = GpuHealth { power_w: Some(50.0), ..Default::default() };
        assert_eq!(sparse.to_log_fragment(), "50.0W");
    }
}
