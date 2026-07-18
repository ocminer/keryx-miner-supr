//! Stratum miner-telemetry extension: `mining.hello` (static rig identity, sent once) +
//! `mining.telemetry` (dynamic per-GPU metrics, sent periodically). Per
//! `KERYX_STRATUM_TELEMETRY_PROPOSAL_v0.7.0.md` (schema `v:1`).
//!
//! TRUST: every field here is DISPLAY / OPS ONLY and attacker-spoofable — it MUST NOT influence
//! reward, tier, difficulty, PoM verification, payout, or any consensus/accounting path. It is the
//! miner's *outbound* self-report, exactly like `user_agent`.
//!
//! Robustness: every source (NVML, `/proc`, model state) is wrapped; a field the miner can't read
//! is OMITTED rather than sent as junk, and no failure here ever propagates to the mining loop.

use serde_json::{json, Map, Value};

/// Telemetry schema version (bump only on a coordinated pool+miner change).
pub const SCHEMA_VERSION: u32 = 1;

/// Cap the serialized `mining.hello` object (proposal §2: ~8 KB) so a pool never has to parse a
/// runaway blob.
const MAX_HELLO_BYTES: usize = 8192;

/// Default `mining.telemetry` send interval (seconds). Overridable via `KERYX_TELEMETRY_INTERVAL`;
/// the pool also enforces a server-side minimum (≥30 s).
pub const DEFAULT_INTERVAL_SECS: u64 = 120;
/// Never send faster than this regardless of the configured interval (protects the pool DB).
pub const MIN_INTERVAL_SECS: u64 = 30;

/// Resolve the telemetry send interval from `KERYX_TELEMETRY_INTERVAL` (clamped to ≥ MIN).
pub fn interval_secs() -> u64 {
    std::env::var("KERYX_TELEMETRY_INTERVAL")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .map(|n| n.max(MIN_INTERVAL_SECS))
        .unwrap_or(DEFAULT_INTERVAL_SECS)
}

/// Telemetry is on by default; `KERYX_TELEMETRY_DISABLE=1` turns it off entirely (never sends
/// `mining.hello`/`mining.telemetry`).
pub fn enabled() -> bool {
    !matches!(std::env::var("KERYX_TELEMETRY_DISABLE").ok().as_deref(), Some("1") | Some("true"))
}

// ── mining.hello (STATIC) ──────────────────────────────────────────────────────────────────────

/// Build the `mining.hello` object. `loaded_model` = `(tier, model_id_hex)` for the resident PoM
/// model, if any. Result is size-capped to `MAX_HELLO_BYTES`.
pub fn build_hello(loaded_model: Option<(u8, String)>) -> Value {
    let mut o = Map::new();
    o.insert("v".into(), json!(SCHEMA_VERSION));
    o.insert(
        "miner".into(),
        json!({
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
            "build": build_flavour(),
            "features": features(),
        }),
    );
    o.insert("host".into(), host_static());
    let gpus = gpu_static();
    if !gpus.is_empty() {
        o.insert("gpus".into(), Value::Array(gpus));
    }
    o.insert(
        "model".into(),
        match loaded_model {
            Some((tier, model_id)) => json!({ "loaded": true, "tier": tier, "model_id": model_id }),
            None => json!({ "loaded": false }),
        },
    );

    let mut v = Value::Object(o);
    // Size cap: if somehow oversized (many GPUs / long strings), drop the optional host block first
    // (the gpus array is the operationally valuable part).
    if serde_json::to_string(&v).map(|s| s.len()).unwrap_or(0) > MAX_HELLO_BYTES {
        if let Some(obj) = v.as_object_mut() {
            obj.remove("host");
        }
    }
    v
}

/// Build-flavour string (walk/inference backend compiled into this binary).
fn build_flavour() -> &'static str {
    #[cfg(feature = "pom-cuda")]
    {
        "cuda"
    }
    #[cfg(all(feature = "pom-opencl", not(feature = "pom-cuda")))]
    {
        "opencl"
    }
    #[cfg(all(feature = "pom-metal", not(feature = "pom-cuda"), not(feature = "pom-opencl")))]
    {
        "metal"
    }
    #[cfg(not(any(feature = "pom-cuda", feature = "pom-opencl", feature = "pom-metal")))]
    {
        "cpu"
    }
}

/// Compiled-in feature tags (display only).
fn features() -> Vec<&'static str> {
    let mut f = vec!["pom-v2", "opoi"]; // this build carries H4 proof-v2 + OPoI inference
    #[cfg(feature = "pom-cuda")]
    f.push("pom-cuda");
    #[cfg(feature = "pom-opencl")]
    f.push("pom-opencl");
    #[cfg(feature = "pom-metal")]
    f.push("pom-metal");
    f
}

/// Static host block. Everything best-effort; unreadable fields are omitted.
fn host_static() -> Value {
    let mut h = Map::new();
    h.insert("os".into(), json!(std::env::consts::OS));
    h.insert("cpu_cores".into(), json!(num_cpus::get()));

    #[cfg(target_os = "linux")]
    {
        if let Some(v) = os_release_pretty_name() {
            h.insert("os_version".into(), json!(v));
        }
        if let Ok(k) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
            h.insert("kernel".into(), json!(k.trim()));
        }
        if let Some(cpu) = proc_cpuinfo_model() {
            h.insert("cpu".into(), json!(cpu));
        }
        if let Some(mb) = proc_meminfo_total_mb() {
            h.insert("ram_mb".into(), json!(mb));
        }
        // Privacy: hostname is opt-in (KERYX_TELEMETRY_HOSTNAME=1).
        if std::env::var("KERYX_TELEMETRY_HOSTNAME").as_deref() == Ok("1") {
            if let Ok(name) = std::fs::read_to_string("/proc/sys/kernel/hostname") {
                h.insert("hostname".into(), json!(name.trim()));
            }
        }
    }
    Value::Object(h)
}

#[cfg(target_os = "linux")]
fn os_release_pretty_name() -> Option<String> {
    let s = std::fs::read_to_string("/etc/os-release").ok()?;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("PRETTY_NAME=") {
            return Some(v.trim().trim_matches('"').to_string());
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn proc_cpuinfo_model() -> Option<String> {
    let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("model name") {
            if let Some(idx) = v.find(':') {
                return Some(v[idx + 1..].trim().to_string());
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn proc_meminfo_total_mb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            let kb: u64 = v.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024);
        }
    }
    None
}

// ── mining.telemetry (DYNAMIC) ─────────────────────────────────────────────────────────────────

/// Build the `mining.telemetry` object. `per_gpu_hashrate[i]` is the miner's own H/s for device `i`
/// (from `MinerStats.devices`, matched to NVML device `i` by index — best-effort).
pub fn build_telemetry(
    uptime_s: u64,
    total_hashrate: f64,
    per_gpu_hashrate: &[f64],
    acc: u64,
    rej: u64,
    stale: u64,
) -> Value {
    let mut o = Map::new();
    o.insert("v".into(), json!(SCHEMA_VERSION));
    o.insert("uptime_s".into(), json!(uptime_s));
    o.insert("hashrate_total".into(), json!(total_hashrate.round() as u64));
    o.insert("local_shares".into(), json!({ "acc": acc, "rej": rej, "stale": stale }));
    let gpus = gpu_dynamic(per_gpu_hashrate);
    if !gpus.is_empty() {
        o.insert("gpus".into(), Value::Array(gpus));
    }
    Value::Object(o)
}

// ── NVML GPU enumeration (best-effort; empty on any failure or non-NVIDIA host) ─────────────────

#[cfg(not(target_os = "macos"))]
fn gpu_static() -> Vec<Value> {
    use nvml_wrapper::{enum_wrappers::device::TemperatureSensor, Nvml};
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return Vec::new(),
    };
    let driver = nvml.sys_driver_version().ok();
    let cuda_rt = nvml.sys_cuda_driver_version().ok().map(|v| {
        // NVML encodes as (major*1000 + minor*10), e.g. 12040 -> "CUDA 12.4".
        format!("CUDA {}.{}", v / 1000, (v % 1000) / 10)
    });
    let count = nvml.device_count().unwrap_or(0);
    let mut out = Vec::new();
    for i in 0..count {
        let dev = match nvml.device_by_index(i) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let mut g = Map::new();
        g.insert("idx".into(), json!(i));
        g.insert("vendor".into(), json!("NVIDIA"));
        if let Ok(name) = dev.name() {
            g.insert("model".into(), json!(name));
        }
        if let Ok(uuid) = dev.uuid() {
            g.insert("uuid".into(), json!(uuid));
        }
        if let Ok(pci) = dev.pci_info() {
            g.insert("pci".into(), json!(pci.bus_id));
        }
        if let Ok(mem) = dev.memory_info() {
            g.insert("vram_mb".into(), json!(mem.total / 1024 / 1024));
        }
        if let Some(d) = &driver {
            g.insert("driver".into(), json!(d));
        }
        if let Some(rt) = &cuda_rt {
            g.insert("runtime".into(), json!(rt));
        }
        if let Ok(cc) = dev.cuda_compute_capability() {
            g.insert("compute".into(), json!(format!("{}.{}", cc.major, cc.minor)));
        }
        // touch a sensor so a card that enumerates but can't be queried is still listed w/o error
        let _ = dev.temperature(TemperatureSensor::Gpu);
        out.push(Value::Object(g));
    }
    out
}

#[cfg(not(target_os = "macos"))]
fn gpu_dynamic(per_gpu_hashrate: &[f64]) -> Vec<Value> {
    use nvml_wrapper::{
        enum_wrappers::device::{Clock, TemperatureSensor},
        Nvml,
    };
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(_) => return Vec::new(),
    };
    let count = nvml.device_count().unwrap_or(0);
    let mut out = Vec::new();
    for i in 0..count {
        let dev = match nvml.device_by_index(i) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let mut g = Map::new();
        g.insert("idx".into(), json!(i));
        if let Some(hr) = per_gpu_hashrate.get(i as usize) {
            g.insert("hashrate".into(), json!(hr.round() as u64));
        }
        if let Ok(t) = dev.temperature(TemperatureSensor::Gpu) {
            g.insert("temp_c".into(), json!(t));
        }
        if let Ok(f) = dev.fan_speed(0) {
            g.insert("fan_pct".into(), json!(f));
        }
        if let Ok(p) = dev.power_usage() {
            g.insert("power_w".into(), json!(p / 1000)); // milliwatts -> watts
        }
        if let Ok(c) = dev.clock_info(Clock::Graphics) {
            g.insert("core_mhz".into(), json!(c));
        }
        if let Ok(c) = dev.clock_info(Clock::Memory) {
            g.insert("mem_mhz".into(), json!(c));
        }
        if let Ok(u) = dev.utilization_rates() {
            g.insert("util_pct".into(), json!(u.gpu));
        }
        out.push(Value::Object(g));
    }
    out
}

// macOS: no NVIDIA/NVML — GPU blocks are simply omitted (best-effort per the proposal).
#[cfg(target_os = "macos")]
fn gpu_static() -> Vec<Value> {
    Vec::new()
}
#[cfg(target_os = "macos")]
fn gpu_dynamic(_per_gpu_hashrate: &[f64]) -> Vec<Value> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_is_wellformed_and_bounded() {
        let v = build_hello(Some((3, "aa".repeat(32))));
        assert_eq!(v["v"], json!(SCHEMA_VERSION));
        assert_eq!(v["miner"]["name"], json!("keryx-miner-supr"));
        assert_eq!(v["model"]["loaded"], json!(true));
        assert_eq!(v["model"]["tier"], json!(3));
        // never exceeds the pool's parse cap
        assert!(serde_json::to_string(&v).unwrap().len() <= MAX_HELLO_BYTES);
    }

    #[test]
    fn hello_no_model() {
        let v = build_hello(None);
        assert_eq!(v["model"]["loaded"], json!(false));
    }

    #[test]
    fn telemetry_is_wellformed() {
        let v = build_telemetry(3600, 65_000_000.0, &[65_000_000.0], 1234, 5, 2);
        assert_eq!(v["v"], json!(SCHEMA_VERSION));
        assert_eq!(v["uptime_s"], json!(3600));
        assert_eq!(v["hashrate_total"], json!(65_000_000u64));
        assert_eq!(v["local_shares"]["acc"], json!(1234));
        assert_eq!(v["local_shares"]["rej"], json!(5));
    }

    #[test]
    fn interval_clamps_to_min() {
        std::env::set_var("KERYX_TELEMETRY_INTERVAL", "5");
        assert_eq!(interval_secs(), MIN_INTERVAL_SECS);
        std::env::remove_var("KERYX_TELEMETRY_INTERVAL");
    }
}
