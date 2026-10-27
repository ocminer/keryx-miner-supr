//! CUDA ordinal → NVML device index.
//!
//! NVML numbers every GPU in the machine in PCI order and ignores `CUDA_VISIBLE_DEVICES`, while
//! the miner's device numbers are CUDA ordinals. With `CUDA_VISIBLE_DEVICES=1`, CUDA ordinal 0 is
//! NVML index 1 — a blind `device_by_index(0)` reports the OTHER card's power, temperature, fan
//! and VRAM (field report: a pinned second 5090 showed GPU0's 68 W / 46 °C). The two are matched
//! here by PCI domain/bus/device, which both APIs expose. When the CUDA side cannot be queried we
//! only fall back to the identity mapping if nothing can have remapped the CUDA devices.

#[cfg(not(target_os = "macos"))]
use std::{collections::HashMap, sync::Mutex, sync::OnceLock};

/// PCI bus id of CUDA ordinal `ordinal` in NVML's format (`00000000:07:00.0`).
#[cfg(feature = "pom-cuda")]
pub fn cuda_pci_bus_id(ordinal: u32) -> Option<String> {
    use candle_core::cuda_backend::cudarc::driver::{result, sys::CUdevice_attribute as A};
    result::init().ok()?;
    let dev = result::device::get(ordinal as i32).ok()?;
    // SAFETY: `dev` was returned by `cuDeviceGet` above.
    let attr = |a| unsafe { result::device::get_attribute(dev, a) }.ok();
    let domain = attr(A::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?;
    let bus = attr(A::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?;
    let device = attr(A::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?;
    Some(format!("{:08x}:{:02x}:{:02x}.0", domain, bus, device))
}

#[cfg(not(feature = "pom-cuda"))]
pub fn cuda_pci_bus_id(_ordinal: u32) -> Option<String> {
    None
}

/// NVML index of CUDA ordinal `ordinal`, or `None` when it cannot be determined reliably.
/// Cached per ordinal: the mapping is fixed for the life of the process.
#[cfg(not(target_os = "macos"))]
pub fn nvml_index_for_cuda_ordinal(nvml: &nvml_wrapper::Nvml, ordinal: u32) -> Option<u32> {
    static CACHE: OnceLock<Mutex<HashMap<u32, Option<u32>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hit) = cache.lock().ok().and_then(|c| c.get(&ordinal).copied()) {
        return hit;
    }
    let mapped = match cuda_pci_bus_id(ordinal) {
        Some(bus) => nvml.device_by_pci_bus_id(bus.as_str()).ok().and_then(|d| d.index().ok()),
        // No CUDA driver query (non-CUDA build or driver error): the identity mapping is only
        // trustworthy when nothing remaps or filters the CUDA device list.
        None if identity_mapping_safe() => Some(ordinal),
        None => None,
    };
    if let Ok(mut c) = cache.lock() {
        c.insert(ordinal, mapped);
    }
    mapped
}

#[cfg(not(target_os = "macos"))]
fn identity_mapping_safe() -> bool {
    std::env::var_os("CUDA_VISIBLE_DEVICES").is_none()
        && std::env::var("CUDA_DEVICE_ORDER").map(|v| v == "PCI_BUS_ID").unwrap_or(false)
}

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    #[test]
    fn identity_mapping_requires_pci_order_and_no_visible_devices() {
        // Only checks the pure policy; NVML/CUDA are not touched.
        let vis = std::env::var_os("CUDA_VISIBLE_DEVICES");
        let ord = std::env::var_os("CUDA_DEVICE_ORDER");
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        assert!(super::identity_mapping_safe());
        std::env::set_var("CUDA_VISIBLE_DEVICES", "1");
        assert!(!super::identity_mapping_safe());
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        std::env::remove_var("CUDA_DEVICE_ORDER");
        assert!(!super::identity_mapping_safe());
        match vis { Some(v) => std::env::set_var("CUDA_VISIBLE_DEVICES", v), None => std::env::remove_var("CUDA_VISIBLE_DEVICES") }
        match ord { Some(v) => std::env::set_var("CUDA_DEVICE_ORDER", v), None => std::env::remove_var("CUDA_DEVICE_ORDER") }
    }
}
