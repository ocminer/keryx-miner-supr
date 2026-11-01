//! Context-pad mining through the same model owner and scoped dispatch lifetime.
use super::{engine, sym, InflightGuard, WALK_INFLIGHT};
use std::ffi::{c_int, c_void};
type Mine = unsafe extern "C" fn(*mut c_void, *const u64, *const u64, *const u64, u64, u64, u32, *const u64, u64) -> i64;
type Read = unsafe extern "C" fn(*mut c_void, u64, *mut u8, usize) -> c_int;
type Release = unsafe extern "C" fn(*mut c_void);

#[derive(Clone, Copy)]
pub(super) struct Api { mine: Mine, read: Read, release: Release }
impl Api {
    pub(super) unsafe fn load(lib: &'static libloading::Library) -> Option<Self> {
        let abi = sym::<super::AbiFn>(lib, "keryx_llama_pom_v5_abi")?;
        if abi() != 1 { return None; }
        Some(Self { mine: sym(lib, "keryx_llama_pom_mine_v5")?, read: sym(lib, "keryx_llama_pom_pad_read")?, release: sym(lib, "keryx_llama_pom_release_pad")? })
    }
    pub(super) fn run(self, model: *mut c_void, p: [u64; 4], s: [u64; 4], time: u64, t: [u64; 4], base: u64, batch: u64, seed: &[u64; 4], tiles: u64, stop: &dyn Fn() -> bool) -> Result<(Option<u64>, u64), String> {
        if tiles == 0 { return Err("PoM v5: zero context-pad tiles".into()); }
        let sub_max = std::env::var("KERYX_VK_V5_SUB_DISPATCH").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(1024).clamp(8, 8192);
        let mut done = 0;
        while done < batch {
            if stop() { break; }
            let n = (batch - done).min(sub_max) as u32;
            let start = base.wrapping_add(done);
            let rc = unsafe { (self.mine)(model, p.as_ptr(), s.as_ptr(), t.as_ptr(), time, start, n, seed.as_ptr(), tiles) };
            match rc {
                -1 => done += n as u64,
                -3 => return Err("PoM v5: context pad allocation or pipeline unavailable".into()),
                nwin if nwin >= 0 && (nwin as u64) < n as u64 => return Ok((Some(start.wrapping_add(nwin as u64)), done + n as u64)),
                _ => return Err("PoM v5: Vulkan dispatch failed".into()),
            }
        }
        Ok((None, done))
    }
}

pub fn mine(p: [u64; 4], s: [u64; 4], time: u64, t: [u64; 4], base: u64, batch: u64, seed: &[u64; 4], tiles: u64, stop: &dyn Fn() -> bool) -> Result<(Option<u64>, u64), String> {
    let (model, api) = {
        let guard = engine().lock().map_err(|_| "Vulkan engine poisoned")?;
        let e = guard.as_ref().ok_or("no Vulkan model owner")?;
        let api = e.v5.ok_or("Vulkan sidecar lacks PoM v5 ABI 1")?;
        if stop() { return Ok((None, 0)); }
        WALK_INFLIGHT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        (e.model, api)
    };
    let _flight = InflightGuard;
    api.run(model, p, s, time, t, base, batch, seed, tiles, stop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pom, pom_v4 as v4, pom_v5 as v5};

    fn cpu_value(index: &pom::WeightIndex, pad: &[u8], seed: u64, pph: &[u8; 32]) -> [u8; 32] {
        cpu_value_with(index, pad.len() as u64 / 1024, seed, pph, |off| pad[off as usize * 1024..(off + 1) as usize * 1024].to_vec())
    }
    fn cpu_value_with(index: &pom::WeightIndex, np: u64, seed: u64, pph: &[u8; 32], mut read_pad: impl FnMut(u64) -> Vec<u8>) -> [u8; 32] {
        let nt = index.n_chunks / 32;
        let kp = v5::v5_pad_steps(nt, np);
        let mut state = v4::v4_initial_state(seed);
        let mut next = vec![0; state.len()];
        let mut tile = [0u8; 1024];
        let mut off = v5::v5_first_offset(seed, nt);
        for step in 1..=256 {
            if off < nt { index.read_chunks_into(off * 32, &mut tile); }
            else { tile.copy_from_slice(&read_pad(off - nt)); }
            if step < 256 { off = v5::v5_next_offset(seed, step, tile[..32].try_into().unwrap(), v5::v5_state_fold(&state), nt, np, kp); }
            v4::v4_transition_into(&mut next, &state, &tile, step as u32);
            std::mem::swap(&mut state, &mut next);
        }
        pom::pom_pow_value(v4::fold64(&v4::v4_state_root(&state)), pph, true)
    }
    fn words(b: &[u8; 32]) -> [u64; 4] { std::array::from_fn(|i| u64::from_le_bytes(b[i*8..i*8+8].try_into().unwrap())) }
    fn decrement(mut x: [u8; 32]) -> [u8; 32] {
        for b in &mut x { if *b != 0 { *b -= 1; return x; } *b = 255; }
        panic!("zero PoW value");
    }
    fn fixture(path: &std::path::Path) {
        // Small standalone GGUF: exercises segment boundaries without touching shared models.
        let mut data = b"GGUF".to_vec(); data.extend(3u32.to_le_bytes());
        data.extend(2u64.to_le_bytes()); data.extend(0u64.to_le_bytes());
        let mut offset = 0u64;
        for (name, elements) in [("a.weight", 8200u64), ("b.weight", 8184u64)] {
            data.extend((name.len() as u64).to_le_bytes()); data.extend(name.as_bytes());
            data.extend(1u32.to_le_bytes()); data.extend(elements.to_le_bytes());
            data.extend(0u32.to_le_bytes()); data.extend(offset.to_le_bytes()); offset += elements * 4;
        }
        data.resize((data.len() + 31) / 32 * 32, 0);
        let mut x = 0x8a12_7d0b_936f_aa15u64;
        for _ in 0..offset { x ^= x << 13; x ^= x >> 7; x ^= x << 17; data.push(x as u8); }
        std::fs::write(path, data).unwrap();
    }

    /// Explicit hardware gate: many pad steps, exact target boundaries, winner order,
    /// actual GPU pad bytes, seed replacement and release/refill. No production model.
    #[test]
    #[ignore]
    fn vk_v5_pad_and_walk_match_node() {
        crate::native_env::export_trace_env();
        let root = std::path::PathBuf::from(std::env::var("KERYX_TEST_OUTPUT").expect("KERYX_TEST_OUTPUT"));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("v5-fixture.gguf"); fixture(&path);
        let device = std::env::var("KERYX_LLAMA_VK_DEVICE").unwrap_or_else(|_| "0".into()).parse().unwrap();
        let pci = super::super::ggml_device_pci(device).expect("selected Vulkan GPU PCI");
        assert_eq!(format!("{:04x}:{:02x}:{:02x}.{:x}",pci.0,pci.1,pci.2,pci.3), std::env::var("KERYX_TEST_PCI").expect("KERYX_TEST_PCI"));
        let index = pom::WeightIndex::build_from_gguf(path.to_str().unwrap(), [0x52;32]).unwrap();
        let mut walk = super::super::MiningWalk::load(path.to_str().unwrap(), pci, &index).unwrap();
        let api = walk.v5.expect("PoM v5 native ABI");
        let np = 257u64; // non-power-of-two modulo, most steps land in the pad
        let pph = [0x45;32]; let time = 0x1020_3040_5060_7080;
        let p = pom::pph_words_for_era(&pph, true); let s = pom::pph_words(&pom::seed_h14_pph(&pph));
        for id in [0x63u8, 0x71] {
            let seed = v5::v5_pad_seed(0, &[id;32], &index.r_t); let sw = v5::v5_pad_seed_words(&seed);
            let pad: Vec<u8> = (0..np).flat_map(|i| v5::v5_pad_tile(&seed, i)).collect();
            for nonce in [0u64, 1, 7, 8, 31, 32, 65535, 65536, u64::MAX] {
                let value = cpu_value(&index, &pad, pom::pom_block_seed_h14(&pph,time,nonce), &pph);
                assert_eq!(walk.mine_v5(p,s,time,words(&value),nonce,1,&sw,np,&||false).unwrap(), (Some(nonce),1), "nonce {nonce} equality");
                assert_eq!(walk.mine_v5(p,s,time,words(&decrement(value)),nonce,1,&sw,np,&||false).unwrap(), (None,1), "nonce {nonce} below");
            }
            let mut read = vec![0u8; pad.len()];
            for (i, b) in read.chunks_mut(65536).enumerate() { assert_eq!(unsafe{(api.read)(walk.model,(i*65536) as u64,b.as_mut_ptr(),b.len())},0); }
            assert_eq!(read, pad, "all actual GPU pad bytes");
            let values: Vec<_> = (0..11).map(|n| cpu_value(&index,&pad,pom::pom_block_seed_h14(&pph,time,n),&pph)).collect();
            let target = values[6];
            let expected = values.iter().position(|v| v.iter().rev().cmp(target.iter().rev()).is_le()).map(|i|i as u64);
            assert_eq!(walk.mine_v5(p,s,time,words(&target),0,11,&sw,np,&||false).unwrap(), (expected,11));
            unsafe{(api.release)(walk.model)};
            let value = values[0];
            assert_eq!(walk.mine_v5(p,s,time,words(&value),0,1,&sw,np,&||false).unwrap(),(Some(0),1));
            assert_eq!(walk.mine_v5(p,s,time,[0;4],0,11,&sw,np,&||true).unwrap(),(None,0));
        }
        eprintln!("Vulkan PoM v5 pad bytes/state-fold/pad routing/target boundaries/refill PASS");
    }

    /// Exercise real allocations on both sides of the 1-GiB pad boundary without
    /// materializing a gigabyte-sized CPU reference. Each sampled tile and each
    /// full walk is independently derived with the node-matching scalar code.
    #[test]
    #[ignore]
    fn vk_v5_sharded_pad_matches_node() {
        crate::native_env::export_trace_env();
        let root = std::path::PathBuf::from(std::env::var("KERYX_TEST_OUTPUT").expect("KERYX_TEST_OUTPUT"));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("v5-shard-fixture.gguf"); fixture(&path);
        let device = std::env::var("KERYX_LLAMA_VK_DEVICE").unwrap_or_else(|_| "0".into()).parse().unwrap();
        let pci = super::super::ggml_device_pci(device).expect("selected Vulkan GPU PCI");
        assert_eq!(format!("{:04x}:{:02x}:{:02x}.{:x}",pci.0,pci.1,pci.2,pci.3), std::env::var("KERYX_TEST_PCI").expect("KERYX_TEST_PCI"));
        let index = pom::WeightIndex::build_from_gguf(path.to_str().unwrap(), [0x52;32]).unwrap();
        let mut walk = super::super::MiningWalk::load(path.to_str().unwrap(), pci, &index).unwrap();
        let api = walk.v5.expect("PoM v5 native ABI");
        let boundary = 1u64 << 20;
        let np = boundary + 257;
        let pph = [0x45;32]; let time = 0x1020_3040_5060_7080;
        let p = pom::pph_words_for_era(&pph, true); let s = pom::pph_words(&pom::seed_h14_pph(&pph));
        for id in [0x63u8, 0x71] {
            let seed = v5::v5_pad_seed(0, &[id;32], &index.r_t); let sw = v5::v5_pad_seed_words(&seed);
            for nonce in [0u64, 7, u64::MAX] {
                let value = cpu_value_with(&index, np, pom::pom_block_seed_h14(&pph,time,nonce), &pph, |off| v5::v5_pad_tile(&seed,off));
                assert_eq!(walk.mine_v5(p,s,time,words(&value),nonce,1,&sw,np,&||false).unwrap(),(Some(nonce),1));
                assert_eq!(walk.mine_v5(p,s,time,words(&decrement(value)),nonce,1,&sw,np,&||false).unwrap(),(None,1));
            }
            for tile in [0, boundary-1, boundary, np-1] {
                let mut actual = vec![0;1024];
                assert_eq!(unsafe{(api.read)(walk.model,tile*1024,actual.as_mut_ptr(),actual.len())},0);
                assert_eq!(actual,v5::v5_pad_tile(&seed,tile),"tile {tile}");
            }
            let mut crossing = vec![0;2048];
            assert_eq!(unsafe{(api.read)(walk.model,(boundary-1)*1024,crossing.as_mut_ptr(),crossing.len())},0);
            let expected: Vec<_> = [boundary-1,boundary].into_iter().flat_map(|tile|v5::v5_pad_tile(&seed,tile)).collect();
            assert_eq!(crossing,expected,"readback spanning two allocations");
            unsafe{(api.release)(walk.model)};
        }
        eprintln!("Vulkan PoM v5 multi-allocation pad/boundary readback/full walk/seed replacement PASS");
    }
}
