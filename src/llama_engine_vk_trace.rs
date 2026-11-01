//! H15 tracing on the existing Vulkan model owner. The parent engine mutex keeps
//! model lifetime, generation and replay serialized; the caller drains mining.
use super::{engine, sym, Engine};
use crate::trace::{SeqCapture, SeqLayer, REPLAY_MAX_CAPTURES};
use crate::vulkan_trace_seq::{parse_kv_state, seq_capture, seq_leaf_bytes, RecurSink, SeqAttn, SeqRecur, SeqSpec};
use std::ffi::{c_char, c_int, c_void, CStr, CString};

type Configure = unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int;
type Count = unsafe extern "C" fn(*mut c_void) -> c_int;
type Read = unsafe extern "C" fn(*mut c_void, *mut u8, usize, *mut u32, usize, *mut u32) -> c_int;
type Generate = unsafe extern "C" fn(*mut c_void, *const c_char, *const i32, c_int, c_int, *const c_char, c_int, *mut c_char, c_int, *mut i32, c_int, *mut c_int, *mut c_int, c_int, c_int) -> c_int;
type Replay = unsafe extern "C" fn(*mut c_void, *const i32, c_int, c_int, *const u32, *const u32, u32, u32, *mut u8, usize, *mut u32) -> c_int;
type SeqConfigure = unsafe extern "C" fn(*mut c_void, u32, u32, *const u32, u32, u32) -> c_int;
type Callback = unsafe extern "C" fn(*mut c_void, Option<unsafe extern "C" fn(*mut c_void, u32) -> c_int>, *mut c_void) -> c_int;
type Recurrent = unsafe extern "C" fn(*mut c_void, *mut u8, usize, *mut u8, usize, *mut u32, *mut u32) -> c_int;
type SeqState = unsafe extern "C" fn(*mut c_void, *mut u8, usize) -> usize;
type Lse = unsafe extern "C" fn(*mut c_void, *mut f32, usize, u32) -> c_int;
type SeqReplay = unsafe extern "C" fn(*mut c_void, *const i32, c_int, c_int) -> c_int;
type Error = unsafe extern "C" fn() -> *const c_char;
type Identity = unsafe extern "C" fn(*mut c_void, *mut c_char, usize, *mut c_char, usize) -> c_int;
const CAPTURE_CAP: usize = 4 << 20;

#[derive(Clone, Copy)]
struct Api {
    configure: Configure, count: Count, read: Read, generate: Generate, replay: Replay,
    sequence: SeqConfigure, callback: Callback, recurrent: Recurrent,
    state: SeqState, lse: Lse, seq_replay: SeqReplay, error: Error, identity: Identity,
}

pub(super) struct State {
    api: Api,
    manifest: Vec<u8>,
    leaves: usize,
    sequence: Option<SeqSpec>,
}

impl State {
    pub(super) unsafe fn load(lib: &'static libloading::Library) -> Option<Self> {
        let abi = sym::<super::AbiFn>(lib, "keryx_llama_vk_trace_abi")?;
        if abi() != 1 { return None; }
        Some(Self {
            api: Api {
                configure: sym(lib, "keryx_llama_trace_configure")?, count: sym(lib, "keryx_llama_trace_leaf_count")?,
                read: sym(lib, "keryx_llama_trace_read")?, generate: sym(lib, "keryx_llama_generate_traced")?,
                replay: sym(lib, "keryx_llama_replay_multi")?, sequence: sym(lib, "keryx_llama_vk_sequence_configure")?,
                callback: sym(lib, "keryx_llama_vk_batch_callback")?, recurrent: sym(lib, "keryx_llama_vk_recurrent_read")?,
                state: sym(lib, "keryx_llama_vk_sequence_state")?, lse: sym(lib, "keryx_llama_vk_lse_read")?,
                seq_replay: sym(lib, "keryx_llama_vk_replay_sequence")?, error: sym(lib, "keryx_llama_last_error")?,
                identity: sym(lib, "keryx_llama_vk_trace_identity")?,
            },
            manifest: Vec::new(), leaves: 0, sequence: None,
        })
    }
}

/// Physical Vulkan device and driver of the resident model. Never claim replay
/// compatibility from a different API's ordinal or an unknown (empty) identity.
pub fn gpu_identity() -> (String, u32, String) {
    let unknown = || (String::new(), 0, String::new());
    let Ok(g) = engine().lock() else { return unknown() };
    let Some(e) = g.as_ref() else { return unknown() };
    let Some(s) = e.trace.as_ref() else { return unknown() };
    let (mut name, mut driver) = ([0u8; 512], [0u8; 1024]);
    let rc = unsafe { (s.api.identity)(e.model, name.as_mut_ptr().cast(), name.len(), driver.as_mut_ptr().cast(), driver.len()) };
    if rc != 0 { return unknown(); }
    let text = |b: &[u8]| String::from_utf8_lossy(&b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())]).into_owned();
    (text(&name), 0, text(&driver))
}

fn detail(api: Api, context: &str) -> String {
    let p = unsafe { (api.error)() };
    let d = if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() };
    format!("{context}: {d}")
}

pub fn trace_configure(gguf: &str, _gpu: usize, manifest: &[u8]) -> Result<usize, String> {
    if crate::trace::TierManifest::parse(manifest).is_none() { return Err("invalid trace manifest".into()); }
    let mut guard = engine().lock().map_err(|_| "Vulkan engine poisoned")?;
    let e = guard.as_mut().filter(|e| e.gguf == gguf).ok_or("no Vulkan engine hosts this model")?;
    let s = e.trace.as_mut().ok_or("Vulkan sidecar lacks H15 trace ABI 1")?;
    // Clear host capability before configuring: a failed native reconfiguration cannot retain it.
    s.leaves = 0;
    s.manifest.clear();
    s.sequence = None;
    let rc = unsafe { (s.api.configure)(e.model, manifest.as_ptr().cast(), manifest.len(), 0) };
    if rc != 0 { return Err(detail(s.api, "Vulkan trace configuration failed")); }
    let n = unsafe { (s.api.count)(e.model) };
    if n <= 0 { return Err("Vulkan trace has no leaves".into()); }
    s.leaves = n as usize;
    s.manifest = manifest.to_vec();
    Ok(s.leaves)
}

pub fn traced_with(gguf: &str, _gpu: usize, manifest: &[u8]) -> bool {
    engine().lock().map_or(false, |g| g.as_ref().is_some_and(|e| e.gguf == gguf && e.trace.as_ref().is_some_and(|s| s.leaves > 0 && s.manifest == manifest)))
}

fn sequence_spec(manifest: &[u8], layers: &[SeqLayer], n_attn: u32, n_recur: u32, heads: u32, interval: u32) -> Result<SeqSpec, String> {
    use keryx_verifiable_inference::manifest::{self, Chain};
    let (entries, _) = manifest::read_file(manifest).ok_or("invalid sequence manifest")?;
    let (mut attn, mut recur) = (Vec::new(), Vec::new());
    for (entry, _) in entries {
        let Chain::Layer(layer) = entry.chain else { continue };
        if let Some(a) = layer.attn {
            if a.softcap != 0 {
                return Err(format!("Vulkan attention layer {} needs additional trace support", layer.layer));
            }
            attn.push(SeqAttn { swa: a.swa_window > 0, mla: a.mla_rank > 0, layer: layer.layer, slot: a.slot, n_head: a.n_head, n_head_kv: a.n_head_kv, head_dim: a.head_dim });
        }
        if let Some(r) = layer.recur {
            if !matches!(r.kind, manifest::RECUR_GDA | manifest::RECUR_KDA) { return Err(format!("unsupported Vulkan recurrent layer {}", layer.layer)); }
            recur.push(SeqRecur { layer: layer.layer, slot: r.slot, kind: if r.kind == manifest::RECUR_GDA { 3 } else { 4 }, n_v_heads: r.n_v_heads, head_dim: r.head_dim });
        }
    }
    attn.sort_by_key(|a| a.slot);
    recur.sort_by_key(|r| r.slot);
    let spec = SeqSpec { n_attn, n_recur, heads, attn, recur, interval };
    spec.check().map_err(|e| e.to_string())?;
    // Bind the independently supplied node layout to the native manifest we actually armed.
    let mut expected: Vec<SeqLayer> = spec.attn.iter().map(|a| [a.layer, if a.mla { 2 } else { 1 }, a.slot, a.n_head_kv, a.head_dim]).chain(spec.recur_rows()).collect();
    let mut actual = layers.to_vec();
    expected.sort_unstable(); actual.sort_unstable();
    if expected != actual { return Err("sequence layout differs from armed manifest".into()); }
    Ok(spec)
}

pub fn seq_configure(gguf: &str, _gpu: usize, layers: &[SeqLayer], n_attn: u32, n_recur: u32, heads: u32, interval: u32) -> Result<usize, String> {
    let mut guard = engine().lock().map_err(|_| "Vulkan engine poisoned")?;
    let e = guard.as_mut().filter(|e| e.gguf == gguf).ok_or("no Vulkan engine hosts this model")?;
    let s = e.trace.as_mut().filter(|s| s.leaves > 0).ok_or("no configured Vulkan trace")?;
    let spec = if layers.is_empty() { None } else { Some(sequence_spec(&s.manifest, layers, n_attn, n_recur, heads, interval)?) };
    if s.sequence == spec { return Ok(spec.as_ref().map_or(0, SeqSpec::n_leaves)); }
    let (nl, nh) = spec.as_ref().map_or((0, 0), SeqSpec::lse_shape);
    let rows = spec.as_ref().map_or_else(Vec::new, SeqSpec::recur_rows);
    s.sequence = None;
    let rc = unsafe { (s.api.sequence)(e.model, nl, nh, rows.as_ptr().cast(), rows.len() as u32, interval) };
    if rc != 0 { return Err(detail(s.api, "Vulkan sequence configuration failed")); }
    let n = spec.as_ref().map_or(0, SeqSpec::n_leaves);
    s.sequence = spec;
    Ok(n)
}

/// Callback data lives at a stable address until the native synchronous call returns.
/// Drop removes the native pointer even on every Rust early-return or panic path.
struct Capture {
    model: *mut c_void, api: Api, sink: Option<RecurSink>, error: Option<String>,
}
impl Capture {
    fn install(e: &Engine, s: &State, keep: &[u32]) -> Result<Box<Self>, String> {
        let mut c = Box::new(Self { model: e.model, api: s.api, sink: s.sequence.as_ref().and_then(|s| RecurSink::new(s, keep)), error: None });
        let rc = unsafe { (s.api.callback)(e.model, Some(after_batch), (&mut *c as *mut Self).cast()) };
        if rc != 0 { return Err("cannot install Vulkan trace capture".into()); }
        Ok(c)
    }
    fn check(&self, n_pos: usize) -> Result<(), String> {
        if let Some(error) = &self.error { return Err(error.clone()); }
        if let Some(s) = &self.sink { s.check_end(n_pos).map_err(|e| e.to_string())?; }
        Ok(())
    }
}
impl Drop for Capture {
    fn drop(&mut self) { unsafe { (self.api.callback)(self.model, None, std::ptr::null_mut()); } }
}
unsafe extern "C" fn after_batch(data: *mut c_void, count: u32) -> c_int {
    if data.is_null() { return -1; }
    let c = &mut *data.cast::<Capture>();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<(), String> {
        if let Some(sink) = c.sink.as_mut() {
            sink.after_batch_with(count as usize, |records, checkpoints| {
                let (mut n, mut checkpoint) = (0, 0);
                let rc = (c.api.recurrent)(c.model, records.as_mut_ptr(), records.len(), checkpoints.as_mut_ptr(), checkpoints.len(), &mut n, &mut checkpoint);
                if rc != 0 { return Err(anyhow::anyhow!("Vulkan recurrent read failed ({rc})")); }
                Ok((n, checkpoint))
            }).map_err(|e| e.to_string())?;
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => { c.error = Some(e); -1 },
        Err(_) => { c.error = Some("panic in Vulkan sequence capture".into()); -1 },
    }
}

fn read_sequence(e: &Engine, s: &State, n_pos: usize) -> Result<Vec<Vec<u8>>, String> {
    let spec = s.sequence.as_ref().ok_or("sequence capture not armed")?;
    let size = unsafe { (s.api.state)(e.model, std::ptr::null_mut(), 0) };
    if size == 0 || size > 8usize << 30 { return Err(format!("invalid Vulkan cache state size {size}")); }
    let mut state = Vec::new();
    state.try_reserve_exact(size).map_err(|e| format!("Vulkan cache readback allocation: {e}"))?;
    state.resize(size, 0u8);
    if unsafe { (s.api.state)(e.model, state.as_mut_ptr(), state.len()) } != size { return Err("Vulkan cache readback incomplete".into()); }
    let cache = parse_kv_state(&state, n_pos, spec).map_err(|e| e.to_string())?;
    let (layers, heads) = spec.lse_shape();
    let n_lse = (layers as usize).checked_mul(n_pos).and_then(|n| n.checked_mul(heads as usize)).ok_or("Vulkan sequence size overflow")?;
    let mut lse = vec![0.0f32; n_lse];
    let rc = unsafe { (s.api.lse)(e.model, lse.as_mut_ptr(), lse.len(), n_pos as u32) };
    if rc != 0 { return Err(detail(s.api, "Vulkan attention mass read failed")); }
    seq_leaf_bytes(spec, &cache, &lse, n_pos).map_err(|e| e.to_string())
}

#[derive(Debug)]
pub enum GenError { Unavailable, PromptTooLong(String), PromptHead(String), Failed(String) }
pub struct PromptPrefix<'a> { pub tokens: &'a [i32], pub head_tokens: u32, pub rest: &'a str }
pub struct TracedGeneration {
    pub text: String, pub tokens: Vec<i32>, pub n_prompt: u32, pub n_pos: u32,
    pub ubatch: Vec<u32>, pub leaves: Vec<[u8; 32]>, pub seq: Option<Vec<[u8; 32]>>,
}

pub fn generate_traced(gguf: &str, _gpu: usize, prompt: &str, prefix: Option<PromptPrefix<'_>>, max_tokens: usize, min_prompt: u32, answer_bytes_per_position: u32) -> Result<TracedGeneration, GenError> {
    let guard = engine().lock().map_err(|_| GenError::Unavailable)?;
    let e = guard.as_ref().filter(|e| e.gguf == gguf).ok_or(GenError::Unavailable)?;
    let s = e.trace.as_ref().filter(|s| s.leaves > 0).ok_or(GenError::Unavailable)?;
    let cp = CString::new(prompt).map_err(|_| GenError::Failed("prompt contains NUL".into()))?;
    let rest = CString::new(prefix.as_ref().map_or("", |p| p.rest)).map_err(|_| GenError::Failed("prompt contains NUL".into()))?;
    if max_tokens > 2048 || min_prompt > c_int::MAX as u32 || answer_bytes_per_position > c_int::MAX as u32 { return Err(GenError::Failed("invalid traced generation bounds".into())); }
    let (ptr, np, head) = match prefix {
        Some(p) if p.tokens.len() <= c_int::MAX as usize && p.head_tokens <= c_int::MAX as u32 => (p.tokens.as_ptr(), p.tokens.len() as c_int, p.head_tokens as c_int),
        Some(_) => return Err(GenError::Failed("invalid prefix bounds".into())),
        None => (std::ptr::null(), 0, 0),
    };
    let capture = Capture::install(e, s, &[]).map_err(GenError::Failed)?;
    let mut buf = vec![0u8; 64 * 1024];
    let mut tokens = vec![0i32; e.n_ctx.max(1) as usize];
    let (mut n_prompt, mut n_fed) = (0, 0);
    let n = unsafe { (s.api.generate)(e.model, cp.as_ptr(), ptr, np, head, rest.as_ptr(), max_tokens as c_int, buf.as_mut_ptr().cast(), buf.len() as c_int, tokens.as_mut_ptr(), tokens.len() as c_int, &mut n_prompt, &mut n_fed, min_prompt as c_int, answer_bytes_per_position as c_int) };
    if n < 0 {
        let d = capture.error.clone().unwrap_or_else(|| detail(s.api, "Vulkan traced generation failed"));
        return Err(match n { -2 => GenError::PromptTooLong(d), -8 => GenError::PromptHead(d), _ => GenError::Failed(d) });
    }
    if n as usize >= buf.len() || n_prompt <= 0 || n_fed < n_prompt || n_fed as usize > tokens.len() { return Err(GenError::Failed("invalid native trace lengths".into())); }
    buf.truncate(n as usize); tokens.truncate(n_fed as usize);
    let text = String::from_utf8(buf).map_err(|_| GenError::Failed("generated text is not UTF-8".into()))?;
    capture.check(tokens.len()).map_err(GenError::Failed)?;
    let mut leaves = vec![[0u8; 32]; tokens.len().checked_mul(s.leaves).ok_or_else(|| GenError::Failed("trace size overflow".into()))?];
    let mut ubatch = vec![0u32; tokens.len()];
    let mut n_pos = 0;
    let rc = unsafe { (s.api.read)(e.model, leaves.as_mut_ptr().cast(), leaves.len() * 32, ubatch.as_mut_ptr(), ubatch.len(), &mut n_pos) };
    if rc != 0 || n_pos as usize != tokens.len() { return Err(GenError::Failed(detail(s.api, "Vulkan trace read incomplete"))); }
    let seq = if s.sequence.is_some() {
        let all = read_sequence(e, s, tokens.len()).map_err(GenError::Failed)?;
        let mut roots: Vec<_> = all.iter().map(|b| *blake3::hash(b).as_bytes()).collect();
        if let Some(sink) = &capture.sink { sink.hashes_into(&mut roots); }
        Some(roots)
    } else { None };
    Ok(TracedGeneration { text, tokens, n_prompt: n_prompt as u32, n_pos, ubatch, leaves, seq })
}

pub fn replay_multi(gguf: &str, _gpu: usize, tokens: &[i32], n_prompt: u32, caps: &[(u32, u32)], slot_bytes: usize) -> Result<Vec<Vec<u8>>, String> {
    if caps.is_empty() || caps.len() > REPLAY_MAX_CAPTURES || slot_bytes == 0 || slot_bytes > CAPTURE_CAP || slot_bytes % 4 != 0 || n_prompt == 0 || n_prompt as usize > tokens.len() || tokens.len() > c_int::MAX as usize { return Err("invalid Vulkan replay request".into()); }
    let guard = engine().lock().map_err(|_| "Vulkan engine poisoned")?;
    let e = guard.as_ref().filter(|e| e.gguf == gguf).ok_or("no Vulkan engine hosts this model")?;
    let s = e.trace.as_ref().filter(|s| s.leaves > 0).ok_or("no traced Vulkan engine")?;
    if caps.iter().any(|&(p, l)| p as usize >= tokens.len() || l as usize >= s.leaves) { return Err("capture outside Vulkan trace".into()); }
    let pos: Vec<_> = caps.iter().map(|c| c.0).collect();
    let leaf: Vec<_> = caps.iter().map(|c| c.1).collect();
    let mut out = vec![0u8; caps.len() * slot_bytes];
    let mut lens = vec![0u32; caps.len()];
    let rc = unsafe { (s.api.replay)(e.model, tokens.as_ptr(), tokens.len() as c_int, n_prompt as c_int, pos.as_ptr(), leaf.as_ptr(), caps.len() as u32, slot_bytes as u32, out.as_mut_ptr(), out.len(), lens.as_mut_ptr()) };
    if rc != 0 { return Err(detail(s.api, "Vulkan product replay failed")); }
    if lens.iter().any(|&l| l as usize > slot_bytes) { return Err("native Vulkan capture exceeds slot".into()); }
    Ok(out.chunks(slot_bytes).zip(lens).map(|(b, n)| b[..n as usize].to_vec()).collect())
}

pub fn seq_replay(gguf: &str, _gpu: usize, tokens: &[i32], n_prompt: u32, spans: &[(u32, u64, u64)], max_chunks: u32) -> Result<Vec<SeqCapture>, String> {
    if n_prompt == 0 || n_prompt as usize > tokens.len() || tokens.len() > c_int::MAX as usize || spans.len() > REPLAY_MAX_CAPTURES || spans.iter().any(|s| s.2 < s.1) { return Err("invalid Vulkan sequence replay".into()); }
    let guard = engine().lock().map_err(|_| "Vulkan engine poisoned")?;
    let e = guard.as_ref().filter(|e| e.gguf == gguf).ok_or("no Vulkan engine hosts this model")?;
    let s = e.trace.as_ref().filter(|s| s.leaves > 0).ok_or("no traced Vulkan engine")?;
    let spec = s.sequence.as_ref().ok_or("no Vulkan sequence configured")?;
    let keep: Vec<_> = spans.iter().map(|s| s.0).collect();
    if keep.iter().any(|&l| l as usize >= spec.n_leaves()) { return Err("sequence capture outside trace".into()); }
    let capture = Capture::install(e, s, &keep)?;
    let rc = unsafe { (s.api.seq_replay)(e.model, tokens.as_ptr(), tokens.len() as c_int, n_prompt as c_int) };
    if rc != 0 { return Err(capture.error.clone().unwrap_or_else(|| detail(s.api, "Vulkan sequence replay failed"))); }
    capture.check(tokens.len())?;
    let all = read_sequence(e, s, tokens.len())?;
    spans.iter().map(|&(leaf, lo, hi)| {
        let bytes = capture.sink.as_ref().and_then(|s| s.kept(leaf)).unwrap_or(&all[leaf as usize]);
        let cap = seq_capture(bytes, lo, hi).map_err(|e| e.to_string())?;
        if cap.cvs.len() > max_chunks as usize { return Err("sequence capture exceeds chunk bound".into()); }
        Ok(cap)
    }).collect()
}
