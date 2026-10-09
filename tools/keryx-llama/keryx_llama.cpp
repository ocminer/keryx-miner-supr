// libkeryx-llama.{so,dylib} — the miner's in-process llama.cpp engine (candle-independence Phase 2
// on CUDA, Phase 3b on Apple Silicon Metal).
//
// One llama.cpp instance per loaded model: it OWNS the resident GGUF copy on the inference GPU
// and exposes (a) per-tensor device pointers so the PoM walk gathers straight over the SAME VRAM
// (zero-dup — proven byte-identical to the on-disk GGUF by tools/llama_zerodup_spike on CUDA), and
// (b) text generation for OPoI. On Apple Silicon (Metal) the walk uses its own packed buffer
// (`pom_gpu_metal` Phase 3a) so the tensor-pointer contract there only feeds the future zero-dup
// Metal walk; today it just satisfies the loader-side count/name enumeration.
//
// The miner dlopens this next to its own binary; absent = other GPU routes are tried, while
// candle-CPU remains an explicit, deprecated emergency fallback only.
// Built by hiveos/build-keryx-llama.sh (CUDA) or hiveos/build-keryx-llama-macos.sh (Metal).
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#include "gguf.h"
#ifdef __APPLE__
// Metal: llama.cpp's ggml-metal backend stores quantized tensors in unified-memory MTLBuffers.
// `t->data` is a CPU-readable pointer into that unified memory (also GPU-visible on Apple Silicon
// via the shared address space), so we don't need cudaPointerGetAttributes — `is_device` is
// always 1 for tensors llama.cpp reports.
#else
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// Lowest CUDA compute capability (major*10+minor) whose SASS/PTX this build carries. Set by the
// build scripts from the first CMAKE_CUDA_ARCHITECTURES entry (modern 75, legacy 61). 0 = no gate.
#ifndef KERYX_LLAMA_MIN_CC
#define KERYX_LLAMA_MIN_CC 0
#endif

// Last load failure, for the host's log (optional ABI export `keryx_llama_last_error`).
static std::mutex g_err_mu;
static std::string g_last_error;
static void keryx_set_last_error(const std::string& e) {
    std::lock_guard<std::mutex> g(g_err_mu);
    g_last_error = e;
    fprintf(stderr, "[keryx-llama] %s\n", e.c_str());
}

struct KeryxLlama {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    llama_sampler* smpl  = nullptr;
    std::vector<std::string> names; // canonical (byte-lexicographic) order — matches pom.rs
    std::mutex gen_lock;
    std::string ctx_info;           // "n_ctx=… kv=… flash_attn=…" for the miner log
};

// GLM-4-0414 rotates 64 of its 128 head dimensions; a GGUF missing the key rotates all of them
// (upstream v0.5.7 3f2b2ac).
static bool keryx_needs_glm4_rope_fix(const char* gguf_path) {
    gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context* g = gguf_init_from_file(gguf_path, params);
    if (!g) return false;
    bool fix = false;
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch >= 0 && gguf_get_kv_type(g, arch) == GGUF_TYPE_STRING
        && std::strcmp(gguf_get_val_str(g, arch), "glm4") == 0) {
        fix = gguf_find_key(g, "glm4.rope.dimension_count") < 0;
    }
    gguf_free(g);
    return fix;
}

// Context with an 8-bit KV cache and flash attention (upstream v0.5.6 d5ccad3): half the
// per-token VRAM of f16, so >=32k tokens fit next to the PoM walk. Falls back to the default f16
// cache for an architecture the fast path cannot serve. *kv receives the cache type used.
static llama_context* keryx_make_ctx(llama_model* model, int n_ctx, const char** kv, llama_context_params* out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? n_ctx : 4096;
    // The logical batch bounds one decode call; the physical one bounds the compute buffers,
    // which grow with it and not with the context.
    cp.n_batch = std::min(2048u, cp.n_ctx);
    cp.n_ubatch = std::min(512u, cp.n_batch);
    // The cache is cleared before every request, so sliding-window layers only need their window
    // (upstream v0.5.7 30d72ed: tier 2 fits a 16 GB card).
    cp.swa_full = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
    *kv = "q8_0";
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        *kv = "f16";
        ctx = llama_init_from_model(model, cp);
    }
    if (ctx && out) *out = cp;
    return ctx;
}

// Free VRAM (MiB) on CUDA ordinal `gpu`, or -1 when it cannot be queried.
// Metal (unified memory): -1, so the context ladder keeps the largest context that loads.
static long keryx_free_mib(int gpu) {
#ifdef __APPLE__
    (void)gpu;
    return -1;
#else
    int prev = -1;
    cudaGetDevice(&prev);
    if (cudaSetDevice(gpu) != cudaSuccess) return -1;
    size_t fr = 0, tot = 0;
    const bool ok = cudaMemGetInfo(&fr, &tot) == cudaSuccess;
    if (prev >= 0) cudaSetDevice(prev);
    return ok ? (long)(fr >> 20) : -1;
#endif
}


// Shared system prompt set by the miner (keryx_llama_set_system_prompt); empty = none.
static std::mutex g_sys_mu;
static std::string g_sys_prompt;
static std::string keryx_system_prompt() { std::lock_guard<std::mutex> g(g_sys_mu); return g_sys_prompt; }

extern "C" {

// ABI version — the miner refuses to use a mismatched .so.
int keryx_llama_abi() { return 2; }

// Optional (v0.14.1): human-readable reason of the last failed load ("" if none). Additive — the
// host resolves it with dlsym and tolerates its absence in older engines.
const char* keryx_llama_last_error() {
    static thread_local std::string copy;
    std::lock_guard<std::mutex> g(g_err_mu);
    copy = g_last_error;
    return copy.c_str();
}

// Optional presentation ABI: expose llama.cpp's logger through wrapper-owned names. ELF/Mach-O
// hosts can also resolve llama_log_get/set directly from historical sidecars; the named bridge is
// required on Windows, whose module-definition file intentionally exports only this wrapper API.
// The host installs a callback once, before backend initialization, which preserves classic logs
// while preventing native stderr writes during an alternate-screen dashboard.
void keryx_llama_log_get_v1(ggml_log_callback* callback, void** user_data) {
    llama_log_get(callback, user_data);
}
void keryx_llama_log_set_v1(ggml_log_callback callback, void* user_data) {
    llama_log_set(callback, user_data);
}

// Shared loader: weights are loaded ONCE; context sizes are tried largest-first from `ladder`.
// A context is kept only if at least `reserve_mib` VRAM stays free afterwards for the PoM walk's
// own buffers (zero-dup gather tables, batch scratch); the last candidate is always kept.
static KeryxLlama* keryx_load_impl(const char* gguf_path, int gpu, const int* ladder, int n_ladder, int reserve_mib) {
#if !defined(__APPLE__) && KERYX_LLAMA_MIN_CC > 0
    // Architecture gate. ggml has no load-time check: on a GPU older than every compiled arch the
    // model loads fine and the FIRST kernel launch dies with "no kernel image is available" inside
    // ggml_abort — taking the whole miner down (field report: Tesla V100 sm_70 on the modern line,
    // whose engines are built for sm_75+). Refuse here with an actionable message instead.
    {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpu) == cudaSuccess &&
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, gpu) == cudaSuccess) {
            const int cc = major * 10 + minor;
            if (cc < KERYX_LLAMA_MIN_CC) {
                char buf[320];
                snprintf(buf, sizeof(buf),
                         "GPU %d has compute capability %d.%d but this build's inference engine carries "
                         "kernels for %d.%d and newer only. Use the build line for your GPU: LEGACY = sm_61+ "
                         "(GTX 10-series, Tesla V100/Volta, CMP 100-210, Turing+; driver 550+), MODERN = sm_75+ "
                         "(driver 575+). Not loading the model on this GPU.",
                         gpu, major, minor, KERYX_LLAMA_MIN_CC / 10, KERYX_LLAMA_MIN_CC % 10);
                keryx_set_last_error(buf);
                return nullptr;
            }
        }
    }
#endif
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE; // ONE GPU — never layer-split across mining cards
    mp.main_gpu     = gpu;
    mp.use_mmap     = true;
    llama_model_kv_override overrides[2] = {};               // zeroed tail terminates the list
    if (keryx_needs_glm4_rope_fix(gguf_path)) {
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::strncpy(overrides[0].key, "glm4.rope.dimension_count", sizeof(overrides[0].key) - 1);
        overrides[0].val_i64 = 64;
        mp.kv_overrides = overrides;
    }
    llama_model* model = llama_model_load_from_file(gguf_path, mp);
    if (!model) return nullptr;

    llama_context* ctx = nullptr;
    llama_context_params cp{};
    const char* kv = "f16";
    long free_after = -1;
    for (int i = 0; i < n_ladder && !ctx; ++i) {
        ctx = keryx_make_ctx(model, ladder[i], &kv, &cp);
        if (!ctx) continue;                                   // context did not fit: next size
        free_after = keryx_free_mib(gpu);
        const bool last = (i == n_ladder - 1);
        if (!last && reserve_mib > 0 && free_after >= 0 && free_after < reserve_mib) {
            llama_free(ctx);                                  // leave room for the walk
            ctx = nullptr;
        }
    }
    if (!ctx) { llama_model_free(model); return nullptr; }

    // Same user-facing sampling the candle path uses (repeat penalty -> temperature 0.7 /
    // top_p 0.9) — the OPoI text is not consensus-relevant, but keep the flavor consistent.
    // The repetition penalty is essential: without it the small quantized models (9B Q4
    // especially) degenerate into verbatim sentence loops. 256-token window because the
    // observed loops are whole sentences (~25 tokens each), far beyond the classic 64-token
    // window; 1.10 is the battle-tested llama.cpp default. (Upstream 76047d9.)
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // DRY (Don't Repeat Yourself) ahead of the flat penalty: the 256-window repeat penalty scores
    // single tokens and cannot see a whole clause repeating with different filler between copies —
    // the exact loop the 9B Q4 models fall into. DRY matches the longest suffix that already
    // appeared and penalizes its continuation, so a re-emerging sentence gets choked at token ~3
    // instead of running to completion. Params are the llama.cpp/text-gen-webui defaults
    // (multiplier 0.8, base 1.75, allowed-length 2, whole-context window -1); the seq breakers stop
    // matches from spanning sentence/quote boundaries. (Ported from upstream Keryx-Labs f8d6ba4b.)
    static const char* dry_breakers[] = { "\n", ":", "\"", "*" };
    llama_sampler_chain_add(smpl, llama_sampler_init_dry(
        llama_model_get_vocab(model), llama_model_n_ctx_train(model),
        0.8f, 1.75f, 2, -1, dry_breakers,
        sizeof(dry_breakers) / sizeof(dry_breakers[0])));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.10f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    auto* h = new KeryxLlama();
    h->model = model; h->ctx = ctx; h->smpl = smpl;
    h->ctx_info = "n_ctx=" + std::to_string(llama_n_ctx(ctx)) + " kv=" + kv +
                  " flash_attn=" + llama_flash_attn_type_name(cp.flash_attn_type) +
                  " n_batch=" + std::to_string(llama_n_batch(ctx)) + " n_ubatch=" + std::to_string(llama_n_ubatch(ctx)) +
                  " vram_free_after=" + std::to_string(free_after) + "MiB";
    for (auto& p : model->tensors_by_name) h->names.push_back(p.first);
    std::sort(h->names.begin(), h->names.end());
    return h;
}

KeryxLlama* keryx_llama_load(const char* gguf_path, int gpu, int n_ctx) {
    const int one[1] = { n_ctx > 0 ? n_ctx : 4096 };
    return keryx_load_impl(gguf_path, gpu, one, 1, 0);
}

// Context ladder + walk reserve (see keryx_load_impl). Optional export: the miner falls back to
// keryx_llama_load when an older library lacks it.
KeryxLlama* keryx_llama_load2(const char* gguf_path, int gpu, const int* ladder, int n_ladder, int reserve_mib) {
    if (!ladder || n_ladder <= 0) return keryx_llama_load(gguf_path, gpu, 4096);
    return keryx_load_impl(gguf_path, gpu, ladder, n_ladder, reserve_mib);
}

// Optional export: the system prompt applied to every request (UTF-8, copied).
void keryx_llama_set_system_prompt(const char* text) {
    std::lock_guard<std::mutex> g(g_sys_mu);
    g_sys_prompt = text ? text : "";
}

// Context parameters the engine ended up with, for the miner log.
const char* keryx_llama_context_info(KeryxLlama* h) { return h ? h->ctx_info.c_str() : ""; }

// Allocated context window in tokens (0 when unavailable).
int keryx_llama_n_ctx(KeryxLlama* h) { return (h && h->ctx) ? (int)llama_n_ctx(h->ctx) : 0; }

size_t keryx_llama_tensor_count(KeryxLlama* h) { return h ? h->names.size() : 0; }

// Tensor i in CANONICAL order. *is_device = the data pointer is CUDA device memory (walkable
// in-place); 0 = host memory (the caller uploads its own device copy for the walk).
bool keryx_llama_tensor_info(KeryxLlama* h, size_t i, const char** name, void** data,
                             size_t* nbytes, int* is_device) {
    if (!h || i >= h->names.size()) return false;
    const ggml_tensor* t = h->model->get_tensor(h->names[i].c_str());
    if (!t || !t->data) return false;
    *name = h->names[i].c_str();
    *data = t->data;
    *nbytes = ggml_nbytes(t);
#ifdef __APPLE__
    // Metal / Apple Silicon unified memory: tensor bytes are in an MTLBuffer that's both CPU- and
    // GPU-visible via the same address. The Metal PoM walk (Phase 3a) doesn't consume `data` for
    // its own gather (it pre-packs from GGUF), so the semantic here is "there's a live pointer
    // to the tensor bytes for anyone who wants to check byte-exactness against GGUF".
    *is_device = 1;
#else
    cudaPointerAttributes attr{};
    cudaPointerGetAttributes(&attr, t->data);
    *is_device = attr.type == cudaMemoryTypeDevice ? 1 : 0;
#endif
    return true;
}

// CUDA ordinal owning tensor i's bytes, or -1 (host memory, unified memory, unknown, or a context
// in error). The possession walk gathers over these pointers, so it must launch on this device —
// the Rust side (foreign_device_tensor) uses this to detect a wrong-device placement before the
// walk dereferences unmapped memory. Optional symbol (miner looks it up soft). Upstream aa29fd2.
int keryx_llama_tensor_device(KeryxLlama* h, size_t i) {
#ifdef __APPLE__
    (void)h; (void)i;
    return -1;
#else
    if (!h || i >= h->names.size()) return -1;
    const ggml_tensor* t = h->model->get_tensor(h->names[i].c_str());
    if (!t || !t->data) return -1;
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, t->data) != cudaSuccess) return -1;
    return attr.type == cudaMemoryTypeDevice ? attr.device : -1;
#endif
}

// Generate up to max_tokens; writes UTF-8 into out (cap bytes, NUL-terminated). Returns written
// length, or -1 on error. Serialized — one generation at a time (OPoI challenges are rare).
int keryx_llama_generate(KeryxLlama* h, const char* prompt, int max_tokens, char* out, int cap) {
    if (!h || !prompt || !out || cap < 2 || max_tokens <= 0 || max_tokens > 2048) return -1;
    // Byte bound = the miner's MAX_INFERENCE_PROMPT_BYTES (1 MiB; was 4096 before H14 private
    // requests, which refused every valid larger prompt). Whether a prompt fits is decided below
    // against this card's context (-2), not by a fixed byte count.
    constexpr size_t kMaxPromptBytes = size_t(1) << 20;
    size_t prompt_len = 0;
    while (prompt_len <= kMaxPromptBytes && prompt[prompt_len] != '\0') ++prompt_len;
    if (prompt_len == 0) return -1;
    if (prompt_len > kMaxPromptBytes) return -2;
    std::lock_guard<std::mutex> g(h->gen_lock);
    const llama_vocab* vocab = llama_model_get_vocab(h->model);

    // Apply the model's chat template so the model sees a proper USER turn and stops at the end of
    // ITS reply. Without this the raw prompt is treated as free text: the small tier-0 models never
    // reach a turn boundary, so they hallucinate a whole fake "user:/assistant:" conversation and
    // ramble until max_tokens (the EOG check below never fires because the model doesn't think it
    // finished a turn). With the template the model emits its end-of-turn token (e.g. <|im_end|>),
    // which IS an EOG token → clean stop. Falls back to the raw prompt if the GGUF has no template.
    // System prompt (upstream v0.5.6 32d641a/dbfb745): the miner sets one shared, vendor-agnostic
    // system prompt via keryx_llama_set_system_prompt. It goes in as a real system turn through the
    // GGUF's own template; GLM-4 ignores the system role, so (as upstream) it is folded into the
    // user turn there — and for any template that cannot take a system role.
    std::string formatted;
    if (const char* tmpl = llama_model_chat_template(h->model, nullptr)) {
        const std::string sys = keryx_system_prompt();
        const bool glm = std::strstr(tmpl, "[gMASK]") != nullptr;
        std::string folded;
        auto apply = [&](const llama_chat_message* msgs, size_t n_msgs) -> bool {
            int need = llama_chat_apply_template(tmpl, msgs, n_msgs, /*add_ass=*/true, nullptr, 0);
            if (need <= 0) return false;
            formatted.resize((size_t)need);
            int wrote = llama_chat_apply_template(tmpl, msgs, n_msgs, true, &formatted[0], need);
            if (wrote <= 0) { formatted.clear(); return false; }
            formatted.resize((size_t)wrote);
            return true;
        };
        // Gemma 4 is NOT the classic <start_of_turn> Gemma: its Jinja template renders turns as
        // `<|turn>role\n…<turn|>\n`, which llama_chat_apply_template cannot detect (b10015 only knows
        // <start_of_turn>), so `apply` fails and — until v0.14.1 — the RAW prompt reached the model:
        // no system turn, no turn boundary. Short prompts (the pool's 1-word "ping" probe) then
        // produced junk; on Volta the junk ended in `<channel|>`, the miner's think-tag stripper
        // left an empty answer, three empty answers withdrew the model and suspended mining (field
        // report: Tesla V100 "can't load inference, candle fails"). Render it by hand, verbatim as
        // upstream keryx-miner's format_prompt_by_name does, with the empty thought channel of the
        // `enable_thinking=false` branch so the visible answer starts immediately. No literal <bos>:
        // this GGUF sets add_bos_token, so llama_tokenize(add_special=true) prepends it.
        const bool gemma4 = std::strstr(tmpl, "<|turn>") != nullptr;
        bool ok = false;
        if (gemma4) {
            formatted.clear();
            if (!sys.empty()) formatted += "<|turn>system\n" + sys + "<turn|>\n";
            formatted += std::string("<|turn>user\n") + prompt + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
            ok = true;
        }
        if (!ok && !sys.empty() && !glm) {
            llama_chat_message two[2] = { { "system", sys.c_str() }, { "user", prompt } };
            ok = apply(two, 2);
        }
        if (!ok && !sys.empty()) {
            folded = sys + "\n\n" + prompt;
            llama_chat_message one{ "user", folded.c_str() };
            ok = apply(&one, 1);
        }
        if (!ok) {
            llama_chat_message one{ "user", prompt };
            apply(&one, 1);
        }
    }
    const char* infer = formatted.empty() ? prompt : formatted.c_str();
    const int infer_len = (int)strlen(infer);

    std::vector<llama_token> toks(infer_len + 16);
    int n = llama_tokenize(vocab, infer, infer_len, toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) return -1;
    toks.resize(n);

    llama_memory_clear(llama_get_memory(h->ctx), true);
    // Penalty/DRY samplers retain accepted-token history. Reset it with the KV cache so one remote
    // request cannot bias or truncate the next independent request.
    llama_sampler_reset(h->smpl);
    // Prompt guard (upstream v0.5.6 1670060): llama_decode GGML_ASSERTs n_tokens <= n_batch and
    // a prompt longer than the context can never be answered — either one used to ABORT the whole
    // miner (every card) on a single oversized request. Refuse with -2 instead, and feed the
    // prompt in n_batch-sized chunks so any prompt that fits the context is served.
    const int n_ctx_tok = (int)llama_n_ctx(h->ctx);
    const int n_batch_tok = std::max(1, (int)llama_n_batch(h->ctx));
    if (n <= 0 || n > n_ctx_tok - 16) return -2;
    for (int i = 0; i + n_batch_tok < n; i += n_batch_tok) {
        if (llama_decode(h->ctx, llama_batch_get_one(toks.data() + i, n_batch_tok)) != 0) return -3;
    }
    const int tail = ((n - 1) % n_batch_tok) + 1;
    llama_batch batch = llama_batch_get_one(toks.data() + (n - tail), (int32_t)tail);
    int written = 0;
    std::string acc; // mirrors `out` for cross-piece stop-string scanning
    for (int i = 0; i < max_tokens; i++) {
        if (llama_decode(h->ctx, batch) != 0) break;
        llama_token tok = llama_sampler_sample(h->smpl, h->ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[256];
        int pn = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (pn < 0) break;
        if (written + pn >= cap - 1) break;
        memcpy(out + written, piece, pn);
        written += pn;
        // Fallback for models that write the turn-end marker as PLAIN TEXT (multi-token, not an
        // atomic EOG token) and roll into a hallucinated next turn — cut at the first such marker so
        // the answer ends cleanly even when EOG never fires. Only unambiguous chat template markers
        // (never valid answer content), so this can't truncate a legitimate reply.
        acc.append(piece, (size_t)pn);
        static const char* const STOPS[] = {
            "<|im_end|>", "<|im_start|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
        };
        size_t cut = std::string::npos;
        for (const char* s : STOPS) { size_t p = acc.find(s); if (p < cut) cut = p; }
        if (cut != std::string::npos) { written = (int)cut; break; }
        batch = llama_batch_get_one(&tok, 1);
    }
    out[written] = 0;
    return written;
}

// Present (=1) when keryx_llama_generate chunks the prompt by n_batch and refuses (-2) a prompt
// that does not fit the context instead of aborting. The miner looks it up softly: an older
// library without it gets a conservative byte cap on the Rust side.
int keryx_llama_prompt_guard() { return 1; }

void keryx_llama_free(KeryxLlama* h) {
    if (!h) return;
    if (h->smpl) llama_sampler_free(h->smpl);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

} // extern "C"
