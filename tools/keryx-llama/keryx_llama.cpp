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
// The miner dlopens this next to its own binary; absent = the candle fallback stays active.
// Built by hiveos/build-keryx-llama.sh (CUDA) or hiveos/build-keryx-llama-macos.sh (Metal).
#include "llama.h"
#include "llama-model.h"
#include "ggml.h"
#ifdef __APPLE__
// Metal: llama.cpp's ggml-metal backend stores quantized tensors in unified-memory MTLBuffers.
// `t->data` is a CPU-readable pointer into that unified memory (also GPU-visible on Apple Silicon
// via the shared address space), so we don't need cudaPointerGetAttributes — `is_device` is
// always 1 for tensors llama.cpp reports.
#else
#include <cuda_runtime.h>
#endif
#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

struct KeryxLlama {
    llama_model*   model = nullptr;
    llama_context* ctx   = nullptr;
    llama_sampler* smpl  = nullptr;
    std::vector<std::string> names; // canonical (byte-lexicographic) order — matches pom.rs
    std::mutex gen_lock;
};

extern "C" {

// ABI version — the miner refuses to use a mismatched .so.
int keryx_llama_abi() { return 2; }

KeryxLlama* keryx_llama_load(const char* gguf_path, int gpu, int n_ctx) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode   = LLAMA_SPLIT_MODE_NONE; // ONE GPU — never layer-split across mining cards
    mp.main_gpu     = gpu;
    mp.use_mmap     = true;
    llama_model* model = llama_model_load_from_file(gguf_path, mp);
    if (!model) return nullptr;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx > 0 ? n_ctx : 4096;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return nullptr; }

    // Same user-facing sampling the candle path uses (repeat penalty -> temperature 0.7 /
    // top_p 0.9) — the OPoI text is not consensus-relevant, but keep the flavor consistent.
    // The repetition penalty is essential: without it the small quantized models (9B Q4
    // especially) degenerate into verbatim sentence loops. 256-token window because the
    // observed loops are whole sentences (~25 tokens each), far beyond the classic 64-token
    // window; 1.10 is the battle-tested llama.cpp default. (Upstream 76047d9.)
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(256, 1.10f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    auto* h = new KeryxLlama();
    h->model = model; h->ctx = ctx; h->smpl = smpl;
    for (auto& p : model->tensors_by_name) h->names.push_back(p.first);
    std::sort(h->names.begin(), h->names.end());
    return h;
}

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
    if (!h || !prompt || !out || cap < 2) return -1;
    std::lock_guard<std::mutex> g(h->gen_lock);
    const llama_vocab* vocab = llama_model_get_vocab(h->model);

    // Apply the model's chat template so the model sees a proper USER turn and stops at the end of
    // ITS reply. Without this the raw prompt is treated as free text: the small tier-0 models never
    // reach a turn boundary, so they hallucinate a whole fake "user:/assistant:" conversation and
    // ramble until max_tokens (the EOG check below never fires because the model doesn't think it
    // finished a turn). With the template the model emits its end-of-turn token (e.g. <|im_end|>),
    // which IS an EOG token → clean stop. Falls back to the raw prompt if the GGUF has no template.
    std::string formatted;
    if (const char* tmpl = llama_model_chat_template(h->model, nullptr)) {
        llama_chat_message msg{ "user", prompt };
        int need = llama_chat_apply_template(tmpl, &msg, 1, /*add_ass=*/true, nullptr, 0);
        if (need > 0) {
            formatted.resize((size_t)need);
            int wrote = llama_chat_apply_template(tmpl, &msg, 1, true, &formatted[0], need);
            if (wrote > 0) formatted.resize((size_t)wrote); else formatted.clear();
        }
    }
    const char* infer = formatted.empty() ? prompt : formatted.c_str();
    const int infer_len = (int)strlen(infer);

    std::vector<llama_token> toks(infer_len + 16);
    int n = llama_tokenize(vocab, infer, infer_len, toks.data(), (int32_t)toks.size(), true, true);
    if (n < 0) return -1;
    toks.resize(n);

    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t)toks.size());
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

void keryx_llama_free(KeryxLlama* h) {
    if (!h) return;
    if (h->smpl) llama_sampler_free(h->smpl);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

} // extern "C"
