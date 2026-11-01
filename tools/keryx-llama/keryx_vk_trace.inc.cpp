// H15 trace bridge for the existing Vulkan model owner. The native Vulkan kernels
// are attributed in vulkan-patches/0001-h15-vulkan-trace.patch. Batch declarations
// follow the node's expected_ubatch rule, independently of Vulkan's physical batch.
// Included after KeryxLlama and the ordinary engine implementation.

struct KeryxVkTrace {
    uint32_t max_ctx = 0;
    uint32_t physical_batch = 8;
    bool configured = false;
    bool sequence = false;
    int (*after_batch)(void*, uint32_t) = nullptr;
    void* callback_data = nullptr;
};

static thread_local std::string keryx_vk_trace_error;

static int keryx_vk_fail(KeryxLlama* h, const char* detail, int rc = -5) {
    keryx_vk_trace_error = detail;
    if (h) {
        const char* native = ggml_backend_vk_keryx_trace_error(h->gpu);
        if (native && *native) keryx_vk_trace_error += std::string(": ") + native;
    }
    return rc;
}

static void keryx_vk_trace_free(KeryxLlama* h) {
    if (!h || !h->trace) return;
    ggml_backend_vk_keryx_trace_release(h->gpu);
    delete h->trace;
    h->trace = nullptr;
}

static int keryx_vk_trace_context(KeryxLlama* h) {
    if (!h || !h->trace || !h->trace->configured) return -5;
    keryx_llama_pom_release_pad(h);
    if (h->ctx) { llama_free(h->ctx); h->ctx = nullptr; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = h->trace->max_ctx;
    cp.n_batch = cp.n_ubatch = 8;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // The proof may open an earlier position of a sliding-window layer after
    // generation. Keep its cache rows instead of recycling their storage.
    cp.swa_full = true;
    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) return keryx_vk_fail(h, "Vulkan traced context allocation failed", -2);
    llama_sampler_reset(h->smpl);
    return 0;
}

static int keryx_vk_trace_feed(KeryxLlama* h, const int32_t* tokens, uint32_t start,
                              uint32_t count, uint32_t n_prompt) {
    uint32_t consumed = 0;
    while (consumed < count) {
        const uint32_t pos = start + consumed;
        const uint32_t declared = pos < n_prompt ? std::min(512u, n_prompt - pos / 512u * 512u) : 1u;
        const uint32_t boundary = pos < n_prompt ? std::min(n_prompt, (pos / 512u + 1) * 512u) : pos + 1;
        const uint32_t n = std::min({h->trace->physical_batch, count - consumed, boundary - pos});
        if (!n || ggml_backend_vk_keryx_trace_ubatch_as(h->gpu, pos, n, declared) != 0)
            return keryx_vk_fail(h, "Vulkan trace batch refused");
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (uint32_t j = 0; j < n; ++j) {
            batch.token[j] = tokens[consumed + j];
            batch.pos[j] = pos + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = j + 1 == n;
        }
        const int rc = llama_decode(h->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) return keryx_vk_fail(h, "Vulkan traced decode failed", -3);
        if (h->trace->after_batch && h->trace->after_batch(h->trace->callback_data, n) != 0)
            return keryx_vk_fail(h, "Vulkan sequence capture failed");
        consumed += n;
    }
    return 0;
}

static int keryx_vk_replay_to(KeryxLlama* h, const int32_t* tokens, uint32_t n_tokens,
                             uint32_t n_prompt, uint32_t last) {
    if (!n_prompt || n_prompt > n_tokens || last >= n_tokens) return -1;
    const uint32_t prompt_end = last < n_prompt ? std::min(n_prompt, (last / 8u + 1u) * 8u) : n_prompt;
    int rc = keryx_vk_trace_feed(h, tokens, 0, prompt_end, n_prompt);
    if (rc != 0) return rc;
    for (uint32_t p = n_prompt; p <= last; ++p) {
        rc = keryx_vk_trace_feed(h, tokens + p, p, 1, n_prompt);
        if (rc != 0) return rc;
    }
    return 0;
}

static llama_token keryx_vk_trace_pad(const llama_vocab* vocab) {
    llama_token pad = llama_vocab_eot(vocab);
    if (pad == LLAMA_TOKEN_NULL) pad = llama_vocab_eos(vocab);
    return pad == LLAMA_TOKEN_NULL ? 0 : pad;
}

extern "C" {

int keryx_llama_vk_trace_abi() { return 1; }
const char* keryx_llama_last_error() { return keryx_vk_trace_error.c_str(); }

// Identity comes from the physical device owning the model, not an OpenCL or
// ggml ordinal in a different enumeration. Driver identity also binds replay.
int keryx_llama_vk_trace_identity(KeryxLlama* h, char* name, size_t name_cap,
                                 char* driver, size_t driver_cap) {
    if (!h || !h->phys || !name || name_cap < 2 || !driver || driver_cap < 2) return -1;
    VkPhysicalDeviceDriverProperties dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &dp;
    vkGetPhysicalDeviceProperties2(h->phys, &props);
    const auto& p = props.properties;
    const std::string gpu = std::string(p.deviceName) + ":" + std::to_string(p.vendorID) + ":" + std::to_string(p.deviceID);
    std::string version = std::to_string(dp.driverID) + ":" + std::to_string(p.driverVersion) + ":" + dp.driverName + ":" + dp.driverInfo + ":";
    const char* hex = "0123456789abcdef";
    for (uint8_t b : p.pipelineCacheUUID) { version += hex[b >> 4]; version += hex[b & 15]; }
    if (gpu.size() >= name_cap || version.size() >= driver_cap) return -1;
    memcpy(name, gpu.c_str(), gpu.size() + 1);
    memcpy(driver, version.c_str(), version.size() + 1);
    return 0;
}

int keryx_llama_trace_configure(KeryxLlama* h, const void* manifest, size_t manifest_len, int max_pos) {
    if (!h || !h->model || !h->loaded_context || !manifest || !manifest_len || max_pos < 0) return -1;
    std::lock_guard<std::mutex> guard(h->gen_lock);
    if (!h->trace) {
        h->trace = new KeryxVkTrace();
        h->trace->max_ctx = h->loaded_context;
        // Head-slice MLA products run one token at a time. Their declared
        // prompt batch/layout remains the consensus 512-token rule.
        char arch[64] = {};
        if (llama_model_meta_val_str(h->model, "general.architecture", arch, sizeof(arch)) > 0 && strcmp(arch, "kimi-linear") == 0)
            h->trace->physical_batch = 1;
    }
    const uint32_t capacity = max_pos ? (uint32_t)max_pos : h->trace->max_ctx;
    const int rc = ggml_backend_vk_keryx_trace_configure(h->gpu, manifest, manifest_len, capacity);
    h->trace->configured = rc == 0;
    h->trace->sequence = false;
    return rc ? keryx_vk_fail(h, "Vulkan trace configuration failed") : 0;
}

int keryx_llama_trace_leaf_count(KeryxLlama* h) {
    return h && h->trace && h->trace->configured ? ggml_backend_vk_keryx_trace_leaf_count(h->gpu) : -1;
}

int keryx_llama_trace_read(KeryxLlama* h, uint8_t* leaves, size_t leaves_cap,
                          uint32_t* ubatch, size_t ubatch_cap, uint32_t* n_pos) {
    if (!h || !h->trace || !h->trace->configured) return -1;
    return ggml_backend_vk_keryx_trace_read(h->gpu, leaves, leaves_cap, ubatch, ubatch_cap, n_pos);
}

int keryx_llama_vk_sequence_configure(KeryxLlama* h, uint32_t layers, uint32_t heads,
                                     const uint32_t* recurrent, uint32_t n_recur, uint32_t interval) {
    if (!h || !h->trace || !h->trace->configured || (n_recur && !recurrent)) return -1;
    int rc = ggml_backend_vk_keryx_lse_configure(h->gpu, layers, heads);
    if (!rc) rc = ggml_backend_vk_keryx_recur_configure(h->gpu, recurrent, n_recur, interval);
    h->trace->sequence = !rc && (layers || n_recur);
    return rc ? keryx_vk_fail(h, "Vulkan sequence configuration failed") : 0;
}

int keryx_llama_vk_batch_callback(KeryxLlama* h, int (*callback)(void*, uint32_t), void* data) {
    if (!h || !h->trace) return -1;
    h->trace->after_batch = callback;
    h->trace->callback_data = data;
    return 0;
}

int keryx_llama_vk_recurrent_read(KeryxLlama* h, uint8_t* records, size_t records_cap,
                                 uint8_t* checkpoints, size_t checkpoints_cap,
                                 uint32_t* n_tokens, uint32_t* has_checkpoint) {
    if (!h || !h->trace) return -1;
    return ggml_backend_vk_keryx_recur_read(h->gpu, records, records_cap, checkpoints,
                                         checkpoints_cap, n_tokens, has_checkpoint);
}

size_t keryx_llama_vk_sequence_state(KeryxLlama* h, uint8_t* out, size_t cap) {
    if (!h || !h->ctx || !h->trace) return 0;
    const size_t need = llama_state_seq_get_size_ext(h->ctx, 0, 1u << 31);
    if (!out) return need;
    if (cap < need) return 0;
    return llama_state_seq_get_data_ext(h->ctx, out, cap, 0, 1u << 31);
}

int keryx_llama_vk_lse_read(KeryxLlama* h, float* out, size_t cap, uint32_t n_pos) {
    return h && h->trace ? ggml_backend_vk_keryx_lse_read(h->gpu, out, cap, n_pos) : -1;
}

int keryx_llama_generate_traced(KeryxLlama* h, const char* prompt, const int32_t* prefix, int n_prefix,
                               int head_tokens, const char* rest, int max_tokens, char* out, int cap,
                               int32_t* tokens, int tokens_cap, int* n_prompt_out, int* n_fed_out,
                               int min_prompt, int answer_bytes_per_position) {
    if (!h || !h->trace || !h->trace->configured || !prompt || !out || cap < 2 || !tokens ||
        tokens_cap < 1 || !n_prompt_out || !n_fed_out || max_tokens < 0 || max_tokens > 2048 ||
        n_prefix < 0 || min_prompt < 0 || answer_bytes_per_position < 0 || (n_prefix && (!prefix || !rest))) return -1;
    std::lock_guard<std::mutex> guard(h->gen_lock);
    const llama_vocab* vocab = llama_model_get_vocab(h->model);
    if ((uint32_t)min_prompt > h->trace->max_ctx || (uint32_t)n_prefix > h->trace->max_ctx)
        return keryx_vk_fail(nullptr, "Prefix or padding exceeds the traced context", -2);
    std::vector<llama_token> prompt_tokens(strlen(prompt) + 16);
    int n = llama_tokenize(vocab, prompt, (int32_t)strlen(prompt), prompt_tokens.data(), (int32_t)prompt_tokens.size(), true, true);
    if (n <= 0) return -1;
    prompt_tokens.resize(n);
    if (n_prefix) {
        if (n != head_tokens) return keryx_vk_fail(nullptr, "Request prefix head token count differs", -8);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_prefix; ++i) {
            if (prefix[i] < 0 || prefix[i] >= n_vocab) return -1;
            prompt_tokens.push_back(prefix[i]);
        }
        std::vector<llama_token> tail(strlen(rest) + 16);
        const int nr = llama_tokenize(vocab, rest, (int32_t)strlen(rest), tail.data(), (int32_t)tail.size(), false, true);
        if (nr < 0) return -1;
        prompt_tokens.insert(prompt_tokens.end(), tail.begin(), tail.begin() + nr);
    }
    const llama_token pad = keryx_vk_trace_pad(vocab);
    if ((size_t)min_prompt > prompt_tokens.size()) prompt_tokens.resize(min_prompt, pad);
    n = (int)prompt_tokens.size();
    if ((uint64_t)n + 16 > h->trace->max_ctx) return keryx_vk_fail(nullptr, "Prompt exceeds the Vulkan traced context", -2);
    if (n > tokens_cap) return -1;
    int rc = keryx_vk_trace_context(h);
    if (rc) return rc;
    if (ggml_backend_vk_keryx_trace_begin(h->gpu)) return keryx_vk_fail(h, "Cannot begin Vulkan trace");
    memcpy(tokens, prompt_tokens.data(), n * sizeof(int32_t));
    int fed = n, written = 0;
    rc = keryx_vk_trace_feed(h, tokens, 0, n, n);
    const int budget = std::min({max_tokens, tokens_cap - n, (int)h->trace->max_ctx - n - 1});
    for (int i = 0; !rc && i < budget; ++i) {
        const llama_token token = llama_sampler_sample(h->smpl, h->ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;
        std::vector<char> piece(256);
        int bytes = llama_token_to_piece(vocab, token, piece.data(), (int)piece.size(), 0, true);
        if (bytes < 0) {
            piece.resize((size_t)-bytes);
            bytes = llama_token_to_piece(vocab, token, piece.data(), (int)piece.size(), 0, true);
        }
        if (bytes < 0) { rc = -3; break; }
        if (bytes >= cap - written) break;
        memcpy(out + written, piece.data(), bytes);
        written += bytes;
        tokens[fed] = token;
        rc = keryx_vk_trace_feed(h, tokens + fed, fed, 1, n);
        ++fed;
    }
    if (!rc && answer_bytes_per_position) {
        const int need = std::min(written / answer_bytes_per_position, budget);
        while (!rc && fed - n < need) {
            tokens[fed] = pad;
            rc = keryx_vk_trace_feed(h, tokens + fed, fed, 1, n);
            ++fed;
        }
    }
    const int ended = ggml_backend_vk_keryx_trace_end(h->gpu);
    if (rc || ended) return keryx_vk_fail(h, "Vulkan traced generation failed", rc ? rc : -5);
    out[written] = 0;
    *n_prompt_out = n;
    *n_fed_out = fed;
    return written;
}

int keryx_llama_replay_multi(KeryxLlama* h, const int32_t* tokens, int n_tokens, int n_prompt,
                            const uint32_t* pos, const uint32_t* leaf, uint32_t n_capture,
                            uint32_t slot_bytes, uint8_t* out, size_t out_cap, uint32_t* lens) {
    if (!h || !h->trace || !tokens || n_tokens <= 0 || n_prompt <= 0 || n_prompt > n_tokens ||
        !pos || !leaf || !n_capture || n_capture > 32 || !slot_bytes || slot_bytes % 4 ||
        !out || !lens || out_cap < (uint64_t)n_capture * slot_bytes || (uint32_t)n_tokens > h->trace->max_ctx) return -1;
    std::lock_guard<std::mutex> guard(h->gen_lock);
    const uint32_t last = *std::max_element(pos, pos + n_capture);
    if (last >= (uint32_t)n_tokens) return -1;
    int rc = keryx_vk_trace_context(h);
    if (rc) return rc;
    rc = ggml_backend_vk_keryx_trace_begin(h->gpu);
    if (!rc) rc = ggml_backend_vk_keryx_trace_arm_captures(h->gpu, pos, leaf, n_capture, slot_bytes);
    if (!rc) rc = keryx_vk_replay_to(h, tokens, n_tokens, n_prompt, last);
    const int ended = ggml_backend_vk_keryx_trace_end(h->gpu);
    if (!rc && !ended) rc = ggml_backend_vk_keryx_trace_read_captures(h->gpu, out, out_cap, lens, n_capture);
    return rc || ended ? keryx_vk_fail(h, "Vulkan product replay failed") : 0;
}

int keryx_llama_vk_replay_sequence(KeryxLlama* h, const int32_t* tokens, int n_tokens, int n_prompt) {
    if (!h || !h->trace || !tokens || n_tokens <= 0 || n_prompt <= 0 || n_prompt > n_tokens ||
        (uint32_t)n_tokens > h->trace->max_ctx) return -1;
    std::lock_guard<std::mutex> guard(h->gen_lock);
    int rc = keryx_vk_trace_context(h);
    if (rc) return rc;
    rc = ggml_backend_vk_keryx_trace_begin(h->gpu);
    if (!rc) rc = keryx_vk_replay_to(h, tokens, n_tokens, n_prompt, n_tokens - 1);
    const int ended = ggml_backend_vk_keryx_trace_end(h->gpu);
    return rc || ended ? keryx_vk_fail(h, "Vulkan sequence replay failed") : 0;
}

} // extern "C"
