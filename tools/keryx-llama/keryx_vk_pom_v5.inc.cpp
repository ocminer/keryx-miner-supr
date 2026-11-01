// PoM v5 context pad on the existing Vulkan model/walk owner. The weights stay
// resident; inference releases only this regenerable pad. V4 pipelines are kept
// separate: a v5 offset depends on the current state and cannot be pre-chased.
struct KeryxVkPad {
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> shards;
    // This small buffer contains the BDA of each 1-GiB shard.
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t address = 0, tiles = 0, seed[4]{};
    VkPipeline fill_pipe = VK_NULL_HANDLE, walk_pipe = VK_NULL_HANDLE;
    VkPipelineLayout fill_pl = VK_NULL_HANDLE, walk_pl = VK_NULL_HANDLE;
    VkShaderModule fill_sm = VK_NULL_HANDLE, walk_sm = VK_NULL_HANDLE;
};

struct PushPadFill {
    uint64_t address, first, words[4];
    uint32_t count, rounds;
};
struct PushWalkV5 {
    PushWalk walk;
    uint64_t pad_address, pad_tiles, pad_inverse;
    uint32_t pad_steps, reserved;
};
static_assert(sizeof(PushPadFill) == 56, "Vulkan pad fill ABI");
static_assert(sizeof(PushWalk) == 168 && sizeof(PushWalkV5) == 200, "Vulkan v5 walk ABI");

static void keryx_vk_pad_release(KeryxLlama* h) {
    if (!h || !h->pad) return;
    auto& p = *h->pad;
    if (p.buffer) vkDestroyBuffer(h->dev, p.buffer, nullptr);
    if (p.memory) vkFreeMemory(h->dev, p.memory, nullptr);
    for (const auto& b : p.shards) {
        if (b.first) vkDestroyBuffer(h->dev, b.first, nullptr);
        if (b.second) vkFreeMemory(h->dev, b.second, nullptr);
    }
    p.shards.clear();
    p.buffer = VK_NULL_HANDLE; p.memory = VK_NULL_HANDLE;
    p.address = p.tiles = 0;
}

static void keryx_vk_pad_free(KeryxLlama* h) {
    if (!h || !h->pad) return;
    keryx_vk_pad_release(h);
    auto& p = *h->pad;
    if (p.fill_pipe) vkDestroyPipeline(h->dev, p.fill_pipe, nullptr);
    if (p.walk_pipe) vkDestroyPipeline(h->dev, p.walk_pipe, nullptr);
    if (p.fill_pl) vkDestroyPipelineLayout(h->dev, p.fill_pl, nullptr);
    if (p.walk_pl) vkDestroyPipelineLayout(h->dev, p.walk_pl, nullptr);
    if (p.fill_sm) vkDestroyShaderModule(h->dev, p.fill_sm, nullptr);
    if (p.walk_sm) vkDestroyShaderModule(h->dev, p.walk_sm, nullptr);
    delete h->pad;
    h->pad = nullptr;
}

// Caller holds walk_lock, so every old dispatch has completed before replacement.
static bool keryx_vk_pad_ensure(KeryxLlama* h, const uint64_t seed[4], uint64_t tiles) {
    if (!tiles || tiles > UINT64_MAX / 1024) return false;
    if (h->pad && h->pad->tiles == tiles && memcmp(h->pad->seed, seed, 32) == 0) return true;
    if (!h->pad) h->pad = new KeryxVkPad();
    auto& p = *h->pad;
    if (!p.fill_pipe) p.fill_pipe = make_pipeline(h, PAD_V5_FILL_SPV, sizeof(PAD_V5_FILL_SPV), sizeof(PushPadFill), &p.fill_sm, &p.fill_pl);
    if (!p.walk_pipe) {
        const char* selected = getenv("KERYX_VK_V5_DOT");
        const bool dot = keryx_vk_integer_dot_product(h->gpu) && !(selected && strcmp(selected, "0") == 0);
        p.walk_pipe = make_pipeline(h, dot ? WALK_V5_DOT_SPV : WALK_V5_SCALAR_SPV,
            dot ? sizeof(WALK_V5_DOT_SPV) : sizeof(WALK_V5_SCALAR_SPV), sizeof(PushWalkV5), &p.walk_sm, &p.walk_pl);
    }
    if (!p.fill_pipe || !p.walk_pipe) return false;
    keryx_vk_pad_release(h);
    // Inference has completed and Rust drained every walk before it ran. Mining
    // keeps the model weights but trades the idle KV context for the context pad;
    // the next generation releases the pad and recreates its full context.
    if (h->ctx) { llama_free(h->ctx); h->ctx = nullptr; }
    std::vector<uint64_t> addresses;
    for (uint64_t first = 0; first < tiles * 1024; first += KERYX_VK_ALLOC_BYTES) {
        VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE;
        if (!make_buffer(h, std::min(KERYX_VK_ALLOC_BYTES, tiles * 1024 - first),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, &buffer, &memory, nullptr)) {
            keryx_vk_pad_release(h); return false;
        }
        p.shards.emplace_back(buffer, memory);
        VkBufferDeviceAddressInfo shard{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO}; shard.buffer = buffer;
        const uint64_t address = vkGetBufferDeviceAddress(h->dev, &shard);
        if (!address) { keryx_vk_pad_release(h); return false; }
        addresses.push_back(address);
    }
    if (!make_buffer(h, addresses.size() * sizeof(uint64_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, &p.buffer, &p.memory, nullptr) ||
        !staged_upload(h, p.buffer, reinterpret_cast<const uint8_t*>(addresses.data()), addresses.size() * sizeof(uint64_t))) {
        keryx_vk_pad_release(h); return false;
    }
    VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    dai.buffer = p.buffer;
    p.address = vkGetBufferDeviceAddress(h->dev, &dai);
    if (!p.address) { keryx_vk_pad_release(h); return false; }
    const uint64_t entries = tiles * 32;
    // Small bounded dispatches keep Windows TDR and older AMD devices responsive.
    constexpr uint32_t slice = 32768;
    for (uint64_t first = 0; first < entries; first += slice) {
        const uint32_t n = (uint32_t)std::min<uint64_t>(slice, entries - first);
        const uint64_t byte = first * 32;
        PushPadFill pf{addresses[byte / KERYX_VK_ALLOC_BYTES] + byte % KERYX_VK_ALLOC_BYTES,
                       first, {seed[0], seed[1], seed[2], seed[3]}, n, 1920};
        if (!run_dispatch(h, p.fill_pipe, p.fill_pl, h->walk_ds, &pf, sizeof(pf), (n + 63) / 64, VK_NULL_HANDLE)) {
            keryx_vk_pad_release(h); return false;
        }
    }
    p.tiles = tiles;
    memcpy(p.seed, seed, 32);
    KERYX_LOG_INFO("keryx-llama-vk: PoM v5 context pad ready: %llu MiB on Vulkan device %zu\n", (unsigned long long)(tiles / 1024), h->gpu);
    return true;
}

extern "C" {
int keryx_llama_pom_v5_abi() { return 1; }

void keryx_llama_pom_release_pad(KeryxLlama* h) {
    if (!h) return;
    std::lock_guard<std::mutex> lock(h->walk_lock);
    keryx_vk_pad_release(h);
}

// Returns -3 for an unavailable pad (e.g. insufficient VRAM), -2 for a failed
// dispatch, -1 for no winner, otherwise the lowest batch-relative winning nonce.
int64_t keryx_llama_pom_mine_v5(KeryxLlama* h, const uint64_t p[4], const uint64_t s[4], const uint64_t t[4],
                               uint64_t timestamp, uint64_t start_nonce, uint32_t batch,
                               const uint64_t pad_seed[4], uint64_t pad_tiles) {
    if (!h || !h->walk_ready || !p || !s || !t || !pad_seed || !batch || batch > 8192 || !pad_tiles) return -2;
    const uint64_t n_tiles = h->n_chunks / 32;
    if (!n_tiles || pad_tiles > UINT64_MAX / 256 || n_tiles > UINT64_MAX - pad_tiles) return -2;
    std::lock_guard<std::mutex> lock(h->walk_lock);
    try {
        if (!keryx_vk_pad_ensure(h, pad_seed, pad_tiles)) return h->device_failed ? -2 : -3;
        PushWalkV5 pv{};
        auto& pw = pv.walk;
        memcpy(pw.p, p, 32); memcpy(pw.s, s, 32); memcpy(pw.t, t, 32);
        pw.timestamp = timestamp; pw.n_tiles = n_tiles; pw.start_nonce = start_nonce;
        pw.k = 256; pw.batch = batch; pw.n_tensors = h->n_tensors; pw.h10 = 1;
        pw.n_tiles_inv = (~0ULL / n_tiles) + (((~0ULL % n_tiles) + 1 == n_tiles) ? 1 : 0);
        pv.pad_address = h->pad->address; pv.pad_tiles = pad_tiles;
        pv.pad_inverse = (~0ULL / pad_tiles) + (((~0ULL % pad_tiles) + 1 == pad_tiles) ? 1 : 0);
        pv.pad_steps = (uint32_t)std::max<uint64_t>(1, std::min<uint64_t>(255, (256 * pad_tiles + (n_tiles + pad_tiles) / 2) / (n_tiles + pad_tiles)));
        if (!run_dispatch(h, h->pad->walk_pipe, h->pad->walk_pl, h->walk_ds, &pv, sizeof(pv), (batch + 7) / 8, h->winner_buf, true)) return -2;
        uint32_t w; memcpy(&w, h->winner_map, 4);
        return w == NO_WINNER ? -1 : (int64_t)w;
    } catch (...) { return -2; }
}

// Bounded diagnostic readback of the actual device-filled pad. This is also used
// by the host's exactness test before a v5 backend is qualified for release.
int keryx_llama_pom_pad_read(KeryxLlama* h, uint64_t first, uint8_t* out, size_t bytes) {
    if (!h || !out || !bytes || bytes > 65536 || first % 4 || bytes % 4) return -1;
    std::lock_guard<std::mutex> lock(h->walk_lock);
    if (!h->pad || first > h->pad->tiles * 1024 || bytes > h->pad->tiles * 1024 - first) return -1;
    VkBuffer buffer = VK_NULL_HANDLE; VkDeviceMemory memory = VK_NULL_HANDLE; void* mapped = nullptr;
    if (!make_buffer(h, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     false, &buffer, &memory, &mapped)) return -1;
    vkResetCommandBuffer(h->cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(h->cmd, &bi);
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(h->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    for (size_t done = 0; done < bytes;) {
        const uint64_t absolute = first + done;
        const uint64_t local = absolute % KERYX_VK_ALLOC_BYTES;
        const size_t n = (size_t)std::min<uint64_t>(bytes - done, KERYX_VK_ALLOC_BYTES - local);
        VkBufferCopy copy{local, done, n};
        vkCmdCopyBuffer(h->cmd, h->pad->shards[absolute / KERYX_VK_ALLOC_BYTES].first, buffer, 1, &copy);
        done += n;
    }
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(h->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(h->cmd);
    const bool ok = submit_and_wait(h);
    if (ok) memcpy(out, mapped, bytes);
    vkUnmapMemory(h->dev, memory); vkDestroyBuffer(h->dev, buffer, nullptr); vkFreeMemory(h->dev, memory, nullptr);
    return ok ? 0 : -1;
}
}
