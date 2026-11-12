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
    uint32_t nonces_per_group = 8;
    uint32_t preferred_batch = 1024;
    bool tuned = false;
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
    h->pages.resize(h->weight_pages);
    update_descriptor_pages(h);
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
    if (!p.fill_pipe) p.fill_pipe = make_pipeline(h, h->descriptor_memory ? PAD_DESCRIPTOR_SPV : PAD_V5_FILL_SPV, h->descriptor_memory ? sizeof(PAD_DESCRIPTOR_SPV) : sizeof(PAD_V5_FILL_SPV), sizeof(PushPadFill), &p.fill_sm, &p.fill_pl);
    if (!p.walk_pipe) {
        const char* selected = getenv("KERYX_VK_V5_DOT");
        const bool dot = keryx_vk_integer_dot_product(h->gpu) && !(selected && strcmp(selected, "0") == 0);
        p.walk_pipe = make_pipeline(h, h->descriptor_memory ? WALK_V5_DESCRIPTOR_SPV : (dot ? WALK_V5_DOT_SPV : WALK_V5_SCALAR_SPV),
            h->descriptor_memory ? sizeof(WALK_V5_DESCRIPTOR_SPV) : (dot ? sizeof(WALK_V5_DOT_SPV) : sizeof(WALK_V5_SCALAR_SPV)), sizeof(PushWalkV5), &p.walk_sm, &p.walk_pl);
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
        const uint64_t address = walk_buffer_address(h,buffer,std::min(KERYX_VK_ALLOC_BYTES, tiles * 1024 - first));
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
    p.address = walk_buffer_address(h,p.buffer,addresses.size()*sizeof(uint64_t));
    update_descriptor_pages(h);
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

// Pipelines belong to one model owner/card. Only exact candidates compete with
// the established kernel, and a small timing difference retains that fallback.
static bool keryx_vk_tune_v5(KeryxLlama* h, const PushWalkV5& job) {
    if (h->device_failed) return false;
    auto& pad = *h->pad;
    if (pad.tuned) return !h->device_failed;
    if (h->descriptor_memory) { pad.tuned=true; return true; } // no BDA/matrix candidates on RDNA1
    const char* mode = getenv("KERYX_VK_V5_MODE");
    const char* dot = getenv("KERYX_VK_V5_DOT");
    if ((mode && (strcmp(mode,"sp")==0 || strcmp(mode,"0")==0)) ||
        (!mode && dot && strcmp(dot,"0")==0)) { pad.tuned = true; return true; }
    VkPhysicalDeviceSubgroupSizeControlProperties control{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &subgroup; subgroup.pNext = &control;
    vkGetPhysicalDeviceProperties2(h->phys, &props);
    VkPhysicalDeviceSubgroupSizeControlFeatures features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f.pNext = &features;
    vkGetPhysicalDeviceFeatures2(h->phys, &f);
    // The pinned ggml backend enables the supported subgroup-size/full-subgroup
    // features. Keep non-AMD backends and unsupported subgroup shapes unchanged.
    const bool usable = props.properties.vendorID == 0x1002 && features.subgroupSizeControl && features.computeFullSubgroups &&
        (subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) &&
        (subgroup.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
    struct Choice {
        VkPipeline pipe; VkPipelineLayout layout; VkShaderModule shader;
        uint32_t npg; const char* name; double seconds = 0; bool exact = true;
    };
    std::vector<Choice> choices{{pad.walk_pipe,pad.walk_pl,pad.walk_sm,8,"established"}};
    auto add = [&](const uint32_t* code, size_t bytes, uint32_t wave, uint32_t npg, const char* name) {
        if (wave < control.minSubgroupSize || wave > control.maxSubgroupSize) return;
        // Fixed-wave GCN reports no requiredSubgroupSizeStages: its native
        // width is already guaranteed. Only request a width when supported.
        const bool request_wave = (control.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
        if (!request_wave && (control.minSubgroupSize != wave || control.maxSubgroupSize != wave)) return;
        Choice c{VK_NULL_HANDLE,VK_NULL_HANDLE,VK_NULL_HANDLE,npg,name};
        c.pipe = make_pipeline(h,code,bytes,sizeof(PushWalkV5),&c.shader,&c.layout,request_wave ? wave : 0,false,true);
        if (c.pipe) choices.push_back(c);
    };
    if (usable) {
        if (keryx_vk_coopmat_int8(h->gpu)) {
            add(WALK_V5_CM_SPV,sizeof(WALK_V5_CM_SPV),32,1,"matrix");
            add(WALK_V5_RESOLVED_SPV,sizeof(WALK_V5_RESOLVED_SPV),32,1,"matrix-resolved");
        }
        if (keryx_vk_integer_dot_product(h->gpu) && (subgroup.subgroupSize==32 || subgroup.subgroupSize==64))
            add(WALK_V5_REDUCE_SPV,sizeof(WALK_V5_REDUCE_SPV),subgroup.subgroupSize,8,"subgroup-dot");
    }
    auto dispose = [&](Choice& c) {
        vkDestroyPipeline(h->dev,c.pipe,nullptr);
        vkDestroyPipelineLayout(h->dev,c.layout,nullptr);
        vkDestroyShaderModule(h->dev,c.shader,nullptr);
    };
    auto run = [&](const Choice& c, const PushWalkV5& p, uint32_t& winner) {
        if (!run_dispatch(h,c.pipe,c.layout,h->walk_ds,&p,sizeof(p),(p.walk.batch+c.npg-1)/c.npg,h->winner_buf,true)) return false;
        memcpy(&winner,h->winner_map,4); return true;
    };
    bool ok = true;
    PushWalkV5 check=job; check.walk.batch=1027;
    check.walk.t[0]=check.walk.t[1]=check.walk.t[2]=~0ULL; check.walk.t[3]=~0ULL>>9;
    for (unsigned round=0;round<8 && ok && choices.size()>1;++round) {
        check.walk.start_nonce=job.walk.start_nonce+0x5eed00000000ULL+round*1027ULL*7919ULL;
        uint32_t reference;
        if (!run(choices[0],check,reference)) { ok=false; break; }
        for (size_t i=1;i<choices.size();++i) if (choices[i].exact) {
            uint32_t got;
            if (!run(choices[i],check,got)) { ok=false; break; }
            if (got!=reference) {
                choices[i].exact=false;
                KERYX_LOG_WARN("keryx-llama-vk: v5 %s failed exactness on device %zu; retaining fallback\n",choices[i].name,h->gpu);
            }
        }
    }
    PushWalkV5 timed=job; timed.walk.batch=4096; memset(timed.walk.t,0,32);
    if (ok && choices.size()>1) {
        const auto begin=std::chrono::steady_clock::now();
        do { uint32_t unused; if (!run(choices[0],timed,unused)) {ok=false;break;} }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<0.25);
        for (unsigned round=0;round<4 && ok;++round) for(size_t pos=0;pos<choices.size();++pos) {
            const size_t i=(round&1)?choices.size()-1-pos:pos;
            if (!choices[i].exact) continue;
            uint32_t unused;
            if (!run(choices[i],timed,unused)) {ok=false;break;}
            const auto start=std::chrono::steady_clock::now();
            if (!run(choices[i],timed,unused)) {ok=false;break;}
            choices[i].seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
    }
    size_t best=0;
    if (ok) for(size_t i=0;i<choices.size();++i) if (choices[i].exact) {
        KERYX_LOG_INFO("keryx-llama-vk: v5 device %zu %s cross-checked: %.3f ms / 4096 nonces\n",h->gpu,choices[i].name,choices[i].seconds*250.0);
        if (choices[i].seconds>0 && choices[i].seconds<choices[best].seconds*0.995) best=i;
    }
    // Explicit diagnostic overrides still have to pass the reference check.
    if (ok && mode) for(size_t i=1;i<choices.size();++i) if(choices[i].exact &&
        ((strcmp(mode,"cm")==0 && strcmp(choices[i].name,"matrix")==0) ||
         (strcmp(mode,"resolved")==0 && strcmp(choices[i].name,"matrix-resolved")==0) ||
         (strcmp(mode,"reduce")==0 && strcmp(choices[i].name,"subgroup-dot")==0))) best=i;
    // Tune bounded sub-dispatches on this card/model as well. Small fixed
    // batches can leave a large GPU underfilled and spend unnecessary CPU time
    // submitting fences. Keep each selected launch below 20 ms.
    if (ok && usable) {
        const uint32_t sizes[] = {1024,4096,8192};
        double seconds[3]{};
        for (unsigned round=0;round<4 && ok;++round) for (unsigned pos=0;pos<3;++pos) {
            const unsigned i=(round&1)?2-pos:pos;
            timed.walk.batch=sizes[i]; uint32_t unused;
            if (!run(choices[best],timed,unused)) {ok=false;break;}
            const auto start=std::chrono::steady_clock::now();
            if (!run(choices[best],timed,unused)) {ok=false;break;}
            seconds[i]+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        }
        double rate=0;
        if (ok) for(unsigned i=0;i<3;++i) {
            const double elapsed=seconds[i]/4.0, candidate=sizes[i]/elapsed;
            if(elapsed>0 && elapsed<=0.020 && candidate>rate*1.005) {
                pad.preferred_batch=sizes[i];rate=candidate;
            }
        }
    }
    // A failed fence may still have a submitted command using any candidate.
    // Match the engine's sticky-failure policy: retain resources until exit.
    if (!ok) return false;
    for(size_t i=0;i<choices.size();++i) if(i!=best) dispose(choices[i]);
    pad.walk_pipe=choices[best].pipe; pad.walk_pl=choices[best].layout;pad.walk_sm=choices[best].shader;
    pad.nonces_per_group=choices[best].npg;pad.tuned=ok;
    if(ok) KERYX_LOG_INFO("keryx-llama-vk: v5 selected %s on device %zu (%u nonces/workgroup, %u nonces/dispatch)\n",choices[best].name,h->gpu,pad.nonces_per_group,pad.preferred_batch);
    return ok;
}

extern "C" {
int keryx_llama_pom_v5_abi() { return 1; }

// Optional ABI-1 extension; older hosts/sidecars retain their fixed batch size.
uint32_t keryx_llama_pom_v5_batch(KeryxLlama* h) {
    if (!h) return 1024;
    std::lock_guard<std::mutex> lock(h->walk_lock);
    return h->pad && h->pad->tuned ? h->pad->preferred_batch : 1024;
}

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
        if (!keryx_vk_tune_v5(h, pv)) return -2;
        const uint32_t npg = h->pad->nonces_per_group;
        if (!run_dispatch(h, h->pad->walk_pipe, h->pad->walk_pl, h->walk_ds, &pv, sizeof(pv), (batch + npg - 1) / npg, h->winner_buf, true)) return -2;
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
