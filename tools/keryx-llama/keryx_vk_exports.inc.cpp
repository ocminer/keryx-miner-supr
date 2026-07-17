// Keryx zero-dup exports for ggml-vulkan — APPENDED to ggml-vulkan.cpp by
// hiveos/build-keryx-llama-vk.sh at build time (every symbol referenced below is file-scope
// static inside ggml-vulkan.cpp, so an end-of-file append compiles against any nearby pin;
// re-verify the byte-identity spike whenever the llama.cpp TAG is bumped).
//
// Purpose: let the miner's own Vulkan compute kernel (the PoM possession walk) read the weight
// tensors ALREADY RESIDENT in this backend's VRAM instead of uploading a second copy. Buffer
// device addresses are only dereferenceable on the device that owns them, so the walk builds its
// pipeline on THIS VkDevice and submits through the queue-mutex-guarded hook so it never races
// ggml's own submissions.

extern "C" bool keryx_vk_raw_handles(size_t dev_num, void ** vk_instance_out,
                                     void ** vk_physical_device_out, void ** vk_device_out,
                                     uint32_t * compute_queue_family_index) {
    try {
        ggml_vk_instance_init();
        if (dev_num >= vk_instance.device_indices.size()) {
            return false;
        }
        vk_device dev = ggml_vk_get_device(vk_instance.device_indices[dev_num]);
        if (!dev->buffer_device_address) {
            return false; // no VK 1.2 bufferDeviceAddress -> tensors unreachable by raw address
        }
        *vk_instance_out            = (void *) (VkInstance) vk_instance.instance;
        *vk_physical_device_out     = (void *) (VkPhysicalDevice) dev->physical_device;
        *vk_device_out              = (void *) (VkDevice) dev->device;
        *compute_queue_family_index = dev->compute_queue.queue_family_index;
        return true;
    } catch (const vk::SystemError &) {
        return false;
    }
}

extern "C" bool keryx_vk_tensor_addr(const struct ggml_tensor * tensor,
                                     uint64_t * gpu_addr, uint64_t * size_bytes) {
    if (tensor == nullptr || tensor->buffer == nullptr || !ggml_backend_buffer_is_vk(tensor->buffer)) {
        return false; // not VK-resident (e.g. a ggml-cpu host buffer) -> caller supplements
    }
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) tensor->buffer->context;
    const vk_buffer & buf = buf_ctx->dev_buffer;
    if (!buf || buf->bda_addr == 0) {
        return false;
    }
    // Identical resolution to ggml's own in-shader BDA paths (dst_addr = bda_addr +
    // vk_tensor_offset + view_offs). Weights are uploaded verbatim from the GGUF (no repack),
    // so an external reader sees the raw quant-block bytes the possession index pins.
    *gpu_addr   = (uint64_t) buf->bda_addr + vk_tensor_offset(tensor) + tensor->view_offs;
    *size_bytes = (uint64_t) ggml_nbytes(tensor);
    return true;
}

// VkDevice that owns a VK-resident tensor's buffer (nullptr when not VK-resident). BDA values
// are only dereferenceable on their owning device, so the walk matches this against
// keryx_vk_raw_handles across dev_num to self-locate the model's device — the loader's main_gpu
// request and ggml's device numbering do not always agree.
extern "C" void * keryx_vk_tensor_device(const struct ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr || !ggml_backend_buffer_is_vk(tensor->buffer)) {
        return nullptr;
    }
    ggml_backend_vk_buffer_context * buf_ctx = (ggml_backend_vk_buffer_context *) tensor->buffer->context;
    vk_device dev = buf_ctx->device.lock(); // vk_device_ref is a weak_ptr at this pin
    if (!dev) {
        return nullptr;
    }
    return (void *) (VkDevice) dev->device;
}

// Pick the discrete GPU for the llama engine, as a `main_gpu` index into GGML'S OWN device list
// (issue #18). ggml enumerates + filters physical devices in its own instance; a device index
// taken from ANY other Vulkan instance (a separate libvulkan enumeration, or GGML_VK_VISIBLE_
// DEVICES computed elsewhere) is NOT guaranteed to map to the same device — on a rig with an
// integrated GPU the orders differ, so such an index silently resolves to the iGPU (model into
// UMA system RAM) or trips a GGML_ASSERT. Resolving it HERE, against ggml's exact device_indices,
// is correct by construction. Prefers a discrete AMD GPU; else any discrete; -1 if none found.
extern "C" int keryx_vk_pick_discrete_device() {
    try {
        ggml_vk_instance_init();
        std::vector<vk::PhysicalDevice> phys = vk_instance.instance.enumeratePhysicalDevices();
        int first_discrete = -1, first_amd_discrete = -1;
        for (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
            size_t raw = vk_instance.device_indices[i];
            if (raw >= phys.size()) {
                continue;
            }
            vk::PhysicalDeviceProperties props = phys[raw].getProperties();
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                if (first_discrete < 0) {
                    first_discrete = (int) i;
                }
                if (props.vendorID == 0x1002 && first_amd_discrete < 0) {
                    first_amd_discrete = (int) i;
                }
            }
        }
        return first_amd_discrete >= 0 ? first_amd_discrete : first_discrete;
    } catch (...) {
        return -1;
    }
}

extern "C" void keryx_vk_queue_submit(size_t dev_num, const void * submit_info, void * fence) {
    vk_device dev = ggml_vk_get_device(vk_instance.device_indices[dev_num]);
    // The same lock every internal ggml submission takes -> external dispatches serialize
    // cleanly against ggml's use of the shared compute queue.
    std::lock_guard<std::mutex> guard(queue_mutex);
    dev->compute_queue.queue.submit({ *reinterpret_cast<const vk::SubmitInfo *>(submit_info) },
                                    vk::Fence((VkFence) fence));
}
