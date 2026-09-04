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

#include <cstdio>
#include <cstdlib>
#include <cstring>

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
// is correct by construction. When the miner publishes KERYX_LLAMA_VK_AUTO_PCI_ALLOWLIST, only
// physical devices whose `dddd:bb:dd.f` PCI identity occurs in that selected-OpenCL-worker list are
// eligible. An explicitly empty allowlist therefore has no auto candidate (fail closed). Without
// the variable, retain direct-library compatibility and consider every discrete device. Among the
// eligible devices prefer the LARGEST discrete AMD GPU (stable tie: lowest ggml index), else the
// largest discrete GPU of any vendor.
static bool keryx_vk_physical_device_pci(
        vk::PhysicalDevice physical,
        uint32_t * domain, uint32_t * bus, uint32_t * device, uint32_t * function) {
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT
    };
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &pci;
    vkGetPhysicalDeviceProperties2((VkPhysicalDevice) physical, &properties);
    if (pci.pciBus == 0 && pci.pciDevice == 0 && pci.pciDomain == 0 && pci.pciFunction == 0) {
        return false;
    }
    *domain = pci.pciDomain;
    *bus = pci.pciBus;
    *device = pci.pciDevice;
    *function = pci.pciFunction;
    return true;
}

// Resolve a ggml `main_gpu` index back to its physical PCI identity. The Rust host calls this
// before model allocation/spawn so a selected non-largest card is dedicated and its resident walk
// released in time; an explicitly pinned inference-only card outside the mining subset instead
// releases the provisional reservation. The index is resolved inside ggml's own filtered list.
extern "C" bool keryx_vk_device_pci(
        int dev_num, uint32_t * domain, uint32_t * bus, uint32_t * device, uint32_t * function) {
    if (domain == nullptr || bus == nullptr || device == nullptr || function == nullptr) {
        return false;
    }
    try {
        ggml_vk_instance_init();
        if (dev_num < 0 || (size_t) dev_num >= vk_instance.device_indices.size()) {
            return false;
        }
        std::vector<vk::PhysicalDevice> phys = vk_instance.instance.enumeratePhysicalDevices();
        size_t raw = vk_instance.device_indices[(size_t) dev_num];
        if (raw >= phys.size()) {
            return false;
        }
        return keryx_vk_physical_device_pci(phys[raw], domain, bus, device, function);
    } catch (...) {
        return false;
    }
}

static bool keryx_vk_auto_pci_allowed(vk::PhysicalDevice physical) {
    const char * allowed = std::getenv("KERYX_LLAMA_VK_AUTO_PCI_ALLOWLIST");
    if (allowed == nullptr) {
        return true;
    }
    if (*allowed == '\0') {
        return false;
    }

    uint32_t domain = 0, bus = 0, device = 0, function = 0;
    if (!keryx_vk_physical_device_pci(physical, &domain, &bus, &device, &function)) {
        return false;
    }

    char key[32];
    std::snprintf(
        key, sizeof(key), "%04x:%02x:%02x.%x",
        domain, bus, device, function);
    const size_t key_len = std::strlen(key);
    const char * cursor = allowed;
    while (*cursor != '\0') {
        const char * comma = std::strchr(cursor, ',');
        const char * end = comma;
        if (comma == nullptr) {
            end = cursor + std::strlen(cursor);
        }
        while (cursor < end && (*cursor == ' ' || *cursor == '\t')) {
            cursor++;
        }
        while (end > cursor && (end[-1] == ' ' || end[-1] == '\t')) {
            end--;
        }
        if ((size_t) (end - cursor) == key_len && std::strncmp(cursor, key, key_len) == 0) {
            return true;
        }
        cursor = comma != nullptr ? comma + 1 : end;
    }
    return false;
}

// Optional capability probe used by a new miner with an older sidecar. Auto-placement is allowed
// to proceed under an active PCI allowlist only when this symbol is present and >= 1; otherwise the
// Rust host refuses to trust an old picker which can still select an unrelated platform/card.
extern "C" int keryx_vk_picker_abi() {
    return 1;
}

extern "C" int keryx_vk_pick_discrete_device() {
    try {
        ggml_vk_instance_init();
        std::vector<vk::PhysicalDevice> phys = vk_instance.instance.enumeratePhysicalDevices();
        int largest_discrete = -1, largest_amd_discrete = -1;
        uint64_t largest_discrete_bytes = 0, largest_amd_discrete_bytes = 0;
        for (size_t i = 0; i < vk_instance.device_indices.size(); i++) {
            size_t raw = vk_instance.device_indices[i];
            if (raw >= phys.size()) {
                continue;
            }
            vk::PhysicalDeviceProperties props = phys[raw].getProperties();
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu && keryx_vk_auto_pci_allowed(phys[raw])) {
                vk::PhysicalDeviceMemoryProperties memory = phys[raw].getMemoryProperties();
                uint64_t device_local_bytes = 0;
                for (uint32_t heap = 0; heap < memory.memoryHeapCount; heap++) {
                    if (memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                        device_local_bytes += (uint64_t) memory.memoryHeaps[heap].size;
                    }
                }
                if (largest_discrete < 0 || device_local_bytes > largest_discrete_bytes) {
                    largest_discrete = (int) i;
                    largest_discrete_bytes = device_local_bytes;
                }
                if (props.vendorID == 0x1002
                    && (largest_amd_discrete < 0 || device_local_bytes > largest_amd_discrete_bytes)) {
                    largest_amd_discrete = (int) i;
                    largest_amd_discrete_bytes = device_local_bytes;
                }
            }
        }
        return largest_amd_discrete >= 0 ? largest_amd_discrete : largest_discrete;
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
