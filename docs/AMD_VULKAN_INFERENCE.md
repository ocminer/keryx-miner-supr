# AMD Vulkan inference and H16 mining

The AMD backend uses Vulkan for both traced inference and H16 mining. The
consensus algorithm and activation gate are unchanged.

Each selected AMD worker owns a Vulkan model and tracing context, matched by its
full PCI identity. Before mining, **that worker** must pass traced inference and
the model-byte gate. Another card's successful self-test cannot qualify it.
Only the serving card pauses and drains its outstanding walk during generation
or an opening; the other cards continue mining. The same resident weights serve
inference and mining. The regenerable context pad is released for inference and
rebuilt when mining resumes. Model loading is serialized to avoid simultaneous
host-memory staging on small mining systems.

An AMD worker without a matching Vulkan device, enough model memory, a passing
trace, or a valid gather stays paused with a diagnostic. It never falls back to
mining-only OpenCL v4 for an H16 job. OpenCL and Vulkan enumeration order is not
assumed to match. For multiple cards, remove a single-card
`KERYX_LLAMA_VK_DEVICE` pin and ensure the Vulkan driver exposes every selected
physical GPU. OpenCL device labels follow the miner's selected worker order.

## RX 5600/5700 compatibility

v0.16.5 corrected Vulkan integer-dot support for canonical traced inference on
RDNA1, but the historical RDNA1 BDA mining exclusion remained. After successful
inference, affected rigs could reach OpenCL v4, which cannot execute H16 PoM v5.
Removing that exclusion alone is unsafe: earlier NAVI10 field reports recorded a
GPU hang in the raw-address gather.

The new native sidecar automatically selects a scalar, descriptor-backed path
for NAVI10/12/14. Weight reads, byte verification, context-pad writes and pad
reads use bounded buffer descriptors instead of physical storage-buffer
pointers. Constant descriptor indices avoid adding descriptor-indexing feature
requirements. Host code checks descriptor limits and bounds; no failed setup
falls back to the old gather. A runtime capability check prevents an older
sidecar from enabling RDNA1 zero-dup mining.

Other supported AMD architectures keep their existing scalar/dot/matrix kernel
selection. Linux AVX and no-AVX engines, and the Windows Vulkan build recipe and
exports, include the compatibility shaders. Copying only the miner executable
without its matching Vulkan engine is insufficient.

## Compatibility status

The descriptor path remains experimental on physical RDNA1 hardware. Windows
GPU execution also requires hardware qualification. Use the matching miner and
Vulkan engine together; a successful build alone does not establish device or
driver compatibility.
