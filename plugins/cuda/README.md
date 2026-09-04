# CUDA support for keryx-miner-supr

## Building

The plugin is a shared library file that resides in the same directory as the miner.
You can build the library by running
```sh
cargo build -p keryxcuda
```

The plugin embeds architecture-specific PTX for sm_61, sm_75, sm_80, sm_86, sm_89, sm_90,
sm_100, and sm_120. Regenerate all of them from the canonical CUDA source with:

```sh
./plugins/cuda/regenerate-ptx.sh
```

The script uses digest-pinned NVIDIA CUDA containers:

- CUDA 12.2.2 for sm_61 through sm_90. PTX ISA 8.2 preserves the original older-driver floor.
- CUDA 12.8.0 for sm_100 and sm_120.

It stages the complete set, verifies the target and `atomicMin` winner contract, then replaces the
resources atomically per file and writes `resources/PTX_MANIFEST.txt` with source and artifact hashes.
Docker is required; compilation itself does not require a GPU.

The winner slot uses `u64::MAX` for no result. CUDA publishes the lowest qualifying nonce with
`atomicMin`, and hash equality is accepted (`hash <= target`) exactly like consensus. A new host
negotiates raw-MAX output through the optional `keryx_plugin_enable_raw_nonce_v1` symbol, which makes
nonce zero representable without changing the Rust plugin trait ABI. Without that handshake the
plugin translates MAX to the historical zero sentinel, preserving compatibility with older miners.
