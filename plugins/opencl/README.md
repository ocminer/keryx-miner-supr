# OpenCL support for Keryx-Miner

This is an experimental plugin to support OpenCL (primarily for AMD GPUs).

## AMD GPU Compatibility

| Generation | Architecture | Archived Binary | JIT Source | Status |
|---|---|---|---|---|
| RX 580/570 | Ellesmere (GCN 4) | Retired | Yes | Supported |
| RX 5700 | gfx1010 (RDNA 1) | Retired | Yes | Supported |
| RX 5500/5600 | gfx1011/1012 (RDNA 1) | Retired | Yes | Supported |
| RX 6600-6900 | gfx1030-1034 (RDNA 2) | Retired | Yes | Supported |
| **MI50/MI60** | gfx906 (GCN 5 / CDNA) | No (JIT v_dot8) | Yes | Supported (v_dot8 default) |
| **RX 7600-7900** | gfx1100-1103 (RDNA 3) | No | Yes | Supported (v_dot8 default) |
| **RX 9070** | gfx1200/1201 (RDNA 4) | No | Yes | Supported (v_dot8 default) |

**v_dot8 by default:** the matrix multiply uses the packed 4-bit `v_dot8_u32_u4`
dot product (8 MACs/instr — half the instructions of `v_dot4_u32_u8`) on every arch
that implements the DLOPS dot instructions, with no flag required. This is now
capability-driven in `worker.rs` (see `vdot8_default`). MI50/MI60 (gfx906) and CDNA
parts JIT-compile this kernel at startup — the previously shipped `gfx906` binary was
a `v_dot4` build (slower, and incompatible with the packed-matrix host layout), so it
was removed. Measured gains vs the old v_dot4 path: RX 7600 XT 287→302 kH→MH/s (+5%),
MI50 285→306 MH/s (+7%). The RDNA 1.5/2 parts (gfx1011/1012/1030-1034) are also
v_dot8-capable; pass `--experimental-amd` to JIT the v_dot8 kernel.

The old architecture-specific files under `resources/bin/` are retained only as
development history. They predate the current `UINT64_MAX` + atomic-min winner
contract and are neither embedded nor loaded. `--opencl-use-amd-binary` remains an
accepted no-op for launch-script compatibility; every production worker JIT-compiles
the canonical source for its actual device and driver.

**RDNA 3/4 users:** AMD is deprecating OpenCL in favor of ROCm/HIP. Install the
`rocm-opencl-runtime` package for the most stable OpenCL experience on newer GPUs.
A native HIP backend may be added in a future release.

## Regenerating Archived AMD Binaries

Download and install [Radeon GPU Analyzer (RGA)](https://gpuopen.com/rga/) to compile
OpenCL binaries for AMD architectures. Do not re-enable the binary loader until all
architectures have been regenerated from the current source and validated against
the source-JIT kernel on matching hardware.

### RDNA 3 / RDNA 4 (v_dot8 capable)

```shell
for arch in gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201
do
  rga --O3 -s opencl -c "$arch" \
    --OpenCLoption "-cl-finite-math-only -cl-mad-enable " \
    -b plugins/opencl/resources/bin/keryx-opencl.bin \
    plugins/opencl/resources/keryx-opencl.cl \
    -D __FORCE_AMD_V_DOT8_U32_U4__=1 -D OPENCL_PLATFORM_AMD -D OFFLINE
done
```

### RDNA 2 / RDNA 1 (v_dot8 capable)

```shell
for arch in gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx906
do
  rga --O3 -s opencl -c "$arch" \
    --OpenCLoption "-cl-finite-math-only -cl-mad-enable " \
    -b plugins/opencl/resources/bin/keryx-opencl.bin \
    plugins/opencl/resources/keryx-opencl.cl \
    -D __FORCE_AMD_V_DOT8_U32_U4__=1 -D OPENCL_PLATFORM_AMD -D OFFLINE
done
```

### RDNA 1 entry-level (v_dot4 only)

```shell
for arch in gfx1010
do
  rga --O3 -s opencl -c "$arch" \
    --OpenCLoption "-cl-finite-math-only -cl-mad-enable " \
    -b plugins/opencl/resources/bin/keryx-opencl.bin \
    plugins/opencl/resources/keryx-opencl.cl \
    -D OPENCL_PLATFORM_AMD
done
```

### GCN 4 (Ellesmere, legacy PAL ABI)

```shell
for arch in Ellesmere
do
  rga --O3 -s opencl -c "$arch" \
    --OpenCLoption "-cl-finite-math-only -cl-mad-enable -target amdgcn-amd-amdpal" \
    -b plugins/opencl/resources/bin/keryx-opencl.bin \
    plugins/opencl/resources/keryx-opencl.cl \
    -D OPENCL_PLATFORM_AMD -D PAL
done
```
