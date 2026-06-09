# OpenCL support for Keryx-Miner

This is an experimental plugin to support OpenCL (primarily for AMD GPUs).

## AMD GPU Compatibility

| Generation | Architecture | Pre-compiled Binary | JIT Source | Status |
|---|---|---|---|---|
| RX 580/570 | Ellesmere (GCN 4) | Yes | Yes | Supported |
| RX 5700 | gfx1010 (RDNA 1) | Yes | Yes | Supported |
| RX 5500/5600 | gfx1011/1012 (RDNA 1) | Yes | Yes | Supported |
| RX 6600-6900 | gfx1030-1034 (RDNA 2) | Yes | Yes | Supported |
| **RX 7600-7900** | gfx1100-1103 (RDNA 3) | No | Yes | Supported (v_dot4/v_dot8) |
| **RX 9070** | gfx1200/1201 (RDNA 4) | No | Yes | Supported (v_dot4/v_dot8) |

**RDNA 3/4 users:** AMD is deprecating OpenCL in favor of ROCm/HIP. Install the
`rocm-opencl-runtime` package for the most stable OpenCL experience on newer GPUs.
A native HIP backend may be added in a future release.

## Compiling AMD Binaries

Download and install [Radeon GPU Analyzer (RGA)](https://gpuopen.com/rga/) to compile
pre-compiled OpenCL binaries for AMD architectures.

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