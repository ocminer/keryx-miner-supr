// Standalone on-device build check for keryx-opencl.cl.
// Compiles the kernel for the AMD gfx1102 (RX 7600 XT) with the exact flags
// the Rust worker's `from_source()` uses, then creates the `heavy_hash` kernel.
// Build: cc kbuild_test.c -lOpenCL -o kbuild_test   (needs -L for libOpenCL.so)
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(n + 1);
    if (fread(buf, 1, n, f) != (size_t)n) { perror("fread"); exit(1); }
    buf[n] = 0; *len = n; fclose(f); return buf;
}

int main(int argc, char **argv) {
    const char *src_path = argc > 1 ? argv[1] : "resources/keryx-opencl.cl";
    // Match worker.rs from_source(): which arch to target via -D __<name>__
    const char *arch = argc > 2 ? argv[2] : "gfx1102";
    // experimental_amd? 0 = default (v_dot4_u32_u8 path), 1 = v_dot8_u32_u4 path
    int experimental = argc > 3 ? atoi(argv[3]) : 0;

    cl_uint nplat = 0;
    clGetPlatformIDs(0, NULL, &nplat);
    cl_platform_id *plats = calloc(nplat, sizeof(*plats));
    clGetPlatformIDs(nplat, plats, NULL);

    cl_platform_id plat = 0; cl_device_id dev = 0;
    char buf[256];
    for (cl_uint p = 0; p < nplat && !dev; p++) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &nd) != CL_SUCCESS) continue;
        cl_device_id *devs = calloc(nd, sizeof(*devs));
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, nd, devs, NULL);
        for (cl_uint d = 0; d < nd; d++) {
            clGetDeviceInfo(devs[d], CL_DEVICE_NAME, sizeof(buf), buf, NULL);
            if (strstr(buf, arch)) { plat = plats[p]; dev = devs[d]; break; }
        }
        free(devs);
    }
    if (!dev) { fprintf(stderr, "no device matching '%s'\n", arch); return 1; }
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(buf), buf, NULL);
    printf("Device: %s\n", buf);
    char ver[128]; clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(ver), ver, NULL);
    printf("Version: %s\n", ver);

    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (err) { fprintf(stderr, "ctx err %d\n", err); return 1; }

    size_t srclen; char *src = slurp(src_path, &srclen);
    const char *srcs[1] = { src };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, &srclen, &err);
    if (err) { fprintf(stderr, "prog err %d\n", err); return 1; }

    // Flags mirror worker.rs from_source() for an AMD gfx1102 OpenCL-2.x device.
    char opts[512];
    snprintf(opts, sizeof(opts),
        "%s-cl-mad-enable -cl-finite-math-only -cl-std=CL2.0 "
        "-DAMD_ACCELERATED_PARALLEL_PROCESSING "
        "-D OPENCL_PLATFORM_AMD -D __%s__ ",
        experimental ? "-D __FORCE_AMD_V_DOT8_U32_U4__=1 " : "",
        arch);
    printf("Build options: %s\n", opts);

    err = clBuildProgram(prog, 1, &dev, opts, NULL, NULL);
    size_t logsz = 0;
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
    if (logsz > 1) {
        char *log = malloc(logsz);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log, NULL);
        printf("---- build log ----\n%s\n-------------------\n", log);
        free(log);
    }
    if (err) { fprintf(stderr, "BUILD FAILED err=%d\n", err); return 1; }

    cl_kernel k = clCreateKernel(prog, "heavy_hash", &err);
    if (err) { fprintf(stderr, "kernel create FAILED err=%d\n", err); return 1; }
    printf("OK: heavy_hash kernel built and created for %s\n", arch);
    clReleaseKernel(k); clReleaseProgram(prog); clReleaseContext(ctx);
    return 0;
}
