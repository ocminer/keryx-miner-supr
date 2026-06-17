// Standalone bench + byte-exact correctness harness for the KeryxHash core.
// Compile (sm_120 example):
//   nvcc -O3 -DKERYX_BENCH -gencode=arch=compute_120,code=sm_120 \
//        --use_fast_math -Xptxas -O3 -o bench_main bench_main.cu
// Run:  ./bench_main [N_nonces] [iters]
//   prints GH/s (compute throughput), an FNV-1a checksum over every output
//   byte (the byte-exact correctness signature), and the first 4 hashes.
// The checksum + first-4 must be IDENTICAL before/after any matmul change.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define KERYX_BENCH
#include "kaspa-cuda.cu"

#define CK(x) do{ cudaError_t _ck=(x); if(_ck){ printf("CUDA err %s @ %d: %s\n",#x,__LINE__,cudaGetErrorString(_ck)); exit(1);} }while(0)

int main(int argc, char** argv) {
    uint64_t N    = (argc > 1) ? strtoull(argv[1], 0, 10) : (1ULL << 20);
    int      iters= (argc > 2) ? atoi(argv[2]) : 50;

    // Deterministic synthetic constants (values irrelevant to correctness
    // comparison as long as both builds use the same ones; matrix kept to
    // nibbles 0..15 to match the real KeryxHash matrix energy profile).
    uint8_t h_matrix[64][64];
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            h_matrix[i][j] = (uint8_t)((i * 7 + j * 13 + 3) & 0x0F);
    uint8_t h_header[72];
    for (int i = 0; i < 72; i++) h_header[i] = (uint8_t)(i * 31 + 7);
    uint256_t h_target;
    for (int i = 0; i < 4; i++) h_target.number[i] = ~0ULL; // unused by dump kernel

    CK(cudaMemcpyToSymbol(matrix, h_matrix, sizeof(h_matrix)));
    CK(cudaMemcpyToSymbol(hash_header, h_header, sizeof(h_header)));
    CK(cudaMemcpyToSymbol(target, &h_target, sizeof(h_target)));

    uint8_t* d_out;
    CK(cudaMalloc(&d_out, N * 32));
#ifdef KERYX_WMMA
    int tpb = 128;                                     // 4 warps; wmma kernel is fixed at 128
    #define DUMP_KERNEL heavy_hash_dump_wmma
    printf("[KERYX_WMMA: tensor-core matmul path]\n");
#else
    int tpb = 256;
    #define DUMP_KERNEL heavy_hash_dump
#endif
    uint64_t blocks = (N + tpb - 1) / tpb;

    cudaEvent_t s, e; CK(cudaEventCreate(&s)); CK(cudaEventCreate(&e));
    DUMP_KERNEL<<<blocks, tpb>>>(0, N, d_out);         // warmup
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(s));
    for (int it = 0; it < iters; it++)
        DUMP_KERNEL<<<blocks, tpb>>>((uint64_t)it * N, N, d_out);
    CK(cudaEventRecord(e)); CK(cudaEventSynchronize(e));
    float ms = 0; CK(cudaEventElapsedTime(&ms, s, e));
    double hashes = (double)N * iters;
    printf("N=%llu iters=%d time=%.3f ms  rate=%.4f GH/s\n",
           (unsigned long long)N, iters, ms, hashes / (ms * 1e6));

    // Correctness signature over the LAST iteration's output.
    uint8_t* h_out = (uint8_t*)malloc(N * 32);
    CK(cudaMemcpy(h_out, d_out, N * 32, cudaMemcpyDeviceToHost));
    uint64_t csum = 1469598103934665603ULL;
    for (uint64_t i = 0; i < N * 32; i++) { csum ^= h_out[i]; csum *= 1099511628211ULL; }
    printf("FNV1a(all)=%016llx\n", (unsigned long long)csum);
    for (int k = 0; k < 4; k++) {
        printf("nonce %d: ", k);
        for (int b = 0; b < 32; b++) printf("%02x", h_out[k * 32 + b]);
        printf("\n");
    }
    free(h_out);
    return 0;
}
