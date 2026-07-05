// H100 memory-ceiling probes (branch h100-pom-opt) — answers "can anything make the walk faster?"
//
// (A) granularity sweep: a dependent random pointer-chase (same shape as the PoM walk) reading RB
//     bytes/step for RB in {32,64,128,256}. If gathers/s is ~flat vs RB the walk is TRANSACTION/
//     latency limited (the 2x over-fetch is irrelevant, no memory trick helps); if gathers/s drops
//     as RB grows it is BYTE-bandwidth limited (over-fetch matters, but HW won't fetch <64B anyway).
// (B) L2 persistence: run the RB=32 chase with a cudaAccessPolicyWindow pinning a slice of the blob
//     in L2 — the "caching knob". Uniform-random access should show ~no gain; this quantifies it.
//
// Build: nvcc -O3 -arch=sm_90 -o probe tools/h100/probe.cu
// Run:   ./probe            (granularity sweep + L2-persist test)

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
typedef unsigned long long u64;
#define CK(x) do{cudaError_t e=(x); if(e){printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ u64 mix64(u64 x){ x^=x>>30; x*=0xbf58476d1ce4e5b9ULL; x^=x>>27; x*=0x94d049bb133111ebULL; x^=x>>31; return x; }

// dependent chase reading RB bytes/step (RB = 16*V, V ulonglong2 loads). nslots = blobbytes/RB.
template<int V>
__global__ void chase(const ulonglong2* __restrict__ blob,u64 nslots,int K,u64 nbase,u64 nn,u64* sink){
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x; if(tid>=nn) return;
    u64 state=mix64(nbase+tid), off=state%nslots;
    for(int i=0;i<K;i++){
        u64 b=off*(u64)V; u64 h=state;
        #pragma unroll
        for(int v=0;v<V;v++){ ulonglong2 a=blob[b+v]; h^=a.x; h^=a.y; }
        state=mix64(h); off=state%nslots;
    }
    if(tid<64) sink[tid]=state;
}

static double run(int V,const ulonglong2* blob,u64 blobbytes,int K,u64 nn){
    u64 nslots=blobbytes/(16ULL*V);
    u64 *sink; CK(cudaMalloc(&sink,64*sizeof(u64)));
    int blk=256; u64 g=(nn+blk-1)/blk;
    auto L=[&](){ if(V==2)chase<2><<<(unsigned)g,blk>>>(blob,nslots,K,0,nn,sink);
                  else if(V==4)chase<4><<<(unsigned)g,blk>>>(blob,nslots,K,0,nn,sink);
                  else if(V==8)chase<8><<<(unsigned)g,blk>>>(blob,nslots,K,0,nn,sink);
                  else chase<16><<<(unsigned)g,blk>>>(blob,nslots,K,0,nn,sink); };
    L(); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0)); L(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    double sec=ms/1e3, gaths=(double)nn*K, useful=gaths*16.0*V/sec/1e9;
    printf("  RB=%3dB: %.2f Ggather/s | %.0f GB/s useful | %.3f ms\n",16*V,gaths/sec/1e9,useful,ms);
    cudaFree(sink);
    return gaths/sec/1e9;
}

int main(){
    size_t blobbytes=(size_t)77604776ULL*32ULL; // 2.48 GB (real footprint)
    ulonglong2* blob; CK(cudaMalloc(&blob,blobbytes));
    CK(cudaMemset(blob,0x5a,blobbytes));
    int K=256; u64 nn=1ULL<<24;

    printf("=== (A) granularity sweep (dependent chase, K=%d, nonces=2^24) ===\n",K);
    double g32=run(2,blob,blobbytes,K,nn);   // 32B
    double g64=run(4,blob,blobbytes,K,nn);   // 64B
    double g128=run(8,blob,blobbytes,K,nn);  // 128B
    run(16,blob,blobbytes,K,nn);             // 256B
    printf("  -> gathers/s 32B vs 64B: %.2fx (%.0f%% of 32B). Flat=transaction-limited; halving=byte-limited.\n",
           g64/g32, 100.0*g64/g32);

    printf("=== (B) L2 persistence (cudaAccessPolicyWindow, RB=32) ===\n");
    int dev=0; cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,dev));
    size_t l2max=prop.persistingL2CacheMaxSize;
    printf("  L2 size=%zu MB, max persisting=%zu MB\n",(size_t)prop.l2CacheSize>>20,l2max>>20);
    // baseline
    printf("  baseline (no persist):"); run(2,blob,blobbytes,K,nn);
    // pin a slice in L2
    CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,l2max));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    cudaStreamAttrValue attr={}; attr.accessPolicyWindow.base_ptr=blob;
    attr.accessPolicyWindow.num_bytes=l2max; attr.accessPolicyWindow.hitRatio=1.0;
    attr.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    CK(cudaStreamSetAttribute(s,cudaStreamAttributeAccessPolicyWindow,&attr));
    u64 nslots=blobbytes/32ULL; u64* sink; CK(cudaMalloc(&sink,64*sizeof(u64)));
    int blk=256; u64 g=(nn+blk-1)/blk;
    chase<2><<<(unsigned)g,blk,0,s>>>(blob,nslots,K,0,nn,sink); CK(cudaStreamSynchronize(s));
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0,s)); chase<2><<<(unsigned)g,blk,0,s>>>(blob,nslots,K,0,nn,sink);
    CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    printf("  with L2 persist (%zu MB pinned): %.2f Ggather/s | %.3f ms\n",l2max>>20,(double)nn*K/(ms/1e3)/1e9,ms);
    cudaFree(blob);

    // (C) tier blob-size sweep: does a bigger model (bigger blob -> lower L2 hit rate) mine slower?
    // The walk is K=256 32B gathers/nonce regardless of N; only the L2 hit rate changes with size.
    printf("=== (C) tier blob-size sweep (32B chase, K=%d) — MH/s vs model size ===\n",K);
    struct { const char* name; double gb; } tiers[] = {
        {"light  Gemma-3-4B  ", 2.48}, {"high   Qwen3-32B   ", 19.5}, {"v-high Llama-70B-Q2", 26.0} };
    for(auto&ti:tiers){
        size_t bb=(size_t)(ti.gb*1e9); bb&=~(size_t)31; // 32B align
        ulonglong2* b2; if(cudaMalloc(&b2,bb)){ printf("  %s: alloc %.1f GB FAILED (skip)\n",ti.name,ti.gb); continue; }
        cudaMemset(b2,0x5a,bb);
        u64 slots=bb/32ULL, *sk; CK(cudaMalloc(&sk,64*sizeof(u64)));
        int bk=256; u64 gg=(nn+bk-1)/bk;
        chase<2><<<(unsigned)gg,bk>>>(b2,slots,K,0,nn,sk); CK(cudaDeviceSynchronize());
        cudaEvent_t a,c; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&c));
        CK(cudaEventRecord(a)); chase<2><<<(unsigned)gg,bk>>>(b2,slots,K,0,nn,sk); CK(cudaEventRecord(c)); CK(cudaEventSynchronize(c));
        float t=0; CK(cudaEventElapsedTime(&t,a,c));
        double mhs=(double)nn/(t/1e3)/1e6;
        printf("  %s (%.1f GB, N=%.0fM): %.1f MH/s\n",ti.name,ti.gb,slots/1e6,mhs);
        cudaFree(b2); cudaFree(sk);
    }
    return 0;
}
