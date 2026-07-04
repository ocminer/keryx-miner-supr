// H100 PoM walk microbenchmark + byte-exact validation harness (branch h100-pom-opt).
//
// Replicates the EXACT walk arithmetic of src/pom_mine.cu (mix64 / seed_fold / gather / pow_fold)
// over a synthetic blob whose chunk bytes are a deterministic function of the chunk index, so a CPU
// reference can recompute any chunk and validate the GPU walk byte-exact WITHOUT storing 2.48 GB on
// the host. Measures nonces/s (the pool metric — each nonce = K gathers) and effective GB/s.
//
// Build:  nvcc -O3 -arch=sm_90 -o bench_pom bench_pom.cu
// Run:    ./bench_pom [variant] [n_nonces_log2] [K]
//   variant: 0=baseline(segmented,scalar)  1=contiguous(T=1)  2=contiguous+vec128  3=contig+vec+ILPn
//
// N is fixed to the real Gemma chunk count (77,604,776 -> 2.48 GB) so the memory footprint / L2
// miss behaviour matches production.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

typedef unsigned long long u64;

#define CK(x) do{ cudaError_t e=(x); if(e){ printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

static const u64 N_CHUNKS = 77604776ULL;   // real Gemma-3-4B chunk count
static const int DEF_K    = 256;           // POM_WALK_STEPS

// ---- exact arithmetic (must match src/pom_mine.cu) ----
__host__ __device__ __forceinline__ u64 mix64(u64 x){
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31; return x;
}
__host__ __device__ __forceinline__ u64 seed_fold(u64 nonce,u64 t,u64 p0,u64 p1,u64 p2,u64 p3){
    u64 s = mix64(nonce ^ 0x4B65727978531ULL);
    s = mix64(s ^ t); s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}
// deterministic synthetic chunk: chunk i's 4 u64 words. Fill kernel + CPU ref both use this.
__host__ __device__ __forceinline__ u64 chunk_word(u64 i, int j){
    return mix64(i * 0x9E3779B97F4A7C15ULL + (u64)j * 0xD1B54A32D192ED03ULL + 0xABCDEF12345ULL);
}

// fill the contiguous blob (4 u64 per chunk)
__global__ void fill_blob(u64* blob, u64 n){
    u64 i = (u64)blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=n) return;
    u64 b=i*4ULL;
    blob[b+0]=chunk_word(i,0); blob[b+1]=chunk_word(i,1);
    blob[b+2]=chunk_word(i,2); blob[b+3]=chunk_word(i,3);
}

// ---- CPU reference walk (recomputes chunk data on the fly) ----
u64 cpu_walk_final(u64 nonce,u64 t,const u64 p[4],u64 n,int K){
    u64 state = seed_fold(nonce,t,p[0],p[1],p[2],p[3]);
    u64 off = state % n;
    for(int i=0;i<K;i++){
        u64 h=state;
        h^=chunk_word(off,0); h^=chunk_word(off,1); h^=chunk_word(off,2); h^=chunk_word(off,3);
        state=mix64(h); off=state%n;
    }
    return state;
}

// ================= VARIANT 0: segmented + binary search (mirrors production src/pom_mine.cu) =====
__global__ void walk_v0(const u64* __restrict__ bases,const u64* __restrict__ prefix,unsigned T,
                        u64 n,int K,u64 p0,u64 p1,u64 p2,u64 p3,u64 t,
                        u64 base_nonce,u64 n_nonces,u64* dbg,u64 dbg_n){
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n_nonces) return;
    u64 nonce=base_nonce+tid;
    u64 state=seed_fold(nonce,t,p0,p1,p2,p3);
    u64 off=state%n;
    for(int i=0;i<K;i++){
        unsigned lo=0,hi=T;
        while(lo+1<hi){ unsigned mid=(lo+hi)>>1; if(prefix[mid]<=off) lo=mid; else hi=mid; }
        u64 local=off-prefix[lo];
        const u64* p=(const u64*)bases[lo];
        u64 b=local*4ULL;
        u64 h=state; h^=p[b]; h^=p[b+1]; h^=p[b+2]; h^=p[b+3];
        state=mix64(h); off=state%n;
    }
    if(tid<dbg_n) dbg[tid]=state;
}

// ================= VARIANT 1: contiguous T=1, scalar reads =================
__global__ void walk_v1(const u64* __restrict__ blob,u64 n,int K,
                        u64 p0,u64 p1,u64 p2,u64 p3,u64 t,
                        u64 base_nonce,u64 n_nonces,u64* dbg,u64 dbg_n){
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n_nonces) return;
    u64 nonce=base_nonce+tid;
    u64 state=seed_fold(nonce,t,p0,p1,p2,p3);
    u64 off=state%n;
    for(int i=0;i<K;i++){
        u64 b=off*4ULL;
        u64 h=state; h^=blob[b]; h^=blob[b+1]; h^=blob[b+2]; h^=blob[b+3];
        state=mix64(h); off=state%n;
    }
    if(tid<dbg_n) dbg[tid]=state;
}

// ================= VARIANT 2: contiguous + 128-bit vector loads =================
__global__ void walk_v2(const u64* __restrict__ blob,u64 n,int K,
                        u64 p0,u64 p1,u64 p2,u64 p3,u64 t,
                        u64 base_nonce,u64 n_nonces,u64* dbg,u64 dbg_n){
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n_nonces) return;
    u64 nonce=base_nonce+tid;
    u64 state=seed_fold(nonce,t,p0,p1,p2,p3);
    u64 off=state%n;
    const ulonglong2* __restrict__ blob2=(const ulonglong2*)blob;
    for(int i=0;i<K;i++){
        u64 b2=off*2ULL;              // 2 ulonglong2 per chunk
        ulonglong2 a=blob2[b2], c=blob2[b2+1];
        u64 h=state; h^=a.x; h^=a.y; h^=c.x; h^=c.y;
        state=mix64(h); off=state%n;
    }
    if(tid<dbg_n) dbg[tid]=state;
}

// ================= VARIANT 3: contiguous + vec + ILP (G independent walks/thread) =================
template<int G>
__global__ void walk_v3(const u64* __restrict__ blob,u64 n,int K,
                        u64 p0,u64 p1,u64 p2,u64 p3,u64 t,
                        u64 base_nonce,u64 n_nonces,u64* dbg,u64 dbg_n){
    u64 gid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    u64 base=base_nonce + gid*(u64)G;
    if(base>=base_nonce+n_nonces) return;
    const ulonglong2* __restrict__ blob2=(const ulonglong2*)blob;
    u64 state[G], off[G];
    #pragma unroll
    for(int g=0;g<G;g++){ state[g]=seed_fold(base+g,t,p0,p1,p2,p3); off[g]=state[g]%n; }
    for(int i=0;i<K;i++){
        ulonglong2 a[G], c[G];
        #pragma unroll
        for(int g=0;g<G;g++){ u64 b2=off[g]*2ULL; a[g]=blob2[b2]; c[g]=blob2[b2+1]; }  // G independent loads in flight
        #pragma unroll
        for(int g=0;g<G;g++){ u64 h=state[g]; h^=a[g].x; h^=a[g].y; h^=c[g].x; h^=c[g].y; state[g]=mix64(h); off[g]=state[g]%n; }
    }
    #pragma unroll
    for(int g=0;g<G;g++){ u64 id=gid*(u64)G+g; if(id<dbg_n) dbg[id]=state[g]; }
}

// ===== VARIANT 4: contiguous + vec + compile-time-constant N (fast div-by-constant, byte-exact) =====
template<unsigned long long NC>
__global__ void walk_v4(const u64* __restrict__ blob,int K,
                        u64 p0,u64 p1,u64 p2,u64 p3,u64 t,
                        u64 base_nonce,u64 n_nonces,u64* dbg,u64 dbg_n){
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n_nonces) return;
    u64 nonce=base_nonce+tid;
    u64 state=seed_fold(nonce,t,p0,p1,p2,p3);
    u64 off=state%NC;
    const ulonglong2* __restrict__ blob2=(const ulonglong2*)blob;
    for(int i=0;i<K;i++){
        u64 b2=off*2ULL;
        ulonglong2 a=blob2[b2], c=blob2[b2+1];
        u64 h=state; h^=a.x; h^=a.y; h^=c.x; h^=c.y;
        state=mix64(h); off=state%NC;      // NC compile-time -> mul/shift, not 64-bit division
    }
    if(tid<dbg_n) dbg[tid]=state;
}

// ===== VARIANT 5: mask instead of modulo (NON-EXACT speed probe — bounds the modulo cost) =====
__global__ void walk_v5(const u64* __restrict__ blob,int K,
                        u64 p0,u64 p1,u64 p2,u64 p3,u64 t,
                        u64 base_nonce,u64 n_nonces,u64* dbg,u64 dbg_n){
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n_nonces) return;
    u64 nonce=base_nonce+tid;
    u64 state=seed_fold(nonce,t,p0,p1,p2,p3);
    const u64 MASK=(1ULL<<26)-1;           // 67,108,863 < N -> always in bounds (NOT byte-exact)
    u64 off=state&MASK;
    const ulonglong2* __restrict__ blob2=(const ulonglong2*)blob;
    for(int i=0;i<K;i++){
        u64 b2=off*2ULL;
        ulonglong2 a=blob2[b2], c=blob2[b2+1];
        u64 h=state; h^=a.x; h^=a.y; h^=c.x; h^=c.y;
        state=mix64(h); off=state&MASK;
    }
    if(tid<dbg_n) dbg[tid]=state;
}

// ===== VARIANTS 6/7/8: contiguous+vec with cache-load hints (probe the 64B->32B fetch) =====
// 6 = __ldcs (cache-streaming, evict-first)   7 = __ldcg (cache-global, bypass L1)   8 = __ldlu (last-use)
template<int HINT>
__global__ void walk_hint(const u64* __restrict__ blob,u64 n,int K,
                        u64 p0,u64 p1,u64 p2,u64 p3,u64 t,
                        u64 base_nonce,u64 n_nonces,u64* dbg,u64 dbg_n){
    u64 tid=(u64)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=n_nonces) return;
    u64 nonce=base_nonce+tid;
    u64 state=seed_fold(nonce,t,p0,p1,p2,p3);
    u64 off=state%n;
    const ulonglong2* __restrict__ blob2=(const ulonglong2*)blob;
    for(int i=0;i<K;i++){
        u64 b2=off*2ULL;
        ulonglong2 a,c;
        if(HINT==6){ a=__ldcs(&blob2[b2]); c=__ldcs(&blob2[b2+1]); }
        else if(HINT==7){ a=__ldcg(&blob2[b2]); c=__ldcg(&blob2[b2+1]); }
        else { a=__ldlu(&blob2[b2]); c=__ldlu(&blob2[b2+1]); }
        u64 h=state; h^=a.x; h^=a.y; h^=c.x; h^=c.y;
        state=mix64(h); off=state%n;
    }
    if(tid<dbg_n) dbg[tid]=state;
}

int main(int argc,char**argv){
    int variant = argc>1?atoi(argv[1]):1;
    int lg      = argc>2?atoi(argv[2]):24;   // 2^24 = 16.7M nonces
    int K       = argc>3?atoi(argv[3]):DEF_K;
    int G       = argc>4?atoi(argv[4]):4;
    unsigned T  = argc>5?(unsigned)atoi(argv[5]):256;   // segments for v0 (mirror Gemma tensor count)
    u64 n_nonces = 1ULL<<lg;

    printf("N=%llu chunks (%.2f GB), K=%d, nonces=2^%d=%llu, variant=%d G=%d\n",
           N_CHUNKS,(double)N_CHUNKS*32.0/1e9,K,lg,n_nonces,variant,G);

    u64* blob; size_t blob_bytes=(size_t)N_CHUNKS*32ULL;
    CK(cudaMalloc(&blob,blob_bytes));
    { u64 fb=(N_CHUNKS+255)/256; fill_blob<<<(unsigned)fb,256>>>(blob,N_CHUNKS); CK(cudaDeviceSynchronize()); }

    // header params (arbitrary but fixed) + target
    u64 p[4]={0x1122334455667788ULL,0x99aabbccddeeff00ULL,0x0f1e2d3c4b5a6978ULL,0xdeadbeefcafef00dULL};
    u64 t=0x0000000012345678ULL;

    u64 dbg_n=64; u64* dbg; CK(cudaMalloc(&dbg,dbg_n*sizeof(u64)));
    CK(cudaMemset(dbg,0,dbg_n*sizeof(u64)));

    // segmented layout for v0: T equal slices of the SAME contiguous blob (byte-identical results)
    u64 *bases_d=nullptr,*prefix_d=nullptr;
    if(variant==0){
        u64* hbases=(u64*)malloc(T*sizeof(u64));
        u64* hprefix=(u64*)malloc((T+1)*sizeof(u64));
        hprefix[0]=0;
        for(unsigned s=0;s<T;s++){
            u64 lo=(N_CHUNKS*(u64)s)/T, hi=(N_CHUNKS*(u64)(s+1))/T;
            hbases[s]=(u64)(blob + lo*4ULL);   // device pointer to this slice
            hprefix[s+1]=hi;
        }
        CK(cudaMalloc(&bases_d,T*sizeof(u64))); CK(cudaMalloc(&prefix_d,(T+1)*sizeof(u64)));
        CK(cudaMemcpy(bases_d,hbases,T*sizeof(u64),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(prefix_d,hprefix,(T+1)*sizeof(u64),cudaMemcpyHostToDevice));
        free(hbases); free(hprefix);
    }

    int block=256;
    auto launch=[&](u64 nn){
        if(variant==0){ u64 g=(nn+block-1)/block; walk_v0<<<(unsigned)g,block>>>(bases_d,prefix_d,T,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
        else if(variant==1){ u64 g=(nn+block-1)/block; walk_v1<<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
        else if(variant==2){ u64 g=(nn+block-1)/block; walk_v2<<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
        else if(variant==3){ u64 threads=(nn+G-1)/G; u64 g=(threads+block-1)/block;
               if(G==2) walk_v3<2><<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n);
               else if(G==4) walk_v3<4><<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n);
               else if(G==8) walk_v3<8><<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n);
               else { printf("bad G\n"); exit(1);} }
        else if(variant==4){ u64 g=(nn+block-1)/block; walk_v4<77604776ULL><<<(unsigned)g,block>>>(blob,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
        else if(variant==5){ u64 g=(nn+block-1)/block; walk_v5<<<(unsigned)g,block>>>(blob,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
        else if(variant==6){ u64 g=(nn+block-1)/block; walk_hint<6><<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
        else if(variant==7){ u64 g=(nn+block-1)/block; walk_hint<7><<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
        else if(variant==8){ u64 g=(nn+block-1)/block; walk_hint<8><<<(unsigned)g,block>>>(blob,N_CHUNKS,K,p[0],p[1],p[2],p[3],t,0,nn,dbg,dbg_n); }
    };

    // ---- validate byte-exact vs CPU on the first dbg_n nonces ----
    launch(dbg_n); CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    u64 hdbg[64]; CK(cudaMemcpy(hdbg,dbg,dbg_n*sizeof(u64),cudaMemcpyDeviceToHost));
    if(variant==5){ printf("VALIDATION SKIPPED (v5 = non-exact mask speed probe)\n"); }
    else {
        int bad=0;
        for(u64 i=0;i<dbg_n;i++){ u64 ref=cpu_walk_final(i,t,p,N_CHUNKS,K); if(ref!=hdbg[i]){ if(bad<4) printf("  MISMATCH nonce %llu: gpu=%016llx cpu=%016llx\n",i,hdbg[i],ref); bad++; } }
        if(bad){ printf("VALIDATION FAILED (%d/%llu mismatch)\n",bad,dbg_n); return 2; }
        printf("VALIDATION OK (%llu/%llu byte-exact vs CPU reference)\n",dbg_n,dbg_n);
    }

    // ---- timed run ----
    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    launch(n_nonces); CK(cudaDeviceSynchronize()); // warm
    CK(cudaEventRecord(e0));
    launch(n_nonces);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1));
    double sec=ms/1e3;
    double mhs=(double)n_nonces/sec/1e6;
    double gaths=(double)n_nonces*(double)K;               // total gathers
    double gbs=gaths*32.0/sec/1e9;                          // effective bytes/s (32B/gather)
    printf("time=%.3f ms  |  %.2f MH/s (nonces)  |  %.1f Ggather/s  |  %.0f GB/s effective\n",
           ms,mhs,gaths/sec/1e9,gbs);
    return 0;
}
