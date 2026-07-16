// AMD zero-dup spike: load a GGUF through libkeryx-llama-vk.so and prove the wrapper's gather
// path (per-tensor prefix table + buffer device addresses + the fetch shader — the EXACT path
// pom_walk_vk.comp reads through) returns bytes IDENTICAL to the on-disk GGUF for EVERY canonical
// chunk (name-sorted tensor order, floor(nbytes/32) chunks — what pom.rs WeightIndex indexes and
// R_T pins). Exit 0 = full-model byte identity + chunk-count identity hold.
//
// Built by build-keryx-llama-vk.sh with KERYX_SPIKE=1; links the .so's own exported gguf_*/ggml
// symbols for the on-disk reference, so there is exactly one llama/ggml in the process.
// Run:  LD_LIBRARY_PATH=<dir> ./spike_vk <model.gguf> [gpu]
#include "gguf.h"
#include "ggml.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>

extern "C" {
struct KeryxLlama;
int keryx_llama_abi();
int keryx_llama_vk_abi();
KeryxLlama* keryx_llama_load(const char* gguf_path, int gpu, int n_ctx);
void keryx_llama_free(KeryxLlama* h);
bool keryx_llama_pom_ready(KeryxLlama* h);
uint64_t keryx_llama_pom_n_chunks(KeryxLlama* h);
uint64_t keryx_llama_pom_supl_bytes(KeryxLlama* h);
bool keryx_llama_pom_fetch_range(KeryxLlama* h, uint64_t first, uint32_t count, uint8_t* out);
}

static constexpr uint64_t CHUNK = 32;
static constexpr uint32_t WINDOW = 131072; // must match FETCH_WINDOW_CHUNKS in the wrapper

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const char* path = argv[1];
    int gpu = argc > 2 ? atoi(argv[2]) : 0;

    printf("abi=%d vk_abi=%d\n", keryx_llama_abi(), keryx_llama_vk_abi());
    KeryxLlama* h = keryx_llama_load(path, gpu, 512);
    if (!h) { fprintf(stderr, "FAIL: engine load\n"); return 1; }
    if (!keryx_llama_pom_ready(h)) { fprintf(stderr, "FAIL: walk not ready (no BDA?)\n"); return 1; }
    const uint64_t n_chunks = keryx_llama_pom_n_chunks(h);
    printf("engine: N=%" PRIu64 " chunks, supplement=%" PRIu64 " bytes\n", n_chunks, keryx_llama_pom_supl_bytes(h));

    // On-disk reference: gguf metadata for canonical order + per-tensor file offsets.
    ggml_context* meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "FAIL: gguf meta\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(g);
    std::vector<std::string> names;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) names.push_back(gguf_get_tensor_name(g, i));
    std::sort(names.begin(), names.end());

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FAIL: open gguf\n"); return 1; }

    uint64_t global = 0; // running canonical chunk index — must track the wrapper's table exactly
    size_t n_bad = 0;
    std::vector<uint8_t> fbuf, vbuf((size_t)WINDOW * CHUNK);
    for (auto& name : names) {
        ggml_tensor* t = ggml_get_tensor(meta, name.c_str());
        int64_t idx = gguf_find_tensor(g, name.c_str());
        if (!t || idx < 0) { printf("NO-META %s\n", name.c_str()); n_bad++; continue; }
        const uint64_t nbytes = ggml_nbytes(t);
        const uint64_t chunks = nbytes / CHUNK;
        if (chunks == 0) continue;
        const size_t foff = data_off + gguf_get_tensor_offset(g, idx);
        fbuf.resize(chunks * CHUNK);
        if (fseeko(f, (off_t)foff, SEEK_SET) != 0 || fread(fbuf.data(), 1, chunks * CHUNK, f) != chunks * CHUNK) {
            printf("FREAD-FAIL %s\n", name.c_str()); n_bad++; global += chunks; continue;
        }
        for (uint64_t done = 0; done < chunks; done += WINDOW) {
            const uint32_t n = (uint32_t)std::min<uint64_t>(WINDOW, chunks - done);
            if (!keryx_llama_pom_fetch_range(h, global + done, n, vbuf.data())) {
                printf("FETCH-FAIL %s @%" PRIu64 "\n", name.c_str(), global + done); n_bad++; break;
            }
            if (memcmp(vbuf.data(), fbuf.data() + done * CHUNK, (size_t)n * CHUNK) != 0) {
                size_t first = 0;
                while (first < (size_t)n * CHUNK && vbuf[first] == fbuf[done * CHUNK + first]) first++;
                printf("BYTE-MISMATCH %s (chunk %" PRIu64 ", byte %zu) type=%s\n",
                       name.c_str(), global + done + first / CHUNK, first, ggml_type_name(t->type));
                n_bad++;
                break;
            }
        }
        global += chunks;
    }
    fclose(f);

    if (global != n_chunks) {
        printf("CHUNK-COUNT MISMATCH: gguf canonical=%" PRIu64 " engine=%" PRIu64 "\n", global, n_chunks);
        n_bad++;
    }
    printf("RESULT tensors=%zu canonical_chunks=%" PRIu64 " bad=%zu\n", names.size(), global, n_bad);
    keryx_llama_free(h);
    return n_bad == 0 ? 0 : 1;
}
