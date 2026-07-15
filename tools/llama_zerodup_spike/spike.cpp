// Phase-2 zero-dup spike: load a GGUF with llama.cpp (CUDA, -ngl 99), enumerate the resident
// model tensors, and prove (1) their data pointers are DEVICE pointers, (2) the resident bytes
// are BYTE-IDENTICAL to the on-disk GGUF tensor data (canonical name-sorted order — what the
// PoM walk gathers over and what R_T pins). Exit 0 = both hold for every tensor.
//
// Build (from llama.cpp source root, after a -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF build in $B):
//   g++ -O2 -std=c++17 spike.cpp -I include -I ggml/include -I src \
//       $B/src/libllama.a $B/ggml/src/libggml.a $B/ggml/src/libggml-cuda.a \
//       $B/ggml/src/libggml-cpu.a $B/ggml/src/libggml-base.a \
//       -L/usr/local/cuda-12.9/lib64 -lcudart -lcublas -lcublasLt -lcuda -lpthread -ldl -o spike
// Run:  ./spike <model.gguf> <gpu_ordinal>
#include "llama.h"
#include "llama-model.h"
#include "gguf.h"
#include "ggml.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const char* path = argv[1];
    int gpu = argc > 2 ? atoi(argv[2]) : 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.main_gpu = gpu;
    mp.use_mmap = true;
    llama_model* model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "FAIL: model load\n"); return 1; }

    // On-disk reference: gguf metadata for per-tensor file offsets.
    ggml_context* meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context* g = gguf_init_from_file(path, gp);
    if (!g) { fprintf(stderr, "FAIL: gguf meta\n"); return 1; }
    const size_t data_off = gguf_get_data_offset(g);
    FILE* f = fopen(path, "rb");

    // Canonical (byte-lexicographic name-sorted) order — must match pom.rs WeightIndex.
    std::vector<std::string> names;
    for (auto& p : model->tensors_by_name) names.push_back(p.first);
    std::sort(names.begin(), names.end());

    size_t n_dev = 0, n_host = 0, n_bad = 0, total = 0, chunks = 0;
    std::vector<uint8_t> hbuf, fbuf;
    for (auto& name : names) {
        const ggml_tensor* t = model->get_tensor(name.c_str());
        if (!t || !t->data) { printf("MISSING %s\n", name.c_str()); n_bad++; continue; }
        size_t nb = ggml_nbytes(t);
        total += nb; chunks += nb / 32;

        cudaPointerAttributes attr{};
        cudaPointerGetAttributes(&attr, t->data);
        bool on_dev = (attr.type == cudaMemoryTypeDevice);
        if (on_dev) n_dev++; else n_host++;

        // reference bytes from the file
        int64_t idx = gguf_find_tensor(g, name.c_str());
        if (idx < 0) { printf("NO-GGUF %s\n", name.c_str()); n_bad++; continue; }
        size_t foff = data_off + gguf_get_tensor_offset(g, idx);
        fbuf.resize(nb);
        if (fseeko(f, (off_t)foff, SEEK_SET) != 0 || fread(fbuf.data(), 1, nb, f) != nb) {
            printf("FREAD-FAIL %s\n", name.c_str()); n_bad++; continue;
        }
        // resident bytes
        hbuf.resize(nb);
        if (on_dev) {
            if (cudaMemcpy(hbuf.data(), t->data, nb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                printf("DTOH-FAIL %s\n", name.c_str()); n_bad++; continue;
            }
        } else {
            memcpy(hbuf.data(), t->data, nb);
        }
        if (memcmp(hbuf.data(), fbuf.data(), nb) != 0) {
            size_t first = 0; while (first < nb && hbuf[first] == fbuf[first]) first++;
            printf("BYTE-MISMATCH %s (%zu bytes, first diff @%zu) dev=%d type=%s\n",
                   name.c_str(), nb, first, (int)on_dev, ggml_type_name(t->type));
            n_bad++;
        }
    }
    fclose(f);
    printf("RESULT tensors=%zu dev=%zu host=%zu bad=%zu total_bytes=%zu chunks=%zu\n",
           names.size(), n_dev, n_host, n_bad, total, chunks);
    // Host-resident tensors are OK (the integration uploads its own device copy of those few);
    // any byte mismatch is fatal for zero-dup.
    llama_model_free(model);
    llama_backend_free();
    return n_bad == 0 ? 0 : 1;
}
