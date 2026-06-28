use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    // PATCH (suprnova keryx): Builder::default() auto-globs src/**/*.cu, which
    // pulls in src/moe/moe_wmma*.cu. Those use bf16 tensor-core WMMA fragments that
    // only compile for sm_80+, so emitting their PTX at our low CUDA_COMPUTE_CAP=70
    // (chosen so the inference PTX forward-JITs across the whole fleet sm_70→sm_120)
    // fails with "incomplete type nvcuda::wmma::fragment<...nv_bfloat16...>". The MOE
    // PTX is unused anyway — it is stripped from ptx.rs by remove_lines below, and
    // keryx rejects MoE models (dense llama-arch only). So exclude src/moe/* from the
    // PTX build; the real inference kernels compile cleanly at sm_70. (The MOE FFI
    // *lib* is still built separately further down, pinned to sm_80.)
    let ptx_kernels: Vec<PathBuf> = glob::glob("src/**/*.cu")
        .expect("valid glob")
        .filter_map(|p| p.ok())
        .filter(|p| !p.components().any(|c| c.as_os_str() == "moe"))
        .collect();
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(ptx_kernels)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs (now also never generated)
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    // PATCH (suprnova keryx): the MOE WMMA FFI lib uses bf16 tensor-core fragments
    // that only compile for sm_80+. The OPoI inference PTX above (build_ptx) is
    // built at CUDA_COMPUTE_CAP (we ship 70 so the `.target sm_70` PTX forward-JITs
    // across the whole fleet sm_70→sm_120). bindgen_cuda::Builder::build_lib uses a
    // SINGLE compute_cap for both, so building the MOE lib at sm_70/75 fails with
    // "incomplete type nvcuda::wmma::fragment<...nv_bfloat16...>". Force the MOE lib
    // to at least sm_80 here. This is SAFE: keryx rejects MoE models (dense
    // llama-arch only — see quantized_llama_split.rs), so these kernels are
    // dead-linked SASS that is never launched; their arch never affects runtime
    // JIT on older cards. Without this, the lowest arch that compiles the whole
    // crate is sm_80 — which would re-break Volta/Turing GPU inference.
    let moe_compute_cap: usize = std::env::var("CUDA_COMPUTE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80)
        .max(80);
    let mut moe_builder = bindgen_cuda::Builder::default()
        .compute_cap(moe_compute_cap)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}
