use std::env;
use time::{format_description, OffsetDateTime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = format_description::parse("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=src/keccakf1600_x86-64.s");
    tonic_build::configure()
        .build_server(false)
        // .type_attribute(".", "#[derive(Debug)]")
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_arch == "x86_64" && target_os != "windows" && target_os != "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64.s").compile("libkeccak.a");
    }
    if target_arch == "x86_64" && target_os == "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64-osx.s").compile("libkeccak.a");
    }

    // PoM CUDA: compile the gather kernel to PTX (JIT'd by the driver at runtime, so a virtual
    // compute_75 arch runs on every Turing+ card — RTX 20xx/30xx/40xx/50xx, A100, H100). The
    // candle-CUDA pom_gpu driver does include_str!(OUT_DIR/pom_mine.ptx). Use a CUDA 12.x nvcc
    // (PATH/CUDA_PATH) to match candle 0.8's cudarc — NOT the CUDA 13 toolkit.
    if env::var("CARGO_FEATURE_POM_CUDA").is_ok() {
        let out = env::var("OUT_DIR").unwrap();
        let ptx = format!("{}/pom_mine.ptx", out);
        println!("cargo:rerun-if-changed=src/pom_mine.cu");
        let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
        let arch = env::var("POM_CUDA_ARCH").unwrap_or_else(|_| "compute_75".to_string());
        let status = std::process::Command::new(&nvcc)
            .args(["-ptx", &format!("-arch={}", arch), "-o", &ptx, "src/pom_mine.cu"])
            .status()
            .expect("pom-cuda: failed to run nvcc (CUDA toolkit required)");
        if !status.success() {
            panic!("pom-cuda: nvcc -ptx src/pom_mine.cu failed");
        }
    }
    Ok(())
}
