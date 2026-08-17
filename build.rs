use std::env;
use time::{format_description, OffsetDateTime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build stamp for field diagnosis: every 0.9.5 asset repack printed the same bare "0.9.5"
    // banner, making it impossible to tell WHICH build a user's log came from. git hash + UTC time.
    let git = std::process::Command::new("git")
        .args(["rev-parse", "--short=9", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    println!("cargo:rustc-env=KERYX_BUILD_STAMP={git} @{secs}");
    println!("cargo:rerun-if-changed=.git/HEAD");

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

    // PoM CUDA walk image. TWO shapes, selected below:
    //  - MODERN (default): ship the prebuilt NATIVE-SASS fatbin `cuda/pom_mine.fatbin` (sm_75;80;86;
    //    89;90;120 + compute_75 PTX fallback). Native sm_120 SASS means the walk runs DIRECTLY on
    //    Blackwell (5070Ti/5080/5090) with NO driver JIT. The old path shipped only compute_75 PTX;
    //    on Windows the driver's JIT of that to sm_120 was ~5x slower than upstream's native fatbin
    //    ("limited on power"). The fatbin is prebuilt with CUDA 12.9 and committed, so even a build
    //    toolkit that predates sm_120 (Windows CI = CUDA 12.5) still ships native Blackwell SASS.
    //  - LEGACY/PASCAL (POM_CUDA_ARCH set, e.g. compute_70/compute_60): compile PTX from source with
    //    the build's nvcc, as before (those old cards aren't the Blackwell-JIT case). Also the dev
    //    fallback when the committed fatbin is absent.
    if env::var("CARGO_FEATURE_POM_CUDA").is_ok() {
        let out = env::var("OUT_DIR").unwrap();
        let image = format!("{}/pom_mine.image", out); // the shipped walk image (fatbin or ptx)
        println!("cargo:rerun-if-changed=src/pom_mine.cu");
        println!("cargo:rerun-if-changed=cuda/pom_mine.fatbin");
        println!("cargo:rerun-if-env-changed=POM_CUDA_ARCH");
        let arch_override = env::var("POM_CUDA_ARCH").ok();
        let committed_fatbin = "cuda/pom_mine.fatbin";
        // Ship the native-SASS fatbin ONLY where the driver's PTX→sm_120 JIT is bad: WINDOWS.
        // Measured on a 5080: Linux JITs compute_75 PTX to ~7.5 kh/s, but WINDOWS JITs the SAME PTX
        // to only ~1.2 kh/s ("limited on power"); native sm_120 SASS runs ~6.8 kh/s regardless of OS.
        // So Windows uses the fatbin (1.2 → ~6.8, matches/beats upstream) while Linux keeps the PTX
        // path (native SASS is ~9% SLOWER than Linux's good JIT — no fleet regression). Override with
        // POM_WALK_IMAGE=fatbin|ptx. Legacy/pascal (POM_CUDA_ARCH set) always use PTX (old cards, no
        // Blackwell-JIT issue, and the fatbin has no sm_60/70).
        let want_fatbin = match env::var("POM_WALK_IMAGE").ok().as_deref() {
            Some("fatbin") => true,
            Some("ptx") => false,
            _ => env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows"),
        };

        if arch_override.is_none() && want_fatbin && std::path::Path::new(committed_fatbin).exists() {
            // MODERN: ship the native-SASS fatbin as-is (no nvcc needed at build time).
            std::fs::copy(committed_fatbin, &image)
                .unwrap_or_else(|e| panic!("pom-cuda: copy {committed_fatbin} -> {image}: {e}"));
            println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=fatbin");
            println!("cargo:rustc-env=POM_PTX_ARCH=sm_75..120-native"); // for the load-error message
        } else {
            // LEGACY/PASCAL or no committed fatbin: compile PTX from source, JIT'd at runtime.
            let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
            let arch = arch_override.unwrap_or_else(|| "compute_75".to_string());
            println!("cargo:rustc-env=POM_PTX_ARCH={}", arch.replace("compute_", "sm_"));
            let status = std::process::Command::new(&nvcc)
                .args(["-ptx", &format!("-arch={}", arch), "-o", &image, "src/pom_mine.cu"])
                .status()
                .expect("pom-cuda: failed to run nvcc (CUDA toolkit required)");
            if !status.success() {
                panic!("pom-cuda: nvcc -ptx src/pom_mine.cu failed");
            }
            println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=ptx");
        }
    }
    Ok(())
}
