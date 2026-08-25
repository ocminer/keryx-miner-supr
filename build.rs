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
        // The tensor-core walk's warps-per-block and pipeline depth are COMPILE-TIME constants in
        // the kernel (`V4_TC_WARPS` / `V4_TC_PIPE`), and the host launch config must agree with them
        // exactly: the kernel derives its nonce index as `blockIdx.x * V4_TC_WARPS + warp`, so a
        // block launched with a different warp count silently walks the wrong nonces (or reads past
        // its shared-memory slice). Publish what the kernel was actually compiled with so the host
        // cannot drift from it.
        for (name, var) in [("V4_TC_WARPS", "POM_V4_TC_WARPS"), ("V4_TC_PIPE", "POM_V4_TC_PIPE")] {
            let def = std::fs::read_to_string("src/pom_mine.cu")
                .ok()
                .and_then(|s| {
                    s.lines()
                        .find_map(|l| {
                            let l = l.trim();
                            l.strip_prefix(&format!("#define {}", name))
                                .and_then(|r| r.split_whitespace().next().map(|v| v.to_string()))
                        })
                })
                .unwrap_or_else(|| panic!("pom-cuda: could not read #define {} from src/pom_mine.cu", name));
            println!("cargo:rustc-env={}={}", var, def);
        }
        // Without these, switching walk image/arch silently reuses the previously built image.
        println!("cargo:rerun-if-env-changed=POM_WALK_IMAGE");
        println!("cargo:rerun-if-env-changed=POM_CUDA_ARCH");
        println!("cargo:rerun-if-env-changed=NVCC");
        println!("cargo:rerun-if-env-changed=POM_CUDA_ARCH");
        let arch_override = env::var("POM_CUDA_ARCH").ok();
        let committed_fatbin = "cuda/pom_mine.fatbin";
        // MODERN ships the committed native-SASS fatbin on EVERY OS since the v4 tensor-core
        // solver (pom_mine_v4_tc): mma.sync.m16n8k32.s8 needs real sm_80+ SASS, and the old
        // Linux default (compute_75 PTX, JIT'd) can only ever carry the sub-sm_80 STUB of the
        // tc kernel — Linux rigs would silently lose the +35% tc path. The historical reason
        // Linux kept PTX (its compute_75 JIT of the dp4a walk was ~9% faster than native SASS)
        // is outweighed by the tc gain. Override with POM_WALK_IMAGE=fatbin|ptx. Legacy/pascal
        // (POM_CUDA_ARCH set) always use PTX (old cards; the fatbin has no sm_60/70, and their
        // cc < 8.0 dispatches the classic pom_mine_v4 anyway).
        let want_fatbin = match env::var("POM_WALK_IMAGE").ok().as_deref() {
            Some("fatbin") => true,
            Some("ptx") => false,
            _ => true,
        };

        if arch_override.is_none() && want_fatbin && std::path::Path::new(committed_fatbin).exists() {
            // MODERN: ship the native-SASS fatbin as-is (no nvcc needed at build time).
            std::fs::copy(committed_fatbin, &image)
                .unwrap_or_else(|e| panic!("pom-cuda: copy {committed_fatbin} -> {image}: {e}"));
            println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=fatbin");
            println!("cargo:rustc-env=POM_PTX_ARCH=sm_75..120-native"); // for the load-error message
        } else {
            // LEGACY/PASCAL or no committed fatbin: compile from source.
            //
            // Prefer a FATBIN carrying `arch` PTX (for the old cards this package exists for) PLUS
            // native SASS for the tensor-core-capable arches. A pure-PTX image compiled at
            // compute_70/75 contains only the sub-sm_80 STUB of pom_mine_v4_tc, so every Ampere+
            // card in a legacy fleet would be limited to the classic walk (or, before the gate fix,
            // silently mine NOTHING). Mixed rigs (e.g. pre-sm_75 CMPs + Ampere) must run this
            // package, so it has to serve both. Falls back to plain PTX if the toolkit is too old
            // to know these arches.
            let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
            let arch = arch_override.unwrap_or_else(|| "compute_75".to_string());
            let tc_sms = ["80", "86", "89", "90"];
            let mut args: Vec<String> = vec!["-fatbin".into(), format!("-gencode=arch={arch},code={arch}")];
            for sm in tc_sms {
                args.push(format!("-gencode=arch=compute_{sm},code=sm_{sm}"));
            }
            args.extend(["-o".to_string(), image.clone(), "src/pom_mine.cu".to_string()]);
            let fat = std::process::Command::new(&nvcc).args(&args).status();
            let fat_ok = matches!(fat, Ok(st) if st.success());
            if fat_ok {
                println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=fatbin");
                println!("cargo:rustc-env=POM_PTX_ARCH={}+sm_80..90-native", arch.replace("compute_", "sm_"));
            } else {
                println!("cargo:warning=pom-cuda: fatbin build failed — falling back to {arch} PTX; \
tensor-core walk will be unavailable (classic kernel only) on Ampere and newer.");
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
    }
    Ok(())
}
