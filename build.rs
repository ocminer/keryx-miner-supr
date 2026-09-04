use sha2::{Digest, Sha256};
use std::env;
use time::{format_description, OffsetDateTime};

fn sha256_file(path: impl AsRef<std::path::Path>) -> Result<String, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn cuda_define(name: &str) -> String {
    std::fs::read_to_string("src/pom_mine.cu")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|l| {
                let l = l.trim();
                l.strip_prefix(&format!("#define {name}")).and_then(|r| r.split_whitespace().next().map(str::to_owned))
            })
        })
        .unwrap_or_else(|| panic!("pom-cuda: could not read #define {name} from src/pom_mine.cu"))
}

fn manifest_sha256(manifest: &str, field: &str) -> Result<String, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(manifest)?;
    let prefix = format!("- {field}: `");
    let value = text
        .lines()
        .find_map(|line| line.strip_prefix(&prefix).and_then(|rest| rest.strip_suffix('`')))
        .ok_or_else(|| format!("{manifest}: missing `{field}` entry"))?;
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{manifest}: `{field}` is not a SHA-256 digest").into());
    }
    Ok(value.to_ascii_lowercase())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build stamp for field diagnosis: every 0.9.5 asset repack printed the same bare "0.9.5"
    // banner, making it impossible to tell WHICH build a user's log came from. git hash + UTC time.
    let mut git = std::process::Command::new("git")
        .args(["rev-parse", "--short=9", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into());
    if git != "unknown"
        && std::process::Command::new("git")
            .args(["diff", "--quiet", "--ignore-submodules", "--"])
            .status()
            .map(|status| !status.success())
            .unwrap_or(false)
    {
        git.push_str("-dirty");
    }
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
    //  - MODERN (default): ship the prebuilt native-SASS fatbin `cuda/pom_mine.fatbin` (all CUDA
    //    13.3 targets from sm_75 through sm_121, plus compute_75 and tensor-capable compute_80 PTX).
    //    Native sm_120 SASS means the walk runs DIRECTLY on
    //    Blackwell (5070Ti/5080/5090) with NO driver JIT. The old path shipped only compute_75 PTX;
    //    on Windows the driver's JIT of that to sm_120 was ~5x slower than upstream's native fatbin
    //    ("limited on power"). The CUDA 13.3 fatbin is committed, so even a build toolkit that
    //    predates sm_120 can still package native Blackwell SASS; see the kernel audit document for
    //    its exact architecture list.
    //  - LEGACY/PASCAL (POM_CUDA_ARCH set, e.g. compute_70/compute_60): compile PTX from source with
    //    the build's nvcc, as before (those old cards aren't the Blackwell-JIT case). Also the dev
    //    fallback when the committed fatbin is absent.
    if env::var("CARGO_FEATURE_POM_CUDA").is_ok() {
        let out = env::var("OUT_DIR").unwrap();
        let image = format!("{}/pom_mine.image", out); // the shipped walk image (fatbin or ptx)
        println!("cargo:rerun-if-changed=src/pom_mine.cu");
        println!("cargo:rerun-if-changed=src/pom_gpu.rs");
        println!("cargo:rerun-if-changed=src/miner.rs");
        println!("cargo:rerun-if-changed=cuda/pom_mine.fatbin");
        println!("cargo:rerun-if-changed=cuda/POM_FATBIN_MANIFEST.md");
        // The tensor-core walk's warps-per-block and pipeline depth are COMPILE-TIME constants in
        // the kernel (`V4_TC_WARPS` / `V4_TC_PIPE`), and the host launch config must agree with them
        // exactly: the kernel derives its nonce index as `blockIdx.x * V4_TC_WARPS + warp`, so a
        // block launched with a different warp count silently walks the wrong nonces (or reads past
        // its shared-memory slice). Publish what the kernel was actually compiled with so the host
        // cannot drift from it.
        let tc_warps = cuda_define("V4_TC_WARPS");
        let tc_pipe = cuda_define("V4_TC_PIPE");
        let ncf_warps = cuda_define("V4_NCF_WARPS");
        for (var, def) in
            [("POM_V4_TC_WARPS", &tc_warps), ("POM_V4_TC_PIPE", &tc_pipe), ("POM_V4_NCF_WARPS", &ncf_warps)]
        {
            println!("cargo:rustc-env={}={}", var, def);
        }
        // Without these, switching walk image/arch silently reuses the previously built image.
        println!("cargo:rerun-if-env-changed=POM_WALK_IMAGE");
        println!("cargo:rerun-if-env-changed=POM_CUDA_ARCH");
        println!("cargo:rerun-if-env-changed=NVCC");
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
            // The image contains compile-time launch constants and cannot safely drift from its
            // source. Refuse a source-only or artifact-only edit until the protected regeneration
            // workflow has updated both hashes in the committed manifest.
            let manifest = "cuda/POM_FATBIN_MANIFEST.md";
            let source_sha = sha256_file("src/pom_mine.cu")?;
            let image_sha = sha256_file(committed_fatbin)?;
            let expected_source_sha = manifest_sha256(manifest, "Source SHA-256")?;
            let expected_image_sha = manifest_sha256(manifest, "Artifact SHA-256")?;
            if source_sha != expected_source_sha || image_sha != expected_image_sha {
                panic!(
                    "pom-cuda: committed walk image/source do not match {manifest}\n\
                     source expected {expected_source_sha}, got {source_sha}\n\
                     image  expected {expected_image_sha}, got {image_sha}\n\
                     Rebuild with cuda/regenerate-pom-fatbin.sh, pass the SASS/exactness gates, \
                     and update the manifest before compiling."
                );
            }
            std::fs::copy(committed_fatbin, &image)
                .unwrap_or_else(|e| panic!("pom-cuda: copy {committed_fatbin} -> {image}: {e}"));
            println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=fatbin");
            println!("cargo:rustc-env=POM_PTX_ARCH=sm_75..121-native+compute_80-tc");
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
            if !want_fatbin {
                // Honour the explicit override literally. Previously `POM_WALK_IMAGE=ptx` still
                // entered the mixed-fatbin branch and silently produced a fatbin.
                let status = std::process::Command::new(&nvcc)
                    .args(["-O3", "-ptx", &format!("-arch={}", arch), "-o", &image, "src/pom_mine.cu"])
                    .status()
                    .expect("pom-cuda: failed to run nvcc (CUDA toolkit required)");
                if !status.success() {
                    panic!("pom-cuda: nvcc -ptx src/pom_mine.cu failed");
                }
                println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=ptx");
                println!("cargo:rustc-env=POM_PTX_ARCH={}", arch.replace("compute_", "sm_"));
            } else {
                // Add every native target this nvcc actually knows. This gives source-fallback
                // builds the same broad coverage as the committed image without making an older
                // toolkit fail the entire fatbin on one unknown architecture.
                let candidates = ["80", "86", "87", "88", "89", "90", "100", "103", "110", "120", "121"];
                let listed = std::process::Command::new(&nvcc)
                    .arg("--list-gpu-code")
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
                let tc_sms: Vec<&str> = match listed {
                    Some(ref codes) => candidates
                        .iter()
                        .copied()
                        .filter(|sm| codes.split_whitespace().any(|code| code == format!("sm_{sm}")))
                        .collect(),
                    None => vec!["80", "86", "89", "90"],
                };
                let mut args: Vec<String> =
                    vec!["-O3".into(), "-fatbin".into(), format!("-gencode=arch={arch},code={arch}")];
                // A compute_75 fallback can only contain the empty tensor-core stubs because the
                // source guards IMMA at __CUDA_ARCH__ >= 800. Carry compute_80 PTX as well so an
                // unlisted/future Ampere-or-newer architecture JITs real seeded TC/NCF kernels.
                if arch != "compute_80" {
                    args.push("-gencode=arch=compute_80,code=compute_80".into());
                }
                for sm in &tc_sms {
                    args.push(format!("-gencode=arch=compute_{sm},code=sm_{sm}"));
                }
                args.extend(["-o".to_string(), image.clone(), "src/pom_mine.cu".to_string()]);
                let fat = std::process::Command::new(&nvcc).args(&args).status();
                let fat_ok = matches!(fat, Ok(st) if st.success());
                if fat_ok {
                    println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=fatbin");
                    println!(
                        "cargo:rustc-env=POM_PTX_ARCH={}+sm_{}-native+compute_80-tc",
                        arch.replace("compute_", "sm_"),
                        tc_sms.join(",")
                    );
                } else {
                    println!(
                        "cargo:warning=pom-cuda: fatbin build failed — falling back to {arch} PTX; \
tensor-core walk will be unavailable (classic kernel only) on Ampere and newer."
                    );
                    println!("cargo:rustc-env=POM_PTX_ARCH={}", arch.replace("compute_", "sm_"));
                    let status = std::process::Command::new(&nvcc)
                        .args(["-O3", "-ptx", &format!("-arch={}", arch), "-o", &image, "src/pom_mine.cu"])
                        .status()
                        .expect("pom-cuda: failed to run nvcc (CUDA toolkit required)");
                    if !status.success() {
                        panic!("pom-cuda: nvcc -ptx src/pom_mine.cu failed");
                    }
                    println!("cargo:rustc-env=POM_WALK_IMAGE_KIND=ptx");
                }
            }
        }

        // Bind the Rust host launch ABI and its persistent autotune decisions to the exact image
        // and source that produced this binary. Package versions are not sufficiently precise:
        // developers routinely rebuild several kernel revisions under one version number.
        let image_sha = sha256_file(&image)?;
        let source_sha = sha256_file("src/pom_mine.cu")?;
        let host_policy_sha = sha256_file("src/pom_gpu.rs")?;
        let abi = format!("v1;tc_warps={tc_warps};tc_pipe={tc_pipe};ncf_warps={ncf_warps}");
        println!("cargo:rustc-env=POM_WALK_IMAGE_SHA256={image_sha}");
        println!("cargo:rustc-env=POM_WALK_SOURCE_SHA256={source_sha}");
        println!("cargo:rustc-env=POM_HOST_POLICY_SHA256={host_policy_sha}");
        println!("cargo:rustc-env=POM_WALK_ABI={abi}");
    }
    Ok(())
}
