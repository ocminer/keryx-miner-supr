//! Environment variables that the bundled native engines (llama.cpp / ggml inside
//! `libkeryx-llama.so` / `keryx-llama.dll`) read with the C runtime's `getenv`.
//!
//! Windows trap (0.15.4 field bug): `std::env::set_var` calls `SetEnvironmentVariableW`, which
//! changes the OS process environment block only. The Universal CRT keeps its OWN copy of the
//! environment (`_environ`), snapshotted when the executable's CRT started, and its `getenv`
//! never looks at the OS block again. Every DLL linked against the shared UCRT (our llama.cpp
//! engine is) therefore never saw `GGML_CUDA_DISABLE_GRAPHS` / `LLAMA_ATTN_ROT_DISABLE` set by
//! `main`: CUDA graphs stayed on and quantized KV rows stayed rotated, so the H15 traced
//! self-check failed on every Windows card, no model became serveable and the rig neither
//! declared nor mined. `_putenv_s` updates the UCRT table AND the OS block (one UCRT instance
//! per process, shared by the exe and every /MD DLL). On unix `setenv` is what `set_var` does.

/// The variables the traced engine needs before llama.cpp / ggml initialize.
pub const TRACE_ENV: [(&str, &str); 2] = [("GGML_CUDA_DISABLE_GRAPHS", "1"), ("LLAMA_ATTN_ROT_DISABLE", "1")];

#[cfg(windows)]
extern "C" {
    fn _putenv_s(name: *const std::os::raw::c_char, value: *const std::os::raw::c_char) -> i32;
}

extern "C" {
    fn getenv(name: *const std::os::raw::c_char) -> *const std::os::raw::c_char;
}

/// Sets `key=value` so that BOTH Rust (`std::env`) and native code using the C runtime's
/// `getenv` see it. Call only during single-threaded startup.
pub fn set(key: &str, value: &str) {
    std::env::set_var(key, value);
    #[cfg(windows)]
    {
        if let (Ok(k), Ok(v)) = (std::ffi::CString::new(key), std::ffi::CString::new(value)) {
            // SAFETY: valid NUL-terminated strings; startup, before any engine thread exists.
            let rc = unsafe { _putenv_s(k.as_ptr(), v.as_ptr()) };
            if rc != 0 {
                log::error!("native env: _putenv_s({key}) failed ({rc})");
            }
        }
    }
}

/// What the C runtime's `getenv` (the one native engines use) returns for `key`.
pub fn crt_get(key: &str) -> Option<String> {
    let k = std::ffi::CString::new(key).ok()?;
    // SAFETY: getenv returns NULL or a NUL-terminated string owned by the CRT; copied at once.
    let p = unsafe { getenv(k.as_ptr()) };
    if p.is_null() {
        return None;
    }
    Some(unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// Exports [`TRACE_ENV`] for the native engine (an explicit `GGML_CUDA_DISABLE_GRAPHS` from the
/// operator still wins; any value disables graphs in ggml). `KERYX_DIAG_NO_TRACE_ENV=1` skips
/// the export — a diagnostic that reproduces the pre-0.15.5 Windows behaviour on any platform.
pub fn export_trace_env() {
    if std::env::var_os("KERYX_DIAG_NO_TRACE_ENV").is_some_and(|v| v == "1") {
        return;
    }
    for (k, v) in TRACE_ENV {
        if k == "GGML_CUDA_DISABLE_GRAPHS" {
            if let Some(cur) = std::env::var_os(k) {
                // Already in the OS block: make sure the CRT table carries it too (Windows).
                set(k, &cur.to_string_lossy());
                continue;
            }
        }
        set(k, v);
    }
}

/// Variables of [`TRACE_ENV`] that the C runtime does NOT see (empty = native engine is set up).
pub fn trace_env_missing() -> Vec<&'static str> {
    TRACE_ENV.iter().filter(|(k, _)| crt_get(k).is_none()).map(|(k, _)| *k).collect()
}

/// `KERYX_NATIVE_ENV_CHECK=1` (CI gate): reports whether the C runtime sees a variable set via
/// `std::env::set_var` alone (the old path) and via [`set`] (the fixed path), plus the trace
/// variables. Exit code 0 only when the fixed path and both trace variables are visible.
pub fn ci_check() -> i32 {
    std::env::set_var("KERYX_ENV_PROBE_STD_ONLY", "1");
    let std_only = crt_get("KERYX_ENV_PROBE_STD_ONLY").is_some();
    set("KERYX_ENV_PROBE_FIXED", "1");
    let fixed = crt_get("KERYX_ENV_PROBE_FIXED").is_some();
    let missing = trace_env_missing();
    println!(
        "native-env-check: std::env::set_var visible to CRT getenv: {std_only}; native_env::set visible: {fixed}; trace vars missing from CRT: {missing:?}"
    );
    if fixed && missing.is_empty() {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn set_is_visible_to_crt_getenv() {
        super::set("KERYX_NATIVE_ENV_TEST", "abc");
        assert_eq!(super::crt_get("KERYX_NATIVE_ENV_TEST").as_deref(), Some("abc"));
        assert_eq!(std::env::var("KERYX_NATIVE_ENV_TEST").as_deref(), Ok("abc"));
    }
}
