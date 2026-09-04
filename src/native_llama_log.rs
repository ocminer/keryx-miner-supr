//! Terminal-safe bridge for the in-process llama.cpp sidecars.
//!
//! llama.cpp and ggml own a process-local global logger inside each sidecar.  Their default
//! callback writes directly to stderr, bypassing Rust's switchable logger and corrupting an active
//! alternate-screen dashboard.  The pinned llama API exposes `llama_log_get`/`llama_log_set`, so
//! install one callback before the first backend/model call and leave it installed for the sidecar
//! lifetime.  The callback reads the host's atomic TUI state: while the dashboard is active it is a
//! no-op; in classic mode it delegates to the exact callback/user-data pair that was installed
//! before us.  No file-descriptor redirection, environment mutation, allocation, or lock is used on
//! the native logging path.

use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

pub(crate) type LogCallback = unsafe extern "C" fn(c_int, *const c_char, *mut c_void);
pub(crate) type LogGet = unsafe extern "C" fn(*mut Option<LogCallback>, *mut *mut c_void);
pub(crate) type LogSet = unsafe extern "C" fn(Option<LogCallback>, *mut c_void);

struct Bridge {
    downstream: LogCallback,
    // Store the opaque C pointer as an address so the immutable bridge remains `Sync`. The native
    // logger owns the pointee and its lifetime is at least that of the never-unloaded sidecar.
    downstream_user_data: usize,
}

static BRIDGE: OnceLock<Bridge> = OnceLock::new();
static INSTALL_RESULT: OnceLock<bool> = OnceLock::new();

unsafe extern "C" fn dispatch(level: c_int, text: *const c_char, user_data: *mut c_void) {
    // This callback crosses a C ABI boundary and therefore deliberately contains only atomic
    // reads, pointer loads, and a call back into the native logger. None of those operations panic.
    if crate::tui_active() || user_data.is_null() {
        return;
    }
    let bridge = &*(user_data as *const Bridge);
    (bridge.downstream)(level, text, bridge.downstream_user_data as *mut c_void);
}

/// Install the bridge into one loaded llama.cpp sidecar.
///
/// Must run before any backend/model operation can create native logging threads. Repeated calls
/// are harmless; initialization is serialized and the native global is written exactly once. CUDA
/// and OpenCL inference engines are mutually exclusive build routes, so one process has at most one
/// in-process llama sidecar.
pub(crate) unsafe fn install(get: LogGet, set: LogSet) -> bool {
    *INSTALL_RESULT.get_or_init(|| {
        let mut downstream = None;
        let mut downstream_user_data = std::ptr::null_mut();
        get(&mut downstream, &mut downstream_user_data);
        let Some(downstream) = downstream else {
            return false;
        };

        // `get_or_init` serializes concurrent per-GPU loaders. Publish the immutable callback
        // state first, then write llama/ggml's unsynchronized global logger exactly once, before
        // either loader is allowed to enter backend initialization.
        if BRIDGE.set(Bridge { downstream, downstream_user_data: downstream_user_data as usize }).is_err() {
            return false;
        }
        let Some(bridge) = BRIDGE.get() else {
            return false;
        };
        set(Some(dispatch), bridge as *const Bridge as *mut c_void);
        true
    })
}

#[cfg(test)]
mod tests {
    use super::{install, LogCallback};
    use std::ffi::{c_char, c_int, c_void};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    static FORWARDED: AtomicUsize = AtomicUsize::new(0);
    static FORWARDED_USER_DATA: AtomicUsize = AtomicUsize::new(0);
    static INSTALLED: Mutex<Option<(LogCallback, usize)>> = Mutex::new(None);

    unsafe extern "C" fn downstream(_level: c_int, _text: *const c_char, user_data: *mut c_void) {
        FORWARDED_USER_DATA.store(user_data as usize, Ordering::Release);
        FORWARDED.fetch_add(1, Ordering::AcqRel);
    }

    unsafe extern "C" fn get(callback: *mut Option<LogCallback>, user_data: *mut *mut c_void) {
        *callback = Some(downstream);
        *user_data = 0x1234usize as *mut c_void;
    }

    unsafe extern "C" fn set(callback: Option<LogCallback>, user_data: *mut c_void) {
        *INSTALLED.lock().expect("test callback lock") = callback.map(|callback| (callback, user_data as usize));
    }

    #[test]
    fn bridge_preserves_classic_output_and_suppresses_dashboard_output() {
        crate::set_tui_active(false);
        assert!(unsafe { install(get, set) });
        let (callback, user_data) = INSTALLED.lock().expect("test callback lock").expect("bridge callback installed");

        unsafe { callback(1, std::ptr::null(), user_data as *mut c_void) };
        assert_eq!(FORWARDED.load(Ordering::Acquire), 1);
        assert_eq!(FORWARDED_USER_DATA.load(Ordering::Acquire), 0x1234);

        crate::set_tui_active(true);
        unsafe { callback(1, std::ptr::null(), user_data as *mut c_void) };
        assert_eq!(FORWARDED.load(Ordering::Acquire), 1);
        crate::set_tui_active(false);
    }
}
