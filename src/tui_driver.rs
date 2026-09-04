//! Lifecycle and logging boundary for the interactive dashboard.
//!
//! The renderer is deliberately isolated in `tui.rs`. This module decides whether a dashboard is
//! appropriate, diverts host log records into a bounded nonblocking ring while it owns the terminal,
//! and provides an idempotent restore hook for every hard-exit and panic path.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::ffi::OsString;
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Once, OnceLock, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::cursor::{Hide, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use log::{Level, LevelFilter, Log, Metadata, Record, SetLoggerError};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::tui::{
    self, BlockView, ConnectionView, DeviceActivity, DeviceView, EscrowView, InferenceState, InferenceView,
    MiningState, MiningView, ServiceBondView, ShareView, TuiAction, TuiState, UiEvent, UiEventKind, UiSnapshot,
};
use keryx_miner::runtime_stats as runtime;
use keryx_miner::PluginTuiLogControl;

// The UI retains the same visible horizon as structured runtime events. Keeping a larger hidden
// ring would only clone and sort invisible strings on every animation frame.
const MAX_LOG_LINES: usize = 256;

static DASHBOARD_ACTIVE: AtomicBool = AtomicBool::new(false);
static EXIT_RESTORE_REQUESTED: AtomicBool = AtomicBool::new(false);
// Raw mode and the alternate screen are separate transitions. Track both so an error or panic
// between them can undo the part of terminal setup that already succeeded.
static RAW_MODE_ACTIVE: AtomicBool = AtomicBool::new(false);
static TERMINAL_ACTIVE: AtomicBool = AtomicBool::new(false);
static LOGGER_INSTALLED: AtomicBool = AtomicBool::new(false);
static DIAGNOSTICS_REPLAYED: AtomicBool = AtomicBool::new(false);
static TERMINAL_IO: Mutex<()> = Mutex::new(());
static LOG_RING: OnceLock<Arc<Mutex<VecDeque<DashboardLog>>>> = OnceLock::new();
static LOG_START: OnceLock<Instant> = OnceLock::new();
static PLUGIN_LOG_CONTROLS: OnceLock<Vec<PluginTuiLogControl>> = OnceLock::new();

thread_local! {
    // The panic hook runs before stack unwinding releases a MutexGuard. Knowing whether this exact
    // thread owns TERMINAL_IO lets it use the unlocked restore path only for a true self-deadlock,
    // while a render-thread panic outside a draw remains serialized like every other panic.
    static CURRENT_THREAD_OWNS_TERMINAL_IO: Cell<bool> = Cell::new(false);
}

struct TerminalIoGuard {
    guard: Option<MutexGuard<'static, ()>>,
}

impl Drop for TerminalIoGuard {
    fn drop(&mut self) {
        // Unlock first, then clear the ownership marker. Neither operation can panic, and this keeps
        // the marker truthful for the entire interval in which another lock attempt would block.
        drop(self.guard.take());
        CURRENT_THREAD_OWNS_TERMINAL_IO.with(|owned| owned.set(false));
    }
}

fn lock_terminal_io() -> TerminalIoGuard {
    let guard = TERMINAL_IO.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    CURRENT_THREAD_OWNS_TERMINAL_IO.with(|owned| {
        debug_assert!(!owned.get(), "recursive TERMINAL_IO acquisition");
        owned.set(true);
    });
    TerminalIoGuard { guard: Some(guard) }
}

fn current_thread_owns_terminal_io() -> bool {
    CURRENT_THREAD_OWNS_TERMINAL_IO.with(Cell::get)
}

fn renderer_should_run(stop: &AtomicBool) -> bool {
    !stop.load(Ordering::Acquire)
        && !EXIT_RESTORE_REQUESTED.load(Ordering::Acquire)
        && dashboard_active()
}

#[derive(Clone, Debug)]
pub struct DashboardLog {
    pub uptime_ms: u64,
    pub level: Level,
    pub target: String,
    pub message: String,
}

fn log_ring() -> &'static Arc<Mutex<VecDeque<DashboardLog>>> {
    LOG_RING.get_or_init(|| Arc::new(Mutex::new(VecDeque::with_capacity(MAX_LOG_LINES))))
}

fn elapsed_ms() -> u64 {
    LOG_START.get_or_init(Instant::now).elapsed().as_millis().min(u64::MAX as u128) as u64
}

/// The pure selection rule is split out so it can be tested without a pseudo-terminal.
fn should_enable_with(
    args: &[OsString],
    stdin_is_terminal: bool,
    stdout_is_terminal: bool,
    stderr_is_terminal: bool,
    term: Option<&str>,
    force_off: bool,
) -> bool {
    let has = |needle: &str| args.iter().any(|arg| arg == needle);
    if force_off
        || has("--no-tui")
        || has("--recover-escrow")
        || has("--help")
        || has("-h")
        || has("--version")
        || has("-V")
    {
        return false;
    }
    if term.is_some_and(|value| value.eq_ignore_ascii_case("dumb")) {
        return false;
    }
    stdin_is_terminal && stdout_is_terminal && stderr_is_terminal
}

/// Decide before plugins are constructed whether they must refrain from installing an independent
/// stderr logger. Clap help/version and recovery remain classic even when launched from a TTY.
pub fn requested_from_process() -> bool {
    let args: Vec<OsString> = std::env::args_os().collect();
    let force_off = std::env::var("KERYX_TUI")
        .is_ok_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no"));
    should_enable_with(
        &args,
        io::stdin().is_terminal(),
        io::stdout().is_terminal(),
        io::stderr().is_terminal(),
        std::env::var("TERM").ok().as_deref(),
        force_off,
    )
}

/// Tell dynamically loaded worker plugins not to install their own stderr loggers. The host logger
/// is installed after clap has parsed the complete plugin-augmented command line.
pub fn mark_requested(requested: bool) {
    keryx_miner::set_tui_requested(requested);
    keryx_miner::set_tui_active(false);
}

/// Install the optional DSO-local logger controls discovered before plugin construction. The
/// pointers remain valid while PluginManager owns their libraries (its lifetime encloses the
/// dashboard guard in `main`).
pub fn configure_plugin_log_controls(controls: Vec<PluginTuiLogControl>) {
    let _ = PLUGIN_LOG_CONTROLS.set(controls);
    set_runtime_tui_active(false);
}

fn set_runtime_tui_active(active: bool) {
    keryx_miner::set_tui_active(active);
    if let Some(controls) = PLUGIN_LOG_CONTROLS.get() {
        for control in controls {
            // SAFETY: each pointer was resolved from a library retained by PluginManager, and its
            // ABI is the versioned `extern "C" fn(u8)` contract.
            unsafe { control(u8::from(active)) };
        }
    }
}

struct MinerLogger {
    inner: env_logger::Logger,
}

impl Log for MinerLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &Record<'_>) {
        if !self.inner.matches(record) {
            return;
        }
        if DASHBOARD_ACTIVE.load(Ordering::Acquire) {
            push_log(record.level(), record.target(), &record.args().to_string());
        } else {
            self.inner.log(record);
        }
    }

    fn flush(&self) {
        self.inner.flush();
    }
}

/// Install the switchable logger used while the TUI owns the terminal. Classic mode continues to
/// use the existing env_logger setup and therefore preserves its line format exactly.
pub fn install_dashboard_logger(level: LevelFilter) -> Result<(), SetLoggerError> {
    let mut builder = env_logger::Builder::new();
    builder.filter_level(level).parse_default_env();
    let inner = builder.build();
    let max_level = inner.filter();
    // Keep logging in classic mode until the alternate-screen thread is actually started. This
    // matters because plugin option processing and `Opt::process` can return actionable startup
    // errors before `start_dashboard`; buffering those messages here would make them disappear.
    DASHBOARD_ACTIVE.store(false, Ordering::Release);
    match log::set_boxed_logger(Box::new(MinerLogger { inner })) {
        Ok(()) => {
            log::set_max_level(max_level);
            LOGGER_INSTALLED.store(true, Ordering::Release);
            install_panic_restore_hook();
            Ok(())
        }
        Err(error) => {
            DASHBOARD_ACTIVE.store(false, Ordering::Release);
            LOGGER_INSTALLED.store(false, Ordering::Release);
            Err(error)
        }
    }
}

pub fn dashboard_active() -> bool {
    DASHBOARD_ACTIVE.load(Ordering::Acquire)
}

pub fn dashboard_logs() -> Vec<DashboardLog> {
    log_ring().try_lock().map(|logs| logs.iter().cloned().collect()).unwrap_or_default()
}

fn push_log(level: Level, target: &str, message: &str) {
    let Some(message) = sanitize_log_message(message) else {
        return;
    };
    let Ok(mut logs) = log_ring().try_lock() else {
        return;
    };
    if logs.len() == MAX_LOG_LINES {
        logs.pop_front();
    }
    logs.push_back(DashboardLog {
        uptime_ms: elapsed_ms(),
        level,
        target: target.rsplit("::").next().unwrap_or(target).chars().take(32).collect(),
        message,
    });
}

fn sanitize_log_message(raw: &str) -> Option<String> {
    let mut clean = String::with_capacity(raw.len().min(500));
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' {
            // Consume CSI, OSC and single-character escape sequences. `[` itself is in the ANSI
            // final-byte range, so it must be consumed before scanning a CSI terminator.
            match chars.next() {
                Some('[') => {
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
                Some(']') => {
                    while let Some(next) = chars.next() {
                        if next == '\u{7}' {
                            break;
                        }
                        if next == '\u{1b}' && chars.peek() == Some(&'\\') {
                            chars.next();
                            break;
                        }
                    }
                }
                Some(_) | None => {}
            }
            continue;
        }
        if ch == '\n' || ch == '\r' {
            if !clean.ends_with(' ') {
                clean.push(' ');
            }
        } else if !ch.is_control() {
            clean.push(ch);
        }
        if clean.chars().count() >= 500 {
            break;
        }
    }
    let trimmed = clean.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Avoid putting payout identities or private escrow setup instructions into a screenshot/log
    // pane. Detailed classic logs remain available via --no-tui for operator-only diagnosis.
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("mining for:") {
        return Some("Mining identity configured".to_string());
    }
    if lower.contains("delegate-escrow") {
        return Some(
            "Escrow delegation requires operator action; use --no-tui for the private setup command".to_string(),
        );
    }
    if lower.contains("escrow delegation cert") {
        return Some(if lower.contains("saved") || lower.contains("loaded") || lower.contains("taken from") {
            "Escrow delegation certificate configured".to_string()
        } else {
            "Escrow delegation certificate requires attention; use --no-tui for private details".to_string()
        });
    }
    if lower.contains("escrow keypair generated") {
        return Some("Escrow keypair generated and stored".to_string());
    }
    if lower.contains("mining to ") && lower.contains("cert") {
        return Some("Escrow payout identity requires attention; use --no-tui for private details".to_string());
    }
    Some(redact_log_tokens(trimmed))
}

fn redact_log_tokens(message: &str) -> String {
    let mut output = Vec::new();
    let mut redact_next = false;
    for token in message.split_whitespace() {
        if redact_next {
            output.push("[redacted]".to_string());
            redact_next = false;
            continue;
        }

        let core = token.trim_matches(|ch: char| {
            matches!(ch, '\'' | '"' | '`' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':' | '.')
        });
        let lower = core.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "--escrow-cert"
                | "--escrow-key-file"
                | "--escrow-cert-file"
                | "--escrow-state-file"
                | "--mining-address"
                | "--mining-privkey"
        ) {
            output.push(token.to_string());
            redact_next = true;
            continue;
        }

        let marker_value =
            ["pubkey=", "txid=", "transaction_id=", "request_hash=", "nonce=", "cid=", "cert=", "address=", "id="]
                .iter()
                .any(|marker| lower.contains(marker));
        let authenticated_url = lower
            .split_once("://")
            .is_some_and(|(_, rest)| rest.split('/').next().is_some_and(|authority| authority.contains('@')));
        let endpoint_or_model_url = lower.contains("://") || lower.contains("/ipfs/");
        let local_path = core.starts_with('/')
            || core.starts_with("./")
            || core.starts_with("../")
            || core.starts_with("~/")
            || lower.contains("/home/")
            || lower.contains("/users/")
            || (core.len() >= 3 && core.as_bytes()[1] == b':' && matches!(core.as_bytes()[2], b'\\' | b'/'));
        let payout_address = lower.starts_with("keryx:") || lower.contains("=keryx:");
        let likely_cid =
            (core.starts_with("Qm") && core.len() >= 40 && core.chars().all(|ch| ch.is_ascii_alphanumeric()))
                || (lower.starts_with("bafy") && core.len() >= 32 && core.chars().all(|ch| ch.is_ascii_alphanumeric()));
        let bare_long_hex = core.len() >= 32 && core.bytes().all(|byte| byte.is_ascii_hexdigit());
        let private_host = looks_like_private_host(core);

        if marker_value
            || authenticated_url
            || endpoint_or_model_url
            || local_path
            || payout_address
            || likely_cid
            || bare_long_hex
            || private_host
        {
            output.push("[redacted]".to_string());
        } else {
            output.push(token.to_string());
        }
    }
    output.join(" ")
}

fn looks_like_private_host(token: &str) -> bool {
    let host = token
        .split_once("://")
        .map_or(token, |(_, rest)| rest)
        .trim_matches(|ch: char| !ch.is_ascii_alphanumeric() && ch != '.' && ch != ':');
    let host = host.split(':').next().unwrap_or(host);
    let octets: Vec<u8> = host.split('.').filter_map(|part| part.parse().ok()).collect();
    if octets.len() != 4 {
        return false;
    }
    octets[0] == 10
        || octets[0] == 127
        || (octets[0] == 192 && octets[1] == 168)
        || (octets[0] == 172 && (16..=31).contains(&octets[1]))
}

pub(crate) fn mark_terminal_active(active: bool) {
    TERMINAL_ACTIVE.store(active, Ordering::Release);
}

fn replay_sanitized_diagnostics() {
    if DIAGNOSTICS_REPLAYED.swap(true, Ordering::AcqRel) {
        return;
    }
    let diagnostics: Vec<DashboardLog> = log_ring()
        .try_lock()
        .map(|logs| {
            let mut selected: Vec<_> = logs
                .iter()
                .rev()
                .filter(|entry| matches!(entry.level, Level::Warn | Level::Error))
                .take(12)
                .cloned()
                .collect();
            selected.reverse();
            selected
        })
        .unwrap_or_default();
    if diagnostics.is_empty() {
        return;
    }

    let stderr = io::stderr();
    let mut out = stderr.lock();
    let _ = writeln!(out, "Recent dashboard warnings/errors (sanitized; use --no-tui for full diagnostics):");
    for entry in diagnostics {
        let level = match entry.level {
            Level::Error => "ERROR",
            _ => "WARN",
        };
        let _ = writeln!(out, "  [{level}] {}", entry.message);
    }
    let _ = out.flush();
}

fn restore_terminal_unlocked() {
    let was_raw_mode_active = RAW_MODE_ACTIVE.swap(false, Ordering::AcqRel);
    let was_terminal_active = TERMINAL_ACTIVE.swap(false, Ordering::AcqRel);
    if was_raw_mode_active {
        let _ = disable_raw_mode();
    }
    if was_terminal_active {
        let mut out = io::stdout();
        let _ = execute!(out, LeaveAlternateScreen, Show);
        let _ = out.flush();
    }
    // Keep host logs buffered until stdout is back on the normal screen. Together with the
    // in-lock render recheck below, this prevents either a log line or a late frame from being
    // emitted between LeaveAlternateScreen and dashboard deactivation.
    let was_dashboard_active = DASHBOARD_ACTIVE.swap(false, Ordering::AcqRel);
    if was_raw_mode_active || was_terminal_active || was_dashboard_active {
        // The owning PluginManager is guaranteed alive while a dashboard is active.
        set_runtime_tui_active(false);
        replay_sanitized_diagnostics();
    } else {
        // Do not call retained DSO pointers after the dashboard guard (and potentially the plugin
        // manager) has already been dropped. The host-side atomic is always safe to clear.
        keryx_miner::set_tui_active(false);
    }
}

/// Safe to call repeatedly, from another thread, during a panic, or immediately before `_exit`.
/// Serializing terminal writes prevents a signal/wedge exit from leaving alternate-screen draw
/// bytes on the restored normal screen.
pub fn restore_terminal_best_effort() {
    let _terminal_io = lock_terminal_io();
    restore_terminal_unlocked();
}

#[cfg(unix)]
fn write_exit_restore_sequence() {
    use std::os::unix::fs::OpenOptionsExt;

    const RESTORE: &[u8] = b"\x1b[?1049l\x1b[?25h";
    // Open a separate nonblocking file description so a full/stalled TTY output queue cannot make
    // SIGTERM wait forever and so O_NONBLOCK never leaks onto the parent shell's inherited stdout.
    for path in ["/dev/tty", "/proc/self/fd/1", "/dev/fd/1"] {
        let Ok(mut tty) = std::fs::OpenOptions::new().write(true).custom_flags(nix::libc::O_NONBLOCK).open(path) else {
            continue;
        };
        if tty.write(RESTORE).is_ok() {
            break;
        }
    }
}

#[cfg(not(unix))]
fn write_exit_restore_sequence() {
    let mut out = io::stdout();
    let _ = execute!(out, LeaveAlternateScreen, Show);
    let _ = out.flush();
}

fn restore_terminal_for_exit_unlocked() {
    let was_raw_mode_active = RAW_MODE_ACTIVE.swap(false, Ordering::AcqRel);
    let was_terminal_active = TERMINAL_ACTIVE.swap(false, Ordering::AcqRel);
    if was_raw_mode_active {
        let _ = disable_raw_mode();
    }
    if was_terminal_active {
        write_exit_restore_sequence();
    }
    let was_dashboard_active = DASHBOARD_ACTIVE.swap(false, Ordering::AcqRel);
    if was_raw_mode_active || was_terminal_active || was_dashboard_active {
        set_runtime_tui_active(false);
    } else {
        keryx_miner::set_tui_active(false);
    }
}

/// Hard-exit restoration must not wait forever behind a blocked terminal write. Stop future frames,
/// give an in-flight draw a short opportunity to finish, then fall back to idempotent nonblocking
/// terminal recovery. The caller invokes `process::exit` immediately after this returns.
pub fn restore_terminal_for_exit() {
    const LOCK_GRACE: Duration = Duration::from_millis(250);

    EXIT_RESTORE_REQUESTED.store(true, Ordering::Release);
    let deadline = Instant::now() + LOCK_GRACE;
    loop {
        match TERMINAL_IO.try_lock() {
            Ok(_terminal_io) => {
                restore_terminal_for_exit_unlocked();
                return;
            }
            Err(TryLockError::Poisoned(poisoned)) => {
                let _terminal_io = poisoned.into_inner();
                restore_terminal_for_exit_unlocked();
                return;
            }
            Err(TryLockError::WouldBlock) if Instant::now() >= deadline => {
                // The process is about to terminate, so preventing an infinite supervisor STOP is
                // more important than waiting on a renderer that may never release its lock.
                restore_terminal_for_exit_unlocked();
                return;
            }
            Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(2)),
        }
    }
}

fn install_panic_restore_hook() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            // The standard panic hook writes straight to stderr. While the alternate screen is
            // active that text disappears when the renderer restores the operator's screen, so
            // retain a bounded, sanitized copy for the post-restore diagnostic tail.
            push_log(Level::Error, "panic", &format!("Thread panic: {info}"));
            if current_thread_owns_terminal_io() {
                // A panic hook runs before unwinding releases this thread's MutexGuard. Restore
                // without relocking to avoid self-deadlock; ownership of TERMINAL_IO still excludes
                // every other setup/draw/restore operation.
                restore_terminal_unlocked();
                previous(info);
            } else {
                // A worker panic must not let the default hook write unsanitized stderr into the
                // alternate screen, overlap a draw, or race the post-restore diagnostic replay.
                // Always participate in TERMINAL_IO, even if the state flags already read false:
                // another thread can have cleared them while it still owns the lock and is writing
                // the replay. If that lock is busy, deactivate the renderer and suppress this one
                // raw write; the bounded sanitized copy above is the terminal-safe diagnostic.
                match TERMINAL_IO.try_lock() {
                    Ok(_terminal_io) => {
                        restore_terminal_unlocked();
                        previous(info);
                    }
                    Err(TryLockError::Poisoned(poisoned)) => {
                        let _terminal_io = poisoned.into_inner();
                        restore_terminal_unlocked();
                        previous(info);
                    }
                    Err(TryLockError::WouldBlock) => {
                        DASHBOARD_ACTIVE.store(false, Ordering::Release);
                    }
                }
            }
        }));
    });
}

/// Owns the dedicated render thread. Dropping it is a graceful fallback: the current terminal is
/// restored first, then the bounded event loop is joined, and ordinary env_logger output resumes.
pub struct DashboardGuard {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for DashboardGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        restore_terminal_best_effort();
    }
}

/// Start the presentation thread. It reads only immutable runtime snapshots and never owns a miner,
/// client, GPU, inference route or escrow watcher.
pub fn start_dashboard() -> io::Result<DashboardGuard> {
    if !LOGGER_INSTALLED.load(Ordering::Acquire) {
        return Err(io::Error::new(io::ErrorKind::Other, "dashboard logger is not installed"));
    }
    if let Ok(mut logs) = log_ring().try_lock() {
        logs.clear();
    }
    DIAGNOSTICS_REPLAYED.store(false, Ordering::Release);
    EXIT_RESTORE_REQUESTED.store(false, Ordering::Release);
    set_runtime_tui_active(true);
    DASHBOARD_ACTIVE.store(true, Ordering::Release);
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let handle = match thread::Builder::new().name("keryx-tui".to_string()).spawn(move || {
        let result = std::panic::catch_unwind(|| render_loop(&thread_stop));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                restore_terminal_best_effort();
                log::warn!("Interactive dashboard stopped ({error}); continuing with classic logs.");
            }
            Err(_) => {
                // The panic hook has already restored the terminal. Never let presentation
                // failure bring down mining or inference.
                restore_terminal_best_effort();
                log::error!("Interactive dashboard panicked; continuing with classic logs.");
            }
        }
    }) {
        Ok(handle) => handle,
        Err(error) => {
            restore_terminal_best_effort();
            return Err(error);
        }
    };
    Ok(DashboardGuard { stop, handle: Some(handle) })
}

fn render_loop(stop: &AtomicBool) -> io::Result<()> {
    let setup_io = lock_terminal_io();
    // A signal/panic can deactivate the dashboard after start_dashboard() spawns us but before this
    // thread wins TERMINAL_IO. Never enter raw/alternate mode after another thread already restored.
    if !renderer_should_run(stop) {
        return Ok(());
    }

    // Set the conservative marker before calling into crossterm: if that call panics after changing
    // termios, the render-thread panic hook still knows that raw mode may need to be disabled.
    RAW_MODE_ACTIVE.store(true, Ordering::Release);
    if let Err(error) = enable_raw_mode() {
        let _ = disable_raw_mode();
        RAW_MODE_ACTIVE.store(false, Ordering::Release);
        return Err(error);
    }
    if !renderer_should_run(stop) {
        restore_terminal_unlocked();
        return Ok(());
    }
    let mut out = io::stdout();
    // Likewise, conservatively mark the alternate screen before the combined write. Sending a
    // redundant LeaveAlternateScreen on a failed Enter is harmless; omitting it after a partial
    // write would strand the operator's terminal.
    mark_terminal_active(true);
    if let Err(error) = execute!(out, EnterAlternateScreen, Hide) {
        restore_terminal_unlocked();
        return Err(error);
    }
    if !renderer_should_run(stop) {
        restore_terminal_unlocked();
        return Ok(());
    }

    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    drop(setup_io);
    let mut state = TuiState::from_environment();
    let mut last_snapshot = UiSnapshot::default();
    let mut clocks = ClockBaselines::default();
    let mut last_frame = Instant::now().checked_sub(Duration::from_secs(1)).unwrap_or_else(Instant::now);

    // A panic on any process thread invokes the global restore hook. Treat dashboard deactivation
    // as a stop signal too, so a recoverable/caught worker panic cannot leave this thread drawing
    // into the normal terminal after the hook has left the alternate screen.
    while renderer_should_run(stop) {
        if event::poll(Duration::from_millis(40))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Release
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char('c' | 'C'))
                {
                    restore_terminal_for_exit();
                    std::process::exit(0);
                }
                if tui::handle_key(&mut state, key) == TuiAction::QuitConfirmed {
                    restore_terminal_for_exit();
                    std::process::exit(0);
                }
            }
        }

        let frame_period = if state.motion_enabled { Duration::from_millis(125) } else { Duration::from_millis(500) };
        if last_frame.elapsed() < frame_period {
            continue;
        }
        state.tick_animation();
        if let Some(snapshot) = runtime::try_snapshot() {
            last_snapshot = adapt_snapshot(snapshot, &mut clocks);
        }
        let drawn = {
            let _terminal_io = lock_terminal_io();
            // The loop condition predates event::poll and snapshot work. Restoration may have won
            // TERMINAL_IO in that interval, so validate again while holding the same lock before
            // allowing any bytes onto the now-normal screen.
            if !renderer_should_run(stop) {
                None
            } else {
                Some(terminal.draw(|frame| tui::draw(frame, &state, &last_snapshot)))
            }
        };
        let Some(drawn) = drawn else {
            break;
        };
        drawn?;
        last_frame = Instant::now();
    }

    restore_terminal_best_effort();
    Ok(())
}

#[derive(Default)]
struct ClockBaselines {
    last_sample_sequence: Option<u64>,
    devices: HashMap<u32, DeviceClocks>,
}

#[derive(Default)]
struct DeviceClocks {
    active_samples: u8,
    core: ClockTrack,
    memory: ClockTrack,
}

#[derive(Default)]
struct ClockTrack {
    baseline: Option<u32>,
    low_streak: u8,
}

impl ClockTrack {
    fn observe(&mut self, current: Option<u32>, active: bool, warmed: bool, fresh_sample: bool) -> Option<u32> {
        let Some(current) = current.filter(|value| *value != 0) else {
            return None;
        };
        if !active {
            return self.baseline;
        }
        if fresh_sample {
            self.baseline = Some(self.baseline.map_or(current, |baseline| baseline.max(current)));
            let baseline = self.baseline?;
            if current.saturating_mul(100) < baseline.saturating_mul(90) {
                self.low_streak = self.low_streak.saturating_add(1);
            } else {
                self.low_streak = 0;
            }
        }
        let baseline = self.baseline?;
        // Suppress universal/one-sample clock alarms. Once warmed, a low clock is compared to the
        // learned card-specific baseline only after three distinct backend telemetry samples.
        if !warmed || (self.low_streak > 0 && self.low_streak < 3) {
            None
        } else {
            Some(baseline)
        }
    }
}

fn adapt_snapshot(snapshot: runtime::Snapshot, clocks: &mut ClockBaselines) -> UiSnapshot {
    let fresh_mining_sample = clocks.last_sample_sequence != Some(snapshot.mining.sample_sequence);
    if fresh_mining_sample {
        clocks.last_sample_sequence = Some(snapshot.mining.sample_sequence);
    }
    let mode = match snapshot.mode {
        runtime::MiningMode::Solo => tui::MiningMode::Solo,
        runtime::MiningMode::Pool | runtime::MiningMode::Unknown => tui::MiningMode::Pool,
    };
    let connection_state = match snapshot.connection {
        runtime::ConnectionState::Connecting => tui::ConnectionState::Connecting,
        runtime::ConnectionState::Connected => tui::ConnectionState::Connected,
        runtime::ConnectionState::Failover => tui::ConnectionState::Failover,
        runtime::ConnectionState::Offline => tui::ConnectionState::Offline,
    };
    let mining_state = if snapshot.inference.staging_error {
        MiningState::Degraded
    } else if snapshot.mining.preparing {
        MiningState::Preparing
    } else if snapshot.mining.inference_paused {
        MiningState::InferencePaused
    } else if matches!(snapshot.connection, runtime::ConnectionState::Offline) {
        MiningState::Stopped
    } else {
        MiningState::Mining
    };

    let inference_state = if snapshot.inference.staging_error {
        InferenceState::Degraded
    } else if snapshot.inference.active != 0 {
        InferenceState::Serving
    } else if snapshot.inference.serveable_models != 0 {
        InferenceState::Ready
    } else if snapshot.mining.preparing {
        InferenceState::Preparing
    } else {
        InferenceState::Unavailable
    };
    let inference_status = inference_status(inference_state, &snapshot.inference);

    // A startup self-test proves a route without becoming an external inference attempt, so its
    // GPU is carried by `route_gpus` rather than `gpu_index`. Only identify a READY host when the
    // route is unique; a genuinely multi-GPU route remains neutral instead of naming one card.
    let inference_gpu = inference_display_gpu(&snapshot.inference);
    let devices = snapshot
        .mining
        .devices
        .iter()
        .map(|device| {
            let inference_host = inference_gpu == Some(device.index);
            let activity = if snapshot.mining.preparing {
                DeviceActivity::Preparing
            } else if snapshot.mining.inference_paused && inference_host {
                DeviceActivity::Inference
            } else if snapshot.mining.inference_paused {
                DeviceActivity::Paused
            } else if matches!(snapshot.connection, runtime::ConnectionState::Offline) {
                DeviceActivity::Offline
            } else if device.hashrate_hs > 0.0 {
                DeviceActivity::Mining
            } else {
                DeviceActivity::Stalled
            };
            let active = activity == DeviceActivity::Mining && device.hashrate_hs > 0.0;
            let tracker = clocks.devices.entry(device.index).or_default();
            if active && fresh_mining_sample {
                tracker.active_samples = tracker.active_samples.saturating_add(1);
            }
            let warmed = tracker.active_samples >= 3;
            let baseline_core_mhz = tracker.core.observe(device.core_mhz, active, warmed, fresh_mining_sample);
            let baseline_mem_mhz = tracker.memory.observe(device.mem_mhz, active, warmed, fresh_mining_sample);
            DeviceView {
                index: device.index,
                name: clean_device_name(&device.label),
                backend: device.backend.clone(),
                hashrate_hs: device.hashrate_hs,
                temp_c: device.temp_c,
                hotspot_c: device.hotspot_c,
                fan_pct: device.fan_pct,
                power_w: device.power_w,
                core_mhz: device.core_mhz,
                mem_mhz: device.mem_mhz,
                baseline_core_mhz,
                baseline_mem_mhz,
                vram_used_mb: device.vram_used_mb,
                vram_total_mb: device.vram_total_mb,
                efficiency_mhs_per_w: device.efficiency_mhs_per_w,
                accepted: device.accepted,
                rejected: device.rejected,
                inference_host,
                activity,
                throttle_reason: None,
            }
        })
        .collect();

    let mut events: Vec<(u64, UiEvent)> = snapshot
        .events
        .iter()
        .map(|event| {
            (
                event.uptime_ms,
                UiEvent {
                    timestamp: format_uptime_clock(event.uptime_ms),
                    kind: map_event_kind(event.kind),
                    message: event.message.clone(),
                },
            )
        })
        .collect();
    events.extend(dashboard_logs().into_iter().map(|entry| {
        let kind = match entry.level {
            Level::Error => UiEventKind::Error,
            Level::Warn => UiEventKind::HealthWarn,
            _ => UiEventKind::Info,
        };
        let message =
            if entry.target.is_empty() { entry.message } else { format!("[{}] {}", entry.target, entry.message) };
        (entry.uptime_ms, UiEvent { timestamp: format_uptime_clock(entry.uptime_ms), kind, message })
    }));
    events.sort_by_key(|(at, _)| *at);
    let trim = events.len().saturating_sub(256);
    let events = events.into_iter().skip(trim).map(|(_, event)| event).collect();

    let escrow = adapt_escrow(&snapshot, mode);
    let service_bond = adapt_service_bond(&snapshot.service_bond);
    let era = snapshot.daa_score.map_or_else(String::new, |daa| {
        if keryx_miner::pom::is_h10_seed_era(daa) {
            "H10".to_string()
        } else {
            "pre-H10".to_string()
        }
    });

    UiSnapshot {
        miner_name: "KERYX // MINING CORE".to_string(),
        version: format!("v{}", env!("CARGO_PKG_VERSION")),
        build: env!("KERYX_BUILD_STAMP").to_string(),
        algorithm: "KeryxHash / PoM v4".to_string(),
        era,
        uptime_secs: snapshot.uptime_secs,
        connection: ConnectionView {
            mode,
            endpoint: snapshot.endpoint,
            state: connection_state,
            latency_ms: snapshot.connection_latency_ms,
            last_job_age_secs: snapshot.last_job_age_secs,
            difficulty: snapshot.difficulty,
            network_difficulty: snapshot.network_difficulty,
            daa_score: snapshot.daa_score,
            failover: if snapshot.failover_index == 0 {
                "primary".to_string()
            } else {
                format!("backup #{}", snapshot.failover_index)
            },
            synced: snapshot.synced,
            message: snapshot.connection_message,
        },
        mining: MiningView {
            state: mining_state,
            total_hashrate_hs: snapshot.mining.total_hashrate_hs,
            average_60s_hs: snapshot.mining.average_60s_hs,
            hashrate_history_hs: snapshot.mining.hashrate_history_hs,
            total_power_w: snapshot.mining.total_power_w,
            efficiency_mhs_per_w: snapshot.mining.efficiency_mhs_per_w,
        },
        shares: ShareView {
            accepted: snapshot.shares.accepted,
            rejected: snapshot.shares.rejected(),
            stale: snapshot.shares.stale,
            low_diff: snapshot.shares.low_diff,
            duplicate: snapshot.shares.duplicate,
            other: snapshot.shares.other,
            pending: snapshot.shares.pending,
            last_accepted_age_secs: snapshot.shares.last_accepted_age_secs,
        },
        blocks: BlockView {
            found: snapshot.blocks.found,
            accepted: (mode == tui::MiningMode::Solo).then_some(snapshot.blocks.accepted),
            rejected: (mode == tui::MiningMode::Solo).then_some(snapshot.blocks.rejected),
            pending: snapshot.blocks.pending,
            last_accepted_age_secs: snapshot.blocks.last_accepted_age_secs,
        },
        inference: InferenceView {
            state: inference_state,
            requested: snapshot.inference.requested,
            prepared: snapshot.inference.prepared,
            served: snapshot.inference.served,
            delivered: snapshot.inference.delivered,
            failed: snapshot.inference.failed,
            busy: snapshot.inference.busy,
            active: snapshot.inference.active,
            queue_depth: snapshot.inference.queue_depth,
            queue_capacity: snapshot.inference.queue_capacity,
            gpu_route_count: snapshot.inference.route_gpus.len(),
            gpu_index: inference_gpu,
            model: snapshot.inference.model_name,
            model_id: if snapshot.inference.model_id_prefix.is_empty() {
                String::new()
            } else {
                format!("{}…", snapshot.inference.model_id_prefix)
            },
            tier: snapshot.inference.tier,
            backend: snapshot.inference.backend,
            last_latency_ms: snapshot.inference.last_latency_ms,
            p95_latency_ms: snapshot.inference.p95_latency_ms,
            last_tokens: snapshot.inference.last_tokens,
            pow_pause_total_secs: snapshot.inference.pow_pause_total_ms / 1_000,
            self_test_ok: if snapshot.inference.serveable_models != 0 {
                Some(true)
            } else if snapshot.inference.staging_error {
                Some(false)
            } else {
                None
            },
            status: inference_status,
        },
        escrow,
        service_bond,
        devices,
        events,
    }
}

fn inference_display_gpu(inference: &runtime::InferenceSnapshot) -> Option<u32> {
    let unique_route_gpu = match inference.route_gpus.as_slice() {
        [gpu] => Some(*gpu),
        _ => None,
    };
    if inference.active == 1 {
        inference.gpu_index.or(unique_route_gpu)
    } else {
        unique_route_gpu
    }
}

fn adapt_service_bond(bond: &runtime::ServiceBondSnapshot) -> ServiceBondView {
    ServiceBondView {
        // `available = false` with no heartbeat/failure is the neutral PENDING state. Do not
        // fabricate health before the solo node's first one-minute service-bond poll.
        available: bond.available,
        consecutive_misses: bond.consecutive_misses,
        last_strike_daa: bond.last_strike_daa,
        burned_claims: bond.burned_claims,
        burned_amount: format_krx(bond.burned_sompi),
        suspended_until_daa: bond.suspended_until_daa,
        heartbeat_alive: bond.last_heartbeat_age_secs.is_some_and(|age| age <= 180),
        last_heartbeat_age_secs: bond.last_heartbeat_age_secs,
        last_failure_age_secs: bond.last_failure_age_secs,
    }
}

fn adapt_escrow(snapshot: &runtime::Snapshot, mode: tui::MiningMode) -> EscrowView {
    let escrow = &snapshot.escrow;
    let enabled = mode == tui::MiningMode::Solo && escrow.enabled;
    let claiming = enabled && (escrow.in_flight_txs != 0 || escrow.status == runtime::EscrowStatus::Claiming);
    let connected =
        matches!(snapshot.connection, runtime::ConnectionState::Connected | runtime::ConnectionState::Failover);
    let alive = enabled
        && connected
        && (escrow.heartbeat_age_secs.is_some_and(|age| age <= 120)
            || escrow.validation_in_progress
            || escrow.in_flight_txs != 0);
    let status = match escrow.status {
        runtime::EscrowStatus::Disabled => "disabled",
        runtime::EscrowStatus::Ready => {
            if alive {
                "alive"
            } else if escrow.heartbeat_age_secs.is_some() {
                "heartbeat lost"
            } else {
                "waiting"
            }
        }
        runtime::EscrowStatus::Validating => "validating",
        runtime::EscrowStatus::Held => "held",
        runtime::EscrowStatus::Claiming => "claiming",
        runtime::EscrowStatus::Degraded => "degraded",
    }
    .to_string();
    let message = if !enabled {
        "solo claim worker inactive in pool mode".to_string()
    } else if escrow.claims_held {
        "claims held for node compatibility; tracked funds remain pending".to_string()
    } else if escrow.validation_in_progress {
        format!("validating {} saved block reference(s)", escrow.validation_pending_blocks)
    } else if claiming {
        format!("{} claim transaction(s) in flight", escrow.in_flight_txs)
    } else if escrow.status == runtime::EscrowStatus::Degraded {
        "state persistence requires attention".to_string()
    } else if alive {
        format!("claim worker heartbeat healthy · DAA {}", escrow.last_seen_daa.unwrap_or(0))
    } else if escrow.heartbeat_age_secs.is_some() {
        "solo claim worker heartbeat is stale".to_string()
    } else {
        "waiting for the solo claim worker heartbeat".to_string()
    };
    EscrowView {
        enabled,
        claiming,
        alive,
        // Keep the A/F/P row output-denominated. One transaction may batch many outputs, so a
        // transaction count beside output failure/pending counts would be misleading.
        claims_accepted: escrow.accepted_outputs,
        // Retriable node responses remain visible in the status message/counters, but are not
        // dishonestly labeled permanent failures. This field is reserved for terminal loss.
        claims_failed: escrow
            .terminal_slashed_outputs
            .saturating_add(escrow.discarded_red_outputs)
            .saturating_add(escrow.discarded_ghost_outputs),
        claims_pending: escrow.pending_live_outputs,
        last_attempt_age_secs: escrow.last_attempt_age_secs,
        last_success_age_secs: escrow.last_success_age_secs,
        claimable_amount: format_krx(escrow.pending_gross_sompi),
        claimed_amount: format_krx(escrow.accepted_net_sompi),
        status,
        message,
    }
}

fn inference_status(state: InferenceState, inference: &runtime::InferenceSnapshot) -> String {
    match state {
        InferenceState::Unavailable => "no proven GPU route".to_string(),
        InferenceState::Preparing => "model staging / self-test".to_string(),
        InferenceState::Ready => format!("{} model route(s) ready", inference.serveable_models),
        InferenceState::Serving => "serving external inference".to_string(),
        InferenceState::Degraded => "GPU inference route degraded".to_string(),
    }
}

fn map_event_kind(kind: runtime::EventKind) -> UiEventKind {
    match kind {
        runtime::EventKind::Info => UiEventKind::Info,
        runtime::EventKind::Job => UiEventKind::Job,
        runtime::EventKind::ShareAccepted => UiEventKind::ShareAccepted,
        runtime::EventKind::ShareRejected => UiEventKind::ShareRejected,
        runtime::EventKind::BlockFound => UiEventKind::BlockFound,
        runtime::EventKind::BlockAccepted => UiEventKind::BlockAccepted,
        runtime::EventKind::BlockRejected => UiEventKind::BlockRejected,
        runtime::EventKind::InferenceOk => UiEventKind::InferenceOk,
        runtime::EventKind::InferenceFailed => UiEventKind::InferenceFailed,
        runtime::EventKind::Escrow => UiEventKind::Escrow,
        runtime::EventKind::HealthWarning => UiEventKind::HealthWarn,
        runtime::EventKind::Error => UiEventKind::Error,
    }
}

fn clean_device_name(label: &str) -> String {
    let trimmed = label.trim();
    if let Some(open) = trimmed.find('(') {
        if let Some(close) = trimmed.rfind(')') {
            if close > open {
                return trimmed[open + 1..close].trim().to_string();
            }
        }
    }
    trimmed.trim_start_matches('#').trim_start_matches(|ch: char| ch.is_ascii_digit()).trim().to_string()
}

fn format_uptime_clock(uptime_ms: u64) -> String {
    let total = uptime_ms / 1_000;
    format!("{:02}:{:02}:{:02}", (total / 3_600) % 100, (total / 60) % 60, total % 60)
}

fn format_krx(sompi: u64) -> String {
    format!("{:.8} KRX", sompi as f64 / 100_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::{
        current_thread_owns_terminal_io, inference_display_gpu, lock_terminal_io, restore_terminal_for_exit,
        runtime, sanitize_log_message, should_enable_with, ClockTrack, EXIT_RESTORE_REQUESTED,
    };
    use std::ffi::OsString;

    fn args(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn selection_requires_three_interactive_streams() {
        assert!(should_enable_with(&args(&["miner"]), true, true, true, Some("xterm-256color"), false));
        assert!(!should_enable_with(&args(&["miner"]), false, true, true, Some("xterm"), false));
        assert!(!should_enable_with(&args(&["miner"]), true, false, true, Some("xterm"), false));
        assert!(!should_enable_with(&args(&["miner"]), true, true, false, Some("xterm"), false));
    }

    #[test]
    fn explicit_and_noninteractive_modes_win() {
        assert!(!should_enable_with(&args(&["miner", "--no-tui"]), true, true, true, Some("xterm"), false));
        assert!(!should_enable_with(&args(&["miner", "--help"]), true, true, true, Some("xterm"), false));
        assert!(!should_enable_with(&args(&["miner", "--recover-escrow"]), true, true, true, Some("xterm"), false));
        assert!(!should_enable_with(&args(&["miner"]), true, true, true, Some("dumb"), false));
        assert!(!should_enable_with(&args(&["miner"]), true, true, true, Some("xterm"), true));
    }

    #[test]
    fn terminal_lock_tracks_same_thread_ownership_for_panic_restore() {
        assert!(!current_thread_owns_terminal_io());
        {
            let _terminal_io = lock_terminal_io();
            assert!(current_thread_owns_terminal_io());
        }
        assert!(!current_thread_owns_terminal_io());
    }

    #[test]
    fn hard_exit_restore_is_bounded_behind_a_wedged_draw_lock() {
        let (locked_tx, locked_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            let _terminal_io = lock_terminal_io();
            locked_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        locked_rx.recv().unwrap();

        let started = std::time::Instant::now();
        restore_terminal_for_exit();
        assert!(started.elapsed() < std::time::Duration::from_secs(2));

        release_tx.send(()).unwrap();
        holder.join().unwrap();
        EXIT_RESTORE_REQUESTED.store(false, std::sync::atomic::Ordering::Release);
    }

    #[test]
    fn log_sanitizer_strips_controls_and_redacts_identities() {
        assert_eq!(sanitize_log_message("hello\u{1b}[31m world\nnext").as_deref(), Some("hello world next"));
        assert_eq!(
            sanitize_log_message("Mining for: keryx:private.worker").as_deref(),
            Some("Mining identity configured")
        );
        assert_eq!(
            sanitize_log_message("OPoI escrow active: pubkey=abcdef next").as_deref(),
            Some("OPoI escrow active: [redacted] next")
        );
        assert_eq!(
            sanitize_log_message("saved /home/operator/private/escrow.cert for keryx:qprivate.worker").as_deref(),
            Some("saved [redacted] for [redacted]")
        );
        assert_eq!(
            sanitize_log_message(
                "response txid=0123456789abcdef0123456789abcdef CID=Qmabcdefghijklmnopqrstuvwxyz0123456789ABCDEFG"
            )
            .as_deref(),
            Some("response [redacted] [redacted]")
        );
        assert_eq!(
            sanitize_log_message("dial grpc://user:secret@192.168.1.7:22110").as_deref(),
            Some("dial [redacted]")
        );
    }

    #[test]
    fn clock_debounce_counts_backend_samples_not_render_frames() {
        let mut track = ClockTrack::default();
        assert_eq!(track.observe(Some(1_500), true, true, true), Some(1_500));

        assert_eq!(track.observe(Some(1_200), true, true, true), None);
        for _ in 0..20 {
            assert_eq!(track.observe(Some(1_200), true, true, false), None);
        }
        assert_eq!(track.low_streak, 1);

        assert_eq!(track.observe(Some(1_200), true, true, true), None);
        assert_eq!(track.observe(Some(1_200), true, true, true), Some(1_500));
    }

    #[test]
    fn ready_route_gpu_is_exact_only_when_unambiguous() {
        let unique = runtime::InferenceSnapshot { route_gpus: vec![3], serveable_models: 1, ..Default::default() };
        assert_eq!(inference_display_gpu(&unique), Some(3));

        let multiple = runtime::InferenceSnapshot { route_gpus: vec![1, 3], serveable_models: 1, ..Default::default() };
        assert_eq!(inference_display_gpu(&multiple), None);

        let serving = runtime::InferenceSnapshot {
            route_gpus: vec![1, 3],
            gpu_index: Some(1),
            active: 1,
            serveable_models: 1,
            ..Default::default()
        };
        assert_eq!(inference_display_gpu(&serving), Some(1));
    }
}
