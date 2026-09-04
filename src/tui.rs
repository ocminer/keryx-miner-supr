//! Presentation-only terminal dashboard for the miner.
//!
//! This module deliberately owns no mining, network, model, escrow, or GPU
//! handles.  The runtime publishes a cheap [`UiSnapshot`]; the renderer only
//! formats that snapshot and turns keyboard input into [`TuiAction`] values.
//! In particular, no key handled here can pause mining, change clocks, select
//! an inference device, or mutate escrow state.
//!
//! Matrix-style animation is constrained to panel titles, reserved logo
//! gutters, and a tiny allowlist of ornamental static body labels.  Numeric
//! telemetry, endpoints, warnings, event messages, device names, counters and
//! status values are never passed through the glyph mutator.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap},
    Frame,
};

const LOGO: [&str; 5] = [
    r" _  __ _____ ____  __   __ __  __",
    r"| |/ /| ____|  _ \ \ \ / / \ \/ /",
    r"| ' / |  _| | |_) | \ V /   \  / ",
    r"| . \ | |___|  _ <   | |    /  \ ",
    r"|_|\_\|_____|_| \_\  |_|   /_/\_\",
];

const MATRIX_UNICODE: [char; 20] =
    ['ｱ', 'ｲ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｸ', 'ｹ', 'ｺ', 'ｻ', 'ｼ', 'ｽ', 'ｾ', 'ｿ', 'ﾊ', 'ﾐ', 'ﾑ', 'ﾒ', 'ﾓ'];
const MATRIX_ASCII: [char; 12] = ['0', '1', '<', '>', '[', ']', '{', '}', ':', '/', '\\', '|'];
const POWERED_BY: &str = "powered by krx.suprnova.cc";
const POWER_SCAN_REST_TICKS: u64 = 40;

/// Pool and solo counters have deliberately different meanings in the UI.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MiningMode {
    #[default]
    Pool,
    Solo,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ConnectionState {
    #[default]
    Connecting,
    Connected,
    Failover,
    Offline,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MiningState {
    #[default]
    Preparing,
    Mining,
    InferencePaused,
    Degraded,
    Stopped,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InferenceState {
    #[default]
    Unavailable,
    Preparing,
    Ready,
    Serving,
    Degraded,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DeviceActivity {
    #[default]
    Preparing,
    Mining,
    Inference,
    Paused,
    Stalled,
    Offline,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum UiEventKind {
    #[default]
    Info,
    Job,
    ShareAccepted,
    ShareRejected,
    BlockFound,
    BlockAccepted,
    BlockRejected,
    InferenceOk,
    InferenceFailed,
    Escrow,
    HealthWarn,
    Error,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConnectionView {
    pub mode: MiningMode,
    /// Display endpoint only.  Wallets, worker secrets and credentials do not
    /// belong in this value or anywhere else in a UI snapshot.
    pub endpoint: String,
    pub state: ConnectionState,
    pub latency_ms: Option<u64>,
    pub last_job_age_secs: Option<u64>,
    pub difficulty: Option<f64>,
    pub daa_score: Option<u64>,
    pub failover: String,
    pub synced: Option<bool>,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MiningView {
    pub state: MiningState,
    pub total_hashrate_hs: f64,
    pub average_60s_hs: Option<f64>,
    /// Oldest to newest.  Keeping this bounded is the producer's job.
    pub hashrate_history_hs: Vec<f64>,
    pub total_power_w: Option<f64>,
    pub efficiency_mhs_per_w: Option<f64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ShareView {
    pub accepted: u64,
    pub rejected: u64,
    pub stale: u64,
    pub low_diff: u64,
    pub duplicate: u64,
    pub other: u64,
    pub pending: u64,
    pub last_accepted_age_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BlockView {
    /// Locally detected network-target candidates.
    pub found: u64,
    /// `None` means the endpoint does not report this fact.  The renderer uses
    /// `--`; it must never infer a pool block from an accepted share.
    pub accepted: Option<u64>,
    pub rejected: Option<u64>,
    pub pending: u64,
    pub last_accepted_age_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct InferenceView {
    pub state: InferenceState,
    pub requested: u64,
    pub prepared: u64,
    /// Successful result generated and queued/cached.
    pub served: u64,
    /// Successful transport acknowledgement, when the protocol provides one.
    pub delivered: u64,
    pub failed: u64,
    /// Terminal admission/deadline refusal, distinct from execution failure.
    pub busy: u64,
    pub active: u64,
    pub queue_depth: u64,
    pub queue_capacity: u64,
    /// Number of exact, currently proven GPU routes for the displayed model.
    pub gpu_route_count: usize,
    /// Exact route when there is only one, or the active route when unambiguous.
    pub gpu_index: Option<u32>,
    pub model: String,
    pub model_id: String,
    pub tier: String,
    pub backend: String,
    pub last_latency_ms: Option<u64>,
    pub p95_latency_ms: Option<u64>,
    pub last_tokens: Option<u64>,
    pub pow_pause_total_secs: u64,
    /// Startup/self-test health; these probes are not included in request
    /// served/failed counters.
    pub self_test_ok: Option<bool>,
    pub status: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EscrowView {
    pub enabled: bool,
    pub claiming: bool,
    /// Liveness/heartbeat of the claim worker, not merely configuration.
    pub alive: bool,
    pub claims_accepted: u64,
    pub claims_failed: u64,
    pub claims_pending: u64,
    pub last_attempt_age_secs: Option<u64>,
    pub last_success_age_secs: Option<u64>,
    /// Display-ready amounts including unit.  The runtime adapter may leave
    /// either empty when the corresponding value is unavailable.
    pub claimable_amount: String,
    pub claimed_amount: String,
    pub status: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ServiceBondView {
    pub available: bool,
    pub consecutive_misses: u64,
    pub last_strike_daa: Option<u64>,
    pub burned_claims: u64,
    /// Display-ready amount including unit.
    pub burned_amount: String,
    pub suspended_until_daa: Option<u64>,
    pub heartbeat_alive: bool,
    pub last_heartbeat_age_secs: Option<u64>,
    pub last_failure_age_secs: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceView {
    pub index: u32,
    pub name: String,
    pub backend: String,
    pub hashrate_hs: f64,
    pub temp_c: Option<u32>,
    pub hotspot_c: Option<u32>,
    pub fan_pct: Option<u32>,
    pub power_w: Option<f64>,
    pub core_mhz: Option<u32>,
    pub mem_mhz: Option<u32>,
    /// Per-card active-mining baseline.  Absolute clock thresholds are not
    /// portable between GPU models.
    pub baseline_core_mhz: Option<u32>,
    pub baseline_mem_mhz: Option<u32>,
    pub vram_used_mb: Option<u64>,
    pub vram_total_mb: Option<u64>,
    pub efficiency_mhs_per_w: Option<f64>,
    pub accepted: u64,
    pub rejected: u64,
    pub inference_host: bool,
    pub activity: DeviceActivity,
    pub throttle_reason: Option<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UiEvent {
    /// Already formatted local time or relative time (for example `14:22:08`).
    pub timestamp: String,
    pub kind: UiEventKind,
    /// A short operator-safe summary.  Inference prompts/results, wallet data,
    /// credentials, filesystem paths and machine identity must not be placed in
    /// UI events by the producer.
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UiSnapshot {
    pub miner_name: String,
    pub version: String,
    pub build: String,
    pub algorithm: String,
    pub era: String,
    pub uptime_secs: u64,
    pub connection: ConnectionView,
    pub mining: MiningView,
    pub shares: ShareView,
    pub blocks: BlockView,
    pub inference: InferenceView,
    pub escrow: EscrowView,
    pub service_bond: ServiceBondView,
    pub devices: Vec<DeviceView>,
    /// Chronological, oldest to newest.
    pub events: Vec<UiEvent>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum DashboardPage {
    #[default]
    Overview,
    Gpus,
    Inference,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ColorMode {
    TrueColor,
    Color256,
    Basic,
    #[default]
    Mono,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiAction {
    None,
    /// The host owns shutdown.  Returning this action is the only thing the
    /// renderer does after a confirmed `Q Q` sequence.
    QuitConfirmed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayoutMode {
    Wide,
    Medium,
    Stacked,
    Tabbed,
    Tiny,
}

#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum Severity {
    Cool,
    Nominal,
    Watch,
    Alert,
    #[default]
    Unknown,
}

#[derive(Clone, Debug)]
pub struct TuiState {
    pub page: DashboardPage,
    pub logs_expanded: bool,
    pub help_visible: bool,
    pub quit_armed: bool,
    pub motion_enabled: bool,
    pub unicode_glyphs: bool,
    pub color_mode: ColorMode,
    /// Number of messages skipped back from the newest event.
    pub log_scroll: usize,
    animation_tick: u64,
}

impl Default for TuiState {
    fn default() -> Self {
        Self {
            page: DashboardPage::Overview,
            logs_expanded: false,
            help_visible: false,
            quit_armed: false,
            motion_enabled: true,
            unicode_glyphs: true,
            color_mode: ColorMode::TrueColor,
            log_scroll: 0,
            animation_tick: 0,
        }
    }
}

impl TuiState {
    /// Environment-aware defaults.  TTY detection and `--no-tui` selection
    /// remain the host's responsibility.
    pub fn from_environment() -> Self {
        let mut state = Self::default();
        let term = std::env::var("TERM").unwrap_or_default().to_ascii_lowercase();
        let color_term = std::env::var("COLORTERM").unwrap_or_default().to_ascii_lowercase();
        state.color_mode = if std::env::var_os("NO_COLOR").is_some() || term == "dumb" {
            ColorMode::Mono
        } else if color_term.contains("truecolor") || color_term.contains("24bit") {
            ColorMode::TrueColor
        } else if term.contains("256color") {
            ColorMode::Color256
        } else {
            ColorMode::Basic
        };

        let locale = std::env::var("LC_ALL")
            .or_else(|_| std::env::var("LC_CTYPE"))
            .or_else(|_| std::env::var("LANG"))
            .unwrap_or_default()
            .to_ascii_lowercase();
        state.unicode_glyphs = locale.contains("utf-8") || locale.contains("utf8");
        state.motion_enabled = term != "dumb"
            && !matches!(std::env::var("KERYX_TUI_MOTION").ok().as_deref(), Some("0") | Some("false") | Some("off"));
        state
    }

    /// Advance only the decorative animation.  Runtime telemetry must be
    /// sampled on its own slower cadence rather than on each UI tick.
    pub fn tick_animation(&mut self) {
        if self.motion_enabled {
            self.animation_tick = self.animation_tick.wrapping_add(1);
        }
    }

    pub fn animation_tick(&self) -> u64 {
        self.animation_tick
    }

    #[cfg(test)]
    fn set_animation_tick(&mut self, tick: u64) {
        self.animation_tick = tick;
    }
}

/// Input handling is intentionally limited to presentation and a confirmed
/// graceful-quit request.  Key release events are ignored so terminals that
/// report press/release pairs do not turn one `Q` into two.
pub fn handle_key(state: &mut TuiState, key: KeyEvent) -> TuiAction {
    if key.kind == KeyEventKind::Release {
        return TuiAction::None;
    }

    if key.modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) {
        // Control sequences (including Ctrl-C) remain owned by the host's
        // signal/shutdown path and can never satisfy a quit confirmation.
        return TuiAction::None;
    }

    match key.code {
        KeyCode::Char('q' | 'Q') => {
            // Holding Q must not turn the terminal's auto-repeat into a second
            // confirmation event.
            if key.kind != KeyEventKind::Press {
                return TuiAction::None;
            }
            state.help_visible = false;
            if state.quit_armed {
                TuiAction::QuitConfirmed
            } else {
                state.quit_armed = true;
                TuiAction::None
            }
        }
        KeyCode::Esc => {
            if state.quit_armed {
                state.quit_armed = false;
            } else if state.help_visible {
                state.help_visible = false;
            } else if state.logs_expanded {
                state.logs_expanded = false;
            }
            TuiAction::None
        }
        KeyCode::Char('m' | 'M') => {
            state.quit_armed = false;
            state.motion_enabled = !state.motion_enabled;
            TuiAction::None
        }
        KeyCode::Char('l' | 'L') => {
            state.quit_armed = false;
            state.help_visible = false;
            state.logs_expanded = !state.logs_expanded;
            TuiAction::None
        }
        KeyCode::Char('1') => {
            state.quit_armed = false;
            state.page = DashboardPage::Overview;
            state.logs_expanded = false;
            TuiAction::None
        }
        KeyCode::Char('2') => {
            state.quit_armed = false;
            state.page = DashboardPage::Gpus;
            state.logs_expanded = false;
            TuiAction::None
        }
        KeyCode::Char('3') => {
            state.quit_armed = false;
            state.page = DashboardPage::Inference;
            state.logs_expanded = false;
            TuiAction::None
        }
        KeyCode::Char('?') => {
            state.quit_armed = false;
            state.help_visible = !state.help_visible;
            TuiAction::None
        }
        KeyCode::Up => {
            state.quit_armed = false;
            state.log_scroll = state.log_scroll.saturating_add(1);
            TuiAction::None
        }
        KeyCode::Down => {
            state.quit_armed = false;
            state.log_scroll = state.log_scroll.saturating_sub(1);
            TuiAction::None
        }
        KeyCode::PageUp => {
            state.quit_armed = false;
            state.log_scroll = state.log_scroll.saturating_add(10);
            TuiAction::None
        }
        KeyCode::PageDown | KeyCode::End => {
            state.quit_armed = false;
            state.log_scroll = if matches!(key.code, KeyCode::End) { 0 } else { state.log_scroll.saturating_sub(10) };
            TuiAction::None
        }
        KeyCode::Left => {
            state.quit_armed = false;
            state.page = match state.page {
                DashboardPage::Overview => DashboardPage::Inference,
                DashboardPage::Gpus => DashboardPage::Overview,
                DashboardPage::Inference => DashboardPage::Gpus,
            };
            TuiAction::None
        }
        KeyCode::Right | KeyCode::Tab => {
            state.quit_armed = false;
            state.page = match state.page {
                DashboardPage::Overview => DashboardPage::Gpus,
                DashboardPage::Gpus => DashboardPage::Inference,
                DashboardPage::Inference => DashboardPage::Overview,
            };
            TuiAction::None
        }
        KeyCode::BackTab => {
            state.quit_armed = false;
            state.page = match state.page {
                DashboardPage::Overview => DashboardPage::Inference,
                DashboardPage::Gpus => DashboardPage::Overview,
                DashboardPage::Inference => DashboardPage::Gpus,
            };
            TuiAction::None
        }
        _ => {
            // A stray key cancels an armed quit so an accidental Q cannot stay
            // latent while the operator navigates the dashboard.
            state.quit_armed = false;
            TuiAction::None
        }
    }
}

pub fn layout_mode(area: Rect) -> LayoutMode {
    match (area.width, area.height) {
        (w, h) if w >= 120 && h >= 34 => LayoutMode::Wide,
        (w, h) if w >= 90 && h >= 28 => LayoutMode::Medium,
        (w, h) if w >= 70 && h >= 30 => LayoutMode::Stacked,
        (w, h) if w >= 70 && h >= 20 => LayoutMode::Tabbed,
        _ => LayoutMode::Tiny,
    }
}

#[derive(Clone, Copy)]
struct Palette {
    background: Color,
    panel: Color,
    border: Color,
    border_focus: Color,
    label: Color,
    text: Color,
    green: Color,
    bright_green: Color,
    cyan: Color,
    yellow: Color,
    red: Color,
    alert_dim: Color,
    muted: Color,
    rain: Color,
    rain_head: Color,
}

impl Palette {
    fn new(mode: ColorMode) -> Self {
        match mode {
            ColorMode::TrueColor => Self {
                background: Color::Rgb(1, 4, 3),
                panel: Color::Rgb(3, 11, 8),
                border: Color::Rgb(9, 45, 25),
                border_focus: Color::Rgb(18, 112, 54),
                label: Color::Rgb(70, 101, 81),
                text: Color::Rgb(151, 177, 159),
                green: Color::Rgb(18, 139, 66),
                bright_green: Color::Rgb(78, 188, 108),
                cyan: Color::Rgb(39, 125, 148),
                yellow: Color::Rgb(194, 143, 35),
                red: Color::Rgb(218, 51, 61),
                alert_dim: Color::Rgb(68, 17, 22),
                muted: Color::Rgb(52, 72, 61),
                rain: Color::Rgb(6, 32, 17),
                rain_head: Color::Rgb(14, 77, 36),
            },
            ColorMode::Color256 => Self {
                background: Color::Indexed(232),
                panel: Color::Indexed(233),
                border: Color::Indexed(22),
                border_focus: Color::Indexed(28),
                label: Color::Indexed(65),
                text: Color::Indexed(108),
                green: Color::Indexed(28),
                bright_green: Color::Indexed(77),
                cyan: Color::Indexed(30),
                yellow: Color::Indexed(178),
                red: Color::Indexed(167),
                alert_dim: Color::Indexed(52),
                muted: Color::Indexed(239),
                rain: Color::Indexed(22),
                rain_head: Color::Indexed(28),
            },
            ColorMode::Basic => Self {
                background: Color::Black,
                panel: Color::Black,
                border: Color::DarkGray,
                border_focus: Color::Green,
                label: Color::DarkGray,
                text: Color::Gray,
                green: Color::Green,
                bright_green: Color::LightGreen,
                cyan: Color::Cyan,
                yellow: Color::Yellow,
                red: Color::Red,
                alert_dim: Color::DarkGray,
                muted: Color::DarkGray,
                rain: Color::DarkGray,
                rain_head: Color::Green,
            },
            ColorMode::Mono => Self {
                background: Color::Reset,
                panel: Color::Reset,
                border: Color::Reset,
                border_focus: Color::Reset,
                label: Color::Reset,
                text: Color::Reset,
                green: Color::Reset,
                bright_green: Color::Reset,
                cyan: Color::Reset,
                yellow: Color::Reset,
                red: Color::Reset,
                alert_dim: Color::Reset,
                muted: Color::Reset,
                rain: Color::Reset,
                rain_head: Color::Reset,
            },
        }
    }

    fn severity(self, severity: Severity) -> Style {
        let color = match severity {
            Severity::Cool => self.cyan,
            Severity::Nominal => self.green,
            Severity::Watch => self.yellow,
            Severity::Alert => self.red,
            Severity::Unknown => self.muted,
        };
        Style::default().fg(color)
    }

    fn canvas(self) -> Style {
        Style::default().fg(self.text).bg(self.background)
    }
}

/// Edge-temperature thresholds.  Hotspot severity is evaluated separately in
/// [`device_severity`].
pub fn temperature_severity(temp_c: Option<u32>) -> Severity {
    match temp_c {
        Some(0..=44) => Severity::Cool,
        Some(45..=69) => Severity::Nominal,
        Some(70..=79) => Severity::Watch,
        Some(_) => Severity::Alert,
        None => Severity::Unknown,
    }
}

pub fn hotspot_severity(temp_c: Option<u32>) -> Severity {
    match temp_c {
        Some(0..=89) => Severity::Nominal,
        Some(90..=99) => Severity::Watch,
        Some(_) => Severity::Alert,
        None => Severity::Unknown,
    }
}

/// Compare a current clock to the learned baseline for this exact card.  A
/// transient low sample should be filtered by the producer; the UI has no
/// timing authority and simply presents the snapshot verdict.
pub fn clock_severity(current_mhz: Option<u32>, baseline_mhz: Option<u32>) -> Severity {
    match (current_mhz, baseline_mhz) {
        (Some(_), Some(0)) => Severity::Unknown,
        (Some(current), Some(baseline)) => {
            let ratio = current as f64 / baseline as f64;
            if ratio >= 0.90 {
                Severity::Nominal
            } else if ratio >= 0.75 {
                Severity::Watch
            } else {
                Severity::Alert
            }
        }
        _ => Severity::Unknown,
    }
}

pub fn device_severity(device: &DeviceView) -> Severity {
    match device.activity {
        DeviceActivity::Offline | DeviceActivity::Stalled => return Severity::Alert,
        DeviceActivity::Preparing | DeviceActivity::Inference | DeviceActivity::Paused | DeviceActivity::Mining => {}
    }

    if device.throttle_reason.as_ref().is_some_and(|r| !r.trim().is_empty()) {
        return Severity::Alert;
    }

    let thermal = max_known_severity(temperature_severity(device.temp_c), hotspot_severity(device.hotspot_c));
    if device.activity != DeviceActivity::Mining {
        // Paused/inference clocks are intentionally cool/blue, but a physically
        // hot card must remain amber/red regardless of why its clock fell.
        return match thermal {
            Severity::Watch | Severity::Alert => thermal,
            _ => Severity::Cool,
        };
    }

    let mut severity = thermal;
    severity = max_known_severity(severity, clock_severity(device.core_mhz, device.baseline_core_mhz));
    severity = max_known_severity(severity, clock_severity(device.mem_mhz, device.baseline_mem_mhz));
    severity
}

fn device_has_overheat_alert(device: &DeviceView) -> bool {
    temperature_severity(device.temp_c) == Severity::Alert || hotspot_severity(device.hotspot_c) == Severity::Alert
}

/// The renderer advances at 8 Hz. Two ticks per visual phase yields a 500 ms on/off cycle: a
/// deliberate 2 Hz blink without relying on terminal-specific BLINK support. Turning motion off
/// is also the accessibility fallback and leaves the warning steadily red and bold.
fn device_status_style(device: &DeviceView, severity: Severity, state: &TuiState, palette: &Palette) -> Style {
    if severity != Severity::Alert || !device_has_overheat_alert(device) {
        return palette.severity(severity);
    }
    if !state.motion_enabled {
        return Style::default().fg(palette.red).add_modifier(Modifier::BOLD);
    }
    if (state.animation_tick / 2) % 2 == 0 {
        Style::default().fg(palette.red).add_modifier(Modifier::BOLD | Modifier::REVERSED)
    } else {
        Style::default().fg(palette.alert_dim).add_modifier(Modifier::DIM)
    }
}

fn max_known_severity(a: Severity, b: Severity) -> Severity {
    match (a, b) {
        (Severity::Unknown, x) | (x, Severity::Unknown) => x,
        _ => a.max(b),
    }
}

/// Strip terminal controls and collapse whitespace.  Runtime strings can
/// originate in remote protocol messages; they must remain inert terminal
/// text.  This does not attempt to redact secrets—the snapshot producer must
/// never publish them in the first place.
pub fn safe_text(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.chars() {
        if ch.is_control() || is_invisible_format_control(ch) {
            pending_space |= ch.is_whitespace();
            continue;
        }
        if ch.is_whitespace() {
            pending_space = !out.is_empty();
            continue;
        }
        if pending_space {
            out.push(' ');
            pending_space = false;
        }
        out.push(ch);
    }
    out.trim().to_owned()
}

fn is_invisible_format_control(ch: char) -> bool {
    matches!(
        ch,
        '\u{200b}'..='\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2060}'..='\u{206f}'
            | '\u{feff}'
    )
}

pub fn sorted_devices(devices: &[DeviceView]) -> Vec<&DeviceView> {
    let mut sorted: Vec<_> = devices.iter().collect();
    sorted.sort_by(|a, b| {
        a.index
            .cmp(&b.index)
            .then_with(|| safe_text(&a.name).to_ascii_lowercase().cmp(&safe_text(&b.name).to_ascii_lowercase()))
            .then_with(|| safe_text(&a.backend).cmp(&safe_text(&b.backend)))
    });
    sorted
}

pub fn draw(frame: &mut Frame<'_>, state: &TuiState, snapshot: &UiSnapshot) {
    let area = frame.area();
    let palette = Palette::new(state.color_mode);
    frame.render_widget(Block::default().style(palette.canvas()), area);

    if area.width < 20 || area.height < 5 {
        render_too_small(frame, area, snapshot, &palette);
        return;
    }

    if state.logs_expanded {
        render_expanded_logs(frame, area, state, snapshot, &palette);
    } else {
        match layout_mode(area) {
            LayoutMode::Wide => render_wide(frame, area, state, snapshot, 62, &palette),
            LayoutMode::Medium => render_wide(frame, area, state, snapshot, 60, &palette),
            LayoutMode::Stacked => render_stacked(frame, area, state, snapshot, &palette),
            LayoutMode::Tabbed => render_tabbed(frame, area, state, snapshot, &palette),
            LayoutMode::Tiny => render_tiny(frame, area, state, snapshot, &palette),
        }
    }

    if state.help_visible {
        render_help(frame, area, state, &palette);
    }
    if state.quit_armed {
        render_quit_confirmation(frame, area, state, &palette);
    }
}

fn render_wide(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &TuiState,
    snapshot: &UiSnapshot,
    left_percent: u16,
    palette: &Palette,
) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(61), Constraint::Min(10), Constraint::Length(1)])
        .split(area);
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(left_percent), Constraint::Percentage(100 - left_percent)])
        .split(rows[0]);
    render_core(frame, top[0], state, snapshot, palette, false);
    render_rig(frame, top[1], state, snapshot, palette, false);
    render_page(frame, rows[1], state, snapshot, palette);
    render_footer(frame, rows[2], state, palette);
}

fn render_stacked(frame: &mut Frame<'_>, area: Rect, state: &TuiState, snapshot: &UiSnapshot, palette: &Palette) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Length(9), Constraint::Min(10), Constraint::Length(1)])
        .split(area);
    render_core(frame, rows[0], state, snapshot, palette, true);
    render_rig(frame, rows[1], state, snapshot, palette, true);
    render_page(frame, rows[2], state, snapshot, palette);
    render_footer(frame, rows[3], state, palette);
}

fn render_tabbed(frame: &mut Frame<'_>, area: Rect, state: &TuiState, snapshot: &UiSnapshot, palette: &Palette) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(10), Constraint::Length(1)])
        .split(area);
    render_core(frame, rows[0], state, snapshot, palette, true);
    match state.page {
        DashboardPage::Overview => render_overview(frame, rows[1], state, snapshot, palette),
        DashboardPage::Gpus => render_gpu_detail(frame, rows[1], state, snapshot, palette),
        DashboardPage::Inference => render_inference_detail(frame, rows[1], state, snapshot, palette),
    }
    render_footer(frame, rows[2], state, palette);
}

fn render_page(frame: &mut Frame<'_>, area: Rect, state: &TuiState, snapshot: &UiSnapshot, palette: &Palette) {
    match state.page {
        DashboardPage::Overview => render_overview(frame, area, state, snapshot, palette),
        DashboardPage::Gpus => render_gpu_detail(frame, area, state, snapshot, palette),
        DashboardPage::Inference => render_inference_detail(frame, area, state, snapshot, palette),
    }
}

fn panel_block(title: &str, salt: u64, state: &TuiState, palette: &Palette, focused: bool) -> Block<'static> {
    let title = matrix_title(title, salt, state, palette);
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if focused { palette.border_focus } else { palette.border }))
        .style(Style::default().fg(palette.text).bg(palette.panel))
        .title(title)
}

fn matrix_title(title: &str, salt: u64, state: &TuiState, palette: &Palette) -> Line<'static> {
    let clean = safe_text(title);
    if !state.motion_enabled || is_warning_text(&clean) {
        return Line::from(Span::styled(
            format!(" {clean} "),
            Style::default().fg(palette.green).add_modifier(Modifier::BOLD),
        ));
    }

    // Four mutation frames in every 23 make the effect legible without turning titles into noise.
    // Only alphabetic characters in our fixed decorative title are eligible; digits are provably
    // untouched, and warning titles are rejected above.
    let phase = state.animation_tick.wrapping_add(salt.wrapping_mul(11)) % 23;
    if phase > 3 {
        return Line::from(Span::styled(
            format!(" {clean} "),
            Style::default().fg(palette.green).add_modifier(Modifier::BOLD),
        ));
    }

    let eligible: Vec<usize> =
        clean.chars().enumerate().filter_map(|(idx, ch)| ch.is_ascii_alphabetic().then_some(idx)).collect();
    if eligible.is_empty() {
        return Line::from(format!(" {clean} "));
    }
    let target = eligible[((state.animation_tick / 23).wrapping_add(salt) as usize) % eligible.len()];
    let glyph = matrix_glyph(state, state.animation_tick.wrapping_add(salt));
    let mut spans = Vec::with_capacity(clean.chars().count() + 2);
    spans.push(Span::raw(" "));
    for (idx, ch) in clean.chars().enumerate() {
        if idx == target {
            spans.push(Span::styled(
                glyph.to_string(),
                Style::default()
                    .fg(if phase == 0 { palette.bright_green } else { palette.green })
                    .add_modifier(if phase <= 1 { Modifier::BOLD | Modifier::REVERSED } else { Modifier::DIM }),
            ));
        } else {
            spans.push(Span::styled(ch.to_string(), Style::default().fg(palette.green).add_modifier(Modifier::BOLD)));
        }
    }
    spans.push(Span::raw(" "));
    Line::from(spans)
}

/// Sparse mutation for explicitly allowlisted, static ornamental copy inside a panel. Callers must
/// never pass snapshot data here. It remains substantially rarer than title motion: three frames
/// in every 59.
fn matrix_body_label(label: &'static str, salt: u64, state: &TuiState, palette: &Palette) -> Vec<Span<'static>> {
    if !state.motion_enabled || is_warning_text(label) {
        return vec![Span::styled(label, Style::default().fg(palette.label))];
    }
    let phase = state.animation_tick.wrapping_add(salt.wrapping_mul(13)) % 59;
    if phase > 2 {
        return vec![Span::styled(label, Style::default().fg(palette.label))];
    }
    let eligible: Vec<usize> =
        label.chars().enumerate().filter_map(|(idx, ch)| ch.is_ascii_alphabetic().then_some(idx)).collect();
    if eligible.is_empty() {
        return vec![Span::styled(label, Style::default().fg(palette.label))];
    }
    let target = eligible[((state.animation_tick / 59).wrapping_add(salt) as usize) % eligible.len()];
    label
        .chars()
        .enumerate()
        .map(|(idx, ch)| {
            if idx == target {
                Span::styled(
                    matrix_glyph(state, state.animation_tick.wrapping_add(salt)).to_string(),
                    Style::default()
                        .fg(if phase == 0 { palette.bright_green } else { palette.green })
                        .add_modifier(if phase == 0 { Modifier::BOLD | Modifier::REVERSED } else { Modifier::DIM }),
                )
            } else {
                Span::styled(ch.to_string(), Style::default().fg(palette.label))
            }
        })
        .collect()
}

fn is_warning_text(value: &str) -> bool {
    let upper = value.to_ascii_uppercase();
    ["WARN", "ALERT", "ERROR", "FAILED", "REJECT"].iter().any(|needle| upper.contains(needle))
}

fn matrix_glyph(state: &TuiState, seed: u64) -> char {
    if state.unicode_glyphs {
        MATRIX_UNICODE[mix64(seed) as usize % MATRIX_UNICODE.len()]
    } else {
        MATRIX_ASCII[mix64(seed) as usize % MATRIX_ASCII.len()]
    }
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn matrix_rail(width: u16, salt: u64, state: &TuiState, palette: &Palette) -> Line<'static> {
    if !state.motion_enabled || width == 0 {
        return Line::raw("");
    }
    let mut spans = Vec::with_capacity(width as usize);
    for col in 0..width {
        let seed = state.animation_tick.wrapping_mul(131).wrapping_add(salt.wrapping_mul(17)).wrapping_add(col as u64);
        let density = mix64(seed) % 9;
        if density <= 1 {
            spans.push(Span::styled(
                matrix_glyph(state, seed).to_string(),
                Style::default()
                    .fg(if density == 0 { palette.rain_head } else { palette.rain })
                    .add_modifier(Modifier::DIM),
            ));
        } else {
            spans.push(Span::raw(" "));
        }
    }
    Line::from(spans)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScanDirection {
    LeftToRight,
    RightToLeft,
}

fn powered_scan_position(state: &TuiState) -> Option<(usize, ScanDirection)> {
    if !state.motion_enabled {
        return None;
    }
    let eligible: Vec<usize> = POWERED_BY
        .chars()
        .enumerate()
        .filter_map(|(index, ch)| ch.is_ascii_alphabetic().then_some(index))
        .collect();
    let scan_ticks = eligible.len() as u64;
    if scan_ticks == 0 {
        return None;
    }
    let cycle = POWER_SCAN_REST_TICKS * 2 + scan_ticks * 2;
    let phase = state.animation_tick % cycle;
    if phase < POWER_SCAN_REST_TICKS {
        return None;
    }
    let phase = phase - POWER_SCAN_REST_TICKS;
    if phase < scan_ticks {
        return Some((eligible[phase as usize], ScanDirection::LeftToRight));
    }
    let phase = phase - scan_ticks;
    if phase < POWER_SCAN_REST_TICKS {
        return None;
    }
    let phase = phase - POWER_SCAN_REST_TICKS;
    Some((eligible[eligible.len() - 1 - phase as usize], ScanDirection::RightToLeft))
}

fn powered_brand_spans(state: &TuiState, palette: &Palette) -> Vec<Span<'static>> {
    let scan = powered_scan_position(state);
    let trail = scan.and_then(|(head, direction)| {
        let eligible: Vec<usize> = POWERED_BY
            .chars()
            .enumerate()
            .filter_map(|(index, ch)| ch.is_ascii_alphabetic().then_some(index))
            .collect();
        let position = eligible.iter().position(|index| *index == head)?;
        match direction {
            ScanDirection::LeftToRight => position.checked_sub(1).map(|index| eligible[index]),
            ScanDirection::RightToLeft => eligible.get(position + 1).copied(),
        }
    });
    POWERED_BY
        .chars()
        .enumerate()
        .map(|(index, ch)| {
            if scan.is_some_and(|(head, _)| head == index) {
                Span::styled(
                    matrix_glyph(state, state.animation_tick.wrapping_add(index as u64)).to_string(),
                    Style::default().fg(palette.bright_green).add_modifier(Modifier::BOLD | Modifier::REVERSED),
                )
            } else if trail == Some(index) {
                Span::styled(ch.to_string(), Style::default().fg(palette.green).add_modifier(Modifier::BOLD))
            } else {
                Span::styled(ch.to_string(), Style::default().fg(palette.label))
            }
        })
        .collect()
}

fn matrix_edge_lines(height: u16, salt: u64, state: &TuiState, palette: &Palette) -> Vec<Line<'static>> {
    (0..height)
        .map(|row| {
            if !state.motion_enabled {
                return Line::raw(" ");
            }
            let seed = state
                .animation_tick
                .wrapping_mul(97)
                .wrapping_add(salt.wrapping_mul(41))
                .wrapping_add(row as u64 * 17);
            let density = mix64(seed) % 5;
            if density <= 1 {
                Line::from(Span::styled(
                    matrix_glyph(state, seed).to_string(),
                    Style::default()
                        .fg(if density == 0 { palette.rain_head } else { palette.rain })
                        .add_modifier(Modifier::DIM),
                ))
            } else {
                Line::raw(" ")
            }
        })
        .collect()
}

fn render_matrix_edges(frame: &mut Frame<'_>, left: Rect, right: Rect, state: &TuiState, palette: &Palette) {
    let style = Style::default().bg(palette.panel);
    frame.render_widget(Paragraph::new(matrix_edge_lines(left.height, 31, state, palette)).style(style), left);
    frame.render_widget(Paragraph::new(matrix_edge_lines(right.height, 47, state, palette)).style(style), right);
}

fn render_core(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &TuiState,
    snapshot: &UiSnapshot,
    palette: &Palette,
    compact: bool,
) {
    let block = panel_block("KERYX // MINING CORE", 1, state, palette, state.page == DashboardPage::Overview);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    let (mining_symbol, mining_label, mining_severity) = mining_status(snapshot.mining.state);
    let (connection_symbol, connection_label, connection_severity) = connection_status(snapshot.connection.state);
    let version = compact_version(snapshot);
    let endpoint = if snapshot.connection.endpoint.trim().is_empty() {
        "--".to_owned()
    } else {
        safe_text(&snapshot.connection.endpoint)
    };
    let mode = match snapshot.connection.mode {
        MiningMode::Pool => "POOL",
        MiningMode::Solo => "SOLO",
    };
    let network = match (snapshot.era.trim(), snapshot.connection.daa_score) {
        ("", None) => "--".to_owned(),
        (era, None) => safe_text(era),
        ("", Some(daa)) => format!("DAA {}", format_integer(daa)),
        (era, Some(daa)) => format!("{} · DAA {}", safe_text(era), format_integer(daa)),
    };
    let connection =
        format!("{connection_symbol} {connection_label}{}", format_latency(snapshot.connection.latency_ms));

    let mut lines = Vec::new();
    if compact || inner.width < 58 {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{mining_symbol} {mining_label}"),
                palette.severity(mining_severity).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" · {mode} · "), Style::default().fg(palette.label)),
            Span::styled(format_rate(snapshot.mining.total_hashrate_hs), Style::default().fg(palette.bright_green)),
            Span::styled(format!(" · {connection}"), palette.severity(connection_severity)),
        ]));
        lines.push(kv_line("ENDPOINT", &endpoint, palette));
        lines.push(Line::from(format!(
            "{} · {} · uptime {} · job {}",
            version,
            display_value(&snapshot.algorithm),
            format_duration(snapshot.uptime_secs),
            format_age(snapshot.connection.last_job_age_secs)
        )));
        match snapshot.connection.mode {
            MiningMode::Pool => lines.push(Line::from(format!(
                "shares A {} / R {} / S {} / P {}",
                snapshot.shares.accepted, snapshot.shares.rejected, snapshot.shares.stale, snapshot.shares.pending
            ))),
            MiningMode::Solo => lines.push(Line::from(format!(
                "blocks found {} / A {} / R {} / P {}",
                snapshot.blocks.found,
                format_optional_count(snapshot.blocks.accepted),
                format_optional_count(snapshot.blocks.rejected),
                snapshot.blocks.pending
            ))),
        }
        lines.push(Line::from(format!(
            "network {network} · diff {}",
            format_difficulty(snapshot.connection.difficulty)
        )));
    } else {
        lines.push(status_version_line(
            &format!("{mining_symbol} {mining_label} · {}", display_value(&snapshot.algorithm)),
            palette.severity(mining_severity),
            &version,
            inner.width,
            palette,
        ));
        lines.push(two_column_line(
            "MODE",
            mode,
            Style::default().fg(palette.bright_green),
            "CONNECTION",
            &connection,
            palette.severity(connection_severity),
            inner.width,
            palette,
        ));
        if !snapshot.connection.message.trim().is_empty() {
            lines.push(Line::from(vec![
                Span::styled("LINK NOTE    ", Style::default().fg(palette.label)),
                Span::styled(safe_text(&snapshot.connection.message), palette.severity(connection_severity)),
            ]));
        }
        lines.push(two_column_line(
            "ENDPOINT",
            &endpoint,
            Style::default().fg(palette.text),
            "LAST JOB",
            &format_age(snapshot.connection.last_job_age_secs),
            Style::default().fg(palette.text),
            inner.width,
            palette,
        ));
        lines.push(two_column_line(
            "RUNTIME",
            &format_duration(snapshot.uptime_secs),
            Style::default().fg(palette.text),
            "FAILOVER",
            &display_value(&snapshot.connection.failover),
            Style::default().fg(palette.text),
            inner.width,
            palette,
        ));
        lines.push(two_column_line(
            "NETWORK",
            &network,
            Style::default().fg(palette.text),
            "DIFFICULTY",
            &format_difficulty(snapshot.connection.difficulty),
            Style::default().fg(palette.text),
            inner.width,
            palette,
        ));
        lines.push(Line::raw(""));
        lines.push(two_column_line(
            "TOTAL HASH",
            &format_rate(snapshot.mining.total_hashrate_hs),
            Style::default().fg(palette.bright_green).add_modifier(Modifier::BOLD),
            "1 MIN AVG",
            &snapshot.mining.average_60s_hs.map(format_rate).unwrap_or_else(|| "--".to_owned()),
            Style::default().fg(palette.green),
            inner.width,
            palette,
        ));
        lines.push(two_column_line(
            "RIG POWER",
            &format_power(snapshot.mining.total_power_w),
            Style::default().fg(palette.text),
            "EFFICIENCY",
            &format_efficiency(snapshot.mining.efficiency_mhs_per_w),
            Style::default().fg(palette.cyan),
            inner.width,
            palette,
        ));

        if inner.height >= 13 {
            let mut history = matrix_body_label("HASH HISTORY", 17, state, palette);
            history.push(Span::raw(" "));
            history.push(Span::styled(
                sparkline_text(
                    &snapshot.mining.hashrate_history_hs,
                    inner.width.saturating_sub(14) as usize,
                    state.unicode_glyphs,
                ),
                Style::default().fg(palette.green),
            ));
            lines.push(Line::from(history));
        }

        match snapshot.connection.mode {
            MiningMode::Pool => {
                lines.push(Line::raw(""));
                lines.push(Line::from(format!(
                    "SHARES       {} accepted · {} rejected · {} stale · {} pending",
                    snapshot.shares.accepted, snapshot.shares.rejected, snapshot.shares.stale, snapshot.shares.pending
                )));
                lines.push(Line::from(format!(
                    "REJECT SPLIT low diff {} · duplicate {} · other {} · last accept {}",
                    snapshot.shares.low_diff,
                    snapshot.shares.duplicate,
                    snapshot.shares.other,
                    format_age(snapshot.shares.last_accepted_age_secs)
                )));
                lines.push(Line::from(format!(
                    "BLOCK CAND. {} found · {} accepted · {} rejected",
                    snapshot.blocks.found,
                    format_optional_count(snapshot.blocks.accepted),
                    format_optional_count(snapshot.blocks.rejected)
                )));
            }
            MiningMode::Solo => {
                lines.push(Line::raw(""));
                lines.push(Line::from(format!(
                    "BLOCKS       {} found · {} accepted · {} rejected · {} pending",
                    snapshot.blocks.found,
                    format_optional_count(snapshot.blocks.accepted),
                    format_optional_count(snapshot.blocks.rejected),
                    snapshot.blocks.pending
                )));
                lines.push(Line::from(format!("LAST BLOCK   {}", format_age(snapshot.blocks.last_accepted_age_secs))));
                if let Some(synced) = snapshot.connection.synced {
                    lines.push(Line::from(format!("NODE         {}", if synced { "synced" } else { "NOT SYNCED" })));
                }
            }
        }
    }

    lines.truncate(inner.height as usize);
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

fn render_rig(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &TuiState,
    snapshot: &UiSnapshot,
    palette: &Palette,
    compact: bool,
) {
    let title =
        format!("NEURAL RIG // {} GPU{}", snapshot.devices.len(), if snapshot.devices.len() == 1 { "" } else { "S" });
    let block = panel_block(&title, 2, state, palette, state.page == DashboardPage::Gpus);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }

    // Reserve actual gutters for the rain so decorative glyphs never overwrite names, rates or
    // warnings. The muted streams deliberately stay inside this one presentation pane.
    let content = if inner.width >= 36 {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(1), Constraint::Min(1), Constraint::Length(1)])
            .split(inner);
        render_matrix_edges(frame, columns[0], columns[2], state, palette);
        columns[1]
    } else {
        inner
    };
    let legend_height = if content.width >= 43 { 1 } else { 2 }.min(content.height);
    let regions = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(legend_height)])
        .split(content);
    let body = regions[0];
    let legend = regions[1];

    let devices = sorted_devices(&snapshot.devices);
    let full_logo = !compact && body.width >= 43 && body.height >= 13;
    let mut lines = Vec::new();
    if full_logo {
        lines.push(matrix_rail(body.width, 7, state, palette));
        lines.extend(LOGO.iter().map(|line| {
            Line::from(Span::styled(
                (*line).to_owned(),
                Style::default().fg(palette.green).add_modifier(Modifier::BOLD),
            ))
            .alignment(Alignment::Center)
        }));
        lines.push(Line::from(powered_brand_spans(state, palette)).alignment(Alignment::Center));
    } else {
        let mut brand = matrix_body_label("K E R Y X", 23, state, palette);
        brand.push(Span::styled("  ·  ", Style::default().fg(palette.rain_head)));
        brand.extend(powered_brand_spans(state, palette));
        lines.push(Line::from(brand).alignment(Alignment::Center));
    }

    let used = lines.len();
    let room = body.height as usize;
    if room > used {
        lines.push(Line::from(Span::styled("─".repeat(body.width as usize), Style::default().fg(palette.border))));
    }

    let available = room.saturating_sub(lines.len());
    let detailed = body.width >= 42 && available >= devices.len().min(3).saturating_mul(2);
    let mut shown = 0usize;
    for device in &devices {
        let card = device_card_lines(device, detailed, body.width, state, palette);
        if lines.len() + card.len() > room {
            break;
        }
        lines.extend(card);
        shown += 1;
    }
    if devices.len() > shown && lines.len() < room {
        lines.push(Line::from(Span::styled(
            format!("+{} more · press 2 for GPU detail", devices.len() - shown),
            Style::default().fg(palette.label),
        )));
    } else if devices.is_empty() && lines.len() < room {
        lines.push(Line::from(Span::styled("Awaiting GPU telemetry.", Style::default().fg(palette.muted))));
    }

    lines.truncate(room);
    frame.render_widget(Paragraph::new(lines), body);
    frame.render_widget(Paragraph::new(gpu_legend_lines(content.width, palette)).alignment(Alignment::Center), legend);
}

fn gpu_legend_lines(width: u16, palette: &Palette) -> Vec<Line<'static>> {
    let nominal = Span::styled("● NOMINAL", palette.severity(Severity::Nominal));
    let paused = Span::styled("◆ PAUSED", palette.severity(Severity::Cool));
    let watch = Span::styled("▲ WATCH", palette.severity(Severity::Watch));
    let alert = Span::styled("■ ALERT", palette.severity(Severity::Alert));
    if width >= 43 {
        vec![Line::from(vec![
            nominal,
            Span::raw("  "),
            paused,
            Span::raw("  "),
            watch,
            Span::raw("  "),
            alert,
        ])]
    } else {
        vec![
            Line::from(vec![nominal, Span::raw("  "), paused]),
            Line::from(vec![watch, Span::raw("  "), alert]),
        ]
    }
}

fn wrap_device_name(value: &str, first_width: usize, continuation_width: usize) -> Vec<String> {
    let clean = safe_text(value);
    let clean = if clean.is_empty() { "unnamed GPU".to_owned() } else { clean };
    let chars: Vec<char> = clean.chars().collect();
    let mut lines = Vec::new();
    let mut cursor = 0usize;
    while cursor < chars.len() {
        while cursor < chars.len() && chars[cursor].is_whitespace() {
            cursor += 1;
        }
        if cursor >= chars.len() {
            break;
        }
        let capacity = if lines.is_empty() { first_width } else { continuation_width }.max(1);
        let mut end = (cursor + capacity).min(chars.len());
        if end < chars.len() {
            if let Some(relative) = chars[cursor..end].iter().rposition(|ch| ch.is_whitespace()) {
                if relative > 0 {
                    end = cursor + relative;
                }
            }
        }
        lines.push(chars[cursor..end].iter().collect());
        cursor = end;
    }
    lines
}

fn append_device_suffix(
    spans: &mut Vec<Span<'static>>,
    device: &DeviceView,
    detailed: bool,
    label: &'static str,
    status_style: Style,
    palette: &Palette,
) {
    spans.push(Span::styled(format!(" · {}", format_rate(device.hashrate_hs)), Style::default().fg(palette.bright_green)));
    if device.inference_host {
        spans.push(Span::styled(" [INF]", Style::default().fg(palette.cyan).add_modifier(Modifier::BOLD)));
    }
    if !detailed {
        spans.push(Span::styled(
            format!("  {}", format_temperature(device.temp_c)),
            palette.severity(temperature_severity(device.temp_c)),
        ));
        spans.push(Span::styled(format!("  {label}"), status_style));
    }
}

fn device_card_lines(
    device: &DeviceView,
    detailed: bool,
    width: u16,
    state: &TuiState,
    palette: &Palette,
) -> Vec<Line<'static>> {
    let (symbol, label, severity) = device_status(device);
    let status_style = device_status_style(device, severity, state, palette);
    let prefix = format!("{symbol} #{} ", device.index);
    let continuation = "    ";
    let first_width = (width as usize).saturating_sub(prefix.chars().count()).max(1);
    let continuation_width = (width as usize).saturating_sub(continuation.chars().count()).max(1);
    let name_lines = wrap_device_name(&device.name, first_width, continuation_width);
    let rate_width = 3 + format_rate(device.hashrate_hs).chars().count();
    let inf_width = usize::from(device.inference_host) * 6;
    let compact_width = if detailed {
        0
    } else {
        2 + format_temperature(device.temp_c).chars().count() + 2 + label.chars().count()
    };
    let suffix_width = rate_width + inf_width + compact_width;
    let last_capacity = if name_lines.len() == 1 { first_width } else { continuation_width };
    let suffix_needs_line = name_lines.last().map_or(false, |name| name.chars().count() + suffix_width > last_capacity);

    let mut lines = Vec::with_capacity(name_lines.len() + usize::from(suffix_needs_line) + usize::from(detailed));
    for (index, name) in name_lines.iter().enumerate() {
        let mut spans = if index == 0 {
            vec![
                Span::styled(format!("{symbol} "), status_style.add_modifier(Modifier::BOLD)),
                Span::styled(format!("#{} ", device.index), Style::default().fg(palette.label)),
            ]
        } else {
            vec![Span::raw(continuation)]
        };
        spans.push(Span::styled(name.clone(), Style::default().fg(palette.text)));
        if index + 1 == name_lines.len() && !suffix_needs_line {
            append_device_suffix(&mut spans, device, detailed, label, status_style, palette);
        }
        lines.push(Line::from(spans));
    }
    if suffix_needs_line {
        let mut spans = vec![Span::raw(continuation)];
        append_device_suffix(&mut spans, device, detailed, label, status_style, palette);
        lines.push(Line::from(spans));
    }
    if detailed {
        let core_sev = if device.activity == DeviceActivity::Mining {
            clock_severity(device.core_mhz, device.baseline_core_mhz)
        } else {
            Severity::Cool
        };
        let mem_sev = if device.activity == DeviceActivity::Mining {
            clock_severity(device.mem_mhz, device.baseline_mem_mhz)
        } else {
            Severity::Cool
        };
        let mut telemetry = vec![
            Span::raw("    "),
            Span::styled(format_temperature(device.temp_c), palette.severity(temperature_severity(device.temp_c))),
            Span::styled(
                format!("  {}  {}  ", format_fan(device.fan_pct), format_power(device.power_w)),
                Style::default().fg(palette.text),
            ),
            Span::styled(format!("C{}  ", format_clock(device.core_mhz)), palette.severity(core_sev)),
        ];
        if width >= 53 {
            telemetry.push(Span::styled(
                format!("M{}  ", format_clock(device.mem_mhz)),
                palette.severity(mem_sev),
            ));
            telemetry.push(Span::styled(
                format!("{}  ", format_vram(device.vram_used_mb, device.vram_total_mb)),
                Style::default().fg(palette.label),
            ));
        }
        telemetry.push(Span::styled(label, status_style));
        lines.push(Line::from(telemetry));
    }
    lines
}

fn render_overview(frame: &mut Frame<'_>, area: Rect, state: &TuiState, snapshot: &UiSnapshot, palette: &Palette) {
    let block =
        panel_block("OPoI // NEURAL FABRIC + LIVE EVENTS", 3, state, palette, state.page == DashboardPage::Overview);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let mut lines = inference_summary_lines(snapshot, palette, inner.width, false);
    if snapshot.connection.mode == MiningMode::Solo {
        lines.push(escrow_summary_line(&snapshot.escrow, palette));
        if service_bond_non_nominal(&snapshot.service_bond) {
            lines.push(service_bond_summary_line(&snapshot.service_bond, palette));
        }
    }
    if lines.len() < inner.height as usize {
        lines.push(Line::from(Span::styled("─".repeat(inner.width as usize), Style::default().fg(palette.border))));
    }
    let event_room = inner.height as usize - lines.len().min(inner.height as usize);
    lines.extend(event_lines(&snapshot.events, state.log_scroll, event_room, palette));
    lines.truncate(inner.height as usize);
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

fn inference_summary_lines(snapshot: &UiSnapshot, palette: &Palette, width: u16, detailed: bool) -> Vec<Line<'static>> {
    let inference = &snapshot.inference;
    let (symbol, label, severity) = inference_status(inference.state);
    let host = match (inference.gpu_index, inference.gpu_route_count) {
        (Some(gpu), _) => format!("GPU #{gpu}"),
        (None, 0) => "no GPU route".to_owned(),
        (None, 1) => "GPU route".to_owned(),
        (None, count) => format!("{count} GPU routes"),
    };
    let queue = if inference.queue_capacity > 0 {
        format!("{}/{}", inference.queue_depth, inference.queue_capacity)
    } else {
        inference.queue_depth.to_string()
    };
    let pow = if snapshot.mining.state == MiningState::InferencePaused {
        ("◆ PAUSED FOR AI", Severity::Cool)
    } else {
        ("● ACTIVE", Severity::Nominal)
    };

    let mut lines = vec![Line::from(vec![
        Span::styled(format!("{symbol} {label}"), palette.severity(severity).add_modifier(Modifier::BOLD)),
        Span::styled(format!(" · {host}"), Style::default().fg(palette.text)),
        Span::styled(
            format!(" · {} · {}", display_value(&inference.model), display_value(&inference.backend)),
            Style::default().fg(palette.label),
        ),
        Span::styled(format!("   POW {}", pow.0), palette.severity(pow.1)),
        Span::styled(format!("   QUEUE {queue} · ACTIVE {}", inference.active), Style::default().fg(palette.text)),
    ])];
    lines.push(Line::from(format!(
        "REQUESTED {} · PREPARED {} · SERVED {} · DELIVERED {} · FAILED {} · BUSY {}",
        inference.requested,
        inference.prepared,
        inference.served,
        inference.delivered,
        inference.failed,
        inference.busy
    )));
    lines.push(Line::from(format!(
        "LAST {} / {} tok · P95 {} · TIER {} · SELF-TEST {} · AI PAUSE {} total",
        format_millis(inference.last_latency_ms),
        inference.last_tokens.map(|v| v.to_string()).unwrap_or_else(|| "--".to_owned()),
        format_millis(inference.p95_latency_ms),
        display_value(&inference.tier),
        format_self_test(inference.self_test_ok),
        format_duration(inference.pow_pause_total_secs)
    )));
    if detailed {
        lines.push(Line::from(format!(
            "MODEL ID {} · STATUS {}",
            fit_text(&display_value(&inference.model_id), (width as usize / 2).max(8)),
            display_value(&inference.status)
        )));
    }
    lines
}

fn escrow_summary_line(escrow: &EscrowView, palette: &Palette) -> Line<'static> {
    let (symbol, label, severity) = escrow_status(escrow);
    Line::from(vec![
        Span::styled("ESCROW ", Style::default().fg(palette.label)),
        Span::styled(format!("{symbol} {label}"), palette.severity(severity).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(
                " · outputs A {} / F {} / P {} · claimable {} · claimed {} · last success {}",
                escrow.claims_accepted,
                escrow.claims_failed,
                escrow.claims_pending,
                display_value(&escrow.claimable_amount),
                display_value(&escrow.claimed_amount),
                format_age(escrow.last_success_age_secs)
            ),
            Style::default().fg(palette.text),
        ),
    ])
}

fn service_bond_summary_line(bond: &ServiceBondView, palette: &Palette) -> Line<'static> {
    let (symbol, label, severity) = service_bond_status(bond);
    Line::from(vec![
        Span::styled("SERVICE BOND ", Style::default().fg(palette.label)),
        Span::styled(format!("{symbol} {label}"), palette.severity(severity).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(
                " · misses {} · burned {} / {} · heartbeat {} · suspended until {}",
                bond.consecutive_misses,
                bond.burned_claims,
                display_value(&bond.burned_amount),
                format_age(bond.last_heartbeat_age_secs),
                bond.suspended_until_daa.map(format_integer).unwrap_or_else(|| "--".to_owned())
            ),
            Style::default().fg(palette.text),
        ),
    ])
}

fn render_gpu_detail(frame: &mut Frame<'_>, area: Rect, state: &TuiState, snapshot: &UiSnapshot, palette: &Palette) {
    let block = panel_block("GPU TELEMETRY // STABLE DEVICE ORDER", 5, state, palette, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let mut lines = Vec::new();
    if inner.width >= 125 {
        lines.push(Line::from(Span::styled(
            " #  GPU / BACKEND                 HASH        TEMP HOT  FAN  POWER   CORE   MEM    VRAM         EFF       A/R      STATE",
            Style::default().fg(palette.label).add_modifier(Modifier::BOLD),
        )));
    } else if inner.width >= 84 {
        lines.push(Line::from(Span::styled(
            " #  GPU / BACKEND              HASH       TEMP FAN POWER CORE   A/R    STATE",
            Style::default().fg(palette.label).add_modifier(Modifier::BOLD),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            " #  GPU                 HASH       TEMP  A/R   STATE",
            Style::default().fg(palette.label).add_modifier(Modifier::BOLD),
        )));
    }

    for device in sorted_devices(&snapshot.devices) {
        let (symbol, label, severity) = device_status(device);
        let status_style = device_status_style(device, severity, state, palette);
        let temp_style = palette.severity(temperature_severity(device.temp_c));
        let core_style = if device.activity == DeviceActivity::Mining {
            palette.severity(clock_severity(device.core_mhz, device.baseline_core_mhz))
        } else {
            palette.severity(Severity::Cool)
        };
        let name = format!("{}{}", safe_text(&device.name), if device.inference_host { " [INF]" } else { "" });
        if inner.width >= 125 {
            lines.push(Line::from(vec![
                Span::styled(format!("{symbol} "), status_style),
                Span::styled(format!("{:<2} ", device.index), Style::default().fg(palette.label)),
                Span::styled(
                    fit_pad(&format!("{} / {}", name, safe_text(&device.backend)), 28),
                    Style::default().fg(palette.text),
                ),
                Span::styled(fit_pad(&format_rate(device.hashrate_hs), 12), Style::default().fg(palette.bright_green)),
                Span::styled(fit_pad(&format_temperature(device.temp_c), 5), temp_style),
                Span::styled(
                    fit_pad(&format_temperature(device.hotspot_c), 5),
                    palette.severity(hotspot_severity(device.hotspot_c)),
                ),
                Span::styled(fit_pad(&format_fan(device.fan_pct), 5), Style::default().fg(palette.text)),
                Span::styled(fit_pad(&format_power(device.power_w), 8), Style::default().fg(palette.text)),
                Span::styled(fit_pad(&format_clock(device.core_mhz), 7), core_style),
                Span::styled(
                    fit_pad(&format_clock(device.mem_mhz), 7),
                    palette.severity(clock_severity(device.mem_mhz, device.baseline_mem_mhz)),
                ),
                Span::styled(
                    fit_pad(&format_vram(device.vram_used_mb, device.vram_total_mb), 13),
                    Style::default().fg(palette.label),
                ),
                Span::styled(
                    fit_pad(&format_efficiency(device.efficiency_mhs_per_w), 10),
                    Style::default().fg(palette.cyan),
                ),
                Span::styled(
                    fit_pad(&format!("{}/{}", device.accepted, device.rejected), 9),
                    Style::default().fg(palette.text),
                ),
                Span::styled(label, status_style),
            ]));
        } else if inner.width >= 84 {
            lines.push(Line::from(vec![
                Span::styled(format!("{symbol} "), status_style),
                Span::styled(format!("{:<2} ", device.index), Style::default().fg(palette.label)),
                Span::styled(fit_pad(&name, 25), Style::default().fg(palette.text)),
                Span::styled(fit_pad(&format_rate(device.hashrate_hs), 11), Style::default().fg(palette.bright_green)),
                Span::styled(fit_pad(&format_temperature(device.temp_c), 5), temp_style),
                Span::styled(fit_pad(&format_fan(device.fan_pct), 4), Style::default().fg(palette.text)),
                Span::styled(fit_pad(&format_power(device.power_w), 7), Style::default().fg(palette.text)),
                Span::styled(fit_pad(&format_clock(device.core_mhz), 7), core_style),
                Span::styled(
                    fit_pad(&format!("{}/{}", device.accepted, device.rejected), 7),
                    Style::default().fg(palette.text),
                ),
                Span::styled(label, status_style),
            ]));
        } else {
            lines.push(Line::from(vec![
                Span::styled(format!("{symbol} "), status_style),
                Span::styled(format!("{:<2} ", device.index), Style::default().fg(palette.label)),
                Span::styled(fit_pad(&name, 20), Style::default().fg(palette.text)),
                Span::styled(fit_pad(&format_rate(device.hashrate_hs), 11), Style::default().fg(palette.bright_green)),
                Span::styled(fit_pad(&format_temperature(device.temp_c), 6), temp_style),
                Span::styled(
                    fit_pad(&format!("{}/{}", device.accepted, device.rejected), 6),
                    Style::default().fg(palette.text),
                ),
                Span::styled(label, status_style),
            ]));
        }
        if let Some(reason) = device.throttle_reason.as_deref().filter(|reason| !reason.trim().is_empty()) {
            if lines.len() < inner.height as usize {
                lines.push(Line::from(vec![
                    Span::styled("    ALERT: ", status_style.add_modifier(Modifier::BOLD)),
                    Span::styled(safe_text(reason), Style::default().fg(palette.red)),
                ]));
            }
        }
        if lines.len() >= inner.height as usize {
            break;
        }
    }
    if snapshot.devices.is_empty() {
        lines.push(Line::from(Span::styled(
            "No mining devices have published telemetry yet.",
            Style::default().fg(palette.muted),
        )));
    }
    lines.truncate(inner.height as usize);
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_inference_detail(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &TuiState,
    snapshot: &UiSnapshot,
    palette: &Palette,
) {
    let block = panel_block("OPoI + SOLO ESCROW // DELIVERY HEALTH", 6, state, palette, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    let mut lines = inference_summary_lines(snapshot, palette, inner.width, true);
    if !snapshot.inference.status.trim().is_empty() {
        let (_, _, severity) = inference_status(snapshot.inference.state);
        lines.push(Line::from(vec![
            Span::styled("INFERENCE STATUS  ", Style::default().fg(palette.label)),
            Span::styled(safe_text(&snapshot.inference.status), palette.severity(severity)),
        ]));
    }

    if snapshot.connection.mode == MiningMode::Solo {
        lines.push(Line::from(Span::styled("─ SOLO ESCROW ─", Style::default().fg(palette.border_focus))));
        lines.push(escrow_summary_line(&snapshot.escrow, palette));
        lines.push(Line::from(format!(
            "LAST ATTEMPT {} · LAST SUCCESS {} · STATUS {}",
            format_age(snapshot.escrow.last_attempt_age_secs),
            format_age(snapshot.escrow.last_success_age_secs),
            display_value(&snapshot.escrow.status)
        )));
        if !snapshot.escrow.message.trim().is_empty() {
            let (_, _, severity) = escrow_status(&snapshot.escrow);
            lines.push(Line::from(Span::styled(safe_text(&snapshot.escrow.message), palette.severity(severity))));
        }
        lines.push(Line::from(Span::styled("─ SERVICE BOND ─", Style::default().fg(palette.border_focus))));
        lines.push(service_bond_summary_line(&snapshot.service_bond, palette));
        lines.push(Line::from(format!(
            "LAST STRIKE DAA {} · LAST FAILURE {} · HEARTBEAT {}",
            snapshot.service_bond.last_strike_daa.map(format_integer).unwrap_or_else(|| "--".to_owned()),
            format_age(snapshot.service_bond.last_failure_age_secs),
            if snapshot.service_bond.heartbeat_alive { "alive" } else { "NOT ALIVE" }
        )));
    }

    if lines.len() < inner.height as usize {
        lines.push(Line::from(Span::styled(
            "─ RECENT INFERENCE / ESCROW EVENTS ─",
            Style::default().fg(palette.border),
        )));
        let remaining = inner.height as usize - lines.len();
        let filtered: Vec<_> = snapshot
            .events
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    UiEventKind::InferenceOk
                        | UiEventKind::InferenceFailed
                        | UiEventKind::Escrow
                        | UiEventKind::Error
                        | UiEventKind::Info
                )
            })
            .cloned()
            .collect();
        lines.extend(event_lines(&filtered, state.log_scroll, remaining, palette));
    }
    lines.truncate(inner.height as usize);
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

fn render_expanded_logs(frame: &mut Frame<'_>, area: Rect, state: &TuiState, snapshot: &UiSnapshot, palette: &Palette) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(4), Constraint::Length(1)])
        .split(area);
    let block = panel_block("LIVE EVENT STREAM // READ-ONLY", 8, state, palette, true);
    let inner = block.inner(rows[0]);
    frame.render_widget(block, rows[0]);
    let lines = event_lines(&snapshot.events, state.log_scroll, inner.height as usize, palette);
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
    render_footer(frame, rows[1], state, palette);
}

fn event_lines(events: &[UiEvent], scroll: usize, max_lines: usize, palette: &Palette) -> Vec<Line<'static>> {
    if max_lines == 0 {
        return Vec::new();
    }
    if events.is_empty() {
        return vec![Line::from(Span::styled("No events yet.", Style::default().fg(palette.muted)))];
    }
    let effective_scroll = scroll.min(events.len().saturating_sub(1));
    let end = events.len().saturating_sub(effective_scroll);
    let start = end.saturating_sub(max_lines);
    events[start..end]
        .iter()
        .map(|event| {
            let (tag, severity) = event_visual(event.kind);
            Line::from(vec![
                Span::styled(fit_pad(&safe_text(&event.timestamp), 10), Style::default().fg(palette.muted)),
                Span::styled(fit_pad(tag, 12), palette.severity(severity).add_modifier(Modifier::BOLD)),
                Span::styled(safe_text(&event.message), Style::default().fg(palette.text)),
            ])
        })
        .collect()
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, state: &TuiState, palette: &Palette) {
    let motion = if state.motion_enabled { "ON" } else { "OFF" };
    let page = match state.page {
        DashboardPage::Overview => "OVERVIEW",
        DashboardPage::Gpus => "GPUs",
        DashboardPage::Inference => "INFERENCE",
    };
    let text = if state.quit_armed {
        "QUIT ARMED: press Q again for graceful shutdown · Esc cancels".to_owned()
    } else if state.logs_expanded {
        format!("PAGE {page}  [L] DASHBOARD  [UP/DOWN] SCROLL  [M] MOTION {motion}  [?] HELP  [Q Q] QUIT")
    } else {
        format!(
            "[1] OVERVIEW  [2] GPUs  [3] INFERENCE  ·  PAGE {page}  [L] LOGS  [M] MOTION {motion}  [?] HELP  [Q Q] QUIT"
        )
    };
    let style = if state.quit_armed {
        Style::default().fg(palette.yellow).bg(palette.panel).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(palette.label).bg(palette.panel)
    };
    frame.render_widget(
        Paragraph::new(fit_text(&text, area.width as usize)).style(style).alignment(Alignment::Center),
        area,
    );
}

fn render_help(frame: &mut Frame<'_>, area: Rect, state: &TuiState, palette: &Palette) {
    let popup = centered_rect(68.min(area.width.saturating_sub(2)), 15.min(area.height.saturating_sub(2)), area);
    frame.render_widget(Clear, popup);
    let block = panel_block("KEYS // OBSERVER CONTROLS ONLY", 9, state, palette, true);
    let lines = vec![
        Line::from("1 / 2 / 3     Overview / GPU telemetry / inference + escrow"),
        Line::from("Left / Right  Change page"),
        Line::from("Up / Down     Scroll event history"),
        Line::from("PageUp/Down   Scroll by ten events; End returns to live tail"),
        Line::from("L             Expand or restore the event stream"),
        Line::from("M             Toggle sparse Matrix title/gutter motion"),
        Line::from("?             Toggle this help"),
        Line::from("Q Q           Request graceful miner shutdown"),
        Line::from("Esc           Cancel quit/help/expanded logs"),
        Line::raw(""),
        Line::from(Span::styled(
            "No UI key changes clocks, GPUs, mining, inference or escrow.",
            Style::default().fg(palette.cyan).add_modifier(Modifier::BOLD),
        )),
    ];
    frame.render_widget(Paragraph::new(lines).block(block).wrap(Wrap { trim: true }), popup);
}

fn render_quit_confirmation(frame: &mut Frame<'_>, area: Rect, state: &TuiState, palette: &Palette) {
    let popup = centered_rect(62.min(area.width.saturating_sub(2)), 7.min(area.height.saturating_sub(2)), area);
    frame.render_widget(Clear, popup);
    let block = panel_block("CONFIRM GRACEFUL SHUTDOWN", 10, state, palette, true)
        .border_style(Style::default().fg(palette.yellow));
    let text = vec![
        Line::raw(""),
        Line::from(Span::styled(
            "Press Q again to ask the host to stop the miner gracefully.",
            Style::default().fg(palette.yellow).add_modifier(Modifier::BOLD),
        )),
        Line::from("Press Esc to keep mining."),
    ];
    frame.render_widget(Paragraph::new(text).alignment(Alignment::Center).block(block), popup);
}

fn render_tiny(frame: &mut Frame<'_>, area: Rect, _state: &TuiState, snapshot: &UiSnapshot, palette: &Palette) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(palette.border))
        .style(Style::default().fg(palette.text).bg(palette.panel))
        .title(Span::styled(" KERYX // COMPACT ", Style::default().fg(palette.green).add_modifier(Modifier::BOLD)));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let (_, mining_label, mining_severity) = mining_status(snapshot.mining.state);
    let mode = if snapshot.connection.mode == MiningMode::Pool { "POOL" } else { "SOLO" };
    let result = if snapshot.connection.mode == MiningMode::Pool {
        format!("shares A/R {}/{}", snapshot.shares.accepted, snapshot.shares.rejected)
    } else {
        format!(
            "blocks F/A/R {}/{}/{}",
            snapshot.blocks.found,
            format_optional_count(snapshot.blocks.accepted),
            format_optional_count(snapshot.blocks.rejected)
        )
    };
    let (inf_symbol, inf_label, inf_severity) = inference_status(snapshot.inference.state);
    let lines = vec![
        Line::from(vec![
            Span::styled(mining_label, palette.severity(mining_severity).add_modifier(Modifier::BOLD)),
            Span::raw(format!(" · {mode} · {}", compact_version(snapshot))),
        ]),
        Line::from(format!(
            "{} · uptime {}",
            format_rate(snapshot.mining.total_hashrate_hs),
            format_duration(snapshot.uptime_secs)
        )),
        Line::from(result),
        Line::from(vec![
            Span::styled(format!("{inf_symbol} {inf_label}"), palette.severity(inf_severity)),
            Span::raw(format!(" · served {} / failed {}", snapshot.inference.served, snapshot.inference.failed)),
        ]),
        Line::from(Span::styled("Resize to at least 70x20 for dashboard pages.", Style::default().fg(palette.muted))),
    ];
    frame.render_widget(Paragraph::new(lines).wrap(Wrap { trim: true }), inner);
}

fn render_too_small(frame: &mut Frame<'_>, area: Rect, snapshot: &UiSnapshot, palette: &Palette) {
    let text = format!("KERYX {} {}", compact_version(snapshot), format_rate(snapshot.mining.total_hashrate_hs));
    frame.render_widget(Paragraph::new(fit_text(&text, area.width as usize)).style(palette.canvas()), area);
}

fn mining_status(state: MiningState) -> (&'static str, &'static str, Severity) {
    match state {
        MiningState::Preparing => ("◆", "PREPARING", Severity::Cool),
        MiningState::Mining => ("●", "MINING", Severity::Nominal),
        MiningState::InferencePaused => ("◆", "AI PAUSE", Severity::Cool),
        MiningState::Degraded => ("▲", "DEGRADED", Severity::Watch),
        MiningState::Stopped => ("■", "STOPPED", Severity::Alert),
    }
}

fn connection_status(state: ConnectionState) -> (&'static str, &'static str, Severity) {
    match state {
        ConnectionState::Connecting => ("◆", "CONNECTING", Severity::Cool),
        ConnectionState::Connected => ("●", "CONNECTED", Severity::Nominal),
        ConnectionState::Failover => ("▲", "FAILOVER", Severity::Watch),
        ConnectionState::Offline => ("■", "OFFLINE", Severity::Alert),
    }
}

fn inference_status(state: InferenceState) -> (&'static str, &'static str, Severity) {
    match state {
        InferenceState::Unavailable => ("■", "UNAVAILABLE", Severity::Alert),
        InferenceState::Preparing => ("◆", "PREPARING", Severity::Cool),
        InferenceState::Ready => ("●", "READY", Severity::Nominal),
        InferenceState::Serving => ("◆", "SERVING", Severity::Cool),
        InferenceState::Degraded => ("▲", "DEGRADED", Severity::Watch),
    }
}

fn escrow_status(escrow: &EscrowView) -> (&'static str, &'static str, Severity) {
    if !escrow.enabled {
        ("○", "DISABLED", Severity::Unknown)
    } else if escrow.status.eq_ignore_ascii_case("degraded") {
        ("■", "DEGRADED", Severity::Alert)
    } else if escrow.status.eq_ignore_ascii_case("held") {
        ("▲", "CLAIMS HELD", Severity::Watch)
    } else if escrow.status.eq_ignore_ascii_case("validating") {
        ("◆", "VALIDATING", Severity::Cool)
    } else if !escrow.alive && escrow.status.eq_ignore_ascii_case("waiting") {
        // The claim worker receives its first heartbeat from the solo block stream. Avoid a false
        // red alarm during that short, expected startup window; a previously-live worker whose
        // heartbeat expires has a non-waiting status and still escalates to WORKER DOWN.
        ("◆", "STARTING", Severity::Cool)
    } else if !escrow.alive {
        ("■", "WORKER DOWN", Severity::Alert)
    } else if escrow.claiming {
        ("◆", "CLAIMING", Severity::Cool)
    } else if escrow.claims_failed > 0 && escrow.claims_accepted == 0 {
        ("▲", "CHECK", Severity::Watch)
    } else {
        ("●", "ALIVE", Severity::Nominal)
    }
}

fn service_bond_status(bond: &ServiceBondView) -> (&'static str, &'static str, Severity) {
    if bond.suspended_until_daa.is_some() {
        ("■", "SUSPENDED", Severity::Alert)
    } else if bond.consecutive_misses != 0 || bond.burned_claims != 0 {
        ("▲", "AT RISK", Severity::Watch)
    } else if !bond.available {
        if bond.last_failure_age_secs.is_some() {
            ("▲", "UNAVAILABLE", Severity::Watch)
        } else {
            ("◆", "PENDING", Severity::Cool)
        }
    } else if !bond.heartbeat_alive {
        if bond.last_heartbeat_age_secs.is_none() && bond.last_failure_age_secs.is_none() {
            ("◆", "STARTING", Severity::Cool)
        } else {
            ("▲", "HEARTBEAT LOST", Severity::Watch)
        }
    } else {
        ("●", "HEALTHY", Severity::Nominal)
    }
}

fn service_bond_non_nominal(bond: &ServiceBondView) -> bool {
    service_bond_status(bond).2 != Severity::Nominal
}

fn device_status(device: &DeviceView) -> (&'static str, &'static str, Severity) {
    match device.activity {
        DeviceActivity::Preparing => {
            let severity = device_severity(device);
            (severity_symbol(severity), "PREPARING", severity)
        }
        DeviceActivity::Inference => {
            let severity = device_severity(device);
            (severity_symbol(severity), "INFERENCE", severity)
        }
        DeviceActivity::Paused => {
            let severity = device_severity(device);
            (severity_symbol(severity), "PAUSED", severity)
        }
        DeviceActivity::Stalled => ("■", "STALLED", Severity::Alert),
        DeviceActivity::Offline => ("■", "OFFLINE", Severity::Alert),
        DeviceActivity::Mining => match device_severity(device) {
            Severity::Cool => ("◆", "COOL", Severity::Cool),
            Severity::Nominal => ("●", "NOMINAL", Severity::Nominal),
            Severity::Watch => ("▲", "WATCH", Severity::Watch),
            Severity::Alert => ("■", "ALERT", Severity::Alert),
            Severity::Unknown => ("○", "MINING", Severity::Unknown),
        },
    }
}

fn event_visual(kind: UiEventKind) -> (&'static str, Severity) {
    match kind {
        UiEventKind::Info => ("[INFO]", Severity::Unknown),
        UiEventKind::Job => ("[JOB]", Severity::Cool),
        UiEventKind::ShareAccepted => ("[SHARE/OK]", Severity::Nominal),
        UiEventKind::ShareRejected => ("[SHARE/NO]", Severity::Watch),
        UiEventKind::BlockFound => ("[BLOCK]", Severity::Cool),
        UiEventKind::BlockAccepted => ("[BLOCK/OK]", Severity::Nominal),
        UiEventKind::BlockRejected => ("[BLOCK/NO]", Severity::Alert),
        UiEventKind::InferenceOk => ("[AI/OK]", Severity::Cool),
        UiEventKind::InferenceFailed => ("[AI/FAIL]", Severity::Watch),
        UiEventKind::Escrow => ("[ESCROW]", Severity::Cool),
        UiEventKind::HealthWarn => ("[HEALTH]", Severity::Watch),
        UiEventKind::Error => ("[ERROR]", Severity::Alert),
    }
}

fn severity_symbol(severity: Severity) -> &'static str {
    match severity {
        Severity::Cool => "◆",
        Severity::Nominal => "●",
        Severity::Watch => "▲",
        Severity::Alert => "■",
        Severity::Unknown => "○",
    }
}

fn compact_version(snapshot: &UiSnapshot) -> String {
    // Keep the primary operator-facing VERSION stable and copyable. Build/commit metadata belongs
    // in diagnostics, never in this field (and the decorative panel title is not the binary name).
    format!("keryx-miner-supr {}", display_value(&snapshot.version))
}

fn two_column_line(
    left_label: &str,
    left_value: &str,
    left_style: Style,
    right_label: &str,
    right_value: &str,
    right_style: Style,
    width: u16,
    palette: &Palette,
) -> Line<'static> {
    let half = (width as usize / 2).max(1);
    let left_label_width = 13.min(half.saturating_sub(1));
    let left_value_width = half.saturating_sub(left_label_width);
    let right_width = width as usize - half;
    let right_label_width = 13.min(right_width.saturating_sub(1));
    let right_value_width = right_width.saturating_sub(right_label_width);
    Line::from(vec![
        Span::styled(fit_pad(left_label, left_label_width), Style::default().fg(palette.label)),
        Span::styled(fit_pad(&safe_text(left_value), left_value_width), left_style),
        Span::styled(fit_pad(right_label, right_label_width), Style::default().fg(palette.label)),
        Span::styled(fit_pad(&safe_text(right_value), right_value_width), right_style),
    ])
}

fn status_version_line(
    status: &str,
    status_style: Style,
    version: &str,
    width: u16,
    palette: &Palette,
) -> Line<'static> {
    let version = safe_text(version);
    let version_width = version.chars().count();
    let right_label_width = 13usize;
    let right_width = (right_label_width + version_width).min(width as usize);
    let left_width = (width as usize).saturating_sub(right_width);
    let left_label_width = 13.min(left_width);
    let left_value_width = left_width.saturating_sub(left_label_width);
    Line::from(vec![
        Span::styled(fit_pad("STATUS", left_label_width), Style::default().fg(palette.label)),
        Span::styled(fit_pad(&safe_text(status), left_value_width), status_style),
        Span::styled(fit_pad("VERSION", right_label_width.min(right_width)), Style::default().fg(palette.label)),
        // `render_core` uses this layout only when enough columns exist. Never feed the canonical
        // primary version through the ellipsis helper.
        Span::styled(version, Style::default().fg(palette.bright_green)),
    ])
}

fn kv_line(label: &str, value: &str, palette: &Palette) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<10}"), Style::default().fg(palette.label)),
        Span::styled(safe_text(value), Style::default().fg(palette.text)),
    ])
}

fn fit_text(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let clean = safe_text(value);
    let count = clean.chars().count();
    if count <= width {
        clean
    } else if width <= 3 {
        clean.chars().take(width).collect()
    } else {
        let mut out: String = clean.chars().take(width - 1).collect();
        out.push('…');
        out
    }
}

fn fit_pad(value: &str, width: usize) -> String {
    let fitted = fit_text(value, width);
    let padding = width.saturating_sub(fitted.chars().count());
    format!("{fitted}{}", " ".repeat(padding))
}

fn display_value(value: &str) -> String {
    let clean = safe_text(value);
    if clean.is_empty() {
        "--".to_owned()
    } else {
        clean
    }
}

fn format_rate(hashes_per_second: f64) -> String {
    let value = if hashes_per_second.is_finite() && hashes_per_second >= 0.0 { hashes_per_second } else { 0.0 };
    if value >= 1_000_000_000.0 {
        format!("{:.3} GH/s", value / 1_000_000_000.0)
    } else if value >= 1_000_000.0 {
        format!("{:.3} MH/s", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.2} kH/s", value / 1_000.0)
    } else {
        format!("{value:.0} H/s")
    }
}

fn format_power(power_w: Option<f64>) -> String {
    power_w
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| format!("{value:.1} W"))
        .unwrap_or_else(|| "--".to_owned())
}

fn format_efficiency(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| if value < 0.01 { format!("{:.2} kH/W", value * 1_000.0) } else { format!("{value:.3} MH/W") })
        .unwrap_or_else(|| "--".to_owned())
}

fn format_temperature(value: Option<u32>) -> String {
    value.map(|v| format!("{v}°C")).unwrap_or_else(|| "--".to_owned())
}

fn format_fan(value: Option<u32>) -> String {
    value.map(|v| format!("{v}%")).unwrap_or_else(|| "--".to_owned())
}

fn format_clock(value: Option<u32>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "--".to_owned())
}

fn format_vram(used_mb: Option<u64>, total_mb: Option<u64>) -> String {
    match (used_mb, total_mb) {
        (Some(used), Some(total)) => format!("{:.1}/{:.1}G", used as f64 / 1024.0, total as f64 / 1024.0),
        _ => "--".to_owned(),
    }
}

fn format_duration(total_secs: u64) -> String {
    let days = total_secs / 86_400;
    let hours = (total_secs % 86_400) / 3_600;
    let minutes = (total_secs % 3_600) / 60;
    let seconds = total_secs % 60;
    if days > 0 {
        format!("{days}d {hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    }
}

fn format_age(age_secs: Option<u64>) -> String {
    match age_secs {
        None => "--".to_owned(),
        Some(0) => "now".to_owned(),
        Some(seconds) if seconds < 60 => format!("{seconds}s ago"),
        Some(seconds) if seconds < 3_600 => format!("{}m {:02}s ago", seconds / 60, seconds % 60),
        Some(seconds) if seconds < 86_400 => format!("{}h {:02}m ago", seconds / 3_600, (seconds % 3_600) / 60),
        Some(seconds) => format!("{}d {:02}h ago", seconds / 86_400, (seconds % 86_400) / 3_600),
    }
}

fn format_latency(latency_ms: Option<u64>) -> String {
    latency_ms.map(|ms| format!(" · {ms} ms")).unwrap_or_default()
}

fn format_millis(value: Option<u64>) -> String {
    value.map(|ms| format!("{ms} ms")).unwrap_or_else(|| "--".to_owned())
}

fn format_difficulty(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "--".to_owned())
}

fn format_optional_count(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "--".to_owned())
}

fn format_self_test(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "passed",
        Some(false) => "FAILED",
        None => "--",
    }
}

fn format_integer(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (idx, ch) in digits.chars().enumerate() {
        if idx > 0 && (digits.len() - idx) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

fn sparkline_text(values: &[f64], width: usize, unicode: bool) -> String {
    if width == 0 || values.is_empty() {
        return "--".to_owned();
    }
    let start = values.len().saturating_sub(width);
    let values = &values[start..];
    let max = values.iter().copied().filter(|value| value.is_finite() && *value >= 0.0).fold(0.0_f64, f64::max);
    if max <= f64::EPSILON {
        return if unicode { "▁" } else { "_" }.repeat(values.len());
    }
    let bars: &[char] = if unicode {
        &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
    } else {
        &['.', ':', '-', '=', '+', '*', '#', '@']
    };
    values
        .iter()
        .map(|value| {
            let value = if value.is_finite() && *value >= 0.0 { *value } else { 0.0 };
            let idx = ((value / max) * (bars.len() - 1) as f64).round() as usize;
            bars[idx.min(bars.len() - 1)]
        })
        .collect()
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width);
    let height = height.min(area.height);
    Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::KeyEvent;
    use ratatui::{backend::TestBackend, buffer::Buffer, Terminal};

    fn snapshot(mode: MiningMode) -> UiSnapshot {
        UiSnapshot {
            miner_name: "keryx-miner-supr".into(),
            version: "v0.13.1-test".into(),
            build: "deadbee".into(),
            algorithm: "KeryxHash / PoM v4".into(),
            era: "H10".into(),
            uptime_secs: 200_478,
            connection: ConnectionView {
                mode,
                endpoint: if mode == MiningMode::Pool {
                    "krx.suprnova.cc:443".into()
                } else {
                    "local keryxd / gRPC".into()
                },
                state: ConnectionState::Connected,
                latency_ms: Some(28),
                last_job_age_secs: Some(1),
                difficulty: Some(8.0),
                daa_score: Some(12_847_291),
                failover: "armed / idle".into(),
                synced: Some(true),
                message: String::new(),
            },
            mining: MiningView {
                state: MiningState::Mining,
                total_hashrate_hs: 3_004_000.0,
                average_60s_hs: Some(2_998_000.0),
                hashrate_history_hs: vec![1.0, 2.0, 4.0, 3.5, 4.2, 4.3],
                total_power_w: Some(691.0),
                efficiency_mhs_per_w: Some(0.00435),
            },
            shares: ShareView {
                accepted: 184,
                rejected: 1,
                stale: 0,
                low_diff: 1,
                duplicate: 0,
                other: 0,
                pending: 0,
                last_accepted_age_secs: Some(78),
            },
            blocks: BlockView {
                found: if mode == MiningMode::Solo { 3 } else { 0 },
                accepted: (mode == MiningMode::Solo).then_some(2),
                rejected: (mode == MiningMode::Solo).then_some(1),
                pending: 0,
                last_accepted_age_secs: Some(360),
            },
            inference: InferenceView {
                state: InferenceState::Ready,
                requested: 44,
                prepared: 43,
                served: 42,
                delivered: 41,
                failed: 1,
                busy: 1,
                active: 0,
                queue_depth: 0,
                queue_capacity: 4,
                gpu_route_count: 1,
                gpu_index: Some(0),
                model: "Gemma-4-12B".into(),
                model_id: "9d13a4c2…".into(),
                tier: "default".into(),
                backend: "Vulkan".into(),
                last_latency_ms: Some(752),
                p95_latency_ms: Some(1_080),
                last_tokens: Some(31),
                pow_pause_total_secs: 102,
                self_test_ok: Some(true),
                status: "serveable".into(),
            },
            escrow: EscrowView {
                enabled: mode == MiningMode::Solo,
                claiming: false,
                alive: mode == MiningMode::Solo,
                claims_accepted: 7,
                claims_failed: 1,
                claims_pending: 2,
                last_attempt_age_secs: Some(40),
                last_success_age_secs: Some(300),
                claimable_amount: "1.250 KRX".into(),
                claimed_amount: "8.750 KRX".into(),
                status: "idle".into(),
                message: "next claim scheduled".into(),
            },
            service_bond: ServiceBondView {
                available: mode == MiningMode::Solo,
                consecutive_misses: 0,
                last_strike_daa: None,
                burned_claims: 0,
                burned_amount: "0.00000000 KRX".into(),
                suspended_until_daa: None,
                heartbeat_alive: mode == MiningMode::Solo,
                last_heartbeat_age_secs: Some(8),
                last_failure_age_secs: None,
            },
            devices: vec![
                DeviceView {
                    index: 2,
                    name: "Radeon MI50".into(),
                    backend: "OpenCL".into(),
                    hashrate_hs: 521_000.0,
                    temp_c: Some(74),
                    fan_pct: Some(78),
                    power_w: Some(225.0),
                    core_mhz: Some(1_680),
                    baseline_core_mhz: Some(1_700),
                    vram_used_mb: Some(10_000),
                    vram_total_mb: Some(16_384),
                    efficiency_mhs_per_w: Some(0.00231),
                    activity: DeviceActivity::Mining,
                    accepted: 12,
                    rejected: 0,
                    ..DeviceView::default()
                },
                DeviceView {
                    index: 0,
                    name: "RX 7900 XTX".into(),
                    backend: "OpenCL".into(),
                    hashrate_hs: 1_781_000.0,
                    temp_c: Some(62),
                    hotspot_c: Some(79),
                    fan_pct: Some(48),
                    power_w: Some(302.0),
                    core_mhz: Some(2_475),
                    baseline_core_mhz: Some(2_450),
                    mem_mhz: Some(1_249),
                    baseline_mem_mhz: Some(1_250),
                    vram_used_mb: Some(15_565),
                    vram_total_mb: Some(24_576),
                    efficiency_mhs_per_w: Some(0.00590),
                    accepted: 31,
                    rejected: 0,
                    inference_host: true,
                    activity: DeviceActivity::Mining,
                    throttle_reason: None,
                },
            ],
            events: vec![
                UiEvent {
                    timestamp: "14:21:57".into(),
                    kind: UiEventKind::Job,
                    message: "New H10 template; all workers active".into(),
                },
                UiEvent {
                    timestamp: "14:22:03".into(),
                    kind: UiEventKind::ShareAccepted,
                    message: "Accepted on GPU #1; round trip 31 ms".into(),
                },
                UiEvent {
                    timestamp: "14:22:08".into(),
                    kind: UiEventKind::InferenceOk,
                    message: "Request served on GPU #0; retained PoW job resumed".into(),
                },
            ],
        }
    }

    fn render(snapshot: &UiSnapshot, state: &TuiState, width: u16, height: u16) -> String {
        let backend = TestBackend::new(width, height);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal.draw(|frame| draw(frame, state, snapshot)).expect("draw");
        buffer_text(terminal.backend().buffer())
    }

    fn buffer_text(buffer: &Buffer) -> String {
        let mut out = String::new();
        for y in buffer.area.y..buffer.area.y + buffer.area.height {
            for x in buffer.area.x..buffer.area.x + buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    fn line_text(line: &Line<'_>) -> String {
        line.spans.iter().map(|span| span.content.as_ref()).collect()
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn responsive_layout_breakpoints_are_explicit() {
        assert_eq!(layout_mode(Rect::new(0, 0, 140, 40)), LayoutMode::Wide);
        assert_eq!(layout_mode(Rect::new(0, 0, 100, 30)), LayoutMode::Medium);
        assert_eq!(layout_mode(Rect::new(0, 0, 80, 32)), LayoutMode::Stacked);
        assert_eq!(layout_mode(Rect::new(0, 0, 80, 24)), LayoutMode::Tabbed);
        assert_eq!(layout_mode(Rect::new(0, 0, 69, 40)), LayoutMode::Tiny);
    }

    #[test]
    fn qq_is_required_and_ctrl_c_remains_host_owned() {
        let mut state = TuiState::default();
        assert_eq!(handle_key(&mut state, key(KeyCode::Char('q'))), TuiAction::None);
        assert!(state.quit_armed);
        assert_eq!(handle_key(&mut state, key(KeyCode::Char('q'))), TuiAction::QuitConfirmed);

        state.quit_armed = false;
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(handle_key(&mut state, ctrl_c), TuiAction::None);
        assert!(!state.quit_armed);

        handle_key(&mut state, key(KeyCode::Char('q')));
        let repeat = KeyEvent::new_with_kind(KeyCode::Char('q'), KeyModifiers::NONE, KeyEventKind::Repeat);
        assert_eq!(handle_key(&mut state, repeat), TuiAction::None);
        assert!(state.quit_armed, "key repeat must not confirm shutdown");
    }

    #[test]
    fn navigation_controls_only_presentation_state() {
        let mut state = TuiState::default();
        handle_key(&mut state, key(KeyCode::Char('2')));
        assert_eq!(state.page, DashboardPage::Gpus);
        handle_key(&mut state, key(KeyCode::Char('3')));
        assert_eq!(state.page, DashboardPage::Inference);
        handle_key(&mut state, key(KeyCode::Char('1')));
        assert_eq!(state.page, DashboardPage::Overview);
        handle_key(&mut state, key(KeyCode::Char('l')));
        assert!(state.logs_expanded);
        let motion = state.motion_enabled;
        handle_key(&mut state, key(KeyCode::Char('m')));
        assert_ne!(state.motion_enabled, motion);
        handle_key(&mut state, key(KeyCode::Up));
        assert_eq!(state.log_scroll, 1);
        handle_key(&mut state, key(KeyCode::Down));
        assert_eq!(state.log_scroll, 0);
        handle_key(&mut state, key(KeyCode::Char('?')));
        assert!(state.help_visible);
    }

    #[test]
    fn severity_uses_temperature_hotspot_clock_and_activity() {
        assert_eq!(temperature_severity(Some(44)), Severity::Cool);
        assert_eq!(temperature_severity(Some(45)), Severity::Nominal);
        assert_eq!(temperature_severity(Some(70)), Severity::Watch);
        assert_eq!(temperature_severity(Some(80)), Severity::Alert);
        assert_eq!(hotspot_severity(Some(89)), Severity::Nominal);
        assert_eq!(hotspot_severity(Some(90)), Severity::Watch);
        assert_eq!(hotspot_severity(Some(100)), Severity::Alert);
        assert_eq!(clock_severity(Some(900), Some(1_000)), Severity::Nominal);
        assert_eq!(clock_severity(Some(899), Some(1_000)), Severity::Watch);
        assert_eq!(clock_severity(Some(749), Some(1_000)), Severity::Alert);

        let paused_hot = DeviceView {
            activity: DeviceActivity::Inference,
            temp_c: Some(83),
            core_mhz: Some(100),
            baseline_core_mhz: Some(2_000),
            ..DeviceView::default()
        };
        assert_eq!(device_severity(&paused_hot), Severity::Alert);

        let paused_normal = DeviceView {
            activity: DeviceActivity::Inference,
            temp_c: Some(62),
            core_mhz: Some(100),
            baseline_core_mhz: Some(2_000),
            ..DeviceView::default()
        };
        assert_eq!(device_severity(&paused_normal), Severity::Cool);
    }

    #[test]
    fn device_order_is_stable_by_ordinal_then_identity() {
        let devices = vec![
            DeviceView { index: 2, name: "Z".into(), ..DeviceView::default() },
            DeviceView { index: 0, name: "B".into(), ..DeviceView::default() },
            DeviceView { index: 0, name: "a".into(), ..DeviceView::default() },
        ];
        let ordered = sorted_devices(&devices);
        assert_eq!(
            ordered.iter().map(|d| (d.index, d.name.as_str())).collect::<Vec<_>>(),
            vec![(0, "a"), (0, "B"), (2, "Z")]
        );
    }

    #[test]
    fn remote_strings_cannot_inject_terminal_controls() {
        assert_eq!(safe_text("pool\x1b[2J.example\n forged"), "pool[2J.example forged");
        assert_eq!(safe_text("  normal\twords  "), "normal words");
        assert_eq!(safe_text("left\u{202e}right\u{200b}"), "leftright");
    }

    #[test]
    fn matrix_mutation_never_changes_digits_or_warning_text() {
        let palette = Palette::new(ColorMode::TrueColor);
        let mut state = TuiState::default();
        for tick in 0..200 {
            state.set_animation_tick(tick);
            let line = matrix_title("GPU 5090 CORE", 1, &state, &palette);
            let text: String = line.spans.iter().map(|span| span.content.as_ref()).collect();
            let digits: String = text.chars().filter(char::is_ascii_digit).collect();
            assert_eq!(digits, "5090");

            let warning = matrix_title("GPU 2 ALERT", 1, &state, &palette);
            let warning_text: String = warning.spans.iter().map(|span| span.content.as_ref()).collect();
            assert_eq!(warning_text.trim(), "GPU 2 ALERT");
        }
    }

    #[test]
    fn matrix_rain_is_dense_visible_and_changes_without_touching_data() {
        let palette = Palette::new(ColorMode::TrueColor);
        let mut state = TuiState::default();
        state.set_animation_tick(7);
        let first = line_text(&matrix_rail(96, 7, &state, &palette));
        let first_density = first.chars().filter(|ch| !ch.is_whitespace()).count();
        assert!(first_density >= 8, "edge rail should be visibly populated: {first_density}");

        state.set_animation_tick(8);
        let second = line_text(&matrix_rail(96, 7, &state, &palette));
        assert_ne!(first, second, "rain must visibly advance between animation frames");

        let edge_density: usize = matrix_edge_lines(30, 31, &state, &palette)
            .iter()
            .map(line_text)
            .map(|line| line.chars().filter(|ch| !ch.is_whitespace()).count())
            .sum();
        assert!(edge_density >= 5, "vertical rain gutter should be visibly populated: {edge_density}");
    }

    #[test]
    fn allowlisted_body_copy_flips_rarely_but_runtime_values_never_enter_it() {
        let palette = Palette::new(ColorMode::TrueColor);
        let mut state = TuiState::default();
        let mut saw_flip = false;
        for tick in 0..404 {
            state.set_animation_tick(tick);
            let spans = matrix_body_label("HASH HISTORY 60", 17, &state, &palette);
            let text: String = spans.iter().map(|span| span.content.as_ref()).collect();
            let digits: String = text.chars().filter(char::is_ascii_digit).collect();
            assert_eq!(digits, "60");
            saw_flip |= text != "HASH HISTORY 60";

            let warning = matrix_body_label("HEALTH ALERT 80", 17, &state, &palette);
            let warning_text: String = warning.iter().map(|span| span.content.as_ref()).collect();
            assert_eq!(warning_text, "HEALTH ALERT 80");
        }
        assert!(saw_flip, "the allowlisted ornamental label should eventually flip");
    }

    #[test]
    fn pool_snapshot_keeps_shares_distinct_from_unknown_blocks() {
        let mut state = TuiState::default();
        state.motion_enabled = false;
        let output = render(&snapshot(MiningMode::Pool), &state, 140, 40);
        assert!(output.contains("SHARES"));
        assert!(output.contains("184 accepted"));
        assert!(output.contains("BLOCK CAND."));
        assert!(output.contains("0 found · -- accepted · -- rejected"));
        assert!(output.contains("krx.suprnova.cc:443"));
        assert!(output.contains("SERVED 42"));
    }

    #[test]
    fn rig_pane_preserves_full_gpu_name_rate_and_bottom_pinned_legend() {
        let mut snap = snapshot(MiningMode::Pool);
        snap.devices.truncate(1);
        snap.devices[0].index = 0;
        snap.devices[0].name = "NVIDIA GeForce GTX 1080 Ti".into();
        snap.devices[0].hashrate_hs = 12_345_000.0;
        let mut state = TuiState::default();
        state.motion_enabled = false;
        let output = render(&snap, &state, 140, 40);
        let rows: Vec<&str> = output.lines().collect();

        let device_y = rows
            .iter()
            .position(|line| line.contains("NVIDIA GeForce GTX 1080 Ti"))
            .expect("full GPU name must be rendered");
        assert!(rows[device_y].contains("12.345 MH/s"), "hash rate must directly follow the full name");
        assert!(!output.contains("NVIDIA GeForce GTX 1080…"));

        let legend_y = rows
            .iter()
            .position(|line| line.contains("● NOMINAL  ◆ PAUSED  ▲ WATCH  ■ ALERT"))
            .expect("GPU legend");
        let page_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(61), Constraint::Min(10), Constraint::Length(1)])
            .split(Rect::new(0, 0, 140, 40));
        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(page_rows[0]);
        let rig_inner_bottom = top[1].y + top[1].height - 2;
        assert_eq!(legend_y as u16, rig_inner_bottom, "legend must stay pinned to the pane's bottom row");
        assert!(legend_y > device_y);
    }

    #[test]
    fn overheat_alert_marker_and_text_blink_at_two_hz_with_steady_fallback() {
        let palette = Palette::new(ColorMode::TrueColor);
        let hot = DeviceView {
            index: 0,
            name: "Hot GPU".into(),
            hashrate_hs: 1_000_000.0,
            temp_c: Some(84),
            activity: DeviceActivity::Mining,
            ..DeviceView::default()
        };
        let mut state = TuiState::default();

        state.set_animation_tick(0);
        let on = device_card_lines(&hot, true, 60, &state, &palette);
        let on_marker = &on[0].spans[0];
        let on_alert = on
            .iter()
            .flat_map(|line| &line.spans)
            .find(|span| span.content == "ALERT")
            .expect("alert label");
        assert_eq!(on_marker.style.fg, Some(palette.red));
        assert!(on_marker.style.add_modifier.contains(Modifier::REVERSED));
        assert_eq!(on_alert.style, on_marker.style);

        state.set_animation_tick(2);
        let off = device_card_lines(&hot, true, 60, &state, &palette);
        let off_marker = &off[0].spans[0];
        let off_alert = off
            .iter()
            .flat_map(|line| &line.spans)
            .find(|span| span.content == "ALERT")
            .expect("alert label");
        assert_eq!(off_marker.style.fg, Some(palette.alert_dim));
        assert!(off_marker.style.add_modifier.contains(Modifier::DIM));
        assert_eq!(off_alert.style.fg, Some(palette.alert_dim));

        state.motion_enabled = false;
        let steady = device_card_lines(&hot, true, 60, &state, &palette);
        let steady_marker = &steady[0].spans[0];
        let steady_alert = steady
            .iter()
            .flat_map(|line| &line.spans)
            .find(|span| span.content == "ALERT")
            .expect("alert label");
        assert_eq!(steady_marker.style.fg, Some(palette.red));
        assert!(steady_marker.style.add_modifier.contains(Modifier::BOLD));
        assert!(!steady_marker.style.add_modifier.contains(Modifier::REVERSED));
        assert_eq!(steady_alert.style.fg, Some(palette.red));
    }

    #[test]
    fn powered_brand_scan_runs_both_directions_and_static_fallback_is_exact() {
        let palette = Palette::new(ColorMode::TrueColor);
        let eligible: Vec<usize> = POWERED_BY
            .chars()
            .enumerate()
            .filter_map(|(index, ch)| ch.is_ascii_alphabetic().then_some(index))
            .collect();
        let mut state = TuiState::default();

        state.set_animation_tick(POWER_SCAN_REST_TICKS);
        let first_ltr = powered_scan_position(&state).expect("left-to-right scan starts");
        state.set_animation_tick(POWER_SCAN_REST_TICKS + 1);
        let second_ltr = powered_scan_position(&state).expect("left-to-right scan advances");
        assert_eq!(first_ltr, (eligible[0], ScanDirection::LeftToRight));
        assert_eq!(second_ltr, (eligible[1], ScanDirection::LeftToRight));

        let rtl_start = POWER_SCAN_REST_TICKS * 2 + eligible.len() as u64;
        state.set_animation_tick(rtl_start);
        let first_rtl = powered_scan_position(&state).expect("right-to-left scan starts");
        state.set_animation_tick(rtl_start + 1);
        let second_rtl = powered_scan_position(&state).expect("right-to-left scan advances");
        assert_eq!(first_rtl, (*eligible.last().unwrap(), ScanDirection::RightToLeft));
        assert_eq!(second_rtl, (eligible[eligible.len() - 2], ScanDirection::RightToLeft));
        assert!(first_rtl.0 > second_rtl.0);

        state.motion_enabled = false;
        assert_eq!(powered_scan_position(&state), None);
        assert_eq!(line_text(&Line::from(powered_brand_spans(&state, &palette))), POWERED_BY);
    }

    #[test]
    fn primary_version_is_canonical_and_excludes_build_metadata() {
        let mut state = TuiState::default();
        state.motion_enabled = false;
        let mut snap = snapshot(MiningMode::Pool);
        snap.version = "v0.13.1".into();
        let output = render(&snap, &state, 120, 40);
        assert!(output.contains("VERSION      keryx-miner-supr v0.13.1"));
        assert!(!output.contains("deadbee"));
        assert!(!output.contains("KERYX // MINING CORE v0.13.1"));
    }

    #[test]
    fn ready_inference_route_never_claims_there_is_no_gpu() {
        let mut state = TuiState::default();
        state.motion_enabled = false;

        let mut unique = snapshot(MiningMode::Pool);
        unique.inference.gpu_index = Some(0);
        unique.inference.gpu_route_count = 1;
        let unique_output = render(&unique, &state, 140, 40);
        assert!(unique_output.contains("GPU #0"));
        assert!(!unique_output.contains("no GPU route"));

        let mut multiple = unique;
        multiple.inference.gpu_index = None;
        multiple.inference.gpu_route_count = 2;
        let multiple_output = render(&multiple, &state, 140, 40);
        assert!(multiple_output.contains("2 GPU routes"));
        assert!(!multiple_output.contains("no GPU route"));
    }

    #[test]
    fn ready_without_a_proven_route_is_not_masked_as_a_generic_gpu_route() {
        let mut state = TuiState::default();
        state.motion_enabled = false;
        let mut inconsistent = snapshot(MiningMode::Pool);
        inconsistent.inference.gpu_index = None;
        inconsistent.inference.gpu_route_count = 0;

        let output = render(&inconsistent, &state, 140, 40);
        assert!(output.contains("no GPU route"));
    }

    #[test]
    fn solo_snapshot_shows_block_and_escrow_lifecycle() {
        let mut state = TuiState::default();
        state.motion_enabled = false;
        let output = render(&snapshot(MiningMode::Solo), &state, 140, 40);
        assert!(output.contains("BLOCKS"));
        assert!(output.contains("3 found · 2 accepted · 1 rejected"));
        assert!(output.contains("ESCROW"));
        assert!(output.contains("outputs A 7 / F 1 / P 2"));
        assert!(output.contains("1.250 KRX"));
    }

    #[test]
    fn escrow_startup_is_cool_but_a_lost_heartbeat_is_an_alert() {
        let starting = EscrowView { enabled: true, status: "waiting".into(), ..EscrowView::default() };
        assert_eq!(escrow_status(&starting), ("◆", "STARTING", Severity::Cool));

        let stale = EscrowView { enabled: true, status: "heartbeat lost".into(), ..EscrowView::default() };
        assert_eq!(escrow_status(&stale), ("■", "WORKER DOWN", Severity::Alert));

        let held = EscrowView { enabled: true, alive: true, status: "held".into(), ..EscrowView::default() };
        assert_eq!(escrow_status(&held), ("▲", "CLAIMS HELD", Severity::Watch));

        let validating =
            EscrowView { enabled: true, alive: true, status: "validating".into(), ..EscrowView::default() };
        assert_eq!(escrow_status(&validating), ("◆", "VALIDATING", Severity::Cool));

        let degraded = EscrowView { enabled: true, alive: true, status: "degraded".into(), ..EscrowView::default() };
        assert_eq!(escrow_status(&degraded), ("■", "DEGRADED", Severity::Alert));
    }

    #[test]
    fn non_nominal_service_bond_is_promoted_to_solo_overview() {
        let mut snap = snapshot(MiningMode::Solo);
        snap.service_bond.consecutive_misses = 2;
        snap.service_bond.burned_claims = 1;
        snap.service_bond.burned_amount = "0.25000000 KRX".into();
        snap.service_bond.suspended_until_daa = Some(12_850_000);
        let mut state = TuiState::default();
        state.motion_enabled = false;
        let output = render(&snap, &state, 140, 40);
        assert!(output.contains("SERVICE BOND"));
        assert!(output.contains("SUSPENDED"));
        assert!(output.contains("misses 2"));
        assert!(output.contains("0.25000000 KRX"));
    }

    #[test]
    fn service_bond_startup_and_pending_are_informative_not_alerts() {
        let pending = ServiceBondView::default();
        assert_eq!(service_bond_status(&pending), ("◆", "PENDING", Severity::Cool));

        let starting = ServiceBondView { available: true, ..ServiceBondView::default() };
        assert_eq!(service_bond_status(&starting), ("◆", "STARTING", Severity::Cool));

        let failed = ServiceBondView { available: false, last_failure_age_secs: Some(2), ..ServiceBondView::default() };
        assert_eq!(service_bond_status(&failed), ("▲", "UNAVAILABLE", Severity::Watch));
    }

    #[test]
    fn rendering_preserves_endpoint_while_titles_animate() {
        let snap = snapshot(MiningMode::Pool);
        let rtl_scan = POWER_SCAN_REST_TICKS * 2
            + POWERED_BY.chars().filter(|ch| ch.is_ascii_alphabetic()).count() as u64;
        for tick in [0, 1, 22, 23, POWER_SCAN_REST_TICKS, POWER_SCAN_REST_TICKS + 1, rtl_scan, rtl_scan + 1] {
            let mut state = TuiState::default();
            state.set_animation_tick(tick);
            let output = render(&snap, &state, 140, 40);
            assert!(output.contains("krx.suprnova.cc:443"));
            assert!(output.contains("184 accepted"));
            assert!(output.contains("FAILED 1"));
        }
    }

    #[test]
    fn compact_render_remains_informative_without_logo_or_motion() {
        let mut state = TuiState::default();
        state.motion_enabled = false;
        let output = render(&snapshot(MiningMode::Solo), &state, 68, 18);
        assert!(output.contains("KERYX // COMPACT"));
        assert!(output.contains("SOLO"));
        assert!(output.contains("blocks F/A/R 3/2/1"));
        assert!(output.contains("served 42 / failed 1"));
    }
}
