//! `--wait-ready`: hold mining AND the OPoI capability declaration until EVERY card in this
//! process has its walk table installed.
//!
//! Why this exists: on RAM-constrained rigs (reported: 4 GB hosts with 8 cards, `--low-ram`),
//! bring-up takes many minutes per card. The moment the FIRST card finishes, its model is
//! declared serveable and the pool starts routing OPoI challenges — which then run starved
//! (host RAM and disk are saturated by the remaining cards' staging) and miss their deadline.
//! Failed challenges cost inference strikes, and a challenge's model-swap/pause machinery
//! landing in the middle of another card's staging can stall bring-up entirely (reported as
//! "stops loading the GPUs and spins in a loop"). Holding the DECLARATION means the pool has
//! nothing to challenge until the rig can actually answer; holding MINING keeps the walk from
//! competing with staging for the same starved host. Off by default; purely opt-in.
//!
//! Semantics:
//!  - Every GPU worker registers itself at thread start; every successful walk install marks
//!    its card ready. The gate holds while any registered card is not ready (plus a short
//!    startup grace so all workers get to register before the first card can win the race).
//!  - The gate is a LATCH: once open it stays open for the process lifetime. Later transient
//!    uninstalls (model swap while serving, fault recovery, tier change) must not re-suspend
//!    a working rig.
//!  - Safety valve: a card that can never stage (unsupported tier, dead card) must not idle
//!    the rig forever. After KERYX_WAIT_READY_TIMEOUT_SECS (default 2700 = 45 min) the gate
//!    opens with whatever is ready, loudly naming the cards that never made it.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static OPENED: AtomicBool = AtomicBool::new(false);
static LAST_LOG_SECS: AtomicU64 = AtomicU64::new(0);

fn started() -> &'static Instant {
    static T: OnceLock<Instant> = OnceLock::new();
    T.get_or_init(Instant::now)
}

/// (registered worker devices, devices with an installed walk)
fn sets() -> &'static Mutex<(BTreeSet<u32>, BTreeSet<u32>)> {
    static S: OnceLock<Mutex<(BTreeSet<u32>, BTreeSet<u32>)>> = OnceLock::new();
    S.get_or_init(|| Mutex::new((BTreeSet::new(), BTreeSet::new())))
}

fn timeout_secs() -> u64 {
    std::env::var("KERYX_WAIT_READY_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(2700)
}

/// Turn the gate on (from the `--wait-ready` CLI flag, before workers spawn).
pub fn enable() {
    let _ = started(); // pin t0 to enable time
    ENABLED.store(true, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// A GPU worker announces itself (idempotent). Called at worker-thread start, so within the
/// startup grace every card that will ever mine is known.
pub fn register_device(device_id: u32) {
    if let Ok(mut g) = sets().lock() {
        g.0.insert(device_id);
    }
}

/// A card's walk table finished installing (idempotent). Called from the driver's `install`.
pub fn mark_ready(device_id: u32) {
    let (ready, total) = match sets().lock() {
        Ok(mut g) => {
            g.1.insert(device_id);
            (g.1.intersection(&g.0).count(), g.0.len())
        }
        Err(_) => return,
    };
    if enabled() && !OPENED.load(Ordering::Relaxed) {
        log::info!("--wait-ready: GPU {} is ready ({}/{} cards).", device_id, ready, total);
    }
}

/// True while mining and the OPoI declaration must stay held. Cheap; called per batch attempt
/// and per declaration pass.
pub fn holds() -> bool {
    if !ENABLED.load(Ordering::Relaxed) || OPENED.load(Ordering::Relaxed) {
        return false;
    }
    let elapsed = started().elapsed();
    // Startup grace: worker threads register within milliseconds of spawn, staging takes
    // minutes — 10 s guarantees the registered set is complete before the gate can open.
    if elapsed.as_secs() < 10 {
        return true;
    }
    let (missing, total): (Vec<u32>, usize) = match sets().lock() {
        Ok(g) => (g.0.difference(&g.1).copied().collect(), g.0.len()),
        Err(_) => return false,
    };
    // No worker ever registered (a backend that doesn't wire the gate yet, or no GPUs): the
    // gate would be a 45-minute dead hold with nothing to wait for — open it, loudly.
    if total == 0 {
        if !OPENED.swap(true, Ordering::Relaxed) {
            log::warn!(
                "--wait-ready: no GPU workers registered on this backend — the flag has no \
                 effect here; mining proceeds normally."
            );
        }
        return false;
    }
    if missing.is_empty() {
        if !OPENED.swap(true, Ordering::Relaxed) {
            log::info!(
                "--wait-ready: all {} cards are set up — starting mining and declaring OPoI \
                 capabilities to the pool.",
                total
            );
        }
        return false;
    }
    if elapsed.as_secs() >= timeout_secs() {
        if !OPENED.swap(true, Ordering::Relaxed) {
            log::warn!(
                "--wait-ready: timeout after {}s with card(s) {:?} still not ready — starting \
                 with the {} that are. Raise KERYX_WAIT_READY_TIMEOUT_SECS if this rig simply \
                 needs longer; check the staging errors above if a card is truly stuck.",
                timeout_secs(),
                missing,
                total - missing.len()
            );
        }
        return false;
    }
    // Progress note every 30 s so a long quiet bring-up is visibly alive, not hung.
    let now = elapsed.as_secs();
    let last = LAST_LOG_SECS.load(Ordering::Relaxed);
    if now >= last + 30
        && LAST_LOG_SECS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        log::info!(
            "--wait-ready: {}/{} cards ready (waiting on {:?}) — mining and OPoI declaration \
             held until all cards are set up.",
            total - missing.len(),
            total,
            missing
        );
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test exercises the whole latch (statics are process-wide, so a single sequential
    // scenario keeps the assertions order-independent from the test harness).
    #[test]
    fn gate_holds_then_latches_open() {
        assert!(!holds(), "gate must be inert while disabled");
        enable();
        register_device(0);
        register_device(1);
        mark_ready(0);
        assert!(holds(), "one of two cards ready — must hold (grace window also active)");
        mark_ready(1);
        // Inside the 10 s startup grace the gate still holds even though all are ready.
        assert!(holds(), "startup grace must hold the gate");
        // After the grace it opens (simulate by waiting out the grace in test time: the grace
        // is wall-clock; keep the test fast by only checking the latch path via OPENED).
        OPENED.store(true, Ordering::Relaxed);
        assert!(!holds(), "open latch must stay open");
        mark_ready(1); // idempotent, no panic, no re-close
        assert!(!holds());
    }
}
