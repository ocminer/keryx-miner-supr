//! Coordination outcomes are separate from model capability. Waiting for another GPU operation
//! must never withdraw a model, demote a mining tier, or trigger inference-host failover.

use std::collections::HashSet;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Mutex;
use std::thread::ThreadId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeferredReason {
    ModelNotRegistered,
    ModelIncomplete,
    CardBusy,
    InstallBusy,
    WalkBusy,
    ProbeBusy,
    HostLoadBusy,
    RetryBackoff,
}

impl std::fmt::Display for DeferredReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ModelNotRegistered => "model lineup is still initializing",
            Self::ModelIncomplete => "model files are still staging",
            Self::CardBusy => "serving GPU is busy",
            Self::InstallBusy => "serving GPU is installing its mining model",
            Self::WalkBusy => "serving GPU walk has not drained",
            Self::ProbeBusy => "this model/GPU probe is already in progress",
            Self::HostLoadBusy => "another model operation owns the low-RAM staging permit",
            Self::RetryBackoff => "failed route is waiting for its next recovery probe",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceError {
    Deferred(DeferredReason),
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfTestOutcome {
    Passed,
    Deferred(DeferredReason),
    Failed,
}

/// Nonblocking, thread-reentrant host-load permit. An install may call its own inference probe
/// while holding this permit, but another installer/warmer defers without holding any GPU locks.
#[derive(Default)]
pub struct HostLoadGate(Mutex<(Option<ThreadId>, usize)>);

impl HostLoadGate {
    pub fn try_enter(&self) -> Option<HostLoadPermit<'_>> {
        let current = std::thread::current().id();
        let mut state = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if state.0.is_some_and(|owner| owner != current) {
            return None;
        }
        state.0 = Some(current);
        state.1 += 1;
        Some(HostLoadPermit { gate: self, _thread_bound: PhantomData })
    }
}

pub struct HostLoadPermit<'a> {
    gate: &'a HostLoadGate,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Drop for HostLoadPermit<'_> {
    fn drop(&mut self) {
        let mut state = self.gate.0.lock().unwrap_or_else(|p| p.into_inner());
        debug_assert_eq!(state.0, Some(std::thread::current().id()));
        state.1 -= 1;
        if state.1 == 0 {
            state.0 = None;
        }
    }
}

type ProbeKey = ([u8; 32], usize);

#[derive(Default)]
pub struct ProbeFlights(Mutex<HashSet<ProbeKey>>);

impl ProbeFlights {
    pub fn try_enter(&self, model: [u8; 32], gpu: usize) -> Option<ProbePermit<'_>> {
        let key = (model, gpu);
        let mut active = self.0.lock().unwrap_or_else(|p| p.into_inner());
        active.insert(key).then(|| ProbePermit { flights: self, key })
    }
}

pub struct ProbePermit<'a> {
    flights: &'a ProbeFlights,
    key: ProbeKey,
}

impl Drop for ProbePermit<'_> {
    fn drop(&mut self) {
        self.flights.0.lock().unwrap_or_else(|p| p.into_inner()).remove(&self.key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_probe_has_one_owner_and_other_routes_can_progress() {
        let flights = ProbeFlights::default();
        let owner = flights.try_enter([1; 32], 2).unwrap();
        std::thread::scope(|s| {
            for _ in 0..3 {
                s.spawn(|| {
                    assert!(flights.try_enter([1; 32], 2).is_none());
                });
            }
            assert!(flights.try_enter([1; 32], 0).is_some());
            assert!(flights.try_enter([2; 32], 2).is_some());
        });
        // No elapsed-time expiry may let a follower steal a still-running slow owner's probe.
        assert!(flights.try_enter([1; 32], 2).is_none());
        drop(owner);
        assert!(flights.try_enter([1; 32], 2).is_some());
    }

    #[test]
    fn low_ram_permit_is_reentrant_and_releases_on_unwind() {
        let gate = HostLoadGate::default();
        let _ = std::panic::catch_unwind(|| {
            let outer = gate.try_enter().unwrap();
            let inner = gate.try_enter().unwrap();
            std::thread::scope(|s| {
                s.spawn(|| assert!(gate.try_enter().is_none())).join().unwrap();
                drop(inner);
                s.spawn(|| assert!(gate.try_enter().is_none())).join().unwrap();
            });
            let _keep = outer;
            panic!("test unwind");
        });
        std::thread::scope(|s| {
            s.spawn(|| assert!(gate.try_enter().is_some())).join().unwrap();
        });
    }
}
