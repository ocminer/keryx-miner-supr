//! Node-authoritative H16 request lifetime and deterministic sibling-rig scheduling.
//! Adapted from upstream ae8e38e, 5b4b88b and b0d9d18, retaining full request keys.
use super::{hash32, AiRequestKey};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

pub(super) const CHECK_WAIT_SECS: u64 = 15;
const SIBLING_MARGIN_SECS: u64 = 15;
const WINDOW_SAFETY_DAA: u64 = 600;

#[derive(Default)]
pub(super) struct Schedule {
    pub node_checks: bool,
    pub queued_at: HashMap<AiRequestKey, Instant>,
    pub checks: HashMap<AiRequestKey, (bool, u64, Instant)>,
    pub siblings: HashMap<u32, Vec<[u8; 32]>>,
}

pub(super) fn sibling_rank(request: &[u8; 32], own: &[u8; 32], siblings: &[[u8; 32]]) -> usize {
    let key = |escrow: &[u8; 32]| {
        let mut input = [0; 64];
        input[..32].copy_from_slice(request);
        input[32..].copy_from_slice(escrow);
        let digest = blake2b_simd::blake2b(&input);
        <[u8; 32]>::try_from(&digest.as_bytes()[..32]).unwrap()
    };
    let mine = key(own);
    siblings.iter().filter(|escrow| *escrow != own && key(escrow) < mine).count()
}

impl Schedule {
    pub fn remove(&mut self, key: &AiRequestKey) {
        self.queued_at.remove(key);
        self.checks.remove(key);
    }

    pub fn closed(&self, key: &AiRequestKey, daa: u64) -> bool {
        self.checks.get(key).is_some_and(|(awaited, end, checked)| {
            (!awaited && self.queued_at.get(key).is_none_or(|queued| checked >= queued))
                || (*awaited && *end > 0 && daa > *end)
        })
    }

    pub fn startable(
        &self,
        key: &AiRequestKey,
        rank: usize,
        now: Instant,
        daa: u64,
        work: u64,
        generation: Option<u64>,
    ) -> bool {
        if self.closed(key, daa) {
            return false;
        }
        if !self.node_checks || rank == 0 {
            return true;
        }
        let queued = self.queued_at.get(key).copied().unwrap_or(now);
        let delay = Duration::from_secs(
            (rank as u64).saturating_mul(generation.map_or(60, |d| d / 10).saturating_add(SIBLING_MARGIN_SECS)),
        );
        let check = self.checks.get(key);
        let end = check.map_or(0, |c| c.1);
        let near = end > 0 && daa.saturating_add(work).saturating_add(WINDOW_SAFETY_DAA) >= end;
        let waited = now.saturating_duration_since(queued);
        if waited < delay && !near {
            return false;
        }
        near || check.is_some_and(|(awaited, _, at)| *awaited && at.saturating_duration_since(queued) >= delay)
            || waited >= delay.saturating_add(Duration::from_secs(CHECK_WAIT_SECS))
    }

    pub fn apply(
        &mut self,
        polled: Vec<(AiRequestKey, [u8; 32])>,
        response: &crate::proto::GetAwaitedOpeningsResponseMessage,
        now: Instant,
    ) {
        self.node_checks = response.requests_checked;
        if !self.node_checks {
            return; // Older nodes do not implement request checks; retain their prior behavior.
        }
        let awaited: HashMap<String, u64> =
            response.awaited_requests.iter().map(|r| (r.request_hash.to_ascii_lowercase(), r.window_end_daa)).collect();
        let closed: HashSet<String> = response.closed_requests.iter().map(|h| h.to_ascii_lowercase()).collect();
        for (key, request) in polled {
            if !self.queued_at.contains_key(&key) {
                continue; // Finished/evicted since the poll: never recreate stale metadata.
            }
            let request = hex::encode(request);
            if let Some(end) = awaited.get(&request) {
                self.checks.insert(key, (true, *end, now));
            } else if closed.contains(&request) {
                self.checks.insert(key, (false, 0, now));
            }
            // Unknown is neither open nor closed: keep it until the ledger knows the request.
        }
    }

    pub fn providers(&mut self, own: &str, response: &crate::proto::GetServiceProvidersResponseMessage) {
        let mut siblings: HashMap<u32, Vec<[u8; 32]>> = HashMap::new();
        for mine in response.providers.iter().filter(|p| p.escrow_pubkey.eq_ignore_ascii_case(own)) {
            let keys = siblings.entry(mine.tier).or_default();
            for p in response.providers.iter().filter(|p| p.tier == mine.tier && p.identity == mine.identity) {
                if let Some(key) = hash32(&p.escrow_pubkey) {
                    keys.push(key);
                }
            }
            keys.sort_unstable();
            keys.dedup();
        }
        self.siblings = siblings;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{
        AwaitedRequestMessage, GetAwaitedOpeningsResponseMessage, GetServiceProvidersResponseMessage,
        ServiceProviderMessage,
    };

    #[test]
    fn h16_request_rpc_distinguishes_open_closed_unknown_and_evicted() {
        let now = Instant::now();
        let mut s = Schedule::default();
        for byte in 1..=3 {
            s.queued_at.insert(AiRequestKey([byte; 32]), now);
        }
        let polled = (1..=4).map(|byte| (AiRequestKey([byte; 32]), [byte; 32])).collect();
        s.apply(
            polled,
            &GetAwaitedOpeningsResponseMessage {
                requests_checked: true,
                awaited_requests: vec![AwaitedRequestMessage {
                    request_hash: hex::encode([1; 32]),
                    window_end_daa: 1000,
                }],
                closed_requests: vec![hex::encode([2; 32]), hex::encode([4; 32])],
                ..Default::default()
            },
            now,
        );
        assert!(!s.closed(&AiRequestKey([1; 32]), 1000));
        assert!(s.closed(&AiRequestKey([1; 32]), 1001));
        assert!(s.closed(&AiRequestKey([2; 32]), 1));
        assert!(!s.closed(&AiRequestKey([3; 32]), 999999));
        assert!(!s.checks.contains_key(&AiRequestKey([4; 32])));
        // A stale closed check cannot discard a newly queued request.
        s.queued_at.insert(AiRequestKey([2; 32]), now + Duration::from_secs(1));
        assert!(!s.closed(&AiRequestKey([2; 32]), 1));
    }

    #[test]
    fn h16_first_rig_starts_immediately_siblings_wait_but_do_not_stall() {
        let now = Instant::now();
        let key = AiRequestKey([1; 32]);
        let mut s = Schedule { node_checks: true, ..Default::default() };
        s.queued_at.insert(key, now);
        assert!(s.startable(&key, 0, now, 1000, 200, None));
        assert!(!s.startable(&key, 1, now, 1000, 200, None));
        assert!(!s.startable(&key, 1, now + Duration::from_secs(89), 1000, 200, None));
        assert!(s.startable(&key, 1, now + Duration::from_secs(90), 1000, 200, None));
        s.checks.insert(key, (true, 2000, now + Duration::from_secs(75)));
        assert!(s.startable(&key, 1, now + Duration::from_secs(75), 1000, 200, None));
        // Deadline approaching: do not wait out another rig's slot.
        s.checks.insert(key, (true, 1700, now));
        assert!(s.startable(&key, 2, now, 1000, 200, None));
        s.checks.insert(key, (false, 0, now));
        assert!(!s.startable(&key, 0, now, 1000, 200, None));
    }

    #[test]
    fn h16_sibling_ranks_match_all_rigs_and_filter_other_addresses_and_tiers() {
        let own = hex::encode([1; 32]);
        let mut s = Schedule::default();
        let row = |tier, identity: &str, key| ServiceProviderMessage {
            tier,
            identity: identity.into(),
            escrow_pubkey: hex::encode([key; 32]),
            ..Default::default()
        };
        s.providers(
            &own,
            &GetServiceProvidersResponseMessage {
                providers: vec![
                    row(0, "ours", 1),
                    row(0, "ours", 2),
                    row(0, "ours", 2),
                    row(1, "ours", 3),
                    row(0, "other", 4),
                ],
                ..Default::default()
            },
        );
        assert_eq!(s.siblings.get(&0).unwrap(), &vec![[1; 32], [2; 32]]);
        assert!(!s.siblings.contains_key(&1));
        let keys: Vec<_> = (1..=5).map(|b| [b; 32]).collect();
        for b in 0..20 {
            let mut ranks: Vec<_> = keys.iter().map(|own| sibling_rank(&[b; 32], own, &keys)).collect();
            ranks.sort_unstable();
            assert_eq!(ranks, vec![0, 1, 2, 3, 4]);
        }
    }
}
