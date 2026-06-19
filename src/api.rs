//! Minimal dependency-free HTTP JSON stats API for monitoring
//! (HiveOS / mmpOS / dashboards).
//!
//! Endpoints (any GET):
//!   `/` or `/stats`  -> generic JSON: miner, version, algo, uptime, total +
//!                       per-device hashrate (hashes/sec), accepted/rejected shares.
//!   `/mmpos`         -> mmpOS custom-miner format:
//!                       {busid, hash, units, air, miner_name, miner_version}.
//!
//! Hand-rolled HTTP/1.1 over a tokio `TcpListener` (no extra deps). Hashrate
//! comes from the miner's live snapshot; shares from the global stratum
//! `ShareStats`.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use log::{info, warn};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::client::stratum::share_stats;
use crate::miner::MinerStats;

const MINER_NAME: &str = "keryx-miner-supr";

/// (accepted, rejected) from the global share counters; (0,0) before the pool connects.
fn shares() -> (u64, u64) {
    match share_stats() {
        Some(s) => {
            let acc = s.accepted.load(Ordering::SeqCst);
            let rej = s.low_diff.load(Ordering::SeqCst)
                + s.stale.load(Ordering::SeqCst)
                + s.duplicate.load(Ordering::SeqCst);
            (acc, rej)
        }
        None => (0, 0),
    }
}

/// Parse the device ordinal from a label like "#0 (NVIDIA GeForce RTX 5090):" -> 0.
fn dev_index(label: &str) -> Option<u64> {
    let after_hash = label.split('#').nth(1)?;
    let digits: String = after_hash.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

fn generic_json(stats: &Arc<Mutex<MinerStats>>, version: &str, uptime: u64) -> String {
    let (total, devices) = {
        let s = stats.lock().unwrap();
        (s.total_hashrate, s.devices.clone())
    };
    let (acc, rej) = shares();
    let devs: Vec<_> = devices
        .iter()
        .map(|d| serde_json::json!({ "name": d.label, "hashrate_hs": d.hashrate }))
        .collect();
    serde_json::json!({
        "miner": MINER_NAME,
        "version": version,
        "algo": "keryxhash",
        "uptime_s": uptime,
        "total_hashrate_hs": total,
        "devices": devs,
        "shares": { "accepted": acc, "rejected": rej },
    })
    .to_string()
}

fn mmpos_json(stats: &Arc<Mutex<MinerStats>>, version: &str) -> String {
    let (total, devices) = {
        let s = stats.lock().unwrap();
        (s.total_hashrate, s.devices.clone())
    };
    let (acc, rej) = shares();
    let mut busid: Vec<u64> = Vec::new();
    let mut hash: Vec<f64> = Vec::new();
    for (i, d) in devices.iter().enumerate() {
        busid.push(dev_index(&d.label).unwrap_or(i as u64));
        hash.push(d.hashrate);
    }
    if hash.is_empty() {
        // CPU-only, or before per-device data exists: report the total on bus 0.
        busid.push(0);
        hash.push(total);
    }
    serde_json::json!({
        "busid": busid,
        "hash": hash,
        "units": "hs",
        "air": [acc.to_string(), "0", rej.to_string()],
        "miner_name": MINER_NAME,
        "miner_version": version,
    })
    .to_string()
}

/// Serve the stats API forever. On bind failure it logs and returns (never crashes the miner).
pub async fn serve(bind: String, stats: Arc<Mutex<MinerStats>>, version: String) {
    let listener = match TcpListener::bind(&bind).await {
        Ok(l) => l,
        Err(e) => {
            warn!("Stats API: cannot bind {} ({}) — API disabled", bind, e);
            return;
        }
    };
    info!("Stats API on http://{}  (/, /stats, /mmpos)", bind);
    let start = Instant::now();
    loop {
        let (mut sock, _) = match listener.accept().await {
            Ok(x) => x,
            Err(_) => continue,
        };
        let stats = Arc::clone(&stats);
        let version = version.clone();
        let uptime = start.elapsed().as_secs();
        tokio::spawn(async move {
            let mut buf = [0u8; 2048];
            let n = sock.read(&mut buf).await.unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]);
            // request line: "GET /path HTTP/1.1"
            let path = req.split_whitespace().nth(1).unwrap_or("/");
            let body = if path.starts_with("/mmpos") {
                mmpos_json(&stats, &version)
            } else {
                generic_json(&stats, &version, uptime)
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = sock.write_all(resp.as_bytes()).await;
            let _ = sock.shutdown().await;
        });
    }
}
