# Devfund cycle — removed (archived here for possible future restore)

The "devfund" mechanism let the miner spend a configurable fraction of blocks
to a developer/pool address by **periodically switching the address it
authorizes with and reconnecting**. The `-supr` fork already defaulted the tax
to 0 %, but the *cycling machinery still ran*, and it had a latent crash-loop
bug (see "Why it was removed"). It was removed wholesale on 2026-06-09 so the
miner always mines to the configured address and never switches.

This file is the canonical archive of the removed code. To restore the feature,
re-apply each snippet to the file/region noted. (Git also has it: the commit
*before* the removal contains the live code.)

---

## Why it was removed (the bug)

`devfund_percent = 0` and `add_devfund()` is only called when
`devfund_percent > 0` (`main.rs`), so in the fork `add_devfund` was **never**
called → `devfund_address` stayed `None` and `mining_dev` was always
`Some(false)`.

The `listen()` guard then reduced to: **`return Ok(())` whenever
`block_template_ctr == 0`**. That counter increments `(v+1) % 10_000` on every
job notify, starting from a random value, so it wraps to 0 every ~10 000 jobs.
The guard sits at the *top* of the loop, **before** reading a message, and the
counter is a shared `Arc` that persists across reconnects — so once it hit 0,
`listen()` returned immediately on every reconnect, never processing a job
(which is the only thing that moves the counter off 0). Result: a **permanent
reconnect crash-loop, never mining.** Removing the cycle is the complete fix.

---

## Removed code, by file

### 1. `src/client.rs` — trait method

```rust
fn add_devfund(&mut self, address: String, percent: u16);
```
(was the first method of `pub trait Client`.)

### 2. `src/cli.rs`

CLI field (in `struct Opt`):
```rust
// Upstream `keryx-miner` hardcoded `keryx:qrxpcusy…` as the devfund and
// forced a 2 % minimum. The `-supr` fork removes the tax entirely: the
// default is 0 and there is no minimum clamp. If you want to donate to
// someone, pass `--devfund-percent N` explicitly and patch
// `Opt::process()` to point `devfund_address` somewhere meaningful.
#[clap(long = "devfund-percent", help = "The percentage of blocks to send to the devfund (default 0 — no tax)", default_value = "0", parse(try_from_str = parse_devfund_percent))]
pub devfund_percent: u16,

#[clap(skip)]
pub devfund_address: String,
```

Parser fn (top-level in `cli.rs`):
```rust
fn parse_devfund_percent(s: &str) -> Result<u16, &'static str> {
    let err = "devfund-percent should be --devfund-percent=XX.YY up to 2 numbers after the dot";
    let mut splited = s.split('.');
    let prefix = splited.next().ok_or(err)?;
    // if there's no postfix then it's 0.
    let postfix = splited.next().ok_or(err).unwrap_or("0");
    // error if there's more than a single dot
    if splited.next().is_some() {
        return Err(err);
    };
    // error if there are more than 2 numbers before or after the dot
    if prefix.len() > 2 || postfix.len() > 2 {
        return Err(err);
    }
    let postfix: u16 = postfix.parse().map_err(|_| err)?;
    let prefix: u16 = prefix.parse().map_err(|_| err)?;
    // can't be more than 99.99%,
    if prefix >= 100 || postfix >= 100 {
        return Err(err);
    }
    // `-supr` fork: no 2 % minimum. Upstream forced `Ok(200u16)` when prefix
    // was below 2, taxing every miner with no opt-out.
    // DevFund is out of 10_000
    Ok(prefix * 100 + postfix)
}
```

Block inside `Opt::process()` (just before `Ok(())`):
```rust
let miner_network = self.mining_address.as_deref().and_then(|a| a.split(':').next());
self.devfund_address = String::from("keryx:qp0vrxc0k5w0pcyem6vau2pjgztje880tsm239rywtm7l7uv2pcxzq55n8khs");
let devfund_network = self.devfund_address.split(':').next();
if miner_network.is_some() && devfund_network.is_some() && miner_network != devfund_network {
    self.devfund_percent = 0;
    log::info!(
        "Mining address ({}) and devfund ({}) are not from the same network. Disabling devfund.",
        miner_network.unwrap(),
        devfund_network.unwrap()
    )
}
```

### 3. `src/main.rs`

Register-time hookup (in `client_main`, before `client.register()`):
```rust
if opt.devfund_percent > 0 {
    client.add_devfund(opt.devfund_address.clone(), opt.devfund_percent);
}
```

Startup log (after `block_template_ctr` is created):
```rust
if opt.devfund_percent > 0 {
    info!(
        "devfund enabled, mining {}.{}% of the time to devfund address: {} ",
        opt.devfund_percent / 100,
        opt.devfund_percent % 100,
        opt.devfund_address
    );
}
```

### 4. `src/client/stratum.rs`

Struct fields:
```rust
devfund_address: Option<String>,
devfund_percent: u16,
mining_dev: Option<bool>,
```

`add_devfund` impl:
```rust
fn add_devfund(&mut self, address: String, percent: u16) {
    self.devfund_address = Some(address);
    self.devfund_percent = percent;
}
```

`register()` pay-address selection (replaced by `let pay_address = self.miner_address.clone();`):
```rust
let pay_address = match &self.devfund_address {
    Some(devfund_address) if self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent => {
        self.mining_dev = Some(true);
        info!("Mining to devfund");
        devfund_address.clone()
    }
    _ => {
        self.mining_dev = Some(false);
        self.miner_address.clone()
    }
};
```

`listen()` swap-exit guard (deleted; was the top of the `loop`):
```rust
{
    if (!self.mining_dev.unwrap_or(true)
        && self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent)
        || (self.mining_dev.unwrap_or(false)
            && self.block_template_ctr.load(Ordering::SeqCst) > self.devfund_percent)
    {
        return Ok(());
    }
}
```

Struct init (in `connect()`):
```rust
devfund_address: None,
devfund_percent: 0,
...
mining_dev: None,
```

### 5. `src/client/grpc.rs`

Struct fields:
```rust
devfund_address: Option<String>,
devfund_percent: u16,
```

`add_devfund` impl:
```rust
fn add_devfund(&mut self, address: String, percent: u16) {
    self.devfund_address = Some(address);
    self.devfund_percent = percent;
}
```

`client_get_block_template()` pay-address selection (replaced by `let pay_address = self.miner_address.clone();`):
```rust
let pay_address = match &self.devfund_address {
    Some(devfund_address) if self.block_template_ctr.load(Ordering::SeqCst) <= self.devfund_percent => {
        devfund_address.clone()
    }
    _ => self.miner_address.clone(),
};
```

Struct init:
```rust
devfund_address: None,
devfund_percent: 0,
```

---

## Notes for a future restore

- `block_template_ctr` is **kept** in the live code — it is also used for the
  `% 200` log-throttle in `stratum.rs` and as the `(v+1) % 10_000` job counter.
  Only the devfund *comparisons* against it were removed.
- If restoring, fix the crash-loop first: the `listen()` guard must not be able
  to `return Ok(())` before processing at least one job, and `devfund_percent == 0`
  must short-circuit the whole swap so the counter wrapping to 0 can't wedge the
  connection.
- The previous devfund/pool address was
  `keryx:qp0vrxc0k5w0pcyem6vau2pjgztje880tsm239rywtm7l7uv2pcxzq55n8khs`.
