# keryx-miner-supr 0.16.0 — H16 preparation (unreleased)

- Mirrors the upstream H16 context-pad table: 200 MiB less per tier after the
  node's `pad_margin_activation` gate. Mining and CPU proofs select the table
  from each job's DAA score, including jobs across a reorganisation.
- Includes the qualified CPU/IPFS and exact CUDA solver improvements from the
  preceding development session, plus cached SM80/SM86 tile loads.
- Adds a finite staging-gate override and a CUDA transition regression covering
  pad replacement, reuse and rollback. No model or inference-engine change.
- **Activation is not yet scheduled upstream.** Mainnet and testnet defaults
  remain disabled until an authoritative activation value is published. This
  preparation must not be advertised as a scheduled, automatic mainnet upgrade.

# keryx-miner-supr 0.15.8 — Tesla V100 / Pascal H15 traces and per-card tier pinning

(0.15.7 was an internal build and was not released.)

- **Tesla V100 / Volta (legacy package): H15 traced self-test fixed.** On Volta every tier failed with
  `trace read failed (-5), N positions traced for M fed`, the card was withdrawn and mined 0 H/s.
  The inference engine now always uses the quantized matrix kernels that record the trace; before,
  Volta handed prompts of 64 tokens or more to cuBLAS, so the prompt was never traced. Turing and
  newer cards already used these kernels, so their inference path is unchanged.
- **Tesla V100 / GTX 10xx (Pascal): Q4_K/Q5_K models pass the H15 self-check.** Cards without integer tensor cores rounded two K-quant scale products differently from the verifier, so models such as Qwen3.5-9B failed with `point 0: Mismatch` on prompts. The engine now rounds them the same way (llama patch 0006). Verified on a Tesla V100: tier 0 and tier 1 pass and mine with 0 rejected shares.
- **Multi-card rigs: no more false `point 0: Mismatch`.** When several cards loaded different models in one process, a card's self-check could read another model's possession index while the other cards were starting up, fail, and withdraw a healthy tier. Each opening now uses the index of its own model.
- **`--force-model` pins one tier per card.** Use `t0`–`t4` (or the old names): `--force-model t0,t1,t2,t3,t4` loads tier 0 on GPU0, tier 1 on GPU1 and so on. With `--cuda-device` the list maps onto the selected cards in that order. A pinned card keeps its tier: a failed self-check is retried on that card instead of moving the tier onto another card.
- `--help` shows the current model for every tier name.
- `KERYX_LLAMA_CTX` is capped at the largest context a trace can cover.
- With the interactive dashboard the miner now writes a plain log file, `keryx-miner.log` (rotated
  at 64 MiB). `--plain-log-file <path>` or `KERYX_PLAIN_LOG_FILE` choose another file, an empty
  path disables it; classic mode (HiveOS, mmpOS, `--no-tui`) writes a file only when asked.
- When the miner stops itself, the reason is printed on the console after the dashboard closes.
- The dashboard no longer highlights healthy `error: false` status text as an alert.
- Stale analysis: a stale, duplicate or low-difficulty share is logged with the age of its job at submit time and the pool's reply time, and every 200 answered shares the miner logs a summary (job age p50/p90/p99 for accepted and stale shares, pool reply time).

# keryx-miner-supr 0.15.6 — AMD H15 mining and verifiable inference

- AMD now supports H15 context-pad mining, traced inference and multi-phase audits through the bundled Vulkan engine.
- Mining and inference share one resident model on the same card. Mining pauses during inference and resumes afterward; a dedicated inference-only card is not required.
- Compact trace records can be uploaded, fetched by a fresh compatible process and replayed for audits. Proof verification remains unchanged.
- Large Vulkan allocations are split into bounded buffers, allowing additional mining cards to load models on drivers with per-allocation limits.
- Windows AMD device selection accepts both architecture identifiers and retail GPU names. Its launcher works from another directory and preserves the exit status.
- HiveOS statistics correctly convert both `Khash/s` and `khash/s`, and retain fractional kilohashes for rates below 1 kH/s.
- Linux AMD packages include AVX and baseline-CPU Vulkan libraries. Keep the executable and its matching libraries together when upgrading.
- Earlier AMD benchmark figures remain explicitly labelled as pre-H15 results. RDNA1 remains excluded from H15 mining; Apple Metal H15 support is unchanged.
- Release packages include the existing project and third-party licenses. NVIDIA retains its established CUDA kernels and H15 proof path.

# keryx-miner-supr 0.15.5 — Windows: H15 self-test fixed

**Every Windows NVIDIA user of 0.15.4 must update.** On Windows no card passed the H15 traced
self-test, so no model became ready: the miner did not mine and printed
`OPoI: no models ready — skipping AiRequest scan` for every block. Linux and the Docker image were
not affected.

- **Windows: the inference engine now gets its trace settings.** The miner turns off CUDA graphs
  and KV-cache rotation for the traced engine via the process environment. On Windows the setting
  only reached the operating-system environment and not the C runtime copy that the engine DLL
  reads, so the traced self-test failed on every card. The miner now sets these variables through
  the C runtime as well, checks at startup that the engine can see them, and the Windows build
  gate tests this.
- **The "no models ready" warning now gives the reason and no longer floods.** It is printed once,
  then at most every 60 s or when the reason changes. The reason is one of: model download in
  progress, no model file staged (with the staging error), self-test pending, self-test FAILED on
  GPU n (with the cause), or this build cannot produce traces (AMD/Vulkan, macOS). The pool mode
  "mining suspended" warning works the same way.

# keryx-miner-supr 0.15.4 — H15 serving fixes

Fixes found by the 0.15.3 test matrix on our rigs.

- **Tier 4 (Kimi-Linear-48B) can be served again.** Its MLA cache now uses a Q8_0 K / f16 V cache, like keryx v0.5.8. 0.15.3 fell back to an f16/f16 cache, which the trace cannot commit. The H15 self-test therefore correctly refused tier 4, and those cards did not mine it.
- **A card is declared to the pool only after its traced self-test passed.** In 0.15.3 the plain warm-up answer already marked the model serveable, about two seconds before the traced check finished.

# keryx-miner-supr 0.15.3 — H15 private inference (traced answers)

**Mandatory for every solo miner that answers private inference requests.** Since the H15 gate
(mainnet DAA 140,115,000, 2026-10-30) keryxd 1.6.6 requires every private answer to carry a
verifiable inference trace, and requesters seal with key schedule 2. 0.15.0–0.15.2 mine H15 blocks
correctly but could not open schedule-2 requests and sent answers without a trace, which the node
rejects (`ResponseWithoutTrace`) — a missed answer costs a service-bond strike. Mining is unchanged.

- **Key schedule 2.** Private requests are opened with schedule 2 (falling back to schedule 1), and
  the answer is sealed with the per-responder key of the schedule that opened it. 66-digit V2
  request secrets are understood everywhere.
- **Traced answers.** Every private answer is generated on the in-process CUDA engine with the
  canonical trace armed: position leaves, sequence leaves (KV cache, attention softmax mass,
  recurrent state) and the token ids of the answer. The `TraceCommitment` (extension `0x81`) is
  attached to the AiResponse **before** it is signed, and the response is only sent once its trace
  is safely on disk.
- **Audits and openings.** The miner follows the node's draw for every traced response (seed
  block, audit, deadline — and the node's own `GetAwaitedOpenings` answer on keryxd ≥ 1.6.6),
  replays the drawn positions on the card that generated the answer, checks every opening with
  the node's verifier code and submits it (trace, attention phases, recurrent and residual
  pieces). Pending traces survive restarts and reconnects (`<escrow state>.traces/`), and the
  chain is rebuilt after a restart so no audit is lost.
- **Engine.** `libkeryx-llama.so` / `keryx-llama.dll` is now engine ABI 4: four new exports for
  traced generation and sequence leaves, and the verifier-exact attention/RoPE kernels of keryx
  v0.5.8. Always ship the engine that comes with this release next to the miner.
- **Serving self-test gates mining.** Before a card mines its tier it must answer a private
  request the way keryxd 1.6.6 requires: a traced generation whose openings (trace, attention,
  recurrent, residual pieces) pass the node's own verifiers for two seeds. A card or build that
  cannot (AMD OpenCL/Vulkan, Apple Metal, CPU, or a failed check) does not mine and says why — mining
  a tier that cannot be served only collects service-bond strikes. `mining.hello` / `mining.telemetry`
  carry `"trace": 1` once the self-test passed on this rig.
- **Pool path (opt-in per request).** A pool can ask for a traced answer with an optional 8th
  `mining.ai_request` parameter `{"trace":{"tier","prompt_len","chain","attention"}}`; the answer then
  carries the hex `TraceCommitment` and the `answer::encode` body for the pool to seal, attach and sign.
  The rig keeps the compact trace on disk for 9 h and builds the unsigned openings when the pool sends
  `mining.ai_opening`. Requests without the parameter are answered exactly as before.
- CUDA graphs and llama.cpp's KV-cache rotation are disabled in-process (as upstream keryx 0.5.8):
  the trace hooks do not run inside a replayed CUDA graph and attention openings need unrotated rows.
- Vendored `inference` / `verifiable-inference` crates are byte-identical to keryx-miner
  v0.5.8-PoM (= keryx-node 1.6.6 consensus code). The AiResponse is resubmitted until the mempool
  accepts it (a response sent before its request's audit arms is refused once).

# keryx-miner-supr 0.15.0 — H15 hard fork

**Mandatory update before the Keryx H15 hard fork** (mainnet DAA 140,115,000, expected around 2026-10-30 15:00 UTC). From that block on, every block needs a PoM v5 proof, a block-task root in its header and — for block candidates — a verifiable trace opening, and pools switch to stratum v4. Miners below 0.15.0 produce invalid blocks after the gate and are refused by pools that enforce the version floor.

The switch is driven by the block DAA in each job, never by wall-clock time: start 0.15.0 now and it keeps mining exactly like 0.14.7 until the gate, then changes over on its own. No restart, pool, wallet or model change is needed.

## What changes at the gate

- **PoM v5 context pad.** The possession walk reads an additional per-tier context pad next to the model weights. It is generated on the GPU and walked with the same fast kernels as before (the chaseless kernel where the card supports it). The pad takes VRAM: while mining, the inference context is shrunk to a small block-task context and restored when the pool sends an inference request. A card with too little VRAM for its tier's pad logs a clear warning and stays idle instead of mining invalid work — move it to a lower tier with `--force-model`.
- **Block task.** For every ~64-block window the miner runs a short canonical inference pass on its own resident model and commits the resulting root into the block header before grinding. Block candidates ship a verifiable opening of that trace; the miner checks every opening with the node's own verifier before submitting, and runs a one-time self-check per card and tier when it starts mining post-gate.
- **Stratum v4.** The pool sends the full block header and the window seed; the miner hashes the header itself and submits a 7-field share. Pre-gate v3 jobs keep working unchanged; post-gate header-less (v3) jobs are skipped with a warning because they cannot carry a block-task root.
- **Engine.** `libkeryx-llama.so` / `keryx-llama.dll` (engine ABI 3) adds the canonical trace. Always ship the engine library that comes with this release next to the miner binary.

## Platform notes

- **NVIDIA (Linux modern/legacy, Windows, Docker):** full H15 support.
- **AMD (OpenCL/Vulkan) and Apple Silicon (Metal):** these backends have no trace engine yet and therefore cannot commit a block-task root. They mine normally until the gate and then idle honestly with a warning instead of burning power on jobs that can never become valid blocks. H15 support for these lines follows in a later release.

## Testnet / staging

`KERYX_TESTNET=1` switches the complete activation table to the Keryx testnet parameters (the same role as upstream's `--testnet`). Individual `KERYX_*_ACTIVATION_DAA` overrides still take precedence. Without either, all gates are the mainnet constants.

## Validation

Validated against a post-gate testnet pool and node: every submitted share accepted, block candidates verified by the node's `verify_block_task` and accepted into the chain; mainnet pre-gate mining verified unchanged against the production pool. Byte-exact unit vectors against the node for the v5 pad walk, the header layouts and the block-task opening are part of the test suite.

## Downloads

| Platform | Package |
|---|---|
| NVIDIA Turing and newer, Linux, driver 575+ | `modern`: standalone, HiveOS, mmpos, SMOS |
| NVIDIA Volta/Pascal, older CMP cards, or driver 550–574 | `legacy`: standalone, HiveOS, mmpos, SMOS |
| AMD, Linux | `amd`: standalone, HiveOS, mmpos |
| NVIDIA, Windows | `keryx-miner-supr-windows-nvidia-pom.zip` |
| AMD, Windows | `keryx-miner-supr-windows-amd.zip` |
| Apple Silicon, macOS | `keryx-miner-supr-macos-arm64-0.15.0.tar.gz` |
| Docker, NVIDIA modern | `ocminersupr/keryx-miner-supr:0.15.0` |

Use the matching SHA256SUMS file to verify downloads.
