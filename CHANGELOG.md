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
- **Backends without a trace engine** (AMD OpenCL/Vulkan, Apple Metal, CPU) no longer answer
  private requests after the gate: they refuse instead of sending an answer the node rejects.
- **Pool mining** is unchanged (pools seal and submit the answers they dispatch).
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
