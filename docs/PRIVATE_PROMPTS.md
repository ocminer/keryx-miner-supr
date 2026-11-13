# Private prompts and pool integration (v0.16.7)

The reference is upstream [keryx-miner v0.6.2-PoM](https://github.com/Keryx-Labs/keryx-miner/releases/tag/v0.6.2-PoM),
commit `3e82e42c1b8e8195c50ae0b158ac1bd17b33f6d1` (`src/client/grpc.rs`,
`src/client/stratum/ai.rs`, and the traced retry in `src/llm.rs`).

## Bytes and trace identity

A pool opens the node's private envelope and forwards those original prompt
bytes, base64-encoded in `mining.ai_request` parameter 4. A prompt starting
with `00 4B 5A 01` contains raw DEFLATE after that marker. The miner reads at
most 1,048,576 decompressed bytes, then applies lossy UTF-8 conversion, exactly
as upstream does. A damaged stream retains partial output, or the raw deflated
bytes if it failed before producing output. This decoder has been shared by
pool and solo paths since v0.16.4.

The Stratum decoded-wire cap is separately 1,048,576 bytes. Decompression is
bounded before UTF-8 conversion: a replacement character can expand an invalid
byte to three UTF-8 bytes. The private traced admission bound allows this
conversion; plain text RPC limits remain unchanged.

Always set `ext.trace.prompt_len` to the original opened byte length **before
inflation**, as represented by the request's plaintext-length metadata. Do not
substitute inflated text, base64 length or encrypted envelope size. As upstream
does, an omitted value defaults to the original dispatched byte count; both
forms are clamped to the 65,536-byte consensus private-payload bound. Invalid
numeric metadata is refused. Preserve model/tier, request hash, custody URLs
and all audit messages. Existing explicit-length handling retains embedded
NUL bytes in traced prompts.

A pool's old 65,536-byte limit on *inflated text* prevents valid requests from
reaching the miner. Remove that routing filter for compatible versions while
retaining node consensus envelope validation and worker readiness checks.
Prefer forwarding original compressed bytes over decoding and re-encoding.
Older unknown miners must not be assumed to implement this contract.

## Context overflow

Every traced request first attempts the full decoded prompt. Only
`PromptTooLong` can trigger one retry. Its byte budget is
`model.ctx_floor - (max_tokens + 512)`, with saturating arithmetic, and the
original `prompt_len` must fit that budget. The retry keeps the head and tail
on UTF-8 boundaries with the exact separator `\n…\n`. A request that already
fits is not shortened. Other failures are not retried this way.

The retry stays on the same leased card, with the same original length,
request-derived prefix and traced-answer/audit path. Only that card pauses;
it resumes after inference, including refusal paths. The rule does not imply
that every 1 MiB prompt fits every model or GPU context. A retry may still be
refused. No fallback to untraced inference or mining-only service is allowed.

## Refusals and deployment

Give each dispatch an id. A valid request that cannot be served receives
`{"id":id,"result":null,"error":[46,"reason",null]}`; malformed requests receive
error 20. A notification without an id cannot receive a matching refusal.
A refusal is not a mining-share ACK or an on-chain answer. Pools can promptly
try a different eligible worker with bounded attempts and the real deadline.

Use off-chain probes for 74,739/79,150 decoded bytes, 70–500 KiB inputs, UTF-8,
raw/compressed forms, the inflate boundary, and actual context overflow.
Verify traced answers, all opening phases, custody/replay and mining resumption.
Test refusals and redispatch separately. Installing the miner cannot repair a
router which still discards the request before dispatch.
