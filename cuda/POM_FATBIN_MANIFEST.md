# PoM CUDA fatbin manifest

0.16.2 lock-step walk: `pom_mine_v5_lockstep_seeded` (four nonces per warp) and
`pom_mine_v5_lockstep2_seeded` (two nonces per warp) walk the H15/H16 context-pad
jobs over one contiguous copy of the model weights in the walk's canonical layout.
The nonces of one warp advance in lock-step, one lane group per nonce, so the
snippet fold is computed once per nonce instead of once per lane, and tile copies
are coalesced. The lock-step kernels require the contiguous canonical blob: on
mining-only cards the raw weight upload, on inference cards the engine's own
weights when the engine places them in canonical order (`KERYX_CANONICAL_WEIGHTS`,
default on).

Before first use on a card, the host cross-checks both widths and tile layouts
against the established fast kernel on the live job and times the exact candidates;
a mismatch leaves the card on the established kernel. The lock-step kernels are
selected only on compute capability 8.0 and newer. `KERYX_POM_V5_LOCKSTEP=0`
disables them. All earlier entry points (including `pom_mine_v5_ncf_fast_seeded`
with its SM80/SM86 cached asynchronous tile loads) are retained unchanged in
behavior and remain the fallback.

- Compiler: NVIDIA CUDA 13.4, `nvcc V13.4.92`
- Source: `src/pom_mine.cu`
- Source SHA-256: `f6deab08bae320390d4cad163eb1f042a4f94a3515d4e91d58e1e8f67c1a8b24`
- Artifact: `cuda/pom_mine.fatbin`
- Artifact SHA-256: `619177f55e797dcdc835158e73195fb10e49bba4c371a3abee5dd1c041218925`
- Artifact size: 10166304 bytes
- Native SASS: sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110,
  sm_120, sm_121
- Forward-compatible PTX: compute_75 and compute_80

The host selects the established fast entry point for context-pad jobs on sm_80, sm_86 and
sm_120 when the lock-step walk is unavailable or disabled. Other architectures and pre-H15
jobs retain the established solver. `KERYX_POM_V5_FAST=0` selects the established path for
diagnosis.

Rebuild with `cuda/regenerate-pom-fatbin.sh OUTPUT BASELINE`; retain the baseline
for the established-kernel SASS comparison.

The optional `pom_mine_v5_inplace2_seeded` and `pom_mine_v5_inplace4_seeded`
kernels preload the matrix fragments before reusing one tile buffer, reducing
shared memory from 3 KiB to 2 KiB per nonce. Merkle parents from independent
nonces share the finish phase. A warmed, interleaved per-card comparison selects
the layout; a new layout must beat the established one by at least 0.5%.
`KERYX_POM_V5_INPLACE=0` excludes the new layouts for diagnosis. All 168
established native functions across the 12 architectures are instruction-identical
to the v0.16.2 baseline; the additional layouts have separate entry points.
