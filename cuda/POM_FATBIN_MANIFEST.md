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

Before first use on a card, the host cross-checks both widths against the
established fast kernel on the live job and keeps the faster width that matches;
a mismatch leaves the card on the established kernel. The lock-step kernels are
selected only on compute capability 8.0 and newer. `KERYX_POM_V5_LOCKSTEP=0`
disables them. All earlier entry points (including `pom_mine_v5_ncf_fast_seeded`
with its SM80/SM86 cached asynchronous tile loads) are retained unchanged in
behavior and remain the fallback.

- Compiler: NVIDIA CUDA 13.4, `nvcc V13.4.92`
- Source: `src/pom_mine.cu`
- Source SHA-256: `30c32bf736a666ecdb66663a9f733ce6cf5b0a934962fdb910179e7ea5ab5071`
- Artifact: `cuda/pom_mine.fatbin`
- Artifact SHA-256: `7777e6f4f219b4ed6b1487cf20d5c8fb89f022a36b79a8dc2e24c2df62015140`
- Artifact size: 7989624 bytes
- Native SASS: sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110,
  sm_120, sm_121
- Forward-compatible PTX: compute_75 and compute_80

The host selects the established fast entry point for context-pad jobs on sm_80, sm_86 and
sm_120 when the lock-step walk is unavailable or disabled. Other architectures and pre-H15
jobs retain the established solver. `KERYX_POM_V5_FAST=0` selects the established path for
diagnosis.

Rebuild with `cuda/regenerate-pom-fatbin.sh OUTPUT BASELINE`; retain the baseline
for the established-kernel SASS comparison.
