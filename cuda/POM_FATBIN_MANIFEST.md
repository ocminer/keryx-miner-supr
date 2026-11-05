# PoM CUDA fatbin manifest

H15/H16 solver candidate: the new `pom_mine_v5_ncf_fast_seeded` uses exact warp XOR
reductions and a division-free pad-step calculation for the consensus K=256.
Established entry points are retained. The rebuild compared all five established
sm_80 kernels against the previously committed image and found identical SASS.
GPU exactness passed on a physical RTX 3070, RTX 5080, RTX 5070 Ti and two RTX 5090s.
Each card passed 700 individual exact-target/one-below checks and 20 batch-target
checks against independently built and verified CPU proofs across five pad extents,
including current and H16 tier-0 sizes. The H16 gate remains disabled.

The established sm_86 and sm_120 kernels also retain instruction-identical SASS
(all five entry points on each architecture). Matched current-era A/B/B/A mining
runs and the traced-answer/audit/custody/restart/mining-resume check passed.
Unreleased H16 network behavior and other GPU architectures were not runtime-tested.

- Compiler: NVIDIA CUDA 13.4, `nvcc V13.4.92`
- Source: `src/pom_mine.cu`
- Source SHA-256: `f7ddd414e51d7ce98cda206cb0ce01b2d21e4a9e612be608457bf0623853fe29`
- Artifact: `cuda/pom_mine.fatbin`
- Artifact SHA-256: `c099b3c3ef6334499502b24d1d9ad99f2d1d42f956dfc53f8903bf4a2f3d5f28`
- Artifact size: 5097208 bytes
- Native SASS: sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110,
  sm_120, sm_121
- Forward-compatible PTX: compute_75 and compute_80

The host selects the new entry point only for context-pad jobs on sm_80, sm_86 and sm_120.
Other architectures and pre-H15 jobs retain the established solver. Setting
`KERYX_POM_V5_FAST=0` selects the established path for diagnosis.

Rebuild with `cuda/regenerate-pom-fatbin.sh OUTPUT BASELINE`; retain the baseline
for the established-kernel SASS comparison. H16 activation remains unscheduled.

CMP 170HX qualification extends the exact fast solver to SM80. SM86 tile copies
use L1-cached asynchronous loads; the SM80/SM120 fast kernels and all established
entry points above remain instruction-identical. Full-miner validation receipts
are recorded separately; this is a local optimization candidate.
