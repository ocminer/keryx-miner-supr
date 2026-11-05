# PoM CUDA fatbin manifest

H15/H16 solver candidate: the new `pom_mine_v5_ncf_fast_seeded` uses exact warp XOR
reductions and a division-free pad-step calculation for the consensus K=256.
Established entry points are retained. The rebuild compared all five established
sm_80 kernels against the previously committed image and found identical SASS.
GPU qualification is recorded separately before this candidate is promoted.

- Compiler: NVIDIA CUDA 13.4, `nvcc V13.4.92`
- Source: `src/pom_mine.cu`
- Source SHA-256: `d6d41e16f60c0f660a2f7636117a32b1ab9bdc4f8adb9df601aff5574a39de5b`
- Artifact: `cuda/pom_mine.fatbin`
- Artifact SHA-256: `e0d40f6c09d6d638a602c0db5f31db41e4b23b0863f3e62531472790153378a7`
- Artifact size: 5097208 bytes
- Native SASS: sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110,
  sm_120, sm_121
- Forward-compatible PTX: compute_75 and compute_80

The host selects the new entry point only for context-pad jobs on sm_86 and sm_120.
Other architectures and pre-H15 jobs retain the established solver. Setting
`KERYX_POM_V5_FAST=0` selects the established path for diagnosis.

Rebuild with `cuda/regenerate-pom-fatbin.sh OUTPUT BASELINE`; retain the baseline
for the established-kernel SASS comparison. H16 activation remains unscheduled.
