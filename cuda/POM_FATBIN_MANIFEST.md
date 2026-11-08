# PoM CUDA fatbin manifest

H15/H16 solver candidate: `pom_mine_v5_ncf_fast_seeded` uses exact warp XOR
reductions and a division-free pad-step calculation for the consensus K=256.
SM80 and SM86 now use L1-cached asynchronous tile loads. All five established
entry points on each of SM80, SM86 and SM120 retain instruction-identical SASS,
as does the SM120 fast kernel. The selected SM80/SM86 fast kernels match the
measured cache prototypes; SM86 also matches the independently benchmarked
embedded build.

The preceding solver qualification covered a physical RTX 3070, RTX 5080,
RTX 5070 Ti and two RTX 5090s. The cache extension adds physical CMP 170HX
and RTX 3070 qualification: each passed 700 individual exact-target/one-below
checks and 20 batch-target checks against independent CPU proofs over five
pad extents, including current and H16 tier-0 sizes. The H16 gate remains
disabled; unreleased H16 network behavior was not tested.

Matched H15 mining comparisons passed on CMP 170HX and RTX 3070. The final
Linux binary passed traced-answer/proof/audit/custody/restart/mining-resume
validation on an RTX 5090. The equivalent CMP workflow remains incomplete:
the power guard stopped replay after custody upload and fetch after restart.
Do not interpret CMP hashing qualification as full CMP workflow qualification.
These are local fixtures, not new production-pool acceptance measurements.
See `docs/benchmarks/2026-11-05-h15-cmp-cache.md` for measured gains and limits.

- Compiler: NVIDIA CUDA 13.4, `nvcc V13.4.92`
- Source: `src/pom_mine.cu`
- Source SHA-256: `30c32bf736a666ecdb66663a9f733ce6cf5b0a934962fdb910179e7ea5ab5071`
- Artifact: `cuda/pom_mine.fatbin`
- Artifact SHA-256: `7777e6f4f219b4ed6b1487cf20d5c8fb89f022a36b79a8dc2e24c2df62015140`
- Artifact size: 5097224 bytes
- Native SASS: sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110,
  sm_120, sm_121
- Forward-compatible PTX: compute_75 and compute_80

The host selects the new entry point only for context-pad jobs on sm_80, sm_86 and sm_120.
Other architectures and pre-H15 jobs retain the established solver. Setting
`KERYX_POM_V5_FAST=0` selects the established path for diagnosis.

Rebuild with `cuda/regenerate-pom-fatbin.sh OUTPUT BASELINE`; retain the baseline
for the established-kernel SASS comparison. H16 activation remains unscheduled.

CMP 170HX qualification extends the exact fast solver to SM80. SM80/86 tile copies
use L1-cached asynchronous loads; the SM120 fast kernel and all established
entry points above remain instruction-identical. Full-miner validation receipts
are recorded separately; this is a local optimization candidate.
