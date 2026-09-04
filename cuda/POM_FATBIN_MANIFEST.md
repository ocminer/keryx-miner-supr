# PoM CUDA fatbin manifest

This manifest identifies the modern walk image committed for the v0.13.0 optimization work. Build
it with `cuda/regenerate-pom-fatbin.sh`; the script refuses incomplete architecture coverage and can
compare all established GA100 entry points against a baseline before an artifact is installed.

- Compiler: NVIDIA CUDA 13.3, `nvcc V13.3.73`
- Source: `src/pom_mine.cu`
- Source SHA-256: `1e43f51f6237c956a876f625ed57ad00f53e5144dd274f42f8cd5b1d73f5ccb1`
- Artifact: `cuda/pom_mine.fatbin`
- Artifact SHA-256: `4159b8db2aec24be63ef54db160da8bcec020a99cec281155996d3b928cfe639`
- Artifact size: 3,808,272 bytes
- Native SASS: sm_75, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110,
  sm_120, sm_121
- Forward-compatible PTX: compute_75 and compute_80

The candidate passed the ignored `v4_sidecar_folds_offsets_and_winners_match_host` test on a
physical sm_80 CMP170HX: all 2,048 model folds; pre-H10 and H10 offset chains at batches
1/31/32/33/255/256/257; and winners from established TC plus both sidecar combinations matched the
host oracle.

The five established sm_80 kernels were instruction-identical to the previous committed image:

| Entry point | SASS SHA-256 |
| --- | --- |
| `pom_mine_v4_seeded` | `05afd978713adcc08bf9adeca20be4a64668441e921a28b68c93c96ddc03e9b6` |
| `pom_mine_v4_chase_seeded` | `9d4febe977dec3fda13306b3f085434c16ec545d62ffffcd4d89f3f2d8219029` |
| `pom_mine_v4_tc_seeded` | `fa79e2e2ed39e913acc998a313fef698b7f9caa10beb7eb68e018e57354c7410` |
| `pom_mine_v4_ncf_seeded` | `7d973862f65dd331e132998ee365d37992fa60c023b6fbbcaae0a34970b6257e` |
| `pom_seed_h10_batch` | `16456c3fa441368cfeaf684efa2b93e1026cb69a9f60e9ad85b7c82f96497758` |

Example protected rebuild:

```console
cuda/regenerate-pom-fatbin.sh /tmp/pom-all.fatbin cuda/pom_mine.fatbin
```

Only replace `cuda/pom_mine.fatbin` after the SASS gate and GPU exactness test pass.
