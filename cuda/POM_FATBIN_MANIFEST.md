# PoM CUDA fatbin manifest

This manifest identifies the modern walk image committed for the v0.13.0 optimization work. Build
it with `cuda/regenerate-pom-fatbin.sh`; the script refuses incomplete architecture coverage and can
compare all established GA100 entry points against a baseline before an artifact is installed.

- Compiler: NVIDIA CUDA 13.2, `nvcc V13.2.86` (adds pom_build_v4_tile_addrs_lut + pom_mine_v4_tc_sidecar_addr_seeded; established sm_80 SASS gate passed)
- Source: `src/pom_mine.cu`
- Source SHA-256: `c719502eb71af94d84632645e08997223599911bb4d92b38f3c9c6d819fdf4ae`
- Artifact: `cuda/pom_mine.fatbin`
- Artifact SHA-256: `cf92a92367251f078314b21d35fd906acd8cdc3b62a146c1d9d93c29839fef1b`
- Artifact size: 4,388,256 bytes
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
