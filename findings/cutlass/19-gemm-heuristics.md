# Gate 2c: official plain-GEMM kernel discovery on B200

Gate 2c replaces the manual Cartesian tuning of findings 16-18 with NVIDIA's
supported discovery workflow: `nvidia-matmul-heuristics` generates kernel
candidates for the exact B200 BF16 problems, `cutlass_profiler` measures
them, and the matched project runner confirms the selected dispatch against
`torch.mm` under the production cold-L2 protocol.

**Result: Gate 2c passes.** Two independent confirmation runs each select a
CUTLASS dispatch within 5% of `torch.mm` in all 18 B200 production cells
(run 4 worst ratio 1.0115, run 5 worst ratio 1.0488, worst cell across both
runs 1.049).

Run it with:

```text
make modal-cutlass GATE=ordinary-gemm-tuning PHASE=discover
make modal-cutlass GATE=ordinary-gemm-tuning PHASE=confirm RUN=<n>
```

The evidence packets are under
`benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning/gate-2c-discovery/`
and `.../gate-2c-confirm-<run>/`.

## Scope

- B200 (`sm_100a`) only. Hopper discovery is deferred to Gate 8.
- Both primary model shapes: `(V, D) = (151936, 4096)` and `(128256, 8192)`.
- The full hidden-state sweep `H in {1,2,4,8,16,32,64,128,256}`.
- `torch.mm` is the sole strong matmul baseline.
- CUTLASS 4.6.1 at the pinned commit with `nvidia-matmul-heuristics==0.1.0.27`.

The GEMM is `W[V,D] @ H[D,H]` with BF16 inputs and output, FP32
accumulation, beta=0, A/B/D all row-major (`layout="ttt"`), matching the
production provider. N is padded to a multiple of 8 for the BF16 TMA store
alignment, so H in {1,2,4,8} share the padded problem N=8. The 12 unique
problems live in the checked-in `gemm_problems_b200.json`; the runner
verifies the file against `ordinary_gemm_common.heuristic_problems()` on
every run.

## Discovery: heuristic generation and profiler search

The image builds two profiler instances at image-build time (the builders
have no GPU, so the heuristics GPU is pinned to `B200`):

1. Top 16 configurations per problem for all 12 problems
   (`gemm_problems_b200.json`).
2. Top 32 configurations for the two N=256 problems
   (`gemm_problems_b200_n256.json`), the stop-rule expansion for the shapes
   that failed the top-16 selection.

The profiler profiles each emitted test case for a fixed 50 ms with
verification disabled. The main testlist has 262 rows (152 unique test
cases after deduplicating identical kernel+argument rows); 110 unique cases
profiled successfully. The expansion adds 42 rows per N=256 problem (24
unique kernels each).

### Coverage audit of the heuristic emission

- **2-SM Blackwell schedules**: present in all 12 problems.
- **1-SM schedules**: absent everywhere, including the top-32 expansion.
  nvidia-matmul-heuristics suggested only 2-SM kernels for these
  very-large-M, small-N shapes. The six manual variants
  (`tile-*-native` = `KernelTmaWarpSpecialized1SmSm100`, plus the
  automatic-schedule controls) and the explicit
  `heur-128x128x64-1sm-c1x2x1` cluster-(1,2,1) control remain in the
  confirmation as the 1-SM control families.
- **Nontrivial static M-axis clusters**: present in all 12 problems
  (cluster shapes (2,1,1), (2,2,1), (4,1,1)). No 1-CTA cluster and no
  (1,2,1) shape was emitted; the latter is covered by the explicit control.
- **StreamK**: emitted for all 12 problems but profiled only with cluster
  (2,1,1). Every StreamK test case with cluster (2,2,1) or (4,1,1) was
  silently omitted by the profiler (return code 0, no CSV rows): 42 unique
  cases across 8 problems. The generation log shows the kernels were
  emitted, so the rejection happens at profiler problem-validation or
  `can_implement`. This is the recorded exclusion for the
  StreamK-plus-multi-CTA-cluster family.
- **Flexible clusters**: not emitted. The heuristic generator
  (`generate_sm100_from_heuristics_configs` at the pinned commit) only
  builds static-cluster `TileDescription`s, so the official workflow cannot
  produce them. Flexible clusters provide launch-fallback behavior rather
  than a new data-movement pattern; the static families that a flexible
  cluster would fall back to are all measured. Recorded as the formal
  exclusion.
- **Split-K**: every emitted configuration has `split_k_slices=1`; the
  heuristic suggested no split-K for these shapes.
- **Raster order and swizzle**: `along_m` everywhere except N=256
  (`along_n`); `swizzle_size=1` in every emitted row of both testlists.
- **CTA tile N**: the emission never exceeds 192
  (`cta_tile_n in {64, 128, 192}`). This gap mattered: the family that
  finally closed the two H=256 cells uses N-tile 256, which came from an
  explicit coverage control, not from the heuristic.

## Discovery result (profiler, warm-L2, 50 ms fixed duration)

After the N=256 expansion, the per-case oracle stays within 5% of the
same-run cold-L2 `torch.mm` in all 12 padded problems (worst 1.013). The
profiler's warm-L2 ranking did not transfer to the cold-L2 confirmation at
H=256: the warm-L2 winner (`256x128x64` 2-SM) measured 1.09-1.34 cold in
the confirmation runs. Gate decisions therefore use only the matched
cold-L2 confirmation below.

## Confirmation: matched cold-L2 protocol

Each confirm run measures, in one process on one host:

- `torch.mm` at every (shape, H) with the production padding policy, in
  cold- and warm-L2 states (25 warmup + 100 measured repetitions, CUDA
  events, preallocated buffers, deterministic per-case seeds).
- Every extension CUTLASS candidate with cold-L2 timing and exactness
  checks: the six manual controls, the transplanted heuristic winners, and
  the small-N GEMV where applicable (H<=8).

Baseline and candidates are interleaved in the same process so Modal
host-class variance cancels in the ratio. Runs 1 and 2 measured them in
separate containers; run 2's torch.mm at V=128,256 H=256 entered a slow
state for ~80 repetitions (0.75 ms vs the typical 0.43-0.44 ms at identical
reported clocks, flipping to fast mid-sweep) while its warm state stayed
fast. That anomaly motivated the same-process protocol used from run 3 on.
The mechanism is not established; per-run slow states remain possible and
are handled by cross-run agreement.

### Transplanted winner families

Kernel families were transplanted into `greedy_provider.cu` as
`PlainGemmVariant` instantiations parameterized by CTA tile, cluster, and
schedule, with raster order and `max_swizzle_size=1` as runtime arguments
(matching the profiler runs). From the top-16 search:

- `heur-256x64x128-c2x1x1`, `heur-128x64x128-c2x1x1`,
  `heur-256x128x64-c2x1x1`
- `heur-256x64x64-c4x1x1`, `heur-128x64x64-c4x1x1`,
  `heur-256x64x128-c4x1x1`, `heur-256x128x64-c4x1x1`

From the N=256 top-32 expansion: `heur-256x128x128-c{2,4}x1x1`,
`heur-128x128x64-c4x1x1`, `heur-128x128x128-c4x1x1`,
`heur-256x192x64-c{2,4}x1x1`. Explicit coverage controls:
`heur-128x128x64-1sm-c1x2x1` (the cuBLAS-style 1-SM family with weight
multicast across hidden-state tile CTAs) and the full-H CTA tiles
`heur-128x256x64-1sm(-c2x1x1)` and `heur-256x256x64-c2x1x1`. Every name has
an `-rn` suffix variant selecting `along_n` raster.

The StreamK winner at V=128,256 N=8/16 was not transplanted: its
default-scheduler sibling is within 0.3-1.1% in the profiler measurements,
and avoiding StreamK keeps the extension on the standard persistent
scheduler.

### Correctness

All tensor-core candidates match the same-case `torch.mm` output
bit-for-bit. The small-N GEMV uses serial-K FP32 accumulation and differs
from cuBLAS split-K accumulation by up to 1 bf16 ULP (max abs difference
1.0-2.0 at output magnitude up to ~400), which the packet records as
within-rounding. Zero candidates were rejected at build, `can_implement`,
or launch in runs 4-5 (the cluster-(1,2,1) control is legal at gemm_n=8).

## Result

Run 4 and run 5 each pass all 18 cells with zero rejections and 728/728
candidates within rounding:

| V | D | H | Run 4 selected (ratio) | Run 5 selected (ratio) |
|---:|---:|---:|---|---|
| 128,256 | 8,192 | 1-64 | `128x64x128-c2x1x1` (0.83-0.98) | same family (0.83-0.98) |
| 128,256 | 8,192 | 128 | `256x128x64-c2x1x1-rn` (1.012) | `256x128x128-c2x1x1` (1.017) |
| 128,256 | 8,192 | 256 | `256x256x64-c2x1x1` (1.004) | `256x256x64-c2x1x1-rn` (1.022) |
| 151,936 | 4,096 | 1-64 | `128x64x128-c2x1x1(-rn)` (0.91-0.94) | same family (0.90-0.93) |
| 151,936 | 4,096 | 128 | `256x128x64-c4x1x1-rn` (1.000) | `256x128x64-c4x1x1` (1.010) |
| 151,936 | 4,096 | 256 | `256x256x64-c2x1x1` (0.985) | `256x256x64-c2x1x1` (1.049) |

The two H=256 cells were the last to close. Runs 1 and 3 (before the full-H
control) failed them: V=128,256 at 1.23-1.25 and V=151,936 at 1.09-1.17. cuBLASLt
logs identify the torch.mm kernels there as algoId 66 with
`MATMUL_TILE_128x256`/`MATMUL_TILE_128x192` (V-tile 256/192 x H-tile 128 in
FMMS orientation, clusters on the hidden-state axis, `customOption` 1 and
3). The full-H CTA family (N-tile 256, 2-SM `256x256x64` cluster (2,1,1))
closed both cells in both final runs. The per-case oracle in the discovery
packet shows the same family leading the profiler's warm-L2 ranking.

The simplified dispatch (smallest variant set within 1% of the per-case
oracle) uses 6-7 variants per run out of 34 measured; see `dispatch.csv`.

## What this gate does not prove

- The dispatch is approved for the **plain** GEMM only. Gate 2d must
  re-derive accumulator ownership for any changed tile/schedule and
  revalidate the fused candidate epilogue before any FMMS use.
- The confirmation is BF16 `ttt` with beta=0 at the two primary model
  shapes on B200; it does not cover other layouts, dtypes, or GPUs.
- torch.mm at V=128,256 H=256 showed a bimodal cold-L2 state in run 2
  (0.75 ms vs 0.43 ms) with identical reported clocks. The cause is not
  established (possibly a cuBLASLt workspace/algo state or host power
  management; the same-process protocol plus cross-run agreement bounds
  its effect on the decision).
- The profiler's 50 ms warm-L2 ranking is a discovery aid only; it
  mis-ranked the H=256 winner relative to the cold-L2 confirmation.
- StreamK with multi-CTA clusters and flexible clusters are formally
  excluded (profiler rejection and generator limitation respectively), not
  measured.

## Handoff to Gate 2d

The approved plain-GEMM dispatch families are the schedule donors for the
fused epilogue: `128x64x128` 2-SM cluster (2,1,1) for H<=64,
`256x128x64`/`256x128x128` 2-SM (cluster (2,1,1) or (4,1,1)) at H=128, and
`256x256x64` 2-SM cluster (2,1,1) at H=256, with `along_m` raster below
N=256. Gate 2d must rerun Gate 1a (accumulator ownership) on B200 for each
donor tile before re-running Gates 1b-2a, then measure the greedy fused
kernel against these plain winners and `torch.mm` plus argmax.
