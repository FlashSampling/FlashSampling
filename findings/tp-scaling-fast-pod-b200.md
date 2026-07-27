# TP Scaling on Fast-Pod B200 (2026-05-04)

## Motivation

The first TP=8 b200 run (committed at `triton-bench/own/b200/tp8/`) landed on
the slow HGX SKU class identified in `findings/tp2-fanout-symm-mem.md`. At low
H it ran ~2.3x slower than the TP=4 result (which had landed on a fast 2-GPU
pod), making TP=8 look like a hard regression vs TP=4. That is contaminated by
host class, not algorithmic scaling. To get an apples-to-apples TP-scaling
read I rolled the Modal scheduler until one TP=8 run hit a fast pod.

## Method

5 parallel `modal-triton-benchmark BENCH_FN=own GPU=b200 N_PROCS=8` runs with
`POSTFIX=-runX`. Each lands on its own host. After the run, classify by
`metadata.json` NIC layout (fast pod: `nic0 ∈ {mlx5_10, mlx5_11, mlx5_bond_0}`,
n_nics ≤ 3; slow HGX: single-digit `mlx5_0..mlx5_9`, n_nics ≥ 8).

| Run | NIC0 | n_nics | State | FMMS H=1 small (ms) |
|---|---|---|---|---|
| run1 | `mlx5_0` | 8 | bench-done (HGX) | 0.225 |
| run2 | `mlx5_0` | 8 | bench-done (HGX) | 0.218 |
| run3 | — | — | timed out at 1200s before any output | — |
| run3b | `mlx5_4` | 8+ | timed out mid-bench (HGX) | — |
| run4 | `mlx5_0` | 8 | bench-done (HGX) | 0.236 |
| run5 | `mlx5_10` | 2 | bench-done (**fast pod**) | **0.095** |

Hit ratio: 1/5 fast pod for TP=8 (vs ~7/11 for TP=2 in the prior fanout
investigation). 2/5 runs hit the 1200s Modal function timeout, both on slow
HGX, never on the fast pod.

run5 was promoted to the canonical `b200/tp8/` location (local + Modal volume).

## Fast-pod TP scaling

FMMS (Triton) latency, ms, all on fast-pod b200:

### small (V=151,936, d=4,096)

| H | TP2 | TP4 | TP8 | TP4/TP2 | TP8/TP4 |
|---|---|---|---|---|---|
| 1 | 0.139 | 0.096 | 0.095 | 0.69 | 0.99 |
| 2 | 0.144 | 0.102 | 0.097 | 0.71 | 0.95 |
| 4 | 0.145 | 0.102 | 0.095 | 0.70 | 0.93 |
| 8 | 0.145 | 0.102 | 0.095 | 0.70 | 0.93 |
| 16 | 0.145 | 0.102 | 0.096 | 0.70 | 0.94 |
| 32 | 0.148 | 0.107 | 0.094 | 0.72 | 0.88 |
| 64 | 0.156 | 0.114 | 0.129 | 0.73 | 1.13 |
| 128 | 0.181 | 0.125 | 0.184 | 0.69 | 1.47 |
| 256 | 0.283 | 0.172 | 0.220 | 0.61 | 1.28 |

### large (V=128,256, d=8,192)

| H | TP2 | TP4 | TP8 | TP4/TP2 | TP8/TP4 |
|---|---|---|---|---|---|
| 1 | 0.200 | 0.121 | 0.084 | 0.61 | 0.69 |
| 2 | 0.206 | 0.127 | 0.089 | 0.62 | 0.70 |
| 4 | 0.205 | 0.127 | 0.088 | 0.62 | 0.69 |
| 8 | 0.205 | 0.127 | 0.088 | 0.62 | 0.69 |
| 16 | 0.204 | 0.127 | 0.088 | 0.62 | 0.69 |
| 32 | 0.208 | 0.200 | 0.091 | 0.96 | 0.45 |
| 64 | 0.216 | 0.136 | 0.096 | 0.63 | 0.71 |
| 128 | 0.245 | 0.162 | 0.107 | 0.66 | 0.66 |
| 256 | 0.404 | 0.225 | 0.150 | 0.56 | 0.66 |

(TP4 large H=32 = 0.200 ms is an autotune outlier vs neighbors at 0.127-0.136.
The H=32 config picked at TP=4 large is worse than the configs at H=16 and
H=64.)

## Pattern

- **Low H (≤ 16-32)**: clean ~30-40% step from TP=2 to TP=4 on both configs.
  TP=4 to TP=8 gains another ~30% on large, but is flat on small (already
  bandwidth-saturated when V/world is small enough).
- **High H (small only)**: TP=8 is *slower* than TP=4 — 1.13x at H=64, 1.47x
  at H=128, 1.28x at H=256. The fan-out symm-mem write cost scales
  O(world_size): each per-tile winner gets stored to all 8 peer buffers, and
  at H=128/256 the matmul has shrunk enough that those 8 stores dominate.
- **High H (large)**: same effect is suppressed because the matmul stays
  large enough to hide the fan-out cost. TP=8/TP=4 holds around 0.66 across
  H=64..256.

So fan-out remains a net win on the large config across all H, and on the
small config up to H=32. Beyond H=32 small, TP=4 is the better choice.

## Slow-HGX timeouts

2 of 5 TP=8 runs hit the 1200s function timeout before producing useful
output. Both were on HGX. None of the fast-pod TP=2/4 runs in prior
investigations have ever timed out at this benchmark size. The most likely
cause is multi-tenant CPU contention on shared HGX hosts during autotune
(the `.autotune.json` cache investigation is still open — see comments in
`core.py` around `cache_results=True`). Raising the Modal timeout
(`modal_lib/utils.py:24`) is a workaround, not a fix.

## Files

- `benchmarking/modal-results/triton-bench/own/b200/tp8/` — fast-pod run5,
  promoted to canonical position (local + Modal volume).
- `benchmarking/modal-results/triton-bench/own/b200-run{1,2,4}/tp8/` —
  HGX runs, retained for cross-class comparison.
- `benchmarking/modal-results/triton-bench/own/b200-run{3,3b}/tp8/` —
  timeout artefacts, kept until cleanup.
- `benchmarking/modal-results/triton-bench/own/b200/tp{2,4}/` — fast-pod
  reference results from earlier work.

## Fresh full-provider reruns (2026-07-26)

The old B200 data were replaced with fresh runs containing all three baselines, FMMS, and the P2P no-overlap ablation.
There are five completed runs at TP1, TP2, and TP4, and four at TP8.
The fifth TP8 submission stopped with a Modal `RemoteError` before producing a CSV.

The apparent TP2-to-TP4 regression was caused by mixing Modal B200 host classes.
The fresh TP2 set contained one fast-host run, while all five TP4 runs and the original four completed TP8 runs landed in the slow cluster.
Because the plot takes the minimum independently at each TP size, it selected the fast TP2 result and compared it with slow-host TP4 and TP8 results.

Median FMMS latency across runs at batch sizes 1, 64, and 256 was:

| TP | B=1 | B=64 | B=256 |
|---:|---:|---:|---:|
| 1 | 0.333 ms | 0.349 ms | 0.754 ms |
| 2 | 0.200 ms | 0.216 ms | 0.361 ms |
| 4 | 0.169 ms | 0.200 ms | 0.208 ms |
| 8 | 0.159 ms | 0.177 ms | 0.174 ms |

These mixed-class results must not be used for tensor-parallel scaling claims.

### Controlled launcher diagnostic

Two TP4 and two TP8 runs were launched on the current image with torchrun but without NUMA binding.
Both TP4 runs and one TP8 run landed in the slow cluster.
Disabling binding did not restore performance on those hosts and was generally slower than the bound runs, which rules out NUMA binding as the cause.

The other TP8 run landed on a host exposing `mlx5_bond_0` and reproduced the old fast results:

| B | Method | Old fast minimum | Diagnostic fast |
|---:|---|---:|---:|
| 16 | FMMS | 0.087 ms | 0.087 ms |
| 16 | Compiled | 0.242 ms | 0.242 ms |
| 16 | FI1 | 0.215 ms | 0.219 ms |
| 16 | FI2 | 0.164 ms | 0.170 ms |
| 64 | FMMS | 0.096 ms | 0.095 ms |
| 64 | Compiled | 0.313 ms | 0.314 ms |
| 64 | FI1 | 0.258 ms | 0.260 ms |
| 64 | FI2 | 0.204 ms | 0.207 ms |

This establishes that the current image, kernels, torchrun launcher, and timing code can reproduce the previous performance.
The remaining problem is obtaining matched fast-host runs at every TP size.
The reliable signal in the diagnostic TP8 pair was the presence of `mlx5_bond_0` on the fast host; the slow host exposed ten individual NICs and no bond.
