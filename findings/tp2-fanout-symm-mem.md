# TP2 Fan-Out Symm-Mem Writes (B200, 2026-05-03)

## Motivation

After `findings/tp2-collective-overhead.md` introduced symmetric memory on the
post-kernel path, the call still had two host-side stages on the critical path:

1. The Triton kernel finishes and writes its per-tile winners to *its own* slot
   in symmetric memory.
2. `symm_mem_hdl.barrier()` ensures all ranks' writes are visible.
3. Each rank reads every peer's slot from symmetric memory, runs `_local_reduce`
   per source rank, then `_stack_and_select_winner`.

Step 3 cannot start until step 1 finishes on every rank, because the per-tile
buffers are still being written. So the post-kernel work is fully sequential
with the kernel.

If the kernel itself wrote each per-tile winner directly into *every* rank's
symmetric-memory buffer (a "fan-out" store) as it produced it, the inter-GPU
traffic would be issued throughout the matmul / sample loop instead of after
it. After the post-kernel barrier, every rank already has every other rank's
data locally and only needs to reduce.

## Implementation

`src/fused_mm_sampling/tensor_parallel_reduce.py`:

- `allocate_symm_mem_outputs` now allocates a `[world_size, num_samples,
  max_grid_size_v, H]` buffer per rank (extra leading source-rank dim) instead
  of `[num_samples, max_grid_size_v, H]`.
- `kraken_post_kernel_reduce_fanout` is the new post-kernel function; it does
  `symm_mem_hdl.barrier()`, then reads only this rank's local buffer (which by
  then contains every source rank's per-tile data thanks to fan-out writes
  during the kernel) and runs the local reduce.

`src/fused_mm_sampling/core.py`:

- The kernel takes a new `symm_mem_buffer_ptrs` argument (each peer's base
  address, exposed via `symm_mem_hdl.buffer_ptrs_dev`).
- For every per-tile winner produced, the kernel iterates
  `tl.static_range(0, tp_world_size)` and `tl.store`s through each peer's
  base pointer at this rank's source slot. Both `gumbel_max` (bf16) and
  `gumbel_max_idx` (int64) are written.
- `gumbel_max_idx_global` adds `tp_rank * vocab_size`, so indices are already
  in the global vocab space and the post-kernel reduce no longer needs a
  `vocab_start_index` offset.

The matmul-load TMA descriptors are unchanged. Only the small per-tile output
stores are touched.

## Benchmark results

5 main runs vs 5 fanout runs on b200 TP2, `bench_fn=own`, 100 bench iterations
each. One fanout run hit a slow Modal node (see "Modal node-class noise"
below) and was excluded; the table is mean over 5 main / 4 fanout. Within-group
std on FMMS is ~0.001 ms.

### FMMS (Triton) latency, ms

| H | small main | small fan | small Δ | large main | large fan | large Δ |
|---|---|---|---|---|---|---|
| 1 | 0.157 | 0.139 | **−11.4%** | 0.217 | 0.199 | **−8.2%** |
| 2 | 0.160 | 0.144 | −10.1% | 0.219 | 0.204 | −7.1% |
| 4 | 0.161 | 0.145 | −10.0% | 0.219 | 0.204 | −6.8% |
| 8 | 0.161 | 0.145 | −10.0% | 0.221 | 0.204 | −7.8% |
| 16 | 0.161 | 0.145 | −9.9% | 0.221 | 0.203 | −8.2% |
| 32 | 0.165 | 0.148 | −10.1% | 0.224 | 0.207 | −7.6% |
| 64 | 0.173 | 0.155 | −10.5% | 0.230 | 0.214 | −7.0% |
| 128 | 0.196 | 0.180 | −8.1% | 0.261 | 0.244 | −6.6% |
| 256 | 0.299 | 0.281 | −6.0% | 0.421 | 0.403 | −4.2% |

`small` = V=151,936, d=4,096; `large` = V=128,256, d=8,192.

The other providers (multinomial-eager, multinomial-compiled, flashinfer)
show ±1% across all H — they don't touch the fan-out path, which confirms
the noise floor and that the FMMS deltas are the kernel change, not run
variance.

### Pattern

The relative gain is largest at low H (8-11% at H≤32 small) and shrinks at
H=256 (4-6%). Consistent with the design: at low H the post-kernel collective
was a larger fraction of call time, and the fan-out hides it behind kernel
work; at H=256 the kernel is large enough that there is little remaining
post-kernel latency to hide.

## Modal node-class noise

Initial single-run comparisons gave wildly different speedup numbers (from
−7% to −66%), all dominated by which Modal SKU the run landed on. Investigating
metadata for 11 runs (5 + 3 + 3) revealed two distinct b200 machine classes:

| Class | NIC count | First NIC name | CPU affinity | FMMS H=1 small (ms) |
|---|---|---|---|---|
| Fast 2-GPU pod | 1-2 | `mlx5_10` or `mlx5_bond_0` | 64-128 cores | 0.137-0.140 |
| Slow HGX 8-GPU shared | 8-10 | `mlx5_0` or `mlx5_4` | 4-96 cores | 0.173-0.214 |

The slow HGX class is ~25-50% slower at low H but **identical at H=256** (both
~0.281 ms). That points to a host-side dispatch overhead (extra PCIe hops,
multi-tenant CPU contention) rather than a fabric issue: GPU↔GPU is `NV18`
(NVLink) on every SKU, and our TP traffic only uses NVLink anyway.

### `cpu=N` cannot filter HGX

We tried `cpu=32` and `cpu=64` on the Modal `@app.function` decorator. Both
were honored, but neither blocked the slow class:

- `cpu=32`: HGX still landed (44-core slot satisfied the request), FMMS ran
  at 0.198 ms.
- `cpu=64`: HGX still landed (Modal allocated 96 cores on the HGX host),
  FMMS ran at 0.173 ms.

So the slow class is *not* a CPU-starvation issue; HGX hosts have plenty of
cores to hand out. The slowdown is from the underlying box (PCIe topology,
cross-tenant noise) and `cpu=` cannot exclude it.

### Reliable filter

The reliable fingerprint is the NIC layout in metadata:

- Fast pod: `nic0 ∈ {mlx5_10, mlx5_11, mlx5_bond_0}`, n_nics ≤ 3.
- Slow HGX: `nic0` is single-digit `mlx5_0..mlx5_9`, n_nics ≥ 8.

Filtering at analysis time (drop runs with n_nics > 3 or low-numbered nic0)
gives a clean comparison without inflating Modal queue time. The `cpu=`
constraint added to `modal_lib/utils.py` during this investigation was
reverted because it provided no real filtering.

## Files / branch

- Implementation lives on the `fanout-symm-mem` branch (commit `927601a`,
  squashed into `tensor_parallel_reduce.py` + `core.py`).
- The pre-fanout reduce (single-slot writes + post-kernel remote reads) is
  available in git history before the fanout commit if A/B testing is needed.
- Benchmark CSVs are in
  `benchmarking/modal-results/triton-bench/own/b200-{main,gpu-fanout}-runX/tp2/`.
