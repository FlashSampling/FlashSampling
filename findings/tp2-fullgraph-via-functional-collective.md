# TP2 fullgraph compile via functional all_gather_tensor

The TP>1 path used to fall back to `torch.compile(...)` (no fullgraph), branching
inside `sample_compiled` / `greedy_sample_compiled` based on `tp.size > 1`.
The reason: `_allgather_logits` was decorated with `@torch.compiler.disable`
because dynamo could not trace through `dist.all_gather` (or
`dist.all_gather_into_tensor`) cleanly, so the surrounding compiled function
had to graph-break at every collective call.

This finding documents the path to a single, fullgraph-compiled path for both
TP1 and TP>1, and the measured impact on B200 TP2.

## What changed

1. `_allgather_logits` was rewritten on top of
   `torch.distributed._functional_collectives.all_gather_tensor`. This is the
   compile-friendly collective primitive: it returns an `AsyncCollectiveTensor`
   that integrates with the FX graph, so dynamo can trace through it without a
   graph break. `gather_dim=1` is supported (the docstring's "currently only
   supports gather_dim = 0" note is stale — the implementation does the
   chunk+cat after a dim-0 gather via `_maybe_view_chunk_cat`).

2. With the collective traceable, `@torch.compiler.disable` on
   `_allgather_logits` was no longer needed.

3. `sample_compiled` and `greedy_sample_compiled` were collapsed to a single
   fullgraph-compiled path (no more `_with_breaks` variant, no more
   `if tp.size > 1` dispatch).

4. `torch.manual_seed(seed)` inside `sample()` blocks fullgraph because dynamo
   intentionally marks `torch.random.manual_seed` as skipped. Pulled the seed
   handling into a thin `sample_compiled` wrapper that seeds eagerly, then
   calls the compiled inner without a seed; the dead `if seed is not None`
   branch in `sample()` is folded at trace time.

## What was attempted along the way (and rejected)

- **`dist.all_gather_into_tensor` with `@torch.compiler.disable` removed**:
  hangs the TP2 distributed pytest at the first compiled provider. dynamo
  attempts to trace through it, hits the torch functional-collective lowering
  (`output_tensor.copy_(all_gather_tensor(...))`), and crashes on a fake-tensor
  shape mismatch where the rank-dim and H-dim get confused
  (`expand: attempting to expand a dimension of length 2 -> 1`). With the
  decorator restored, this works but defeats the goal of fullgraph.

- **Compiling `sample` directly with `fullgraph=True` while keeping the
  in-body `torch.manual_seed`**: dynamo errors with
  `Attempted to call function marked as skipped: manual_seed`.

## What is NOT affected

The FMMS Triton kernel does not call `_allgather_logits` (it uses
`kraken_post_kernel_reduce` over symmetric memory), so the
`fused-triton`/`fused-triton-greedy`/`fused-triton-ret-logits` providers are
unchanged. Same for `helion`, `fused-cuda`, `fused-topk`, `jl-compiled`,
`sequential-compiled`. The change only touches the baseline providers that
gather logits across ranks.

## Measured impact (B200 TP2, "small" config, BENCH_FN=own)

3 runs on `main` and 3 on the branch (`-main-runX` / `-fg-runX`), each with 0
NUMA-binding warnings in the log. All other Modal noise was uncontrolled.

### `Multinomial Sampling (Compiled)` — the column that changes

This is `naive-compiled` (= `sample_compiled` with TP>1). Latencies in ms:

| H | main mean ± std | fg mean ± std | delta |
|---|---|---|---|
| 1 | 0.394 ± 0.137 | 0.284 ± 0.077 | **-28%** |
| 8 | 0.400 ± 0.127 | 0.286 ± 0.042 | **-28%** |
| 32 | 0.418 ± 0.089 | 0.340 ± 0.002 | **-19%** |
| 64 | 0.468 ± 0.059 | 0.414 ± 0.000 | **-12%** |
| 128 | 0.564 ± 0.001 | 0.593 ± 0.001 | **+5%** |
| 256 | 0.916 ± 0.003 | 0.976 ± 0.003 | **+6%** |

Two effects:

- **Big win at low H**: dropping the per-call graph-break overhead matters
  most when the kernel itself is small. -28% at H=1.
- **Small loss at high H**: a real ~5-6% regression at H=128/256. Std on
  both sides is ~0.001-0.003 ms, so the gap is many sigmas above zero, not
  noise. Likely cause: inductor makes a different fusion/scheduling choice
  on the larger fullgraph at those shapes that happens to be worse than
  what `_with_breaks` was producing. Worth investigating if the operating
  point of interest is H >= 128. Did not investigate yet.

### `flashinfer:*` — also affected (both paths call `_allgather_logits`)

Both flashinfer providers are 2-12% faster on the branch across H. At
H=128/256 the std is tiny so the gap is highly significant; at low H the
noise is bigger but the direction is consistent.

### `FMMS (Triton)` and `Multinomial Sampling (Eager)` — should not change

- FMMS: -3% to -8% mean delta, std ~25% of mean (large run-to-run variance) —
  not statistically distinguishable from zero. Confirms FMMS is unaffected
  by the branch.
- Eager: -1% to -3%, with very tiny std on both sides. Significant by
  t-ratio but real-world negligible. Probably small environmental drift
  between the run cohorts; the eager `sample` doesn't go through compile so
  there is no code-level reason it should move.

### Variance pattern

`main`'s `Multinomial Sampling (Compiled)` has std ~30% of mean at low H but
~0% at high H. The branch's std is uniformly small. So `_with_breaks` has some
warmup/recompile-per-shape behavior that makes early-batch latencies vary a
lot run-to-run, while fullgraph specializes once and is consistent. That
alone is a reason to prefer fullgraph at low H even ignoring the mean.

## Operational note

The Modal pytest-distributed worker doesn't print Triton autotune output
unless `TRITON_PRINT_AUTOTUNING=1` is set. Without it, a cold-cache autotune
looks identical to a hang in the run log. `set_volume_caches()` now exports
the env var by default so every modal entrypoint has visibility.
