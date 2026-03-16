# Proton scopes don't work inside persistent Triton kernels

## Problem

After converting the FMMS Triton kernel to a persistent kernel (`tl.range` with `flatten=True`), the existing Proton intra-kernel scopes (`pl.enter_scope`/`pl.exit_scope`) stopped working. Two issues:

1. **Duplicate scope names.** The compiler's software pipelining peels the first loop iteration into a prologue, duplicating the `pl.enter_scope("matmul-tile")` call. Proton's trace backend rejects duplicate start records for the same scope name.

2. **Scope placement after compiler rewrites.** Even if names are fixed, the compiler reorders operations across the prologue and loop body. The `end "matmul-tile"` ends up inside a conditional (`scf.if` checking the last inner-loop iteration), while the corresponding `start` is in a different conditional. The scopes no longer bracket the intended operations.

The Triton docs explicitly warn about this: "Triton's higher-level IR undergoes aggressive compiler rewrites (loop pipelining, instruction re-ordering, IR duplication, etc.). These transformations can invalidate naive instrumentation."

## TTGIR override approach (attempted, failed)

Triton supports a 3-step TTGIR override workflow:
1. Dump TTGIR with `TRITON_KERNEL_DUMP=1`
2. Modify the TTGIR (fix scope names, remove unpaired scopes)
3. Re-run with `TRITON_KERNEL_OVERRIDE=1`

A `deduplicate_proton_scopes.py` script was written to rename duplicate scopes (`"matmul-tile"` -> `"matmul-tile:0"`, `"matmul-tile:1"`) and remove unpaired prologue starts.

This partially worked for bsz=1 (single hidden state), but **failed for bsz>=2** because:

- `proton.start(backend="instrumentation")` globally modifies Triton's compilation pipeline, even without `hook="triton"` or `pl.enable_semantic("triton")`.
- The `_local_reduce` function (stage 2 of FMMS) uses `@torch.compile(fullgraph=True)`, which generates an inductor Triton kernel.
- This inductor kernel breaks when compiled through the Proton-modified pipeline: `AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'`.
- There is no way to selectively enable Proton instrumentation for only our kernel while leaving inductor kernels unaffected.

Isolating Triton and inductor caches between steps did not help because the corruption happens at compile time within the same process, not from cached artifacts.

## What works

- **Proton `pcsampling` mode** (PC sampling) still works with the persistent kernel. It doesn't use intra-kernel scopes, just samples instruction counters.
- **Proton `trace` mode without scopes** works but gives no per-phase breakdown (matmul vs sampling).
- **Gluon DSL kernels** can use Proton scopes inside loops because the Gluon IR doesn't undergo the same aggressive rewrites.

## Conclusion

Proton intra-kernel scopes are incompatible with persistent Triton kernels. The scopes were removed from the kernel. For matmul vs sampling breakdown, NCU profiling (which uses NVTX ranges at the host level) remains the primary tool.
