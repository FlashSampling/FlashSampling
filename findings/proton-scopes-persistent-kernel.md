# Proton intra-kernel profiling for the persistent FMMS kernel

## Problem

DSL-level Proton scopes (`pl.enter_scope`/`pl.exit_scope`) don't work inside persistent Triton kernels.
The compiler's software pipelining peels the first loop iteration into a prologue, duplicating scope entries.
Even if names are deduplicated, the compiler reorders operations across the prologue and loop body, breaking scope placement.

**Confirmed by Triton engineer (Corbin Robeck, 2026-03-17):** DSL-level scopes inside loops are explicitly disabled because the compiler may hoist them out of the loop entirely. This is by design, not a bug.

## Solution: TTGIR-level injection (implemented)

The Triton team recommends injecting `proton.record` statements at the TTGIR level, after compiler rewrites.
Official tutorial and reference scripts:

- Example: https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/intra_kernel/example_override.py
- Injection script: https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/intra_kernel/insert_proton_records

### Workflow

Three Makefile targets (`make proton-profile` runs all three):

1. **Dump TTGIR** (`make proton-dump-ttgir`): Uses `dump_ttgir.sh` to run the kernel and capture TTGIR.
2. **Inject proton.record** (`make proton-inject`): `insert_proton_records.py` pattern-matches the TTGIR and inserts scope pairs.
3. **Run with override** (`make proton-run`): Runs with `TRITON_KERNEL_OVERRIDE=1` so the instrumented TTGIR is used.

### Scopes

The persistent kernel fuses the D-loop and tile loop into a single `scf.for`.
Every HIDDEN_SIZE/BLOCK_SIZE_D iterations (e.g. 128), an `scf.if` epilogue fires with masking, sampling, and store.
We instrument six scopes:

| Scope | What it covers | Fires |
|-------|---------------|-------|
| `kernel` | Full kernel (tt.func → tt.return) | once |
| `setup` | Temperature load, grid dims, TMA descriptors, first tile prefetch | once |
| `mask` | V-masking and temperature scaling | once per tile |
| `tile-mgmt` | Next-tile coordinate computation (swizzle grouping) | once per tile |
| `sample` | Gumbel noise generation and argmax reduce | once per tile |
| `store` | Global index offset, address computation, tt.store | once per tile |

Matmul time is derived as: `kernel - setup - mask - tile-mgmt - sample - store`.

### Key constraints

- **No per-chunk matmul scope possible.** The D-loop is fused into the persistent `scf.for` (not unrolled), so there is only one `tt.dot` instruction in the TTGIR. A scope around it fires 128 times per tile, overflowing the shared buffer (128 slots). Proton's validation also prevents spanning a single scope across multiple loop iterations (start/end must pair within the same iteration).
- **Scope ordering matters.** When multiple insertions target the same line index, a secondary sort key ensures `end` records appear before `start` records, and the outermost `kernel` scope nests correctly around inner scopes.
- **Buffer overflow at high bsz.** Each CTA generates 4 (kernel+setup, once) + 8 (mask+tile-mgmt+sample+store, per tile) events. At bsz=256 with ~58 tiles/CTA, that's ~468 events, exceeding the 256-slot shared buffer. Use `BUFFER_TYPE.GLOBAL` (HBM) for high batch sizes, or accept missing kernel/setup scopes with shared buffer.
- **Warp sampling.** `SAMPLING_STRATEGY.SELECTIVE` with `sampling_options="0"` profiles only warp 0, reducing events per CTA.
- **HBM vs shared buffer.** Comparison at bsz=1 and bsz=128 shows identical ratios (within noise), so the HBM write overhead is negligible with this few events per tile.

### Results (RTX 3090, V=151936, D=4096)

| bsz | matmul% | sample% | BLOCK_SIZE_H |
|-----|---------|---------|--------------|
| 1 | 98.7 | 1.0 | 16 |
| 4 | 98.7 | 1.1 | 16 |
| 16 | 97.2 | 2.4 | 16 |
| 64 | 87.8 | 11.6 | 64 |
| 128 | 79.6 | 19.9 | 64 |
| 256 | 76.5 | 23.0 | 64 |

The sampling fraction grows with batch size because BLOCK_SIZE_H increases (16 → 64), making the Gumbel noise + argmax disproportionately expensive.
At BLOCK_SIZE_H=64, the Philox PRNG operates on 128x64=8192 elements (vs 128x16=2048 at BLOCK_SIZE_H=16).
NCU confirms massive register spilling at bsz=256 (118M local memory spilling requests, 76% of L1TEX accesses are local memory), which is driven by the sampling phase's register pressure.
The tensor-core matmul scales efficiently with wider tiles; the ALU/SFU-bound sampling phase does not.

## Inductor conflict

`proton.start(backend="instrumentation")` is process-global and breaks `torch.compile`-generated inductor kernels (`AttributeError: 'KernelMetadata' object has no attribute 'cluster_dims'`).
The solution is to profile only the FMMS Triton kernel directly (`proton_profile.py`), skipping `_local_reduce` (stage 2).
