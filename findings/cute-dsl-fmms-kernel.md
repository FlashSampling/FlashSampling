# Plan: CuTe DSL FMMS kernel

## Context

The existing Triton FMMS kernel works well but the user wants to target
Hopper (SM90) and Blackwell (SM100+). CuTe DSL (`nvidia-cutlass-dsl` v4.3.5,
already installed) compiles Python kernels via MLIR to PTX/CUBIN — no C++
compilation needed, architecture-portable, and provides an upgrade path to
TMA/warpgroup-MMA for future optimization.

## Kernel design

### Pre-generated Gumbel noise

CuTe DSL has **no native RNG API**. Gumbel noise is pre-generated in Python
and passed as a tensor:

```python
u = torch.rand(V, H * num_samples, device=device)
gumbel = -torch.log(-torch.log(u.clamp(min=1e-10)))
```

Memory cost: V x H x S x 4B. For decode (V=128K, H=1, S=1): 512KB.
For correctness tests (V=256, H=2, S=10K): 20MB. Both fine.

### Work decomposition

Same as CUDA C++ / Triton kernel:

- Thread block: 256 threads (8 warps), TILE_V=128 vocab rows
- Each warp: ROWS_PER_WARP=16 rows
- 32 lanes cooperate on D-dimension dot product (coalesced scalar loads)
- Swizzle2d for L2 cache reuse (GROUP_SIZE=4)
- Per-sample loop: add noise, warp argmax, block argmax via smem

### CuTe DSL primitives

| Operation | CuTe DSL API |
|-----------|-------------|
| Thread/block IDs | `cute.arch.thread_idx()`, `block_idx()`, `lane_idx()`, `warp_idx()` |
| D-reduction | `cute.arch.warp_reduction_sum(val)` (butterfly shuffle) |
| Warp argmax | Custom loop with `cute.arch.shuffle_sync_bfly(val, offset)` |
| Block reduction | `cute.arch.alloc_smem(Float32, N_WARPS)` + `cute.arch.sync_threads()` |
| Max comparison | `cute.arch.fmax(a, b)` |
| Tensor access | `cute.runtime.from_dlpack(torch_tensor)` |
| Compilation | `cute.compile(jit_fn, *args)` → cached compiled function |

### Kernel structure

```python
@cute.kernel
def fmms_stage1_kernel(gW, gH, gNoise, gMaxOut, gMaxOutIdx, ...):
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    bid_x, bid_y, _ = cute.arch.block_idx()

    # Swizzle (same formula as CUDA kernel)
    # pid_v, pid_h = swizzle2d(bid_x, bid_y, ...)

    # D-loop: each lane accumulates partial dot products
    # for d in range(lane, hidden_size, WARP_SIZE): ...
    # Warp sum: acc = cute.arch.warp_reduction_sum(acc)

    # Scale by 1/temperature, add pre-generated noise
    # Warp argmax via shuffle_sync_bfly
    # Block argmax via shared memory
    # Write (max_val, max_idx) to output
```

### Stage 2 (Python-side)

Identical to Triton/CUDA wrapper:
```python
idxs = maxs.max(dim=0).indices
samples = maxs_idx.gather(0, idxs.unsqueeze(0)).squeeze(0)
```

### Risk: CuTe DSL limitations

CuTe DSL is designed for structured GEMM (copy atoms, MMA atoms, tiled
partitions). Using low-level primitives for a scalar GEMV + custom reduction
may hit limitations:

1. **Runtime-bounded loops** — `for d in range(0, hidden_size, 32)` where
   `hidden_size` is runtime. MLIR supports `scf.for` so likely OK.
2. **Scalar global loads** — Pointer arithmetic for individual bf16 loads.
3. **bf16 to f32 conversion** — Need explicit cast on load.

Mitigation: Start with a minimal kernel (just dot product + warp sum) to
verify compilation, then build the full kernel incrementally.

## Files to create

1. **`src/fused_mm_sampling/cute_dsl_impl.py`**
   - `fmms_stage1_kernel` — `@cute.kernel` device function
   - `fmms_host_fn` — `@cute.jit` host function (kernel launch)
   - `fused_mm_sample_cute_dsl()` — Python wrapper:
     pre-gen noise, compile, launch, stage 2 reduction
   - Pattern: `cute.compile(fmms_host_fn, *args)` cached globally

## Files to modify

2. **`src/fused_mm_sampling/core.py`** — add `"fused-cute-dsl"` case to
   `get_sampler()` (lazy import from `cute_dsl_impl`).

3. **`tests/test_core.py`** — add `"fused-cute-dsl"` to the parametrized
   `provider` list in `test_sampling_distribution`.

4. **`src/fused_mm_sampling/bench/speed_test.py`** — add `"fused-cute-dsl"`
   to `all_providers` list.

5. **`src/fused_mm_sampling/bench/triton_benchmark.py`** — add
   `"fused-cute-dsl": "FMMS (CuTe DSL)"` to `provider_names` dict and add
   style entry.

## Verification

```bash
# Chi-squared correctness test
.venv/bin/pytest tests/test_core.py::test_sampling_distribution -k "fused-cute-dsl" -v

# Speed comparison
python benchmarking/speed-test.py --name fused-cute-dsl --case large
```
