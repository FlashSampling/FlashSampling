# Triton TMA (Tensor Memory Access) pitfalls

TMA uses `tl.make_tensor_descriptor`, `desc.load()`, and `desc.store()` for hardware-accelerated memory access on Hopper and Blackwell GPUs.
This document records five implementation constraints.

## 1. Innermost dimension must be aligned to 16 bytes

TMA descriptors require the innermost stride-one dimension to be a multiple of 16 bytes.
For bfloat16, that means a multiple of eight elements.
Nonaligned dimensions can cause silent data corruption.

```
K=304 (304 % 8 == 0) → PASS
K=300 (300 % 8 == 4) → FAIL, max_err=92.0
N=200 (200 % 8 == 0) → PASS
N=33  (33 % 8 == 1)  → FAIL, max_err=34.75
```

The current `matmul()` wrapper in `src/fused_mm_sampling/tl_matmul.py` rejects nonaligned K or N dimensions with `ValueError`.
If a caller needs arbitrary shapes, pad before calling the wrapper and slice the result afterward.
Do not refer to the removed `_tma_pad()` helper.

## 2. `tl.dot(a, b.T)` does NOT work with TMA-loaded blocks

`.T` only swaps the logical view without rearranging shared memory layout. Tensor core MMA instructions depend on physical (row-major) layout, so the dot product produces wrong results. You must pre-transpose the matrix in the wrapper to make it physically contiguous in the layout the kernel expects.

## 3. Triton enforces `strides[-1] == 1`

You cannot describe a transpose through TMA strides.
Triton's `semantic.py` checks that the final stride is one and raises `CompilationError` otherwise.
Pretranspose the matrix and make it contiguous in the required layout.

## 4. TMA store descriptors with degenerate dims silently no-op on Blackwell

On B200 (sm_100, Triton 3.6), a `tl.make_tensor_descriptor(...).store(...)` with a 3D `block_shape=[1, 1, BLOCK_SIZE_H]` (two singleton dims) over a 3D output buffer silently drops the store on most iterations of a persistent loop. Symptoms:

- Only the first ~2 of N persistent-loop iterations actually write to GMEM.
- Remaining slots stay uninitialized.
- The `bf16` and `int64` variants of the same descriptor pattern are both affected.
- The same descriptor works correctly on Hopper (sm_90) and on RTX 3090 (sm_86, no warp specialization).
- Disabling `WARP_SPECIALIZE` does NOT fix it.
- Forcing the development-machine autotune config (`BLOCK_SIZE_D=32, num_warps=4, num_stages=2, maxnreg=255`) does not fix it either.
  Every config in the search space is affected.

The Triton tutorial `09-persistent-matmul.py` only ever uses 2D TMA store descriptors with no singleton dims (e.g. `block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]`). Deviating to 3D-with-singleton-dims is the trigger.

Drop TMA store descriptors for tiny scattered output writes because TMA gives no bandwidth benefit at this size.
Replace `desc.store(...)` with `tl.store(ptr + offset, val, mask=...)` and keep TMA for the matmul load descriptors.
See `findings/tma-store-blackwell-singleton-dims.md` and `fused_mm_sample_triton_kernel` in `src/fused_mm_sampling/core.py` for the canonical fix.

## 5. Allocator setup belongs in the wrapper

`fused_mm_sample_triton()` calls `set_torch_allocator_for_tma_descriptors_cached()` before launching TMA kernels.
Clients using the sampler or wrapper APIs do not need to call it directly.
Keep explicit calls only in raw TMA launch paths that bypass the wrapper.
