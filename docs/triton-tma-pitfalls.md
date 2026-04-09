# Triton TMA (Tensor Memory Access) pitfalls

TMA uses `tl.make_tensor_descriptor` / `desc.load()` / `desc.store()` for hardware-accelerated memory access on H100. Three hard-won lessons:

## 1. Innermost dimension must be aligned to 16 bytes

TMA descriptors require the **innermost (stride-1) dimension** to be a multiple of 16 bytes. For bfloat16 (2 bytes/element), that means **multiples of 8 elements**. Non-aligned dimensions cause **silent data corruption** — no error, just wrong results.

```
K=304 (304 % 8 == 0) → PASS
K=300 (300 % 8 == 4) → FAIL, max_err=92.0
N=200 (200 % 8 == 0) → PASS
N=33  (33 % 8 == 1)  → FAIL, max_err=34.75
```

**Fix:** Pad tensors in the Python wrapper before passing to the kernel. Zero-padding doesn't affect matmul results. See `_tma_pad()` in `tl_matmul.py`. After the kernel, slice output back to the original dimensions.

## 2. `tl.dot(a, b.T)` does NOT work with TMA-loaded blocks

`.T` only swaps the logical view without rearranging shared memory layout. Tensor core MMA instructions depend on physical (row-major) layout, so the dot product produces wrong results. You must pre-transpose the matrix in the wrapper to make it physically contiguous in the layout the kernel expects.

## 3. Triton enforces `strides[-1] == 1`

You cannot describe a transpose via TMA strides — Triton's `semantic.py` checks that the last stride is 1 and raises `CompilationError` otherwise. The only option is to pre-transpose and make the matrix contiguous in the desired layout.

## 4. TMA store descriptors with degenerate dims silently no-op on Blackwell

On B200 (sm_100, Triton 3.6), a `tl.make_tensor_descriptor(...).store(...)` with a 3D `block_shape=[1, 1, BLOCK_SIZE_H]` (two singleton dims) over a 3D output buffer silently drops the store on most iterations of a persistent loop. Symptoms:

- Only the first ~2 of N persistent-loop iterations actually write to GMEM.
- Remaining slots stay uninitialized.
- The `bf16` and `int64` variants of the same descriptor pattern are both affected.
- The same descriptor works correctly on Hopper (sm_90) and on RTX 3090 (sm_86, no warp specialization).
- Disabling `WARP_SPECIALIZE` does NOT fix it.
- Forcing the dev-machine autotune config (`BLOCK_SIZE_D=32, num_warps=4, num_stages=2, maxnreg=255`) does NOT fix it either — every config in the search space is affected.

The Triton tutorial `09-persistent-matmul.py` only ever uses 2D TMA store descriptors with no singleton dims (e.g. `block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]`). Deviating to 3D-with-singleton-dims is the trigger.

**Fix:** Drop TMA store descriptors entirely for tiny scattered output writes — TMA gives no bandwidth win below ~32 KB per launch anyway. Replace `desc.store(...)` with plain `tl.store(ptr + offset, val, mask=...)`. Keep TMA for the matmul *load* descriptors (where it actually pays off). See `findings/tma-store-blackwell-singleton-dims.md` and the FMMS kernel in `core.py:fused_mm_sample_triton_kernel` for the canonical fix.
