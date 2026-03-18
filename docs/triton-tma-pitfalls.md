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
