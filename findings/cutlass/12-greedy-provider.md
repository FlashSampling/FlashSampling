# CUTLASS greedy TP1 provider

## Result

Gate 2a of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
All 52 production-path cases returned the expected greedy token indices.

The provider is exposed as `fused-cutlass-greedy` through `get_sampler()`.
Run the gate with `make modal-cutlass GATE=greedy-provider`.
The human-verifiable packet is written to `benchmarking/modal-results/cutlass/09-greedy-provider/`.

## Method

The PyTorch extension includes the exact Gate 1h CUTLASS EVT candidate implementation and launches a GPU Stage 2 reduction with the same packed FP32-value/i32-index comparator.
The wrapper accepts contiguous BF16 weights in `[V, D]` layout and hidden states in `[H, D]` layout, returns int64 indices in `[H, 1]` layout, and rejects TP greater than one or `num_samples` other than one.
It JIT compiles only for SM90 and SM100 against the pinned CUTLASS source tree.

The correctness matrix covers V={100,127,128,129,255,256,257}, a deterministic cross-tile tie, and the two primary model shapes `(V=151,936, D=4,096)` and `(V=128,256, D=8,192)`.
Both model shapes run H={1,2,4,8,16,32,64,128,256} on both architectures.
H values below four are internally padded to four columns because the current FP32 TMA epilogue requires four-element alignment.
Stage 2 returns only the original H outputs.

The existing `test_greedy_sampling` pytest is parameterized over the Triton and CUTLASS greedy providers.
The CUTLASS parameter runs V={100,127,128,129,200,255,256,257,512} at H={1,2}, producing 18 passing cases on each architecture.
Gate 2a saves the independent pytest outputs as `pytest-sm90.txt` and `pytest-sm100.txt` and cannot pass without them.
These provider-level tests use the standard FMMS Modal image with the pinned CUTLASS source tree layered onto it.
The standalone CUTLASS toolchain image remains limited to the low-level Gates 0 and 1.

## Constraint review

Keeping the provider BF16, TP1, greedy, and SM90/SM100-only remains useful for Gate 2a.
These constraints isolate the production binding and dynamic-shape integration before RNG and tensor parallelism add separate correctness domains.

Reusing the Gate 1h source by inclusion remains useful because it prevents the candidate representation, accumulator ownership formulas, and deterministic comparator from drifting.
The production extension uses PyTorch's current CUDA stream for the CUTLASS GEMM and Stage 2 launch.

The diagnostic FP32 D output inherited from Gate 1h is no longer a useful production constraint.
It allocates and writes a padded `[V, H]` tensor even though the provider returns only indices.
Gate 2b must remove that store while preserving the approved candidate and Stage 2 path before making a performance feasibility decision.
Performance measurements from the current diagnostic-store path must not be used to approve the kernel.

## Failure signatures

The gate fails if:

- Either H100 or B200 is absent.
- The provider falls back to another implementation or cannot build through `get_sampler()`.
- Any boundary, tie, primary model shape, or H value is absent.
- Any returned index differs from the deterministic expected result.
- A cross-tile tie does not select the lowest global vocabulary index.
- The wrapper accepts TP greater than one or an unsupported architecture.
- The CUTLASS extension, GEMM, or Stage 2 launch fails.

## Limitations and next gate

The test inputs construct exact, well-separated BF16 maxima and deterministic ties.
This gate does not establish general GEMM numerical tolerances, performance, RNG correctness, tensor parallelism, or top-k behavior.
The extension currently requires a local CUTLASS 4.6.1 source tree at the
pinned Gate 0 commit, and first use JIT compiles the architecture-specific
module.

Gate 2b should first replace the diagnostic D store with a no-output epilogue that retains the Gate 1h auxiliary candidates.
It should then compare the exact corrected provider against plain CUTLASS GEMM, CUTLASS GEMM plus argmax, Triton FMMS, and cuBLAS plus argmax over the full H sweep on both architectures.
