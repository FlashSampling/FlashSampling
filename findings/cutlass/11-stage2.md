# CUTLASS GPU Stage 2

## Result

Gate 1h of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
All 2,176 intermediate per-tile candidates and 944 final outputs matched exactly across both architectures.
Compute Sanitizer memcheck reported zero errors, and racecheck reported zero hazards, errors, and warnings on both architectures.

The implementation reuses `src/fused_mm_sampling/csrc/cutlass/evt_candidates.cu` with `FMMS_GATE_STAGE2` enabled.
Run it with `make modal-cutlass GATE=stage2`.
The human-verifiable packet is written to `benchmarking/modal-results/cutlass/08-stage2/`.

## Method

The Gate 1g CUTLASS GEMM and EVT path remains unchanged.
The Gate 1h build launches a separate CUDA Stage 2 kernel after the EVT writes one packed FP32-value/i32-index candidate for each M tile and N column.
One Stage 2 thread owns one N column and merges its M-tile candidates with the same deterministic comparator used by the EVT.

The packet retains every intermediate candidate and every final output.
Both are compared exactly against a host reference constructed from the real GEMM inputs.
The final matrix covers global winners in the first, middle, and last M tiles, complete and partial M/N shapes, all-negative values, within-tile ties, and equal global maxima in different tiles.
Cross-tile ties select the lowest global vocabulary index.

## Constraint review

Keeping Stage 2 as a separate kernel remains useful.
It isolates the global merge from the warp-specialized CUTLASS epilogue, matches the existing two-stage FMMS structure, and avoids adding a new cross-CTA synchronization mechanism before performance data justifies it.

Compiling Gate 1h from the Gate 1g source with a feature definition also remains useful.
It prevents the deterministic kernel path from drifting between gates while preserving the original Gate 1g executable and evidence format.

The Stage 2 kernel intentionally uses one thread per output column.
Gate 1h validates deterministic correctness, not the eventual launch geometry or performance.
Gate 2 may change the Stage 2 implementation only if it preserves this gate's exact candidate representation and comparator behavior.

## Failure signatures

The gate fails if:

- Either architecture or any declared test family is absent.
- Any intermediate candidate value bit pattern or global index differs.
- Any final value bit pattern or global index differs.
- First-, middle-, or last-tile winner coverage is absent.
- A cross-tile tie does not choose the lowest global index.
- Any candidate or final coordinate is missing.
- The CUDA launch, synchronization, allocation, or copy fails.
- Memcheck reports an error.
- Racecheck reports a hazard, error, or warning.

## Limitations and next gate

The deterministic inputs intentionally make the GEMM reference exactly reproducible.
This gate does not measure numerical error for general dense BF16 inputs, performance, sampling, tensor parallelism, or top-k.
The SM90 build still reports that WGMMA instructions may serialize across a function-call boundary.

Gate 2a should wrap this deterministic kernel in the production sampler interface as a BF16, TP1, greedy provider for H100 and B200.
Gate 2b must then measure the exact correctness-approved path and resolve or quantify the SM90 serialization warning before the project adds RNG or tensor parallelism.
