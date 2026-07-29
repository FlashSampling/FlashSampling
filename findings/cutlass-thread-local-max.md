# CUTLASS thread-local max-with-index

## Result

Gate 1b of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
All 9,728 comparisons matched an independent CPU reference exactly.
The expected and actual FP32 bit patterns and integer indices agreed in every row.

The primitive is implemented in `src/fused_mm_sampling/csrc/cutlass_thread_local_max.cu`.
Run it with `make modal-cutlass-thread-local-max`.
The runner writes `summary.json`, `cases.csv`, and the complete `log.txt` under `benchmarking/modal-results/cutlass-thread-local-max/`.

## Method

Each architecture runs a separately compiled CUDA binary in the pinned Gate 0 image.
One block represents the 128 CUTLASS consumer threads observed in Gate 1a.
Each CUDA thread independently reduces a 16-value FP32 fragment and returns a value-index pair.

The comparison chooses the larger value, then the lower global index when values tie.
The test matrix contains a unique maximum in every one of the 16 fragment slots, an all-negative case, a tie whose lower index appears in the earlier slot, and a tie whose lower index appears in the later slot.
Every case is reduced in ascending and descending fragment visitation order.
This catches an order-dependent tie implementation instead of validating only one convenient traversal.

The complete matrix is:

- 2 architectures.
- 19 input cases.
- 2 visitation orders.
- 128 consumer threads.

This produces 9,728 exact comparisons.

## Human verification

Start with `benchmarking/modal-results/cutlass-thread-local-max/VERIFY.md`.
Then inspect `case-summary.csv`, which has one row per architecture, case, and visitation order instead of one row per CUDA thread.

The expected outcome is:

- Both `sm90` and `sm100` are present.
- There are 19 cases and two visitation orders per architecture, for 76 summary rows.
- Every summary row covers 128 consumer threads.
- Every `mismatch_count` is zero.
- Expected and actual FP32 bit patterns and integer indices match.

The actual outcome is 76 complete summary rows, 128 threads in every row, and zero mismatches.
`summary.json` records 9,728 expected comparisons, 9,728 actual comparisons, and zero failures.
The full thread-level evidence remains in `cases.csv`.

The quickest shell checks are:

```bash
column -s, -t benchmarking/modal-results/cutlass-thread-local-max/case-summary.csv | less -S
rg -n '"expected_count"|"actual_count"|"failure_count"' benchmarking/modal-results/cutlass-thread-local-max/summary.json
rg -ni 'error|exception|skipped|nan|fallback' benchmarking/modal-results/cutlass-thread-local-max/log.txt
```

The first command should show every `thread_count` as 128, every `mismatch_count` as zero, and matching expected/actual columns.
The second should show 9,728, 9,728, and zero.
The third should print nothing.

## Constraint review

The restriction against warp shuffles and shared memory is useful at this gate.
It isolates the value-index comparison and serial fragment reduction from the architecture-specific communication added in Gates 1c and 1d.
Relaxing it would make a failure ambiguous between comparison, shuffle, and synchronization logic.

The complete-fragment restriction is also intentional, but only temporary.
Boundary predication changes which values and indices participate in the reduction and is tested separately in Gate 1f.
Combining predication with the first comparison primitive would make it harder to identify whether a wrong winner came from masking or comparison.
No dependency constraint was required for this gate beyond the reproducible Gate 0 toolchain, and the plan permits upgrading that toolchain if a later CUTLASS or CUDA facility materially simplifies the implementation.

## Failure signatures

The runner fails if either architecture is absent, a case or visitation order is missing, any consumer thread is missing, the row count differs from the expected Cartesian product, an FP32 bit pattern differs, or an index differs.
CUDA allocation, copy, launch, and synchronization errors terminate the architecture binary.

## Limitations

This gate proves only the deterministic thread-local comparison and reduction primitive.
It deliberately does not use warp shuffles, shared memory, partial fragments, boundary predication, or the CUTLASS EVT callback.
Gate 1c must establish cross-lane communication independently.
