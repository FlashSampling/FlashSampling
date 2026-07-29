# CUTLASS warp-local max-with-index

## Result

Gate 1c of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
All 4,832 output-lane comparisons matched an independent CPU reference exactly.
The expected and actual FP32 bit patterns and integer indices agreed in every row.

The shuffle reduction is implemented in `src/fused_mm_sampling/csrc/cutlass/warp_max.cu`.
It reuses the comparison primitive in `src/fused_mm_sampling/csrc/cutlass/max_with_index.cuh`, which is also used by the Gate 1b harness.
Run the gate with `make modal-cutlass GATE=warp-max`.
The generated packet is under `benchmarking/modal-results/cutlass/03-warp-max/` and remains ignored by Git.

## Architecture-specific warp domains

Gate 1a showed that one N column uses different M-lane layouts on the two architectures.
SM90 uses lanes 0, 4, 8, 12, 16, 20, 24, and 28 within each consumer warp.
SM100 uses all 32 lanes.

The Gate 1c primitive uses `__shfl_xor_sync` with architecture-specific masks and strides.
SM90 reduces with mask `0x11111111` and XOR offsets 4, 8, and 16.
SM100 reduces with mask `0xffffffff` and XOR offsets 1, 2, 4, 8, and 16.
Every participating lane receives and validates the final winner.

## Test matrix

Both architectures test all four CUTLASS consumer warps observed in Gate 1a.
Each architecture has a unique-maximum case for every participating input lane, an all-negative case, and two cross-lane tie cases.
The tie cases place the lowest index first and last in lane order so the result cannot depend on shuffle order.

The complete matrix is:

- SM90: 11 cases by 4 warps by 8 output lanes, for 352 comparisons.
- SM100: 35 cases by 4 warps by 32 output lanes, for 4,480 comparisons.
- Total: 4,832 exact comparisons.

Gate 1b was rerun after extracting the shared comparator.
All 9,728 Gate 1b comparisons still passed.

## Human verification

Start with `benchmarking/modal-results/cutlass/03-warp-max/VERIFY.md`.
Then inspect `case-summary.csv`, which has one row per architecture, case, and warp.

The expected outcome is:

- Both `sm90` and `sm100` are present.
- SM90 unique-winner cases cover lanes 0, 4, 8, 12, 16, 20, 24, and 28.
- SM100 unique-winner cases cover every lane from 0 through 31.
- Warps 4 through 7 run every case.
- SM90 compact rows each represent 8 output lanes.
- SM100 compact rows each represent 32 output lanes.
- Every `mismatch_count` is zero and every `pass` is one.
- Expected and actual FP32 bit patterns and integer indices match.

The actual outcome is 184 complete compact rows and zero mismatches.
`summary.json` records 4,832 expected comparisons, 4,832 actual comparisons, and zero failures.
The full output-lane evidence remains in `cases.csv`.

The quickest shell checks are:

```bash
column -s, -t benchmarking/modal-results/cutlass/03-warp-max/case-summary.csv | less -S
rg -n '"expected_count"|"actual_count"|"failure_count"' benchmarking/modal-results/cutlass/03-warp-max/summary.json
rg -ni 'error|exception|skipped|nan|fallback' benchmarking/modal-results/cutlass/03-warp-max/log.txt
```

The first command should show matching expected and actual columns, zero mismatches, and a pass value of one.
The second should show 4,832, 4,832, and zero.
The third should print nothing.

## Constraint review

Using warp shuffles is necessary at this gate because cross-lane communication is the behavior under test.
Supporting both architecture-specific masks is also necessary because Gate 1a empirically showed different participating lanes.

The restriction against shared memory remains useful.
Adding shared memory would combine warp and cross-warp failure domains before the shuffle primitive is established.
Gate 1d introduces shared memory separately.

Testing complete warp-local domains remains appropriate here.
Boundary predication is still isolated in Gate 1f so a masking failure cannot be mistaken for a shuffle failure.
No dependency upgrade would simplify this primitive, because the required CUDA shuffle intrinsics are supported by the pinned Gate 0 toolchain.

## Failure signatures

The runner fails if either architecture, any consumer warp, any required unique-winner lane, a tie case, or the all-negative case is absent.
It also fails if an output lane is missing, a row count differs from the declared Cartesian product, an FP32 bit pattern differs, or an index differs.
CUDA allocation, copy, launch, and synchronization errors terminate the architecture binary.

## Limitations

This gate proves the standalone warp-local shuffle and shared comparison primitive.
It does not yet execute inside a CUTLASS EVT callback or combine candidates from different warps.
It does not use shared memory, partial tiles, or boundary predication.
Gate 1d must establish the cross-warp shared-memory reduction independently.
