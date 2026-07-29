# CUTLASS accumulator callback layout

## Result

Gate 1a of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
The diagnostic covered all 16,384 coordinates of one 128 by 128 CTA output tile exactly once on each architecture.
No coordinate was missing or had multiple owners.

The diagnostic is implemented in
`src/fused_mm_sampling/csrc/cutlass/accumulator_layout.cu`.
Run it with `make modal-cutlass GATE=accumulator-layout`.
The runner writes the full log and the raw coordinate mappings to
`benchmarking/modal-results/cutlass/01-accumulator-layout/`.

## Method

The custom EVT leaf ignores the accumulator value and returns a losslessly encoded FP32 ownership record.
The record contains `threadIdx.x`, fragment slot, `epi_v`, `epi_m`, and `epi_n`.
The ordinary CUTLASS epilogue stores this record at the output coordinate owned by that callback invocation.
The Modal runner decodes the records, constructs a pandas coordinate index, and requires exact equality with the complete 128 by 128 coordinate grid.

This tests the actual compiled callback layout instead of inferring it from CUTLASS types.
It also avoids adding reduction logic before coordinate ownership is known.

## Observed architecture difference

Both schedules use consumer threads 128 through 255 and expose 16 accumulator values per callback.
Their epilogue iteration structure differs:

| Architecture | Schedule | `epi_m` | `epi_n` |
|---|---|---:|---:|
| SM90 | `KernelTmaWarpSpecialized` and `TmaWarpSpecialized` | 0 through 1 | 0 through 3 |
| SM100 | `KernelTmaWarpSpecialized1SmSm100` and `TmaWarpSpecialized1Sm` | 0 | 0 through 7 |

On SM90, one thread owns two M positions separated by eight within each 64-row `epi_m` region.
Its 16 fragment slots cover eight N groups and the two M positions.
For example, thread 128 owns M 0 and 8 in the first `epi_m` region, then M 64 and 72 in the second region.

On SM100, one thread owns one M position and 16 adjacent N positions per callback.
For example, thread 128 owns M 0, while fragment slots 0 through 15 own consecutive N positions.
The eight `epi_n` iterations advance across the complete 128-column tile.

The next reduction micro-gate must therefore use architecture-specific ownership tests.
It must not assume that the SM90 fragment layout carries over to SM100.
