# CUTLASS CTA multi-column max-with-index

## Result

Gate 1e of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
The standalone CTA harness produced exact FP32 value bits and M indices for all 128 columns of a complete 128 by 128 output tile.
Both deterministic cases used every M position exactly once as a column winner.
All 512 architecture, case, and column comparisons passed.
Compute Sanitizer racecheck reported zero hazards, errors, and warnings on both architectures.

Run `make modal-cutlass-cta-multi-column-max`.
The generated verification packet is under `benchmarking/modal-results/cutlass-cta-multi-column-max/`.

## Architecture-specific routing

Gate 1a established different accumulator ownership layouts on SM90 and SM100.
Gate 1e encodes those layouts independently.

On SM90, each epilogue N iteration spans 32 columns.
A thread owns two M positions separated by eight within each of the two 64-row epilogue M iterations.
For one column, eight lanes with the same `lane % 4` value reduce their four thread-local M candidates with a shifted mask.
The four warp winners then pass through the shared-memory CTA reduction.
Four epilogue N iterations cover columns 0 through 127.

On SM100, each epilogue N iteration spans 16 columns.
Each thread owns one M position and 16 adjacent N positions.
All 32 lanes reduce one M candidate per column before the four warp winners pass through the same shared-memory CTA reduction.
Eight epilogue N iterations cover columns 0 through 127.

The shuffle implementation is now shared in `max_with_index.cuh`.
Gates 1c and 1d use the same primitive as Gate 1e.

## Test construction

The `independent_unique` case uses:

```text
winner_m(n) = (37n + 11) mod 128
winner_value(n) = 1000 + n
```

The `all_negative` case uses:

```text
winner_m(n) = (53n + 7) mod 128
winner_value(n) = -0.25 - n / 256
```

Both M formulas are permutations because 37 and 53 are coprime with 128.
Every column therefore has a different winning M position, and every M position wins exactly once in each case.
The distinct per-column values and indices make cross-column contamination observable.

## Evidence

The combined packet contains:

- 512 exact per-column rows in `cases.csv`.
- 24 compact architecture, case, and epilogue-iteration rows in `case-summary.csv`.
- One `summary.json` with exact counts, iteration widths, comparison policy, and racecheck status.
- Separate complete `racecheck-sm90.txt` and `racecheck-sm100.txt` outputs.
- A `VERIFY.md` review entry point and the complete `log.txt`.

SM90 covers four 32-column epilogue N iterations.
SM100 covers eight 16-column epilogue N iterations.
The first and last column of every iteration are labeled explicitly and checked by the runner.

After moving the shuffle helper into shared code, Gates 1b, 1c, and 1d were rerun.
They retained 9,728, 4,832, and 14 exact passing comparisons respectively.
Gate 1d racecheck remained clean on both architectures.

## Constraint review

The complete-tile constraint still reduces risk.
It isolates architecture-specific column routing from the boundary predication introduced in Gate 1f.
Adding partial M or N tiles here would make an ownership-routing failure difficult to distinguish from an invalid-lane masking failure.

The standalone harness uses physical warps 0 through 3 to model CUTLASS consumer-warp roles.
It does not instantiate a warp-specialized CUTLASS GEMM.
That separation remains useful until the reduction and boundary behavior are both proven independently.

## Failure signatures

The gate fails if:

- Either architecture or deterministic case is absent.
- Any of the 128 columns is missing.
- The expected winner indices do not form a complete M permutation in either case.
- Either endpoint of any architecture-specific epilogue N iteration is absent.
- Any expected and actual FP32 bit pattern or M index differs.
- The CUDA launch, synchronization, or copy fails.
- Racecheck reports a hazard, error, or warning.
- The verification packet has an unexpected row count.

## Limitations and next gate

This gate covers one complete 128 by 128 tile.
It does not test partial M or N tiles, global vocabulary offsets, EVT callback integration, warp-specialized scheduling, or multi-tile Stage 2 reduction.
Gate 1f should preserve the proven routing while adding explicit predication for the boundary-shape matrix.
