# CUTLASS CTA boundary max-with-index

## Result

Gate 1f of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
The standalone CTA harness produced exact tile-local FP32 value bits and M indices for every required M and N boundary shape.
All 14,336 comparisons across 112 architecture-shape combinations passed.
Compute Sanitizer memcheck reported zero errors on both architectures.
Compute Sanitizer racecheck reported zero hazards, errors, and warnings on both architectures.

Run `make modal-cutlass GATE=cta-boundary-max`.
The generated verification packet is under `benchmarking/modal-results/cutlass/06-cta-boundary-max/`.

## Predication design

Gate 1f retains the architecture-specific SM90 and SM100 ownership routing proven by Gate 1e.
It adds explicit M and N extent checks without adding the global indices reserved for Gate 1g.

Each test operates on the final 128 by 128 tile implied by the requested global shape.
A dimension divisible by 128 has a full final tile.
Otherwise, the final tile contains the dimension remainder.
The reported `m_tile` and `n_tile` identify the final tile, while expected and actual indices remain tile-local.

Invalid M rows contain values above every valid value.
If M predication fails, a padded sentinel wins immediately.
Invalid N columns have output slots initialized to a canary and receive no store.
If N predication fails, the canary comparison fails.

## Shape coverage

The M extents are:

```text
100, 127, 128, 129, 255, 256, 257
```

The N extents are:

```text
1, 2, 63, 64, 65, 127, 128, 129
```

Both architectures cover the full Cartesian product.
For each of the 56 shapes per architecture, `cases.csv` records all 128 output columns.
Valid columns must select the final valid M coordinate.
Padded columns must retain the output canary.

Shapes 127, 128, and 129 exercise the positions immediately below, at, and above the tile boundary.
Shapes 255, 256, and 257 repeat that test at the next M boundary.
The N set additionally covers the 64-column midpoint and its adjacent dimensions.

## Evidence

The verification packet contains:

- 14,336 exact rows in `cases.csv`.
- 112 compact architecture-shape rows in `case-summary.csv`.
- One `summary.json` with shape sets, exact counts, index scope, comparison policy, and sanitizer status.
- Separate complete memcheck reports for SM90 and SM100.
- Separate complete racecheck reports for SM90 and SM100.
- A `VERIFY.md` review entry point and the complete `log.txt`.

Every partial M shape records a nonzero padded-row count and the FP32 bit pattern of its first larger sentinel.
Every partial N shape records a nonzero padded-column count.
Every shape has zero mismatches.

## Constraint review

Keeping indices tile-local still reduces risk.
It isolates invalid-lane predication from the global coordinate arithmetic and cross-tile tie behavior introduced in Gate 1g.
The final-tile construction covers both full and partial tiles while avoiding a second reduction domain.

The standalone harness uses physical warps 0 through 3 to model CUTLASS consumer-warp roles.
It does not instantiate a warp-specialized CUTLASS GEMM.
That remains appropriate because Gate 1f proves boundary masking around the shared reduction primitive, not EVT integration.

Running both memcheck and racecheck is necessary.
Racecheck validates the shared-memory handoff but does not establish that boundary loads and stores stay in bounds.
Memcheck covers that separate failure mode.

## Failure signatures

The gate fails if:

- Either architecture is absent.
- Any pair in the declared M and N Cartesian product is absent.
- Any shape does not record all 128 output columns.
- A valid column does not select the final valid tile-local M coordinate.
- A larger padded M sentinel wins.
- A padded N output does not retain its initialization canary.
- Any expected and actual FP32 bit pattern or M index differs.
- The CUDA launch, synchronization, allocation, or copy fails.
- Memcheck reports an error.
- Racecheck reports a hazard, error, or warning.
- The verification packet has an unexpected row or shape count.

## Limitations and next gate

This gate validates explicit boundary predication around the standalone CTA reduction.
It does not validate global vocabulary indices, ties across different M tiles, real GEMM accumulators, EVT callbacks, warp-specialized scheduling, or Stage 2 reduction.
Gate 1g should add global M offsets and deterministic lowest-global-index ties across multiple CTA coordinates while preserving this predication.
