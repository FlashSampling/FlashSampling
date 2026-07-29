# CUTLASS CTA-local max-with-index

## Result

Gate 1d of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
The harness combined four simulated CUTLASS consumer-warp winners through shared memory and produced the exact CPU-reference FP32 value bits and lowest tie index in all 14 architecture and case combinations.
Compute Sanitizer racecheck reported zero hazards, errors, and warnings on both architectures.

Run `make modal-cutlass-cta-max`.
The generated verification packet is under `benchmarking/modal-results/cutlass-cta-max/`.

## What this gate proves

The CUDA kernel launches one 128-thread CTA whose physical warps 0 through 3 represent the CUTLASS consumer-warp roles 4 through 7 identified in Gate 1a.
It does not instantiate a warp-specialized CUTLASS GEMM or its producer warps.
Each warp first applies the Gate 1c shuffle reduction to its architecture-specific M lanes.
Lane zero writes the warp winner to a four-element shared-memory array.
After a CTA barrier, the first warp loads those four candidates and applies the same deterministic max-with-index comparator.
The final winner is written by one lane.

SM90 uses lanes 0,4,...,28 for one N column.
SM100 uses lanes 0,...,31.
The shared-memory exchange is architecture independent once each warp has produced one candidate.

The cases include a unique winner from every contributing warp, an all-negative input, and equal-value ties whose lowest index appears in either the earlier or later warp.
Expected and actual FP32 bit patterns and indices are compared exactly.

## Evidence

The combined packet contains:

- 14 exact result rows in `cases.csv`.
- 14 compact review rows in `case-summary.csv`.
- One `summary.json` with the expected and actual counts, exact-comparison policy, and per-architecture racecheck status.
- Separate complete `racecheck-sm90.txt` and `racecheck-sm100.txt` outputs.
- A `VERIFY.md` review entry point and the complete `log.txt`.

Both architectures passed all seven cases.
Racecheck reported `0 hazards displayed (0 errors, 0 warnings)` for each architecture.

## Constraint review

The one-column, complete-M-tile constraint still reduces risk.
It isolates shared-memory publication, CTA synchronization, and cross-warp comparison from the multi-column ownership mapping introduced in Gate 1e.
Relaxing it now would make a column-routing error indistinguishable from a shared-memory reduction error.

The harness starts from the proven thread-local and warp-local candidates rather than GEMM accumulators.
On SM90, each participating lane represents the thread-local state that covers multiple M coordinates observed in Gate 1a.
On SM100, all 32 lanes per warp participate.
This abstraction tests the complete CTA candidate hierarchy, but it does not retest fragment visitation or warp-specialized scheduling.

## Failure signatures

The gate fails if:

- Either architecture is absent.
- Any consumer warp never supplies a unique winner.
- Either cross-warp tie order selects the wrong index.
- Any expected and actual FP32 bit pattern or index differs.
- The CUDA launch, synchronization, or copy fails.
- Racecheck reports a hazard, error, or warning.
- The result stream or verification packet is incomplete.

## Limitations and next gate

This gate covers one complete 128-row M tile and one N column.
It does not prove independent state for multiple columns, boundary predication, global vocabulary indices, EVT callback integration, warp-specialized execution, or multi-tile Stage 2 reduction.
Gate 1e should reuse this CTA primitive while adding the architecture-specific per-column routing established by Gate 1a.
