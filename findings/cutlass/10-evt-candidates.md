# CUTLASS EVT per-tile candidates

## Result

Gate 1g of the CUTLASS FMMS plan passed on H100 and B200 on 2026-07-29.
All 1,744 emitted candidates matched their real GEMM M-tile references exactly across 12 architecture-case combinations.
Compute Sanitizer memcheck reported zero errors, and racecheck reported zero hazards, errors, and warnings on both architectures.

The implementation is in `src/fused_mm_sampling/csrc/cutlass/evt_candidates.cu`.
Run it with `make modal-cutlass GATE=evt-candidates`.
The human-verifiable packet is written to `benchmarking/modal-results/cutlass/07-evt-candidates/`.

## Method

The harness runs a warp-specialized CUTLASS BF16 GEMM with FP32 accumulation.
A split-tree EVT first packs each accumulator and its global M coordinate into one `uint64_t`.
The high 32 bits preserve the FP32 value bits, and the low 32 bits preserve the signed i32 vocabulary index.
An auxiliary `Sm90RowReduction` uses `FinalReduction=false` to emit one packed candidate per M tile and N column.
The root writes a disposable FP32 diagnostic output, so Stage 2 is not present.

The input construction is deterministic and exactly representable.
Only K coordinate zero is nonzero, so each reference logit is one BF16 value multiplied by one.
This still executes the real CUTLASS GEMM mainloop and passes its FP32 accumulators into the EVT.
The harness reconstructs every tile slice on the host and compares the packed value bits and global index exactly.

The test families cover:

- Nonzero M-tile offsets and winners at tile-local M coordinates 0 and 127.
- Complete 128 by 128 tiles and partial M and N tiles.
- All-negative logits and equal maxima within one tile.
- Equal maxima in different full tiles, with each losing or winning tile candidate retained before Stage 2.

## Architecture-specific accumulator ownership

The callback cannot derive an accumulator fragment's M coordinate from `tCcD`.
That tensor describes the epilogue store-copy partition, which is not the accumulator ownership mapping used by the custom input leaf.
Using it produced the correct value but index 200 instead of 255 for the second SM90 M tile.

The final implementation uses the ownership mappings measured in Gate 1a.
On SM90, M depends on consumer warp, the lane group, `epi_m`, and the fragment pair.
On SM100, each consumer thread owns one M coordinate across all its fragment slots.
Both formulas then add `m_tile * 128` to form the global vocabulary index.

This dependency is deliberate and guarded by the architecture-specific compile definitions.
Any epilogue schedule, tile shape, or CUTLASS version change requires rerunning Gate 1a before trusting these formulas.

## Constraint review

Keeping `FinalReduction=false` still isolates EVT integration from Stage 2 and remains useful.
Keeping packed FP32 plus i32 candidates also remains useful because CUTLASS shuffle and shared-memory reductions move the pair together and deterministic ties never lose their index.

The TMA epilogue requires the contiguous FP32 N extent to be aligned to four elements.
The partial-N cases therefore use N=68 and N=132 rather than unaligned extents such as 65 and 129.
They remain genuine partial 128-column tiles while satisfying the production kernel's TMA contract.
Dropping TMA or adding a second direct-store implementation solely to test unaligned N would expand the gate without reducing risk because Gate 1f already covers arbitrary N predication.

The FP32 root output is diagnostic only.
The candidate buffer is the meaningful output, and a later production integration can remove the disposable D store if its epilogue construction supports an auxiliary-only EVT on both architectures.
The SM90 build also emits a ptxas warning that WGMMA instructions may serialize across a function-call boundary.
This correctness gate makes no performance claim, and the warning must be resolved or measured before the later greedy-performance gate.

## Failure signatures

The gate fails if:

- Either architecture or any declared test family is absent.
- Any expected `(m_tile, column)` coordinate is absent.
- Any candidate FP32 bit pattern or global index differs.
- A within-tile tie does not choose the lowest global index.
- Equal maxima in different tiles do not retain the correct per-tile indices.
- A partial M row contributes a padded coordinate.
- The CUDA launch, synchronization, allocation, or copy fails.
- Memcheck reports an error.
- Racecheck reports a hazard, error, or warning.

## Limitations and next gate

The deterministic inputs intentionally make the GEMM reference exactly reproducible.
This gate does not measure numerical error for general dense BF16 inputs, performance, Stage 2, sampling, or tensor parallelism.
It also does not prove that a different CUTLASS schedule preserves the recorded callback ownership.

Gate 1h should merge these packed candidates with the existing GPU Stage 2.
Its packet must retain both the intermediate candidates and final values and indices, including first-, middle-, and last-tile winners and deterministic cross-tile ties.
