# Gate 4 B200 Gumbel-Max TP1

Gate 4 integrates the Gate 3 Philox4x32-10 primitive and an open-interval Gumbel transform into the correctness-approved B200 CUTLASS schedules.
The sampling provider is compiled as a separate `fused-cutlass` extension, so the approved `fused-cutlass-greedy` binary remains unchanged.

## Correctness

Run deterministic checks with:

```text
make modal-cutlass GATE=gumbel-provider
```

Nine cases passed on B200.
They cover exact CPU Philox and Gumbel agreement over three seeds, temperatures 0.5, 1.0, and 2.0, same-seed reproducibility, different-seed separation, exact 17-sample batching, and invariance across the H=64, H=128, and H=256 donor transitions.

The sampling-only EVT derives RNG coordinates from the Gate 2d ownership formulas.
Its counter tuple is `(seed, sample_idx, hidden_idx, global_vocab_idx)` and does not depend on launch order.
The Python provider tiles multiple samples into sample-major hidden-state batches of at most 256 GEMM columns.

## Distribution

Run the ordered distribution phase with:

```text
make modal-cutlass GATE=gumbel-provider MODAL_ARGS="--phase distribution"
```

At V=128,000 and 10M samples, the test covered 123,333 bins and 99.8411% of the expected probability mass.
Reduced chi-squared was 0.994756 and p=0.903758.
The distribution phase passes.

## Performance decision

Run paired timings with:

```text
make modal-cutlass GATE=gumbel-provider MODAL_ARGS="--phase performance"
```

The measurement alternates provider order on one B200 and records 50 cold-L2 CUDA-event repetitions per cell.
The predeclared acceptance limit was 1.20x the matching greedy provider in every cell.

The D=8,192 ratios were 1.06--1.11x through H=64, then 3.43x at H=128 and 5.29x at H=256.
The D=4,096 ratios were 1.16--1.17x through H=16, 1.32x at H=32/64, 6.37x at H=128, and 8.48x at H=256.
Gate 4 therefore fails its performance criterion.

## NCU evidence

Run the matched profiles with:

```text
make modal-cutlass GATE=gumbel-ncu
```

At V=151,936 and D=4,096, H=64 uses 134 registers for Gumbel and 115 for greedy, with zero local-memory instructions for both.
The profiled GEMM durations were 247 and 192 microseconds.

At H=128 both kernels use 255 registers.
Gumbel executes 263,548 local loads and 447,564 local stores, versus 118,724 and 133,020 for greedy.
The profiled durations were 1.451 and 0.212 ms.

At H=256 both kernels again use 255 registers.
Gumbel executes 527,096 local loads and 889,320 local stores, versus 237,448 and 266,040 for greedy.
The profiled durations were 2.200 and 0.288 ms.

Gumbel issue activity falls below greedy at H=128/256 while its XU/SFU pipe remains active.
The matched profiles support spill amplification as a major contributor to the high-H regression.
They do not establish that spill traffic explains the entire duration difference.

## Current handoff

Gate 4 correctness and distribution pass, but performance is no-go.

The first bounded fallback moved Philox plus the Gumbel transform into a device `noinline` helper.
All deterministic cases and the complete 10M-sample distribution test pass unchanged.
At D=4,096, the H=128 ratio improved from 6.37x to 3.30x and H=256 improved from 8.48x to 4.37x.
At D=8,192, H=128 improved from 3.43x to 1.91x and H=256 improved from 5.29x to 2.63x.

NCU confirms that H=64 registers fell from 134 to 126.
At H=128, Gumbel local loads/stores fell from 263,548/447,564 to 166,180/308,740 and duration fell from 1.451 ms to 0.767 ms.
At H=256, they fell from 527,096/889,320 to 332,360/617,480 and duration fell from 2.200 ms to 1.213 ms.
The helper is retained because it produces a large measured recovery, but Gate 4 still fails.

The next attempted fallback was a non-warp-specialized epilogue candidate for the H=128/256 donor families.
The stock CUTLASS schedule did not compile with the provider's `ElementD=void` contract, so that drop-in experiment is closed.

## Partial-unroll follow-up

The non-warp-specialized epilogue is not directly usable with this provider because CUTLASS rejects `ElementD=void`, which suppresses the ordinary GEMM output.
Rolling the Gumbel callback loop reduces the high-H latency without changing ownership or RNG semantics.
The selected mixed configuration keeps the H<=64 128x64 callback fully unrolled, uses `#pragma unroll 2` for the cluster-2 256x128 donors, and uses `#pragma unroll 4` for the D=4,096 cluster-4 256x128 donor.
All nine deterministic Gate 4 cases pass for this exact configuration.

The paired B200 timing packet measures ratios of 1.06x/1.52x/2.30x for D=8,192 at H=64/128/256 and 1.22x/2.41x/3.39x for D=4,096 at the same H values.
The performance gate still fails the 1.20x requirement at high H, but this improves on the noinline-helper ratios of 1.91x/2.63x and 3.30x/4.37x.
Final NCU still reports 255 registers at H=128/256.
Gumbel local loads/stores are 555,900/883,848 at H=128 and 1,111,800/1,767,696 at H=256, so the change reshapes spill scheduling rather than eliminating spills.
The exact packets are under `benchmarking/modal-results/cutlass/18-gumbel-provider/`.

## Next steps

Gate 4 remains no-go, and Gate 5 is blocked.
The canonical bounded recovery plan is in `findings/cutlass/01-fmms-kernel-plan.md`.
First audit generated SASS and compiler spill slots for the selected partial-unroll control.
Then compare one smaller-granularity custom SM100 epilogue and one maintained vendor-backed Philox candidate.
Each candidate must use the current control and identical greedy donor in paired H=64/128/256 timings.
Only materially faster candidates earn matched NCU and the complete deterministic, 10M-sample distribution, and performance validation sequence.
If neither candidate removes RNG-caused spilling and passes the pointwise 1.20x limit, stop the complete CUTLASS sampling roadmap before Gate 5 unless the project explicitly adopts a greedy-only scope.
