# CUTLASS multi-warpgroup epilogue and partitioned TMEM loads

## Result

The loading inefficiency is fixed.
Each four-warpgroup consumer now loads only its four owned FP32 accumulator values from TMEM, and each two-warpgroup consumer loads only its eight owned values.
Fixed register destinations avoid the local-memory materialization caused by runtime register-array indexing.
The selected high-H schedule also removes the now-redundant shared-memory accumulator round trip.

The current working-tree production dispatch uses the accurate single-warpgroup immediate-reduction kernel for H<=64 and the accurate four-warpgroup partitioned kernel for H=128 and H=256.
Other H values retain the generic provider because they have not received matched schedule validation.
The fast-log variants remain experimental because the 10-million-sample distribution gate rejected their numerical behavior.

## Correctness and ownership fixes

The original striped schedule let every physical consumer group execute a complete 16-value TMEM load even though a four-group schedule used only four values per group.
Raw `SM100_TMEM_LOAD_32dp32b4x` and `SM100_TMEM_LOAD_32dp32b8x` paths now load contiguous owned fragments for four and two groups respectively.
A full-load versus partial-load diagnostic proved that the raw TMEM addresses and loaded bits were correct.

The first partial-load implementation still produced deterministic mismatches because the patched CUTLASS epilogue computed its group from `threadIdx.x % ThreadCount`.
The four producer warps occupy threads 0 through 127, so the first physical consumer group was incorrectly normalized as group one.
The corrected index is `(threadIdx.x - 128) % ThreadCount`, which now agrees with the callback's consumer-group mapping.

The first correct partial-load implementation wrote into `rAcc(runtime_group * count + i)`.
That runtime index forced the compiler to materialize the register array in local memory.
The final implementation always writes the partial TMEM load into fixed local registers 0 through 3 or 0 through 7.
The callback computes only the global fragment base at runtime and accesses each local accumulator with a compile-time-unrolled index.

The upstream accumulator-pipeline barrier expects every epilogue thread to release the accumulator stage.
Attempts to reduce release arrivals or synchronize release across groups either violated that contract or hung.
The final patch leaves the original per-thread `consumer_release` behavior unchanged.

## Timing result

The table reports the accurate production candidates divided by interleaved Triton latency on B200.

| D | H=64, one group | H=128, four groups | H=256, four groups |
| ---: | ---: | ---: | ---: |
| 4,096 | 1.12x | 1.04x | 1.05x |
| 8,192 | 0.99x | 0.89x | 0.78x |

The accurate one-group packet is under `benchmarking/modal-results/cutlass/experiments/accurate-warpgroup-vs-fastmath/`.
The accurate four-group packet is under `benchmarking/modal-results/cutlass/experiments/accurate-4wg-vs-fastmath/`.
Every candidate passed the two focused exact-winner cases before timing.

Removing shared-memory accumulator staging from the fixed-register four-group kernel improved D=4,096 and H=128 by 5.4% to 12.2% across three same-process packets.
It improved D=4,096 and H=256 by 7.3% to 9.5%.
The corresponding packets are `no-stage-4wg-vs-staged`, `no-stage-4wg-vs-staged-r2`, and `no-stage-4wg-vs-staged-r3` under the experiment result root.

The faster `__logf` and `__fdividef` diagnostic reached 0.86x to 0.98x Triton at D=4,096 and H=128 and 0.89x to 0.93x at H=256 across those runs.
It is not selected because the standard 10-million-sample distribution gate produced p=0.000415, below the predeclared 0.001 threshold.

## Production comparison with greedy

The final accurate production packet uses cold-L2 CUDA events with alternating provider order.

| D | H | CUTLASS Gumbel, ms | CUTLASS greedy, ms | Gumbel / greedy |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 64 | 0.328 | 0.242 | 1.36x |
| 4,096 | 128 | 0.395 | 0.258 | 1.53x |
| 4,096 | 256 | 0.604 | 0.333 | 1.81x |
| 8,192 | 64 | 0.416 | 0.356 | 1.17x |
| 8,192 | 128 | 0.467 | 0.388 | 1.20x |
| 8,192 | 256 | 0.647 | 0.495 | 1.31x |

The old pointwise 1.20x-greedy gate therefore still fails, with a worst measured ratio of 1.81x.
The remaining high-H difference from greedy is accurate Gumbel generation and reduction work rather than duplicated TMEM loads.
The packet is `benchmarking/modal-results/cutlass/18-gumbel-provider/performance-summary.csv`.

## Profiler result

At D=4,096 and H=256, the unstaged fast-log four-group diagnostic measured 451.264 microseconds, 54 registers, zero local loads and stores, 52.82% issue activity, and 209.14 million executed instructions.
Matched Triton measured 476.736 microseconds and 238.06 million instructions in that packet.

At the same shape, the selected accurate four-group kernel measured 527.200 microseconds, 72 registers, zero local loads and stores, 56.66% issue activity, and 266.78 million instructions.
Matched Triton measured 459.520 microseconds and 238.06 million instructions.
This isolates the remaining D=4,096 gap to accurate Gumbel arithmetic rather than spills or insufficient issue activity.

At D=8,192 and H=256, the accurate kernel measured 525.792 microseconds versus Triton's 639.424 microseconds.
It allocated 72 registers and executed 1,026,048 local loads and 384,768 local stores, so accurate libdevice arithmetic still creates a D=8,192 spill path.
That spill is a remaining optimization target, but it does not prevent the kernel from beating Triton at that shape.

Moving Gumbel generation into a no-inline device function did not remove this spill.
It lowered allocation from 72 to 62 registers but retained exactly 1,026,048 local loads and 384,768 local stores.
It increased executed instructions from 237.51 million to 259.52 million and slowed NCU duration from 525.792 to 584.096 microseconds.
The no-inline variant is rejected and removed from the registry.

The raw reports and local exports are under `benchmarking/modal-results/cutlass/experiments/warpgroup-4wg-partitioned/` and the matching shared-Volume paths recorded in its NCU summaries.

## Validation and decision

Validation command: `make modal-cutlass GATE=gumbel-provider`.
Expected result: all deterministic cases pass with the production shape dispatch.
Actual result: all nine cases passed, including the exact H=2 batched case and H=64, H=128, and H=256 schedule invariance.
Possible failures were incorrect physical-group normalization, incomplete fragment ownership, small-H schedule incompatibility, or a pipeline-release hang.
Artifacts are `benchmarking/modal-results/cutlass/18-gumbel-provider/summary.json` and `cases.csv`.

Validation command: `make modal-cutlass GATE=gumbel-provider MODAL_ARGS='--phase distribution'`.
Expected result: the accurate production path exceeds the predeclared p-value threshold of 0.001.
Actual result: 10,000,000 samples passed with p=0.903758, reduced chi-squared 0.994756, 123,333 tested bins, and 0.998411 tested probability mass.
Possible failures were fast-log bias, RNG stream mismatch, or incorrect sample chunk offsets.
The fast-log production probe did fail this gate at p=0.000415 and was rejected.
The passing artifact is `benchmarking/modal-results/cutlass/18-gumbel-provider/distribution-summary.json`.

Validation command: `make modal-cutlass GATE=gumbel-provider MODAL_ARGS='--phase performance'`.
Expected result: the legacy feasibility gate requires every Gumbel latency to be no greater than 1.20x matched greedy.
Actual result: the loading fix materially improves the kernel, but the accurate production dispatch still reaches 1.81x greedy at D=4,096 and H=256.
Possible failures were host-class variation, cold-L2 state variation, fast-log numerical rejection, and accurate-log instruction or spill cost.
The artifact is `benchmarking/modal-results/cutlass/18-gumbel-provider/performance-decision.json`.

Keep the accurate dispatch in the working tree as the correct implementation state.
Do not promote `__logf` or `__fdividef` without a new statistical result that passes the declared gate.
The next bounded performance targets are accurate-log instruction cost at D=4,096 and the accurate D=8,192 local-memory path.
