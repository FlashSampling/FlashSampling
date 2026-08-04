# Gate 4 Gumbel performance recovery

## Outcome

The generic CUTLASS Gumbel-Max callback, not the CUTLASS GEMM mainloop, causes the high-H performance loss against Triton.
The complete accumulator remains in TMEM in both implementations.
The original CUTLASS callback spills because it retains a large packed row-reduction state, while Triton reduces each loaded fragment immediately with more consumer warps and dynamic register allocation.

The custom CUTLASS reduction now removes the persistent state and all callback atomics.
Shared-memory fragment staging also removes all measured local-memory traffic.
The remaining gap is lower issue activity under CUTLASS's stock 128-consumer-thread SM100 epilogue schedule.

The fastest spill-free experiment combines the shared-memory path with fast logarithm and fast division.
It is essentially tied with the rolled fast-math path while removing all measured local-memory traffic.
Neither has been promoted into the production provider.

## Matched baseline

Two same-process B200 runs compared CUTLASS greedy, production CUTLASS Gumbel-Max, and Triton Gumbel-Max with rotating provider order, cold-L2 preparation, and 50 CUDA-event repetitions.
For V=128,256 and D=8,192, production CUTLASS is 1.33x Triton at H=128 and 1.54x to 1.56x at H=256.
For V=151,936 and D=4,096, it is 2.01x at H=128 and 2.15x to 2.17x at H=256.
CUTLASS greedy remains approximately 33% and 37% faster than Triton Gumbel-Max in the corresponding H=256 cells.
This isolates the regression to the sampling expression and does not show that Triton is intrinsically faster than CUTLASS.

The timing packets are under `benchmarking/modal-results/cutlass/22-gumbel-vs-triton/`.

## SASS attribution and rejected vendor RNG

The production 256x128 Gumbel donors use 255 registers and 288 to 304 byte compiler stack frames.
Static disassembly attributes 64 local loads and 16 to 32 local stores per donor to materializing the 16-element packed candidate return array.
These static counts motivated the custom void-output reduction, while matched NCU supplies the dynamic traffic evidence below.
The disassembly packet is under `benchmarking/modal-results/cutlass/19-gumbel-sass/`.

The maintained cuRAND `curandStatePhilox4_32_10_t` candidate is rejected.
It was 4.5% to 7.1% slower than the existing stateless helper at H=128/256, retained the identical 255-register allocation and local traffic, and executed 6.7% more warp instructions.
The packet is under `benchmarking/modal-results/cutlass/20-gumbel-vendor-philox/`.

The generic 128x8 epilogue experiment is also rejected.
It retained 255 registers, increased dynamic local loads by 111% and local stores by 70%, and did not implement the required void-output custom reduction.
The packet is under `benchmarking/modal-results/cutlass/21-gumbel-small-epilogue/`.

## Direct answer about the accumulator

Neither the Triton kernel nor the Blackwell CUTLASS kernel keeps the complete GEMM accumulator tile in ordinary registers.
Both kernels accumulate the matrix product in Blackwell tensor memory, or TMEM.
The relevant performance difference begins when each kernel loads a completed FP32 accumulator fragment from TMEM and applies Gumbel-Max.

The Triton TTGIR allocates a double-buffered `2x128x64xf32` TMEM object with `ttng.tmem_alloc`.
It loads one completed `128x64` tile with `ttng.tmem_load`, applies temperature scaling, Philox, two logarithms, and `tt.reduce`, then releases that tile through the pipeline barrier.
The CUTLASS SM100 epilogue also uses a `cute::tmem_ptr<float>` accumulator engine and presents 16 FP32 accumulator values to each consumer callback.

## Why Triton does not spill

The matched Triton cubin launches with 128 registers per thread and reports zero dynamic local loads and stores at D={4,096,8,192} and H={128,256}.
Its PTX uses warp-specialized dynamic register allocation and raises consumer warps to 176 registers with `setmaxnreg.inc.sync.aligned.u32 176` while producer warps surrender registers.
The complete matrix accumulator remains in TMEM, and the live consumer fragment is reduced immediately instead of being retained as a persistent per-thread row state.
The generated Triton artifacts are under `benchmarking/modal-results/cutlass/24-triton-liveness/`.

## Why the generic CUTLASS callback spills

The selected generic CUTLASS row reduction allocates 128 packed FP32-value/i32-index candidates per consumer thread.
That state alone represents 256 32-bit register words before the 16 FP32 accumulator values and Philox/Gumbel temporaries are considered.
The resulting production kernels allocate 255 registers and execute substantial local-memory traffic.
At D=4,096, H=128 the selected control executes 555,900 local loads and 883,848 local stores.
At D=8,192, H=128 it executes 617,232 local loads and 765,528 local stores.
The corresponding Triton kernels execute zero local loads and stores.

## No-persistent-state reduction experiments

The first custom visitor removed the persistent 128-candidate row state and reduced each 16-value callback fragment immediately inside each warp.
It then used four deterministic 64-bit CAS reductions per physical CTA and output column.
The implementation was bit-exact, but the four-atomic version was 2.36x to 4.55x slower than Triton at H=128/256 in its first timing packet.

NCU showed that this visitor allocated only 40 registers at D=4,096 and 47 registers at D=8,192.
It nevertheless executed 617,320 local loads at D=4,096, H=128 and 522,323 local loads at D=8,192, H=128.
This rejected the simple claim that lowering the allocated register count had removed the spill path.

A 512-byte shared-memory scratch then merged the four warp winners before global output.
This reduced the global update count from four CAS operations to one CAS per physical CTA and output column.
At H=256 it reduced latency from 2.22 ms to 1.37 ms for D=4,096 and from 1.78 ms to 1.14 ms for D=8,192.

The next version stored the merged winner directly into the physical CTA's unique candidate slot and retained the existing Stage 2 tile merge.
This removed atomics from the custom path entirely.
It improved D=4,096 by approximately 4% to 5% and was neutral within host variance at D=8,192.
The single remaining CAS was therefore measurable but did not explain the remaining Triton gap.

## Rolled fragment indexing

Inlining Philox and the Gumbel transform reduced executed instructions and raised issue activity, but did not remove local traffic.
The inlined direct-store kernel executes exactly 607,744 local loads at V=151,936 and H=128.
That count equals `V * H / 32`, which is one warp-level local load per output element.
The same identity holds for all four profiled shapes, including 1,215,488 loads at V=151,936 and H=256, 513,024 loads at V=128,256 and H=128, and 1,026,048 loads at V=128,256 and H=256.
This exact relationship identifies the rolled `accumulators[i]` access as the remaining local-load path.

The inlined kernel still improved the paired timing result.
At D=8,192 it measured 1.22x Triton at H=128 and 1.36x at H=256.
At D=4,096 it measured 1.97x Triton at H=128 and 2.17x at H=256.
Matched NCU measured 60 registers for D=4,096, 66 registers for D=8,192, and issue activity of approximately 21% to 24%.
Triton retained zero local traffic and approximately 33% to 55% issue activity in the same profiles.

## Constant-index experiments

The full-unroll candidate made all 16 accumulator indices compile-time constants.
It passed exact correctness but regressed to 2.29x and 2.61x Triton at D=8,192 for H=128 and H=256.
At D=4,096 it regressed to 3.54x and 3.09x.
Two cold NCU attempts were stopped after approximately eleven minutes without profiling a kernel, so this candidate does not yet have dynamic local-traffic evidence.

The shared-memory candidate wrote the 16 constant-index accumulator values to an 8 KiB CTA scratch and then consumed them through the rolled loop.
It eliminated local loads and stores completely in all four matched NCU cells.
The kernels allocated 64 registers at D=4,096 and 68 registers at D=8,192.
They measured 2.02x and 2.22x Triton at D=4,096 and 1.24x and 1.39x at D=8,192 for H=128 and H=256.
Those results are 2% to 4% slower than the inlined direct-store candidate despite eliminating local memory.

The chunked candidate instead used constant accumulator indices and compiler memory barriers after each four-value group.
It also eliminated all local loads and stores, while allocating 96 registers at D=4,096 and 100 registers at D=8,192.
Issue activity fell to 9% to 16%, compared with 20% to 23% for the shared-memory version and 32% to 55% for Triton.
Its timing regressed to 2.34x to 3.59x Triton at H=128 and 2.56x to 3.29x at H=256.
This rejects local-memory elimination by itself as a sufficient performance criterion.
The implementation must preserve enough instruction-level and warp-level latency hiding while bounding liveness.

## Logarithm-lowering probe

The zero-local-memory shared-memory profile executes roughly the same total instruction count as Triton.
At D=8,192 and H=128 it executes 106.4 million SM-subpartition instructions versus Triton's 117.0 million, yet its issue activity is 20.5% versus 31.7%.
The CUTLASS kernel has eight active warps per SM while Triton has twelve, because the stock CUTLASS SM100 epilogue fixes `ThreadCount = 128` consumer threads and the complete kernel launches 256 threads.
Triton launches 384 threads and uses dynamic register allocation to give its consumer warps 176 registers.

The NCU pipeline counters also identify a mathematical execution difference that requires a narrower probe.
At D=8,192 and H=128 the CUTLASS shared-memory candidate executes 2.57 million XU-pipe instructions, while Triton executes 0.23 million.
The CUTLASS helper calls accurate CUDA `logf` twice for each Gumbel value.
The saved Triton PTX contains `__nv_logf` exit blocks, so the counter gap does not by itself prove that Triton uses a faster logarithm implementation.
The isolated fast-log probe retained local traffic but improved timing, which admitted the later fast-division and spill-free shared-memory combinations.
Correctness and distribution must be rechecked before either approximation is promoted because both change numerical behavior.

## Bounded multi-value ILP experiments

The two-value and four-value candidates generated independent Philox counters round by round, then reduced each completed Gumbel candidate before starting the next bounded group.
Both variants passed exact winner correctness and eliminated all dynamic local loads and stores in every completed NCU cell.
The result nevertheless regressed sharply because the larger straight-line regions reduced scheduler issue activity instead of hiding the Philox and logarithm latency.

At D=4,096, the two-value path allocated 72 registers and reached only 8.1% and 11.6% issue activity at H=128 and H=256.
It measured 3.51x and 3.47x Triton in the timing packet.
At D=8,192, it allocated 84 registers, reached 7.7% and 8.4% issue activity, and measured 2.13x and 2.55x Triton.

The four-value path allocated 88 registers at D=4,096 and reached 8.2% and 11.9% issue activity.
It measured 3.13x and 3.43x Triton at H=128 and H=256.
The D=8,192 timing result measured 1.93x and 2.32x Triton.
Its matched D=8,192 profile was stopped after 671.70 seconds without completing, as recorded in the CUTLASS development-infrastructure finding.

These results reject bounded Philox batching as a recovery mechanism.
They also strengthen the distinction between eliminating spills and preserving latency hiding: the rolled fast-log candidate retains local traffic but reaches approximately 19% to 21% issue activity, while Triton combines zero local traffic with approximately 33% to 55% issue activity and twelve active warps.

The pinned CUTLASS 4.6.1 SM100 kernel computes its block size from four fixed infrastructure warps plus `CollectiveEpilogue::ThreadCount`.
The stock TMA epilogue fixes that thread count at 128, producing the observed 256-thread, eight-warp block.
There is no public dense SM100 schedule tag that selects a second epilogue warp-group.
The supported-layout experiment therefore split high-H work into 64-column CTA tiles to increase independent CTAs without forking the CUTLASS kernel internals.

## Sixty-four-column schedule probe

The `warpgroup-n64` candidate uses supported two-SM schedules with 256x64x128 tiles at D=8,192 and 256x64x64 tiles at D=4,096.
It passed the focused exact-winner check.
At D=8,192 it measured 0.99x, 1.15x, and 1.26x Triton at H=64, H=128, and H=256.
At D=4,096 it measured 1.02x, 1.76x, and 2.02x Triton.

The D=8,192 profiles allocated 56 registers, retained 513,024 and 1,026,048 dynamic local loads, and reached 20.57% and 21.37% issue activity at H=128 and H=256.
The D=4,096 profiles allocated 52 registers, retained 607,744 and 1,215,488 dynamic local loads, and reached 21.04% and 21.41% issue activity.
The smaller N tile modestly improves some cells but does not remove the rolled fragment access or the fixed four-consumer-warp structure.

## Spill-free fast-log path

The `warpgroup-fastlog-smem` candidate combines CUDA `__logf` with constant-index shared-memory accumulator staging.
It passed the focused exact-winner check and reports zero dynamic local loads and stores in all four matched high-H profiles.
It allocates 55 registers at D=4,096 and 59 registers at D=8,192.

At D=8,192 it measured 1.02x, 1.22x, and 1.32x Triton at H=64, H=128, and H=256.
At D=4,096 it measured 1.07x, 1.89x, and 2.11x Triton.
Issue activity remains 18.42% to 20.51% at D=8,192 and 19.15% to 19.94% at D=4,096.
This is the cleanest direct evidence that the CUTLASS spill is fixed while the schedule still fails to hide the epilogue latency as effectively as Triton.

## Fast-division probe

The `warpgroup-fastmath` candidate adds CUDA `__fdividef` to the rolled, immediate, direct-store fast-log path.
It passed the focused exact-winner check and is the fastest completed custom candidate.
At D=8,192 it measured 0.99x, 1.11x, and 1.21x Triton at H=64, H=128, and H=256.
At D=4,096 it measured 0.98x, 1.73x, and 1.89x Triton.

The fast division removes approximately 4 million and 8 million executed instructions at D=8,192 for H=128 and H=256 relative to the fast-log control.
It removes approximately 5 million and 10 million instructions at D=4,096.
The profiles allocate 56 registers but retain the exact rolled-path local-load counts.
This shows that arithmetic lowering is independently valuable, although it does not solve the local-memory path.

## Combined fast-math and spill-free path

The `warpgroup-fastmath-smem` candidate combines `__logf` and `__fdividef` with constant-index shared-memory accumulator staging.
It passed both focused exact-winner checks and reports zero dynamic local loads and stores in all four matched NCU cells.
It allocates 54 registers at D=4,096 and 58 registers at D=8,192.

At D=8,192 it measured 0.99x, 1.12x, and 1.20x Triton at H=64, H=128, and H=256.
At D=4,096 it measured 1.03x, 1.63x, and 1.85x Triton.
Relative to the spill-free fast-log control, fast division removes 4.47% to 4.73% of executed instructions and 14.28% of XU-pipe instructions.
Issue activity changes by less than 0.22 percentage points and remains 18.52% to 20.69%, so cheaper division does not address the stock epilogue schedule's latency-hiding limit.

The timing packet completed in 255.59 seconds, of which the cold extension build consumed 231.57 seconds, correctness consumed 1.20 seconds, and timing consumed 3.08 seconds.
After the cache fill, the D=4,096 and D=8,192 NCU jobs ran concurrently and completed in 35.96 and 36.36 seconds.
Their warm extension loads took at most 1.83 seconds.
The packet is under `benchmarking/modal-results/cutlass/experiments/warpgroup-fastmath-smem/`.

## Four-output Philox probe

Triton 3.6 `tl.rand` calls `randint`, which consumes only the first output produced by `randint4x`.
The four-output CUTLASS candidate is therefore a new RNG mapping experiment, not an explanation for why the current Triton kernel avoids spilling.
It maps four adjacent global sample streams to the four results from one Philox counter and handles unaligned stream groups with a second counter.

The candidate passed the focused exact-winner check against a reference implementing that mapping.
It reports zero dynamic local loads and stores and allocates 74 registers at D=4,096 and 86 registers at D=8,192.
It nevertheless measures 2.95x and 3.08x Triton at D=4,096 and 1.79x and 2.06x Triton at D=8,192 for H=128 and H=256.
Issue activity is only 7.85% and 11.42% at D=4,096 and 7.32% and 8.11% at D=8,192.
This rejects four-output Philox batching in its current straight-line form and supplies a second independent example where eliminating all local traffic is insufficient.

## Current conclusion

The CUTLASS accumulator is not kept as a complete register tile, and the generic callback's 255-register spill is not intrinsic to the GEMM.
Immediate reduction plus constant-index shared-memory staging removes the measured spill completely.
The remaining measured gap is the epilogue's lower issue activity under the stock 128-consumer-thread SM100 schedule.

The combined fast-math and spill-free path remains materially slower at D=4,096 and does not raise issue activity.
Further progress requires a gated custom SM100 epilogue or kernel schedule with more consumer warpgroups rather than more unrolling or Philox batching inside the same four consumer warps.

## Retained experiment surface

The compile-time experiment registry retains the fastest rolled candidate, the spill-free fast-log control, and their combined fast-math spill-free candidate.
Rejected cuRAND, small-epilogue, atomic, full-unroll, chunked, batched-Philox, N64, value-first, and Philox4 implementations were removed after their evidence was recorded above.

Run a retained timing experiment with:

```text
make modal-cutlass GATE=gumbel-experiment CUTLASS_VARIANT=warpgroup-fastmath
make modal-cutlass GATE=gumbel-experiment CUTLASS_VARIANT=warpgroup-fastlog-smem
make modal-cutlass GATE=gumbel-experiment CUTLASS_VARIANT=warpgroup-fastmath-smem
```

Run matched NCU with:

```text
make modal-cutlass GATE=gumbel-experiment-ncu CUTLASS_VARIANT=<variant> CUTLASS_HIDDEN_SIZE=4096
make modal-cutlass GATE=gumbel-experiment-ncu CUTLASS_VARIANT=<variant> CUTLASS_HIDDEN_SIZE=8192
```

The shared timing runner writes raw repetitions, a pandas summary, exact-winner checks, and a decision file under the variant's ignored experiment directory.
The shared NCU runner writes one D-specific log, kernel table, and summary file to the same directory.
