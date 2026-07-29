# Gate 2b greedy performance feasibility

Gate 2b initially reached a no-go decision under the predeclared 5% threshold.

The benchmark compares the correctness-approved no-D CUTLASS FMMS provider with
plain CUTLASS GEMM, CUTLASS GEMM plus argmax, Triton FMMS, and cuBLAS plus
argmax.
It covers V=151,936 with D=4,096 and V=128,256 with D=8,192 at H=1,2,4,...,256
on H100 and B200.
Each point contains 25 warmup iterations and 100 cold-L2 CUDA-event
repetitions.

## Decision

The declared metric is:

```text
median(CUTLASS FMMS) / median(CUTLASS GEMM plus argmax) <= 1.05
```

Only 12 of 36 configurations passed.
H100 passed 6 of 18 configurations, with a median ratio of 1.14 and a worst
ratio of 1.33.
B200 passed 6 of 18 configurations, with a median ratio of 1.20 and a worst
ratio of 1.38.
The worst case was V=151,936, D=4,096, H=1 on B200: CUTLASS FMMS measured
0.386 ms versus 0.280 ms for CUTLASS GEMM plus argmax.

CUTLASS FMMS was slower than Triton FMMS in 35 of 36 configurations and slower
than cuBLAS plus argmax in all 36 configurations.
These comparisons include each provider's complete current wrapper path,
including allocations and input padding.

The project must not begin Gate 3 stateless Philox work on this implementation.
The epilogue or wrapper path must first be reworked and Gate 2b rerun, or the
CUTLASS port should stop.

## Profiling follow-up

Targeted component and Nsight Compute profiling found that the original Stage
2 kernel serially scanned all vocabulary-tile candidates in one active thread
per hidden state.
The parallel Stage 2 rewrite reduced that component from 0.092-0.144 ms to
0.004-0.006 ms and passed the complete Gate 2a matrix.

The updated Gate 2b sweep passes 29 of 36 configurations.
H100 passes 15/18 and B200 passes 14/18.
The worst ratio is now 1.14.
Gate 2b remains no-go because seven low-H configurations still exceed 1.05.
The profile, optimization, and current next step are documented in
`findings/cutlass/15-greedy-profile-stage2.md`.

## Static resources

The fused GEMM uses 187 registers per thread on H100 and 255 registers per
thread plus 64 bytes of static local memory per thread on B200.
The CUDA occupancy API reports one active fused-GEMM block per SM on both
architectures.
Stage 2 uses 28 registers per thread on H100 and 32 on B200, with eight active
blocks per SM on both.

These attributes identify useful profiling targets but do not establish why
the low-H cases are slow.
Causal claims require component timing and measured hardware counters.

## Correctness handling

Gate 2a remains the exact deterministic correctness authority.
Random dense inputs can select different near-tie winners across CUTLASS
epilogues, Triton, and cuBLAS because their floating-point accumulation
schedules differ.
Gate 2b therefore verifies output shape, dtype, and index range and records
cross-baseline agreement diagnostically.
It does not reinterpret random cross-library disagreement as a correctness
failure.

## Reproduction and evidence

Run:

```text
make modal-cutlass GATE=greedy-performance
```

The runner launches the H100 and B200 jobs concurrently.
Each GPU executes providers and shapes sequentially.
The packet is under
`benchmarking/modal-results/cutlass/10-greedy-performance/` and contains
`VERIFY.md`, `summary.json`, `case-summary.csv`, `cases.csv`,
`correctness.csv`, `kernel-attributes.csv`, and `log.txt`.

The first run intentionally stopped when random cross-library argmax equality
was treated as a requirement.
The corrected run completed the full matrix and retained the no-go evidence.

## Follow-up

Before attempting an epilogue redesign, profile representative failing and
passing points to separate the fused GEMM, Stage 2, padding, allocation, and
wrapper costs.
Use NCU to confirm that the void-D path performs no D writes and to collect
measured occupancy and local-memory traffic.
This profiling is diagnostic follow-up for a rework decision.
It cannot convert the present measurements into a Gate 2b pass.
