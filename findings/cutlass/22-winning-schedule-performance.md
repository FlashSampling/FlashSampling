# Gate 2d winning-schedule fused performance

The first Gate 2d performance sweep measures the complete production donor
dispatch against the matching plain CUTLASS schedules, `torch.mm` plus argmax,
and Triton FMMS.

The Gate 2b harness now dispatches the plain B200 comparison to the same Gate
2c family as the fused kernel for every primary model-shape cell.
Run B200-only follow-ups with:

```text
make modal-cutlass GATE=greedy-performance MODAL_ARGS="--b200-only"
```

## Result

The original packed-CAS final reduction made H=128 and H=256 substantially
slower than the matching plain GEMMs.
Nsight Compute attributed the H=256 regression to both contended atomics and
register spilling: the fused kernels used 255 registers per thread, issued
12.4--14.7 million local-load sectors and 16.9--20.0 million local-store
sectors, and wrote 172--175 MB to DRAM.
The matching plain kernels used 77 registers or fewer and issued no local
memory traffic.

The production replacement emits one packed candidate per physical CTA and
uses a separate cooperative Stage 2 merge.
On Blackwell 2-SM schedules, a logical 256-row M tile consists of two CTAs that
each own 128 vocabulary rows, so candidate addressing advances in 128-row
units.
The focused Gate 2d harness passed 8,612 exact intermediate and final candidate
comparisons across all five donor schedules, plus memcheck and racecheck.
The production B200 provider suite also passed.

H=256 now reuses 256x128 donors over two N tiles instead of using the 256x256
donor.
D=4,096 uses K64 cluster-4, while D=8,192 uses K64 cluster-2.
At V=151,936, D=4,096, H=256, the final full sweep measures 0.327 ms and is
18.2% faster than Triton and 3.2% faster than `torch.mm` plus argmax.
At V=128,256, D=8,192, H=256, it measures 0.484 ms and is 42.3% faster than
Triton and 4.2% faster than `torch.mm` plus argmax.
The H=128 path is 4.2--7.7% faster than Triton in the same B200 run.
All 90 correctness rows pass, and the predeclared fused-to-CUTLASS-GEMM-plus-
argmax ratio passes every cell with a worst value of 1.043.

These measurements are in
`benchmarking/modal-results/cutlass/10-greedy-performance/`.
The full invocation also repeated H100, but Gate 2d changes only B200.
Subsequent Gate 2d timing and profiling should use the B200-only runner.

## Reduction experiments

Native packed `atomicMax` is not a substitute for the guarded CAS loop here.
The `atomicMax` experiment passed exact correctness but made H=256 latency
roughly 1.00--1.04 ms.
Every losing candidate performs an atomic read-modify-write with `atomicMax`,
while the CAS implementation first performs an atomic load and returns without
an update for most losers.
Under roughly one thousand competing M tiles, the latter behavior measured
better despite its retry loop.

An attempted replacement used CUTLASS `Sm90RowReduction` with
`FinalReduction=true` and a non-atomic `CandidateReduce`.
The first B200 boundary case returned indices `0;0` instead of `13;10`.
The stock final-reduction path is therefore not correctness-approved for these
2-SM schedules and was removed from production.

## Current Nsight Compute result

The 128-column H=256 path has zero atomic sectors and reduces spilling by
15--27x relative to the atomic 256-column path.
At D=4,096 it issues 0.95 million local-load sectors and 1.06 million
local-store sectors, writes 16.2 MB, and reaches 69.3% tensor-pipe utilization.
At D=8,192 it issues 0.77 million local-load sectors and 0.74 million
local-store sectors, writes 20.7 MB, and reaches 76.6% tensor-pipe utilization
with the final K64 cluster-2 donor.
Both kernels still allocate 255 registers per thread, so spilling is reduced
but not eliminated.

An existing 128x64 donor was tested at H=256 to reduce callback state further.
It regressed to 0.409 ms at D=4,096 and 0.716 ms at D=8,192, 6.5--8.1% slower
than Triton, so it was rejected without another NCU run.
The 256x128 K64 donors are the current measured winners.
At D=8,192, changing from cluster-2 K128 to cluster-4 K64 improved timing to
0.550 ms but still missed the production-baseline threshold.
Holding K64 fixed and changing cluster-4 to cluster-2 improved it to 0.484 ms.
NCU shows unchanged local spill traffic but tensor-pipe utilization increasing
from 58.8% to 76.6%, so the cluster-2 improvement is execution efficiency rather
than spill elimination.

## Profiling and workflow

`greedy-ncu` now profiles only the changed H=256 fused and matching-plain GEMMs
for the two primary model shapes.
It filters out allocation, initialization, and Stage 2 kernels, reducing the
packet from eight NCU invocations to four.
Previous H=128 profiles remain the control until that dispatch changes.

CUTLASS provider sources, tests, and the NCU target are runtime mounts rather
than image copies.
Source edits no longer invalidate the dependency, NCU, or CUTLASS image layers.
After one cache migration, a clean NCU launch performed zero image builds and
went directly from mount creation to profiling.
