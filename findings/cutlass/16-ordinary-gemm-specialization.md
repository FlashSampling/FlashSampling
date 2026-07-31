# Ordinary GEMM specialization prerequisite

The ordinary-GEMM prerequisite now compares CUTLASS and cuBLAS with matched BF16 inputs, outputs, logical dimensions, preallocated buffers, and cold-L2 timing.

Run it with:

```text
make modal-cutlass GATE=ordinary-gemm
```

The evidence packet is under `benchmarking/modal-results/cutlass/13-ordinary-gemm/`.

## Specialization

One tensor-core GEMM schedule is not appropriate for H=1 through H=256.

H=1 and H=2 now use a dedicated BF16 small-N kernel.

One warp owns one vocabulary row, loads each weight once, accumulates one or two outputs in FP32 registers, performs a warp reduction, and stores BF16 outputs.

The inner loop loads two adjacent BF16 values per lane with `__nv_bfloat162`.

H>=4 uses the 128x128x64 tensor-core kernel with CUTLASS automatic mainloop and epilogue schedules.

Both providers use logical unpadded dimensions for H=1 and H=2.

Both providers use N padded to a multiple of eight for H>=4, which gives the CUTLASS BF16 TMA output a 16-byte row stride.

## Result

The specialization validates the need for runtime dispatch.

On H100, the custom H=1 and H=2 path is within the 5% threshold for both primary model shapes.

On B200, H=2 is faster than the measured cuBLAS path for both shapes and H=1 passes for V=151,936 and D=4,096.

B200 H=1 at V=128,256 and D=8,192 remains 10.5% slower than cuBLAS.

Across the complete sweep, H100 passes 7/18 configurations and B200 passes 10/18.

The worst remaining ratio is 1.35 at V=128,256, D=8,192, and H=256 on H100.

The ordinary-GEMM prerequisite therefore remains tuning-required.

## Rejected tile experiments

Changing the tensor-core tile from 128x128x64 to 128x256x64 did not pass the prerequisite.

It improved some H=256 points but remained 15-25% behind cuBLAS there and regressed most smaller-H configurations.

A 256x128x64 tile is not a cross-architecture option.

CUTLASS 4.6.1 rejects it on SM100 because one-CTA BF16 UMMA permits M=64 or M=128, not M=256.

## Next step

Keep the small-N specialization and tune tensor-core schedules separately by architecture and H regime.

The next sweep should preserve every candidate result instead of overwriting one packet and should include tile shapes, cluster shapes, explicit mainloop schedules, explicit epilogue schedules, and stage counts.

Do not transplant a new schedule into the fused epilogue until the ordinary-GEMM prerequisite passes and any changed epilogue visitation has been re-derived through Gate 1a.

## Handoff

This handoff is superseded by `findings/cutlass/17-ordinary-gemm-tuning.md`.

The active implementation is in `src/fused_mm_sampling/csrc/cutlass/greedy_provider.cu`.

`small_n_gemv_kernel<N>` is the diagnostic H=1 and H=2 specialization.

`PlainGemmKernel` is the ordinary tensor-core baseline.

The benchmark and packet writer are in `src/fused_mm_sampling/modal_lib/cutlass/ordinary_gemm.py`.

The gate is registered as `make modal-cutlass GATE=ordinary-gemm`.

The current packet contains only the final selected 128x128x64 hybrid run because the exploratory runner overwrites the same output directory.

The authoritative ratios are in `benchmarking/modal-results/cutlass/13-ordinary-gemm/case-summary.csv`.

The next agent should:

1. Refactor the ordinary baseline into templated kernel variants without changing the validated fused `GemmKernel`.
2. Add a tuning runner that records the variant name and writes all candidates to one DataFrame instead of overwriting earlier runs.
3. Start with architecture-specific legal tile shapes.
4. Keep SM100 M in {64,128} for one-CTA BF16 UMMA.
5. Measure 64x128x64, 128x64x64, and 128x128x64 before expanding the search.
6. Sweep automatic versus explicit architecture-native mainloop and epilogue schedules, then stage counts and legal cluster shapes.
7. Select candidates per architecture and H regime using the predeclared `CUTLASS/cuBLAS <= 1.05` requirement.
8. Retain the custom H=1 and H=2 path unless a measured tensor-core variant is faster.
9. Investigate B200 V=128,256, D=8,192, H=1 separately because it is the only failing small-N case in the selected run.
10. Investigate H=256 separately on both architectures because it contains the largest remaining ratios.
11. Rerun the complete ordinary-GEMM gate after implementing runtime dispatch.
12. Only after the prerequisite passes, move approved schedules into the fused candidate epilogue and rerun Gate 1a plus every dependent correctness gate before Gate 2b.

Do not begin Philox Gate 3 while this prerequisite remains tuning-required.
