# Ordinary GEMM manual stage-count no-go

The bounded explicit-stage search did not make the ordinary CUTLASS prerequisite competitive with `torch.mm`.

This finding closes only the manually selected `1x1x1` cluster stage-count
family, whose explicit Blackwell native schedule was 1-SM and which did not
audit named 2-SM coverage.

It does not close the CUTLASS FMMS port or establish that CUTLASS cannot match
the `torch.mm` baseline.

The revised plan reopens plain-GEMM discovery with NVIDIA Matmul Heuristics
and `cutlass_profiler` in Gate 2c of
`findings/cutlass/01-fmms-kernel-plan.md`.

## Screen

The screen retained all six existing tile and schedule controls.

It added explicit stage counts for 128x64x64 and 128x128x64 tiles with both automatic and architecture-native schedules.

H100 tested stages 7, 8, and 9 for 128x64x64 and stages 5, 6, and 7 for 128x128x64.

B200 tested stages 6, 7, and 8 for 128x64x64 and stages 4, 5, and 6 for 128x128x64.

The screen covered both primary model shapes at H=128 and H=256 with 25 warmups and 30 measured cold-L2 repetitions.

All 144 explicit-stage correctness cases matched the corresponding `torch.mm` BF16 output bit-for-bit and were finite.

## Result

Per-case selection passed four of eight screened configurations.

The four H=256 configurations remained outside the 1.05 threshold.

| Architecture | V | D | H | Best selected variant | CUTLASS/torch.mm |
|---|---:|---:|---:|---|---:|
| B200 | 128,256 | 8,192 | 256 | 128x128x64 auto, 6 stages | 1.17 |
| B200 | 151,936 | 4,096 | 256 | 128x128x64 auto, 6 stages | 1.24 |
| H100 | 128,256 | 8,192 | 256 | 128x128x64 auto control | 1.35 |
| H100 | 151,936 | 4,096 | 256 | 128x128x64 auto control | 1.38 |

No stage candidate met the promotion rule.

On H100, the largest improvement over the matching best control was 1.04%, and that candidate regressed another screened configuration by 37.35%.

On B200, the best explicit-stage result improved its matching control by 0.36% in one configuration and regressed another by 3.15%.

An earlier automatic-schedule-only packet showed a 3.47% B200 improvement paired with a 3.31% regression.

The complete automatic-plus-native packet did not reproduce that improvement, so it does not justify promotion.

## Decision

The original experiment made cluster tuning conditional on an explicit-stage candidate either passing the 1.05 threshold or improving a failing control by at least 3% without regressing another screened configuration by more than 3%.

No candidate met that condition, so no cluster search was launched.

That condition was too restrictive because cluster multicast changes hidden-state reuse independently of pipeline stage count.

The ordinary-GEMM prerequisite remains tuning-required, and this bounded manual search reached its predeclared stop condition.

The rejected explicit-stage instantiations and dispatch branches were removed from the compiled extension.

The canonical retained-candidate runner again contains only the six controls.

The ignored evidence packets are under `benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning/stage-screen/` and `benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning/stage-screen-complete/`.

The CUTLASS FMMS path remains a correctness-validated experimental provider.

Further sampling features remain paused until the official heuristic and profiler search in Gate 2c either finds a competitive plain kernel or reaches its stronger audited stop condition.

Gate 2c now targets B200 only.
Hopper work resumes after the complete B200 CUTLASS provider beats Triton
FlashSampling pointwise across its declared kernel and end-to-end matrix.
