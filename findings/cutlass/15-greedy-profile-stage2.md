# Greedy profiling and parallel Stage 2

The first Gate 2b implementation spent most of its avoidable latency in Stage 2.

The original Stage 2 kernel launched one CTA for every 256 hidden states.
Each active thread owned one hidden state and serially scanned all vocabulary-tile candidates.
At V=151,936, each active thread executed a dependency chain over 1,187 candidates.

## Evidence

Run the component timing packet with:

```text
make modal-cutlass GATE=greedy-profile
```

Run the targeted Nsight Compute packet with:

```text
make modal-cutlass GATE=greedy-ncu
```

The runners use the existing FlashInfer CUPTI benchmark helper and standard Modal Nsight Compute image.
The NCU target places separate NVTX ranges around the fused GEMM, Stage 2, and matching plain CUTLASS GEMM.

Before the fix, Stage 2 took 0.092 to 0.144 ms across the representative H=1 and H=128 cases.
NCU measured 0.100 to 0.137 ms for the kernel while it read only 80 KB at H=1 or approximately 1.22 MB at H=128.
The combination of low traffic and nearly constant duration across H identified the serial per-column reduction as the cause.

The fused GEMM also had a smaller low-H gap against the matching plain GEMM.
At V=151,936, D=4,096, and H=1, NCU measured 0.620 ms versus 0.543 ms on H100 and 0.307 ms versus 0.272 ms on B200.
The fused kernel used 187 versus 154 registers per thread on H100 and 255 versus 81 on B200.
The H100 fused mainloop used five stages while the plain GEMM used six.
These measurements identify epilogue resource use and mainloop depth as the next profiling targets, but they do not by themselves prove which difference causes the remaining latency.

## Fix

Stage 2 now launches one CTA per hidden state.
Its 256 threads scan the vocabulary tiles cooperatively and merge their local winners with a deterministic shared-memory tree reduction.
The reduction continues to use the exact packed FP32-value and int32-index comparator from Gate 1.

The complete Gate 2a matrix passed again on H100 and B200.
All 52 deterministic boundary, tie, and primary-shape cases passed, together with the shared CUTLASS pytest cases on each architecture.

After the fix, CUPTI measured Stage 2 at 0.004 to 0.006 ms.
This is a 20x to 31x reduction and leaves Stage 2 at 0.4% to 2.4% of the preallocated pipeline.

## Updated Gate 2b decision

The full H100 and B200 sweep improved from 12/36 to 29/36 passing configurations under the predeclared 1.05 ratio.
H100 now passes 15/18 configurations and B200 passes 14/18.
The worst ratio improved from 1.38 to 1.14.

The remaining failures are H=1,2,4 for V=151,936 and D=4,096 on H100, H=1,2 for that shape on B200, and H=1,2 for V=128,256 and D=8,192 on B200.
Gate 2b therefore remains no-go.

The next optimization must target the low-H fused GEMM epilogue and its effect on register use and mainloop staging.
Do not spend more work on Stage 2 unless a later profile shows a regression.

## Revised feasibility requirement

### Immediate next action

Implement one new diagnostic gate that compares ordinary CUTLASS GEMM directly with cuBLAS GEMM using identical BF16 inputs, BF16 output, logical dimensions, padding policy, and cold-L2 timing across both primary shapes and H=1 through H=256.
Do not modify the fused epilogue before this diagnostic exists and its results are understood.

The schedule-matched plain CUTLASS GEMM is not sufficiently competitive with the current cuBLAS-plus-argmax path.
CUTLASS GEMM plus argmax is slower than cuBLAS plus argmax in all 36 measured configurations.
Its median latency ratio is 1.17 on H100 and 1.27 on B200.
The worst ratios are 2.05 on H100 and 2.58 on B200.

This comparison is not yet a clean ordinary-GEMM comparison.
The diagnostic CUTLASS GEMM stores FP32 logits, while the PyTorch cuBLAS path stores BF16 logits, and the CUTLASS schedule was selected to match the FMMS kernel rather than tuned independently.
The result therefore does not prove that CUTLASS cannot match cuBLAS.
It does prove that the current CUTLASS schedule cannot serve as the foundation for a useful FMMS port.

Pause epilogue register-pressure work until an ordinary-GEMM prerequisite is complete.
Use this sequence:

1. Add dtype-matched ordinary CUTLASS and cuBLAS GEMM baselines with identical logical M, N, K, layouts, output dtype, padding policy, and cold-L2 timing.
2. Tune CUTLASS tile shape, cluster shape, mainloop schedule, epilogue schedule, and small-N specialization across both primary shapes and H=1 through H=256.
3. Require the tuned ordinary CUTLASS GEMM to remain within 5% of cuBLAS for every supported configuration.
4. Stop the CUTLASS port if that prerequisite cannot pass.
5. Only after it passes, transplant the FMMS candidate epilogue onto the approved schedules.
6. Treat register reduction as a justified experiment without claiming that spilling is the established cause.
7. If the epilogue tile or visitation changes, rerun Gate 1a and derive new ownership formulas before rerunning the dependent Gate 1b through Gate 1g tests.
8. Collect registers, local-memory traffic, mainloop stages, and component latency through `greedy-ncu`.
9. Adopt the epilogue variant only after Gate 2a passes, then rerun Gate 2b against both the approved CUTLASS baseline and the production cuBLAS baseline.

The current seven failures are the decision points to optimize first.
They are H=1,2,4 for V=151,936 and D=4,096 on H100, H=1,2 for that shape on B200, and H=1,2 for V=128,256 and D=8,192 on B200.
Do not begin Philox Gate 3 until every configuration meets the existing 1.05 threshold.
