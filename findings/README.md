# Findings index

The files in this directory record measured behavior, failed experiments, bug investigations, and design decisions.
Read only the findings relevant to the current task.
Put new empirical or historical knowledge here instead of expanding `AGENTS.md`.
Stable operational instructions belong under `docs/` and should link back to the evidence here when useful.

## Sampling correctness and algorithms

- `upcasting-before-softmax.md`
- `fused-top-k-top-p-feasibility.md`
- `argsort-topk-complexity.md`
- `preliminary-topk-top-benchs.md`
- `multinomial-validation-overhead.md`

## Triton and Helion kernels

- `b200-warp-spec-single-tile-crash.md`
- `helion-autotune-per-batch-size.md`
- `helion-barrier-single-kernel.md`
- `helion-hl-rand-specialize-1-bug.md`
- `helion-kernel-slow-at-large-batch-sizes.md`
- `register-spilling-bsz256.md`
- `tma-cache-modifiers.md`
- `tma-store-blackwell-singleton-dims.md`
- `triton-autotune-batch-size-key.md`

## Performance analysis and profiling

- `arithmetic-intensity-decode-matmul.md`
- `gemv-kernel-for-bsz1.md`
- `lm-head-configurations.md`
- `matplotlib-layout-pitfalls.md`
- `ncu-speedup-analysis.md`
- `proton-scopes-persistent-kernel.md`
- `rebuttal-benchmark-summary.md`

## Tensor parallelism and scale-out

- `inter-node-scale-out.md`
- `torch-compile-overhead-tp2.md`
- `tp-scaling-fast-pod-b200.md`
- `tp2-collective-overhead.md`
- `tp2-dispatch-asymmetry.md`
- `tp2-fanout-symm-mem.md`
- `tp2-fullgraph-via-functional-collective.md`

## vLLM

- `modal-vllm-run-anomalies.md`
- `qwen3-8b-tpot-gap-at-high-concurrency.md`
- `vllm-integration.md`

## CUTLASS and CuTe DSL

- `cutlass/README.md` is the ordered CUTLASS index and points to the current handoff.
- `cute-dsl-fmms-kernel.md` covers the separate CuTe DSL investigation.
