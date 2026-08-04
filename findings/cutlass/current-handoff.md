# Current CUTLASS handoff

Read `README.md` and the numbered findings it references before continuing the CUTLASS implementation.
This file tracks the mutable handoff and should be updated when the active gate, blocker, or production dispatch changes.

## Current state

Gate 4 Gumbel-Max correctness and distribution pass on B200, but performance remains a no-go.
The selected per-family partial-unroll control measures 1.06x, 1.52x, and 2.30x at D=8,192 and 1.22x, 2.41x, and 3.39x at D=4,096 for H=64, 128, and 256.
NCU still reports 255 registers and substantial local-memory traffic at H>=128.
The stock no-shared-memory epilogue path is closed because it rejects `ElementD=void`.

The bounded Gate 4 recovery plan is to audit SASS spill ownership, compare one smaller-granularity custom SM100 epilogue, and compare one maintained vendor-backed Philox candidate.
Gate 5 stays blocked unless a candidate removes RNG-caused spill and meets the pointwise 1.20x limit, or the project explicitly changes scope to a greedy-only provider.

Run Gate 4 with:

```bash
make modal-cutlass GATE=gumbel-provider
make modal-cutlass GATE=gumbel-ncu
```

The provider gate has `deterministic`, `distribution`, and `performance` phases.
See `24-gumbel-max-tp1.md` for the evidence and rejected paths.

## Production dispatch

The B200 greedy provider emits one packed candidate per physical CTA and uses a cooperative Stage 2 merge.

- H<=64 uses `128x64x128` with cluster `(2,1,1)`.
- H=128 uses K64 with cluster `(4,1,1)` at D=4,096 and K128 with cluster `(2,1,1)` at D=8,192.
- H=256 uses two N tiles and K64 with cluster `(4,1,1)` at D=4,096 and `(2,1,1)` at D=8,192.

The focused gate passed 8,612 exact intermediate and final comparisons plus memcheck and racecheck.
The production B200 correctness suite also passed.
See `20-winning-schedule-accumulator-layout.md`, `21-winning-schedule-evt.md`, and `22-winning-schedule-performance.md` for layout, reduction, and performance details.

## Gate 5 TP experiment

Gate 5a must compare per-tile symmetric-memory fan-out with a locally atomic-reduced packed-MAX path.
The packed path communicates one 64-bit candidate per hidden state and rank but cannot overlap communication before the local GEMM finishes.
Raw FP32 bits and indices are not directly MAX-sortable.
Use an order-preserving FP32 transform, invert the global index for lower-index tie-breaking, and validate signed collective semantics before an integer-MAX all-reduce.
Compare total paired timings because launch latency may dominate the `8H`-byte payload.

## Development workflow

Nsight Compute is a continuous CUTLASS development tool.
After material schedule, epilogue, reduction, or memory-path changes, use matched timings to select representative fast and slow cells and profile the exact production kernels before making causal claims.
Refresh stale profiles instead of applying results from an older kernel.

Run all Modal gates through `make modal-cutlass GATE=<gate>`.
The top-level recipe propagates pipe failures so `tee` cannot hide a failed Modal run.
During prior TLS and heartbeat failures, `uv tool run --from modal --with pydantic-settings modal run --detach ...` was more reliable than the system Modal client.

CUTLASS provider sources, tests, and the NCU target are runtime mounts.
Source changes should not rebuild dependency, NCU, or CUTLASS image layers.
After changing image composition, expect one cache migration and then verify that the next startup uses mounts without image builds.
Stop stale interrupted Modal apps before relaunching to avoid duplicate builds, profiling, or log writers.

CUTLASS PyTorch JIT extensions must mount the shared `fused-mm-sample` volume and call `set_volume_caches()`.
This places `TORCH_EXTENSIONS_DIR` on the shared cache so workers reuse multi-minute SM90 and SM100 builds.
`_extension_name()` hashes all local CUTLASS `.cu`, `.cuh`, and `.patch` inputs so header changes cannot reuse a stale shared object.
Keep architecture and feature variants under distinct prefixes, and do not launch concurrent first builds of the same content-keyed extension.

The shared correctness-gate helpers live in `src/fused_mm_sampling/modal_lib/cutlass/gate_common.py` and the max-harness CUDA helper lives in `src/fused_mm_sampling/csrc/cutlass/max_harness.h`.
Put new common sanitizer, CSV, pass-detection, packet, and CUDA-check logic in those helpers instead of copying it into another gate.
Production-driven runners such as the small-N GEMV intentionally keep their own orchestration.
Use `CUTLASS_RESULT_POSTFIX`, exposed as Make's `POSTFIX`, to direct a gate to a separate evidence directory.

## External context

[NVIDIA CUTLASS PR #3426](https://github.com/NVIDIA/cutlass/pull/3426) remains open as of 2026-08-04 and introduces a separate public `cutlass_compiler/` MLIR stack.
Its initial version did not modify or integrate the existing CuTe Python DSL or CUTLASS C++ template frontend.
The preliminary ACM Europe MLIR School 2026 program places compiler fundamentals and the MLIR IR model on Day 1, followed by ODS and transformations on Day 2.
For PR #3426, learn operations, regions, blocks, SSA, and dialects first, then map its `.td` definitions and lowering passes onto the transformation material.
