# Current CUTLASS handoff

Read the directory policy and the consolidated Gate 4 recovery finding before continuing the CUTLASS implementation.
This file tracks the mutable handoff and should be updated when the active gate, blocker, or production dispatch changes.

Checkpoint date: 2026-08-04.
Branch: `cutlass-kernel`.
Rebased commit: `06c69f8`.

## Required experiment driver

Use specialized ownership, layout, compilation, and sanitizer gates while a candidate has not reached shared timing or profiling.
Once a candidate is registered as `CUTLASS_VARIANT`, run its timing packet through:

```bash
make infra-sync
make check-dev-env
make modal-cutlass-experiment \
    CUTLASS_VARIANT=<variant> \
    CUTLASS_DEV_LABEL=<short-description>
```

The driver builds or loads the selected extension through one cache writer, commits it, then runs timing against the published cache entry.
Profiling is never implicit.
Pass only the configurations needed for the current question, for example `CUTLASS_PROFILE_CONFIGS='[{"hidden_size":4096,"n_hidden_states":128}]'`.
The available menu contains `hidden_size` values 4,096 and 8,192 crossed with `n_hidden_states` values 128 and 256.
See the driver documentation for the complete JSON menu.
Each selected configuration produces a candidate report and a matched Triton report after timing succeeds.
Do not launch timing and NCU independently against a cold fingerprint because that can compile the same extension more than once.
A failed build prevents timing and profiling from launching, and a failed timing packet prevents profiling from launching.
Raw `.ncu-rep` files are required evidence and are retained on the shared Volume rather than deleted after CSV export.
The local NCU summaries record their exact paths.
See [the CUTLASS experiment development driver](../../docs/modal-benchmarking.md#cutlass-experiment-development-driver) for direct debugging gates, workflow records, artifact locations, and report-download commands.
Use [the CUTLASS compile-time study driver](../../docs/modal-benchmarking.md#cutlass-compile-time-study-driver) for compiler baseline, split-compilation, and Compile Time Advisor work.

## Current state

Gates 0 and 1 are complete, and the production B200 greedy provider from Gate 2 remains correct and performant.
The original Gate 4 Gumbel-Max provider passed deterministic and distribution checks, but its high-H performance remains a no-go.
The register-liveness investigation has now produced custom experimental paths that remove the generic EVT's persistent row state and global atomics.
None of those paths has been promoted into the production `fused-cutlass` provider.

The best completed timing candidate is `warpgroup-fastmath`.
It uses the rolled immediate reduction, direct per-CTA candidate stores, `__logf`, and `__fdividef`.
It matches Triton at H=64 and reaches 1.11x and 1.21x Triton at D=8,192 for H=128 and H=256, but it remains 1.73x and 1.89x Triton at D=4,096.
It still executes the dynamic local-memory path associated with rolled `accumulators[i]` access.

The best completed spill-free candidate is `warpgroup-fastlog-smem`.
It stages each constant-index accumulator fragment through shared memory, allocates 55 to 59 registers in the matched profiles, and reports zero dynamic local loads and stores.
It remains 1.22x and 1.32x Triton at D=8,192 and 1.89x and 2.11x Triton at D=4,096 for H=128 and H=256.
This proves that the CUTLASS spill can be removed, but also proves that spill removal alone does not recover Triton's latency hiding.

The latest `warpgroup-fastmath-smem` candidate combines that zero-local-memory path with fast division.
It reports zero dynamic local loads and stores, allocates 54 to 58 registers, and measures 1.12x and 1.20x Triton at D=8,192 and 1.63x and 1.85x at D=4,096 for H=128 and H=256.
Fast division removes 4.47% to 4.73% of executed instructions relative to the spill-free fast-log control, but issue activity remains 18.52% to 20.69%.
The result leaves the stock epilogue schedule, rather than arithmetic instruction count or register spilling, as the next bounded target.

The four-output Philox candidate is rejected.
It passed its focused exact-winner check under the new four-stream mapping, but measured 1.79x and 2.06x Triton at D=8,192 and 2.95x and 3.08x Triton at D=4,096 for H=128 and H=256.
It reports zero dynamic local loads and stores, allocates 74 registers at D=4,096 and 86 registers at D=8,192, and reaches only 7.32% to 11.42% issue activity.

Gate 4 therefore remains experimental and Gate 5 remains blocked.
The current production provider must not be described as containing the custom immediate-reduction, fast-math, or Philox4 paths.

## What the profiling established

The complete GEMM accumulator is in Blackwell TMEM in both Triton and CUTLASS.
The liveness problem begins after the epilogue loads an FP32 accumulator fragment from TMEM.

The matched Triton cubin launches 384 threads, uses warp-specialized dynamic register allocation, gives consumer warps up to 176 registers, and reports zero dynamic local loads and stores.
The generic CUTLASS row-reduction callback retains 128 packed value/index candidates per consumer thread, reaches 255 registers, and spills heavily.

The custom immediate reduction removes that persistent row state.
Its direct-store form emits one candidate per physical CTA and output column into a unique slot, uses no global atomics in the GEMM callback, and retains the existing Stage 2 merge.
The remaining rolled fragment access produces exactly one warp-level local load per output element.

Constant-index shared-memory staging removes all measured local traffic, but the stock SM100 TMA epilogue still supplies only 128 consumer threads.
Together with four infrastructure warps, the complete CUTLASS kernel launches 256 threads, or eight warps, compared with Triton's 384 threads, or twelve warps.
The zero-local CUTLASS profiles show materially lower issue activity than Triton, so the remaining gap is associated with insufficient latency hiding in this epilogue schedule rather than an unavoidable CUTLASS GEMM limitation.

See the consolidated Gate 4 performance-recovery finding for the full causal chain and matched counters.

## Latest completed candidate matrix

The table reports CUTLASS latency divided by the interleaved Triton latency on B200.

| Candidate | D | H=64 | H=128 | H=256 | Local-memory result |
| --- | ---: | ---: | ---: | ---: | --- |
| `warpgroup-fastmath-smem` | 8,192 | 0.99x | 1.12x | 1.20x | Zero dynamic local loads and stores |
| `warpgroup-fastmath-smem` | 4,096 | 1.03x | 1.63x | 1.85x | Zero dynamic local loads and stores |
| `warpgroup-fastmath` | 8,192 | 0.99x | 1.11x | 1.21x | Dynamic local loads and stores remain |
| `warpgroup-fastmath` | 4,096 | 0.98x | 1.73x | 1.89x | Dynamic local loads and stores remain |
| `warpgroup-fastlog-smem` | 8,192 | 1.02x | 1.22x | 1.32x | Zero dynamic local loads and stores |
| `warpgroup-fastlog-smem` | 4,096 | 1.07x | 1.89x | 2.11x | Zero dynamic local loads and stores |
| `warpgroup-n64` | 8,192 | 0.99x | 1.15x | 1.26x | Dynamic local loads and stores remain |
| `warpgroup-n64` | 4,096 | 1.02x | 1.76x | 2.02x | Dynamic local loads and stores remain |
| `warpgroup-philox4` | 8,192 | 0.99x | 1.79x | 2.06x | Zero dynamic local loads and stores |
| `warpgroup-philox4` | 4,096 | 0.99x | 2.95x | 3.08x | Zero dynamic local loads and stores |

Each completed candidate passed its focused exact-winner check.
These checks are narrower than the full production Gate 4 deterministic and distribution suites.

## Production integration state

`src/fused_mm_sampling/cutlass_impl.py::_get_sampling_module()` still builds the earlier generic Gate 4 implementation.
The experimental surface is consolidated into one CPU-importable compile-time registry, one generic sampling API, one timing runner, and one NCU runner.
The root `Makefile` exposes one timing gate and one NCU gate parameterized by `CUTLASS_VARIANT`.
Only `warpgroup-fastmath`, `warpgroup-fastlog-smem`, and their combined `warpgroup-fastmath-smem` candidate remain registered.
The rejected one-off APIs, loaders, Modal wrappers, Make targets, C++ branches, and findings were removed after their evidence was consolidated.
The specialized production SASS audit and Triton compiler-dump runners remain separate because their artifact protocols differ from candidate timing and NCU.

The complete Gate 4 production matrix, the large-vocabulary 10-million-sample distribution test, and the production performance gate have not been rerun with any custom candidate.

The consolidated runners passed their initial B200 validation on both retained controls and then ran the combined candidate without adding another runner or Make gate.
Each variant compiled from the shared runner, passed both focused exact-winner cases, and completed all six interleaved timing cells in its own result directory.
The local infrastructure suite passed all 21 tests, and registry validation rejected an unknown variant before any Modal allocation.
This validates the consolidated experiment plumbing without promoting either candidate into production.

## Next steps

1. Start the custom SM100 epilogue-schedule work with an ownership-layout gate for more consumer warpgroups.
   The combined fast-math spill-free result shows that further arithmetic changes inside the same four consumer warps are unlikely to close the D=4,096 gap.
2. Add a pipeline and barrier correctness gate before integrating the Gumbel callback into that schedule.
   Do not change the stock `ThreadCount=128` constant without independently validating ownership and synchronization.
3. Run the resulting candidate two or three times with CUTLASS and Triton interleaved in the same remote function.
   Require cross-run agreement before applying the pointwise 1.20x performance gate.
4. Promote one selected dispatch into `_get_sampling_module()` only after it passes the timing gate.
   Do not leave production behavior dependent on an experimental Make target.
5. Rerun the full deterministic, distribution, B200 correctness, and performance suites after promotion.
   Fast logarithm or fast division requires the 10-million-sample large-vocabulary distribution check, not only exact winner checks against a matching reference.
6. Begin Gate 5 only after Gate 4 is promoted or the project explicitly changes scope to a greedy-only CUTLASS provider.

## Commands and evidence

Run the current production Gate 4 with:

```bash
make modal-cutlass GATE=gumbel-provider
make modal-cutlass GATE=gumbel-ncu
```

Run any registered candidate by setting `CUTLASS_VARIANT`:

```bash
make modal-cutlass GATE=gumbel-experiment CUTLASS_VARIANT=warpgroup-fastmath-smem
make modal-cutlass GATE=gumbel-experiment-ncu CUTLASS_VARIANT=warpgroup-fastmath-smem CUTLASS_HIDDEN_SIZE=4096 CUTLASS_N_HIDDEN_STATES=128
```

The shared runner writes current candidate packets under `benchmarking/modal-results/cutlass/experiments/<variant>/`.
The consolidated Gate 4 performance-recovery finding records the historical numbered evidence directories for completed and rejected probes.

See `24-gumbel-max-tp1.md` for the original production Gate 4 evidence.
See the CUTLASS development-infrastructure finding for measured delays and concrete optimization targets.
See the consolidated Gate 4 performance-recovery finding for the experimental sequence and rejection evidence.

## Repository checkpoint

The working tree contains staged and unstaged CUTLASS implementation, experiment-runner, Makefile, and findings changes.
No checkpoint commit has been created.
`stash@{0}` is the rebase autostash and is intentionally retained.
Do not pop or drop it without first auditing its overlap with the current working tree.

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
The extension fingerprint follows recursive quoted local includes and also hashes compiler flags, architecture, both applied patches, the pinned CUTLASS revision, and Python, Torch, and CUDA ABI identities.
Standalone correctness harnesses are excluded so their edits do not invalidate production extensions.
Keep architecture and feature variants under distinct prefixes, and do not launch concurrent first builds of the same content-keyed extension.
Use `make cutlass-dev-metrics` to summarize end-to-end gate duration, startup, extension load time, cache state, failures, and residual time.

The shared correctness-gate helpers live in `src/fused_mm_sampling/modal_lib/cutlass/gate_common.py` and the max-harness CUDA helper lives in `src/fused_mm_sampling/csrc/cutlass/max_harness.h`.
Put new common sanitizer, CSV, pass-detection, packet, and CUDA-check logic in those helpers instead of copying it into another gate.
Production-driven runners such as the small-N GEMV intentionally keep their own orchestration.
Use `CUTLASS_RESULT_POSTFIX`, exposed as Make's `POSTFIX`, to direct a gate to a separate evidence directory.

## External context

[NVIDIA CUTLASS PR #3426](https://github.com/NVIDIA/cutlass/pull/3426) remains open as of 2026-08-04 and introduces a separate public `cutlass_compiler/` MLIR stack.
Its initial version did not modify or integrate the existing CuTe Python DSL or CUTLASS C++ template frontend.
The preliminary ACM Europe MLIR School 2026 program places compiler fundamentals and the MLIR IR model on Day 1, followed by ODS and transformations on Day 2.
For PR #3426, learn operations, regions, blocks, SSA, and dialects first, then map its `.td` definitions and lowering passes onto the transformation material.
