# Current CUTLASS handoff

Read the directory policy and the consolidated Gate 4 recovery finding before continuing the CUTLASS implementation.
This file tracks the mutable handoff and should be updated when the active gate, blocker, or production dispatch changes.

Checkpoint date: 2026-08-05.
Branch: `cutlass-kernel`.
HEAD commit: `8195478`.

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
Production greedy and sampling extensions omit ordinary-GEMM tuning kernels by default, while ordinary-GEMM APIs opt into a separate tuning extension with `FMMS_ENABLE_GEMM_TUNING`.
The controlled sampling build fell from 147.86 seconds to 93.23 seconds while retaining embedded PTX.
Read [the CUTLASS compilation design guide](../../docs/cutlass-compilation.md) before changing CUTLASS sources, includes, template variants, or extension boundaries.

## Current state

Gates 0 and 1 are complete, and the production B200 greedy provider from Gate 2 remains correct and performant.
The working-tree Gate 4 provider now uses immediate per-fragment reduction, direct per-CTA candidate stores, and a shape-dispatched epilogue.
H<=64 uses the accurate single-warpgroup kernel.
H=128 and H=256 use four physical consumer warpgroups with contiguous four-value TMEM loads and no accumulator shared-memory staging.
All other H values retain the generic provider because they have not received matched schedule validation.

The partial-load path is correct.
The epilogue normalizes physical consumer threads with `(threadIdx.x - 128) % ThreadCount`, which accounts for the four producer warps.
Each four-group consumer writes its raw x4 TMEM load into fixed registers 0 through 3.
The callback uses a runtime global fragment base only for output indexing and accesses accumulator registers with compile-time-unrolled local indices.
This removes the local-memory materialization caused by `rAcc(runtime_group * count + i)`.

The selected accurate dispatch passes all nine deterministic production cases and the 10-million-sample large-vocabulary distribution gate.
The distribution result is p=0.903758 with reduced chi-squared 0.994756.
The corresponding `__logf` and `__fdividef` production probe failed at p=0.000415 and is rejected.

The accurate candidates measure 1.12x Triton at D=4,096 and H=64, 1.04x at H=128, and 1.05x at H=256 in the focused interleaved packets.
They measure 0.99x, 0.89x, and 0.78x Triton at D=8,192 for the same H values.
The legacy pointwise 1.20x-greedy production gate still fails because accurate Gumbel generation reaches 1.81x greedy at D=4,096 and H=256.
Gate 4 therefore remains a performance no-go under that declared threshold even though the loading and scheduling defect is fixed.
Gate 5 remains blocked by the Gate 4 feasibility decision.

See the multi-warpgroup epilogue finding for the ownership correction, raw partial loads, fixed-register change, no-staging result, numerical rejection, production validation, and profiler evidence.

## What the profiling established

The complete GEMM accumulator is in Blackwell TMEM in both Triton and CUTLASS.
The liveness problem begins after the epilogue loads an FP32 accumulator fragment from TMEM.

The matched Triton cubin launches 384 threads, uses warp-specialized dynamic register allocation, gives consumer warps up to 176 registers, and reports zero dynamic local loads and stores.
The generic CUTLASS row-reduction callback retains 128 packed value/index candidates per consumer thread, reaches 255 registers, and spills heavily.

The custom immediate reduction removes that persistent row state.
Its direct-store form emits one candidate per physical CTA and output column into a unique slot, uses no global atomics in the GEMM callback, and retains the existing Stage 2 merge.
The selected four-group path partitions the 16-value TMEM fragment into four contiguous x4 loads.
Fixed local register destinations and fixed local callback indices eliminate dynamic local traffic at D=4,096.
Removing the redundant accumulator shared-memory stage raises D=4,096 and H=256 issue activity to 52.82% for the fast-log diagnostic and 56.66% for the accurate kernel.

At D=4,096 and H=256, the accurate kernel reports 72 registers, zero local traffic, and 266.78 million instructions.
Its 527.200-microsecond NCU duration versus Triton's 459.520 microseconds isolates the remaining gap to accurate Gumbel arithmetic.
At D=8,192 and H=256, accurate libdevice arithmetic produces 1,026,048 local loads and 384,768 local stores, but the kernel still measures 525.792 microseconds versus Triton's 639.424 microseconds.
An accurate no-inline probe retained the same local traffic, increased instructions to 259.52 million, and slowed the kernel to 584.096 microseconds, so function outlining is rejected.

See the consolidated Gate 4 performance-recovery finding for the full causal chain and matched counters.

## Latest completed candidate matrix

The table reports CUTLASS latency divided by the interleaved Triton latency on B200.

| Candidate | D | H=64 | H=128 | H=256 | Local-memory result |
| --- | ---: | ---: | ---: | ---: | --- |
| Accurate selected dispatch | 8,192 | 0.99x | 0.89x | 0.78x | D=8,192 accurate path still has libdevice local traffic |
| Accurate selected dispatch | 4,096 | 1.12x | 1.04x | 1.05x | Zero dynamic local loads and stores in the profiled H=256 cell |
| Fast-log selected-shape diagnostic | 8,192 | 0.93x to 0.99x | 0.87x to 0.94x | 0.75x to 0.78x | Rejected by distribution gate |
| Fast-log selected-shape diagnostic | 4,096 | 0.99x to 1.09x | 0.86x to 0.98x | 0.89x to 0.93x | Rejected by distribution gate |

Each completed candidate passed its focused exact-winner check.
The accurate selected dispatch also passed the full production deterministic and distribution suites.

## Production integration state

`src/fused_mm_sampling/cutlass_impl.py::_get_sampling_module()` now selects `warpgroup` for H<=64 and `warpgroup-4wg-partitioned` for H=128 and H=256.
It selects the generic module for unmeasured intermediate H values.
The experimental surface is consolidated into one CPU-importable compile-time registry, one generic sampling API, one timing runner, and one NCU runner.
The root `Makefile` exposes timing, build, and NCU gates parameterized by `CUTLASS_VARIANT`.
The registry retains staged, partitioned, accurate, and fast-log controls needed to reproduce the loading and numerical decisions.
The specialized production SASS audit and Triton compiler-dump runners remain separate because their artifact protocols differ from candidate timing and NCU.

The final accurate production dispatch passed all nine deterministic cases and the 10-million-sample distribution test.
The production performance packet completed all 18 provider-shape pairs but failed the predeclared 1.20x-greedy threshold with a worst ratio of 1.81x.
The local infrastructure suite passes all 34 tests.

## Next steps

1. Keep accurate `logf` and division in the production dispatch unless a replacement passes the same 10-million-sample gate.
2. Target accurate Gumbel instruction cost at D=4,096 and H=256, where the selected kernel has zero local traffic and issue activity already exceeds Triton's.
3. Target the accurate D=8,192 libdevice spill path, which executes 1,026,048 local loads and 384,768 local stores at H=256.
4. Reconfirm any arithmetic change with deterministic, distribution, interleaved timing, and matched NCU evidence.
5. Begin Gate 5 only after Gate 4 passes the declared greedy-performance threshold or the project explicitly changes that criterion.

## Commands and evidence

Run the current production Gate 4 with:

```bash
make modal-cutlass GATE=gumbel-provider
make modal-cutlass GATE=gumbel-ncu
```

Run any registered candidate by setting `CUTLASS_VARIANT`:

```bash
make modal-cutlass GATE=gumbel-experiment CUTLASS_VARIANT=warpgroup-4wg-partitioned
make modal-cutlass GATE=gumbel-experiment-ncu CUTLASS_VARIANT=warpgroup-4wg-partitioned CUTLASS_HIDDEN_SIZE=4096 CUTLASS_N_HIDDEN_STATES=256
```

The shared runner writes current candidate packets under `benchmarking/modal-results/cutlass/experiments/<variant>/`.
The consolidated Gate 4 performance-recovery finding records the historical numbered evidence directories for completed and rejected probes.

See `24-gumbel-max-tp1.md` for the original production Gate 4 evidence.
See the CUTLASS development-infrastructure finding for measured delays and concrete optimization targets.
See the consolidated Gate 4 performance-recovery finding for the experimental sequence and rejection evidence.

## Repository checkpoint

The working tree contains the uncommitted multi-warpgroup ownership and pipeline gates, partial-TMEM-load CUTLASS patch, registered schedule variants, production shape dispatch, infrastructure test update, Makefile routing, the consolidated finding, and this handoff update.
`ws.code-workspace` contains an unrelated user change and must remain untouched.
No checkpoint commit has been created for this gate.
`stash@{0}` and `stash@{1}` are named pre-update safety stashes and are intentionally retained.
Do not pop or drop either stash without first auditing its overlap with the current working tree.

## Production dispatch

The B200 greedy provider emits one packed candidate per physical CTA and uses a cooperative Stage 2 merge.

The working-tree Gumbel provider uses accurate logarithm and division.

- H<=64 uses the single-warpgroup immediate-reduction module.
- H=128 and H=256 use four epilogue warpgroups with contiguous x4 TMEM loads and no accumulator shared-memory staging.
- Other H values use the generic module until they receive matched validation.

The greedy schedule remains:

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
The extension fingerprint follows recursive quoted local includes and also hashes compiler flags, architecture, all applied patches, the pinned CUTLASS revision, and Python, Torch, and CUDA ABI identities.
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
