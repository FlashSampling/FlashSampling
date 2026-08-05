# CUTLASS development infrastructure roadmap

This roadmap improves the development loop around the CUTLASS FMMS kernel without changing kernel algorithms.
The work is deliberately staged so each infrastructure change must demonstrate value before the next one begins.

## Measured starting point

The production PyTorch extensions currently derive their names by hashing every `.cu`, `.cuh`, and `.patch` file under `csrc/cutlass`.
That directory contains 14 such files at the start of this work.
The production translation units are `greedy_provider.cu` and `winning_schedule_provider.cu`.
Their quoted local include closure contains `evt_candidates.cu` and `stateless_philox.cuh`, plus the two patches applied to the pinned CUTLASS checkout.
Standalone accumulator-layout and max-reduction harnesses are not production extension inputs, but edits to them currently produce a new extension name.

The gate runner records a human-readable log but no structured development-loop timings.
It is therefore difficult to separate Modal startup, extension compilation, benchmark work, result writing, and failures across repeated runs.

## 2026-08-05 workload assessment

The retained development metrics now contain 59 runs: 45 successes, nine failures, and five interruptions.
The 14 non-successful runs represent 23.7% of the observed attempts.
Across successful cold runs, median wall time is 262.67 seconds and median extension-load time is 210.75 seconds.
The extension load accounts for a median 77.60% of successful cold-run wall time.
The focused correctness and timing stages normally take 0.94 and 3.20 seconds, respectively.
A warm extension load takes a median 0.17 seconds.

The records contain 44 cold loads for 22 extension fingerprints.
Twenty-two cold loads therefore duplicated a build of the same semantic binary.
Those duplicate loads consumed an estimated 4,541 seconds of aggregate compiler time.
Parallel launch sometimes reduced wall time, but it duplicated work and GPU allocation cost.

The evidence admits the durable-profiler-artifact phase.
At assessment time, the shared NCU runner exported CSV from a temporary `.ncu-rep` and then deleted the raw report.
Repeated profiling and several stalled NCU workers make the raw report and per-profile progress events necessary debugging evidence.

The telemetry is sufficient for bottleneck ranking but not exact accounting for every historical run.
Eight records have negative residual time because observed remote durations exceeded the local observer's wall interval or overlapped another timed stage.
Future records must preserve this mismatch explicitly instead of reporting a negative residual.

The local environment contract is also incomplete.
`AGENTS.md` requires the repository `.venv`, but the main macOS checkout had no `.venv` or `.python-version` and the Makefile selected ambient `python3`.
The first frozen sync exposed an unconditional Triton dependency with no macOS distribution.
Infrastructure tooling needs a small CPU-importable dependency set that can be installed on macOS, while CUDA and Triton dependencies remain part of the Linux and Modal environment.

## Active infrastructure tranche

The next implementation has three bounded steps.

1. Restore the local environment contract with a locked infrastructure extra, a repository `.venv` bootstrap target, a fail-fast environment check, and `.venv/bin/python` as the default project interpreter.
   Validation command: `make infra-sync && make check-dev-env` from the main development checkout.
   Expected result: the repository `.venv` contains the CPU-side infrastructure test dependencies and imports the project without selecting an ambient interpreter.
   Possible failures include an incompatible lockfile, a platform-specific dependency leaking into the infrastructure extra, or an unsupported `uv` version.
   The `.venv` is ignored and remains in the main development checkout rather than this infrastructure worktree.
2. Add a single-writer experiment build gate and an orchestration command that waits for an explicit Volume commit before launching timing and any explicitly requested NCU configurations.
   Every consumer must reload the Volume before loading the extension.
   Validation commands: `make check-dev-env`, `.venv/bin/pytest benchmarking/tests/test_cutlass_infra.py -q`, and a controlled `make modal-cutlass-experiment CUTLASS_VARIANT=warpgroup-fastmath-smem CUTLASS_DEV_LABEL=build-fanout-validation` packet.
   Expected result: exactly one cache writer for the selected fingerprint, no consumer cache misses, distinct logs and result files, and a failed build that prevents all consumers from launching.
   Possible failures include stale Volume mounts, an extension file still open during commit, concurrent consumers writing the same artifact, or orchestration that loses a child exit code.
   Generated evidence belongs under `benchmarking/modal-results/cutlass/experiments/<variant>/` and `benchmarking/modal-results/cutlass/dev-metrics/` and must not be committed.
3. Preserve each raw NCU report under a unique development run identifier and emit query, profile, export, commit, and extension-load-start events.
   Preserve the signed residual and record UTC wall time separately from observer-active time when remote events cannot be reconciled with the local interval.
   Validation commands: the local infrastructure suite above followed by the controlled Modal packet from Step 2.
   Expected result: every requested NCU configuration retains its candidate and Triton raw reports, summaries identify their Volume paths, progress is visible before a long child process finishes, and metrics never claim a negative residual.
   Possible failures include Volume commit conflicts, missing run identifiers, open files preventing reload, raw reports exceeding practical packet size, or a profiler child that stalls before emitting progress.
   Raw profiler artifacts belong under the shared Volume's CUTLASS experiment artifact prefix and must not be committed.

Do not begin preprocessed-source fingerprints until this tranche has measured whether the single-writer barrier removes the dominant recurring delay.

### Active tranche result

The local environment contract is implemented with a locked `infra` extra and a Linux-only marker on Triton.
The real `.venv` is under the main development checkout, and the infrastructure worktree uses that environment without keeping a second copy.
`make check-dev-env` passed with Python 3.12.4 and the repository-local Modal client.

The local infrastructure suite passed all 31 tests.
The suite covers precise fingerprints, registry validation, event recording, non-negative residual accounting, exact workflow command construction, and the rule that a failed build launches no consumers.
Both the build-gate and complete-workflow Make dry runs resolved the expected modules, arguments, labels, and result paths.
Python bytecode compilation and `git diff --check` also passed.

The controlled B200 command was:

```text
make modal-cutlass-experiment CUTLASS_VARIANT=warpgroup-fastmath-smem CUTLASS_DEV_LABEL=build-fanout-validation
```

The complete workflow passed in 63.86 seconds.
The build barrier completed in 15.39 seconds, loaded the already-cached extension in 1.12 seconds, committed the Volume in 0.61 seconds, and exited before any consumer was launched.
The timing consumer completed in 25.73 seconds and retained 240 repetitions across 12 provider cells.
Both focused exact-winner cases passed.
The D=4,096 and D=8,192 NCU consumers ran in parallel and completed in 46.85 and 48.22 seconds.
All five consumer extension-load events were cache hits between 0.14 and 0.37 seconds.
Every metrics record in the controlled packet reported consistent accounting.

The historical negative residuals had two measured causes.
Two failed runners timed extension loading both as an `extension_load` and inside an enclosing correctness stage, which double-counted 223.57 and 231.01 seconds.
Six other records came from three launch batches where the local UTC interval exceeded the observer's `perf_counter()` interval by 27.42 to 108.25 seconds while remote work continued.
The previous runner mislabeled observer-active time as wall time and compared it with remote durations that continued during the local suspension.
The runner now records UTC wall time, observer-active time, and their signed difference separately.
It preserves a signed unattributed residual and marks inconsistent records instead of clamping them.
The summarizer reconstructs UTC wall time for legacy records from their retained start and end timestamps.
The metrics summary reports accounting failures and excludes inconsistent residuals from its additive median.
The post-fix audit reduced the legacy accounting failures from eight to three without changing any raw record.
The two nested-timer failures remain invalid at -225.18 and -215.21 seconds, and the resumed shared-memory run remains invalid at -15.37 seconds.

The original validation requested the full four-configuration menu.
It retained candidate and Triton reports for H=128 and H=256 at both hidden sizes, for eight raw `.ncu-rep` files in total.
The local NCU summaries record the exact Volume paths and unique development run identifiers.

The workflow record is `benchmarking/modal-results/cutlass/experiments/warpgroup-fastmath-smem/workflow-20260805T071813Z-build-fanout-validation-f98318ca.json`.
The structured records share the `20260805T071813Z-build-fanout-validation-f98318ca` workflow identifier under `benchmarking/modal-results/cutlass/dev-metrics/`.
The raw reports are under `cutlass-profiler/experiments/warpgroup-fastmath-smem/` on the `fused-mm-sample` Volume.
These generated artifacts are ignored and must not be committed.

This packet validates ordering, explicit commit and reload, parallel fan-out, progress events, summaries, and durable raw reports against a warm fingerprint.

The cold-path validation used a unique experimental fingerprint and completed in 285.87 seconds.
The build barrier was the only cache miss and spent 208.26 seconds compiling before a 2.64-second explicit Volume commit.
Timing then loaded the published extension in 0.57 seconds, passed two focused correctness cases, and completed its timing stage in 3.36 seconds.
Both profiler consumers loaded the same published extension without compiling.
The D=4,096 and D=8,192 profiler processes completed in 46.65 and 60.73 seconds and retained all eight reports from the then-default full matrix.
The workflow record is `benchmarking/modal-results/cutlass/experiments/infra-cold-probe-20260805/workflow-20260805T073212Z-cold-single-writer-validation-bf7ba1f8.json`.
The structured records share the `20260805T073212Z-cold-single-writer-validation-bf7ba1f8` identifier under `benchmarking/modal-results/cutlass/dev-metrics/`.
The raw reports remain under `cutlass-profiler/experiments/infra-cold-probe-20260805/` on the shared Volume.
The temporary registry entry was removed after measurement, while its profiler artifacts were retained.

The cold packet validates the single-writer cache handoff, but it also shows that profiling every menu item is unnecessary recurring work.
The workflow now defaults to build and timing only.
The client must explicitly pass a JSON list with named parameters, such as `CUTLASS_PROFILE_CONFIGS='[{"hidden_size":4096,"n_hidden_states":128}]'`, when NCU evidence is needed.
The full menu crosses `hidden_size` values 4,096 and 8,192 with `n_hidden_states` values 128 and 256 and is never selected implicitly.
Each selected configuration creates exactly two retained raw reports, one candidate and one matched Triton baseline.
Timing must pass before the selected profiler jobs launch in parallel.

The selective B200 validation requested only `hidden_size=4096` and `n_hidden_states=128` and completed successfully in 90.03 seconds.
The build barrier loaded the warm extension in 0.72 seconds and committed it in 0.89 seconds.
The timing consumer loaded it in 0.30 seconds, passed correctness in 0.74 seconds, and completed timing in 3.09 seconds.
Only after timing passed, the selected profiler loaded the extension in 0.66 seconds and retained exactly two reports for the candidate and Triton.
The workflow record is `benchmarking/modal-results/cutlass/experiments/warpgroup-fastmath-smem/workflow-20260805T074107Z-explicit-profile-validation-04ee57e8.json`.
Its NCU summary is `benchmarking/modal-results/cutlass/experiments/warpgroup-fastmath-smem/ncu-d4096-h128-summary.json`.
Possible failures were an implicit profile launch, a profile starting before timing completed, a stale cache mount, or a report path collision between configurations.
None occurred in this packet.

## Next infrastructure tranche: reduce unavoidable recompilation

The single-writer barrier removed duplicate builds for one semantic extension, but a genuinely new fingerprint still spent 208.26 seconds compiling.
The focused correctness and timing work after that build took 4.22 seconds.
Recompilation is therefore the dominant remaining development-loop cost.

Do not replace the conservative extension fingerprint with a custom preprocessed-source fingerprint yet.
First test established CUDA compiler instrumentation, intra-translation-unit parallelism, and an object cache in isolation.
Keep the source snapshot, CUTLASS variant, architecture, optimization level, and validation packet fixed while changing one infrastructure variable at a time.
Give every build configuration a distinct extension key and artifact directory.

### Step 1: compile-time baseline

Add a build-only compile-study driver for the registered `warpgroup-fastmath-smem` variant.
Pass `--time=-` to both CUDA translation units and retain the phase CSV in the process log, `.ninja_log`, `build.ninja`, compiler output, object sizes, and shared-library size.
Record the requested and observed CPU resources, CUDA and NVCC versions, compile flags, cache state, translation-unit timings, link timing when available, and total extension-load time.

Validation command: `make modal-cutlass-compile-study CUTLASS_COMPILE_STUDY=baseline`.
Expected result: one guaranteed-cold extension build with phase rows for both CUDA translation units and a durable build packet.
Actual result: the first preflight found that `ctadvisor` was absent from the PyTorch CUDA development image, so the study image now installs the CUDA 13.0 advisor package.
The first traced cold build then spent 197.25 seconds before both PTXAS invocations rejected their independently named, incomplete JSON streams with `Parsed trace file not in expected format`.
Both failed trace files and the full local process log were preserved.
The comparison harness now uses `--time=-`, which NVIDIA documents as phase CSV output, because the failure is inside the CUDA 13.0 `sm_100a` device-trace path rather than a Ninja filename collision.
The final cold baseline completed in 147.86 seconds.
Ninja recorded 147.58 seconds for `greedy_provider.cu`, 116.22 seconds for `winning_schedule_provider.cu`, and 0.25 seconds for linking, with both CUDA translation units starting concurrently.
Compile Time Advisor runs as a separate trace-only diagnostic and is not allowed to alter the optimization comparison.
Possible failures include interleaved phase rows from concurrent compiler processes, incomplete process output, or generated artifacts that are not committed to the shared Volume.
Generated evidence belongs under `benchmarking/modal-results/cutlass/compile-cache-study/baseline/` locally and `cutlass-compile-study/baseline/` on the shared Volume.

### Step 2: NVCC split compilation

Repeat the baseline with exactly one additional compiler setting at a time, first `--split-compile=4` and then `--split-compile=8`.
Request a fixed CPU allocation for every comparison so scheduler-dependent CPU availability does not confound the result.
Do not test `--split-compile=0` until fixed limits show useful scaling because Ninja can run both CUDA translation units concurrently and an unlimited setting can oversubscribe the container.
Do not use `--split-compile-extended` because NVIDIA documents that it can affect runtime performance and requires device link-time optimization.

Validation commands: `make modal-cutlass-compile-study CUTLASS_COMPILE_STUDY=split4` and `make modal-cutlass-compile-study CUTLASS_COMPILE_STUDY=split8`.
Expected result: successful cold builds whose phase packets show whether device optimization parallelizes and whether four or eight threads reduce wall time without increasing failures or memory pressure.
Actual result: `--split-compile=4` took 204.94 seconds, which is 38.60% slower than the 147.86-second baseline.
`--split-compile=8` took 219.74 seconds, which is 48.61% slower than baseline.
The dominant `greedy_provider.cu` object increased from 147.58 seconds to 203.74 seconds and 218.82 seconds respectively.
Reject both settings for this workflow and retain the default compiler behavior.
The CUDA 13.2 trace-only advisor packet explains the result: only 2.59 seconds of optimization work is parallelizable, while device frontend work consumes 121.65 seconds of 152.23 gross traced seconds.
The advisor attributes 69.23 seconds across 26 `cutlass::device_kernel` instantiations and reports 58,651 to 85,290 recursive template instantiations in the most expensive examples.
The most expensive project include is `evt_candidates.cu` at 6.80 seconds across the two translation units, followed by the broad `torch/extension.h` include chain at 6.49 seconds.
CUDA 13.0.88 could not produce complete traces for these large sources, and a CUDA 13.2 full-object trace later failed in PTXAS for the larger source.
The working advisor lane therefore stops after PTX generation, marks the extension load as `trace_only`, and combines its frontend report with the CUDA 13.0 phase CSV for host compilation and PTXAS timing.
Its two retained raw JSON traces are 7.29 GB and 2.73 GB and must not be deleted.
Possible failures include oversubscription between Ninja and NVCC, memory exhaustion, no parallelizable optimizer region, or a compiler defect exposed by split compilation.
Compare extension-load wall time, per-translation-unit trace duration, CPU utilization when available, binary resource usage, focused exact-winner correctness, and interleaved timing against the baseline.
Generated evidence belongs under sibling `split4/` and `split8/` directories in the local and shared compile-study roots.

### Step 3: persistent object cache

The default compiler behavior won the split-compilation comparison.
Add `ccache` in front of its CUDA 13.0 NVCC invocation and use a dedicated rebuildable cache location rather than the experiment-artifact tree.
Verify the compiler wrapper used by the pinned PyTorch 2.11 extension builder instead of assuming behavior from a newer PyTorch implementation.
Configure stable paths or `CCACHE_BASEDIR` only after inspecting the actual command lines.
Preserve `ccache --show-stats` before and after every build.

Measure four cases in order: an empty-cache build, an exact rebuild in a fresh extension directory, a source change confined to one translation unit, and a feature-flag change that preprocesses away in one translation unit.
Keep the extension-level fingerprint conservative even if the object cache gets a hit.
The object cache may reuse an object only when its own compiler-derived key proves compatibility.

Validation command: `make modal-cutlass-compile-study CUTLASS_COMPILE_STUDY=ccache-<case>` for each allowlisted case.
Expected result: the empty-cache build matches the selected split-compilation baseline, the exact rebuild obtains two object hits, and adjacent changes recompile only objects whose effective compiler inputs changed.
Actual result: the empty-cache build took 224.57 seconds and recorded two misses.
An exact rebuild in a fresh extension directory took 1.59 seconds and added two direct hits with no misses.
A controlled header change confined to `greedy_provider.cu` took 211.38 seconds, with a 0.96-second direct hit for `winning_schedule_provider.cu` and one new miss for the affected object.
A controlled global feature flag that preprocesses away from `winning_schedule_provider.cu` took 141.32 seconds, with a 2.37-second preprocessed hit for the unaffected object and one new miss for `greedy_provider.cu`.
The cache therefore gives large exact and unaffected-object savings, but it does not reduce a cold frontend and the empty-cache overhead needs a same-host control before enabling it globally.
Possible failures include the pinned PyTorch builder bypassing the wrapper, absolute paths preventing hits, network filesystem overhead exceeding saved compilation time, excessive cache inode growth, or unsafe cache reuse caused by overly loose configuration.
Validate every reused binary with focused correctness and compare performance from a normally optimized build before accepting it as experiment evidence.
Generated statistics and build artifacts belong under `benchmarking/modal-results/cutlass/compile-cache-study/ccache-<case>/`, while cache entries belong in a dedicated disposable cache Volume.

### Step 4: remove redundant architecture frontend work

The CUDA 13.0 phase CSV shows that the `-arch=sm_100a` shorthand runs device frontend work for both `compute_100` and `compute_100a`.
Measure an explicit `--generate-code=arch=compute_100a,code=sm_100a` target that emits the B200 architecture-specific cubin without embedding either PTX target.
Keep this as a separately keyed study until focused correctness and interleaved performance agree with the normal build.

Validation command: `make modal-cutlass-compile-study CUTLASS_COMPILE_STUDY=sass-only`.
Expected result: one `compute_100a` `cicc` row per CUDA translation unit, no `compute_100` row, a loadable extension, and lower cold extension-load time.
Actual result: the cold build completed in 122.33 seconds, 25.54 seconds or 17.27% below the 147.86-second baseline.
Ninja recorded 121.94 seconds for `greedy_provider.cu`, 106.32 seconds for `winning_schedule_provider.cu`, and 0.32 seconds for linking.
The phase CSV contains exactly one `compute_100a` device frontend and one `sm_100a` PTXAS invocation per translation unit.
The extension linked and loaded successfully, but this build has not yet passed focused kernel correctness or interleaved performance validation.
Possible failures include losing required forward compatibility, accidentally omitting the B200 cubin, runtime incompatibility, or a code-generation difference that changes kernel performance.
Generated evidence belongs under `benchmarking/modal-results/cutlass/compile-cache-study/sass-only/` locally and `cutlass-compile-study/sass-only/20260805T091045Z-compile-study-compile-study-sass-only-902de41c/` on the shared Volume.

### Step 5: make experiment-only kernels positive opt-ins

Compile Time Advisor showed that the sampling extension instantiated 23 ordinary-GEMM tuning variants even though its Gumbel-only Python module did not export those entry points.
It also compiled a fifth winning schedule that no production dispatch could reach.
The production-compatible `pruned` study is the historical comparison label.
Its implementation now leaves experiment-only ordinary-GEMM symbols disabled unless the caller explicitly enables them, while retaining the normal `-arch=sm_100a` target and its embedded PTX.

Validation command: `make modal-cutlass-compile-study CUTLASS_COMPILE_STUDY=pruned`, followed by `make modal-cutlass GATE=gumbel-experiment CUTLASS_VARIANT=warpgroup-fastmath-smem CUTLASS_DEV_LABEL=compile-pruning-timing`.
Expected result: a loadable extension, lower cold compile time, exact-winner agreement at H=128 and H=256, and timing consistent with the retained kernel implementation.
Actual result: the cold build completed in 93.23 seconds, 54.63 seconds or 36.95% below the 147.86-second baseline.
The focused and full experiment checks passed both exact-winner cases.
The six interleaved timing cells remained consistent with the established candidate range, from 0.96x to 1.89x Triton.
Production greedy and sampling extensions now exclude tuning code by default.
Ordinary-GEMM APIs opt into the same tuning code through a separate extension with `FMMS_ENABLE_GEMM_TUNING` and a distinct fingerprint.
This positive inclusion model prevents a new production loader from accidentally compiling tuning variants because it forgot a pruning flag.
The single-writer experiment build now requests the same 16-CPU allocation as the controlled compile studies instead of relying on Modal's default CPU request.
The durable compile packet is `cutlass-compile-study/pruned/20260805T093011Z-compile-study-compile-study-pruned-06492acc/` on the shared Volume.
The local correctness and timing artifacts are under `benchmarking/modal-results/cutlass/experiments/warpgroup-fastmath-smem/`.
The positive-opt-in implementation was validated again with the same compile-study command.
That independent cold build completed in 126.36 seconds and passed both focused exact-winner cases.
This rerun validates loading and correctness, but it is not used as a replacement speed comparison because its object timings differed materially from the earlier controlled packet.
Its durable packet is `cutlass-compile-study/pruned/20260805T110748Z-compile-study-compile-study-pruned-9bbb6b92/`, and its local summary is under `benchmarking/modal-results/cutlass/compile-cache-study/pruned/`.

A follow-up split the four retained winning schedules into separate CUDA translation units.
It passed focused correctness but took 96.07 seconds, 3.05% longer than the pruned monolith, because concurrent NVCC processes inflated individual object times to 87.98 through 95.08 seconds.
Reject the split and keep the monolithic winning-schedule translation unit.
The rejected implementation was removed, while its packet remains at `cutlass-compile-study/pruned-split-tu/20260805T105508Z-compile-study-compile-study-pruned-split-tu-78da0192/`.

### Later decisions

Refactor the Python binding, stable Stage 2 code, and candidate CUTLASS instantiations into narrower translation units only if the traces or cache misses identify a reusable boundary.
Test `--Ofast-compile=min` only as a separately keyed development lane after cache and split compilation are measured.
Never use a fast-compile binary for a performance decision, and require a normal `-O3` rebuild before promotion.
Treat a CuTe DSL implementation as a kernel-development project rather than an infrastructure optimization because it changes the implementation and integration surface.

Choose the next action from the measured result of each step.
Stop adding complexity if a technique does not materially reduce cold or adjacent-change wall time.

## Phase 1: precise build caching and friction telemetry

Phase 1 is the only implementation authorized by this finding initially.

Add a dependency-scoped extension fingerprint that includes the recursive local include closure, compiler flags, architecture, applied patches, pinned CUTLASS revision, and Python, Torch, and CUDA ABI identity.
Keep architecture and feature variants under distinct prefixes.
Emit structured extension-load events with cache state and duration.

Wrap the existing `make modal-cutlass` subprocess without changing its gate mapping or result layout.
Record end-to-end duration, time to the first remote event, extension-load events, exit status, log size, Git state, and any explicitly timed stages.
Keep these generated metrics under the ignored CUTLASS results directory.
Summarize them with pandas so the next infrastructure decision is based on observed recurring friction.

Validate the change with a controlled B200 packet.
The packet must compare the legacy broad fingerprint with the precise fingerprint, exercise cold and warm extension loads, and show that an unrelated harness edit no longer changes the production name while a transitive dependency edit still does.
Representative greedy and Gumbel calls must retain their outputs.

Expected outcome: warm loads reuse the same extension after unrelated harness edits, relevant edits still invalidate it, and the telemetry reports where the run time went.
Possible failures include an incomplete include closure, stale shared objects, hidden toolchain inputs, a wrapper that hides Modal failures, or telemetry that adds unstable parsing dependencies.
Generated evidence belongs under `benchmarking/modal-results/cutlass/dev-infra-phase1/` and must not be committed.

### Phase 1 result

The dependency audit reduced the production extension input set from 14 local files to six.
The six retained inputs are the two translation units, their two transitive local includes, and the two patches applied to the pinned CUTLASS checkout.
Edits to the eight standalone harness inputs no longer change a production extension name.

The controlled B200 run completed in 411.97 seconds.
Modal startup through the explicit remote marker took 10.83 seconds.
Cold extension loads took 197.94 seconds for greedy and 200.64 seconds for Gumbel, accounting for 96.75% of total wall time together.
The same-process warm reloads found both compiled shared objects and took 0.00028 and 0.00016 seconds.
The greedy and Gumbel smoke outputs were identical before and after the warm reload.

The first attempted launch was intentionally stopped after telemetry showed that placing the metadata environment layer before `apt_install` invalidated downstream image layers.
Moving that environment layer to the end reduced the relaunch to two lightweight environment layers and reused the existing apt and CUTLASS layers.
A second attempt failed locally before requesting a GPU because the Modal CLI resolved an older installed package for an absolute import.
The observed runner now prepends the current worktree's `src` directory to `PYTHONPATH`, preventing mixed-worktree submissions.
Both failures and the successful run remain in the ignored development-metrics packet, where the combined success rate is one of three attempts.

Validation command: `make modal-cutlass GATE=dev-infra CUTLASS_DEV_LABEL=precise-cache`.
Expected result: six precise inputs, cold misses followed by warm hits, and unchanged provider outputs.
Actual result: the gate passed with both output comparisons true and the timings above.
Possible failures were an incomplete include closure, stale reuse, an omitted patch or ABI input, a warm Ninja rebuild, or mixed-worktree submission.
Artifacts are under `benchmarking/modal-results/cutlass/dev-infra-phase1/` and `benchmarking/modal-results/cutlass/dev-metrics/`.

## Phase 2: durable profiler artifacts

Persist raw `.ncu-rep`, `.nsys-rep`, cubins, full SASS, metric inventories, and exported tables under unique run identifiers.
Do not discard raw profiler reports in remote temporary directories.
Admit this phase only if telemetry shows repeated profiling or artifact loss is a material source of delay.

## Phase 3: shared benchmark protocols

Extract paired same-process timing, cache policy, repetition capture, coverage validation, and pointwise decisions into small reusable helpers.
Retain raw repetitions and use pandas for summaries and comparisons.
Admit this phase only if duplicated harness work or inconsistent protocols are recurring costs.

## Phase 4: immutable experiment packets

Give every run an immutable identifier and a provenance manifest.
Record source digests, toolchain identity, command arguments, GPU metadata, status, and artifact checksums.
Use pointers for latest and selected runs rather than overwriting evidence.

## Phase 5: declarative experiment registry

Replace repeated Makefile maps and module constants with one CPU-importable experiment registry.
Keep `make modal-cutlass GATE=<gate>` compatible.
Add candidate comparison and promotion commands only after the smaller abstractions have stabilized.

### Phase 3 and Phase 5 result

The Gate 4 recovery step created nine one-off timing, profiling, and layout modules plus sixteen near-identical experimental APIs and loader functions.
That duplication satisfied the roadmap's admission rule for shared protocols and a declarative registry.

The consolidated implementation keeps one CPU-importable compile-time variant registry, one timing runner, one NCU runner, and one generic experimental sampling API.
The Makefile now has one timing gate and one NCU gate parameterized by `CUTLASS_VARIANT`, while each variant still writes to a distinct ignored result directory.
The production sampling API remains separate from the experiment API.

Only the fastest rolled candidate, the spill-free shared-memory control, and their combined fast-math spill-free candidate remain registered.
Completed rejected implementations were removed after their measured outcomes were consolidated into one Gate 4 recovery finding.
The specialized SASS audit and Triton compiler-dump tools remain separate because they produce different artifact protocols and are not per-candidate benchmark wrappers.

Validation commands:

```text
PYTHONPATH=src python3 -m pytest benchmarking/tests/test_cutlass_infra.py -q
make -n modal-cutlass GATE=gumbel-experiment CUTLASS_VARIANT=warpgroup-fastmath
make -n modal-cutlass GATE=gumbel-experiment-ncu CUTLASS_VARIANT=warpgroup-fastlog-smem CUTLASS_HIDDEN_SIZE=8192
```

The expected result is one resolved module and result directory per generic gate, registry validation of the selected variant, and no import of Torch by the registry.
The dry runs resolve the expected shared modules and variant-specific ignored directories.
The local infrastructure suite passed all 21 tests, and an unknown variant failed locally before creating a result directory or requesting a Modal GPU.
The two retained variants then compiled and ran concurrently on separate B200s with distinct result paths.
`warpgroup-fastmath` completed in 254.47 seconds, including a 232.10-second cold extension build, 0.98-second correctness stage, and 3.29-second timing stage.
`warpgroup-fastlog-smem` completed in 248.20 seconds, including a 210.75-second cold extension build, 0.82-second correctness stage, and 8.37-second timing stage.
Both focused correctness checks passed, both timing matrices completed, and parallel execution kept the observed validation wall time near the slower 254.47-second run instead of the 502.67-second sum.
The packets are under `benchmarking/modal-results/cutlass/experiments/` and `benchmarking/modal-results/cutlass/dev-metrics/`.
Possible failures include an unknown variant reaching a remote compile, two variants sharing a result path, production accidentally loading an experimental module, or a stale one-off module remaining in the Make allowlist.

The first new registry-only candidate then completed in 255.59 seconds, including a 231.57-second cold extension build, 1.20-second correctness stage, and 3.08-second timing stage.
After the cache-fill event, its two NCU shapes launched in parallel and completed in 35.96 and 36.36 seconds with warm extension loads no longer than 1.83 seconds.
This validates the intended workflow: pay for one cold build, then fan independent diagnostics out in parallel without duplicating compilation or waiting for timing analysis.

## Decision rule for later phases

Choose the next phase from the collected timing and failure data instead of following this roadmap mechanically.
Prioritize the largest recurring avoidable delay or most frequent failure source.
Do not migrate completed historical gates merely for consistency.

## Recorded recurring friction

The following concrete workloads motivated the shared benchmark protocol and declarative experiment registry.
They remain here so infrastructure work can optimize measured failures rather than inferred ones.

### Infrastructure problem

The eight-element SM100 epilogue experiment spent about 11 minutes compiling three closely related PyTorch CUDA extensions before a timing sweep that completed in less than one minute.
This cold-build path materially limits the rate at which CUTLASS kernel hypotheses can be tested on Modal.
The infrastructure opportunity is to avoid serial recompilation of shared CUTLASS translation units and to expose phase timing in every experiment log.

### Concrete workload

The experiment changed the SM100 epilogue tile from 128x16 to 128x8 for the three production 256x128 schedule donors.
It then compared the greedy provider, the current Gumbel-Max provider, and the eight-element candidate on one B200.
The timing matrix contained two model shapes, H={64,128,256}, three providers, five warmups, and 20 cold-L2 repetitions.
The runner made 462 provider calls in total, of which 360 were measured calls and each measured call first zeroed a 256 MiB cache tensor.
The historical command, superseded by the generic experiment runner, was:

```text
make modal-cutlass GATE=gumbel-small-epilogue
```

The Modal app was `ap-XfzbH2dLb4ofXQJwQ5j4AK`.
It ran from 2026-08-04 15:42:56 CEST to 15:54:21 CEST, for 11 minutes 25 seconds.
The local log is `benchmarking/modal-results/cutlass/21-gumbel-small-epilogue/timing-log.txt`.

### What the cold path compiled

The runner lazily loaded these three content-keyed extensions in sequence:

```text
fmms_cutlass_sampling_sm100_fb5f2f772a0f
fmms_cutlass_sampling_small_epilogue_sm100_fb5f2f772a0f
fmms_cutlass_greedy_sm100_fb5f2f772a0f
```

Every extension separately compiled `greedy_provider.cu` and `winning_schedule_provider.cu` and then linked a separate shared library.
The source digest in `cutlass_impl.py` hashes every `.cu`, `.cuh`, and `.patch` file in the CUTLASS source directory.
Consequently, a change anywhere in that directory gives every provider a new extension name and invalidates all of their cached builds.
The shared objects are cached under `/vol-fused-mm-sample/cache/torch_extensions` on the `fused-mm-sample` Modal volume.

The volume metadata gives this minute-resolution sequence:

| Extension | `build.ninja` created | Shared library created | Shared library size |
|---|---:|---:|---:|
| Current Gumbel-Max | 15:43 | 15:46 | 15.7 MiB |
| Eight-element candidate | 15:46 | 15:50 | 16.2 MiB |
| Greedy | 15:50 | 15:54 | 14.0 MiB |

The sequence accounts for nearly the entire 11-minute app lifetime.
The app stopped 21 seconds after the last shared library's minute timestamp, so the timing and correctness work itself took less than 81 seconds and possibly much less.
The current logs do not permit a more exact split because `torch.utils.cpp_extension.load` ran with verbose output disabled and the runner emitted no phase timestamps.

### Parallel work already used

The independent ownership gate ran concurrently in app `ap-dRPSpd7I36ESfTwGOEQn7d` and used a different local log.
It ran from 15:42:50 to 15:45:29, for 2 minutes 39 seconds.
Its Modal image build compiled three standalone ownership binaries and explicitly reported 126.09 seconds for the main image layer.
Running it in parallel kept it off the timing gate's critical path.

The NCU diagnostic was intentionally launched after the timing job because it needed the same newly keyed extensions and concurrent first builds must not race in the shared volume.
App `ap-kEPKEbDCO5jcoGWNQgWR1i` then ran from 15:54:36 to 15:55:43, for 1 minute 7 seconds.
It found all three extensions in the shared cache and immediately executed six four-pass NCU profiles.
That warm-cache behavior is additional evidence that the first job was dominated by extension compilation rather than the benchmark matrix.

### Experiment result

All six deterministic comparisons passed.
The candidate was rejected because it ranged from a 2.1% improvement to a 14.4% regression at H>=128 and remained 1.48x to 3.75x slower than greedy.
NCU showed that the smaller fragment kept the 255-register allocation but increased dynamic local loads by 111% and local stores by 70% at H=128 and H=256.
The kernel result matters only as a reproducible workload for the infrastructure problem.

### Requested infrastructure improvements

The highest-value change is a single experiment module that exposes greedy, current, and candidate entrypoints without compiling the same translation units three times.
A second option is fine-grained extension hashes based on each module's actual dependency closure instead of one digest for the entire CUTLASS source directory.
A build-only Modal phase could compile distinct content keys concurrently and commit them before allocating a B200 for correctness, timing, or NCU.
Any concurrent builder must use distinct extension directories and must not race on the same content key.
The runner should record timestamps before and after every extension load, CUDA allocation, correctness phase, warmup phase, and measurement phase.
It should enable `FMMS_CUTLASS_VERBOSE=1` for cold builds and save each `.ninja_log`, compiler command, cache-hit decision, object size, and link time in the result packet.
These measurements would let an infrastructure change report developer-time improvement directly rather than infer it from minute-resolution volume metadata.

### Parallel result suffix pitfall

The first parallel CUTLASS-versus-Triton repetition used distinct `run1-log.txt` and `run2-log.txt` files, but `RUN=2` was only an environment value and was not passed to the Modal local entrypoint.
Both jobs therefore wrote the `run1` CSV packet even though their logs were distinct.
The Make gate now passes `--run $(RUN)` explicitly, and repetition 2 was rerun successfully.
Future parallel-run helpers should derive both logs and every result filename from one validated run identifier and print the resolved paths before launch.

### Instrumented post-rebase follow-up

Commit `06c69f8` added dependency-scoped extension names and structured development-loop events.
The first post-rebase immediate-warp run used app `ap-ZEqe9Lchtft8qnEeaKM1rQ` and the label `post-rebase-cache`.
It completed in 262.67 seconds, or 4 minutes 22.67 seconds of experiment time.
The one cold extension load took 193.99 seconds and accounted for 73.9% of the complete run.
The precise fingerprint contained six inputs instead of every CUTLASS harness source.
The remaining 68.68 seconds covered Modal startup, correctness, timing, result transfer, and shutdown, but the runner could not split them because this gate did not yet emit a remote-start marker or timed-stage events.
The warp-reduction runner now emits the remote-start marker and explicit correctness and timing stages for future runs.

Two independent NCU jobs then reused the exact compiled binary in parallel.
At D=4,096 the first process loaded the cached shared object in 5.39 seconds and the second in 0.18 seconds.
At D=8,192 the corresponding loads took 4.88 and 0.10 seconds.
Those jobs wrote distinct D-specific logs and CSV packets, so they did not contend for a result path.
The structured record for the cache-fill run is `benchmarking/modal-results/cutlass/dev-metrics/20260804T173156Z-gumbel-immediate-warp-post-rebase-cache-750e1c3c.json`.

### Register-liveness experiment friction

The full-unroll and shared-memory register-liveness experiments exposed several additional failures whose experiment time was much larger than their GPU measurement time.
The first full-unroll build spent 231.01 seconds compiling, passed the 0.85-second correctness stage, and then failed after 0.008 seconds in the timing stage because `_run_timings` did not receive the extracted `variant` argument.
The record is `benchmarking/modal-results/cutlass/dev-metrics/20260804T175245Z-gumbel-warpgroup-full-full-unroll-9c5c704f.json`.
A source edit for the independent shared-memory variant changed the shared translation-unit fingerprint and prevented the corrected full-unroll runner from reusing that binary.
The next full-unroll attempt made no observable progress for 119.48 seconds before the Modal function was cancelled.
The detached retry then spent another 199.79 seconds compiling before correctness and timing completed in 0.91 and 2.55 seconds.
The successful record is `benchmarking/modal-results/cutlass/dev-metrics/20260804T181418Z-gumbel-warpgroup-full-full-unroll-detach-ad00e817.json`.

The first shared-memory build routed the staged callback into the unrelated `256x256` donor and spent 223.57 seconds compiling before the callback's `FragmentSize == 16` static assertion rejected its 32-element fragment.
After the selector was fixed, the next attempt ran for 123.69 seconds before remote cancellation.
The detached resume still spent 206.78 seconds loading or compiling the extension, while exact correctness and the complete paired timing matrix took only 0.95 and 2.52 seconds.
Those records are `benchmarking/modal-results/cutlass/dev-metrics/20260804T175245Z-gumbel-warpgroup-smem-smem-stage-dfa70b54.json`, `benchmarking/modal-results/cutlass/dev-metrics/20260804T175722Z-gumbel-warpgroup-smem-smem-stage-selector-fix-a87b3d1e.json`, and `benchmarking/modal-results/cutlass/dev-metrics/20260804T180636Z-gumbel-warpgroup-smem-smem-stage-resume-f22d26dc.json`.

Two full-unroll NCU jobs were launched in parallel with a cold timing build at 20:16:58 CEST.
Both profilers connected to their Python targets at 20:17:07 CEST but emitted no extension-load or kernel-progress event during the next ten minutes.
They were explicitly stopped before relaunch because they had not profiled a kernel.
The D=4,096 and D=8,192 runs consumed 657.69 and 661.13 seconds of experiment time, all but about nine seconds unattributed.
The Modal apps were `ap-vm8wLDlq5lGugXmL6pG2w3` and `ap-KtbZKDVyYpiU8OmsEJ4JZX`.
The records are `benchmarking/modal-results/cutlass/dev-metrics/20260804T181658Z-gumbel-warpgroup-full-ncu-5e3c31c2.json` and `benchmarking/modal-results/cutlass/dev-metrics/20260804T181658Z-gumbel-warpgroup-full-ncu-33b27eee.json`.
This observation does not identify whether the profiler, extension file baton, or compilation itself caused the stall.

The next chunked-unroll experiment provides a useful counterexample.
Its timing and two NCU jobs cold-built the same fingerprint concurrently and completed successfully.
Their extension-load durations were 215.31, 225.00, and 213.47 seconds, respectively.
Parallel launch reduced wall-clock delay but duplicated about 10.9 minutes of aggregate compile work for one semantic binary.
The desired infrastructure path is therefore a build-only cache fill followed immediately by parallel timing and NCU fan-out, without making performance timing a prerequisite for profiling.

The dependency-scoped fingerprint from commit `06c69f8` solved invalidation from unrelated harness files.
It does not solve conditional invalidation within the two production translation units.
Any edit to `evt_candidates.cu` or `winning_schedule_provider.cu` still changes every experimental provider key, even when preprocessing would remove the changed branch for that provider.
A future infrastructure improvement should fingerprint the preprocessed translation unit or consolidate feature variants into one compiled module so nearby experiments can reuse common objects.

### Batch-4 guard failure and stalled parallel profile

The first four-way Philox/Gumbel ILP experiment launched timing and both NCU shapes concurrently at 2026-08-04 18:52:50 UTC.
All three workers compiled the same fingerprint independently and failed only after spending 232.99, 190.07, and 185.65 seconds in extension loading.
The complete timing job consumed 288.91 seconds of experiment time, including 45.16 seconds before the remote function started and 10.76 unattributed seconds after compilation failed.
The failure was a preprocessing error in `evt_candidates.cu`: an eager `#error` inside an uninstantiated batch callback fired while the extension compiled the unrelated 1-SM `greedy_provider.cu` translation unit.
The guard did not detect a runtime or template-instantiation error in the intended 2-SM path.
It was replaced by relying on the existing production selector, which instantiates the callback only for the validated SM100 2-SM schedule.
The records are `20260804T185250Z-gumbel-warpgroup-batch4-a3276bcd.json`, `20260804T185250Z-gumbel-warpgroup-batch4-ncu-48b92e0a.json`, and `20260804T185250Z-gumbel-warpgroup-batch4-ncu-1ca13dcc.json` under `benchmarking/modal-results/cutlass/dev-metrics/`.

The independent D=8,192 fast-log/value-first NCU run was stopped after 723.50 seconds of experiment time.
Its remote function started after 81.90 seconds, then emitted no extension-load or kernel-profile progress for the remaining 641.60 seconds.
The sibling D=4,096 profile completed from the same launch packet, so this is an orchestration or cache-path stall rather than evidence about the kernel.
The interrupted record is `20260804T184710Z-gumbel-warpgroup-fastlog-value-ncu-ccff6a8c.json`.

The corrected batch-4 fan-out again produced asymmetric orchestration behavior.
Timing and the D=4,096 profile completed in 269.81 and 208.72 seconds, including cold extension loads of 227.87 and 147.41 seconds.
The D=8,192 profile was stopped after 671.70 seconds because it emitted only `remote_start` and the initial NCU process connection.
Its 40.55-second startup was followed by 631.15 unattributed seconds with no extension-load or profile progress.
The interrupted record is `20260804T185949Z-gumbel-warpgroup-batch4-ncu-batch4-guard-fix-7b4c8c1c.json`.
This is the third instance in this experiment family where one parallel NCU worker stalls while a sibling using the same source snapshot and Modal image completes, so the infrastructure investigation should compare file-lock, volume-cache, and NCU child-process state across those sibling apps.
