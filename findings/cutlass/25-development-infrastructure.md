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

## Decision rule for later phases

Choose the next phase from the collected timing and failure data instead of following this roadmap mechanically.
Prioritize the largest recurring avoidable delay or most frequent failure source.
Do not migrate completed historical gates merely for consistency.
