# Ordinary GEMM retained-candidate tuning

The explicit-stage follow-up reached the bounded manual-search stop condition.

See `findings/cutlass/18-ordinary-gemm-stage-no-go.md` for that family's result.

The active handoff is now Gate 2c in
`findings/cutlass/01-fmms-kernel-plan.md`.
It replaces the manual stage-and-cluster proposal below with NVIDIA Matmul
Heuristics, `cutlass_profiler`, an explicit supported-family coverage audit,
and `torch.mm` as the sole strong baseline.
The active search targets B200 only.
Hopper kernel discovery is deferred until the complete B200 implementation
passes the pointwise superiority gate.

The ordinary-GEMM prerequisite now has a persistent multi-variant tuning runner.

Run it with:

```text
make modal-cutlass GATE=ordinary-gemm-tuning
```

The evidence packet is under `benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning/`.

## Scope

The fused candidate `GemmKernel` remains unchanged.

The diagnostic ordinary GEMM instantiates 64x128x64, 128x64x64, and 128x128x64 tiles.

Each tile uses both CUTLASS automatic schedules and the architecture-native schedule already validated by the fused kernel.

The native schedule is `KernelTmaWarpSpecialized` with `TmaWarpSpecialized` on SM90.

It is `KernelTmaWarpSpecialized1SmSm100` with `TmaWarpSpecialized1Sm` on SM100.

All variants use matched BF16 inputs and outputs, preallocated buffers, identical N padding, and cold-L2 timing.

The runner retains every raw repetition and candidate result instead of overwriting exploratory runs.

## Result

All 216 candidate outputs matched `torch.mm` bit-for-bit and were finite.

Selecting the fastest measured variant per architecture, model shape, and H passed 31 of 36 configurations.

The five remaining failures were:

| Architecture | V | D | H | Selected variant | CUTLASS/torch.mm |
|---|---:|---:|---:|---|---:|
| B200 | 128,256 | 8,192 | 256 | 128x128x64 native | 1.18 |
| B200 | 151,936 | 4,096 | 128 | 128x128x64 auto | 1.06 |
| B200 | 151,936 | 4,096 | 256 | 128x128x64 native | 1.23 |
| H100 | 128,256 | 8,192 | 256 | 128x128x64 auto | 1.32 |
| H100 | 151,936 | 4,096 | 256 | 128x128x64 auto | 1.56 |

The automatic and explicit native schedules each win some configurations.

Neither schedule family closes the high-H throughput gap.

The ordinary-GEMM prerequisite therefore remains tuning-required.

The worst ratio changed across the two tuning runs because the `torch.mm` medians also changed.

The packet records p10, p90, standard deviation, and all raw repetitions so that schedule decisions do not rely on the worst ratio alone.

## Superseded manual follow-up

The instructions below record the bounded manual search that produced Gate 18.
Do not use them as the active implementation plan.
The active Gate 2c workflow is in
`findings/cutlass/01-fmms-kernel-plan.md`.

Keep the six current variants as controls.

Add explicit stage-count candidates for the 128x64x64 and 128x128x64 tiles, then test legal cluster shapes for the H=128 and H=256 regime.

Preserve the full H sweep in the runner so a high-H improvement cannot silently regress smaller H values.

Do not add runtime production dispatch until one selected implementation passes `CUTLASS/torch.mm <= 1.05` everywhere.

Do not move any schedule into the fused epilogue until the ordinary prerequisite passes and Gate 1a re-derives any changed epilogue visitation.

### Historical handoff for the manual search

Continue the ordinary-GEMM prerequisite before doing more work on the fused epilogue.

The current bounded question is whether CUTLASS stage-count or cluster specialization can close the remaining H=128 and H=256 gap.

Use the existing retained-candidate runner, keep the current variants as controls, and preserve correctness and raw timing evidence.

The historical decision boundary required the selected ordinary CUTLASS path to remain within 5% of `torch.mm` in all 36 configurations before its schedule moved into the fused kernel.

If a reasonable bounded search cannot meet that threshold, record a no-go decision before adding Philox, tensor parallelism, or top-k.

The details below are starting points derived from the current evidence, not a requirement to follow a predetermined implementation.

The next agent should adjust the candidate set or screening method when CUTLASS constraints or new measurements justify it, and document that reasoning.

### Current implementation

The ordinary-GEMM template and named dispatch live in `src/fused_mm_sampling/csrc/cutlass/greedy_provider.cu`.

`PlainGemmVariant` currently parameterizes the tile shape, mainloop schedule, and epilogue schedule.

Extend that template with cluster-shape and stage-count template parameters rather than copying the builder.

The Python binding wrapper is in `src/fused_mm_sampling/cutlass_impl.py`.

The retained-candidate runner is `src/fused_mm_sampling/modal_lib/cutlass/ordinary_gemm_tuning.py`.

Shared dimensions and timing code are in `src/fused_mm_sampling/modal_lib/cutlass/ordinary_gemm_common.py`.

The gate is registered as `make modal-cutlass GATE=ordinary-gemm-tuning`.

The current evidence packet is `benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning/`.

Keep that gate, runner, and artifact directory canonical.

Do not create a second stage-tuning gate or overwrite one candidate with another inside the packet.

### Constraints that remain intentional

Use only 128x64x64 and 128x128x64 tiles in the next search.

The 64x128x64 tile did not win any unresolved H=128 or H=256 case.

Keep the existing six variants as controls in every final sweep.

Use BF16 inputs and outputs, preallocated buffers, identical padding for CUTLASS and `torch.mm`, cold-L2 timing, and CUDA events.

Continue to require exact BF16 equality and finite outputs for every compiled candidate.

Do not infer a performance cause from tile, stage, or cluster correlations alone.

Use separate names for every candidate with the format:

```text
tile-<M>x<N>x<K>-<auto|native>-stages-<count>-cluster-<M>x<N>x<K>
```

Use `stages-auto` for the existing automatic carveout controls.

### Suggested first investigation: explicit stage counts

Parameterize `PlainGemmVariant` so the mainloop builder accepts either the existing `StageCountAutoCarveout<sizeof(EpilogueStorage)>` or an explicit `cutlass::gemm::collective::StageCount<count>`.

Confirm the exact type name against the pinned CUTLASS 4.6.1 headers before editing the template.

The previous compiler output showed these automatic stage counts:

| Architecture | Tile | Automatic stages |
|---|---|---:|
| H100 | 128x64x64 | 9 |
| H100 | 128x128x64 | 7 |
| B200 | 128x64x64 | 8 |
| B200 | 128x128x64 | 6 |

The architecture-specific stage candidates below are a reasonable starting point.

They cover the automatic count and the two lower counts without expanding into an unbounded search:

| Architecture | Tile | Explicit stage counts |
|---|---|---|
| H100 | 128x64x64 | 7, 8, 9 |
| H100 | 128x128x64 | 5, 6, 7 |
| B200 | 128x64x64 | 6, 7, 8 |
| B200 | 128x128x64 | 4, 5, 6 |

Instantiate the stage candidates only for the architecture being compiled with the existing `FMMS_ARCH_SM90` and `FMMS_ARCH_SM100` guards.

Use both the automatic schedule family and the explicit architecture-native schedule family only if CUTLASS accepts the combination.

Treat a compile-time-invalid stage and schedule combination as a rejected tuning candidate, not as evidence about performance.

Record each rejection and its first useful compiler diagnostic in the finding.

Do not silently remove failed candidates.

For the first screen, benchmark only both primary model shapes at H=128 and H=256.

Use 25 warmups and 30 measured repetitions for this screen.

Retain `torch.mm` and the six existing variants as controls.

Promote a stage candidate only if it either passes `CUTLASS/torch.mm <= 1.05` or improves the matching best control by at least 3% in at least one unresolved configuration without regressing another screened configuration by more than 3%.

Discard all other stage candidates before adding cluster variants.

### Suggested follow-up: cluster shapes

Apply cluster tuning only to the promoted tile, schedule, and stage combinations from Phase 1.

Start from these cluster shapes:

```text
Shape<_1, _1, _1>
Shape<_2, _1, _1>
Shape<_1, _2, _1>
Shape<_2, _2, _1>
```

Do not assume every shape is legal on both architectures or with every schedule.

Confirm legality through the pinned builders, compilation, `can_implement`, and an actual launch on the target GPU.

Record rejected shapes and the rejection stage: builder, compile, `can_implement`, or launch.

Do not add a larger cluster search unless one of these candidates improves a still-failing case and the evidence suggests a specific next shape.

Use the same four high-H screening configurations, 25 warmups, 30 measured repetitions, and Phase 1 promotion thresholds.

### Full confirmation before promotion

After selecting at most one candidate per architecture and H regime, restore the complete H sweep and 100 measured repetitions.

Run:

```text
make modal-cutlass GATE=ordinary-gemm-tuning
```

The final packet must contain:

- All six existing controls.
- Every promoted stage or cluster candidate.
- Raw repetitions in `cases.csv`.
- Per-candidate medians, p10, p90, standard deviations, ratios, and pass flags in `case-summary.csv`.
- Exact correctness results in `correctness.csv`.
- One selected candidate per architecture, shape, and H in `selected.csv`.
- The overall decision and worst selected ratio in `summary.json`.
- Review instructions in `VERIFY.md`.
- Complete stdout and stderr in `log.txt`.

Do not select a candidate from one run and compare it with a `torch.mm` median from another run.

The ordinary prerequisite passes only when the selected candidate satisfies `CUTLASS/torch.mm <= 1.05` in all 36 configurations in the same final packet.

If the final sweep fails, repeat it once before attributing a regression to noise.

Keep both complete packets under distinct run subdirectories or filenames so the rerun does not erase the first result.

Summarize both runs with pandas.

Do not use pointwise minima across runs.

### Promotion after a pass

Only after the ordinary prerequisite passes:

1. Add the measured runtime dispatch by architecture and H regime.
2. Rerun the canonical ordinary-GEMM gate and confirm the dispatched path passes all 36 configurations.
3. Move only the approved schedule into a separate fused-kernel experiment.
4. Rerun Gate 1a because a changed tile or epilogue schedule may change accumulator ownership.
5. Rerun every correctness gate that depends on Gate 1a through Gate 2a.
6. Rerun Gate 2b against both `torch.mm` plus argmax and the approved ordinary CUTLASS baseline.
7. Begin Philox Gate 3 only if Gate 2b passes its existing threshold.

### Stop condition

Stop the CUTLASS port before Philox if the bounded stage and cluster search cannot make the ordinary baseline pass after the required confirmation rerun.

Record the no-go decision with the retained packets and the best measured candidate for each failed configuration.

Do not expand the search indefinitely or weaken the 5% threshold after seeing the results.

### Required cleanup and documentation

Remove candidates that were neither promoted nor retained as controls from the compiled extension after recording their results.

Keep the runner capable of reproducing the final selected and control variants.

Update this finding with the exact candidate matrix, rejected combinations, both architecture results, and the decision.

Update `AGENTS.md` with the new result and next canonical handoff.

Run Python syntax checks and `git diff --check`.

End by auditing `git status --short` and explaining every retained gate-specific file.
