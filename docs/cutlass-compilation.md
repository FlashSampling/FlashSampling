# CUTLASS compilation design

Read this guide before adding CUTLASS kernels, schedules, template variants, source files, or includes.
CUDA frontend time is part of the development-loop cost and should be treated as a design constraint alongside runtime performance and correctness.

## Why the sampling sources were edited

Compile Time Advisor found that the Gumbel-only sampling extension instantiated 23 ordinary-GEMM tuning variants that its Python module did not export.
It also instantiated one winning schedule that no runtime dispatch could reach.
Those unused instantiations contributed 49.74 seconds of traced template work.

`greedy_provider.cu` now includes the ordinary-GEMM template definitions, launchers, small-N GEMV kernels, and Python bindings only when `FMMS_ENABLE_GEMM_TUNING` is defined.
Production greedy and sampling extensions do not define that flag.
Ordinary-GEMM APIs load a separate `fmms_cutlass_gemm_tuning_*` extension that explicitly enables the tuning surface.
The fifth winning schedule was removed from the production provider because no extension entry point or runtime dispatch called it.
Its dedicated winning-schedule experiment remains available independently.
This is a compile-surface change rather than a kernel-algorithm change.
The retained kernels, runtime dispatch, compiler optimization level, and PTX compatibility remain unchanged.

The controlled cold build fell from 147.86 seconds to 93.23 seconds.
Focused exact-winner checks passed at H=128 and H=256, and the six-cell interleaved timing packet remained in the established candidate range.
The detailed evidence is in `findings/cutlass/25-development-infrastructure.md`.

## Rules for new CUTLASS work

Instantiate only kernels that a public entry point or registered experiment can reach.
A configuration menu is not a reason to compile every configuration into every extension.
When the client needs a subset, pass that subset explicitly and give it a distinct extension fingerprint.

Keep production, experiment, tuning, and standalone correctness surfaces separate.
Do not add search-only GEMM variants or diagnostic launchers to a production sampling translation unit unless that module exports and uses them.
If code must share a source file, place experiment-only template definitions and launch calls behind a narrowly named positive compile-time guard before they can instantiate.
Prefer `FMMS_ENABLE_<FEATURE>` owned by the extension that needs it over a negative `PRUNE_UNUSED` flag owned by every extension that does not.

Treat every quoted include as part of the extension dependency closure.
Editing a shared implementation such as `evt_candidates.cu` invalidates every object that includes it.
Prefer a small stable declaration header when consumers need only types or function declarations.
Do not include a `.cu` implementation from additional translation units merely for convenience.

Keep Python bindings thin.
Include broad PyTorch binding headers only in the translation unit that defines the module when practical.
CUDA implementation files should include the narrowest ATen, CUDA, and CUTLASS headers their signatures and bodies require.
Measure header cleanup because the current advisor attributed only 6.49 seconds to the `torch/extension.h` chain, much less than template instantiation.

Do not assume more compiler parallelism is faster.
`--split-compile=4` and `--split-compile=8` were 38.60% and 48.61% slower than the baseline.
Splitting the four retained schedules into separate CUDA translation units also regressed the pruned build from 93.23 seconds to 96.07 seconds because concurrent NVCC processes inflated each object time.
Keep the monolithic winning-schedule translation unit unless a new measurement on a changed workload reverses that result.

Use ccache for exact or unaffected-object rebuilds, not as a substitute for reducing template work.
An exact rebuild measured 1.59 seconds, but a changed CUDA translation unit still required 141 to 211 seconds in the unpruned study.
Keep the extension fingerprint conservative even when the object cache can prove that one object is reusable.

## Required workflow for compile-surface changes

Start with `make modal-cutlass-compile-study CUTLASS_COMPILE_STUDY=pruned` when checking the current production-compatible compiler lane.
Add a separately keyed study when changing one compiler flag, architecture target, source partition, or template set.
Do not reuse an existing extension suffix for a semantically different study.

Record total extension-load time, every Ninja object duration, the compiler phase CSV, flags, source dependency closure, binary size, and the durable artifact path.
The expected result must name the object or phase that should improve.
The actual result must compare against the nearest controlled baseline and report regressions as well as wins.

Before promotion, run focused correctness through the compile-study driver and then run:

```bash
make modal-cutlass-experiment \
    CUTLASS_VARIANT=<variant> \
    CUTLASS_DEV_LABEL=<short-description>
```

The build must pass exact-winner checks and interleaved timing before a source-organization change becomes the default.
Preserve failed compiler output and successful packets under the documented local and shared-Volume result roots.

## Review checklist

- Does every new template instantiation have a reachable caller in this extension?
- Could a tuning or diagnostic variant live in its existing specialized gate instead?
- Does a shared include unnecessarily invalidate unrelated translation units?
- Is the client selecting the configurations it needs instead of compiling a full menu?
- Is the extension fingerprint distinct for the changed source and flags?
- Did a cold compile packet identify the affected object and phase?
- Did focused correctness and interleaved timing pass after the compile change?
- Were rejected wrappers, runners, and source partitions removed after their evidence was recorded?
