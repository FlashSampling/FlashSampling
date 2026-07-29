# Gate 2b no-D epilogue

Gate 2b now removes the diagnostic FP32 `[V,H]` allocation and output store from the production greedy provider.

## Implementation

The provider sets `ElementD=void` while retaining the Gate 1g split-tree candidate EVT and Gate 1h Stage 2 comparator.
`CandidateEVT` advertises `ElementAux=float`, which gives CUTLASS a concrete internal type for epilogue layout and register conversion.

CUTLASS 4.6.1 already implements this contract for SM90, but its SM100 builder passes the public void destination into a collective that still instantiates D layout arithmetic and a D TMA descriptor.
The local `sm100-void-d.patch` applies the missing SM90-style conditions to the pinned SM100 TMA collective.
It:

- substitutes the EVT auxiliary type for internal D layout machinery;
- leaves the public destination type as `void`;
- disables D-dependent shared-memory reuse and delayed TMA-store scheduling;
- skips D TMA descriptor construction and prefetch;
- skips the D register-to-shared-memory output copy;
- skips the D shared-to-global TMA store.

The collective retains a D-shaped shared-memory allocation because `cst_callbacks.reduce` uses it as reduction workspace.
No GEMM output is copied into that buffer or written to global memory.
Removing or shrinking this workspace requires a separate EVT reduction redesign and is not necessary to eliminate the `[V,H]` allocation and traffic.

The patch is applied only to provider images and is checked against the exact pinned CUTLASS 4.6.1 source.
Gate 1 diagnostic binaries remain unpatched and retain their FP32 D output for direct inspection.

## TODO: upstream the CUTLASS fix

The downstream source patch is a temporary compatibility workaround, not the intended permanent architecture.
Extract a minimal SM100 `ElementD=void` reproducer with an EVT auxiliary output, add a focused regression test, and submit the fix to NVIDIA/CUTLASS.
Retain the local patch until the project pins an upstream CUTLASS release containing the fix and the complete Gate 2a matrix passes without the patch.

## Verification

Run:

```text
make modal-cutlass GATE=greedy-provider
```

The post-change Gate 2a rerun passed all 52 boundary, tie, and primary model-shape cases on H100 and B200.
The shared `test_greedy_sampling` suite also passed 18/18 CUTLASS cases independently on each architecture.
The generated packet is under `benchmarking/modal-results/cutlass/09-greedy-provider/`.

This rerun verifies compilation, runtime pipeline behavior, exact indices, boundary predication, deterministic ties, and H=1 through H=256.
It does not prove the absence of physical D writes through profiling.
Gate 2b performance profiling must confirm that claim with memory traffic and kernel metrics before the performance decision.

## Gate 2b result

The end-to-end Gate 2b sweep reached a no-go decision under the predeclared 5%
threshold.
The complete result is documented in
`findings/cutlass/14-greedy-performance.md`.
