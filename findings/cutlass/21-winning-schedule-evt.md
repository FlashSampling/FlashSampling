# Gate 2d winning-schedule fused EVT

## Status

The first 2-SM fused-EVT experiment passes deterministic correctness on B200.

Run the experiment with:

```text
make modal-cutlass GATE=winning-schedule-evt
```

The gate currently targets the B200 128x64x128 schedule with cluster `(2,1,1)`.

## Direct per-tile reduction failure

Using the Gate 1g `Sm90RowReduction` with `FinalReduction=false` lets both
cooperating CTAs write the same compact candidate slot.
The first complete tile passed, but the second tile returned value `-40` at
index 192 instead of value `13` at index 255.
The existing visitor therefore does not provide the required cross-CTA merge
for this 2-SM schedule.

## Atomic final-reduction experiment

The follow-up uses `FinalReduction=true` with a custom 64-bit CAS reducer.
The packed comparator preserves FP32 value ordering and selects the lower index
on ties.
CUTLASS recognizes the custom reducer through an `is_atomic` specialization.

The pinned CUTLASS `fill_workspace` supports only 1-, 2-, and 4-byte elements.
The local `sm90-row-reduction-uint64.patch` lets callers explicitly initialize
8-byte atomic reduction outputs, and the harness fills every output with the
packed negative-infinity identity before launch.

The first atomic path correctly merged distinct maxima but failed the
`edge_winners` tie case: it produced value `-872` at index 254 instead of the
same value at the lower index 126.

A standalone two-thread contention kernel invokes the exact same packed CAS
functor with equal values at indices 254 and 126.
It deterministically returns index 126, so the comparator and CAS loop pass in
isolation.
An atomic trace showed that index 126 was never submitted to the reducer.
The failing code interpreted `tile_coord_mnkl.m` as a 128-row cluster tile and
added 64 rows for cluster rank 1.
For this 2-SM schedule, CUTLASS already exposes a per-CTA M coordinate and each
CTA owns 64 rows.
The extra cluster-rank offset double-counted CTA 1.

The corrected global index is `cta_m * 64 + consumer_thread % 64`.
With this mapping, all 400 exact candidate comparisons pass across complete and
partial M/N shapes, all-negative inputs, within-tile ties, cross-CTA ties, and
cross-cluster ties.
Compute Sanitizer memcheck reports zero errors and racecheck reports zero
hazards, errors, or warnings.
The original Gate 1g regression also still passes all 1,744 exact comparisons
and both sanitizers on H100 and B200.

## Modal runner note

The system Modal client repeatedly failed TLS and heartbeat operations during
this work.
An isolated current client with `uv tool run --from modal --with
pydantic-settings modal run --detach ...` remained connected.
The top-level `modal-cutlass` recipe now enables pipe failure propagation so a
failing `modal run` is no longer masked by a successful `tee`.
Gate logs and expected result files should still be inspected before approval.
