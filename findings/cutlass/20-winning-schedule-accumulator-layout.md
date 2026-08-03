# Gate 2d winning-schedule accumulator layout

Gate 2c selected five distinct B200 2-SM schedule instantiations that can
reach the fused candidate epilogue: 128x64x128 with cluster M=2,
256x128x64 with cluster M=2 or 4, 256x128x128 with cluster M=2, and
256x256x64 with cluster M=2.

The original Gate 1a diagnostic now accepts compile-time tile dimensions,
cluster M, and the SM100 2-SM schedule while preserving its original defaults.
The new runner compiles each winning schedule independently and verifies exact,
duplicate-free coverage of its complete CTA output tile with pandas.

Run it with:

```text
make modal-cutlass GATE=winning-schedule-layout
```

Results are written to
`benchmarking/modal-results/cutlass/15-winning-schedule-layout/`.

## Result

The B200 run passed all five schedule instantiations on 2026-08-03.
Every output coordinate had exactly one owner.
The runner now checks the formulas below against every raw row rather than only
recording summary ranges.

Let `t = threadIdx.x - 128`, `c` be the CTA rank encoded by `blockIdx.x`, `f`
be the fragment slot, and `e` be `epi_n`.

| Schedule tile | M coordinate | N coordinate |
|---|---|---|
| 128x64 | `(t % 64) + 64c` | `32 floor(t / 64) + 16e + f` |
| 256x128 | `t + 128c` | `16e + f` |
| 256x256 | `t + 128c` | `32e + f` |

The 256x128 formula is identical for K=64 and K=128 and for cluster M=2 and
cluster M=4.
The 256x256 schedule exposes 32 fragment slots per callback; the other
schedules expose 16.
All schedules use consumer threads 128 through 255, `epi_m=0`, and `epi_v=0`.

The original diagnostic record lacked a CTA identifier, which made the two M
halves look like duplicate ownership patterns.
The record now includes the CTA rank while remaining exactly representable in
FP32.

Gate 2d can now implement these three architecture-specific ownership formula
families in the fused candidate epilogue.
