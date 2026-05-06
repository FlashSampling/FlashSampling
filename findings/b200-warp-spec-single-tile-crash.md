# B200 + Triton 3.6 + warp specialization + single-tile persistent loop = compiler crash

## Symptom

`make modal-pytest-distributed` (default `GPU=b200`) failed in both ranks during
the very first kernel compile of `verify_sampling_distribution_tp2`:

```
/root/src/fused_mm_sampling/core.py:454:0: error: Failures have been detected
while processing an MLIR pass pipeline
note: Pipeline failed while executing
[`TritonGPUAutomaticWarpSpecialization` on 'builtin.module' operation,
 `NVWSInsertAref` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

`make modal-speed-test` ran clean on the same image.

## Root cause

The chi-squared TP2 test calls the kernel with synthetic shapes from
`make_synthetic_inputs` in `src/fused_mm_sampling/testing.py`:
`vocab_size=100, n_hidden_states=1, hidden_size=10` (padded to D=16).

With `BLOCK_SIZE_V=128, BLOCK_SIZE_H=16` the persistent kernel sees
`num_pid_v=1, num_pid_h=1, num_tiles=1` and the host-side grid is
`min(NUM_SMS, num_tiles)=1`. The persistent loop becomes a single-iteration
`scf.for ... step=148` and `tl.range(..., warp_specialize=True)` (`core.py:521`)
asks Triton 3.6's `NVWSInsertAref` pass to lower it. That lowering crashes on
B200 (sm_100). Bumping `D` (10 -> 64, padded to 72) gave the matmul a second
D-step but did not move the needle - the trigger is the single-tile persistent
loop, not the small D.

`make modal-speed-test` does not hit this because production shapes
(V >= 128k, D in {4096, 8192}) produce hundreds of tiles.

## Fix

Gate `WARP_SPECIALIZE` on having enough V tiles for the largest autotune
`BLOCK_SIZE_V` (`2 * MIN_BLOCK_SIZE_V = 256`) to give >= 1 V tile, applied at
`src/fused_mm_sampling/core.py:297`:

```python
WARP_SPECIALIZE=supports_warp_specialization() and V > 2 * MIN_BLOCK_SIZE_V,
```

Production V (>=128k) is far above 256 and keeps WS on. Tiny test vocabs
(100, 200, 256) skip the buggy compile path and run the non-WS lowering, which
is correct (just unoptimized for prod-sized inputs).

## Verification

- Local RTX 3090 (cc 8.6, WS unavailable): `pytest tests/test_core.py` -
  95 passed, 2 skipped (TP2 cases). No regressions.
- Modal B200 x2: `make modal-pytest-distributed` - all 36 cases pass
  (5 providers x 3 vocabs x 2 H plus 6 greedy).

## What did not work

- Bumping `make_synthetic_inputs(hidden_size=10 -> 64)`: D went from 16 to 72
  (one bias column + TMA align), still crashed. Reverted.
- Increasing `vocab_size` would also work, but it skews the chi-squared test
  away from the small-bin diversity it needs.
