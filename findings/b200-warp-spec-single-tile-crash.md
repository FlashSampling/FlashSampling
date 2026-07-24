# B200 + Triton 3.6 + warp specialization on correctness-test shapes = compiler crash

## Symptom

`make modal-pytest-distributed` (default `GPU=b200`) failed in both ranks during
the very first kernel compile of `verify_sampling_distribution_tp`:

```
/root/src/fused_mm_sampling/core.py:454:0: error: Failures have been detected
while processing an MLIR pass pipeline
note: Pipeline failed while executing
[`TritonGPUAutomaticWarpSpecialization` on 'builtin.module' operation,
 `NVWSInsertAref` on 'builtin.module' operation]
RuntimeError: PassManager::run failed
```

`make modal-speed-test` ran clean on the same image.

## Observed trigger

The chi-squared TP2 test calls the kernel with synthetic shapes from
`make_synthetic_inputs` in `src/fused_mm_sampling/testing.py`:
`vocab_size=100, n_hidden_states=1, hidden_size=10` (padded to D=16).

With `BLOCK_SIZE_V=128, BLOCK_SIZE_H=16` the persistent kernel sees
`num_pid_v=1, num_pid_h=1, num_tiles=1` and the host-side grid is
`min(NUM_SMS, num_tiles)=1`. The persistent loop becomes a single-iteration
`scf.for ... step=148` and `tl.range(..., warp_specialize=True)` (`core.py:521`)
asks Triton 3.6's `NVWSInsertAref` pass to lower it.
That lowering crashes on B200 (sm_100).
Bumping `D` (10 -> 64, padded to 72) gave the matmul a second D-step but did not fix the failure.

The original vocabulary guard disabled warp specialization for `V <= 256`.
After adding `V=512` to the test, the TP1 run passed V=100 and V=256 but failed at `V=512, H=1, num_samples=10_000` in the same `NVWSInsertAref` pass.
That shape has 2--4 tiles depending on `BLOCK_SIZE_V`, so `num_tiles=1` is not a complete characterization of the trigger.
Both failures combine a small shape, warp specialization, and the compile-time `num_samples=10_000` loop used by the distribution test.
The failure occurs during compilation before any samples are generated.

`make modal-speed-test` does not hit this because production shapes
(V >= 128k, D in {4096, 8192}) produce hundreds of tiles.

## Workaround

Restrict warp specialization to the production sampling path, which requests one sample per row:

```python
WARP_SPECIALIZE=(
    supports_warp_specialization_cached()
    and num_samples == 1
    and V > 2 * MIN_BLOCK_SIZE_V
),
```

Production decode keeps warp specialization because it requests one sample.
The chi-squared test requests 10,000 samples and uses the non-WS lowering.

## Verification

- Local RTX 3090 (cc 8.6, WS unavailable): `pytest tests/test_core.py` -
  95 passed, 2 skipped (TP2 cases). No regressions.
- The earlier vocabulary-only guard passed Modal B200 x2 before `V=512` was added to the test.
- Modal B200 TP1: `make modal-verify-correctness-tp1` passed all 30 distribution checks (5 providers x 3 vocabularies x 2 H) and all 6 greedy checks, including FMMS at `V=512`.

## What did not work

- Bumping `make_synthetic_inputs(hidden_size=10 -> 64)`: D went from 16 to 72
  (one bias column + TMA align), still crashed. Reverted.
- The `V > 2 * MIN_BLOCK_SIZE_V` guard does not cover `V=512`.
- Increasing `vocab_size` alone does not characterize the trigger because `V=512` still crashes.
