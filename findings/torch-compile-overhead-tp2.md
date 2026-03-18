# torch.compile overhead at TP2

## Experiment

Ran the triton-bench on B200 TP2 with `torch._dynamo.config.disable = True` to measure the overhead of `torch.compile` on all providers. The `DISABLE_COMPILE=1` flag in the Makefile controls this.

## Affected code paths

Several functions in `core.py` use `@torch.compile(fullgraph=True)`:

- `_local_reduce` (FMMS Stage 2: reduce across V-tiles)
- `_stack_and_select_winner` (FMMS Stage 3: pick global winner across TP ranks)
- `flashinfer_create_logits_and_indices` (matmul + index creation for FlashInfer)
- `sample_compiled_fullgraph` / `sample_compiled_with_breaks` (multinomial baseline)

## Results

Speedup from disabling torch.compile (>1 means eager is faster):

### Small config (V=151,936, d=4,096)

| H | Multinomial Compiled | fi:top_k_top_p | fi:sampling |
|---|---|---|---|
| 1 | 1.43x | 1.18x | 1.20x |
| 2 | 1.38x | 1.23x | 1.23x |
| 4 | 1.37x | 1.09x | 1.14x |
| 8 | 1.33x | 1.07x | 1.17x |
| 16 | 1.24x | 0.99x | 1.09x |
| 32 | 1.11x | 0.96x | 1.00x |
| 64 | 0.96x | 0.95x | 0.94x |
| 128 | 0.91x | 0.94x | 0.91x |
| 256 | 0.88x | 0.96x | 0.93x |

### Large config (V=128,256, d=8,192)

| H | Multinomial Compiled | fi:top_k_top_p | fi:sampling |
|---|---|---|---|
| 1 | 0.96x | 1.04x | 0.97x |
| 2 | 1.09x | 1.01x | 0.97x |
| 4 | 1.10x | 0.95x | 0.97x |
| 8 | 1.07x | 0.97x | 0.97x |
| 16 | 1.04x | 0.98x | 0.97x |
| 32 | 0.92x | 0.96x | 0.95x |
| 64 | 0.88x | 0.95x | 0.94x |
| 128 | 0.85x | 0.94x | 0.93x |
| 256 | 0.80x | 0.94x | 0.93x |

## Key findings

1. **At TP2, torch.compile hurts all baselines at low batch sizes (small config).** The multinomial compiled baseline is up to 1.43x slower than eager at H=1. FlashInfer is up to 1.23x slower due to compiled `flashinfer_create_logits_and_indices`.

2. **The effect is much stronger for small config than large config.** At large config the overhead is smaller, and the crossover where compile starts winning happens at lower batch sizes.

3. **At high batch sizes (H>=64), compile wins for all providers.** The fused graph execution amortizes the overhead.

4. **FMMS Triton is the exception: compile helps at all batch sizes.** Disabling compile makes FMMS 8-22% slower because `_local_reduce` and `_stack_and_select_winner` are tiny functions where compile's fusion benefit outweighs its overhead.

5. **The multinomial eager baseline is unaffected** (as expected, since it never uses torch.compile). This serves as a control confirming the two runs are comparable.

## Conclusion

torch.compile overhead at TP2 is a fixed cost (~0.05-0.13ms) that dominates at low batch sizes. This does not affect FMMS Triton performance (its compiled functions are small enough to benefit), but it inflates baseline times, making FMMS appear relatively better than it would be against eager baselines.

For fair benchmarking at TP2, consider using multinomial eager as the primary baseline instead of multinomial compiled.
