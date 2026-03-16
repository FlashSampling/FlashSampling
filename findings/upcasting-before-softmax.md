# Upcasting to float32 before softmax for torch.multinomial

## Problem

`torch.multinomial` produces incorrect sampling distributions when given bfloat16 probabilities. The root cause is precision loss during the cumulative sum (CDF) computation that `torch.multinomial` performs internally. With bfloat16's limited mantissa (7 bits vs float32's 23 bits), the CDF accumulates rounding errors that shift the effective sampling distribution.

## Evidence

Using a chi-squared goodness-of-fit test comparing 10,000 empirical samples against theoretical softmax probabilities (V=256, temperature=5.0):

- **Without upcasting** (bfloat16 softmax): chi-squared statistic ~323, p-value < 0.001 (fails)
- **With upcasting** (float32 softmax): chi-squared statistic ~22, p-value > 0.001 (passes)

## Fix

Upcast logits to float32 before computing softmax:

```python
# Before (incorrect distributions):
probs = (logits / temperature).softmax(dim=1)

# After (correct distributions):
probs = (logits.float() / temperature).softmax(dim=1)
```

This is applied in `src/fused_mm_sampling/core.py` in the `sample()` function.

## Discussion

- PyTorch docs for `torch.multinomial` do not mention this precision requirement
- The issue affects any code path that feeds bfloat16 probabilities to `torch.multinomial`
- vLLM and other inference frameworks apply similar upcasting before sampling
- Related: https://github.com/pytorch/pytorch/issues/100124
