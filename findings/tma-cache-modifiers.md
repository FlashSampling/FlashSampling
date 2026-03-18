# TMA loads do not support cache modifiers

## The idea

The FMMS kernel has an asymmetric reuse pattern:

- **Hidden states** (`[H, D]`): reused by every V tile. At H=1, D=8192: just 16 KB. At H=64: 1 MB. Should be kept warm in L2.
- **Weights** (`[V, D]`): each V tile is loaded once per H tile. With `H < BLOCK_SIZE_H` (typically 16-64), there is only 1 H tile, so each weight tile is loaded exactly once. No reuse until `H > BLOCK_SIZE_H`.

The ideal cache strategy would be:
- Weights: `eviction_policy="evict_first"` to stream through without polluting cache.
- Hidden states: `eviction_policy="evict_last"` to keep warm for reuse across V tiles.

## Why it doesn't work

The kernel uses TMA (`tl.make_tensor_descriptor` / `desc.load()`), which is hardware-accelerated async bulk copy on Hopper+. TMA compiles to `cp.async.bulk.tensor` PTX instructions, which use a different hardware path (the TMA unit) than regular loads (`ld.global` via the LD/ST unit).

Regular `tl.load()` supports cache modifiers (`.ca`, `.cg`, `.cv`) and eviction policies (`evict_first`, `evict_last`) because `ld.global` has those fields in the PTX ISA. TMA's `cp.async.bulk.tensor` does not, because it moves data directly from L2 into shared memory, bypassing L1 entirely. There is no L1 cache modifier to set.

In Triton's source (triton 3.6.0), `desc.load()` hardcodes both parameters:

```python
# triton/language/core.py
def load(self, offsets, _semantic=None):
    return _semantic.descriptor_load(self, offsets, "", "")
```

The underlying IR (`create_descriptor_load`) accepts cache_modifier and eviction_policy arguments, and `descriptor_gather` has `assert cache_modifier == "", "cache modifier is not supported yet"`, suggesting future support may be planned. But today it's a no-op.

## Does it matter in practice?

Probably not. The GROUP_SIZE_V=4 tile swizzling already provides the main L2 benefit: 4 consecutive V tiles share the same hidden_states, which is the most important reuse pattern.

For the decode case (H=1), hidden_states is 16 KB vs L2 capacity of 50 MB on H100. Even with fully random eviction, the probability of evicting a 16 KB working set while streaming 2 GB of weights is negligible (0.03% of L2).

At larger batch sizes (H=64+), hidden_states grows to ~1 MB but L2 is still 50 MB. And at those sizes, weight tiles start getting reused across H tiles, so `evict_first` would actually hurt.

## Alternatives considered

Switching from TMA to regular `tl.load()` just to get cache modifiers would likely be a net negative. TMA provides async hardware-prefetch that overlaps loads with computation (managed by `num_stages` pipelining). The L2 eviction benefit would not compensate for losing TMA's async copy pipeline.
