"""
Triton argsort: bitonic sort that returns both sorted values and original indices.

Adapted from https://github.com/triton-lang/triton/issues/3698 (hgl71964's implementation),
updated to match Triton's current internal sort API (hypercube reshape approach).
"""

import triton
import triton.language as tl
from triton.language import core
from triton.runtime.jit import constexpr_function


@triton.jit
def argsort(
    x,
    ids,
    dim: core.constexpr = None,
    descending: core.constexpr = core.CONSTEXPR_0,
):
    """Sort `x` along `dim` and apply the same permutation to `ids`.

    Args:
        x: values tensor (e.g. logits)
        ids: index tensor, same shape as x (e.g. vocab offsets)
        dim: dimension to sort along (only last dim supported)
        descending: sort order

    Returns:
        (sorted_x, permuted_ids)
    """
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(
        _dim == len(x.shape) - 1,
        "only minor dimension is currently supported",
    )
    log_n: core.constexpr = _log2(x.shape[_dim])
    n_dims: core.constexpr = _log2(x.numel)

    # Reshape to hypercube of shape [2, 2, ..., 2]
    hx = core.reshape(x, [2] * n_dims if n_dims else [1])
    hi = core.reshape(ids, [2] * n_dims if n_dims else [1])

    # Run only log_n stages (sort-axis size), not n_dims (total).
    # The alternating flip pattern creates independent sorting networks per row.
    for stage in core.static_range(1, log_n + 1):
        hx, hi = _bitonic_merge(hx, hi, stage, 2 if stage < log_n else descending, n_dims)

    x = core.reshape(hx, x.shape)
    ids = core.reshape(hi, ids.shape)
    return x, ids


@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, order: core.constexpr, n_dims: core.constexpr):
    if order == 2:
        flip = _indicator(n_dims, stage)
    else:
        flip = order
    for i in core.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, stage - 1 - i, n_dims)
    return x, ids


@triton.jit
def _compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    # Get the paired element via XOR along dimension i
    idtype = _get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ix = x.to(idtype, bitcast=True)
    iy = ix ^ tl.xor_sum(ix, n_dims - 1 - i, True)
    y = iy.to(x.dtype, bitcast=True)

    # Same XOR trick for indices
    iids = ids ^ tl.xor_sum(ids, n_dims - 1 - i, True)

    # Are we in the right (vs left) position along axis i?
    is_right = _indicator(n_dims, i)

    # Conditional swap: swap both values and indices together.
    # When values are tied (x == y), swapping values is a no-op, but swapping
    # ids would duplicate one id and lose the other. Only swap ids when values
    # actually differ.
    cond = (x > y) != (flip ^ is_right)
    ret = core.where(cond, y, x)
    swap_ids = cond & (x != y)
    ret_ids = core.where(swap_ids, iids, ids)
    return ret, ret_ids


# --- helpers (mirroring triton.language.standard internals) ---


@constexpr_function
def _log2(i):
    log2 = 0
    n = i
    while n > 1:
        n >>= 1
        log2 += 1
    return log2


_get_int_dtype = constexpr_function(core.get_int_dtype)


@triton.jit
def _indicator(n_dims: core.constexpr, j: core.constexpr):
    ar = core.arange(0, 2)
    ar = core.reshape(ar, [1] * (n_dims - j - 1) + [2] + [1] * j)
    return ar
