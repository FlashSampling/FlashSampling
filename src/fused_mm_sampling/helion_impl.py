"""
Helion implementation for benchmarking purposes.
This module is separate from core to avoid making helion a required dependency.
It is only used for performance comparisons.
"""

import os
from pathlib import Path

# Since the helion kernel samples (is stochastic), we should not verify exactness.
os.environ["HELION_AUTOTUNE_ACCURACY_CHECK"] = "0"
# Store autotuned configs in this repo so they persist across machines and /tmp cleanups.
os.environ.setdefault(
    "HELION_CACHE_DIR", str(Path(__file__).resolve().parent.parent.parent / "helion-cache")
)

import helion
import helion.language as hl
import torch

BLOCK_SIZE_V = 128


# autotune_effort: "none" / "quick" / "full". Override via HELION_AUTOTUNE_EFFORT env var.
@helion.kernel(autotune_effort=os.environ.get("HELION_AUTOTUNE_EFFORT", "quick"))
def fused_sample_helion_kernel(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [D, H]
    tile_maxs: torch.Tensor,  # [n_tiles, H], initialized to -inf
    tile_max_idxs: torch.Tensor,  # [n_tiles, H], output token indices per tile
    temperature: torch.Tensor,  # [1]
    seed: int,
):
    """Stage 1: each (V, H) tile computes its local max and argmax in parallel."""
    assert weights.size(1) == hidden_states.size(0)
    V, D = weights.size()  # noqa: N806
    H = hidden_states.size(1)  # noqa: N806

    for tile_v, tile_h in hl.tile([V, H], block_size=[BLOCK_SIZE_V, None]):
        logits_blk = hl.zeros([tile_v, tile_h], dtype=torch.float32)
        for tile_d in hl.tile(D):
            mm = torch.matmul(weights[tile_v, tile_d], hidden_states[tile_d, tile_h])
            logits_blk = logits_blk + mm
        logits_blk = logits_blk / temperature[0]

        unif_noise = hl.rand([tile_v, tile_h], seed=seed)
        gumbel_noise = -(-unif_noise.log()).log()
        summed = logits_blk + gumbel_noise

        tile_maxs[tile_v.id, tile_h] = hl.reduce(torch.max, summed, dim=0, other=float("-inf"))
        # torch.argmax in Helion returns global indices (includes tile offset).
        tile_max_idxs[tile_v.id, tile_h] = torch.argmax(summed, dim=0)


def fused_mm_sample_helion(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [H, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
) -> torch.Tensor:
    temperature = temperature.reshape(1)  # Helion kernel needs 1D tensor for indexing
    V = weights.size(0)  # noqa: N806
    H = hidden_states.size(0)  # noqa: N806
    n_tiles = helion.cdiv(V, BLOCK_SIZE_V)
    hs_t = hidden_states.T.contiguous()  # [D, H]
    results = []
    for i in range(num_samples):
        tile_maxs = torch.full((n_tiles, H), float("-inf"), device=weights.device)
        tile_max_idxs = torch.empty((n_tiles, H), dtype=torch.long, device=weights.device)
        seed = torch.randint(0, 2**31, (1,)).item() + i
        fused_sample_helion_kernel(weights, hs_t, tile_maxs, tile_max_idxs, temperature, seed)
        # Stage 2: reduce across tiles
        best_tiles = tile_maxs.argmax(dim=0)  # [H]
        sample_idx = tile_max_idxs.gather(dim=0, index=best_tiles.unsqueeze(0)).squeeze(0)
        results.append(sample_idx)
    return torch.stack(results, dim=1)  # [H, num_samples]


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty([m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out
