from typing import Optional

import torch
import triton
import triton.language as tl


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_N": bsz_v,
                "BLOCK_SIZE_K": bsz_d,
                "BLOCK_SIZE_M": bsz_h,
                "GROUP_SIZE_M": 4,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for bsz_v in [32, 4 * 32, 8 * 32]
        for bsz_d in [32, 64]
        for bsz_h in [16, 64, 128, 256]
        for num_warps in [8]  # Default 4
        for num_stages in [3]  # Higher values increase SRAM requirements, but 4 outperfomed 2.
    ],
    key=["M", "N", "K"],
    cache_results=True,
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,  # [M, K]
    b_ptr,  # [K, N] (pre-transposed by wrapper)
    c_ptr,
    # Matrix dimensions
    M,  # noqa: N803
    N,  # noqa: N803
    K,  # noqa: N803
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_N: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_K: tl.constexpr,  # noqa: N803
    GROUP_SIZE_M: tl.constexpr,  # noqa: N803
    ACTIVATION: tl.constexpr,  # noqa: N803
):
    """Kernel for computing the matmul C = A x B.T.
    A has shape (M, K), B is pre-transposed to (K, N), and C has shape (M, N).
    Uses device-side tensor descriptors (TMA) for efficient memory access.

    NOTE: tl.dot(a, b.T) does NOT work correctly with TMA-loaded blocks —
    .T only swaps the logical view without rearranging shared memory, but
    tensor core MMA instructions depend on physical layout. And TMA enforces
    strides[-1] == 1 so we can't describe the transpose via strides either.
    The wrapper pre-transposes B to [K, N] contiguous.
    """
    # -----------------------------------------------------------
    # Create device-side tensor descriptors for TMA
    # A: [M, K] row-major contiguous
    # B: [K, N] row-major contiguous (pre-transposed by wrapper)
    # C: [M, N] row-major contiguous
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # Process N dimension first, then M (matching fused kernel pattern).
    # This enables processing many N blocks for the same M block, allowing
    # A matrix (small M dimension) to be reused from L2 cache.
    pid_n = tl.program_id(axis=0)  # N dimension first (like vocab in fused kernel)
    pid_m = tl.program_id(axis=1)  # M dimension second (like hidden_states in fused kernel)

    # Swizzle for L2 cache optimization (N first to match fused kernel)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_n, pid_m = tl.swizzle2d(pid_n, pid_m, num_pid_n, num_pid_m, GROUP_SIZE_M)

    # Block starting offsets
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K
        # Load blocks using tensor descriptors (TMA handles bounds automatically)
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_k, offs_bn])
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)

    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.bfloat16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C using tensor descriptor
    offs_cm = pid_m * BLOCK_SIZE_M
    offs_cn = pid_n * BLOCK_SIZE_N
    c_desc.store([offs_cm, offs_cn], c)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape  # noqa: N806
    N, K = b.shape  # noqa: N806

    # TMA requires the innermost (stride-1) dimension to be aligned to 16 bytes.
    # For bfloat16 (2 bytes per element) that means multiples of 8 elements.
    tma_align = 16 // a.element_size()
    if K % tma_align != 0:
        raise ValueError(
            f"K={K} is not a multiple of {tma_align}. "
            f"TMA descriptors require the innermost dimension to be aligned to 16 bytes."
        )
    if N % tma_align != 0:
        raise ValueError(
            f"N={N} is not a multiple of {tma_align}. "
            f"TMA descriptors require the innermost dimension to be aligned to 16 bytes."
        )

    # Pre-transpose B: [N, K] -> [K, N] contiguous.
    # TMA enforces strides[-1]==1, so we can't describe the transpose via strides.
    b_t = b.T.contiguous()
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # 2D launch kernel with N first to match fused kernel pattern.
    # This enables processing many N blocks for the same M block,
    # allowing A matrix (small M dimension) to be reused from L2 cache.
    def grid(meta):
        return (
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
        )

    matmul_kernel[grid](
        a,
        b_t,
        c,
        M,  # noqa: N803
        N,  # noqa: N803
        K,  # noqa: N803
        ACTIVATION=activation,  #
    )
    return c


def get_cublas():
    from triton._C.libtriton import nvidia

    device_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(device_workspace)
    return cublas
