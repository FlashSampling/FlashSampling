import pytest
import torch
import triton
import triton.language as tl

from fused_mm_sampling.tl_argsort import argsort


@triton.jit
def argsort_desc_kernel(x_ptr, out_vals_ptr, out_ids_ptr, N: tl.constexpr):  # noqa: N803
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    ids = offs.to(tl.int32)
    sorted_x, sorted_ids = argsort(x, ids, descending=True)
    tl.store(out_vals_ptr + offs, sorted_x)
    tl.store(out_ids_ptr + offs, sorted_ids)


@triton.jit
def argsort_asc_kernel(x_ptr, out_vals_ptr, out_ids_ptr, N: tl.constexpr):  # noqa: N803
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    ids = offs.to(tl.int32)
    sorted_x, sorted_ids = argsort(x, ids, descending=False)
    tl.store(out_vals_ptr + offs, sorted_x)
    tl.store(out_ids_ptr + offs, sorted_ids)


@triton.jit
def argsort_2d_kernel(
    x_ptr,
    out_vals_ptr,
    out_ids_ptr,
    stride_m,
    N: tl.constexpr,  # noqa: N803
    M: tl.constexpr,  # noqa: N803
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, N)
    x = tl.load(x_ptr + pid * stride_m + offs_n)
    ids = offs_n.to(tl.int32)
    sorted_x, sorted_ids = argsort(x, ids, descending=True)
    tl.store(out_vals_ptr + pid * stride_m + offs_n, sorted_x)
    tl.store(out_ids_ptr + pid * stride_m + offs_n, sorted_ids)


@pytest.mark.parametrize("N", [16, 64, 128, 256])
def test_argsort_descending(N):  # noqa: N803
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    out_vals = torch.empty_like(x)
    out_ids = torch.empty(N, dtype=torch.int32, device="cuda")

    argsort_desc_kernel[(1,)](x, out_vals, out_ids, N=N)

    expected_vals, expected_ids = torch.sort(x, descending=True)
    assert torch.allclose(out_vals, expected_vals)
    assert torch.equal(out_ids, expected_ids.to(torch.int32))


@pytest.mark.parametrize("N", [16, 64, 128, 256])
def test_argsort_ascending(N):  # noqa: N803
    x = torch.randn(N, dtype=torch.float32, device="cuda")
    out_vals = torch.empty_like(x)
    out_ids = torch.empty(N, dtype=torch.int32, device="cuda")

    argsort_asc_kernel[(1,)](x, out_vals, out_ids, N=N)

    expected_vals, expected_ids = torch.sort(x, descending=False)
    assert torch.allclose(out_vals, expected_vals)
    assert torch.equal(out_ids, expected_ids.to(torch.int32))


def test_argsort_2d():
    M, N = 4, 128  # noqa: N806
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")
    out_vals = torch.empty_like(x)
    out_ids = torch.empty(M, N, dtype=torch.int32, device="cuda")

    argsort_2d_kernel[(M,)](x, out_vals, out_ids, stride_m=N, N=N, M=M)

    expected_vals, expected_ids = torch.sort(x, dim=-1, descending=True)
    assert torch.allclose(out_vals, expected_vals)
    assert torch.equal(out_ids, expected_ids.to(torch.int32))
