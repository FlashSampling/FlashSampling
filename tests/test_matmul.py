import torch

from fused_mm_sampling.tl_matmul import matmul


def test_matmul():
    device = torch.device("cuda")
    torch.manual_seed(0)
    M, N, K = 100, 200, 304  # noqa: N806  (K must be multiple of 8 for TMA)
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=device, dtype=torch.bfloat16)
    c = matmul(a, b)
    assert torch.allclose(c, a @ b.T, atol=0.01)
