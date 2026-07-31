"""Shared ordinary-GEMM benchmark dimensions and timing helper."""

MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
HIDDEN_STATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WARMUP_REPETITIONS = 25
BENCHMARK_REPETITIONS = 100
MAXIMUM_RATIO = 1.05


def benchmark(function) -> list[float]:
    import torch

    cache = torch.empty(
        256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda"
    )
    for _ in range(WARMUP_REPETITIONS):
        function()
    torch.cuda.synchronize()
    starts = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(BENCHMARK_REPETITIONS)
    ]
    ends = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(BENCHMARK_REPETITIONS)
    ]
    for start, end in zip(starts, ends):
        cache.zero_()
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]
