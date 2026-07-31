"""Shared ordinary-GEMM benchmark dimensions, padding policy, and timing."""

MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
HIDDEN_STATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WARMUP_REPETITIONS = 25
BENCHMARK_REPETITIONS = 100
MAXIMUM_RATIO = 1.05
N_ALIGNMENT = 8


def pad_gemm_n(n: int) -> int:
    """Round the GEMM N dimension up to the BF16 TMA store alignment."""
    return ((n + N_ALIGNMENT - 1) // N_ALIGNMENT) * N_ALIGNMENT


def baseline_cases() -> list[dict]:
    """One row per (model shape, H, padding policy) torch.mm measurement.

    Whenever the padded GEMM N differs from the logical H (H=1, 2, 4), the
    case is measured both unpadded (matching the small-N GEMV specialization)
    and padded (matching the tensor-core GEMM path). Otherwise the single
    padded row covers both providers.
    """
    cases = []
    for vocab_size, hidden_size in MODEL_SHAPES:
        for n_hidden_states in HIDDEN_STATES:
            padded_n = pad_gemm_n(n_hidden_states)
            policies = (
                [("logical", n_hidden_states), ("padded", padded_n)]
                if padded_n != n_hidden_states
                else [("padded", padded_n)]
            )
            for padding, gemm_n in policies:
                cases.append(
                    {
                        "vocab_size": vocab_size,
                        "hidden_size": hidden_size,
                        "n_hidden_states": n_hidden_states,
                        "padding": padding,
                        "gemm_n": gemm_n,
                    }
                )
    return cases


def case_seed(vocab_size: int, hidden_size: int, n_hidden_states: int) -> int:
    """Deterministic per-case seed shared by the baseline and candidate runs."""
    return (
        vocab_size * 1_000_003 + hidden_size * 1_009 + n_hidden_states
    ) % (2**31 - 1)


def heuristic_problems() -> list[dict]:
    """The unique padded GEMM problems that gemm_problems_b200.json lists."""
    problems = []
    for vocab_size, hidden_size in MODEL_SHAPES:
        for gemm_n in sorted({pad_gemm_n(h) for h in HIDDEN_STATES}):
            problems.append(
                {"m": vocab_size, "n": gemm_n, "k": hidden_size}
            )
    return problems


def benchmark(function, flush_l2: bool = True) -> list[float]:
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
        if flush_l2:
            cache.zero_()
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]
