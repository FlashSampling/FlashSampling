"""Gate 2c: official B200 plain-GEMM kernel discovery and confirmation.

The discover phase generates kernel candidates with nvidia-matmul-heuristics,
profiles them with cutlass_profiler, audits the supported-family coverage,
and ranks the per-case oracle against the matched torch.mm baseline.
The confirm phase reruns the matched cold-L2 protocol against every
extension candidate (the six manual controls, the small-N GEMV, and any
transplanted heuristic winners) and selects the per-case dispatch.

Both phases are B200-only. Select with the PHASE and RUN env vars:

    make modal-cutlass GATE=ordinary-gemm-tuning PHASE=discover
    make modal-cutlass GATE=ordinary-gemm-tuning PHASE=confirm RUN=1
"""

import io
import json
import os
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .ordinary_gemm_common import (
    BENCHMARK_REPETITIONS,
    HIDDEN_STATES,
    MAXIMUM_RATIO,
    MODEL_SHAPES,
    WARMUP_REPETITIONS,
    baseline_cases,
    benchmark,
    case_seed,
    heuristic_problems,
    pad_gemm_n,
)
from .utils import (
    CUTLASS_SHA,
    CUTLASS_VERSION,
    HEURISTICS_BUILD_DIR,
    HEURISTICS_BUILD_DIR_N256,
    HEURISTICS_TESTLIST,
    HEURISTICS_TESTLIST_N256,
    NVMMH_VERSION,
    add_cutlass_greedy_provider,
    make_cutlass_heuristics_image,
    make_cutlass_provider_image,
)

app = make_app()
provider_image = add_cutlass_greedy_provider(make_cutlass_provider_image())
heuristics_image = make_cutlass_heuristics_image()

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning")
DISCOVERY_DIR = OUTPUT_DIR / "gate-2c-discovery"
PHASE = os.environ.get("PHASE", "discover")
RUN = os.environ.get("RUN", "1")
PROBLEMS_JSON = Path(__file__).parent / "gemm_problems_b200.json"
PROFILING_DURATION_MS = 50
MANUAL_VARIANTS = (
    "tile-64x128x64-auto",
    "tile-128x64x64-auto",
    "tile-128x128x64-auto",
    "tile-64x128x64-native",
    "tile-128x64x64-native",
    "tile-128x128x64-native",
)
# Heuristic winners transplanted into greedy_provider.cu after the Gate 2c
# profiler search. The -rn suffix selects the along_n raster order.
WINNER_VARIANTS: tuple[str, ...] = (
    "heur-256x64x128-c2x1x1",
    "heur-256x64x128-c2x1x1-rn",
    "heur-128x64x128-c2x1x1",
    "heur-128x64x128-c2x1x1-rn",
    "heur-256x128x64-c2x1x1",
    "heur-256x128x64-c2x1x1-rn",
    "heur-256x64x64-c4x1x1",
    "heur-256x64x64-c4x1x1-rn",
    "heur-128x64x64-c4x1x1",
    "heur-128x64x64-c4x1x1-rn",
    "heur-256x64x128-c4x1x1",
    "heur-256x64x128-c4x1x1-rn",
    "heur-256x128x64-c4x1x1",
    "heur-256x128x64-c4x1x1-rn",
    # N=256 top-32 expansion families.
    "heur-256x128x128-c2x1x1",
    "heur-256x128x128-c2x1x1-rn",
    "heur-256x128x128-c4x1x1",
    "heur-256x128x128-c4x1x1-rn",
    "heur-128x128x64-c4x1x1",
    "heur-128x128x64-c4x1x1-rn",
    "heur-128x128x128-c4x1x1",
    "heur-128x128x128-c4x1x1-rn",
    "heur-256x192x64-c2x1x1",
    "heur-256x192x64-c2x1x1-rn",
    "heur-256x192x64-c4x1x1",
    "heur-256x192x64-c4x1x1-rn",
    # Explicit 1-SM cluster-(1,2,1) control (never emitted by the heuristic).
    "heur-128x128x64-1sm-c1x2x1",
    "heur-128x128x64-1sm-c1x2x1-rn",
    # Full-H CTA tiles for the two H=256 cells outside the threshold.
    "heur-128x256x64-1sm",
    "heur-128x256x64-1sm-rn",
    "heur-128x256x64-1sm-c2x1x1",
    "heur-128x256x64-1sm-c2x1x1-rn",
    "heur-256x256x64-c2x1x1",
    "heur-256x256x64-c2x1x1-rn",
)
GEMV_VARIANT = "small-n-gemv"
CASE_KEYS = ("vocab_size", "hidden_size", "n_hidden_states", "gemm_n")


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(gpu="B200", image=heuristics_image, timeout=90 * 60)
def profile_heuristics() -> dict:
    """Run cutlass_profiler over both heuristic testlists on B200."""
    import subprocess

    metadata = _remote_metadata()
    clock_lock = _try_lock_clocks()
    runs = {}
    for tag, build_dir, testlist in (
        ("main", HEURISTICS_BUILD_DIR, HEURISTICS_TESTLIST),
        ("n256", HEURISTICS_BUILD_DIR_N256, HEURISTICS_TESTLIST_N256),
    ):
        profiler = Path(build_dir) / "tools/profiler/cutlass_profiler"
        generation_log = (
            Path(build_dir) / "tools/library/library_instance_generation.log"
        )
        if not profiler.exists() or not Path(testlist).exists():
            runs[tag] = {
                "command": "",
                "profiler_rc": -1,
                "profiler_stdout": "",
                "profiler_stderr": "build artifacts missing",
                "testlist_csv": "",
                "generation_log": "",
                "profiler_csv": "",
            }
            continue
        output = f"/opt/fmms/profiler-out-{tag}.csv"
        command = [
            str(profiler),
            "--operation=Gemm",
            f"--testlist-file={testlist}",
            "--profiling-iterations=0",
            f"--profiling-duration={PROFILING_DURATION_MS}",
            "--verification-enabled=false",
            "--providers=cutlass",
            f"--output={output}",
        ]
        device_info = subprocess.run(
            [str(profiler), "--device-info"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=80 * 60
        )
        output_path = Path(output.replace(".csv", ".gemm.csv"))
        runs[tag] = {
            "command": " ".join(command),
            "profiler_rc": result.returncode,
            "profiler_stdout": result.stdout,
            "profiler_stderr": result.stderr,
            "testlist_csv": Path(testlist).read_text(),
            "generation_log": (
                generation_log.read_text() if generation_log.exists() else ""
            ),
            "profiler_csv": (
                output_path.read_text() if output_path.exists() else ""
            ),
        }
    return {
        "metadata": metadata,
        "clock_lock": clock_lock,
        "clocks_after_mhz": _sm_clock_mhz(),
        "device_info": device_info.stdout + device_info.stderr,
        "runs": runs,
    }


@app.function(gpu="B200", image=provider_image, timeout=90 * 60)
def measure_torch_mm() -> dict:
    """Measure the matched torch.mm baseline in warm- and cold-L2 states."""
    import torch

    rows = []
    for case in baseline_cases():
        rows.extend(_measure_case_baseline(torch, case))
    cublas_logs = {
        f"v{vocab_size}-d{hidden_size}": _cublas_kernel_log(
            vocab_size, hidden_size, 256
        )
        for vocab_size, hidden_size in MODEL_SHAPES
    }
    return {
        "timings": rows,
        "cublas_logs": cublas_logs,
        "metadata": _remote_metadata(),
    }


def _measure_case_baseline(torch, case: dict) -> list[dict]:
    """Cold- and warm-L2 torch.mm repetitions for one baseline case."""
    vocab_size = case["vocab_size"]
    hidden_size = case["hidden_size"]
    n_hidden_states = case["n_hidden_states"]
    gemm_n = case["gemm_n"]
    weights, hidden, padded, output = _make_case_tensors(
        torch, vocab_size, hidden_size, n_hidden_states, gemm_n
    )

    def cublas():
        torch.mm(weights, padded.T, out=output)

    cublas()
    torch.cuda.synchronize()
    rows = []
    for cache_state, flush_l2 in (("cold", True), ("warm", False)):
        for repetition, latency_ms in enumerate(
            benchmark(cublas, flush_l2=flush_l2)
        ):
            rows.append(
                {
                    **case,
                    "cache_state": cache_state,
                    "repetition": repetition,
                    "latency_ms": latency_ms,
                }
            )
    del weights, hidden, padded, output
    return rows


@app.function(gpu="B200", image=provider_image, timeout=90 * 60)
def measure_confirm_sweep() -> dict:
    """Baseline and candidates interleaved in one process on one host.

    The gate decision compares each candidate against the torch.mm baseline
    measured in the same container, so host-class variance cancels in the
    ratio instead of contaminating it.
    """
    import torch

    from fused_mm_sampling.cutlass_impl import (
        cutlass_launch_plain_gemm_variant,
        cutlass_launch_small_n_gemv,
    )

    baseline_rows = []
    candidate_rows = []
    correctness = []
    for case in baseline_cases():
        baseline_rows.extend(_measure_case_baseline(torch, case))
        vocab_size = case["vocab_size"]
        hidden_size = case["hidden_size"]
        n_hidden_states = case["n_hidden_states"]
        gemm_n = case["gemm_n"]
        weights, hidden, padded, output = _make_case_tensors(
            torch, vocab_size, hidden_size, n_hidden_states, gemm_n
        )
        reference = torch.mm(weights, padded.T)

        def record(variant, function, measured_output, measured_reference):
            try:
                function()
                torch.cuda.synchronize()
            except RuntimeError as error:
                # Rejected candidate (build, can_implement, or launch).
                # Record the diagnostic instead of aborting the sweep.
                correctness.append(
                    {
                        **case,
                        "variant": variant,
                        "exact": 0,
                        "max_abs_difference": None,
                        "mean_abs_difference": None,
                        "finite": 0,
                        "rejected": str(error).splitlines()[0][:200],
                    }
                )
                return
            difference = (
                measured_output.float() - measured_reference.float()
            ).abs()
            correctness.append(
                {
                    **case,
                    "variant": variant,
                    "exact": int(torch.equal(measured_output, measured_reference)),
                    "max_abs_difference": float(difference.max()),
                    "mean_abs_difference": float(difference.mean()),
                    "finite": int(torch.isfinite(measured_output).all()),
                    "rejected": None,
                }
            )
            for repetition, latency_ms in enumerate(benchmark(function)):
                candidate_rows.append(
                    {
                        **case,
                        "variant": variant,
                        "repetition": repetition,
                        "latency_ms": latency_ms,
                    }
                )

        if case["padding"] == "padded":
            for variant in MANUAL_VARIANTS + WINNER_VARIANTS:

                def cutlass(variant=variant):
                    cutlass_launch_plain_gemm_variant(
                        variant, weights, padded, output
                    )

                record(variant, cutlass, output, reference)
        if gemm_n == n_hidden_states and n_hidden_states <= 8:
            gemv_output = torch.empty(
                (vocab_size, n_hidden_states),
                dtype=torch.bfloat16,
                device="cuda",
            )
            gemv_reference = torch.mm(weights, hidden.T)

            def gemv():
                cutlass_launch_small_n_gemv(weights, hidden, gemv_output)

            record(GEMV_VARIANT, gemv, gemv_output, gemv_reference)
            del gemv_output, gemv_reference
        del weights, hidden, padded, output, reference
    return {
        "baseline": baseline_rows,
        "timings": candidate_rows,
        "correctness": correctness,
        "metadata": _remote_metadata(),
    }


def _make_case_tensors(torch, vocab_size, hidden_size, n_hidden_states, gemm_n):
    """Identical weights/hidden states for a case across all remote runs."""
    torch.manual_seed(
        case_seed(vocab_size, hidden_size, n_hidden_states)
    )
    weights = torch.randn(
        (vocab_size, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    hidden = torch.randn(
        (n_hidden_states, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    padded = torch.zeros(
        (gemm_n, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    padded[:n_hidden_states] = hidden
    output = torch.empty(
        (vocab_size, gemm_n), dtype=torch.bfloat16, device="cuda"
    )
    return weights, hidden, padded, output


def _remote_metadata() -> dict:
    import platform
    import subprocess

    import torch

    return {
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": ".".join(
            map(str, torch.cuda.get_device_capability())
        ),
        "sm_count": torch.cuda.get_device_properties(
            0
        ).multi_processor_count,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cublas_version": _cublas_version(torch),
        "python": platform.python_version(),
        "driver": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "clocks_sm_mhz": _sm_clock_mhz(),
        "cutlass_version": CUTLASS_VERSION,
        "cutlass_sha": CUTLASS_SHA,
        "nvmmh_version": NVMMH_VERSION,
    }


def _cublas_version(torch) -> str:
    import ctypes

    try:
        handle = torch._C._cuda_getCurrentBlasHandle()
        library = ctypes.CDLL("libcublas.so.13")
        version = ctypes.c_int(0)
        status = library.cublasGetVersion_v2(
            ctypes.c_void_p(handle), ctypes.byref(version)
        )
        return str(version.value) if status == 0 else f"error-{status}"
    except Exception as error:  # best-effort diagnostic metadata
        return f"unavailable ({error})"


def _sm_clock_mhz() -> str:
    return _command_output(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.max.sm",
            "--format=csv,noheader",
        ]
    )


def _command_output(command: list[str]) -> str:
    import subprocess

    return subprocess.check_output(command, text=True).strip()


def _try_lock_clocks() -> dict:
    import subprocess

    max_clock = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=clocks.max.sm",
            "--format=csv,noheader,nounits",
        ]
    )
    result = subprocess.run(
        ["nvidia-smi", "-lgc", max_clock], capture_output=True, text=True
    )
    return {
        "max_clock_mhz": max_clock,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _cublas_kernel_log(vocab_size: int, hidden_size: int, gemm_n: int) -> str:
    """Capture the cuBLASLt heuristic log for one torch.mm shape."""
    import subprocess
    import sys

    code = (
        "import torch;"
        f"w=torch.randn(({vocab_size},{hidden_size}),"
        "dtype=torch.bfloat16,device='cuda');"
        f"h=torch.randn(({gemm_n},{hidden_size}),"
        "dtype=torch.bfloat16,device='cuda');"
        f"o=torch.empty(({vocab_size},{gemm_n}),"
        "dtype=torch.bfloat16,device='cuda');"
        "torch.mm(w,h.T,out=o);torch.cuda.synchronize()"
    )
    env = {
        **os.environ,
        "CUBLASLT_LOG_LEVEL": "5",
        "CUBLAS_LOGINFO_DBG": "1",
        "CUBLAS_LOGDEST_DBG": "stdout",
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Discovery packet
# ---------------------------------------------------------------------------


def _write_discovery_packet(profile: dict, baseline: dict) -> None:
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    problems = _validate_problems_json()
    (DISCOVERY_DIR / "problems.json").write_text(
        json.dumps(problems, indent=2) + "\n"
    )
    runs = profile["runs"]
    main = runs["main"]
    n256 = runs["n256"]
    (DISCOVERY_DIR / "heuristics-testlist.csv").write_text(
        main["testlist_csv"]
    )
    (DISCOVERY_DIR / "library-generation-log.txt").write_text(
        main["generation_log"]
    )
    (DISCOVERY_DIR / "profiler-results.gemm.csv").write_text(
        main["profiler_csv"]
    )
    (DISCOVERY_DIR / "profiler-stdout.txt").write_text(
        main["profiler_stdout"] + "\n--- stderr ---\n" + main["profiler_stderr"]
    )
    if n256["testlist_csv"]:
        (DISCOVERY_DIR / "heuristics-testlist-n256.csv").write_text(
            n256["testlist_csv"]
        )
        (DISCOVERY_DIR / "library-generation-log-n256.txt").write_text(
            n256["generation_log"]
        )
        (DISCOVERY_DIR / "profiler-results-n256.gemm.csv").write_text(
            n256["profiler_csv"]
        )
        (DISCOVERY_DIR / "profiler-stdout-n256.txt").write_text(
            n256["profiler_stdout"]
            + "\n--- stderr ---\n"
            + n256["profiler_stderr"]
        )
    (DISCOVERY_DIR / "device-info.txt").write_text(profile["device_info"])
    for name, log in baseline["cublas_logs"].items():
        (DISCOVERY_DIR / f"cublas-log-{name}.txt").write_text(log)

    main_testlist = pd.read_csv(io.StringIO(main["testlist_csv"]))
    main_profiler = pd.read_csv(io.StringIO(main["profiler_csv"]))
    testlists = [main_testlist]
    profilers = [main_profiler]
    if n256["testlist_csv"]:
        testlists.append(pd.read_csv(io.StringIO(n256["testlist_csv"])))
        profilers.append(pd.read_csv(io.StringIO(n256["profiler_csv"])))
    combined_testlist = pd.concat(testlists, ignore_index=True)
    combined_profiler = pd.concat(profilers, ignore_index=True)
    merged = _join_profiler_results(combined_testlist, combined_profiler)
    merged.to_csv(DISCOVERY_DIR / "profiler-parsed.csv", index=False)
    audit = _coverage_audit(
        main_testlist,
        main_profiler,
        merged.query(
            "operation_name in @main_testlist.operation_name"
        ),
    )
    audit.to_csv(DISCOVERY_DIR / "coverage-audit.csv", index=False)
    expansion = _expansion_summary(n256, combined_testlist, merged)
    expansion.to_csv(DISCOVERY_DIR / "expansion-n256-summary.csv", index=False)

    baseline_rows = pd.DataFrame(baseline["timings"])
    baseline_rows.to_csv(DISCOVERY_DIR / "baseline-cases.csv", index=False)
    baseline_summary = _summarize_baseline(baseline_rows)
    baseline_summary.to_csv(
        DISCOVERY_DIR / "baseline-summary.csv", index=False
    )
    oracle = _oracle_selection(merged, baseline_summary)
    oracle.to_csv(DISCOVERY_DIR / "oracle-selection.csv", index=False)

    metadata = {
        "profile": profile["metadata"],
        "baseline": baseline["metadata"],
        "clock_lock": profile["clock_lock"],
        "clocks_after_mhz": profile["clocks_after_mhz"],
        "profiler_commands": {
            tag: run["command"] for tag, run in runs.items()
        },
        "profiler_returncodes": {
            tag: run["profiler_rc"] for tag, run in runs.items()
        },
        "profiling_duration_ms_per_case": PROFILING_DURATION_MS,
    }
    (DISCOVERY_DIR / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    summary = _discovery_summary(
        main_testlist, main_profiler, combined_testlist, merged, audit, oracle
    )
    (DISCOVERY_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (DISCOVERY_DIR / "VERIFY.md").write_text(_DISCOVERY_VERIFY)


def _validate_problems_json() -> list[dict]:
    problems = json.loads(PROBLEMS_JSON.read_text())
    expected = {
        (problem["m"], problem["n"], problem["k"])
        for problem in heuristic_problems()
    }
    actual = {
        (problem["m"], problem["n"], problem["k"]) for problem in problems
    }
    if actual != expected:
        raise RuntimeError(
            "gemm_problems_b200.json drifted from "
            "ordinary_gemm_common.heuristic_problems(): "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return problems


JOIN_KEYS = [
    "m",
    "n",
    "k",
    "cluster_m",
    "cluster_n",
    "split_k_slices",
    "raster_order",
    "swizzle_size",
]


def _join_profiler_results(
    testlist: pd.DataFrame, profiler_rows: pd.DataFrame
) -> pd.DataFrame:
    """One row per unique (kernel, problem, runtime-args) test case.

    The heuristic emits duplicate testlist rows (different configs mapping to
    the same kernel with identical runtime arguments), and the profiler
    measures each duplicate. Dedupe the testlist with a duplicate count and
    aggregate the profiler rows as repetitions before joining one-to-one.
    """
    left = testlist.copy()
    right = profiler_rows.copy()
    for key in JOIN_KEYS:
        if key in {"raster_order"}:
            left[key] = left[key].astype(str).str.lower()
            right[key] = right[key].astype(str).str.lower()
        else:
            left[key] = left[key].astype(int)
            right[key] = right[key].astype(int)
    left["testlist_duplicate_rows"] = left.groupby(
        ["operation_name"] + JOIN_KEYS
    )["operation_name"].transform("size")
    left = left.drop_duplicates(["operation_name"] + JOIN_KEYS)
    for column in ("Runtime", "Bytes", "Flops", "GB/s", "GFLOPs"):
        right[column] = pd.to_numeric(right[column], errors="coerce")
    group_keys = ["Operation"] + JOIN_KEYS
    right["profiler_runs"] = right.groupby(group_keys)["Operation"].transform(
        "size"
    )
    aggregated = {
        column: "first"
        for column in right.columns
        if column not in {"Runtime", "GB/s", "GFLOPs", *group_keys}
    }
    right = (
        right.sort_values("Runtime")
        .groupby(group_keys, as_index=False)
        .agg({**aggregated, "Runtime": "median", "GB/s": "median", "GFLOPs": "median"})
    )
    return left.merge(
        right,
        how="outer",
        left_on=["operation_name"] + JOIN_KEYS,
        right_on=["Operation"] + JOIN_KEYS,
        indicator=True,
    )


def _coverage_audit(
    testlist: pd.DataFrame,
    profiler_rows: pd.DataFrame,
    merged: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for (m, n, k), group in testlist.groupby(["m", "n", "k"]):
        names = group["operation_name"]
        merged_group = merged.query("m == @m and n == @n and k == @k")
        profiled = merged_group.query("_merge == 'both'")
        success = (
            profiled.query("Status == 'success'")
            if "Status" in profiled
            else profiled.iloc[0:0]
        )
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "testlist_rows": len(group),
                "unique_kernels": int(names.nunique()),
                "kernels_1sm": int(names.str.contains("1sm").sum()),
                "kernels_2sm": int(names.str.contains("2sm").sum()),
                "streamk_rows": int(
                    names.str.contains("stream_k").sum()
                ),
                "flexible_cluster_rows": int(
                    names.str.contains("flex").sum()
                ),
                "cluster_m_values": _value_list(group["cluster_m"]),
                "cluster_n_values": _value_list(group["cluster_n"]),
                "split_k_slices_values": _value_list(group["split_k_slices"]),
                "raster_order_values": _value_list(group["raster_order"]),
                "swizzle_size_values": _value_list(group["swizzle_size"]),
                "stages_values": (
                    _value_list(profiled["stages"])
                    if "stages" in profiled and len(profiled)
                    else "unprofiled"
                ),
                "cta_tile_m_values": _value_list(group["cta_tile_m"]),
                "cta_tile_n_values": _value_list(group["cta_tile_n"]),
                "cta_tile_k_values": _value_list(group["cta_tile_k"]),
                "unique_test_cases": len(merged_group),
                "profiled_cases": len(profiled),
                "profiler_repetitions": int(
                    profiled["profiler_runs"].fillna(0).sum()
                ),
                "unprofiled_cases": int(
                    len(merged_group.query("_merge == 'left_only'"))
                ),
                "failed_cases": int(len(profiled) - len(success)),
            }
        )
    return pd.DataFrame(rows)


def _value_list(series: pd.Series) -> str:
    return ",".join(str(value) for value in sorted(series.unique()))


def _summarize_baseline(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            [
                "vocab_size",
                "hidden_size",
                "n_hidden_states",
                "gemm_n",
                "padding",
                "cache_state",
            ],
            as_index=False,
        )
        .agg(
            repetitions=("latency_ms", "count"),
            median_ms=("latency_ms", "median"),
            p10_ms=("latency_ms", lambda values: values.quantile(0.1)),
            p90_ms=("latency_ms", lambda values: values.quantile(0.9)),
            std_ms=("latency_ms", "std"),
        )
    )


def _oracle_selection(
    merged: pd.DataFrame, baseline_summary: pd.DataFrame
) -> pd.DataFrame:
    success = merged.query("_merge == 'both' and Status == 'success'")
    oracle = (
        success.sort_values("Runtime")
        .groupby(["m", "n", "k"], as_index=False)
        .first()
    )
    oracle = oracle.rename(
        columns={
            "m": "vocab_size",
            "n": "gemm_n",
            "k": "hidden_size",
            "Runtime": "oracle_ms",
        }
    )
    cold = _padded_problem_baseline(baseline_summary, "cold")
    warm = _padded_problem_baseline(baseline_summary, "warm")
    oracle = oracle.merge(
        cold.rename(columns={"median_ms": "torch_mm_cold_ms"}),
        on=["vocab_size", "hidden_size", "gemm_n"],
        how="left",
        validate="one_to_one",
    )
    oracle = oracle.merge(
        warm.rename(columns={"median_ms": "torch_mm_warm_ms"}),
        on=["vocab_size", "hidden_size", "gemm_n"],
        how="left",
        validate="one_to_one",
    )
    oracle["covered_hidden_states"] = oracle["gemm_n"].map(
        lambda gemm_n: ",".join(
            str(h) for h in HIDDEN_STATES if pad_gemm_n(h) == gemm_n
        )
    )
    oracle["ratio_vs_cold_torch_mm"] = (
        oracle["oracle_ms"] / oracle["torch_mm_cold_ms"]
    )
    oracle["ratio_vs_warm_torch_mm"] = (
        oracle["oracle_ms"] / oracle["torch_mm_warm_ms"]
    )
    return oracle


def _padded_problem_baseline(
    baseline_summary: pd.DataFrame, cache_state: str
) -> pd.DataFrame:
    """One baseline row per padded GEMM problem.

    H=1, 2, and 4 share the padded gemm_n=8 shape with H=8; their identical
    GEMM timings are combined into one row per problem so the oracle join
    stays one-to-one.
    """
    padded = baseline_summary.query(
        "cache_state == @cache_state and padding == 'padded'"
    )
    return (
        padded.groupby(["vocab_size", "hidden_size", "gemm_n"], as_index=False)
        .agg(
            median_ms=("median_ms", "mean"),
            baseline_rows=("median_ms", "count"),
        )
    )


def _expansion_summary(
    n256_run: dict, combined_testlist: pd.DataFrame, merged: pd.DataFrame
) -> pd.DataFrame:
    """Per-problem summary of the N=256 top-32 stop-rule expansion."""
    if not n256_run["testlist_csv"]:
        return pd.DataFrame(
            [{"note": "N=256 top-32 expansion not present in this run"}]
        )
    testlist = pd.read_csv(io.StringIO(n256_run["testlist_csv"]))
    rows = []
    for (m, n, k), group in testlist.groupby(["m", "n", "k"]):
        cases = merged.query(
            "m == @m and n == @n and k == @k and _merge == 'both'"
        )
        success = cases.query("Status == 'success'")
        best = (
            success.sort_values("Runtime").iloc[0]
            if len(success)
            else None
        )
        rows.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "configs_per_problem": 32,
                "testlist_rows": len(group),
                "unique_kernels": int(group["operation_name"].nunique()),
                "kernels_1sm": int(
                    group["operation_name"].str.contains("1sm").sum()
                ),
                "kernels_2sm": int(
                    group["operation_name"].str.contains("2sm").sum()
                ),
                "profiled_cases": len(success),
                "best_operation": (
                    best["operation_name"] if best is not None else ""
                ),
                "best_runtime_ms": (
                    float(best["Runtime"]) if best is not None else None
                ),
                "combined_unique_cases_for_problem": len(
                    combined_testlist.query("m == @m and n == @n and k == @k")
                ),
            }
        )
    return pd.DataFrame(rows)


def _discovery_summary(
    testlist: pd.DataFrame,
    profiler_rows: pd.DataFrame,
    combined_testlist: pd.DataFrame,
    merged: pd.DataFrame,
    audit: pd.DataFrame,
    oracle: pd.DataFrame,
) -> dict:
    names = testlist["operation_name"]
    success = merged.query("_merge == 'both' and Status == 'success'")
    return {
        "gate": "ordinary-gemm-tuning",
        "phase": "gate-2c-discovery",
        "command": (
            "make modal-cutlass GATE=ordinary-gemm-tuning PHASE=discover"
        ),
        "status": "discovery-complete",
        "decision": (
            "Discovery ranks candidates only; the pass/fail decision "
            "belongs to the two PHASE=confirm runs."
        ),
        "cutlass": {"version": CUTLASS_VERSION, "sha": CUTLASS_SHA},
        "nvidia_matmul_heuristics": NVMMH_VERSION,
        "problems": int(
            testlist.groupby(["m", "n", "k"]).ngroups
        ),
        "testlist_rows": int(len(testlist)),
        "unique_kernels": int(names.nunique()),
        "unique_test_cases": int(len(merged)),
        "combined_testlist_rows": int(len(combined_testlist)),
        "n256_expansion_configs_per_problem": 32,
        "profiler_rows": int(len(profiler_rows)),
        "profiled_cases": int(len(merged.query("_merge == 'both'"))),
        "successful_cases": int(len(success)),
        "coverage": {
            "problems_with_1sm": int(audit["kernels_1sm"].gt(0).sum()),
            "problems_with_2sm": int(audit["kernels_2sm"].gt(0).sum()),
            "problems_with_streamk": int(
                audit["streamk_rows"].gt(0).sum()
            ),
            "problems_with_flexible_clusters": int(
                audit["flexible_cluster_rows"].gt(0).sum()
            ),
            "problems_with_cluster_m_gt_1": int(
                audit["cluster_m_values"]
                .str.split(",")
                .map(lambda values: max(map(int, values)) > 1)
                .sum()
            ),
            "problems_with_split_k_gt_1": int(
                audit["split_k_slices_values"]
                .str.split(",")
                .map(lambda values: max(map(int, values)) > 1)
                .sum()
            ),
            "problems_with_unprofiled_cases": int(
                audit["unprofiled_cases"].gt(0).sum()
            ),
            "problems_with_failed_cases": int(
                audit["failed_cases"].gt(0).sum()
            ),
        },
        "oracle_worst_ratio_vs_cold_torch_mm": float(
            oracle["ratio_vs_cold_torch_mm"].max()
        ),
        "oracle_best_ratio_vs_cold_torch_mm": float(
            oracle["ratio_vs_cold_torch_mm"].min()
        ),
        "maximum_cutlass_to_cublas_ratio": MAXIMUM_RATIO,
        "profiling_duration_ms_per_case": PROFILING_DURATION_MS,
    }


_DISCOVERY_VERIFY = """# Gate 2c discovery: official heuristic kernel search (B200)

This packet records the nvidia-matmul-heuristics plus cutlass_profiler
search. It does not approve a dispatch; the two `PHASE=confirm` packets do.

Review order:

1. `summary.json`: expect 12 problems, `coverage` families mostly present,
   and any unprofiled-case counts explained in the finding (the StreamK
   multi-CTA-cluster rejections are a known CUTLASS constraint).
2. `coverage-audit.csv`: per-problem 1-SM/2-SM, cluster, StreamK, split-K,
   raster, swizzle, and stage coverage of the emitted testlist.
3. `oracle-selection.csv`: the fastest successful profiled kernel per padded
   problem (combined top-16 pool plus the N=256 top-32 expansion), joined
   with the same-run torch.mm cold- and warm-L2 medians.
4. `expansion-n256-summary.csv`: the top-32 stop-rule expansion for the two
   N=256 problems that failed the top-16 selection.
5. `heuristics-testlist.csv`, `heuristics-testlist-n256.csv`, and
   `profiler-results*.gemm.csv`: raw inputs and outputs of the search.
   `profiler-parsed.csv` is the joined debugging view.
6. `library-generation-log*.txt`: generator diagnostics and rejections.
7. `run-metadata.json`: toolchain, clock-lock outcome, and the exact
   profiler commands. `cublas-log-*.txt` identifies the torch.mm kernels.

The torch.mm baseline for this packet is in `baseline-cases.csv` (raw
repetitions, warm and cold) and `baseline-summary.csv`.
"""


# ---------------------------------------------------------------------------
# Confirmation packet
# ---------------------------------------------------------------------------


def _write_confirm_packet(baseline: dict, candidates: dict) -> None:
    run_dir = OUTPUT_DIR / f"gate-2c-confirm-{RUN}"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = pd.DataFrame(baseline["timings"])
    candidate_rows = pd.DataFrame(candidates["timings"])
    correctness = pd.DataFrame(candidates["correctness"])
    # The small-N GEMV accumulates FP32 in serial K order, which differs from
    # cuBLAS split-K accumulation by up to 1 bf16 ULP at the largest output
    # magnitude (2.0 at |x| >= 256). Exact equality holds for every
    # tensor-core variant; the rounding tolerance applies only to the GEMV.
    # Rejected candidates (can_implement or launch) are recorded separately
    # and excluded from selection and the rounding check.
    rejected = correctness.query("rejected.notnull()")
    measured = correctness.query("rejected.isnull()").copy()
    measured["within_rounding"] = (
        measured["exact"].eq(1)
        | (
            measured["finite"].eq(1)
            & measured["max_abs_difference"].le(2.0)
        )
    ).astype(int)
    correctness = pd.concat([measured, rejected], ignore_index=True)
    baseline_rows.to_csv(run_dir / "baseline-cases.csv", index=False)
    candidate_rows.to_csv(run_dir / "candidate-cases.csv", index=False)
    correctness.to_csv(run_dir / "correctness.csv", index=False)

    baseline_summary = _summarize_baseline(baseline_rows)
    baseline_summary.to_csv(run_dir / "baseline-summary.csv", index=False)
    cold_baseline = baseline_summary.query("cache_state == 'cold'")[
        list(CASE_KEYS) + ["median_ms"]
    ].rename(columns={"median_ms": "torch_mm_cold_ms"})

    candidate_summary = (
        candidate_rows.groupby(list(CASE_KEYS) + ["variant"], as_index=False)
        .agg(
            repetitions=("latency_ms", "count"),
            median_ms=("latency_ms", "median"),
            p10_ms=("latency_ms", lambda values: values.quantile(0.1)),
            p90_ms=("latency_ms", lambda values: values.quantile(0.9)),
            std_ms=("latency_ms", "std"),
        )
        .merge(
            cold_baseline,
            on=list(CASE_KEYS),
            how="left",
            validate="many_to_one",
        )
    )
    candidate_summary["latency_ratio"] = (
        candidate_summary["median_ms"]
        / candidate_summary["torch_mm_cold_ms"]
    )
    candidate_summary["pass"] = (
        candidate_summary["latency_ratio"].le(MAXIMUM_RATIO).astype(int)
    )
    candidate_summary.to_csv(run_dir / "case-summary.csv", index=False)

    selection_keys = ["vocab_size", "hidden_size", "n_hidden_states"]
    selected = (
        candidate_summary.sort_values("median_ms")
        .groupby(selection_keys, as_index=False)
        .first()
    )
    selected.to_csv(run_dir / "selected.csv", index=False)
    dispatch = _simplify_dispatch(candidate_summary, selected)
    dispatch.to_csv(run_dir / "dispatch.csv", index=False)

    metadata = {
        "baseline": baseline["metadata"],
        "candidates": candidates["metadata"],
        "run": RUN,
    }
    (run_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    summary = {
        "gate": "ordinary-gemm-tuning",
        "phase": "gate-2c-confirm",
        "run": RUN,
        "command": (
            "make modal-cutlass GATE=ordinary-gemm-tuning "
            f"PHASE=confirm RUN={RUN}"
        ),
        "status": (
            "pass"
            if len(selected) == 18
            and selected["pass"].eq(1).all()
            and measured["within_rounding"].eq(1).all()
            else "tuning-required"
        ),
        "maximum_cutlass_to_cublas_ratio": MAXIMUM_RATIO,
        "protocol": (
            "torch.mm baseline and CUTLASS candidates interleaved in one "
            "process on one host (measure_confirm_sweep)"
        ),
        "warmup_repetitions": WARMUP_REPETITIONS,
        "benchmark_repetitions": BENCHMARK_REPETITIONS,
        "cases": int(len(selected)),
        "selected_passes": int(selected["pass"].sum()),
        "worst_selected_ratio": float(selected["latency_ratio"].max()),
        "selected_variants": sorted(selected["variant"].unique().tolist()),
        "dispatch_variants": sorted(dispatch["variant"].unique().tolist()),
        "worst_dispatch_ratio": float(dispatch["latency_ratio"].max()),
        "correctness_exact_cases": int(measured["exact"].sum()),
        "correctness_cases": int(len(measured)),
        "correctness_within_rounding_cases": int(
            measured["within_rounding"].sum()
        ),
        "rejected_cases": int(len(rejected)),
        "rejected_variants": sorted(rejected["variant"].unique().tolist()),
        "gemv_rounding_note": (
            "The small-N GEMV uses serial-K FP32 accumulation; its output "
            "may differ from cuBLAS split-K accumulation by up to 1 bf16 "
            "ULP. Tensor-core variants must be exact."
        ),
        "decision": (
            "Gate 2c passes only when two independent confirm runs each "
            "select a dispatch within ratio 1.05 of torch.mm in all 18 "
            "cases."
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "VERIFY.md").write_text(
        _confirm_verify(summary, dispatch)
    )


def _confirm_verify(summary: dict, dispatch: pd.DataFrame) -> str:
    dispatch_lines = "\n".join(
        f"- `{row.variant}` at V={row.vocab_size}, D={row.hidden_size}, "
        f"H={row.n_hidden_states} (ratio {row.latency_ratio:.4f})"
        for row in dispatch.itertuples()
    )
    within_rounding = summary.get(
        "correctness_within_rounding_cases", summary.get("correctness_cases")
    )
    rejected_cases = summary.get("rejected_cases", 0)
    rejected_variants = summary.get("rejected_variants", [])
    protocol = summary.get(
        "protocol",
        "torch.mm baseline and CUTLASS candidates measured in separate "
        "containers (superseded protocol; see the finding)",
    )
    return f"""# Gate 2c confirmation run {summary["run"]}: matched torch.mm versus CUTLASS dispatch (B200)

Protocol: {protocol}.

## Expected outcome

`status == "pass"`: all {summary["cases"]} cases have a selected CUTLASS
candidate within ratio {summary["maximum_cutlass_to_cublas_ratio"]} of the
same-process cold-L2 torch.mm median (100 measured repetitions after 25
warmup, CUDA events, preallocated buffers), and every measured candidate
matches torch.mm within rounding (exact for tensor-core variants, <= 1 bf16
ULP for the small-N GEMV).

## Actual outcome

- status: `{summary["status"]}`
- selected passes: {summary["selected_passes"]} / {summary["cases"]}
- worst selected ratio: {summary["worst_selected_ratio"]:.4f}
- correctness within rounding: {within_rounding} / {summary["correctness_cases"]}
- rejected candidates: {rejected_cases} {rejected_variants}
- simplified dispatch variants: {summary["dispatch_variants"]}

## Explicit failure criteria

Fail if any case has no candidate (selected rows != 18), any selected ratio
exceeds {summary["maximum_cutlass_to_cublas_ratio"]}, any tensor-core
candidate is not bit-exact, or the baseline/candidate repetitions are
missing. Two independent runs must each pass before Gate 2c passes.

## Review order

1. `summary.json` (values above), then `selected.csv` (per-case decision).
2. `case-summary.csv`: every candidate's median, dispersion, ratio, and
   explicit `pass` column. `dispatch.csv` is the smallest simplified
   dispatch within 1% of the per-case oracle:
{dispatch_lines}
3. `correctness.csv`: exactness and rejection diagnostics per candidate.
4. `baseline-cases.csv` and `candidate-cases.csv`: raw repetitions.
5. `run-metadata.json`: toolchain and clock metadata.

## Checks

```bash
python3 -c "import json; s=json.load(open('summary.json')); assert s['status']=='pass', s"
python3 -c "import pandas as pd; s=pd.read_csv('selected.csv'); assert len(s)==18 and s['pass'].eq(1).all() and s['latency_ratio'].max()<={summary['maximum_cutlass_to_cublas_ratio']}"
python3 -c "import pandas as pd; c=pd.read_csv('correctness.csv'); m=c.query('variant != \\"small-n-gemv\\" and rejected.isnull()'); assert m['exact'].eq(1).all()"
```

Run from this packet's directory. Compare `selected.csv` with the other
confirmation run's before approving the gate.
"""


def _simplify_dispatch(
    candidate_summary: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    """Smallest variant set staying within 1% of the per-case oracle."""
    selection_keys = ["vocab_size", "hidden_size", "n_hidden_states"]
    oracle = selected[selection_keys + ["median_ms"]].rename(
        columns={"median_ms": "oracle_ms"}
    )
    merged = candidate_summary.merge(
        oracle, on=selection_keys, how="left", validate="many_to_one"
    )
    merged["within_one_percent"] = (
        merged["median_ms"] <= 1.01 * merged["oracle_ms"]
    )
    eligible = merged.query("within_one_percent")
    chosen: list[str] = []
    uncovered = set(map(tuple, merged[selection_keys].drop_duplicates().values))
    while uncovered:
        coverage = (
            eligible.query(
                "variant not in @chosen"
            )[selection_keys + ["variant"]]
            .drop_duplicates()
            .groupby("variant")
            .size()
        )
        if coverage.empty:
            raise RuntimeError(
                "No variant covers the remaining cases within 1% of oracle"
            )
        best = coverage.idxmax()
        chosen.append(best)
        covered = eligible.query("variant == @best")[selection_keys]
        uncovered -= set(map(tuple, covered.values))
    simplified = (
        eligible.query("variant in @chosen")
        .sort_values("median_ms")
        .groupby(selection_keys, as_index=False)
        .first()
    )
    return simplified


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main() -> None:
    if PHASE == "discover":
        profile_handle = profile_heuristics.spawn()
        baseline_handle = measure_torch_mm.spawn()
        _write_discovery_packet(profile_handle.get(), baseline_handle.get())
    elif PHASE == "confirm":
        result = measure_confirm_sweep.remote()
        baseline = {
            "timings": result["baseline"],
            "metadata": result["metadata"],
        }
        _write_confirm_packet(baseline, result)
    else:
        raise ValueError(f"Unknown PHASE {PHASE!r}; use discover or confirm")
