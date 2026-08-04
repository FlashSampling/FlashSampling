"""Controlled B200 validation for precise CUTLASS extension caching."""

import json
import os
import tempfile
from pathlib import Path

from ...cutlass_build import (
    ExtensionBuildSpec,
    broad_source_fingerprint,
    extension_fingerprint,
)
from ..utils import make_app, make_volumes, set_volume_caches
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/dev-infra-phase1")


@app.function(gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60)
def record_sm100() -> dict:
    import torch

    from fused_mm_sampling import cutlass_impl
    from fused_mm_sampling.dev_metrics import emit_dev_event

    set_volume_caches()
    emit_dev_event("remote_start", stage="dev_infra_validation", gpu="B200")
    original_extensions_dir = os.environ["TORCH_EXTENSIONS_DIR"]
    with tempfile.TemporaryDirectory(prefix="fmms-cutlass-dev-infra-") as cache_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = cache_dir
        try:
            _reset_modules(cutlass_impl)
            cold_modules = _load_modules(cutlass_impl)
            cold_outputs = _smoke_outputs(cutlass_impl, torch, "cold")
            _reset_modules(cutlass_impl)
            warm_modules = _load_modules(cutlass_impl)
            warm_outputs = _smoke_outputs(cutlass_impl, torch, "warm")
        finally:
            os.environ["TORCH_EXTENSIONS_DIR"] = original_extensions_dir
            _reset_modules(cutlass_impl)
    result = {
        "greedy_equal": bool(torch.equal(cold_outputs["greedy"], warm_outputs["greedy"])),
        "gumbel_equal": bool(torch.equal(cold_outputs["gumbel"], warm_outputs["gumbel"])),
        "cold_binaries": cold_modules,
        "warm_binaries": warm_modules,
    }
    emit_dev_event("remote_end", stage="dev_infra_validation", status="success")
    return result


def _load_modules(cutlass_impl) -> dict[str, str]:
    greedy = cutlass_impl._get_module()
    sampling = cutlass_impl._get_sampling_module()
    return {"greedy": greedy.__file__, "sampling": sampling.__file__}


def _smoke_outputs(cutlass_impl, torch, cache_state: str) -> dict:
    from fused_mm_sampling.dev_metrics import timed_dev_stage

    with timed_dev_stage("provider_smoke", cache_state=cache_state):
        torch.manual_seed(0)
        weights = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
        hidden_states = torch.randn((2, 128), dtype=torch.bfloat16, device="cuda")
        temperature = torch.tensor(1.0, device="cuda")
        greedy = cutlass_impl.fused_mm_sample_cutlass_greedy(weights, hidden_states, 1, temperature)
        gumbel = cutlass_impl.fused_mm_sample_cutlass(
            weights, hidden_states, 1, temperature, seed=17
        )
        torch.cuda.synchronize()
    return {"greedy": greedy, "gumbel": gumbel}


def _reset_modules(cutlass_impl) -> None:
    cutlass_impl._module = None
    cutlass_impl._sampling_module = None


@app.local_entrypoint()
def main() -> None:
    csrc_root = Path("src/fused_mm_sampling/csrc/cutlass").resolve()
    sources = (
        csrc_root / "greedy_provider.cu",
        csrc_root / "winning_schedule_provider.cu",
    )
    legacy = broad_source_fingerprint(csrc_root)
    precise = extension_fingerprint(
        ExtensionBuildSpec(
            prefix="audit",
            source_root=csrc_root,
            sources=sources,
            cuda_flags=("-O3",),
            architecture="100",
            toolchain_identity="audit",
            python_abi="audit",
            torch_version="audit",
            cuda_version="audit",
            supplemental_inputs=(
                csrc_root / "sm100-void-d.patch",
                csrc_root / "sm90-row-reduction-uint64.patch",
            ),
        )
    )
    result = record_sm100.remote()
    passed = result["greedy_equal"] and result["gumbel_equal"]
    summary = {
        "gate": "dev-infra-phase1",
        "command": "make modal-cutlass GATE=dev-infra CUTLASS_DEV_LABEL=precise-cache",
        "status": "pass" if passed else "fail",
        "expected": (
            "The precise input set excludes standalone harnesses, cold and warm "
            "loads produce identical greedy and Gumbel outputs, and the warm load "
            "reports a cache hit in the development metrics packet."
        ),
        "actual": result,
        "legacy_input_count": len(legacy.dependencies),
        "legacy_inputs": list(legacy.dependencies),
        "precise_input_count": len(precise.dependencies),
        "precise_inputs": list(precise.dependencies),
        "possible_failures": [
            "incomplete local include closure",
            "stale extension reuse",
            "missing toolchain fingerprint input",
            "warm load unexpectedly invokes Ninja",
        ],
        "metrics_location": "benchmarking/modal-results/cutlass/dev-metrics/",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not passed:
        raise RuntimeError("CUTLASS development infrastructure smoke outputs changed")
