"""Standalone Proton intra-kernel profiling for the fused-triton kernel.

Runs ONLY the FMMS Triton kernel (stage 1) under Proton instrumentation,
skipping _local_reduce (stage 2) which uses torch.compile and would conflict
with Proton's process-global instrumentation backend.

Usage (3-step TTGIR override workflow):

  # 1. Dump TTGIR
  ./dump_ttgir.sh python proton_profile.py --n_hidden_states=1

  # 2. Inject proton.record statements into TTGIR
  python insert_proton_records.py

  # 3. Run with TTGIR override
  TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_OVERRIDE=1 TRITON_OVERRIDE_DIR=ttgir_dump \\
    python proton_profile.py --n_hidden_states=1
"""

import os

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch
import triton
import triton.profiler as proton
from pydantic_settings import BaseSettings
from triton._C.libtriton.proton import BUFFER_TYPE, SAMPLING_STRATEGY
from triton.profiler.mode import Default

from fused_mm_sampling.bench.sys_metadata import get_gpu_name
from fused_mm_sampling.core import (
    MIN_BLOCK_SIZE_V,
    fused_mm_sample_triton_kernel,
    set_torch_allocator_for_tma_descriptors,
    supports_warp_specialization,
)

device = torch.device("cuda")
set_torch_allocator_for_tma_descriptors()


# "small" case: Qwen3-8B / Llama 3 8B dimensions
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 4096


class Args(BaseSettings):
    n_hidden_states: int = 1
    n_samples: int = 1
    vocab_size: int = VOCAB_SIZE


class CliArgs(Args, cli_parse_args=True):
    pass


def run_proton_profile(args: Args) -> None:
    v, d, h = args.vocab_size, HIDDEN_SIZE, args.n_hidden_states
    print(f"GPU: {get_gpu_name()}")
    print(f"V={v}, D={d}, n_hidden_states={h}, n_samples={args.n_samples}")

    weights = torch.randn((v, d), dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn((h, d), dtype=torch.bfloat16, device=device)
    temperature = torch.tensor(1.0, device=device)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    max_grid_size_v = triton.cdiv(v, MIN_BLOCK_SIZE_V)

    # Pre-allocate output buffers (reused across iterations)
    maxs = torch.empty((args.n_samples, max_grid_size_v, h), dtype=torch.bfloat16, device=device)
    maxs_idx = torch.empty_like(maxs, dtype=torch.long)
    logits_out = torch.empty((v, h), dtype=torch.float32, device=device)

    def grid(meta):
        gv = triton.cdiv(v, meta["BLOCK_SIZE_V"])
        gh = triton.cdiv(h, meta["BLOCK_SIZE_H"])
        return (min(num_sms, gv * gh),)

    kernel_kwargs = dict(
        weights_ptr=weights,
        hidden_states_ptr=hidden_states,
        max_out_ptr=maxs,
        max_out_idx_ptr=maxs_idx,
        vocab_size=v,
        hidden_size=d,
        n_hidden_states=h,
        num_samples=args.n_samples,
        temperature_ptr=temperature,
        seed=42,
        max_grid_size_v=max_grid_size_v,
        logits_out_ptr=logits_out,
        WARP_SPECIALIZE=supports_warp_specialization(),
        NUM_SMS=num_sms,
        GREEDY_SAMPLING=False,
        RETURN_LOGITS=False,
        USE_PROTON_SCOPES=False,
    )

    # Start Proton before the first kernel call. When using the TTGIR override
    # workflow, the proton dialect must be registered before the overridden TTGIR
    # (which contains proton.record ops) is parsed.
    mode = Default(
        sampling_strategy=SAMPLING_STRATEGY.SELECTIVE,
        sampling_options="0",  # only profile warp 0
        buffer_type=BUFFER_TYPE.GLOBAL,
    )
    proton.start(name="kernel", data="trace", backend="instrumentation", mode=mode)

    print("Profiling...")
    fused_mm_sample_triton_kernel[grid](**kernel_kwargs)
    torch.cuda.synchronize()

    proton.finalize()
    print("Done. Output: kernel.chrome_trace")


if __name__ == "__main__":
    args = CliArgs()
    run_proton_profile(args)
