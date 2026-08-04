"""Dump the selected B200 Triton FMMS compiler IR and machine code."""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..utils import (
    add_library_code,
    make_app,
    make_image,
    make_volumes,
    set_volume_caches,
)

app = make_app()
image = add_library_code(make_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/24-triton-liveness")
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 4_096
N_HIDDEN_STATES = 256


@app.function(gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60)
def record_sm100() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        dump_dir = Path(directory)
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["TRITON_KERNEL_DUMP"] = "1"
        os.environ["TRITON_DUMP_DIR"] = str(dump_dir)

        import torch

        from fused_mm_sampling.core import get_sampler

        set_volume_caches()
        torch.manual_seed(0)
        weights = torch.randn(
            (VOCAB_SIZE, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
        )
        hidden_states = torch.randn(
            (N_HIDDEN_STATES, HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        sampler = get_sampler("fused-triton", weights=weights)
        sampler.sample(
            weights=weights,
            hidden_states=hidden_states,
            num_samples=1,
            temperature=torch.tensor(1.0, device="cuda"),
            seed=17,
        )
        torch.cuda.synchronize()
        return _collect_dump(dump_dir)


def _collect_dump(dump_dir: Path) -> dict:
    artifacts = {}
    rows = []
    selected_directories = sorted(
        {
            path.parent
            for path in dump_dir.rglob("*.ttgir")
            if "fused_mm_sample_triton_kernel" in path.name
        }
    )
    for index, directory in enumerate(selected_directories):
        prefix = f"kernel-{index}"
        for suffix in ("ttir", "ttgir", "llir", "ptx"):
            matches = sorted(directory.glob(f"*.{suffix}"))
            if matches:
                artifacts[f"{prefix}.{suffix}"] = matches[0].read_text()
        cubins = sorted(directory.glob("*.cubin"))
        for cubin_index, cubin in enumerate(cubins):
            disassembly = subprocess.run(
                ["nvdisasm", "--print-line-info", str(cubin)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            resource_usage = subprocess.run(
                ["cuobjdump", "--dump-resource-usage", str(cubin)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            name = f"{prefix}-{cubin_index}"
            artifacts[f"{name}-resource-usage.txt"] = resource_usage
            local_lines = [
                line
                for line in disassembly.splitlines()
                if re.search(r"\b(?:LDL|STL)(?:\.|\b)", line)
            ]
            artifacts[f"{name}-local-memory.sass.txt"] = "\n".join(local_lines)
            rows.append(
                {
                    "kernel": index,
                    "cubin": cubin_index,
                    "local_load_instructions": sum(
                        bool(re.search(r"\bLDL(?:\.|\b)", line))
                        for line in local_lines
                    ),
                    "local_store_instructions": sum(
                        bool(re.search(r"\bSTL(?:\.|\b)", line))
                        for line in local_lines
                    ),
                }
            )
    if not selected_directories:
        dumped = sorted(str(path.relative_to(dump_dir)) for path in dump_dir.rglob("*"))
        raise RuntimeError(f"No FMMS TTGIR found. Dump contained: {dumped}")
    return {"artifacts": artifacts, "rows": rows}


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = record_sm100.remote()
    for name, contents in result["artifacts"].items():
        (OUTPUT_DIR / name).write_text(contents)
    machine_code = pd.DataFrame(result["rows"])
    machine_code.to_csv(OUTPUT_DIR / "machine-code-summary.csv", index=False)
    summary = {
        "gate": "4-triton-register-liveness",
        "status": "complete",
        "gpu": "B200",
        "shape": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "n_hidden_states": N_HIDDEN_STATES,
        },
        "compiler_ir_count": sum(name.endswith(".ttgir") for name in result["artifacts"]),
        "machine_code_images": len(machine_code),
        "local_load_instructions": int(machine_code["local_load_instructions"].sum()),
        "local_store_instructions": int(machine_code["local_store_instructions"].sum()),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
