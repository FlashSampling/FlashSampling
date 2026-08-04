"""Generated-code and spill-slot audit for the Gate 4 control."""

import json
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..utils import make_app, make_volumes, set_volume_caches
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/19-gumbel-sass")
COMPONENTS = ("greedy", "gumbel")


@app.function(gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60)
def record_sm100() -> dict:
    import torch

    from fused_mm_sampling import cutlass_impl

    set_volume_caches()
    weights = torch.zeros((256, 128), dtype=torch.bfloat16, device="cuda")
    hidden_states = torch.zeros((128, 128), dtype=torch.bfloat16, device="cuda")
    temperature = torch.tensor(1.0, device="cuda")
    cutlass_impl.fused_mm_sample_cutlass_greedy(weights, hidden_states, 1, temperature)
    cutlass_impl.fused_mm_sample_cutlass(weights, hidden_states, 1, temperature, seed=17)
    torch.cuda.synchronize()

    modules = {
        "greedy": cutlass_impl._get_module(),
        "gumbel": cutlass_impl._get_sampling_module(),
    }
    results = {}
    for component, module in modules.items():
        results[component] = _audit_binary(Path(module.__file__))
    return results


def _audit_binary(binary: Path) -> dict:
    resource_usage = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sass = subprocess.run(
        ["cuobjdump", "--dump-sass", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with tempfile.TemporaryDirectory() as directory:
        extracted = Path(directory)
        subprocess.run(
            ["cuobjdump", "--extract-elf", "all", str(binary)],
            cwd=extracted,
            check=True,
            capture_output=True,
            text=True,
        )
        cubins = sorted(
            path for path in extracted.iterdir() if path.is_file() and path.stat().st_size > 0
        )
        if not cubins:
            raise RuntimeError(f"cuobjdump extracted no cubins from {binary}")
        disassemblies = []
        for cubin in cubins:
            completed = subprocess.run(
                ["nvdisasm", "--print-line-info", str(cubin)],
                check=True,
                capture_output=True,
                text=True,
            )
            disassemblies.append(completed.stdout)
    line_disassembly = "\n".join(disassemblies)
    return _summarize_disassembly(sass, line_disassembly, resource_usage)


def _summarize_disassembly(sass: str, line_disassembly: str, resource_usage: str) -> dict:
    rows = []
    source_rows = []
    excerpts = []
    for symbol, section in _function_sections(sass):
        local_instructions = [line for line in section.splitlines() if _local_opcode(line)]
        if not local_instructions:
            continue
        demangled = _demangle(symbol)
        if "device_kernel" not in demangled and "gumbel_noise" not in demangled:
            continue
        offsets = [
            int(match.group(1), 16)
            for line in local_instructions
            if (match := re.search(r"\[R\w+\+0x([0-9a-fA-F]+)\]", line))
        ]
        rows.append(
            {
                "symbol": symbol,
                "function": demangled,
                "variant": _variant_name(demangled),
                "local_load_instructions": sum(
                    bool(re.search(r"\bLDL(?:\.|\b)", line)) for line in local_instructions
                ),
                "local_store_instructions": sum(
                    bool(re.search(r"\bSTL(?:\.|\b)", line)) for line in local_instructions
                ),
                "max_local_offset_bytes": max(offsets, default=-1),
            }
        )
    for symbol, section in _function_sections(line_disassembly):
        demangled = _demangle(symbol)
        if "device_kernel" not in demangled and "gumbel_noise" not in demangled:
            continue
        current_source = "unknown"
        selected_lines = []
        for line in section.splitlines():
            if line.lstrip().startswith("//##"):
                current_source = line.strip()
            if _local_opcode(line):
                opcode = "load" if re.search(r"\bLDL(?:\.|\b)", line) else "store"
                source_rows.append(
                    {
                        "symbol": symbol,
                        "function": demangled,
                        "variant": _variant_name(demangled),
                        "operation": opcode,
                        "source": current_source,
                    }
                )
                selected_lines.extend((current_source, line.rstrip()))
        excerpts.append("\n".join([f"Function: {demangled}", *selected_lines]))
    return {
        "functions": rows,
        "source_locations": source_rows,
        "resource_usage": resource_usage,
        "local_memory_excerpts": "\n\n".join(excerpts),
        "cuobjdump_head": "\n".join(sass.splitlines()[:200]),
        "nvdisasm_head": "\n".join(line_disassembly.splitlines()[:200]),
    }


def _function_sections(disassembly: str):
    pattern = re.compile(
        r'(?m)^\s*\.section\s+\.text\.([^,\s"]+).*?$|' r"^\s*Function\s*:\s*(\S+)\s*$"
    )
    matches = list(pattern.finditer(disassembly))
    for index, match in enumerate(matches):
        symbol = match.group(1) or match.group(2)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(disassembly)
        yield symbol, disassembly[match.end() : end]


def _local_opcode(line: str) -> bool:
    return bool(re.search(r"\b(?:LDL|STL)(?:\.|\b)", line))


def _demangle(symbol: str) -> str:
    return subprocess.run(
        ["cu++filt", symbol], check=True, capture_output=True, text=True
    ).stdout.strip()


def _variant_name(function: str) -> str:
    match = re.search(r"fmms_winning_([0-9x]+_c[24])::CandidateEVT", function)
    return match.group(1).replace("_", "-") if match else "base-128x128"


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = record_sm100.remote()
    function_frames = []
    source_frames = []
    for component in COMPONENTS:
        packet = result[component]
        function_frame = pd.DataFrame(packet["functions"])
        function_frame.insert(0, "component", component)
        function_frames.append(function_frame)
        source_frame = pd.DataFrame(packet["source_locations"])
        source_frame.insert(0, "component", component)
        source_frames.append(source_frame)
        (OUTPUT_DIR / f"{component}-resource-usage.txt").write_text(packet["resource_usage"])
        (OUTPUT_DIR / f"{component}-local-memory.sass.txt").write_text(
            packet["local_memory_excerpts"]
        )
        (OUTPUT_DIR / f"{component}-cuobjdump-head.txt").write_text(packet["cuobjdump_head"])
        (OUTPUT_DIR / f"{component}-nvdisasm-head.txt").write_text(packet["nvdisasm_head"])
    functions = pd.concat(function_frames, ignore_index=True)
    sources = pd.concat(source_frames, ignore_index=True)
    functions.to_csv(OUTPUT_DIR / "functions.csv", index=False)
    if sources.empty:
        source_summary = pd.DataFrame(
            columns=(
                "component",
                "function",
                "variant",
                "operation",
                "source",
                "instruction_count",
            )
        )
    else:
        source_summary = (
            sources.groupby(
                ["component", "variant", "function", "operation", "source"],
                as_index=False,
            )
            .agg(instruction_count=("symbol", "size"))
            .sort_values(
                ["component", "variant", "function", "instruction_count"],
                ascending=[True, True, True, False],
            )
        )
    source_summary.to_csv(OUTPUT_DIR / "source-locations.csv", index=False)
    summary = {
        "gate": "4-sass-audit",
        "status": "complete",
        "gpu": "B200",
        "components": list(COMPONENTS),
        "function_count_with_local_memory": len(functions),
        "outputs": [
            "functions.csv",
            "source-locations.csv",
            "greedy-resource-usage.txt",
            "gumbel-resource-usage.txt",
            "greedy-local-memory.sass.txt",
            "gumbel-local-memory.sass.txt",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
