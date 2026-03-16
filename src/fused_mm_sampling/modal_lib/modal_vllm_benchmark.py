import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import modal

from .utils import make_app, make_volumes, set_volume_caches, volume_path

_repo_root = Path(__file__).resolve().parents[3]
_bench_params_dir = _repo_root / "benchmarking" / "vllm"

SERVE_FLAGS = "--max-model-len 1024 --no-enable-prefix-caching --uvicorn-log-level warning"
BENCH_FLAGS = (
    "--dataset-name hf --dataset-path AI-MO/aimo-validation-aime"
    " --hf-output-len 256 --temperature 0.6 --top-k -1 --top-p 1.0"
    " --backend openai-chat --endpoint /v1/chat/completions"
)
AFTER_BENCH = (
    "curl -s -X POST http://localhost:8000/reset_prefix_cache"
    " && curl -s -X POST http://localhost:8000/reset_mm_cache"
    " && curl -s -X POST http://localhost:8000/reset_encoder_cache"
)

ALL_VARIANTS = {
    "baseline": {},
    "fmms-triton": {"VLLM_USE_FMMS_SAMPLER": "1", "VLLM_FMMS_PROVIDER": "fused-triton"},
}


def make_vllm_image() -> modal.Image:
    return (
        modal.Image.from_registry("pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel")
        .apt_install("git", "curl")
        .run_commands("pip install --break-system-packages uv")
        .run_commands(
            "git clone --depth 1 -b feature/fmms-sampler"
            " https://github.com/tomasruizt/vllm.git /opt/vllm",
            # Pin torch to match the precompiled .so (built for torch 2.10.0+cu130).
            # Without the constraint, uv downgrades torch to 2.9.1, causing ABI mismatch.
            "cd /opt/vllm && VLLM_USE_PRECOMPILED=1 uv pip install --system -e '.[bench]'",
            # The precompiled .so is built for torch 2.10.0+cu130 but vLLM metadata
            # pins torch==2.9.1, causing an ABI mismatch. Reinstall the base image's
            # torch after vLLM is installed.
            "uv pip install --system 'torch==2.10.0' 'torchvision>=0.25' 'torchaudio>=2.10'",
        )
        .add_local_dir(
            str(_repo_root / "src"),
            remote_path="/opt/fused-mm-sample/src",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_file(
            str(_repo_root / "pyproject.toml"),
            remote_path="/opt/fused-mm-sample/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            str(_repo_root / "README.md"), remote_path="/opt/fused-mm-sample/README.md", copy=True
        )
        .run_commands(
            "uv pip install --system tabulate /opt/fused-mm-sample",
        )
    )


@dataclass
class Args:
    model: str = "openai/gpt-oss-120b"
    sweep: str = "quick"  # "quick" or "all"
    tgt_dir: str = "/vol-fused-mm-sample/vllm-bench"
    num_runs: int = 5
    variants: str = ""  # comma-separated, e.g. "baseline,fmms-triton". Empty = all.
    bench_params_json: str = ""  # JSON string with bench params (read from file locally)


app = make_app()
gpu = os.getenv("GPU", "b200")
model = os.getenv("MODEL", "openai/gpt-oss-120b")
sweep = os.getenv("SWEEP", "quick")
tgt_dir = os.getenv("TGT_DIR", "/vol-fused-mm-sample/vllm-bench")
variants = os.getenv("VARIANTS", "")


@app.function(
    gpu=gpu,
    image=make_vllm_image(),
    volumes=make_volumes(),
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})],
    timeout=2 * 60 * 60,
)
def function(args: Args):
    set_volume_caches()
    os.environ["HF_HOME"] = f"{volume_path}/hf-cache"

    model_slug = args.model.split("/")[-1]
    is_quick = args.sweep == "quick"
    params = json.loads(args.bench_params_json)
    num_runs = 1 if is_quick else args.num_runs
    enforce_eager = "--enforce-eager" if is_quick else ""

    if args.variants:
        variant_keys = [v.strip() for v in args.variants.split(",")]
        run_variants = [(k, ALL_VARIANTS[k]) for k in variant_keys]
    else:
        run_variants = list(ALL_VARIANTS.items())

    with tempfile.TemporaryDirectory() as tmpdir:
        params_file = Path(tmpdir) / "bench-params.json"
        params_file.write_text(json.dumps(params, indent=2))

        for variant_name, env_vars in run_variants:
            out_dir = f"{args.tgt_dir}/{model_slug}/{variant_name}"

            env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())
            if env_prefix:
                serve_cmd = (
                    f"env {env_prefix} vllm serve {args.model} {SERVE_FLAGS} {enforce_eager}"
                )
            else:
                serve_cmd = f"vllm serve {args.model} {SERVE_FLAGS} {enforce_eager}"

            cmd = (
                f"vllm bench sweep serve"
                f' --serve-cmd "{serve_cmd}"'
                f' --bench-cmd "vllm bench serve {BENCH_FLAGS}"'
                f' --bench-params "{params_file}"'
                f' --after-bench-cmd "{AFTER_BENCH}"'
                f" --num-runs {num_runs} --show-stdout"
                f" --server-ready-timeout 1200"
                f' -o "{out_dir}"'
            )

            print(f"\n{'=' * 60}")
            print(f"Running variant: {variant_name}")
            print(f"Command: {cmd}")
            print(f"{'=' * 60}\n", flush=True)

            result = subprocess.run(cmd, shell=True, text=True)
            if result.returncode != 0:
                print(f"WARNING: {variant_name} exited with code {result.returncode}")

            # Move the sweep log into the latest timestamped directory
            variant_dir = Path(out_dir)
            if variant_dir.exists():
                timestamped_dirs = sorted(d for d in variant_dir.iterdir() if d.is_dir())
                if timestamped_dirs:
                    tmp_log = variant_dir.parent / f"{variant_name}.tmp"
                    if tmp_log.exists():
                        shutil.move(str(tmp_log), str(timestamped_dirs[-1] / "sweep.log"))

        modal.Volume.from_name("fused-mm-sample").commit()


@app.local_entrypoint()
def main():
    params_file = "quick-bench-params.json" if sweep == "quick" else "bench-params.json"
    bench_params_json = (_bench_params_dir / params_file).read_text()
    args = Args(
        model=model,
        sweep=sweep,
        tgt_dir=tgt_dir,
        variants=variants,
        bench_params_json=bench_params_json,
    )
    function.remote(args=args)
