"""Print torch / triton / flashinfer / numpy versions on the modal images."""

from .modal_vllm_benchmark import make_vllm_image
from .utils import make_app, make_image

app = make_app()

PKGS = ("torch", "triton", "flashinfer-python", "numpy")


def _print_versions(label: str) -> None:
    import importlib.metadata as md

    print(f"--- {label} ---")
    for pkg in PKGS:
        try:
            print(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")


@app.function(gpu="t4", image=make_image(), timeout=5 * 60)
def show_triton():
    _print_versions("triton-bench image")


@app.function(gpu="t4", image=make_vllm_image(), timeout=5 * 60)
def show_vllm():
    _print_versions("vllm image")


@app.local_entrypoint()
def main():
    show_triton.remote()
    show_vllm.remote()
