"""Print torch / triton / flashinfer versions on the modal image."""

from .utils import make_app, make_image

app = make_app()


@app.function(gpu="t4", image=make_image(), timeout=5 * 60)
def show():
    import importlib.metadata as md

    for pkg in ("torch", "triton", "flashinfer-python", "numpy"):
        try:
            print(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")


@app.local_entrypoint()
def main():
    show.remote()
