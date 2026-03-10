import modal


def make_app():
    return modal.App("fused-matmul-sample")


def make_image():
    img = modal.Image.from_registry("pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel")
    deps = [
        "flashinfer-python",
        "pandas",
        "pydantic-settings",
        "matplotlib",
        "nvtx",
        "llnl-hatchet",
        "scipy",
    ]
    return img.uv_pip_install(deps)


volume_path = "/vol-fused-mm-sample"


def make_volumes():
    return {volume_path: modal.Volume.from_name("fused-mm-sample")}
