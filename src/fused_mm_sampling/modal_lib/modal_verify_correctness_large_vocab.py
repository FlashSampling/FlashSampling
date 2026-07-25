from .utils import ModalEnvConfig, make_app, make_image, make_volumes, set_volume_caches


class Config(ModalEnvConfig):
    vocab_size: int = 32_768
    num_samples: int = 1_000_000
    samples_per_call: int = 10_000


cfg = Config()
app = make_app()


@app.function(gpu=cfg.gpu_spec, image=make_image(), volumes=make_volumes(), timeout=cfg.timeout)
def verify_correctness_large_vocab(
    vocab_size: int,
    num_samples: int,
    samples_per_call: int,
) -> None:
    from ..testing import assert_sampling_distribution_large_vocab

    set_volume_caches()
    assert_sampling_distribution_large_vocab(
        vocab_size=vocab_size,
        num_samples=num_samples,
        samples_per_call=samples_per_call,
    )


@app.local_entrypoint()
def main(
    vocab_size: int = cfg.vocab_size,
    num_samples: int = cfg.num_samples,
    samples_per_call: int = cfg.samples_per_call,
) -> None:
    verify_correctness_large_vocab.remote(vocab_size, num_samples, samples_per_call)
