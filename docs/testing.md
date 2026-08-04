# Testing

## Local and Modal entry points

- Use `make modal-verify-correctness-tp1` for one-GPU correctness on Modal.
- Use `make modal-pytest-distributed N_PROCS=2` for distributed correctness on two NVLink-connected GPUs.
- Set `N_PROCS` explicitly to test another world size.
- Use `make pytest-distributed` locally.
  It automatically skips on a single-GPU machine.

Both Modal correctness paths use torchrun, including TP1, so workers can construct `TPInfo.from_world()` with an initialized process group.

## Sampling distribution tests

`test_sampling_distribution` uses a chi-squared goodness-of-fit test against theoretical softmax probabilities.
It covers every provider, vocab sizes 100 and 256, and hidden-state counts 1 and 2 to exercise tile boundaries and dimensional edge cases.
Bins with expected counts below five are excluded, and expected counts are rescaled to the observed total.

`make_synthetic_inputs()` in `src/fused_mm_sampling/testing.py` constructs weights and hidden states that produce known ascending or descending logits through SVD and a pseudoinverse.

## Large-vocabulary validation

Run the TP1 large-vocabulary chi-squared check with:

```bash
make modal-verify-correctness-large-vocab \
  VOCAB_SIZE=... NUM_SAMPLES=... SAMPLES_PER_CALL=...
```

The runner batches sampling, accumulates counts on GPU, reports the test statistic and probability-mass coverage, and forwards the parameters as Modal CLI options.
See `findings/upcasting-before-softmax.md` and `findings/rebuttal-benchmark-summary.md` for prior distribution findings.
