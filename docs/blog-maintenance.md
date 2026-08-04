# Blog maintenance

The blog post lives at `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/index.qmd`.
Keep its benchmark numbers in sync with the README.
The kernel benchmark section presents both the large configuration (V=128,256, D=8,192) and the small configuration (V=151,936, D=4,096) as the outermost tabset.

## Quarto conventions

- Panel tabsets use `::: {.panel-tabset group="name"}` with `# Tab Name` headers.
- The `group=` attribute synchronizes tab selection across tabsets with the same group name.
- Nested tabsets use four colons for the outer tabset and three for the inner tabset.
- Outer tabs use `#` headers and inner tabs use `##` headers.

```markdown
:::: {.panel-tabset group="baseline"}
# vs PyTorch Compiled
::: {.panel-tabset group="gpu"}
## B300
![](imgs/relative-perf-vs-pytorch-b300.png)
## H100
![](imgs/relative-perf-vs-pytorch-h100.png)
:::
# vs FlashInfer
...
::::
```

## Presentation rules

- Show the plot before its data table.
- Order GPUs as B300, B200, H200, H100, A100.
- Use at most two decimal places for numeric table values.
- Store blog images under `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/imgs/` and reference them as `![](imgs/filename.png)`.
- Remove completed items from the commented HTML TODO section near the top instead of striking them through.
- FMMS uses bold red (`#d62728`) and baselines use gray or blue.
- The palette is defined by `PROVIDER_COLORS` in `benchmarking/plot-triton-bench.py` and `VARIANT_COLORS` in `benchmarking/vllm/plot_tpot.py`.

After regenerating plots, copy them with:

```bash
make -C ~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample copy-imgs
```
