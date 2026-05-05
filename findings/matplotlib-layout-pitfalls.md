# Matplotlib layout pitfalls (legends, tight_layout, savefig)

Notes from a debugging session that ate ~15 iterations on legend placement.
The root cause was three layout mechanisms fighting each other; once you see how
they interact, the fix is a single pattern.

## TL;DR

**Use this pattern for any plot with a legend outside the axes:**

```python
fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
# ... draw data ...
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=N,
    title="Method",
    fontsize=fs,
    title_fontsize=fs,
)
fig.savefig(path, dpi=300)   # NO bbox_inches="tight"
```

This guarantees:
- Saved canvas always equals `figsize` (no silent shrinking when legend grows).
- Legend always sits flush above the axes and grows upward (no overlap if entries
  or fontsize change).
- The constrained layout engine reserves space for the legend automatically;
  changing `ncol` resizes the axes instead of pushing the legend into the plot.

## The three interactions that cause grief

### 1. `loc` controls which edge of the legend bbox is anchored

Common mistake: `loc="upper center"` + `bbox_to_anchor=(0.5, 1.3)`.
This anchors the legend's *top* at y=1.3 (axes coords), so the legend extends
**downward**. As you add entries or bump `fontsize`, the legend gets taller and
its bottom dips back into the axes. The fix is to anchor the *opposite* edge:

- `loc="lower center"` + `bbox_to_anchor=(0.5, 1.02)`: legend's bottom is just
  above the axes top, legend grows **upward**, never overlaps the data.

Symmetric: for legends below the axes, use `loc="upper center"` +
`bbox_to_anchor=(0.5, -0.05)` so the legend grows downward away from the plot.

### 2. `tight_layout()` does not reserve space for legends placed outside the axes

`tight_layout` only knows about labels, titles, and tick labels. A legend
positioned with `bbox_to_anchor=(..., 1.x)` is invisible to it. Result: the axes
do not shrink to make room, and the legend just floats in white space (which
then either gets cropped on save, or expands the saved canvas, depending on
`bbox_inches`).

`constrained_layout` *does* reserve space for legends placed via
`Axes.legend(..., bbox_to_anchor=...)`. Use it instead:

```python
fig, ax = plt.subplots(layout="constrained")
```

Caveats:
- `constrained_layout` does **not** handle `Figure.legend()` placement (yet —
  matplotlib 3.10 docs flag this). Stick to `Axes.legend()`.
- Don't mix with `fig.tight_layout()`. Pick one.

### 3. `savefig(bbox_inches="tight")` expands the saved canvas to fit out-of-axes artists

`bbox_inches="tight"` recomputes the bounding box of *all* visible artists
(labels, titles, annotations, legends) and saves that region. So:

- If the legend is wider than the axes (e.g. 3-column legend with long labels),
  the saved PDF is wider than `figsize`.
- If you place that next to a different plot whose legend is narrower, both
  PDFs render at different widths. Display them at the same on-screen width and
  the wider one looks like its axes have shrunk.

Concrete symptom we hit: relative-perf-vs-FlashInfer (3 providers, 3-col legend)
looked narrower than relative-perf-vs-PyTorch (2 providers, 2-col legend) when
viewed side-by-side. Same `figsize=(10, 6)` in code, different saved widths.

Fixes (any one of these):
- Drop `bbox_inches="tight"` from savefig and use `constrained_layout` to
  reserve space inside `figsize`. This is the cleanest.
- Keep `bbox_inches="tight"` but call `leg.set_in_layout(False)` to exclude
  the legend from both layout calculations and the tight bbox.
- Use a fixed `pad_inches` and accept that different plots may have different
  saved sizes.

## Why we keep tweaking `bbox_to_anchor` y values

Each combination of (fontsize, ncol, number of entries, figsize) changes the
height of the legend in axes-fraction units. With `loc="upper center"` you have
to retune the anchor every time. With `loc="lower center"` + small positive
offset (e.g. 1.02) you don't — the legend's bottom is pinned to the axes top,
and the engine grows it upward.

Mantra: **anchor the edge of the legend that is closest to the plot, not the
edge furthest from it.**

## Other small things worth knowing

- **`ax.legend(..., loc="outside upper center")`** — only works with
  `fig.legend()` and `constrained_layout`. Cleanest API for outside legends but
  loses per-axes control.
- **`leg.set_in_layout(False)`** — escape hatch to keep an existing legend out
  of both layout engines and tight bbox computations. Useful for floating
  annotations.
- **Single fontsize variable per plot** — declare `fontsize = 18` once and pass
  it to all of `set_xlabel`, `set_ylabel`, `tick_params(labelsize=...)`,
  `legend(fontsize=..., title_fontsize=...)`. Avoids drift where the legend ends
  up a different size from the axis labels.
- **DPI: 150 for review, 300 for production** — review renders should be small
  enough to inspect quickly. Iteration cap: 3 render-review cycles before
  stepping back to question the approach (per the matplotlib-render-review
  skill discipline).

## What we changed in this codebase

- `plot_roofline` migrated to the unified pattern: `layout="constrained"`,
  `loc="lower center"`, `bbox_to_anchor=(0.5, 1.02)`, no `tight_layout`,
  no `bbox_inches="tight"` in its savefig call.
- `plot_memory_throughput` and `plot_relative_performance` still use the
  older `tight_layout()` + `bbox_inches="tight"` + `loc="upper center"` pattern
  — they happen to work because their legends are short (2 rows). They will
  need migration if their legend layout ever grows.

## Sources

- [Matplotlib Legend guide](https://matplotlib.org/stable/users/explain/axes/legend_guide.html)
- [Matplotlib Constrained layout guide](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html)
- [Matplotlib Tight layout guide](https://matplotlib.org/stable/users/explain/axes/tight_layout_guide.html)
- [Issue #11681: bbox='tight' does not respect figsize](https://github.com/matplotlib/matplotlib/issues/11681)
- [Issue #10194: legend not present with bbox_inches='tight'](https://github.com/matplotlib/matplotlib/issues/10194)
