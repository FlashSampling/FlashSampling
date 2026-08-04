# Rebuttal benchmark summary

This note preserves benchmark facts that were previously mixed into `AGENTS.md`.
Use the underlying result packets for decisions, and rerun measurements when the code or environment has materially changed.

## TP overlap ablation

Use `fused-triton-p2p-no-overlap` to isolate TP communication overlap.
It writes candidates locally in FMMS and then uses a separate Triton kernel to fan out the same values and indices to peer symmetric-memory buffers before the existing barrier and reduction.
The default `fused-triton` provider performs the same remote stores inside FMMS.
The ablation therefore holds the P2P mechanism, payload, destination layout, and reduction fixed while moving communication after computation.
`make plot-tp-scaling` includes this provider as `FlashSampling (P2P No Overlap)`.

The superseded NCCL all-gather experiment is under `benchmarking/modal-results/triton-bench/own/{b200,b200-rerun1,...,b200-rerun9}/tp2/` when the ignored result packets are present.
It reduced locally before exchanging candidates and changed the transport, so it did not isolate overlap.

The TP2 distributed correctness suite passed for V=100, 256, and 512 with H=1 and 2 on B200.
A TP2 large-configuration CUDA-event sweep measured a 0.039 to 0.049 ms reduction at H=1 through 128 and a 0.004 ms reduction at H=256.
The paired results are under `triton-bench/own/b200/tp2/fused-mm-sample-batch-scaling-large.csv`.

Multiple B200 reruns did not support a relationship between NUMA placement and the overlap result.
Both same-NUMA and split-NUMA TP2 groups contained near-zero and roughly 0.046 ms differences.
TP4 and TP8 results also showed overlap gains across different placements.
Launcher generation remains a confounder because the newer split-NUMA TP2 runs used torchrun binding while older runs did not.

For reviewer Q4, define overlap speedup as `latency(no overlap) / latency(overlap)`.
Average across batch sizes within each run before summarizing runs because pointwise minima can fall below one due to noise.
Interpret the growing TP benefit through increasing P2P fan-out rather than as proof that P2P alone is effective.

The checked-in Figure 3 summary reports average FMMS speedups of 2.24x over compiled multinomial sampling and 1.63x over FI2 across TP1, TP2, TP4, and TP8.
Mean overlap speedup is 1.13x across TP2, TP4, and TP8, contributing about 10% and 21% of the respective excess speedups.
See `tp-scaling-fast-pod-b200.md` for host-class hazards and the full scaling interpretation.

## Large-vocabulary distribution

At V=128,000 with 10 million samples on B200, bfloat16 per-tile maxima introduced measurable bias.
Float32 maxima passed with reduced chi-squared 0.99844, p=0.6503, and 99.84% covered probability mass.
The RNG also uses separate sample streams and unique tile-element offsets to avoid collisions.

## Memory traffic

Run the memory profile with:

```bash
make modal-memory-traffic-all CASE=large N_HIDDEN_STATES=64
```

It profiles FMMS, its `return_logits=True` ablation, and the three paper baselines concurrently.
Each provider produces `report.ncu-rep`, `traffic.csv`, `memory.json`, and `log.txt`, and `parse-memory-traffic` aggregates the results with pandas.

On B200 with the large case at B=1, 64, and 256, FMMS used 0.05, 0.77, and 2.97 MiB of peak temporary memory.
That was a 98.48% to 99.53% reduction against the three baselines.
The HBM-read reduction grew from 0.17% to 0.33% at B=1 to 4.33% to 26.52% at B=256.
The HBM-write reduction grew from 37.98% to 43.30% at B=1 to 95.73% to 97.94% at B=256.

For rebuttal Q3, use FP32 logits consistently.
At B=64 and V=128,256, full logits use 31.31 MiB, while theoretical FP32-value and int64-index candidates use 0.734 MiB and measured candidates use 0.77 MiB.
Validate the `2B/D` I/O term by toggling only the FP32 logits store inside FMMS.

A paired B200 NCU profile at B=64 found that `return_logits=True` left reads unchanged, added 27.45 MiB of physical HBM writes, and added 32.00 MiB of peak temporary allocation.
The added writes were below the 31.31 MiB logical FP32 logits size, so excess DRAM bytes did not explain the timing gap.
Kernel-duration and execution-metric profiling was still required to identify the cause.
