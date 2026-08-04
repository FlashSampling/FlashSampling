# Modal vLLM run anomalies

Treat repetitions within one `vllm bench sweep serve` process as correlated measurements from one Modal host.
If a curve conflicts with kernel-level bounds, or unrelated metrics such as TTFT degrade together, rerun that provider as a fresh isolated sweep before drawing conclusions.

## Qwen3-1.7B concurrent sweep

The first concurrent Qwen3-1.7B TP1 B200 sweep produced implausible high-concurrency slowdowns for FMMS and FI2.
Isolated reruns did not reproduce them.
At concurrency 64, FI2 TPOT fell from 3.190 ms to 2.109 ms and TTFT fell from 28.102 ms to 11.403 ms.
The isolated FI2 curve increased only 0.236 ms from concurrency 1 to 64, consistent with the sub-0.45 ms FI2 kernel microbenchmark.

## Qwen3-8B retained result

The retained rebuttal experiment uses baseline `20260727_135427`, FI2 `20260727_141150`, and FMMS `20260727_154330`.
After taking the median over five runs per batch size and then across batch sizes 1 through 64, TPOT is 3.86 ms, 3.91 ms, and 3.72 ms, respectively.
A later independent battery was removed locally and from Modal because its low-batch provider ordering did not reproduce this experiment.

A combined Qwen3-8B invocation stopped after 27 minutes during FI2.
The baseline completed, FI2 reached batch size 8, and FMMS never started.
This was not the two-hour function timeout, and the cause was not established.
Use one app per provider and resume partial experiments when applicable.

## Interpretation details

- The `all` sweep covers concurrency 1, 2, 4, 8, 16, 32, and 64.
- Figure 5 and its rebuttal aggregate exclude 128 and 256.
- Dates in vLLM `summary.csv` are UTC.
- `modal app list --json` renders timestamps in the local timezone, so convert before comparing completion and app lifetime.
- Completed Qwen3-8B FI2 and FMMS apps stopped within five to six seconds of their final summary row.
