"""LLM-as-a-judge evaluation for GSM8K samples.

lm-eval's regex scoring misses answers in \\boxed{X} format. This script
re-evaluates the JSONL samples using a judge LLM to get real accuracy.

Usage:
    # Single variant:
    .venv/bin/python benchmarking/vllm/judge_eval.py benchmarking/vllm/eval/baseline

    # Compare variants:
    .venv/bin/python benchmarking/vllm/judge_eval.py eval/baseline eval/fmms-triton

Requires OPENROUTER_API_KEY env var.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

JUDGE_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
CONCURRENCY = 16
TAIL_CHARS = 500
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0

JUDGE_PROMPT = """\
The correct answer to a math problem is: {reference}

Here is the end of a model's response to that problem:
---
{response_tail}
---

Does the model's final answer equal {reference}? Reply YES or NO only."""


BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42


async def main():
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY env var not set", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(base_url=JUDGE_BASE_URL, api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY)

    # {name: {doc_id: bool}}
    all_results = {}
    for eval_dir in args.eval_dirs:
        name = Path(eval_dir).name
        all_results[name] = await judge_dir(eval_dir, client, sem)

    print()
    print_results(all_results)


def print_results(all_results: dict[str, dict[int, bool]]):
    """Print per-variant accuracy with bootstrap CIs, and pairwise comparisons."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = list(all_results.keys())

    # Per-variant accuracy + CI
    for name in names:
        by_doc = all_results[name]
        correct = sum(by_doc.values())
        total = len(by_doc)
        arr = np.array(list(by_doc.values()), dtype=np.float64)
        lo, hi = bootstrap_ci(arr, rng)
        print(
            f"{name}: {correct}/{total} correct ({100 * correct / total:.1f}%, 95% CI [{lo:.1f}%, {hi:.1f}%])"
        )

    # Pairwise paired bootstrap for each pair
    if len(names) >= 2:
        print()
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, nb = names[i], names[j]
                paired_bootstrap(all_results[na], all_results[nb], na, nb, rng)


def bootstrap_ci(arr: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of a binary array. Returns (lo%, hi%)."""
    n = len(arr)
    idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return 100 * lo, 100 * hi


def paired_bootstrap(
    results_a: dict[int, bool],
    results_b: dict[int, bool],
    name_a: str,
    name_b: str,
    rng: np.random.Generator,
):
    """Paired bootstrap test: is there a significant difference between two variants?

    Both variants answer the same set of questions, so we use a *paired* test:
    compute per-question differences diff[i] = A[i] - B[i] (each is +1, -1, or 0),
    then bootstrap the mean of that difference vector.  This cancels out shared
    question difficulty — if both variants get the same 150 hard questions wrong
    and only differ on 3 borderline ones, the paired test sees the tiny signal
    clearly, while an unpaired test would be swamped by shared noise.

    Concretely:
    1. Align on shared doc_ids → two boolean arrays of length n.
    2. diff = A - B.  Observed accuracy gap = mean(diff).
    3. Draw BOOTSTRAP_N resamples (with replacement) of size n from diff,
       compute the mean of each → distribution of plausible gaps.
    4. 95% CI = 2.5th / 97.5th percentiles of bootstrap means.
    5. Two-sided p-value = 2 * fraction of bootstrap means with opposite sign
       to the observed gap (clamped to 1.0).
    """
    shared_ids = sorted(set(results_a) & set(results_b))
    if not shared_ids:
        print(f"No shared doc_ids between {name_a} and {name_b}")
        return

    arr_a = np.array([results_a[d] for d in shared_ids], dtype=np.float64)
    arr_b = np.array([results_b[d] for d in shared_ids], dtype=np.float64)
    diff = arr_a - arr_b  # per-question difference

    observed_diff = diff.mean()
    n = len(diff)
    idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
    boot_diffs = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_diffs, [2.5, 97.5])

    # Two-sided p-value: fraction of bootstrap samples where sign flips
    if observed_diff >= 0:
        p_value = (boot_diffs <= 0).mean()
    else:
        p_value = (boot_diffs >= 0).mean()
    p_value = min(2 * p_value, 1.0)  # two-sided

    sig = "" if p_value < 0.05 else " (not significant)"
    print(
        f"{name_a} vs {name_b}: "
        f"diff = {100 * observed_diff:+.1f}pp, "
        f"95% CI [{100 * lo:+.1f}pp, {100 * hi:+.1f}pp], "
        f"p = {p_value:.3f}{sig}"
    )
    print(f"  ({len(shared_ids)} shared questions, {BOOTSTRAP_N:,} bootstrap samples)")


async def judge_dir(eval_dir: str, client: AsyncOpenAI, sem: asyncio.Semaphore) -> dict[int, bool]:
    """Judge all samples in an eval directory. Returns {doc_id: correct}."""
    samples_path = find_samples(eval_dir)
    if not samples_path:
        print(f"No samples JSONL found in {eval_dir}", file=sys.stderr)
        return {}

    samples = load_samples(samples_path)
    print(f"Judging {len(samples)} samples from {samples_path}")

    results_path = samples_path.parent / samples_path.name.replace("samples_", "judge_")
    existing = load_existing_results(results_path)
    if existing:
        print(
            f"  Resuming: {len(existing)} already judged, {len(samples) - len(existing)} remaining"
        )

    pending = []
    for sample in samples:
        doc_id = sample["doc_id"]
        if doc_id in existing:
            continue
        ref = extract_reference(sample["doc"]["answer"])
        pending.append(judge_one(sample, ref, client, sem))

    n_existing = len(existing)
    done = 0
    total = n_existing + len(pending)

    with open(results_path, "a") as f:
        for coro in asyncio.as_completed(pending):
            r = await coro
            f.write(json.dumps(r) + "\n")
            f.flush()
            existing[r["doc_id"]] = r
            done += 1
            if done % 100 == 0:
                print(f"  {n_existing + done}/{total} judged...")

    return {doc_id: r["correct"] for doc_id, r in existing.items()}


async def judge_one(
    sample: dict, ref_answer: str, client: AsyncOpenAI, sem: asyncio.Semaphore
) -> dict:
    """Ask the judge LLM whether the model's response matches the reference."""
    doc_id = sample["doc_id"]
    response_text = sample["resps"][0][0]
    response_tail = response_text[-TAIL_CHARS:]

    prompt = JUDGE_PROMPT.format(reference=ref_answer, response_tail=response_tail)

    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                completion = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8,
                    temperature=0,
                )
                judgment = completion.choices[0].message.content.strip()
                break
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    await asyncio.sleep(delay)
                    continue
                print(f"  doc_id={doc_id}: API error: {e}", file=sys.stderr)
                judgment = "ERROR"
                break

    correct = judgment.upper().startswith("YES")
    return {"doc_id": doc_id, "correct": correct, "judgment": judgment}


def find_samples(eval_dir: str) -> Path | None:
    """Find the latest samples_gsm8k*.jsonl under eval_dir."""
    pattern = os.path.join(eval_dir, "**", "samples_gsm8k*.jsonl")
    matches = sorted(glob(pattern, recursive=True))
    if not matches:
        return None
    return Path(matches[-1])


def load_samples(path: Path) -> list[dict]:
    """Load samples from a JSONL file, deduplicated by doc_id.

    lm-eval writes one line per (doc_id, filter) combination. The response
    text is the same across filters, so we keep only the first occurrence.
    """
    seen = set()
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            doc_id = sample["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                samples.append(sample)
    return samples


def load_existing_results(path: Path) -> dict[int, dict]:
    """Load already-judged results keyed by doc_id, for resume support."""
    results = {}
    if not path.exists():
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                results[r["doc_id"]] = r
    return results


def extract_reference(doc_answer: str) -> str:
    """Extract the final numeric answer after '#### ' in the GSM8K answer string."""
    match = re.search(r"####\s*(.+)$", doc_answer)
    if match:
        return match.group(1).strip()
    return doc_answer.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-a-judge for GSM8K eval samples")
    parser.add_argument("eval_dirs", nargs="+", help="One or more eval directories to judge")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
