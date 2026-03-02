"""Parse Proton profiles to extract intra-kernel matmul vs sampling breakdown.

Triton Proton provides two complementary views:
  1. Chrome trace (warp-level scope timing): exact ns durations for user-annotated
     scopes (setup, matmul-tile, sample, store) across all warps and CTAs.
  2. Line-by-line PC sampling: % of samples attributed to each source line,
     which we categorize into matmul / sampling / setup phases.

Usage:
    python parse_proton_intrakernel.py --dir profiles/sweeps/bsz/proton/case-small
    python parse_proton_intrakernel.py --dir profiles/sweeps/bsz/proton/case-small > summary.txt

Expects directory layout:
    <dir>/bsz1/kernel.chrome_trace
    <dir>/bsz1/kernel-by-line.txt
    <dir>/bsz1/speed-test.txt
    <dir>/bsz4/...
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# ── Source-line categorization ──
# Lines from core.py are categorized by line number ranges.
# Lines from triton library files are categorized by file path.

CORE_MATMUL_RANGE = (275, 295)  # for loop + loads + tl.dot
CORE_SAMPLE_RANGE = (296, 362)  # temperature, gumbel noise, argmax, store

LIBRARY_CATEGORIES = {
    "random.py": "sampling",  # Philox RNG for tl.rand
    "standard.py": "sampling",  # tl.max / argmax reduction (+ swizzle2d, cdiv)
}


def categorize_line(filepath: str, lineno: int) -> str:
    """Assign a source line to matmul, sampling, or setup."""
    if "core.py" in filepath:
        if CORE_MATMUL_RANGE[0] <= lineno <= CORE_MATMUL_RANGE[1]:
            return "matmul"
        elif CORE_SAMPLE_RANGE[0] <= lineno <= CORE_SAMPLE_RANGE[1]:
            return "sampling"
        else:
            return "setup"
    for lib_file, cat in LIBRARY_CATEGORIES.items():
        if lib_file in filepath:
            return cat
    return "other"


def parse_line_by_line(path: Path) -> dict | None:
    """Parse kernel-by-line.txt and return categorized PC sample percentages."""
    if not path.exists():
        return None
    content = path.read_text()
    pattern = re.compile(r"[├└]─\s+([\d.]+)\s+[\d.nan]+\s+(.+):(\d+)@fused_mm_sample_triton_kernel")
    categories: dict[str, float] = defaultdict(float)
    lines_detail: list[tuple[str, str, int, float]] = []

    for m in pattern.finditer(content):
        pct = float(m.group(1))
        filepath = m.group(2)
        lineno = int(m.group(3))
        cat = categorize_line(filepath, lineno)
        categories[cat] += pct
        lines_detail.append((cat, filepath, lineno, pct))

    if not categories:
        return None
    return {"categories": dict(categories), "lines": lines_detail}


def parse_chrome_trace(path: Path) -> dict | None:
    """Parse chrome trace and return aggregate scope durations."""
    if not path.exists():
        return None
    data = json.load(path.open())
    events = data["traceEvents"]

    scope_dur: dict[str, float] = defaultdict(float)
    scope_cnt: dict[str, int] = defaultdict(int)

    for e in events:
        name = e["name"]
        scope_dur[name] += e["dur"]
        scope_cnt[name] += 1

    total = sum(scope_dur.values())
    if total == 0:
        return None
    n_warps = scope_cnt.get("setup", 0)

    return {
        "scopes": dict(scope_dur),
        "counts": dict(scope_cnt),
        "total_ns": total,
        "n_warp_events": n_warps,
    }


def parse_speed_test(path: Path) -> dict | None:
    """Extract config info from speed-test.txt."""
    if not path.exists():
        return None
    content = path.read_text()
    info = {}
    for line in content.splitlines():
        if "vocab_size" in line:
            info["vocab_size"] = int(line.split(":")[1].strip())
        elif "hidden_size" in line:
            info["hidden_size"] = int(line.split(":")[1].strip())
        elif line.strip().startswith("n_hidden_states:"):
            info["n_hidden_states"] = int(line.split(":")[1].strip())
        elif "GPU:" in line:
            info["gpu"] = line.split("GPU:")[1].strip()
    return info


SCOPE_ORDER = ["setup", "matmul-tile", "sample", "store"]
SCOPE_TO_PHASE = {
    "setup": "setup",
    "matmul-tile": "matmul",
    "sample": "sampling",
    "store": "store",
}


def trace_phase_pcts(trace: dict) -> dict[str, float]:
    """Return {matmul, sampling, setup} percentages from a chrome trace."""
    total = trace["total_ns"]
    phase_ns: dict[str, float] = defaultdict(float)
    for scope in SCOPE_ORDER:
        phase_ns[SCOPE_TO_PHASE[scope]] += trace["scopes"].get(scope, 0)
    # Merge store into sampling
    phase_ns["sampling"] += phase_ns.pop("store", 0)
    return {k: v / total * 100 for k, v in phase_ns.items()}


def lbl_phase_pcts(lbl: dict) -> dict[str, float]:
    """Return {matmul, sampling, setup} percentages from PC sampling (normalized)."""
    cats = lbl["categories"]
    total = sum(cats.values())
    if total == 0:
        return {}
    return {cat: cats.get(cat, 0) / total * 100 for cat in ["matmul", "sampling", "setup"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Proton sweep directory with bszN/ subdirs")
    args = parser.parse_args()

    base = Path(args.dir)
    bsz_dirs = sorted(base.glob("bsz*"), key=lambda p: int(p.name[3:]))

    if not bsz_dirs:
        print(f"No bszN/ directories found in {base}")
        return

    # ── Config from first bsz dir ──
    config = parse_speed_test(bsz_dirs[0] / "speed-test.txt")
    if config:
        print(f"GPU: {config.get('gpu', '?')}")
        print(f"Config: V={config.get('vocab_size', '?')}, d={config.get('hidden_size', '?')}")

    print(f"Proton Intra-Kernel Sweep: {base}")
    print("=" * 90)

    # ── Collect data ──
    traces: dict[int, dict] = {}
    lbls: dict[int, dict] = {}
    for d in bsz_dirs:
        bsz = int(d.name[3:])
        t = parse_chrome_trace(d / "kernel.chrome_trace")
        if t:
            traces[bsz] = t
        lbl = parse_line_by_line(d / "kernel-by-line.txt")
        if lbl:
            lbls[bsz] = lbl

    sorted_bsz = sorted(set(list(traces) + list(lbls)))

    # ── Section 1: Scope-level breakdown across batch sizes ──
    if traces:
        print()
        print("1. SCOPE-LEVEL BREAKDOWN (% of kernel time, from chrome trace)")
        print()
        print(
            f"   {'N':>4}  {'setup':>7}  {'matmul':>7}  {'sample':>7}  {'store':>7}  {'warp-events':>12}"
        )
        print(f"   {'-' * 55}")
        for bsz in sorted_bsz:
            t = traces.get(bsz)
            if not t:
                continue
            total = t["total_ns"]
            n = t["n_warp_events"]
            pcts = {s: t["scopes"].get(s, 0) / total * 100 for s in SCOPE_ORDER}
            print(
                f"   {bsz:>4}"
                f"  {pcts['setup']:>6.2f}%"
                f"  {pcts['matmul-tile']:>6.2f}%"
                f"  {pcts['sample']:>6.2f}%"
                f"  {pcts['store']:>6.2f}%"
                f"  {n:>12}"
            )

    # ── Section 2: Condensed matmul vs sampling ──
    if traces:
        print()
        print("2. MATMUL vs SAMPLING (condensed, from chrome trace)")
        print("   sampling = sample + store scopes")
        print()
        print(f"   {'N':>4}  {'matmul':>7}  {'sampling':>9}  {'setup':>7}  {'total (us)':>11}")
        print(f"   {'-' * 50}")
        for bsz in sorted_bsz:
            t = traces.get(bsz)
            if not t:
                continue
            pcts = trace_phase_pcts(t)
            n = t["n_warp_events"]
            # Per-warp total → approximate kernel time
            # total_ns is sum across all warps; kernel time ≈ total_ns / n_warps
            kernel_us = t["total_ns"] / n / 1000 if n else 0
            print(
                f"   {bsz:>4}"
                f"  {pcts['matmul']:>6.2f}%"
                f"  {pcts['sampling']:>8.2f}%"
                f"  {pcts['setup']:>6.2f}%"
                f"  {kernel_us:>11.2f}"
            )

    # ── Section 3: PC sampling breakdown ──
    if lbls:
        print()
        print("3. PC SAMPLING BREAKDOWN (normalized to kernel=100%)")
        print()
        print(f"   {'N':>4}  {'matmul':>7}  {'sampling':>9}  {'setup':>7}")
        print(f"   {'-' * 35}")
        for bsz in sorted_bsz:
            lbl = lbls.get(bsz)
            if not lbl:
                continue
            pcts = lbl_phase_pcts(lbl)
            if not pcts:
                continue
            print(
                f"   {bsz:>4}"
                f"  {pcts['matmul']:>6.2f}%"
                f"  {pcts['sampling']:>8.2f}%"
                f"  {pcts['setup']:>6.2f}%"
            )

    # ── Section 4: Per-scope absolute times ──
    if traces:
        print()
        print("4. PER-WARP AVERAGE TIME (ns) BY SCOPE")
        print()
        print(
            f"   {'N':>4}  {'setup':>8}  {'matmul':>8}  {'sample':>8}  {'store':>8}  {'total':>8}"
        )
        print(f"   {'-' * 50}")
        for bsz in sorted_bsz:
            t = traces.get(bsz)
            if not t:
                continue
            n = t["n_warp_events"]
            if not n:
                continue
            avgs = {s: t["scopes"].get(s, 0) / n for s in SCOPE_ORDER}
            total_avg = t["total_ns"] / n
            print(
                f"   {bsz:>4}"
                f"  {avgs['setup']:>8.2f}"
                f"  {avgs['matmul-tile']:>8.2f}"
                f"  {avgs['sample']:>8.2f}"
                f"  {avgs['store']:>8.2f}"
                f"  {total_avg:>8.2f}"
            )

    # ── Section 5: Top hotspots (from largest bsz with PC data) ──
    if lbls:
        # Pick the largest batch size for the hotspot detail
        largest_bsz = max(lbls)
        lbl = lbls[largest_bsz]
        print()
        print(f"5. TOP SOURCE LINES (N={largest_bsz}, >0.01% of kernel)")
        print(f"   {'%':>7}  {'Category':>9}  Source")
        print(f"   {'-' * 70}")
        sorted_lines = sorted(lbl["lines"], key=lambda x: -x[3])
        for cat, filepath, lineno, pct in sorted_lines:
            if pct < 0.01:
                break
            short = filepath.split("/")[-1]
            print(f"   {pct:>7.3f}  {cat:>9}  {short}:{lineno}")


if __name__ == "__main__":
    main()
