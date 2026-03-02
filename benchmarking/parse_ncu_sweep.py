"""Parse NCU text exports from a batch-size sweep and produce a summary.

Usage:
    python parse_ncu_sweep.py --dir profiles/sweeps/bsz/ncu-txt/case-small
    python parse_ncu_sweep.py --dir profiles/sweeps/bsz/ncu-txt/case-small > summary.txt

Expects directory layout:
    <dir>/bsz1/fused-triton.txt
    <dir>/bsz1/naive-compiled.txt
    <dir>/bsz1/flashinfer-sampling.txt        (optional)
    <dir>/bsz1/flashinfer-top-k-top-p.txt     (optional)
    <dir>/bsz4/fused-triton.txt
    ...
"""

import argparse
import re
from pathlib import Path


def parse_durations(path: Path) -> list[float]:
    """Extract all Duration values (in us) from an NCU text export."""
    content = path.read_text()
    durations = re.findall(r"Duration\s+(us|ms)\s+([\d,.]+)", content)
    times_us = []
    for unit, val in durations:
        t = float(val.replace(",", ""))
        if unit == "ms":
            t *= 1000
        times_us.append(t)
    return times_us


def parse_metric(path: Path, metric: str) -> str | None:
    """Extract first occurrence of a metric value from NCU text export."""
    content = path.read_text()
    m = re.search(rf"{re.escape(metric)}\s+[\w/%]+\s+([\d,.]+)", content)
    return m.group(1).replace(",", "") if m else None


def parse_kernel_names(path: Path) -> list[str]:
    """Extract kernel names from NCU Profiling lines."""
    content = path.read_text()
    return re.findall(r'Profiling "([^"]+)"', content)


def parse_method(path: Path) -> dict | None:
    """Parse an NCU text export and return summary metrics."""
    if not path.exists():
        return None
    times = parse_durations(path)
    if not times:
        return None
    total = sum(times)
    bw = parse_metric(path, "DRAM Throughput")
    return {
        "total_us": total,
        "matmul_us": times[0],
        "post_us": sum(times[1:]),
        "n_kernels": len(times),
        "bw_pct": bw,
        "times": times,
        "kernel_names": parse_kernel_names(path),
    }


# Methods to look for in each bszN/ directory: (filename, display name)
METHODS = [
    ("fused-triton.txt", "FMMS"),
    ("naive-compiled.txt", "Naive"),
    ("flashinfer-sampling.txt", "FI-sample"),
    ("flashinfer-top-k-top-p.txt", "FI-topkp"),
]


def fmt_bw(m: dict) -> str:
    return f"{float(m['bw_pct']):.0f}" if m.get("bw_pct") else "?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Directory with bszN/ subdirs")
    args = parser.parse_args()

    base = Path(args.dir)
    bsz_dirs = sorted(base.glob("bsz*"), key=lambda p: int(p.name[3:]))

    if not bsz_dirs:
        print(f"No bszN/ directories found in {base}")
        return

    print(f"NCU Batch-Size Sweep Summary: {base}")
    print("=" * 100)

    # Detect which methods are present
    present = []
    for fname, label in METHODS:
        if any((d / fname).exists() for d in bsz_dirs):
            present.append((fname, label))

    # Collect data: bsz -> {label: parsed}
    data: dict[int, dict[str, dict]] = {}
    for d in bsz_dirs:
        bsz = int(d.name[3:])
        data[bsz] = {}
        for fname, label in present:
            parsed = parse_method(d / fname)
            if parsed:
                data[bsz][label] = parsed

    sorted_bsz = sorted(data)

    # ── Section 1: Total time per method ──
    print()
    print("1. TOTAL TIME (us)")
    print()
    hdr = f"{'N':>4}"
    for _, label in present:
        hdr += f"  {label:>11}"
    print(hdr)
    print("-" * 100)
    for bsz in sorted_bsz:
        row = f"{bsz:>4}"
        for _, label in present:
            m = data[bsz].get(label)
            row += f"  {m['total_us']:>11.1f}" if m else f"  {'—':>11}"
        print(row)

    # ── Section 2: Speedup vs FMMS ──
    has_fmms = "FMMS" in {lab for _, lab in present}
    baselines = [(f, lab) for f, lab in present if lab != "FMMS"]

    if has_fmms and baselines:
        print()
        print("2. SPEEDUP (baseline_total / fmms_total)")
        print()
        hdr2 = f"{'N':>4}"
        for _, label in baselines:
            hdr2 += f"  {label:>11}"
        print(hdr2)
        print("-" * 100)
        for bsz in sorted_bsz:
            fmms = data[bsz].get("FMMS")
            if not fmms:
                continue
            row = f"{bsz:>4}"
            for _, label in baselines:
                m = data[bsz].get(label)
                if m:
                    sp = m["total_us"] / fmms["total_us"]
                    marker = " " if sp > 1.0 else "*"
                    row += f"  {sp:>10.2f}x{marker}"
                else:
                    row += f"  {'—':>11} "
            print(row)
        print("  (* = FMMS is slower than baseline)")

    # ── Section 3: Matmul vs post-matmul decomposition ──
    print()
    print("3. DECOMPOSITION: matmul vs post-matmul (us)")
    print()
    print(
        f"{'N':>4}  {'cuBLAS MM':>10}  {'cuBLAS BW%':>10}"
        f"  {'FMMS MM':>10}  {'FMMS BW%':>10}  {'MM delta':>10}"
    )
    print("-" * 100)
    for bsz in sorted_bsz:
        fmms = data[bsz].get("FMMS")
        # Use any baseline for cuBLAS matmul (they all share it)
        cublas_m = None
        for _, label in baselines:
            cublas_m = data[bsz].get(label)
            if cublas_m:
                break
        if not fmms or not cublas_m:
            continue
        delta = fmms["matmul_us"] - cublas_m["matmul_us"]
        sign = "+" if delta >= 0 else ""
        print(
            f"{bsz:>4}  {cublas_m['matmul_us']:>10.0f}  {fmt_bw(cublas_m):>10}"
            f"  {fmms['matmul_us']:>10.0f}  {fmt_bw(fmms):>10}  {sign}{delta:>9.0f}"
        )

    # ── Section 4: Post-matmul cost per baseline ──
    if baselines:
        print()
        print("4. POST-MATMUL COST — eliminated by FMMS")
        print()
        hdr4 = f"{'N':>4}"
        for _, label in baselines:
            hdr4 += f"  {label + ' post':>11}  {'%MM':>4}  {'%post':>5}  {'#K':>3}"
        print(hdr4)
        print("-" * 100)
        for bsz in sorted_bsz:
            row = f"{bsz:>4}"
            for _, label in baselines:
                m = data[bsz].get(label)
                if m:
                    pct_mm = m["matmul_us"] / m["total_us"] * 100
                    pct_post = m["post_us"] / m["total_us"] * 100
                    row += (
                        f"  {m['post_us']:>11.1f}"
                        f"  {pct_mm:>3.0f}%  {pct_post:>4.0f}%"
                        f"  {m['n_kernels'] - 1:>3}"
                    )
                else:
                    row += f"  {'—':>11}  {'—':>4}  {'—':>5}  {'—':>3}"
            print(row)
        print()
        print("  FMMS is a single kernel — NCU cannot split matmul vs sampling time within it.")

    # ── Section 5: Net advantage breakdown ──
    if has_fmms and baselines:
        print()
        print("5. NET ADVANTAGE BREAKDOWN vs each baseline (us)")
        print("   saved = post-matmul eliminated − matmul overhead")
        for _, bl_label in baselines:
            print()
            print(f"  vs {bl_label}:")
            print(
                f"  {'N':>4}  {'post elim':>10}  {'MM overhead':>11}"
                f"  {'net saved':>10}  {'speedup':>8}"
            )
            print(f"  {'-' * 55}")
            for bsz in sorted_bsz:
                fmms = data[bsz].get("FMMS")
                bl = data[bsz].get(bl_label)
                if not fmms or not bl:
                    continue
                post_elim = bl["post_us"]
                mm_overhead = fmms["matmul_us"] - bl["matmul_us"]
                net = post_elim - mm_overhead
                sp = bl["total_us"] / fmms["total_us"]
                print(
                    f"  {bsz:>4}  {post_elim:>+10.0f}  {mm_overhead:>+11.0f}"
                    f"  {net:>+10.0f}  {sp:>7.2f}x"
                )

    # ── Section 6: Per-bsz kernel breakdown ──
    print()
    print("6. KERNEL BREAKDOWNS")
    for d in bsz_dirs:
        bsz = int(d.name[3:])
        for fname, label in present:
            m = data[bsz].get(label)
            if not m or m["n_kernels"] <= 1:
                continue

            print()
            print(f"--- {label} kernel breakdown (N={bsz}) ---")
            print(f"  {'#':>3}  {'Duration(us)':>12}  Kernel")
            print(f"  {0:>3}  {m['times'][0]:>12.1f}  cuBLAS matmul")
            for i, (name, t) in enumerate(zip(m["kernel_names"], m["times"][1:]), 1):
                print(f"  {i:>3}  {t:>12.1f}  {name[:65]}")


if __name__ == "__main__":
    main()
