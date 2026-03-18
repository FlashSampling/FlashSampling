#!/usr/bin/env python3
"""Insert proton.record statements into the FMMS Triton kernel's TTGIR.

Adapted from the Triton intra-kernel profiling tutorial:
https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/intra_kernel/insert_proton_records

In the persistent kernel, the compiler fuses the D-loop and persistent tile
loop into one scf.for. Every HIDDEN_SIZE/BLOCK_SIZE_D iterations, an scf.if
epilogue fires with masking, sampling, and store. We instrument:

  - "kernel":    the full kernel from tt.func entry to tt.return
  - "setup":     kernel entry through just before the persistent scf.for
                 (temperature load, grid dims, TMA descriptors, first tile prefetch)
  - "mask":      V-masking and temperature scaling
  - "tile-mgmt": next-tile coordinate computation (swizzle grouping)
  - "sample":    Gumbel noise generation and argmax reduce
  - "store":     global index offset, address computation, and tt.store ops

Matmul time is derived as:
  kernel - setup - mask - tile-mgmt - sample - store.

Usage:
    # After dump_ttgir.sh has populated ttgir_dump/:
    python insert_proton_records.py [--dir ttgir_dump]
"""

import argparse
import glob
import os
import re
import sys


def add_proton_records(input_file: str) -> None:
    """Add proton.record statements to the FMMS kernel TTGIR file.

    Uses a two-pass approach: first scan for insertion points, then insert
    from bottom to top so line indices remain stable.
    """
    with open(input_file) as f:
        lines = f.readlines()

    if any("proton.record" in line for line in lines):
        raise AssertionError(
            "File already contains `proton.record` statements! "
            "Re-dump the TTGIR without proton scopes first."
        )

    insertions = _find_insertion_points(lines)
    _validate_insertions(insertions, input_file)

    # Insert from bottom to top so earlier indices stay valid.
    # At the same line index, the last-processed item ends up first in the file.
    # Secondary key: "end" before "start" in the output (so end=0, start=1).
    # Tertiary key: "kernel" is the outermost scope, so its start must appear
    # first (processed last → nesting=0 for start) and its end must appear
    # last (processed first → nesting=0 for end).
    def _sort_key(item):
        line_idx, text = item
        is_start = 1 if text.startswith("start") else 0
        is_kernel = '"kernel"' in text
        # For starts at same line: kernel (nesting=0) processed last → first in file
        # For ends at same line: kernel (nesting=1) processed first → last in file
        nesting = (0 if is_kernel else 1) if is_start else (1 if is_kernel else 0)
        return (line_idx, is_start, nesting)

    for line_idx, text in sorted(insertions, key=_sort_key, reverse=True):
        lines.insert(line_idx, f"      proton.record {text} loc(#loc1)\n")

    with open(input_file, "w") as f:
        f.writelines(lines)

    scopes = sorted({t.split('"')[1] for _, t in insertions})
    print(f"Added proton records to {input_file}: {scopes}")


def _find_insertion_points(lines: list[str]) -> list[tuple[int, str]]:
    """Scan the TTGIR and return (line_index, record_text) pairs.

    Each pair means "insert this proton.record line BEFORE lines[line_index]".
    """
    insertions: list[tuple[int, str]] = []

    # ── Kernel ──
    # Full kernel scope from entry to return.
    func_line = _find_first(lines, "tt.func public")
    return_line = _find_first(lines, "tt.return")
    if func_line is not None:
        insertions.append((func_line + 1, 'start "kernel"'))
    if return_line is not None:
        insertions.append((return_line, 'end "kernel"'))

    # ── Setup ──
    # From kernel entry (after tt.func) to just before the persistent scf.for.
    # Covers: temperature load, grid dims, TMA descriptors, first tile prefetch.
    persistent_for = _find_first(lines, "scf.for")
    if func_line is not None:
        insertions.append((func_line + 1, 'start "setup"'))
    if persistent_for is not None:
        insertions.append((persistent_for, 'end "setup"'))

    # Find the epilogue scf.if: the one whose body contains tt.reduce.
    # This is the guard that fires when a tile's matmul is complete.
    epilogue_if = _find_epilogue_if(lines)
    if epilogue_if is None:
        print("WARNING: could not find epilogue scf.if", file=sys.stderr)
        return insertions

    # ── Mask ──
    # V-masking and temperature scaling (first ops in the epilogue).
    divf_line = _find_first_after(lines, "arith.divf", epilogue_if)
    if epilogue_if is not None:
        insertions.append((epilogue_if + 1, 'start "mask"'))
    if divf_line is not None:
        insertions.append((divf_line + 1, 'end "mask"'))

    # ── Tile-mgmt ──
    # Next-tile coordinate computation (swizzle grouping). Sits between
    # temperature scaling and Gumbel noise seed.
    gumbel_line = _find_first_after(lines, "gumbel_noise", epilogue_if)
    if divf_line is not None and gumbel_line is not None:
        insertions.append((divf_line + 1, 'start "tile-mgmt"'))
        insertions.append((gumbel_line, 'end "tile-mgmt"'))

    # ── Sample ──
    # Gumbel noise generation through tt.reduce (argmax).
    reduce_close = _find_reduce_close(lines)
    if gumbel_line is not None:
        insertions.append((gumbel_line, 'start "sample"'))
    if reduce_close is not None:
        insertions.append((reduce_close + 1, 'end "sample"'))

    # ── Store ──
    # From global index offset (after tt.reduce) through last tt.store.
    # Includes address computation, masking, and the stores themselves.
    store_start = _find_first_after(lines, "tt.splat", reduce_close)
    store_end = _find_last(lines, "tt.store")
    if store_start is not None:
        insertions.append((store_start, 'start "store"'))
    if store_end is not None:
        insertions.append((store_end + 1, 'end "store"'))

    return insertions


def _find_epilogue_if(lines: list[str]) -> int | None:
    """Find the scf.if that guards the epilogue (masking + sample + store).

    This is the scf.if whose body contains a tt.reduce (the argmax).
    """
    for i, line in enumerate(lines):
        if "scf.if" not in line:
            continue
        # Look ahead to see if this scf.if contains a tt.reduce
        for j in range(i + 1, min(i + 300, len(lines))):
            if '"tt.reduce"' in lines[j]:
                return i
            # Stop if we hit the matching else/closing brace at the same nesting
            if "} else {" in lines[j] or lines[j].strip().startswith("} loc("):
                break
    return None


def _find_reduce_close(lines: list[str]) -> int | None:
    """Find the closing line of the last tt.reduce block.

    Returns the index of the `}) : ...` line after tt.reduce.return.
    """
    last_reduce_return = _find_last(lines, "tt.reduce.return")
    if last_reduce_return is None:
        return None
    for i in range(last_reduce_return + 1, len(lines)):
        if re.search(r"\}\)", lines[i]):
            return i
    return None


def _find_first(lines: list[str], pattern: str) -> int | None:
    for i, line in enumerate(lines):
        if pattern in line:
            return i
    return None


def _find_first_after(lines: list[str], pattern: str, after: int | None) -> int | None:
    start = (after or 0) + 1
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return None


def _find_last(lines: list[str], pattern: str) -> int | None:
    result = None
    for i, line in enumerate(lines):
        if pattern in line:
            result = i
    return result


def _validate_insertions(insertions: list[tuple[int, str]], filepath: str) -> None:
    """Check that we found all expected start/end pairs."""
    texts = [t for _, t in insertions]
    for scope in ["kernel", "setup", "mask", "tile-mgmt", "sample", "store"]:
        has_start = f'start "{scope}"' in texts
        has_end = f'end "{scope}"' in texts
        if has_start != has_end:
            print(
                f"WARNING: {filepath}: found {'start' if has_start else 'end'} "
                f"but not {'end' if has_start else 'start'} for scope {scope!r}",
                file=sys.stderr,
            )
        if not has_start and not has_end:
            print(
                f"WARNING: {filepath}: could not find scope {scope!r}",
                file=sys.stderr,
            )


def find_and_process_ttgir(dump_dir: str) -> None:
    """Find all FMMS kernel TTGIR files in the dump directory and process them."""
    if not os.path.isdir(dump_dir):
        print(f"Error: directory not found: {dump_dir}", file=sys.stderr)
        sys.exit(1)

    ttgir_files = glob.glob(os.path.join(dump_dir, "**", "*.ttgir"), recursive=True)
    if not ttgir_files:
        print(f"No .ttgir files found in {dump_dir}", file=sys.stderr)
        sys.exit(1)

    fmms_files = [f for f in ttgir_files if "fused_mm_sample" in os.path.basename(f)]
    if not fmms_files:
        print(
            f"No fused_mm_sample*.ttgir files found in {dump_dir}. "
            f"Found: {[os.path.basename(f) for f in ttgir_files]}",
            file=sys.stderr,
        )
        sys.exit(1)

    for ttgir_file in fmms_files:
        try:
            print(f"Processing {ttgir_file}...")
            add_proton_records(ttgir_file)
        except AssertionError as e:
            print(f"Skipping {ttgir_file}: {e}")
        except Exception as e:
            print(f"Error processing {ttgir_file}: {e}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default="ttgir_dump",
        help="Directory containing TTGIR dump files (default: ttgir_dump)",
    )
    args = parser.parse_args()
    find_and_process_ttgir(args.dir)
