#!/usr/bin/env python3
"""Give each NVCC translation unit a distinct explicit device-trace file."""

import os
import sys
from pathlib import Path


def main() -> None:
    real_nvcc = os.environ["FMMS_REAL_NVCC"]
    trace_dir = Path(os.environ["FMMS_NVCC_TRACE_DIR"])
    arguments = sys.argv[1:]
    if os.environ.get("FMMS_NVCC_TRACE_PTX_ONLY") == "1":
        arguments[arguments.index("-c")] = "--ptx"
    try:
        output = Path(arguments[arguments.index("-o") + 1]).name
    except (ValueError, IndexError) as error:
        raise RuntimeError("The NVCC trace wrapper requires an -o output") from error
    trace_base = trace_dir / f"{output}.device-time-trace"
    os.execv(
        real_nvcc,
        [real_nvcc, *arguments, f"--fdevice-time-trace={trace_base}"],
    )


if __name__ == "__main__":
    main()
