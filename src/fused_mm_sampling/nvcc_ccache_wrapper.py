#!/usr/bin/env python3
"""Apply controlled per-translation-unit inputs before invoking ccache and NVCC."""

import os
import sys
from pathlib import Path


def main() -> None:
    arguments = sys.argv[1:]
    source = next(Path(value) for value in arguments if value.endswith(".cu"))
    mode = os.environ["FMMS_CCACHE_PROBE_MODE"]
    if source.name == "greedy_provider.cu":
        arguments.extend(["-include", os.environ["FMMS_CCACHE_PROBE_HEADER"]])
    if mode == "feature-flag":
        arguments.append("-DFMMS_CCACHE_PROBE_VALUE=2")
    real_nvcc = os.environ["FMMS_REAL_NVCC"]
    os.execvp("ccache", ["ccache", real_nvcc, *arguments])


if __name__ == "__main__":
    main()
