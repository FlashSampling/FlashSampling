# CUTLASS findings

Numeric prefixes preserve the order in which findings entered the CUTLASS implementation work.
The directory listing is the index, so this file intentionally does not duplicate individual finding names.

Read the mutable handoff in this directory before continuing implementation.
Use the numbered findings for the evidence behind completed gates, rejected experiments, infrastructure problems, and design decisions.

Run a gate with `make modal-cutlass GATE=<gate>`.
The top-level `Makefile` contains the available gate names and maps each one to its result directory under `benchmarking/modal-results/cutlass/`.
