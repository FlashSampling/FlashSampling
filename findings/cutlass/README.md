# CUTLASS findings

The numeric prefixes record the order in which these findings entered the CUTLASS implementation work.
They keep the research context and completed gates in a stable reading order.

1. `00-2cta-mma-operand-swap-regression.md`
2. `01-fmms-kernel-plan.md`
3. `02-topk-softmax-epilogue.md`
4. `03-dsmem-cluster-reduction.md`
5. `04-accumulator-layout.md`
6. `05-thread-local-max.md`
7. `06-warp-max.md`
8. `07-cta-max.md`
9. `08-cta-multi-column-max.md`

Run a gate with `make modal-cutlass GATE=<gate>`.
The top-level `Makefile` contains the available gate names and maps each one to a numbered directory under `benchmarking/modal-results/cutlass/`.
