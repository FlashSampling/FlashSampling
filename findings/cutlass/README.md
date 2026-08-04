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
10. `09-cta-boundary-max.md`
11. `10-evt-candidates.md`
12. `11-stage2.md`
13. `12-greedy-provider.md`
14. `13-void-d-epilogue.md`
15. `14-greedy-performance.md`
16. `15-greedy-profile-stage2.md`
17. `16-ordinary-gemm-specialization.md`
18. `17-ordinary-gemm-tuning.md`
19. `18-ordinary-gemm-stage-no-go.md`
20. `19-gemm-heuristics.md`
21. `20-winning-schedule-accumulator-layout.md`
22. `21-winning-schedule-evt.md`
23. `22-winning-schedule-performance.md`
24. `23-stateless-philox.md`
25. `24-gumbel-max-tp1.md`
26. `25-development-infrastructure.md`

The active roadmap in `01-fmms-kernel-plan.md` is B200-first.
Read `current-handoff.md` for the active gate, blocker, production dispatch, and next bounded experiments.
Read `25-development-infrastructure.md` for the staged process-optimization roadmap and its empirical admission rules.

Run a gate with `make modal-cutlass GATE=<gate>`.
The top-level `Makefile` contains the available gate names and maps each one to a numbered directory under `benchmarking/modal-results/cutlass/`.
