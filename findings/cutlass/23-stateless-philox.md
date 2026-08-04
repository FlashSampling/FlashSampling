# Gate 3 stateless Philox

Gate 3 validates a standalone Philox4x32-10 primitive on B200.
The counter is derived only from global `(sample_idx, hidden_idx, global_vocab_idx)` coordinates, and the 64-bit seed supplies the Philox key.
The implementation does not depend on block, warp, lane, scheduler, cluster, or tile order.

Run the canonical gate with:

```text
make modal-cutlass GATE=stateless-philox
```

The packet is under `benchmarking/modal-results/cutlass/17-stateless-philox/`.

## Correctness and statistics

The gate generated 1,048,576 full 128-bit Philox blocks for each of four seeds.
Three launch layouts covered 128, 256, and 512 threads with vocabulary tiles of 64, 128, and 256 elements.
All layouts produced identical output words.
No duplicate 128-bit blocks were found within any seed, and 32 GPU vectors matched the independent CPU implementation exactly.

Uniformity used 256 bins over the high byte of the first Philox word.
The predeclared family alpha was 0.01 with Bonferroni correction across four seeds, giving a per-test alpha of 0.0025.
All tests passed, and the minimum p-value was 0.1047.

## Cost profile

Nsight Compute measured the exact tile-native validation kernel.
The predeclared limits were at most 32 registers per thread, no local-memory load or store instructions, at most 6 executed warp instructions per output block, and no MUFU instructions.
The kernel used 20 registers, no local-memory instructions, and 3.3125 executed warp instructions per output block.
Issue activity was 62.55% of peak sustained active.
Blackwell NCU exposes a shared XU/SFU pipe counter rather than a MUFU-specific runtime counter, so the packet pairs that counter with a kernel-scoped SASS audit.
XU/SFU utilization was 0%, and the kernel contained no `MUFU` opcodes.

An earlier flat-index harness introduced three `MUFU` instructions through runtime integer-division lowering.
That cost was an artifact of decoding `(sample, hidden, vocab)` from one flat index.
The accepted harness uses a 3D `(vocab tile, hidden, sample)` grid, matching the coordinates already available to a CUTLASS callback and removing the artificial SFU work.

## Handoff

Gate 3 passes.
Gate 4 must integrate this exact global-coordinate primitive and the Gumbel transform into the correctness-approved B200 greedy donor.
Gate 4 must rerun deterministic stream checks, the 10M-sample large-vocabulary distribution test, and matched greedy-versus-Gumbel NCU and timing profiles before sampling is approved.
