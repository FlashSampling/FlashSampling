# Register spilling at batch size 256

## Problem

At `n_hidden_states=256`, the FMMS Triton kernel spills 118 MB to local memory on RTX 3090 (SM 8.6), accounting for 76% of all L1TEX sector requests and 17% of executed instructions.
The kernel runs at 20 ms, significantly slower than expected.

## Root cause

With `BLOCK_SIZE_V=128, BLOCK_SIZE_H=64` (returned by `bsz_h(256)`), and `#mma<warpsPerCTA=[2,2]>`, each thread holds ~64 elements of each `[128, 64]` f32 tensor.
The persistent tile loop carries the logits accumulator as an `scf.for` iter_arg (yielded at the end of each iteration for the next tile's zero-init).
At the point where Gumbel noise is added, three `[128, 64]` f32 tensors are simultaneously live:

| Value | What | Regs/thread |
|-------|------|-------------|
| Next-iteration accumulator | iter_arg, live from definition to yield | ~64 |
| Scaled logits | `logits_blk / temperature` | ~64 |
| Gumbel noise | `-log(-log(rand(...)))` | ~64 |

That's ~192 registers of tensor data with `maxnreg=128`, forcing ptxas to spill the excess.

## How `tl.max` works (not the culprit)

The reduction `tl.max(tensor, axis=0, return_indices=True)` compiles to three phases:

1. **Intra-thread**: register compare/select, no extra resources
2. **Intra-warp**: `shfl.sync.bfly.b32` (50 shuffle instructions), register-to-register
3. **Cross-warp**: ~128 bytes exchanged via shared memory with `bar.sync`

The reduction itself adds negligible register pressure.

## Fix: raise `maxnreg` from 128 to 255

The kernel is persistent (grid = `NUM_SMS` = 82 blocks on 82 SMs).
Each SM gets exactly 1 block, so the occupancy argument for limiting registers does not apply.
With `maxnreg=128`, the register file can fit 4 blocks per SM (`65536 / (128 * 128) = 4`), but only 1 is ever scheduled.
Raising `maxnreg=255` uses the idle register file instead of spilling.

### NCU comparison (RTX 3090, V=151936, D=4096, H=256)

| Metric | maxnreg=128 | maxnreg=255 |
|--------|-------------|-------------|
| Duration | 20.00 ms | **11.50 ms** |
| Local memory spilling | 118,109,048 | 20,743,232 |
| Spill % of L1TEX sectors | 76.13% | 26.22% |
| Executed instructions | 693M | 578M |
| IPC active | 0.32 | 0.44 |
| Top stall | L1TEX scoreboard (40%) | Execution pipe (32%) |
| Achieved occupancy | 8.82% | 8.33% |

1.74x speedup from a one-line change.
Some spilling remains (20.7 MB) because 255 is the hardware maximum and the kernel still needs more.

## Attempted: fuse Gumbel noise into the matmul accumulator

**Idea:** initialize the accumulator with `noise * T` instead of zeros.
Then `argmax(logits + T*noise) = argmax(logits/T + noise)`, so no temperature division is needed and no separate noise tensor exists post-matmul.

**Result:** spilling dropped to 38 KB (essentially zero), but the kernel **regressed to 15.02 ms** (from 11.50 ms).

**Why:** the non-zero accumulator broke the D-loop software pipelining.
With a zero-initialized accumulator, the Triton compiler hoists the first tile prefetch before the `scf.for` loop and carries `!ttg.async.token` iter_args across iterations, enabling copy(N+1) to overlap with compute(N).
With the noise-initialized accumulator, the compiler serialized all copies: `async_copy → async_wait(num=0) → local_load → dot`, with zero overlap between memory and compute.

Verified by diffing the TTGIR: the pipelined version has 10 async operations and `!ttg.async.token` in the iter_args; the broken version has 7 async operations with immediate waits and no tokens in iter_args.

## Attempted: in-place `logits_blk += noise` (no effect)

Replacing `tl.max(logits_blk + gumbel_noise, ...)` with `logits_blk += noise; tl.max(logits_blk, ...)` produces identical TTGIR.
At the SSA level, both are `%result = arith.addf %logits, %noise`.
Virtual register count was unchanged at 8,663.

## Final state

`maxnreg=255` on RTX 3090 (the only change that helped).
The datacenter configs keep `maxnreg=128` because warp specialization adds 4 extra warps (8+4=384 threads), and `255 * 384 = 97,920` exceeds the 65,536 register file.
B200 benchmarks confirmed no regression from the change.

## Possible future work

- Investigate `maxnreg=170` on datacenter GPUs (max that fits with warp specialization: `170 * 384 = 65,280 < 65,536`).
- Chunked matmul+reduce: split `BLOCK_SIZE_V` into smaller chunks, computing noise and reducing incrementally. This would eliminate the noise tensor from registers entirely without breaking pipelining, at the cost of redundant `hidden_states` loads (likely cached in L2).
- File a Triton issue about pipelining failing with non-zero accumulator init.
