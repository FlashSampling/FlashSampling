# Inter-node scale-out options

## Status

Inter-node scaling is a potential follow-up rather than part of the NeurIPS rebuttal.
Implementing and evaluating it requires a multi-node RDMA environment and more time than is available for the rebuttal.
The current FMMS implementation should therefore continue to be described as targeting a single NVLink domain.

## Two meanings of inter-node

The tensor-parallel FMMS kernel writes each rank's tile winners directly into every peer GPU's symmetric-memory buffer.
It obtains raw peer CUDA addresses from `symm_mem_hdl.buffer_ptrs_dev` and uses regular Triton `tl.store` operations.
This requires every participating GPU to belong to one directly addressable NVLink domain.

This limitation is not identical to requiring one host.
An NVL72 rack contains multiple compute hosts but presents all 72 GPUs as one NVLink domain.
PyTorch 2.11's CUDA symmetric-memory backend detects CUDA fabric support and exchanges `CU_MEM_HANDLE_TYPE_FABRIC` handles across hosts.
It explicitly validates that all ranks have the same NVLink fabric clique ID and maps every peer allocation into each process.
The current raw-pointer FMMS path can therefore work across the hosts of an NVL72 rack without using NVSHMEM.

GPUs in separate NVLink domains, such as conventional nodes connected only by InfiniBand or Ethernet, remain different.
Their allocations cannot be mapped as ordinary peer CUDA pointers.
Cross-domain access must use NVSHMEM network operations or a hierarchical collective.

## NVL72 path

NVIDIA documents GB200 NVL72 as a default single NVLink partition containing 72 GPUs across 18 compute nodes.
With PyTorch 2.11, CUDA 13, Fabric Manager, IMEX, and an allocation group containing ranks from the same fabric clique, the current CUDA symmetric-memory allocation and barrier are designed to operate across those hosts.

The FMMS algorithm should therefore run on NVL72 with relatively little algorithmic change.
The distributed launcher must start one rank per GPU across all hosts, and the default process group used by the symmetric-memory allocation must contain the intended tensor-parallel ranks.
The system administrator must expose one NVLink partition to the job and configure CUDA fabric-handle import through IMEX.

This is a compatibility expectation rather than a performance claim.
The FMMS kernel statically fans each candidate out to every tensor-parallel rank, so its communication and generated store sequence grow linearly with tensor-parallel size.
Every rank sends to 71 peers at TP72.
Compilation cost, register pressure, mapped-memory capacity, barrier latency, and fan-out time therefore require measurement.
The existing TP8 results already show that fan-out can dominate after tensor parallelism has made the local matrix multiplication sufficiently small.

An NVL72 experiment would demonstrate cross-host scaling within a rack-scale NVLink domain.
It would not answer performance over InfiniBand because the traffic would use multi-node NVLink rather than the scale-out network.

## Option 1: global NVSHMEM fan-out

PyTorch exposes NVSHMEM device operations to Triton through `@requires_nvshmem` and functions such as `nvshmem.put`.
This makes it possible to preserve the current global fan-out algorithm across separate NVLink domains without implementing an InfiniBand transport directly.
See the [PyTorch scale-out documentation](https://docs.pytorch.org/docs/main/symmetric_memory.html#scale-out).

The approximate flow would be:

1. Select the `NVSHMEM` symmetric-memory backend before the first allocation.
2. Allocate and rendezvous the FMMS output buffers across the global tensor-parallel group.
3. Write each tile's winning values and indices to the local symmetric allocation.
4. Issue NVSHMEM puts from that allocation to every peer.
5. Explicitly complete the remote operations and synchronize the ranks.
6. Run the existing local reduction over all received candidates.

This is not a direct replacement of `tl.store` with `nvshmem.put`.
The current winner values are held in registers, whereas `nvshmem.put` copies from a memory-backed source.
The kernel must first store them locally and then transfer them.
Values and indices are separate buffers, so the simplest implementation needs two puts per peer.

`nvshmem.put` guarantees that the local source can be reused when it returns, but it does not guarantee delivery to the remote destination.
Correctness therefore requires an explicit completion protocol.
A simple initial design is `put`, `nvshmem.quiet()`, and a process-group barrier before the reduction.
A lower-latency design could use put-with-signal operations and receiver-side signal waits.

Global NVSHMEM fan-out is the closest scale-out version of the current algorithm and would answer whether the existing P2P scheme remains faster than all-gather at TP16.
It may nevertheless perform poorly because each rank sends small transfers to every remote rank.
For two nodes with eight GPUs each, the two directions contain 128 cross-node sender-to-receiver relationships.
NVSHMEM provides the transport, but it does not remove message latency or NIC contention from this communication pattern.

## Option 2: hierarchical FMMS

A hierarchical implementation would use the existing direct P2P path only within each NVLink domain.
Each node would then communicate its reduced winner to the other nodes through NCCL or NVSHMEM.

For TP16 across two eight-GPU nodes, the flow would be:

1. Create one node-local process group per node.
2. Partition the vocabulary across all 16 global ranks as usual.
3. Run the current fused computation and P2P fan-out across the eight local ranks.
4. Reduce the local ranks to one `(maximum value, global token index)` candidate per batch row and sample.
5. Exchange the node-level candidates over InfiniBand.
6. Select the global winner locally.

This remains exact because taking the maximum within each node and then across nodes is equivalent to taking the maximum across all vocabulary shards.
The global token index must retain the offset of the original global tensor-parallel rank.

The inter-node traffic is proportional to the batch size, number of samples, and number of nodes.
It is independent of vocabulary size.
It also avoids sending every rank's candidate separately to every GPU on the remote node.

This is likely the better production design.
It requires more restructuring because FMMS currently reduces directly to token indices and does not expose the winning values for a second reduction stage.
It also overlaps only the intra-node communication with the matrix multiplication.
The small inter-node exchange happens after the node-local reduction.

## Option 3: hybrid direct and network fan-out

A hybrid kernel could retain raw `tl.store` operations for directly accessible peers and use NVSHMEM puts only for network peers.
This avoids routing fast NVLink traffic through the more general network operation.
It preserves the global fan-out structure and may reduce its intra-node overhead.

This option is more complex than a uniform NVSHMEM path.
The kernel needs topology information, separate peer lists, correct PE mappings, and a completion protocol covering both direct stores and NVSHMEM operations.
It still performs redundant cross-node fan-out and is therefore less attractive than the hierarchical design.

## Recommended follow-up sequence

The first follow-up should implement global NVSHMEM fan-out as a correctness and feasibility prototype.
It most directly tests the reviewer's question because it preserves the existing algorithm while changing the transport.

The prototype should be tested in this order:

1. Verify that the PyTorch image reports `symm_mem.is_nvshmem_available()`.
2. Validate two-rank intra-node operation with the NVSHMEM backend.
3. Validate TP16 sampling correctness across two RDMA-connected nodes.
4. Compare FMMS against the existing compiled, FlashInfer one-stage, and FlashInfer two-stage baselines.
5. Measure batch sizes 16, 64, and 256 with repeated independent allocations.
6. Record latency, inter-node bytes, selected transport, and NIC topology.

If the direct scale-out path is limited by small-message fan-out, implement the hierarchical version and repeat the comparison.
The final evaluation should distinguish algorithmic scaling from differences in pod topology, NIC placement, and RDMA configuration.

## Rebuttal position

The rebuttal can distinguish cross-host NVLink from cross-domain scale-out.
The current CUDA symmetric-memory design is compatible in principle with an NVL72 fabric, but it has not been tested at TP72 and no performance advantage should be claimed there.
PyTorch's NVSHMEM Triton operations provide a practical route across separate NVLink domains, and a hierarchical reduction can reduce inter-node traffic further.
NVL72 validation and cross-domain designs remain future work until they have been measured.
