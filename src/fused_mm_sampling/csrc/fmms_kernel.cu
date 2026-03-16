// Fused Matrix-Multiply & Sampling (FMMS) — CUDA C++ kernel
//
// Work decomposition
// ==================
// Thread block: 256 threads (8 warps), TILE_V=128 vocab rows.
// Each warp handles ROWS_PER_WARP=16 rows.  Within a warp, 32 lanes
// cooperate on the D-dimension dot product via coalesced scalar loads.
//
// For each hidden-state column h ∈ [0, H_batch):
//   Stage 1 – per-block: fmms_stage1_kernel
//     1. Tiled matmul: logit[v] = dot(W[v,:], hs[h,:])
//     2. Scale by 1/temperature
//     3. Add Gumbel noise  -log(-log(u)),  u ~ Uniform(0,1)
//     4. Warp-local argmax across ROWS_PER_WARP rows (butterfly shuffle)
//     5. Block-level argmax across 8 warps (shared memory)
//     6. Write (max_val, max_idx) to global output buffers
//   Stage 2 – Python-side:
//     idxs = maxs.max(dim=0).indices
//     samples = maxs_idx.gather(0, idxs.unsqueeze(0)).squeeze(0)
//
// Memory access
// =============
// 32 lanes load 32 consecutive bf16 values = 64 bytes = one L2 cache line.
// Hidden states (small for decode: 16 KB at H=1, D=8192) stay in L2 cache.

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>

// ──────────────────────── constants ────────────────────────

static constexpr int TILE_V = 128;
static constexpr int ROWS_PER_WARP = 16;  // 128 / 8 warps
static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_THREADS = 256;  // 8 warps
static constexpr int N_WARPS = BLOCK_THREADS / WARP_SIZE;

// ──────────────────────── helpers ────────────────────────

// Warp-level sum reduction (all lanes get the result)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp-level max reduction with index tracking across `width` lanes.
__device__ __forceinline__ void warp_reduce_max_idx(
    float& val, int& idx, int width
) {
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        int other_idx = __shfl_xor_sync(0xFFFFFFFF, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

// ──────────────────────── main kernel ────────────────────────

__global__ void fmms_stage1_kernel(
    const __nv_bfloat16* __restrict__ weights,       // [V, D] row-major
    const __nv_bfloat16* __restrict__ hidden_states,  // [H, D] row-major
    float*               __restrict__ max_out,        // [n_tiles_v, H, num_samples]
    int64_t*             __restrict__ max_out_idx,    // [n_tiles_v, H, num_samples]
    const float*         __restrict__ temperature_ptr,// 0-d tensor
    int vocab_size,
    int hidden_size,
    int n_hidden_states,
    int num_samples,
    unsigned long long base_seed
) {
    // Grid mapping with swizzle for L2 reuse (mirrors Triton's swizzle2d)
    const int num_tiles_v = (vocab_size + TILE_V - 1) / TILE_V;
    constexpr int GROUP_SIZE = 4;
    int pid_v, pid_h;
    {
        int num_pid_h = gridDim.y;
        int ij = blockIdx.x * num_pid_h + blockIdx.y;
        int group_id = ij / (GROUP_SIZE * num_pid_h);
        int first_pid_v = group_id * GROUP_SIZE;
        int group_size_v = min(num_tiles_v - first_pid_v, GROUP_SIZE);
        int ij_in_group = ij % (group_size_v * num_pid_h);
        pid_v = first_pid_v + ij_in_group % group_size_v;
        pid_h = ij_in_group / group_size_v;
    }

    const int h_idx = pid_h;
    if (h_idx >= n_hidden_states) return;

    const int v_start = pid_v * TILE_V;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    // Temperature from 0-d GPU tensor (no CPU sync)
    const float inv_temperature = 1.0f / __ldg(temperature_ptr);

    // ── Matmul: each warp computes dot products for ROWS_PER_WARP rows ──
    // Accumulators: each lane holds partial sums for all of its warp's rows
    float acc[ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        acc[r] = 0.0f;
    }

    // Pointer to the hidden state row for this h_idx
    const __nv_bfloat16* hs_row = hidden_states + (int64_t)h_idx * hidden_size;

    // D-loop: 32 lanes load 32 consecutive bf16 values per step (coalesced)
    for (int d = lane_id; d < hidden_size; d += WARP_SIZE) {
        float hs_val = __bfloat162float(hs_row[d]);

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int v_idx = v_start + warp_id * ROWS_PER_WARP + r;
            if (v_idx < vocab_size) {
                float w_val = __bfloat162float(
                    weights[(int64_t)v_idx * hidden_size + d]
                );
                acc[r] += w_val * hs_val;
            }
        }
    }

    // Warp-level reduction across 32 lanes to get complete logits
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        acc[r] = warp_reduce_sum(acc[r]);
    }

    // After xor-reduce, all lanes have the same value for each row.
    // Assign one logit per lane (lanes 0..15) for the argmax reduction.
    float logit_val = -INFINITY;
    int global_v_idx = -1;
    if (lane_id < ROWS_PER_WARP) {
        int v_idx = v_start + warp_id * ROWS_PER_WARP + lane_id;
        if (v_idx < vocab_size) {
            logit_val = acc[lane_id] * inv_temperature;
            global_v_idx = v_idx;
        }
    }

    // Shared memory for block-level reduction across warps
    __shared__ float smem_max_val[N_WARPS];
    __shared__ int smem_max_idx[N_WARPS];

    // ── Gumbel noise + argmax per sample ──
    for (int s = 0; s < num_samples; s++) {
        float noisy_logit = -INFINITY;
        int noisy_idx = -1;

        if (lane_id < ROWS_PER_WARP && global_v_idx >= 0) {
            // Gumbel noise: -log(-log(u)), u ~ Uniform(0,1)
            // Unique sequence per (block, hidden_state, sample, vocab_row)
            unsigned long long seq = (unsigned long long)pid_v * 100000ULL
                + (unsigned long long)h_idx * 1000ULL
                + (unsigned long long)s * 10ULL
                + (unsigned long long)(warp_id * ROWS_PER_WARP + lane_id);
            curandStatePhilox4_32_10_t state;
            curand_init(base_seed, seq, 0, &state);
            float u = curand_uniform(&state);
            u = fmaxf(u, 1e-10f);
            float gumbel = -logf(-logf(u));
            noisy_logit = logit_val + gumbel;
            noisy_idx = global_v_idx;
        }

        // Warp-level argmax across ROWS_PER_WARP=16 lanes
        warp_reduce_max_idx(noisy_logit, noisy_idx, ROWS_PER_WARP);

        // Lane 0 of each warp writes to shared memory
        if (lane_id == 0) {
            smem_max_val[warp_id] = noisy_logit;
            smem_max_idx[warp_id] = noisy_idx;
        }
        __syncthreads();

        // Thread 0 reduces across all warps
        if (threadIdx.x == 0) {
            float best_val = smem_max_val[0];
            int best_idx = smem_max_idx[0];
            #pragma unroll
            for (int w = 1; w < N_WARPS; w++) {
                if (smem_max_val[w] > best_val) {
                    best_val = smem_max_val[w];
                    best_idx = smem_max_idx[w];
                }
            }

            // Write to output: max_out[pid_v, h_idx, s]
            int64_t out_offset = (int64_t)pid_v * n_hidden_states * num_samples
                + (int64_t)h_idx * num_samples
                + s;
            max_out[out_offset] = best_val;
            max_out_idx[out_offset] = best_idx;
        }
        __syncthreads();
    }
}

// ──────────────────────── host wrapper ────────────────────────

void fmms_stage1(
    torch::Tensor weights,        // [V, D] bfloat16
    torch::Tensor hidden_states,  // [H, D] bfloat16
    torch::Tensor max_out,        // [n_tiles_v, H, num_samples] float32
    torch::Tensor max_out_idx,    // [n_tiles_v, H, num_samples] int64
    torch::Tensor temperature,    // 0-d float32
    int64_t seed
) {
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
    TORCH_CHECK(weights.dtype() == torch::kBFloat16, "weights must be bfloat16");
    TORCH_CHECK(hidden_states.dtype() == torch::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(temperature.is_cuda(), "temperature must be a CUDA tensor");

    int V = weights.size(0);
    int D = weights.size(1);
    int H = hidden_states.size(0);
    TORCH_CHECK(hidden_states.size(1) == D,
        "hidden_states dim 1 (", hidden_states.size(1), ") != weights dim 1 (", D, ")");

    int num_samples = max_out.size(2);
    int n_tiles_v = (V + TILE_V - 1) / TILE_V;

    // Grid: (n_tiles_v, H) — one block per (V-tile, hidden-state)
    dim3 grid(n_tiles_v, H);
    dim3 block(BLOCK_THREADS);

    fmms_stage1_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>()),
        max_out.data_ptr<float>(),
        max_out_idx.data_ptr<int64_t>(),
        temperature.data_ptr<float>(),
        V, D, H, num_samples,
        static_cast<unsigned long long>(seed)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fmms_stage1", &fmms_stage1, "FMMS Stage 1 kernel (CUDA)");
}
