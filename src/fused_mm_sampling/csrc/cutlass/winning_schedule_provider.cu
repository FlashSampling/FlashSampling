// Production launchers for the Gate 2c B200 schedule donors.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(FMMS_ARCH_SM100)

#define FMMS_CUTLASS_LIBRARY
#define FMMS_CUTLASS_DISABLE_D
#define FMMS_PER_CTA_CANDIDATES
#define FMMS_SM100_2SM
#define FMMS_TILE_M 128
#define FMMS_TILE_N 64
#define FMMS_TILE_K 128
#define FMMS_CLUSTER_M 2
#define fmms_evt_candidates fmms_winning_128x64x128_c2
#include "evt_candidates.cu"
#undef fmms_evt_candidates
#undef FMMS_TILE_M
#undef FMMS_TILE_N
#undef FMMS_TILE_K
#undef FMMS_CLUSTER_M

#if defined(FMMS_WARPGROUP_REDUCTION)
#define FMMS_USE_WARPGROUP_REDUCTION
#endif

#define FMMS_GUMBEL_PARTIAL_UNROLL 2
#define FMMS_TILE_M 256
#define FMMS_TILE_N 128
#define FMMS_TILE_K 64
#define FMMS_CLUSTER_M 2
#define fmms_evt_candidates fmms_winning_256x128x64_c2
#include "evt_candidates.cu"
#undef fmms_evt_candidates
#undef FMMS_TILE_M
#undef FMMS_TILE_N
#undef FMMS_TILE_K
#undef FMMS_CLUSTER_M
#undef FMMS_GUMBEL_PARTIAL_UNROLL

#define FMMS_GUMBEL_PARTIAL_UNROLL 2
#define FMMS_TILE_M 256
#define FMMS_TILE_N 128
#define FMMS_TILE_K 128
#define FMMS_CLUSTER_M 2
#define fmms_evt_candidates fmms_winning_256x128x128_c2
#include "evt_candidates.cu"
#undef fmms_evt_candidates
#undef FMMS_TILE_M
#undef FMMS_TILE_N
#undef FMMS_TILE_K
#undef FMMS_CLUSTER_M
#undef FMMS_GUMBEL_PARTIAL_UNROLL

#define FMMS_GUMBEL_PARTIAL_UNROLL 4
#define FMMS_TILE_M 256
#define FMMS_TILE_N 128
#define FMMS_TILE_K 64
#define FMMS_CLUSTER_M 4
#define fmms_evt_candidates fmms_winning_256x128x64_c4
#include "evt_candidates.cu"
#undef fmms_evt_candidates
#undef FMMS_TILE_M
#undef FMMS_TILE_N
#undef FMMS_TILE_K
#undef FMMS_CLUSTER_M
#undef FMMS_GUMBEL_PARTIAL_UNROLL

#if defined(FMMS_WARPGROUP_REDUCTION)
#undef FMMS_USE_WARPGROUP_REDUCTION
#endif

namespace fmms_cutlass_winning {

using namespace cute;
using PackedCandidate = uint64_t;

__global__ void initialize_candidates_kernel(
    PackedCandidate* candidates, int count, PackedCandidate identity) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    candidates[index] = identity;
  }
}

__global__ void merge_candidate_indices_kernel(
    PackedCandidate const* candidates,
    int64_t* indices,
    int m_tiles,
    int candidate_stride,
    int n) {
  int column = blockIdx.x;
  if (column >= n) {
    return;
  }
  __shared__ PackedCandidate partials[256];
  PackedCandidate winner =
      fmms_winning_128x64x128_c2::pack_candidate(-INFINITY, INT_MAX);
  for (int tile = threadIdx.x; tile < m_tiles; tile += blockDim.x) {
    winner = fmms_winning_128x64x128_c2::choose_candidate(
        winner, candidates[tile * candidate_stride + column]);
  }
  partials[threadIdx.x] = winner;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partials[threadIdx.x] =
          fmms_winning_128x64x128_c2::choose_candidate(
              partials[threadIdx.x], partials[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    indices[column] = fmms_winning_128x64x128_c2::candidate_index(partials[0]);
  }
}

template <
    class Gemm,
    class GemmKernel,
    class CandidateEVT,
    class ElementA,
    class ElementB>
void launch_winning_schedule(
    torch::Tensor weights,
    torch::Tensor padded_hidden_states,
    torch::Tensor candidates,
    torch::Tensor output,
    int gemm_n,
    int rounded_n,
    int tile_m,
    int tile_n,
#if defined(FMMS_GUMBEL)
    torch::Tensor temperature,
    uint64_t seed,
    int original_n,
    int sample_base,
#endif
    char const* label) {
  int m = int(weights.size(0));
  int k = int(weights.size(1));
  int m_tiles = (m + tile_m / 2 - 1) / (tile_m / 2);
  TORCH_CHECK(rounded_n >= gemm_n && rounded_n % tile_n == 0,
              label, " received invalid hidden-state padding");
  TORCH_CHECK(candidates.numel() ==
                  m_tiles * int64_t(rounded_n) * sizeof(PackedCandidate),
              "winning-schedule candidate storage has the wrong size");

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  StrideA stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, make_shape(gemm_n, k, 1));
  StrideC stride_c{int64_t(rounded_n), _1{}, int64_t(m) * rounded_n};
  StrideD stride_d{int64_t(rounded_n), _1{}, int64_t(m) * rounded_n};
  auto* candidate_ptr =
      reinterpret_cast<PackedCandidate*>(candidates.data_ptr<uint8_t>());
  PackedCandidate identity =
      fmms_winning_128x64x128_c2::pack_candidate(-INFINITY, INT_MAX);
  typename CandidateEVT::Arguments evt_arguments{
#if defined(FMMS_GUMBEL)
      {seed, temperature.data_ptr<float>(), original_n, sample_base},
#else
      {},
#endif
      {{}, {candidate_ptr, identity, {}}}, {}};
  cutlass::KernelHardwareInfo hardware_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
          weights.get_device());
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, gemm_n, k, 1},
      {reinterpret_cast<ElementA*>(weights.data_ptr()), stride_a,
       reinterpret_cast<ElementB*>(padded_hidden_states.data_ptr()), stride_b},
      {evt_arguments, nullptr, stride_c, nullptr, stride_d},
      hardware_info};

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weights.get_device());
  constexpr int threads = 256;
  initialize_candidates_kernel<<<(m_tiles * rounded_n + threads - 1) / threads,
                                   threads, 0, stream>>>(
      candidate_ptr, m_tiles * rounded_n, identity);
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      {int64_t(workspace_size)}, weights.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "CUTLASS cannot implement ", label);
  TORCH_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream) ==
                  cutlass::Status::kSuccess,
              "CUTLASS ", label, " initialization failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
              "CUTLASS ", label, " launch failed");
  merge_candidate_indices_kernel<<<gemm_n, threads, 0, stream>>>(
      candidate_ptr, output.data_ptr<int64_t>(), m_tiles, rounded_n,
      int(output.size(0)));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "CUTLASS winning-schedule launch failed");
}

#if defined(FMMS_GUMBEL)
#define FMMS_GUMBEL_FUNCTION_PARAMETERS                                      \
      , torch::Tensor temperature, uint64_t seed, int original_n, int sample_base
#define FMMS_GUMBEL_CALL_ARGUMENTS                                           \
      , temperature, seed, original_n, sample_base
#else
#define FMMS_GUMBEL_FUNCTION_PARAMETERS
#define FMMS_GUMBEL_CALL_ARGUMENTS
#endif

#define FMMS_DEFINE_WINNING_LAUNCH(FUNCTION_NAME, KERNEL_NAMESPACE, LABEL)     \
  void FUNCTION_NAME(                                                         \
      torch::Tensor weights, torch::Tensor padded_hidden_states,              \
      torch::Tensor candidates, torch::Tensor output, int gemm_n,             \
      int rounded_n FMMS_GUMBEL_FUNCTION_PARAMETERS) {                        \
    launch_winning_schedule<                                                  \
        KERNEL_NAMESPACE::Gemm, KERNEL_NAMESPACE::GemmKernel,                 \
        KERNEL_NAMESPACE::CandidateEVT, KERNEL_NAMESPACE::ElementA,           \
        KERNEL_NAMESPACE::ElementB>(                                          \
        weights, padded_hidden_states, candidates, output, gemm_n, rounded_n, \
        KERNEL_NAMESPACE::kTileM, KERNEL_NAMESPACE::kTileN                    \
        FMMS_GUMBEL_CALL_ARGUMENTS, LABEL);                                   \
  }

FMMS_DEFINE_WINNING_LAUNCH(launch_128x64x128_c2,
                           fmms_winning_128x64x128_c2, "128x64x128-c2")
FMMS_DEFINE_WINNING_LAUNCH(launch_256x128x128_c2,
                           fmms_winning_256x128x128_c2, "256x128x128-c2")
FMMS_DEFINE_WINNING_LAUNCH(launch_256x128x64_c2,
                           fmms_winning_256x128x64_c2, "256x128x64-c2")
FMMS_DEFINE_WINNING_LAUNCH(launch_256x128x64_c4,
                           fmms_winning_256x128x64_c4, "256x128x64-c4")

#undef FMMS_GUMBEL_FUNCTION_PARAMETERS
#undef FMMS_GUMBEL_CALL_ARGUMENTS

}  // namespace fmms_cutlass_winning

#endif
