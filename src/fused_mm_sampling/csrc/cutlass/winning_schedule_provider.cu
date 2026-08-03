// Production launcher for the Gate 2c low-H B200 schedule donor.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(FMMS_ARCH_SM100)

#define FMMS_CUTLASS_LIBRARY
#define FMMS_CUTLASS_DISABLE_D
#define FMMS_FINAL_REDUCTION
#define FMMS_SM100_2SM
#define FMMS_TILE_M 128
#define FMMS_TILE_N 64
#define FMMS_TILE_K 128
#define FMMS_CLUSTER_M 2
#define fmms_evt_candidates fmms_winning_128x64x128_c2
#include "evt_candidates.cu"
#undef fmms_evt_candidates

namespace fmms_cutlass_winning {

using namespace cute;
using namespace fmms_winning_128x64x128_c2;

__global__ void initialize_candidates_kernel(
    PackedCandidate* candidates, int count, PackedCandidate identity) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < count;
       index += blockDim.x * gridDim.x) {
    candidates[index] = identity;
  }
}

__global__ void extract_indices_kernel(
    PackedCandidate const* candidates, int64_t* indices, int n) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column < n) {
    indices[column] = candidate_index(candidates[column]);
  }
}

void launch_128x64x128_c2(
    torch::Tensor weights,
    torch::Tensor padded_hidden_states,
    torch::Tensor candidates,
    torch::Tensor output,
    int gemm_n,
    int rounded_n) {
  int m = int(weights.size(0));
  int k = int(weights.size(1));
  TORCH_CHECK(gemm_n <= kTileN, "128x64x128-c2 supports H <= 64");
  TORCH_CHECK(candidates.numel() == rounded_n * int64_t(sizeof(PackedCandidate)),
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
  PackedCandidate identity = pack_candidate(-INFINITY, INT_MAX);
  typename CandidateEVT::Arguments evt_arguments{
      {}, {{}, {candidate_ptr, identity, {}}}, {}};
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
  initialize_candidates_kernel<<<(rounded_n + threads - 1) / threads,
                                   threads, 0, stream>>>(
      candidate_ptr, rounded_n, identity);
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty(
      {int64_t(workspace_size)}, weights.options().dtype(torch::kUInt8));
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "CUTLASS cannot implement 128x64x128-c2");
  TORCH_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream) ==
                  cutlass::Status::kSuccess,
              "CUTLASS 128x64x128-c2 initialization failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess,
              "CUTLASS 128x64x128-c2 launch failed");
  extract_indices_kernel<<<(gemm_n + threads - 1) / threads,
                            threads, 0, stream>>>(
      candidate_ptr, output.data_ptr<int64_t>(), int(output.size(0)));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "CUTLASS winning-schedule launch failed");
}

}  // namespace fmms_cutlass_winning

#endif
