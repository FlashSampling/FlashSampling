// Production-facing BF16 TP1 greedy provider built from the Gate 1h kernel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define FMMS_GATE_STAGE2
#define FMMS_CUTLASS_LIBRARY
#include "evt_candidates.cu"

namespace fmms_cutlass_greedy {

using namespace cute;
using namespace fmms_evt_candidates;

__global__ void stage2_indices_kernel(
    PackedCandidate const* candidates,
    int64_t* indices,
    int m_tiles,
    int candidate_stride,
    int n) {
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= n) {
    return;
  }
  PackedCandidate winner = pack_candidate(-INFINITY, INT_MAX);
  for (int tile = 0; tile < m_tiles; ++tile) {
    winner = choose_candidate(
        winner, candidates[tile * candidate_stride + column]);
  }
  indices[column] = candidate_index(winner);
}

torch::Tensor greedy(torch::Tensor weights, torch::Tensor hidden_states) {
  TORCH_CHECK(weights.is_cuda() && hidden_states.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(weights.scalar_type() == torch::kBFloat16, "weights must be bfloat16");
  TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be bfloat16");
  TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");
  TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
  TORCH_CHECK(weights.dim() == 2 && hidden_states.dim() == 2, "inputs must be matrices");
  TORCH_CHECK(weights.size(1) == hidden_states.size(1), "hidden dimensions must match");
  TORCH_CHECK(weights.size(1) % 8 == 0, "hidden dimension must be a multiple of 8");
  TORCH_CHECK(weights.size(0) > 0 && hidden_states.size(0) > 0, "dimensions must be nonzero");
  TORCH_CHECK(weights.size(0) <= INT_MAX && hidden_states.size(0) <= INT_MAX &&
              weights.size(1) <= INT_MAX, "dimensions exceed CUTLASS int32 limits");

  int m = int(weights.size(0));
  int n = int(hidden_states.size(0));
  int k = int(weights.size(1));
  int gemm_n = ((n + 3) / 4) * 4;
  int m_tiles = (m + kTileM - 1) / kTileM;
  int rounded_n = ((gemm_n + kTileN - 1) / kTileN) * kTileN;
  auto byte_options = weights.options().dtype(torch::kUInt8);
  auto padded_hidden_states =
      torch::zeros({gemm_n, k}, hidden_states.options());
  padded_hidden_states.narrow(0, 0, n).copy_(hidden_states);
  auto candidates = torch::empty(
      {m_tiles, rounded_n * int64_t(sizeof(PackedCandidate))}, byte_options);
  auto diagnostic_d = torch::empty(
      {m, rounded_n}, weights.options().dtype(torch::kFloat32));
  auto output = torch::empty({n, 1}, weights.options().dtype(torch::kInt64));

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  StrideA stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(gemm_n, k, 1));
  StrideC stride_c{int64_t(rounded_n), _1{}, int64_t(m) * rounded_n};
  StrideD stride_d{int64_t(rounded_n), _1{}, int64_t(m) * rounded_n};
  auto* candidate_ptr =
      reinterpret_cast<PackedCandidate*>(candidates.data_ptr<uint8_t>());
  PackedCandidate identity = pack_candidate(-INFINITY, INT_MAX);
  typename CandidateEVT::Arguments evt_arguments{
      {},
      {{}, {candidate_ptr, identity, {}}},
      {}};
  cutlass::KernelHardwareInfo hardware_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
          weights.get_device());
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, gemm_n, k, 1},
      {
          reinterpret_cast<ElementA*>(weights.data_ptr()),
          stride_a,
          reinterpret_cast<ElementB*>(padded_hidden_states.data_ptr()),
          stride_b,
      },
      {
          evt_arguments,
          nullptr,
          stride_c,
          diagnostic_d.data_ptr<float>(),
          stride_d,
      },
      hardware_info};

  Gemm gemm;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weights.get_device());
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::empty({int64_t(workspace_size)}, byte_options);
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "CUTLASS cannot implement this shape");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.data_ptr(), stream) ==
          cutlass::Status::kSuccess,
      "CUTLASS initialization failed");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "CUTLASS launch failed");

  constexpr int threads = 256;
  stage2_indices_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      candidate_ptr,
      output.data_ptr<int64_t>(),
      m_tiles,
      rounded_n,
      n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUTLASS Stage 2 launch failed");
  return output;
}

}  // namespace fmms_cutlass_greedy

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("greedy", &fmms_cutlass_greedy::greedy, "CUTLASS BF16 TP1 greedy FMMS");
}
