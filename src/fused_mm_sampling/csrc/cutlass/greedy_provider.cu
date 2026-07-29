// Production-facing BF16 TP1 greedy provider built from the Gate 1h kernel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define FMMS_GATE_STAGE2
#define FMMS_CUTLASS_LIBRARY
#define FMMS_CUTLASS_DISABLE_D
#include "evt_candidates.cu"

namespace fmms_cutlass_greedy {

using namespace cute;
using namespace fmms_evt_candidates;

using PlainElementC = void;
using PlainElementD = float;
using PlainCollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        EpilogueTile,
        ElementAccumulator,
        ElementAccumulator,
        PlainElementC,
        LayoutC,
        kAlignmentC,
        PlainElementD,
        LayoutD,
        kAlignmentD,
        EpilogueSchedule>::CollectiveOp;
using PlainCollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        ElementA,
        LayoutA,
        kAlignmentA,
        ElementB,
        LayoutB,
        kAlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename PlainCollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;
using PlainGemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    PlainCollectiveMainloop,
    PlainCollectiveEpilogue>;
using PlainGemm =
    cutlass::gemm::device::GemmUniversalAdapter<PlainGemmKernel>;

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
          nullptr,
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

torch::Tensor plain_gemm(torch::Tensor weights, torch::Tensor hidden_states) {
  TORCH_CHECK(weights.is_cuda() && hidden_states.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(weights.scalar_type() == torch::kBFloat16, "weights must be bfloat16");
  TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be bfloat16");
  TORCH_CHECK(weights.is_contiguous() && hidden_states.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(weights.dim() == 2 && hidden_states.dim() == 2, "inputs must be matrices");
  TORCH_CHECK(weights.size(1) == hidden_states.size(1), "hidden dimensions must match");

  int m = int(weights.size(0));
  int n = int(hidden_states.size(0));
  int k = int(weights.size(1));
  int gemm_n = ((n + 3) / 4) * 4;
  int rounded_n = ((gemm_n + kTileN - 1) / kTileN) * kTileN;
  auto padded_hidden_states =
      torch::zeros({gemm_n, k}, hidden_states.options());
  padded_hidden_states.narrow(0, 0, n).copy_(hidden_states);
  auto output = torch::empty(
      {m, rounded_n}, weights.options().dtype(torch::kFloat32));
  auto byte_options = weights.options().dtype(torch::kUInt8);

  using StrideA = typename PlainGemmKernel::StrideA;
  using StrideB = typename PlainGemmKernel::StrideB;
  using StrideC = typename PlainGemmKernel::StrideC;
  using StrideD = typename PlainGemmKernel::StrideD;
  StrideA stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(gemm_n, k, 1));
  StrideC stride_c{};
  StrideD stride_d{int64_t(rounded_n), _1{}, int64_t(m) * rounded_n};
  cutlass::KernelHardwareInfo hardware_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<PlainGemmKernel>(
          weights.get_device());
  typename PlainGemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, gemm_n, k, 1},
      {
          reinterpret_cast<ElementA*>(weights.data_ptr()),
          stride_a,
          reinterpret_cast<ElementB*>(padded_hidden_states.data_ptr()),
          stride_b,
      },
      {
          {},
          nullptr,
          stride_c,
          output.data_ptr<float>(),
          stride_d,
      },
      hardware_info};

  PlainGemm gemm;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weights.get_device());
  size_t workspace_size = PlainGemm::get_workspace_size(arguments);
  auto workspace = torch::empty({int64_t(workspace_size)}, byte_options);
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "CUTLASS cannot implement the plain GEMM shape");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.data_ptr(), stream) ==
          cutlass::Status::kSuccess,
      "CUTLASS plain GEMM initialization failed");
  TORCH_CHECK(
      gemm.run(stream) == cutlass::Status::kSuccess,
      "CUTLASS plain GEMM launch failed");
  return output.narrow(1, 0, n);
}

pybind11::dict kernel_attributes() {
  cudaFuncAttributes gemm_attributes;
  cudaFuncAttributes stage2_attributes;
  TORCH_CHECK(
      cudaFuncGetAttributes(
          &gemm_attributes,
          reinterpret_cast<void const*>(cutlass::device_kernel<GemmKernel>)) ==
          cudaSuccess,
      "Could not query CUTLASS FMMS kernel attributes");
  TORCH_CHECK(
      cudaFuncGetAttributes(
          &stage2_attributes,
          reinterpret_cast<void const*>(stage2_indices_kernel)) == cudaSuccess,
      "Could not query CUTLASS Stage 2 kernel attributes");
  int gemm_active_blocks = 0;
  int stage2_active_blocks = 0;
  TORCH_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &gemm_active_blocks,
          cutlass::device_kernel<GemmKernel>,
          gemm_attributes.maxThreadsPerBlock,
          gemm_attributes.sharedSizeBytes) == cudaSuccess,
      "Could not query CUTLASS FMMS occupancy");
  TORCH_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &stage2_active_blocks,
          stage2_indices_kernel,
          256,
          stage2_attributes.sharedSizeBytes) == cudaSuccess,
      "Could not query CUTLASS Stage 2 occupancy");

  pybind11::dict output;
  output["gemm_num_regs"] = gemm_attributes.numRegs;
  output["gemm_local_size_bytes"] = gemm_attributes.localSizeBytes;
  output["gemm_shared_size_bytes"] = gemm_attributes.sharedSizeBytes;
  output["gemm_max_threads_per_block"] = gemm_attributes.maxThreadsPerBlock;
  output["gemm_active_blocks_per_sm"] = gemm_active_blocks;
  output["stage2_num_regs"] = stage2_attributes.numRegs;
  output["stage2_local_size_bytes"] = stage2_attributes.localSizeBytes;
  output["stage2_shared_size_bytes"] = stage2_attributes.sharedSizeBytes;
  output["stage2_active_blocks_per_sm"] = stage2_active_blocks;
  return output;
}

}  // namespace fmms_cutlass_greedy

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("greedy", &fmms_cutlass_greedy::greedy, "CUTLASS BF16 TP1 greedy FMMS");
  module.def(
      "plain_gemm",
      &fmms_cutlass_greedy::plain_gemm,
      "Plain CUTLASS BF16 GEMM with FP32 output");
  module.def(
      "kernel_attributes",
      &fmms_cutlass_greedy::kernel_attributes,
      "Static kernel resources and theoretical active blocks per SM");
}
