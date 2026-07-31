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
using PlainElementD = cutlass::bfloat16_t;
constexpr int PlainAlignmentD = 8;
template <
    class TileShape_,
    class MainloopSchedule_ = cutlass::gemm::collective::KernelScheduleAuto,
    class EpilogueSchedule_ =
        cutlass::epilogue::collective::EpilogueScheduleAuto>
struct PlainGemmVariant {
  using MainloopSchedule = MainloopSchedule_;
  using EpilogueSchedule = EpilogueSchedule_;
  using EpilogueTile =
      cutlass::epilogue::collective::EpilogueTileAuto;
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          cutlass::arch::OpClassTensorOp,
          TileShape_,
          ClusterShape,
          EpilogueTile,
          ElementAccumulator,
          ElementAccumulator,
          PlainElementC,
          LayoutC,
          kAlignmentC,
          PlainElementD,
          LayoutD,
          PlainAlignmentD,
          EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop =
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
          TileShape_,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<
              static_cast<int>(
                  sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

using PlainVariant64x128 =
    PlainGemmVariant<Shape<_64, _128, _64>>;
using PlainVariant128x64 =
    PlainGemmVariant<Shape<_128, _64, _64>>;
using PlainVariant128x128 =
    PlainGemmVariant<Shape<_128, _128, _64>>;
using PlainVariant64x128Native =
    PlainGemmVariant<Shape<_64, _128, _64>, MainloopSchedule, EpilogueSchedule>;
using PlainVariant128x64Native =
    PlainGemmVariant<Shape<_128, _64, _64>, MainloopSchedule, EpilogueSchedule>;
using PlainVariant128x128Native =
    PlainGemmVariant<Shape<_128, _128, _64>, MainloopSchedule, EpilogueSchedule>;
using PlainGemmKernel = typename PlainVariant128x128::Kernel;
using PlainGemm = typename PlainVariant128x128::Gemm;

template <int N>
__global__ void small_n_gemv_kernel(
    __nv_bfloat16 const* weights,
    __nv_bfloat16 const* hidden_states,
    __nv_bfloat16* output,
    int m,
    int k) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (row >= m) {
    return;
  }
  float sums[N] = {};
  for (int column = lane * 2; column < k; column += 64) {
    auto weight_pair = *reinterpret_cast<__nv_bfloat162 const*>(
        weights + int64_t(row) * k + column);
    float2 weight = __bfloat1622float2(weight_pair);
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < N; ++n) {
      auto hidden_pair = *reinterpret_cast<__nv_bfloat162 const*>(
          hidden_states + int64_t(n) * k + column);
      float2 hidden = __bfloat1622float2(hidden_pair);
      sums[n] += weight.x * hidden.x + weight.y * hidden.y;
    }
  }
  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < N; ++n) {
    for (int offset = 16; offset > 0; offset /= 2) {
      sums[n] += __shfl_down_sync(0xffffffff, sums[n], offset);
    }
    if (lane == 0) {
      output[int64_t(row) * N + n] = __float2bfloat16_rn(sums[n]);
    }
  }
}

void launch_small_n_gemv(
    torch::Tensor weights,
    torch::Tensor hidden_states,
    torch::Tensor output) {
  TORCH_CHECK(weights.is_cuda() && hidden_states.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(weights.scalar_type() == torch::kBFloat16 &&
              hidden_states.scalar_type() == torch::kBFloat16,
              "inputs must be bfloat16");
  TORCH_CHECK(weights.is_contiguous() && hidden_states.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(weights.dim() == 2 && hidden_states.dim() == 2,
              "inputs must be matrices");
  TORCH_CHECK(weights.size(1) == hidden_states.size(1),
              "hidden dimensions must match");
  int m = int(weights.size(0));
  int n = int(hidden_states.size(0));
  int k = int(weights.size(1));
  TORCH_CHECK(n >= 1 && n <= 8, "small-N GEMV supports H=1 through H=8");
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
              output.is_contiguous() && output.size(0) == m &&
              output.size(1) == n, "output must be contiguous BF16 [V, H]");
  constexpr int threads = 256;
  int blocks = (m + threads / 32 - 1) / (threads / 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weights.get_device());
  auto* weights_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(weights.data_ptr());
  auto* hidden_ptr =
      reinterpret_cast<__nv_bfloat16 const*>(hidden_states.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
#define FMMS_LAUNCH_SMALL_N_GEMV(N)                                           \
  small_n_gemv_kernel<N><<<blocks, threads, 0, stream>>>(                     \
      weights_ptr, hidden_ptr, output_ptr, m, k)
  switch (n) {
    case 1: FMMS_LAUNCH_SMALL_N_GEMV(1); break;
    case 2: FMMS_LAUNCH_SMALL_N_GEMV(2); break;
    case 3: FMMS_LAUNCH_SMALL_N_GEMV(3); break;
    case 4: FMMS_LAUNCH_SMALL_N_GEMV(4); break;
    case 5: FMMS_LAUNCH_SMALL_N_GEMV(5); break;
    case 6: FMMS_LAUNCH_SMALL_N_GEMV(6); break;
    case 7: FMMS_LAUNCH_SMALL_N_GEMV(7); break;
    case 8: FMMS_LAUNCH_SMALL_N_GEMV(8); break;
  }
#undef FMMS_LAUNCH_SMALL_N_GEMV
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "small-N BF16 GEMV launch failed");
}

torch::Tensor small_n_gemv(
    torch::Tensor weights, torch::Tensor hidden_states) {
  auto output = torch::empty(
      {weights.size(0), hidden_states.size(0)}, weights.options());
  launch_small_n_gemv(weights, hidden_states, output);
  return output;
}

__global__ void stage2_indices_kernel(
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
  PackedCandidate winner = pack_candidate(-INFINITY, INT_MAX);
  for (int tile = threadIdx.x; tile < m_tiles; tile += blockDim.x) {
    winner = choose_candidate(
        winner, candidates[tile * candidate_stride + column]);
  }
  partials[threadIdx.x] = winner;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      partials[threadIdx.x] = choose_candidate(
          partials[threadIdx.x], partials[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    indices[column] = candidate_index(partials[0]);
  }
}

void launch_greedy_gemm(
    torch::Tensor weights,
    torch::Tensor padded_hidden_states,
    torch::Tensor candidates,
    int gemm_n,
    int rounded_n) {
  int m = int(weights.size(0));
  int k = int(weights.size(1));
  int m_tiles = (m + kTileM - 1) / kTileM;
  TORCH_CHECK(
      padded_hidden_states.size(0) == gemm_n &&
          padded_hidden_states.size(1) == k,
      "padded hidden-state shape does not match the GEMM problem");
  TORCH_CHECK(
      candidates.numel() ==
          m_tiles * int64_t(rounded_n) * int64_t(sizeof(PackedCandidate)),
      "candidate storage has the wrong size");

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
  auto workspace = torch::empty(
      {int64_t(workspace_size)}, weights.options().dtype(torch::kUInt8));
  TORCH_CHECK(
      gemm.can_implement(arguments) == cutlass::Status::kSuccess,
      "CUTLASS cannot implement this shape");
  TORCH_CHECK(
      gemm.initialize(arguments, workspace.data_ptr(), stream) ==
          cutlass::Status::kSuccess,
      "CUTLASS initialization failed");
  TORCH_CHECK(
      gemm.run(stream) == cutlass::Status::kSuccess,
      "CUTLASS launch failed");
}

void launch_greedy_stage2(
    torch::Tensor candidates,
    torch::Tensor output,
    int m_tiles,
    int rounded_n,
    int n) {
  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(candidates.get_device());
  auto* candidate_ptr =
      reinterpret_cast<PackedCandidate*>(candidates.data_ptr<uint8_t>());
  stage2_indices_kernel<<<n, threads, 0, stream>>>(
      candidate_ptr,
      output.data_ptr<int64_t>(),
      m_tiles,
      rounded_n,
      n);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUTLASS Stage 2 launch failed");
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

  launch_greedy_gemm(
      weights, padded_hidden_states, candidates, gemm_n, rounded_n);
  launch_greedy_stage2(candidates, output, m_tiles, rounded_n, n);
  return output;
}

pybind11::tuple make_greedy_buffers(
    torch::Tensor weights, torch::Tensor hidden_states) {
  int m = int(weights.size(0));
  int n = int(hidden_states.size(0));
  int k = int(weights.size(1));
  int gemm_n = ((n + 3) / 4) * 4;
  int m_tiles = (m + kTileM - 1) / kTileM;
  int rounded_n = ((gemm_n + kTileN - 1) / kTileN) * kTileN;
  auto padded_hidden_states =
      torch::zeros({gemm_n, k}, hidden_states.options());
  padded_hidden_states.narrow(0, 0, n).copy_(hidden_states);
  auto candidates = torch::empty(
      {m_tiles, rounded_n * int64_t(sizeof(PackedCandidate))},
      weights.options().dtype(torch::kUInt8));
  auto output = torch::empty({n, 1}, weights.options().dtype(torch::kInt64));
  return pybind11::make_tuple(
      padded_hidden_states, candidates, output, gemm_n, rounded_n, m_tiles);
}

template <class Variant>
void launch_plain_gemm_variant_impl(
    torch::Tensor weights,
    torch::Tensor padded_hidden_states,
    torch::Tensor output) {
  TORCH_CHECK(weights.is_cuda() && padded_hidden_states.is_cuda() && output.is_cuda(),
              "inputs and output must be CUDA tensors");
  TORCH_CHECK(weights.scalar_type() == torch::kBFloat16, "weights must be bfloat16");
  TORCH_CHECK(padded_hidden_states.scalar_type() == torch::kBFloat16,
              "hidden states must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(weights.is_contiguous() && padded_hidden_states.is_contiguous() &&
              output.is_contiguous(), "inputs and output must be contiguous");
  TORCH_CHECK(weights.dim() == 2 && padded_hidden_states.dim() == 2 &&
              output.dim() == 2, "inputs and output must be matrices");
  TORCH_CHECK(weights.size(1) == padded_hidden_states.size(1),
              "hidden dimensions must match");

  int m = int(weights.size(0));
  int gemm_n = int(padded_hidden_states.size(0));
  int k = int(weights.size(1));
  TORCH_CHECK(output.size(0) == m && output.size(1) == gemm_n,
              "output shape must match the GEMM problem");
  auto byte_options = weights.options().dtype(torch::kUInt8);

  using Kernel = typename Variant::Kernel;
  using Gemm = typename Variant::Gemm;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  StrideA stride_a =
      cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b =
      cutlass::make_cute_packed_stride(StrideB{}, make_shape(gemm_n, k, 1));
  StrideC stride_c{};
  StrideD stride_d{int64_t(gemm_n), _1{}, int64_t(m) * gemm_n};
  cutlass::KernelHardwareInfo hardware_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(
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
          {},
          nullptr,
          stride_c,
          reinterpret_cast<PlainElementD*>(output.data_ptr()),
          stride_d,
      },
      hardware_info};

  Gemm gemm;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(weights.get_device());
  size_t workspace_size = Gemm::get_workspace_size(arguments);
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
}

void launch_plain_gemm(
    torch::Tensor weights,
    torch::Tensor padded_hidden_states,
    torch::Tensor output) {
  launch_plain_gemm_variant_impl<PlainVariant128x128>(
      weights, padded_hidden_states, output);
}

void launch_plain_gemm_variant(
    std::string const& variant,
    torch::Tensor weights,
    torch::Tensor padded_hidden_states,
    torch::Tensor output) {
  if (variant == "tile-64x128x64-auto") {
    launch_plain_gemm_variant_impl<PlainVariant64x128>(
        weights, padded_hidden_states, output);
  } else if (variant == "tile-128x64x64-auto") {
    launch_plain_gemm_variant_impl<PlainVariant128x64>(
        weights, padded_hidden_states, output);
  } else if (variant == "tile-128x128x64-auto") {
    launch_plain_gemm_variant_impl<PlainVariant128x128>(
        weights, padded_hidden_states, output);
  } else if (variant == "tile-64x128x64-native") {
    launch_plain_gemm_variant_impl<PlainVariant64x128Native>(
        weights, padded_hidden_states, output);
  } else if (variant == "tile-128x64x64-native") {
    launch_plain_gemm_variant_impl<PlainVariant128x64Native>(
        weights, padded_hidden_states, output);
  } else if (variant == "tile-128x128x64-native") {
    launch_plain_gemm_variant_impl<PlainVariant128x128Native>(
        weights, padded_hidden_states, output);
  } else {
    TORCH_CHECK(false, "Unknown plain GEMM variant: ", variant);
  }
}

pybind11::tuple make_plain_gemm_buffers(
    torch::Tensor weights, torch::Tensor hidden_states) {
  int m = int(weights.size(0));
  int n = int(hidden_states.size(0));
  int k = int(weights.size(1));
  int gemm_n = ((n + PlainAlignmentD - 1) / PlainAlignmentD) * PlainAlignmentD;
  auto padded_hidden_states =
      torch::zeros({gemm_n, k}, hidden_states.options());
  padded_hidden_states.narrow(0, 0, n).copy_(hidden_states);
  auto output = torch::empty({m, gemm_n}, weights.options());
  return pybind11::make_tuple(padded_hidden_states, output, gemm_n);
}

torch::Tensor plain_gemm(torch::Tensor weights, torch::Tensor hidden_states) {
  auto buffers = make_plain_gemm_buffers(weights, hidden_states);
  auto padded_hidden_states = buffers[0].cast<torch::Tensor>();
  auto output = buffers[1].cast<torch::Tensor>();
  int n = int(hidden_states.size(0));
  launch_plain_gemm(weights, padded_hidden_states, output);
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
      "make_greedy_buffers",
      &fmms_cutlass_greedy::make_greedy_buffers,
      "Allocate and populate diagnostic greedy buffers");
  module.def(
      "launch_greedy_gemm",
      &fmms_cutlass_greedy::launch_greedy_gemm,
      "Launch only the preallocated greedy GEMM");
  module.def(
      "launch_greedy_stage2",
      &fmms_cutlass_greedy::launch_greedy_stage2,
      "Launch only the preallocated greedy Stage 2 reduction");
  module.def(
      "plain_gemm",
      &fmms_cutlass_greedy::plain_gemm,
      "Plain CUTLASS BF16 GEMM with BF16 output");
  module.def(
      "make_plain_gemm_buffers",
      &fmms_cutlass_greedy::make_plain_gemm_buffers,
      "Allocate identically padded plain GEMM buffers");
  module.def(
      "launch_plain_gemm",
      &fmms_cutlass_greedy::launch_plain_gemm,
      "Launch plain CUTLASS BF16 GEMM into preallocated output");
  module.def(
      "launch_plain_gemm_variant",
      &fmms_cutlass_greedy::launch_plain_gemm_variant,
      "Launch a named diagnostic CUTLASS BF16 GEMM variant");
  module.def(
      "small_n_gemv",
      &fmms_cutlass_greedy::small_n_gemv,
      "Launch the BF16 H=1 through H=8 specialization");
  module.def(
      "launch_small_n_gemv",
      &fmms_cutlass_greedy::launch_small_n_gemv,
      "Launch preallocated BF16 H=1 through H=8 specialization");
  module.def(
      "kernel_attributes",
      &fmms_cutlass_greedy::kernel_attributes,
      "Static kernel resources and theoretical active blocks per SM");
}
