// Production-facing BF16 TP1 greedy provider built from the Gate 1h kernel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#define FMMS_GATE_STAGE2
#define FMMS_CUTLASS_LIBRARY
#define FMMS_CUTLASS_DISABLE_D
#include "evt_candidates.cu"

#if defined(FMMS_GUMBEL)
#define FMMS_GUMBEL_PARAMETERS                                               \
  , torch::Tensor temperature, uint64_t seed, int original_n, int sample_base
#define FMMS_GUMBEL_ARGUMENTS , temperature, seed, original_n, sample_base
#else
#define FMMS_GUMBEL_PARAMETERS
#define FMMS_GUMBEL_ARGUMENTS
#endif

#if defined(FMMS_ARCH_SM100)
namespace fmms_cutlass_winning {
void launch_128x64x128_c2(
    torch::Tensor weights,
    torch::Tensor padded_hidden_states,
    torch::Tensor candidates,
    torch::Tensor output,
    int gemm_n,
    int rounded_n FMMS_GUMBEL_PARAMETERS);
void launch_256x128x128_c2(
    torch::Tensor weights, torch::Tensor padded_hidden_states,
    torch::Tensor candidates, torch::Tensor output, int gemm_n,
    int rounded_n FMMS_GUMBEL_PARAMETERS);
void launch_256x128x64_c2(
    torch::Tensor weights, torch::Tensor padded_hidden_states,
    torch::Tensor candidates, torch::Tensor output, int gemm_n,
    int rounded_n FMMS_GUMBEL_PARAMETERS);
void launch_256x128x64_c4(
    torch::Tensor weights, torch::Tensor padded_hidden_states,
    torch::Tensor candidates, torch::Tensor output, int gemm_n,
    int rounded_n FMMS_GUMBEL_PARAMETERS);
void launch_256x256x64_c2(
    torch::Tensor weights, torch::Tensor padded_hidden_states,
    torch::Tensor candidates, torch::Tensor output, int gemm_n,
    int rounded_n FMMS_GUMBEL_PARAMETERS);
}
#endif

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
        cutlass::epilogue::collective::EpilogueScheduleAuto,
    class ClusterShape_ = ClusterShape>
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
          ClusterShape_,
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
          ClusterShape_,
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

#if defined(FMMS_ARCH_SM100)
// Gate 2c heuristic winners transplanted from the cutlass_profiler search.
// Every family uses the 2-SM Blackwell schedule; the raster order is a
// runtime argument selected by the variant name suffix.
using KernelSchedule2Sm = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule2Sm = cutlass::epilogue::TmaWarpSpecialized2Sm;
using HeurVariant256x64x128C2 = PlainGemmVariant<
    Shape<_256, _64, _128>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_2, _1, _1>>;
using HeurVariant128x64x128C2 = PlainGemmVariant<
    Shape<_128, _64, _128>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_2, _1, _1>>;
using HeurVariant256x128x64C2 = PlainGemmVariant<
    Shape<_256, _128, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_2, _1, _1>>;
using HeurVariant256x64x64C4 = PlainGemmVariant<
    Shape<_256, _64, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
using HeurVariant128x64x64C4 = PlainGemmVariant<
    Shape<_128, _64, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
using HeurVariant256x64x128C4 = PlainGemmVariant<
    Shape<_256, _64, _128>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
using HeurVariant256x128x64C4 = PlainGemmVariant<
    Shape<_256, _128, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
// N=256 top-32 expansion families and the explicit 1-SM cluster-(1,2,1)
// control (the family nvidia-matmul-heuristics never emits but cuBLAS uses
// at H=256: weight multicast across the two hidden-state tile CTAs).
using HeurVariant256x128x128C2 = PlainGemmVariant<
    Shape<_256, _128, _128>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_2, _1, _1>>;
using HeurVariant256x128x128C4 = PlainGemmVariant<
    Shape<_256, _128, _128>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
using HeurVariant128x128x64C4 = PlainGemmVariant<
    Shape<_128, _128, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
using HeurVariant128x128x128C4 = PlainGemmVariant<
    Shape<_128, _128, _128>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
using HeurVariant256x192x64C2 = PlainGemmVariant<
    Shape<_256, _192, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_2, _1, _1>>;
using HeurVariant256x192x64C4 = PlainGemmVariant<
    Shape<_256, _192, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_4, _1, _1>>;
using HeurVariant128x128x64C1x2 = PlainGemmVariant<
    Shape<_128, _128, _64>, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::TmaWarpSpecialized1Sm, Shape<_1, _2, _1>>;
// Full-H CTA tiles for the two H=256 cells that the audited heuristic
// search leaves outside the 1.05 threshold. Neither nvidia-matmul-heuristics
// nor the manual controls cover N-tile=256 families.
using HeurVariant128x256x64C1x1 = PlainGemmVariant<
    Shape<_128, _256, _64>, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::TmaWarpSpecialized1Sm, Shape<_1, _1, _1>>;
using HeurVariant128x256x64C2x1 = PlainGemmVariant<
    Shape<_128, _256, _64>, cutlass::gemm::KernelTmaWarpSpecialized1SmSm100,
    cutlass::epilogue::TmaWarpSpecialized1Sm, Shape<_2, _1, _1>>;
using HeurVariant256x256x64C2x1 = PlainGemmVariant<
    Shape<_256, _256, _64>, KernelSchedule2Sm, EpilogueSchedule2Sm,
    Shape<_2, _1, _1>>;
#endif

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
    int rounded_n FMMS_GUMBEL_PARAMETERS) {
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
#if defined(FMMS_GUMBEL)
      {seed, temperature.data_ptr<float>(), original_n, sample_base},
#else
      {},
#endif
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

torch::Tensor greedy(
    torch::Tensor weights, torch::Tensor hidden_states
    FMMS_GUMBEL_PARAMETERS) {
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
#if defined(FMMS_GUMBEL)
  TORCH_CHECK(
      temperature.is_cuda() && temperature.scalar_type() == torch::kFloat32 &&
          temperature.numel() == 1,
      "temperature must be a CUDA float32 scalar");
#endif
  int gemm_n = ((n + 3) / 4) * 4;
#if defined(FMMS_ARCH_SM100)
  if (n <= 256) {
    int winning_tile_n = n <= 64 ? 64 : 128;
    int winning_tile_m = n <= 64 ? 128 : 256;
    int winning_m_tiles =
        (m + winning_tile_m / 2 - 1) / (winning_tile_m / 2);
    int rounded_n =
        ((gemm_n + winning_tile_n - 1) / winning_tile_n) * winning_tile_n;
    auto padded_hidden_states =
        torch::zeros({gemm_n, k}, hidden_states.options());
    padded_hidden_states.narrow(0, 0, n).copy_(hidden_states);
    auto candidates = torch::empty(
        {winning_m_tiles,
         rounded_n * int64_t(sizeof(PackedCandidate))},
        weights.options().dtype(torch::kUInt8));
    auto output = torch::empty({n, 1}, weights.options().dtype(torch::kInt64));
    if (n <= 64) {
      fmms_cutlass_winning::launch_128x64x128_c2(
          weights, padded_hidden_states, candidates, output, gemm_n, rounded_n
          FMMS_GUMBEL_ARGUMENTS);
    } else if (k <= 4096) {
      fmms_cutlass_winning::launch_256x128x64_c4(
          weights, padded_hidden_states, candidates, output, gemm_n, rounded_n
          FMMS_GUMBEL_ARGUMENTS);
    } else if (n > 128) {
      fmms_cutlass_winning::launch_256x128x64_c2(
          weights, padded_hidden_states, candidates, output, gemm_n, rounded_n
          FMMS_GUMBEL_ARGUMENTS);
    } else {
      fmms_cutlass_winning::launch_256x128x128_c2(
          weights, padded_hidden_states, candidates, output, gemm_n, rounded_n
          FMMS_GUMBEL_ARGUMENTS);
    }
    return output;
  }
#endif
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
      weights, padded_hidden_states, candidates, gemm_n, rounded_n
      FMMS_GUMBEL_ARGUMENTS);
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
    torch::Tensor output,
    cutlass::gemm::kernel::detail::RasterOrderOptions raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic) {
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
  if (raster_order !=
      cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic) {
    arguments.scheduler.raster_order = raster_order;
    arguments.scheduler.max_swizzle_size = 1;
  }
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
  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;
  const bool along_n = variant.size() >= 3 &&
                       variant.compare(variant.size() - 3, 3, "-rn") == 0;
  const RasterOrderOptions raster =
      along_n ? RasterOrderOptions::AlongN : RasterOrderOptions::AlongM;
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
#if defined(FMMS_ARCH_SM100)
  } else if (variant == "heur-256x64x128-c2x1x1" ||
             variant == "heur-256x64x128-c2x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x64x128C2>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-128x64x128-c2x1x1" ||
             variant == "heur-128x64x128-c2x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant128x64x128C2>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x128x64-c2x1x1" ||
             variant == "heur-256x128x64-c2x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x128x64C2>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x64x64-c4x1x1" ||
             variant == "heur-256x64x64-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x64x64C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-128x64x64-c4x1x1" ||
             variant == "heur-128x64x64-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant128x64x64C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x64x128-c4x1x1" ||
             variant == "heur-256x64x128-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x64x128C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x128x64-c4x1x1" ||
             variant == "heur-256x128x64-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x128x64C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x128x128-c2x1x1" ||
             variant == "heur-256x128x128-c2x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x128x128C2>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x128x128-c4x1x1" ||
             variant == "heur-256x128x128-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x128x128C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-128x128x64-c4x1x1" ||
             variant == "heur-128x128x64-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant128x128x64C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-128x128x128-c4x1x1" ||
             variant == "heur-128x128x128-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant128x128x128C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x192x64-c2x1x1" ||
             variant == "heur-256x192x64-c2x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x192x64C2>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x192x64-c4x1x1" ||
             variant == "heur-256x192x64-c4x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x192x64C4>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-128x128x64-1sm-c1x2x1" ||
             variant == "heur-128x128x64-1sm-c1x2x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant128x128x64C1x2>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-128x256x64-1sm" ||
             variant == "heur-128x256x64-1sm-rn") {
    launch_plain_gemm_variant_impl<HeurVariant128x256x64C1x1>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-128x256x64-1sm-c2x1x1" ||
             variant == "heur-128x256x64-1sm-c2x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant128x256x64C2x1>(
        weights, padded_hidden_states, output, raster);
  } else if (variant == "heur-256x256x64-c2x1x1" ||
             variant == "heur-256x256x64-c2x1x1-rn") {
    launch_plain_gemm_variant_impl<HeurVariant256x256x64C2x1>(
        weights, padded_hidden_states, output, raster);
#endif
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
#if defined(FMMS_GUMBEL)
  module.def(
      "sample", &fmms_cutlass_greedy::greedy,
      "CUTLASS BF16 TP1 Gumbel-Max FMMS");
#else
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
#endif
}

#undef FMMS_GUMBEL_PARAMETERS
#undef FMMS_GUMBEL_ARGUMENTS
