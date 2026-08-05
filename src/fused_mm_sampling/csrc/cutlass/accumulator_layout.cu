// Records which CUTLASS epilogue thread and fragment slot owns each output coordinate.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace fmms_layout {

#ifndef FMMS_TILE_M
#define FMMS_TILE_M 128
#endif
#ifndef FMMS_TILE_N
#define FMMS_TILE_N 128
#endif
#ifndef FMMS_TILE_K
#define FMMS_TILE_K 64
#endif

constexpr int kM = FMMS_TILE_M;
constexpr int kN = FMMS_TILE_N;
constexpr int kK = FMMS_TILE_K;

#ifndef FMMS_PROBLEM_M
#define FMMS_PROBLEM_M FMMS_TILE_M
#endif
#ifndef FMMS_PROBLEM_N
#define FMMS_PROBLEM_N FMMS_TILE_N
#endif
#ifndef FMMS_PROBLEM_K
#define FMMS_PROBLEM_K FMMS_TILE_K
#endif

constexpr int kProblemM = FMMS_PROBLEM_M;
constexpr int kProblemN = FMMS_PROBLEM_N;
constexpr int kProblemK = FMMS_PROBLEM_K;

void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void check_cutlass(cutlass::Status status, char const* operation) {
  if (status != cutlass::Status::kSuccess) {
    std::cerr << operation << " failed: " << cutlassGetStatusString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

struct FragmentOwnershipEncoder {
  struct SharedStorage {};
  struct Arguments {
    uint32_t* owner_counts = nullptr;
  };
  struct Params {
    uint32_t* owner_counts;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& arguments, void*) {
    return {arguments.owner_counts};
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) {
    return true;
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&,
      Arguments const&,
      void*,
      cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool is_C_load_needed() const {
    return false;
  }

  CUTLASS_HOST_DEVICE FragmentOwnershipEncoder() : params{nullptr} {}

  CUTLASS_HOST_DEVICE FragmentOwnershipEncoder(
      Params const& params_,
      SharedStorage const&)
      : params(params_) {}

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    uint32_t* owner_counts;

    template <
        typename ElementAccumulator,
        typename ElementInput,
        int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<float, FragmentSize> visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const& accumulators,
        int epi_v,
        int epi_m,
        int epi_n,
        cutlass::Array<ElementInput, FragmentSize> const&) {
      cutlass::Array<float, FragmentSize> output;
      CUTLASS_PRAGMA_UNROLL
      for (int fragment = 0; fragment < FragmentSize; ++fragment) {
#if defined(FMMS_VALIDATE_PIPELINE)
        output[fragment] = float(accumulators[fragment]);
#else
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
        static_assert(kM == 256 && kN == 128);
        int consumer_thread = int(threadIdx.x) - 128;
        int consumer_warpgroup = consumer_thread / 128;
        consumer_thread %= 128;
        int m = consumer_thread % 128 + 128 * block_rank_in_cluster();
        int n = FragmentSize * epi_n + fragment;
#if defined(FMMS_FIVE_EPILOGUE_WARPGROUPS)
        if (fragment % 5 == consumer_warpgroup) {
          atomicAdd(owner_counts + m * kN + n, 1u);
          atomicOr(
              owner_counts + m * kN + n,
              1u << (8 + consumer_warpgroup));
        }
#elif defined(FMMS_FOUR_EPILOGUE_WARPGROUPS)
        if (fragment % 4 == consumer_warpgroup) {
          atomicAdd(owner_counts + m * kN + n, 1u);
          atomicOr(
              owner_counts + m * kN + n,
              1u << (8 + consumer_warpgroup));
        }
#elif defined(FMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED)
        if (fragment % 2 == consumer_warpgroup) {
          atomicAdd(owner_counts + m * kN + n, 1u);
          atomicOr(
              owner_counts + m * kN + n,
              1u << (8 + consumer_warpgroup));
        }
#else
        atomicAdd(owner_counts + m * kN + n, 1u);
        atomicOr(
            owner_counts + m * kN + n,
            1u << (8 + consumer_warpgroup));
#endif
#endif
        // The complete record remains below 2^24 and is therefore exactly
        // representable in the diagnostic FP32 output.
        uint32_t code = uint32_t(threadIdx.x)
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
            | (uint32_t(fragment) << 9)
#else
            | (uint32_t(fragment) << 8)
#endif
            | (uint32_t(epi_v) << 13)
            | (uint32_t(epi_m) << 15)
            | (uint32_t(epi_n) << 18)
            | (uint32_t(block_rank_in_cluster() & 0x3) << 22);
        output[fragment] = float(code);
#endif
      }
      return output;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&) {
    return ConsumerStoreCallbacks{{}, params.owner_counts};
  }
};

#if defined(FMMS_ARCH_SM90)
using ArchTag = cutlass::arch::Sm90;
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using EpilogueTile = Shape<_64, _32>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
#elif defined(FMMS_ARCH_SM100)
using ArchTag = cutlass::arch::Sm100;
#if defined(FMMS_SM100_2SM)
#ifndef FMMS_CLUSTER_M
#define FMMS_CLUSTER_M 2
#endif
using TileShape = Shape<Int<FMMS_TILE_M>, Int<FMMS_TILE_N>, Int<FMMS_TILE_K>>;
using ClusterShape = Shape<Int<FMMS_CLUSTER_M>, _1, _1>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
#else
using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using EpilogueSchedule =
    cutlass::epilogue::TmaWarpSpecialized1Sm;
#endif
using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
#else
#error "Compile with FMMS_ARCH_SM90 or FMMS_ARCH_SM100"
#endif

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kAlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
constexpr int kAlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
constexpr int kAlignmentC = 128 / cutlass::sizeof_bits_v<ElementC>;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;

using DiagnosticEVT = cutlass::epilogue::fusion::Sm90EVT<
    FragmentOwnershipEncoder,
    cutlass::epilogue::fusion::Sm90AccFetch>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        EpilogueTile,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        LayoutC,
        kAlignmentC,
        ElementD,
        LayoutD,
        kAlignmentD,
        EpilogueSchedule,
        DiagnosticEVT>::CollectiveOp;

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
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void run_diagnostic(int iterations) {
  ElementA* matrix_a;
  ElementB* matrix_b;
  ElementD* matrix_d;
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
  uint32_t* owner_counts;
#endif
  check_cuda(
      cudaMalloc(&matrix_a, kProblemM * kProblemK * sizeof(ElementA)),
      "cudaMalloc(matrix_a)");
  check_cuda(
      cudaMalloc(&matrix_b, kProblemK * kProblemN * sizeof(ElementB)),
      "cudaMalloc(matrix_b)");
  check_cuda(
      cudaMalloc(&matrix_d, kProblemM * kProblemN * sizeof(ElementD)),
      "cudaMalloc(matrix_d)");
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
  check_cuda(
      cudaMalloc(&owner_counts, kProblemM * kProblemN * sizeof(uint32_t)),
      "cudaMalloc(owner_counts)");
#endif
#if defined(FMMS_VALIDATE_PIPELINE)
  std::vector<ElementA> host_a(kProblemM * kProblemK, ElementA(1.0f));
  std::vector<ElementB> host_b(kProblemK * kProblemN, ElementB(1.0f));
#elif defined(FMMS_DIAGNOSE_PARTITIONED_TMEM_LOAD)
  std::vector<ElementA> host_a(kProblemM * kProblemK);
  std::vector<ElementB> host_b(kProblemK * kProblemN);
  for (int m = 0; m < kProblemM; ++m) {
    for (int k = 0; k < kProblemK; ++k) {
      host_a[m * kProblemK + k] = ElementA(1 + m % 7);
    }
  }
  for (int k = 0; k < kProblemK; ++k) {
    for (int n = 0; n < kProblemN; ++n) {
      host_b[k * kProblemN + n] = ElementB(1 + n % 11);
    }
  }
#endif
#if defined(FMMS_VALIDATE_PIPELINE) || \
    defined(FMMS_DIAGNOSE_PARTITIONED_TMEM_LOAD)
  check_cuda(
      cudaMemcpy(
          matrix_a,
          host_a.data(),
          host_a.size() * sizeof(ElementA),
          cudaMemcpyHostToDevice),
      "initialize A");
  check_cuda(
      cudaMemcpy(
          matrix_b,
          host_b.data(),
          host_b.size() * sizeof(ElementB),
          cudaMemcpyHostToDevice),
      "initialize B");
#else
  check_cuda(
      cudaMemset(matrix_a, 0, kProblemM * kProblemK * sizeof(ElementA)),
      "zero A");
  check_cuda(
      cudaMemset(matrix_b, 0, kProblemK * kProblemN * sizeof(ElementB)),
      "zero B");
#endif
  check_cuda(
      cudaMemset(matrix_d, 0xff, kProblemM * kProblemN * sizeof(ElementD)),
      "fill D");
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
  check_cuda(
      cudaMemset(
          owner_counts,
          0,
          kProblemM * kProblemN * sizeof(uint32_t)),
      "zero owner counts");
#endif

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  StrideA stride_a =
      cutlass::make_cute_packed_stride(
          StrideA{}, make_shape(kProblemM, kProblemK, 1));
  StrideB stride_b =
      cutlass::make_cute_packed_stride(
          StrideB{}, make_shape(kProblemN, kProblemK, 1));
  StrideC stride_c =
      cutlass::make_cute_packed_stride(
          StrideC{}, make_shape(kProblemM, kProblemN, 1));
  StrideD stride_d =
      cutlass::make_cute_packed_stride(
          StrideD{}, make_shape(kProblemM, kProblemN, 1));

  cutlass::KernelHardwareInfo hardware_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(0);
  typename DiagnosticEVT::Arguments evt_arguments{
      {},
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
      {owner_counts}
#else
      {}
#endif
  };
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {kProblemM, kProblemN, kProblemK, 1},
      {matrix_a, stride_a, matrix_b, stride_b},
      {evt_arguments, nullptr, stride_c, matrix_d, stride_d},
      hardware_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size != 0) {
    check_cuda(cudaMalloc(&workspace, workspace_size), "cudaMalloc(workspace)");
  }
  check_cutlass(gemm.can_implement(arguments), "can_implement");
  check_cutlass(gemm.initialize(arguments, workspace), "initialize");
  for (int iteration = 0; iteration < iterations; ++iteration) {
    check_cutlass(gemm.run(), "run");
  }
  check_cuda(cudaDeviceSynchronize(), "synchronize");

  std::vector<ElementD> output(kProblemM * kProblemN);
  check_cuda(
      cudaMemcpy(
          output.data(),
          matrix_d,
          output.size() * sizeof(ElementD),
          cudaMemcpyDeviceToHost),
      "copy output");
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
  std::vector<uint32_t> host_owner_counts(kProblemM * kProblemN);
  check_cuda(
      cudaMemcpy(
          host_owner_counts.data(),
          owner_counts,
          host_owner_counts.size() * sizeof(uint32_t),
          cudaMemcpyDeviceToHost),
      "copy owner counts");
#endif

#if defined(FMMS_VALIDATE_PIPELINE)
  for (ElementD value : output) {
    if (value != ElementD(kProblemK)) {
      std::cerr << "Unexpected output: " << value
                << ", expected " << kProblemK << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  std::cout << "{\"passed\":true,\"iterations\":" << iterations
            << ",\"m\":" << kProblemM
            << ",\"n\":" << kProblemN
            << ",\"k\":" << kProblemK
            << ",\"coordinates\":" << output.size() << "}\n";
#else
  std::cout << "m,n,thread,fragment,epi_v,epi_m,epi_n,cta";
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
  std::cout << ",owner_count,owner_group_mask";
#endif
  std::cout << '\n';
  for (int m = 0; m < kProblemM; ++m) {
    for (int n = 0; n < kProblemN; ++n) {
      uint32_t code = uint32_t(output[m * kProblemN + n]);
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
      uint32_t thread = code & 0x1ff;
      uint32_t fragment = (code >> 9) & 0xf;
#else
      uint32_t thread = code & 0xff;
      uint32_t fragment = (code >> 8) & 0x1f;
#endif
      uint32_t epi_v = (code >> 13) & 0x3;
      uint32_t epi_m = (code >> 15) & 0x7;
      uint32_t epi_n = (code >> 18) & 0xf;
      uint32_t cta = (code >> 22) & 0x3;
      std::cout << m << ',' << n << ',' << thread << ',' << fragment << ','
                << epi_v << ',' << epi_m << ',' << epi_n << ',' << cta;
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
      uint32_t owner_word = host_owner_counts[m * kProblemN + n];
      std::cout << ',' << (owner_word & 0xff) << ',' << (owner_word >> 8);
#endif
      std::cout << '\n';
    }
  }
#endif

  if (workspace != nullptr) {
    check_cuda(cudaFree(workspace), "cudaFree(workspace)");
  }
#if defined(FMMS_VALIDATE_OWNER_COUNTS)
  check_cuda(cudaFree(owner_counts), "cudaFree(owner_counts)");
#endif
  check_cuda(cudaFree(matrix_d), "cudaFree(matrix_d)");
  check_cuda(cudaFree(matrix_b), "cudaFree(matrix_b)");
  check_cuda(cudaFree(matrix_a), "cudaFree(matrix_a)");
}

}  // namespace fmms_layout

int main(int argc, char** argv) {
  int iterations = argc == 2 ? std::stoi(argv[1]) : 1;
  fmms_layout::run_diagnostic(iterations);
  return 0;
}
