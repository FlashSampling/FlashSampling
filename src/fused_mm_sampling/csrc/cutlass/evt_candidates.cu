// Gate 1g: reduce real CUTLASS GEMM accumulators to per-M-tile candidates.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

namespace fmms_evt_candidates {

#ifndef FMMS_TILE_M
#define FMMS_TILE_M 128
#endif
#ifndef FMMS_TILE_N
#define FMMS_TILE_N 128
#endif
#ifndef FMMS_TILE_K
#define FMMS_TILE_K 64
#endif

constexpr int kTileM = FMMS_TILE_M;
constexpr int kTileN = FMMS_TILE_N;
constexpr int kK = FMMS_TILE_K;

#if defined(FMMS_ARCH_SM90)
constexpr char const* kArchitecture = "sm90";
using ArchTag = cutlass::arch::Sm90;
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;
using EpilogueTile = Shape<_64, _32>;
#elif defined(FMMS_ARCH_SM100)
constexpr char const* kArchitecture = "sm100";
using ArchTag = cutlass::arch::Sm100;
#if defined(FMMS_SM100_2SM)
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;
#else
using MainloopSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
#endif
using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
#else
#error "Compile with FMMS_ARCH_SM90 or FMMS_ARCH_SM100"
#endif

using PackedCandidate = uint64_t;

CUTLASS_HOST_DEVICE
PackedCandidate pack_candidate(float value, int index) {
  union {
    float value;
    uint32_t bits;
  } encoded{value};
  return (uint64_t(encoded.bits) << 32) | uint32_t(index);
}

CUTLASS_HOST_DEVICE
float candidate_value(PackedCandidate candidate) {
  union {
    uint32_t bits;
    float value;
  } decoded{uint32_t(candidate >> 32)};
  return decoded.value;
}

CUTLASS_HOST_DEVICE
int candidate_index(PackedCandidate candidate) {
  return int(uint32_t(candidate));
}

CUTLASS_HOST_DEVICE
PackedCandidate choose_candidate(PackedCandidate lhs, PackedCandidate rhs) {
  float lhs_value = candidate_value(lhs);
  float rhs_value = candidate_value(rhs);
  int lhs_index = candidate_index(lhs);
  int rhs_index = candidate_index(rhs);
  return rhs_value > lhs_value ||
          (rhs_value == lhs_value && rhs_index < lhs_index)
      ? rhs
      : lhs;
}

template <class T>
struct CandidateReduce {
  CUTLASS_HOST_DEVICE T operator()(T const& lhs, T const& rhs) const {
    T output;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < int(T::kElements); ++i) {
      output[i] = choose_candidate(lhs[i], rhs[i]);
    }
    return output;
  }
};

template <>
struct CandidateReduce<PackedCandidate> {
  CUTLASS_HOST_DEVICE PackedCandidate operator()(
      PackedCandidate lhs,
      PackedCandidate rhs) const {
    return choose_candidate(lhs, rhs);
  }
};

template <class T>
struct AtomicCandidateReduce {
  CUTLASS_DEVICE void operator()(T* pointer, T candidate) const {
    auto* bits = reinterpret_cast<unsigned long long*>(pointer);
    unsigned long long observed = atomicCAS(bits, 0, 0);
    while (true) {
      unsigned long long assumed = observed;
      T winner = choose_candidate(T(assumed), candidate);
      if (winner == T(assumed)) {
        return;
      }
      observed = atomicCAS(bits, assumed, static_cast<unsigned long long>(winner));
      if (observed == assumed) {
        return;
      }
    }
  }
};

}  // namespace fmms_evt_candidates

namespace cutlass {

template <class T>
struct is_atomic<fmms_evt_candidates::AtomicCandidateReduce<T>>
    : platform::true_type {};

}  // namespace cutlass

namespace fmms_evt_candidates {

#if defined(FMMS_FINAL_REDUCTION)
void check_cuda(cudaError_t status, char const* operation);

__global__ void atomic_candidate_tie_kernel(PackedCandidate* output) {
  int index = threadIdx.x == 0 ? 254 : 126;
  AtomicCandidateReduce<PackedCandidate>{}(
      output, pack_candidate(-872.0f, index));
}

void verify_atomic_candidate_tie() {
  PackedCandidate identity = pack_candidate(-INFINITY, INT_MAX);
  PackedCandidate* device_output;
  check_cuda(cudaMalloc(&device_output, sizeof(PackedCandidate)), "malloc atomic tie");
  check_cuda(
      cudaMemcpy(
          device_output,
          &identity,
          sizeof(PackedCandidate),
          cudaMemcpyHostToDevice),
      "initialize atomic tie");
  atomic_candidate_tie_kernel<<<1, 2>>>(device_output);
  check_cuda(cudaGetLastError(), "launch atomic tie");
  PackedCandidate actual;
  check_cuda(
      cudaMemcpy(
          &actual,
          device_output,
          sizeof(PackedCandidate),
          cudaMemcpyDeviceToHost),
      "copy atomic tie");
  check_cuda(cudaFree(device_output), "free atomic tie");
  if (actual != pack_candidate(-872.0f, 126)) {
    std::cerr << "standalone atomic tie failed: expected index 126, actual "
              << candidate_index(actual) << '\n';
    std::exit(EXIT_FAILURE);
  }
}
#endif

struct PackCandidate {
  struct SharedStorage {};
  struct Arguments {};
  struct Params {};

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const&, void*) {
    return {};
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

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }
  CUTLASS_HOST_DEVICE PackCandidate() : params() {}
  CUTLASS_HOST_DEVICE PackCandidate(
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
    int tile_m;

    template <
        typename ElementAccumulator,
        int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<PackedCandidate, FragmentSize> visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const& accumulators,
        int epi_v,
        int epi_m,
        int epi_n) {
      cutlass::Array<PackedCandidate, FragmentSize> output;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
#if defined(FMMS_ARCH_SM90)
        int consumer_thread = int(threadIdx.x) - 128;
        int warp = consumer_thread / 32;
        int lane = consumer_thread % 32;
        int local_m =
            warp * 16 + lane / 4 + epi_m * 64 + ((i % 4) / 2) * 8;
#else
#if defined(FMMS_SM100_2SM)
        int consumer_thread = int(threadIdx.x) - 128;
        // SM100 2-SM schedules expose an M tile coordinate per cooperating
        // CTA. Each CTA owns 64 rows, so tile_m already incorporates the
        // cluster rank. Adding the rank here would double-count CTA 1.
        int local_m = consumer_thread % 64;
#else
        int local_m = int(threadIdx.x) - 128;
#endif
#endif
#if defined(FMMS_SM100_2SM)
        int global_m = tile_m * (kTileM / 2) + local_m;
#else
        int global_m = tile_m * kTileM + local_m;
#endif
        output[i] = pack_candidate(float(accumulators[i]), global_m);
      }
      return output;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks{
        {}, int(get<0>(args.tile_coord_mnkl))};
  }
};

struct DiscardPackedOutput {
  struct SharedStorage {};
  struct Arguments {};
  struct Params {};

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const&, void*) {
    return {};
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

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }
  CUTLASS_HOST_DEVICE DiscardPackedOutput() : params() {}
  CUTLASS_HOST_DEVICE DiscardPackedOutput(
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
    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<float, FragmentSize> visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&,
        int,
        int,
        int) {
      cutlass::Array<float, FragmentSize> output;
      output.fill(0.0f);
      return output;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const&) {
    return ConsumerStoreCallbacks{};
  }
};

using TileShape = Shape<Int<FMMS_TILE_M>, Int<FMMS_TILE_N>, Int<FMMS_TILE_K>>;
#if defined(FMMS_SM100_2SM)
using ClusterShape = Shape<_2, _1, _1>;
#else
using ClusterShape = Shape<_1, _1, _1>;
#endif
using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
// The packed auxiliary candidate buffer is the gate output. Gate 1 keeps an
// FP32 D store for direct EVT inspection, while the production provider
// disables D and emits only the candidates.
using ElementC = float;
#if defined(FMMS_CUTLASS_DISABLE_D)
using ElementD = void;
#else
using ElementD = float;
#endif
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using RowReduction = cutlass::epilogue::fusion::Sm90RowReduction<
    CandidateReduce,
    CandidateReduce,
#if defined(FMMS_FINAL_REDUCTION)
    AtomicCandidateReduce,
#else
    CandidateReduce,
#endif
    0,
    TileShape,
    PackedCandidate,
    PackedCandidate,
    cutlass::FloatRoundStyle::round_to_nearest,
    Stride<_0, _1, _0>,
    2,
    false,
#if defined(FMMS_FINAL_REDUCTION)
    true,
#else
    false,
#endif
    true>;
using CandidateReductionEVT = cutlass::epilogue::fusion::Sm90EVT<
    RowReduction,
    cutlass::epilogue::fusion::Sm90AccFetch>;
struct CandidateEVT
    : cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
          PackCandidate,
          DiscardPackedOutput,
          CandidateReductionEVT> {
  using Base = cutlass::epilogue::fusion::Sm90SplitTreeVisitor<
      PackCandidate,
      DiscardPackedOutput,
      CandidateReductionEVT>;
  using Base::Base;
  using ElementAux = float;
};

constexpr int kAlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
constexpr int kAlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
constexpr int kAlignmentC = 128 / cutlass::sizeof_bits_v<ElementC>;
constexpr int kAlignmentD = 128 / cutlass::sizeof_bits_v<float>;

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
        CandidateEVT>::CollectiveOp;

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
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

struct Case {
  std::string family;
  std::string name;
  int m;
  int n;
  int pattern;
};

void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void check_cutlass(cutlass::Status status, char const* operation) {
  if (status != cutlass::Status::kSuccess) {
    std::cerr << operation << " failed: "
              << cutlassGetStatusString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

float logit_for(int pattern, int m, int tile) {
  int local_m = m % kTileM;
  if (pattern == 0) {
    return -1000.0f + float(local_m);
  }
  if (pattern == 1) {
    return -10.0f - float(local_m);
  }
  if (pattern == 2) {
    return local_m == 3 || local_m == 97 ? 7.0f : -20.0f;
  }
  if (pattern == 3) {
    return local_m == 5 ? 11.0f : -30.0f;
  }
  if (pattern == 5) {
    return m == 0 ? 17.0f : -50.0f;
  }
  if (pattern == 6) {
    return m == 128 ? 18.0f : -50.0f;
  }
  if (pattern == 7) {
    return m == 256 ? 19.0f : -50.0f;
  }
  if (pattern == 8) {
    return m == 7 || m == 263 ? 20.0f : -50.0f;
  }
  int winner = tile % 2 == 0 ? 0 : 127;
  return local_m == winner ? 13.0f : -40.0f;
}

#if defined(FMMS_GATE_STAGE2)
__global__ void stage2_kernel(
    PackedCandidate const* candidates,
    PackedCandidate* final_candidates,
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
  final_candidates[column] = winner;
}
#endif

void run_case(Case const& test_case) {
  int m_tiles = (test_case.m + kTileM - 1) / kTileM;
  int rounded_n = ((test_case.n + kTileN - 1) / kTileN) * kTileN;
  std::vector<ElementA> matrix_a(test_case.m * kK, ElementA(0.0f));
  std::vector<ElementB> matrix_b(kK * test_case.n, ElementB(0.0f));
  for (int m = 0; m < test_case.m; ++m) {
    matrix_a[m * kK] =
        ElementA(logit_for(test_case.pattern, m, m / kTileM));
  }
  for (int n = 0; n < test_case.n; ++n) {
    matrix_b[n * kK] = ElementB(1.0f);
  }

  ElementA* device_a;
  ElementB* device_b;
  ElementD* device_d;
  PackedCandidate* device_candidates;
#if defined(FMMS_GATE_STAGE2)
  PackedCandidate* device_final_candidates;
#endif
  check_cuda(cudaMalloc(&device_a, matrix_a.size() * sizeof(ElementA)), "malloc A");
  check_cuda(cudaMalloc(&device_b, matrix_b.size() * sizeof(ElementB)), "malloc B");
  check_cuda(
      cudaMalloc(&device_d, test_case.m * rounded_n * sizeof(ElementD)),
      "malloc D");
  check_cuda(
      cudaMalloc(
          &device_candidates,
          m_tiles * rounded_n * sizeof(PackedCandidate)),
      "malloc candidates");
#if defined(FMMS_GATE_STAGE2)
  check_cuda(
      cudaMalloc(
          &device_final_candidates,
          test_case.n * sizeof(PackedCandidate)),
      "malloc final candidates");
#endif
  check_cuda(
      cudaMemcpy(
          device_a,
          matrix_a.data(),
          matrix_a.size() * sizeof(ElementA),
          cudaMemcpyHostToDevice),
      "copy A");
  check_cuda(
      cudaMemcpy(
          device_b,
          matrix_b.data(),
          matrix_b.size() * sizeof(ElementB),
          cudaMemcpyHostToDevice),
      "copy B");

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, make_shape(test_case.m, kK, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, make_shape(test_case.n, kK, 1));
  StrideC stride_c{
      int64_t(rounded_n), _1{}, int64_t(test_case.m) * rounded_n};
  StrideD stride_d{
      int64_t(rounded_n), _1{}, int64_t(test_case.m) * rounded_n};
  PackedCandidate identity = pack_candidate(-INFINITY, INT_MAX);
#if defined(FMMS_FINAL_REDUCTION)
  std::vector<PackedCandidate> candidate_identities(
      m_tiles * rounded_n, identity);
  check_cuda(
      cudaMemcpy(
          device_candidates,
          candidate_identities.data(),
          candidate_identities.size() * sizeof(PackedCandidate),
          cudaMemcpyHostToDevice),
      "initialize candidate identities");
#endif

  typename CandidateEVT::Arguments evt_arguments{
      {},
      {{}, {device_candidates, identity, {}}},
      {}};
  cutlass::KernelHardwareInfo hardware_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(0);
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {test_case.m, test_case.n, kK, 1},
      {device_a, stride_a, device_b, stride_b},
      {evt_arguments, nullptr, stride_c, device_d, stride_d},
      hardware_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  void* workspace = nullptr;
  if (workspace_size != 0) {
    check_cuda(cudaMalloc(&workspace, workspace_size), "malloc workspace");
  }
  check_cutlass(gemm.can_implement(arguments), "can_implement");
  check_cutlass(gemm.initialize(arguments, workspace), "initialize");
  check_cutlass(gemm.run(), "run");
#if defined(FMMS_GATE_STAGE2)
  constexpr int kStage2Threads = 256;
  stage2_kernel<<<
      (test_case.n + kStage2Threads - 1) / kStage2Threads,
      kStage2Threads>>>(
      device_candidates,
      device_final_candidates,
      m_tiles,
      rounded_n,
      test_case.n);
  check_cuda(cudaGetLastError(), "launch Stage 2");
#endif
  check_cuda(cudaDeviceSynchronize(), "synchronize");

  std::vector<PackedCandidate> candidates(m_tiles * rounded_n);
  check_cuda(
      cudaMemcpy(
          candidates.data(),
          device_candidates,
          candidates.size() * sizeof(PackedCandidate),
          cudaMemcpyDeviceToHost),
      "copy candidates");

#if defined(FMMS_GATE_STAGE2)
  std::vector<PackedCandidate> final_candidates(test_case.n);
  check_cuda(
      cudaMemcpy(
          final_candidates.data(),
          device_final_candidates,
          final_candidates.size() * sizeof(PackedCandidate),
          cudaMemcpyDeviceToHost),
      "copy final candidates");
#endif

#if defined(FMMS_FINAL_REDUCTION)
  for (int n = 0; n < test_case.n; ++n) {
    PackedCandidate expected = identity;
    for (int m = 0; m < test_case.m; ++m) {
      expected = choose_candidate(
          expected,
          pack_candidate(
              float(matrix_a[m * kK]) * float(matrix_b[n * kK]),
              m));
    }
    PackedCandidate actual = candidates[n];
    int passed = actual == expected;
    std::cout << kArchitecture << ',' << test_case.family << ','
              << test_case.name << ',' << test_case.m << ','
              << test_case.n << ',' << kK << ",-1," << n
              << ",0," << test_case.m << ','
              << uint32_t(expected >> 32) << ','
              << uint32_t(actual >> 32) << ','
              << candidate_index(expected) << ','
              << candidate_index(actual) << ',' << passed << '\n';
    if (!passed) {
      std::exit(EXIT_FAILURE);
    }
  }
#else
  for (int tile = 0; tile < m_tiles; ++tile) {
    int begin = tile * kTileM;
    int end = std::min(begin + kTileM, test_case.m);
    for (int n = 0; n < test_case.n; ++n) {
      PackedCandidate expected = identity;
      for (int m = begin; m < end; ++m) {
        expected = choose_candidate(
            expected,
            pack_candidate(
                float(matrix_a[m * kK]) * float(matrix_b[n * kK]),
                m));
      }
      PackedCandidate actual = candidates[tile * rounded_n + n];
      int passed = actual == expected;
#if defined(FMMS_GATE_STAGE2)
      std::cout << kArchitecture << ',' << test_case.family << ','
                << test_case.name << ',' << test_case.m << ','
                << test_case.n << ',' << kK << ",candidate," << tile << ','
                << n << ',' << begin << ',' << end << ','
                << uint32_t(expected >> 32) << ','
                << uint32_t(actual >> 32) << ','
                << candidate_index(expected) << ','
                << candidate_index(actual) << ',' << passed << '\n';
#else
      std::cout << kArchitecture << ',' << test_case.family << ','
                << test_case.name << ',' << test_case.m << ','
                << test_case.n << ',' << kK << ',' << tile << ',' << n
                << ',' << begin << ',' << end << ','
                << uint32_t(expected >> 32) << ','
                << uint32_t(actual >> 32) << ','
                << candidate_index(expected) << ','
                << candidate_index(actual) << ',' << passed << '\n';
#endif
      if (!passed) {
        std::exit(EXIT_FAILURE);
      }
    }
  }
#endif

#if defined(FMMS_GATE_STAGE2)
  for (int n = 0; n < test_case.n; ++n) {
    PackedCandidate expected = identity;
    for (int m = 0; m < test_case.m; ++m) {
      expected = choose_candidate(
          expected,
          pack_candidate(
              float(matrix_a[m * kK]) * float(matrix_b[n * kK]),
              m));
    }
    PackedCandidate actual = final_candidates[n];
    int passed = actual == expected;
    std::cout << kArchitecture << ',' << test_case.family << ','
              << test_case.name << ',' << test_case.m << ','
              << test_case.n << ',' << kK << ",final,-1," << n
              << ",0," << test_case.m << ','
              << uint32_t(expected >> 32) << ','
              << uint32_t(actual >> 32) << ','
              << candidate_index(expected) << ','
              << candidate_index(actual) << ',' << passed << '\n';
    if (!passed) {
      std::exit(EXIT_FAILURE);
    }
  }
  check_cuda(cudaFree(device_final_candidates), "free final candidates");
#endif

  if (workspace != nullptr) {
    check_cuda(cudaFree(workspace), "free workspace");
  }
  check_cuda(cudaFree(device_candidates), "free candidates");
  check_cuda(cudaFree(device_d), "free D");
  check_cuda(cudaFree(device_b), "free B");
  check_cuda(cudaFree(device_a), "free A");
}

}  // namespace fmms_evt_candidates

#if !defined(FMMS_CUTLASS_LIBRARY)
int main() {
  using fmms_evt_candidates::Case;
#if defined(FMMS_FINAL_REDUCTION)
  fmms_evt_candidates::verify_atomic_candidate_tie();
#endif
#if defined(FMMS_GATE_STAGE2)
  std::vector<Case> cases{
      {"winner_tiles", "winner_first_tile", 257, 4, 5},
      {"winner_tiles", "winner_middle_tile", 257, 68, 6},
      {"winner_tiles", "winner_last_tile", 257, 132, 7},
      {"boundaries", "complete_tiles", 256, 128, 4},
      {"boundaries", "partial_m_n", 129, 68, 0},
      {"negative_ties", "all_negative", 255, 4, 1},
      {"negative_ties", "within_tile_tie", 128, 64, 2},
      {"cross_tile_ties", "equal_global_maxima", 264, 4, 8},
  };
  std::cout
      << "architecture,family,case,m,n,k,row_type,m_tile,column,tile_begin,"
         "tile_end,expected_value_bits,actual_value_bits,expected_index,"
         "actual_index,pass\n";
#else
  std::vector<Case> cases{
      {"boundaries", "complete_tiles", 256, 128, 4},
      {"tile_offsets", "edge_winners", 257, 132, 0},
      {"boundaries", "partial_m_n", 129, 68, 0},
      {"negative_ties", "all_negative", 255, 4, 1},
      {"negative_ties", "within_tile_tie", 128, 64, 2},
      {"cross_tile_ties", "equal_tile_candidates", 257, 4, 3},
  };
  std::cout
      << "architecture,family,case,m,n,k,m_tile,column,tile_begin,tile_end,"
         "expected_value_bits,actual_value_bits,expected_index,actual_index,pass\n";
#endif
  for (Case const& test_case : cases) {
    fmms_evt_candidates::run_case(test_case);
  }
  return 0;
}
#endif
