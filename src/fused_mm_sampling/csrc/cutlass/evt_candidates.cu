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
#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#if defined(FMMS_GUMBEL)
#include "stateless_philox.cuh"
#endif

using namespace cute;

namespace fmms_evt_candidates {

#if defined(FMMS_GUMBEL)
#if defined(FMMS_INLINE_GUMBEL)
__device__ __forceinline__
#else
__device__ __noinline__
#endif
float gumbel_noise(
    uint64_t seed,
    uint32_t sample_idx,
    uint32_t hidden_idx,
    uint64_t vocab_idx) {
  auto random = fmms::philox4x32_10(
      seed, sample_idx, hidden_idx, vocab_idx);
  float uniform = fmms::uniform_open_open(random.x);
#if defined(FMMS_FAST_LOG)
  return -__logf(-__logf(uniform));
#else
  return -logf(-logf(uniform));
#endif
}
#endif

#ifndef FMMS_TILE_M
#define FMMS_TILE_M 128
#endif
#ifndef FMMS_TILE_N
#define FMMS_TILE_N 128
#endif
#ifndef FMMS_TILE_K
#define FMMS_TILE_K 64
#endif
#ifndef FMMS_CLUSTER_M
#define FMMS_CLUSTER_M 2
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
#if defined(FMMS_GUMBEL)
  struct Arguments {
    uint64_t seed;
    float const* temperature;
    int original_n;
    int sample_base;
  };
  struct Params {
    uint64_t seed;
    float const* temperature;
    int original_n;
    int sample_base;
  };
#else
  struct Arguments {};
  struct Params {};
#endif

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& arguments, void*) {
#if defined(FMMS_GUMBEL)
    return {
        arguments.seed, arguments.temperature,
        arguments.original_n, arguments.sample_base};
#else
    return {};
#endif
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
    int problem_m;
#if defined(FMMS_GUMBEL)
    int tile_n;
    Params params;
#endif

    template <
        typename ElementAccumulator,
        int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<PackedCandidate, FragmentSize> visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const& accumulators,
        int epi_v,
        int epi_m,
        int epi_n) {
      cutlass::Array<PackedCandidate, FragmentSize> output;
#if defined(FMMS_GUMBEL) && defined(FMMS_GUMBEL_PARTIAL_UNROLL)
#if FMMS_GUMBEL_PARTIAL_UNROLL == 2
#pragma unroll 2
#elif FMMS_GUMBEL_PARTIAL_UNROLL == 4
#pragma unroll 4
#elif FMMS_GUMBEL_PARTIAL_UNROLL == 8
#pragma unroll 8
#else
#error "Unsupported FMMS_GUMBEL_PARTIAL_UNROLL value"
#endif
#else
      CUTLASS_PRAGMA_UNROLL
#endif
      for (int i = 0; i < FragmentSize; ++i) {
        output[i] = visit_one<ElementAccumulator, FragmentSize>(
            accumulators[i], i, epi_v, epi_m, epi_n);
      }
      return output;
    }

    template <
        typename ElementAccumulator,
        int FragmentSize>
    CUTLASS_DEVICE PackedCandidate visit_one(
        ElementAccumulator accumulator,
        int i,
        int epi_v,
        int epi_m,
        int epi_n) {
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
        // CTA. A CTA owns half of the cluster's M tile, so tile_m already
        // incorporates the cluster rank. Adding the rank here would
        // double-count the second CTA. Gate 2d's ownership diagnostic shows
        // that the 128-row family folds the two 64-thread consumer groups
        // onto the same M rows, while each thread owns one M row for the
        // 256-row families.
        int local_m = kTileM == 128 ? consumer_thread % 64 : consumer_thread;
#else
        int local_m = int(threadIdx.x) - 128;
#endif
#endif
#if defined(FMMS_SM100_2SM)
        int global_m = tile_m * (kTileM / 2) + local_m;
#else
        int global_m = tile_m * kTileM + local_m;
#endif
        if (global_m >= problem_m) {
          return pack_candidate(-INFINITY, INT_MAX);
        }
#if defined(FMMS_GUMBEL)
        int global_n = global_n_for<FragmentSize>(i, epi_n);
        int hidden_idx = global_n % params.original_n;
        int sample_idx = params.sample_base + global_n / params.original_n;
        float gumbel = gumbel_noise(
            params.seed, uint32_t(sample_idx), uint32_t(hidden_idx),
            uint64_t(global_m));
#if defined(FMMS_FAST_DIV)
        float scaled = __fdividef(float(accumulator), *params.temperature);
#else
        float scaled = float(accumulator) / *params.temperature;
#endif
        float value = scaled + gumbel;
        return pack_candidate(value, global_m);
#else
        return pack_candidate(float(accumulator), global_m);
#endif
    }

    template <int FragmentSize>
    CUTLASS_DEVICE int global_n_for(int i, int epi_n) const {
#if defined(FMMS_SM100_2SM)
      int consumer_thread = int(threadIdx.x) - 128;
      int local_n;
      if constexpr (kTileM == 128) {
        local_n =
            2 * FragmentSize * (consumer_thread / 64) +
            FragmentSize * epi_n + i;
      } else {
        local_n = FragmentSize * epi_n + i;
      }
#else
      int local_n = 16 * epi_n + i;
#endif
#if defined(FMMS_GUMBEL)
      return tile_n * kTileN + local_n;
#else
      return local_n;
#endif
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const& args) {
    return ConsumerStoreCallbacks{
        {}, int(get<0>(args.tile_coord_mnkl)),
        int(get<0>(args.problem_shape_mnkl))
#if defined(FMMS_GUMBEL)
        , int(get<1>(args.tile_coord_mnkl)), params
#endif
    };
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
#if defined(FMMS_SM100_2SM) && defined(FMMS_PER_CTA_CANDIDATES)
using ReductionTileShape =
    Shape<Int<FMMS_TILE_M / 2>, Int<FMMS_TILE_N>, Int<FMMS_TILE_K>>;
#else
using ReductionTileShape = TileShape;
#endif
#if defined(FMMS_SM100_2SM)
using ClusterShape = Shape<Int<FMMS_CLUSTER_M>, _1, _1>;
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
    ReductionTileShape,
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

struct WarpGroupOutput {
  using Arguments = typename CandidateReductionEVT::Arguments;
  struct SharedStorage {
    alignas(16) PackedCandidate warp_candidates[4 * 16];
  };
  struct Params {
    PackedCandidate* output;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& arguments, void*) {
    return {reinterpret_cast<PackedCandidate*>(arguments.op_1.ptr_row)};
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
  CUTLASS_HOST_DEVICE WarpGroupOutput() : params{}, warp_candidates(nullptr) {}
  CUTLASS_HOST_DEVICE WarpGroupOutput(
      Params const& params_,
      SharedStorage const& storage)
      : params(params_),
        warp_candidates(
            const_cast<PackedCandidate*>(storage.warp_candidates)) {}

  Params params;
  PackedCandidate* warp_candidates;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    PackedCandidate* output;
    PackedCandidate* warp_candidates;
    int tile_m;
    int rounded_n;
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const& args) {
    int problem_n = int(get<1>(args.problem_shape_mnkl));
    int rounded_n = (problem_n + kTileN - 1) / kTileN * kTileN;
    return ConsumerStoreCallbacks{
        {}, params.output, warp_candidates,
        int(get<0>(args.tile_coord_mnkl)), rounded_n};
  }
};

struct StagedWarpGroupOutput {
  using Arguments = typename CandidateReductionEVT::Arguments;
  struct SharedStorage {
    alignas(16) PackedCandidate warp_candidates[4 * 16];
    alignas(16) float accumulator_fragments[128 * 16];
  };
  struct Params {
    PackedCandidate* output;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& arguments, void*) {
    return {reinterpret_cast<PackedCandidate*>(arguments.op_1.ptr_row)};
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
  CUTLASS_HOST_DEVICE StagedWarpGroupOutput()
      : params{}, warp_candidates(nullptr), accumulator_fragments(nullptr) {}
  CUTLASS_HOST_DEVICE StagedWarpGroupOutput(
      Params const& params_,
      SharedStorage const& storage)
      : params(params_),
        warp_candidates(
            const_cast<PackedCandidate*>(storage.warp_candidates)),
        accumulator_fragments(
            const_cast<float*>(storage.accumulator_fragments)) {}

  Params params;
  PackedCandidate* warp_candidates;
  float* accumulator_fragments;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(
      cutlass::epilogue::fusion::detail::ProducerLoadArgs<Args...> const&) {
    return cutlass::epilogue::fusion::EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks
      : cutlass::epilogue::fusion::EmptyConsumerStoreCallbacks {
    PackedCandidate* output;
    PackedCandidate* warp_candidates;
    float* accumulator_fragments;
    int tile_m;
    int rounded_n;
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const& args) {
    int problem_n = int(get<1>(args.problem_shape_mnkl));
    int rounded_n = (problem_n + kTileN - 1) / kTileN * kTileN;
    return ConsumerStoreCallbacks{
        {}, params.output, warp_candidates, accumulator_fragments,
        int(get<0>(args.tile_coord_mnkl)), rounded_n};
  }
};

struct PackedCandidateEVT
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

template <class Output, bool StageAccumulators>
struct WarpGroupCandidateEVTImpl
    : cutlass::epilogue::fusion::detail::Sm90VisitorImpl<
          PackCandidate,
          Output,
          DiscardPackedOutput> {
  using Impl = cutlass::epilogue::fusion::detail::Sm90VisitorImpl<
      PackCandidate,
      Output,
      DiscardPackedOutput>;
  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;
  using Impl::Impl;
  using ElementAux = float;

  template <class CallbacksImpl>
  struct ConsumerStoreCallbacks : CallbacksImpl {
    CUTLASS_DEVICE ConsumerStoreCallbacks(CallbacksImpl&& impl)
        : CallbacksImpl(cute::forward<CallbacksImpl>(impl)) {}

    using CallbacksImpl::callbacks_tuple;

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const& accumulators,
        int epi_v,
        int epi_m,
        int epi_n) {
      static_assert(FragmentSize == 16);
      auto& pack_callbacks = get<0>(callbacks_tuple);
      auto& output_callbacks = get<1>(callbacks_tuple);
      int consumer_thread = int(threadIdx.x) - 128;
      int warp = consumer_thread / 32;
      int lane = consumer_thread % 32;
      if constexpr (StageAccumulators) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          output_callbacks.accumulator_fragments[
              consumer_thread * FragmentSize + i] = float(accumulators[i]);
        }
      }
#if defined(FMMS_GUMBEL_PARTIAL_UNROLL)
#if FMMS_GUMBEL_PARTIAL_UNROLL == 2
#pragma unroll 2
#elif FMMS_GUMBEL_PARTIAL_UNROLL == 4
#pragma unroll 4
#elif FMMS_GUMBEL_PARTIAL_UNROLL == 8
#pragma unroll 8
#else
#error "Unsupported FMMS_GUMBEL_PARTIAL_UNROLL value"
#endif
#else
      CUTLASS_PRAGMA_UNROLL
#endif
      for (int i = 0; i < FragmentSize; ++i) {
        ElementAccumulator accumulator;
        if constexpr (StageAccumulators) {
          accumulator = ElementAccumulator(
              output_callbacks.accumulator_fragments[
                  consumer_thread * FragmentSize + i]);
        }
        else {
          accumulator = accumulators[i];
        }
        PackedCandidate candidate =
            pack_callbacks.template visit_one<ElementAccumulator, FragmentSize>(
                accumulator, i, epi_v, epi_m, epi_n);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset > 0; offset /= 2) {
          uint32_t rhs_value_bits = __shfl_down_sync(
              0xFFFFFFFF, uint32_t(candidate >> 32), offset);
          uint32_t rhs_index = __shfl_down_sync(
              0xFFFFFFFF, uint32_t(candidate), offset);
          PackedCandidate rhs =
              (uint64_t(rhs_value_bits) << 32) | rhs_index;
          candidate = choose_candidate(candidate, rhs);
        }
        if (lane == 0) {
          output_callbacks.warp_candidates[warp * FragmentSize + i] =
              candidate;
        }
      }
      cutlass::arch::NamedBarrier::sync(128, 0);
      if (warp == 0 && lane < FragmentSize) {
        PackedCandidate candidate =
            output_callbacks.warp_candidates[lane];
        CUTLASS_PRAGMA_UNROLL
        for (int source_warp = 1; source_warp < 4; ++source_warp) {
          candidate = choose_candidate(
              candidate,
              output_callbacks.warp_candidates[
                  source_warp * FragmentSize + lane]);
        }
        int global_n =
            pack_callbacks.template global_n_for<FragmentSize>(lane, epi_n);
        output_callbacks.output[
            output_callbacks.tile_m * output_callbacks.rounded_n + global_n] =
                candidate;
      }
      cutlass::arch::NamedBarrier::sync(128, 0);
      return get<2>(callbacks_tuple).visit(
          accumulators, epi_v, epi_m, epi_n);
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(
      cutlass::epilogue::fusion::detail::ConsumerStoreArgs<Args...> const& args) {
    auto callbacks_impl =
        Impl::template get_consumer_store_callbacks<ReferenceSrc>(args);
    return ConsumerStoreCallbacks<decltype(callbacks_impl)>(
        cute::move(callbacks_impl));
  }
};

using WarpGroupCandidateEVT =
    WarpGroupCandidateEVTImpl<WarpGroupOutput, false>;
using StagedWarpGroupCandidateEVT =
    WarpGroupCandidateEVTImpl<StagedWarpGroupOutput, true>;

#if defined(FMMS_USE_WARPGROUP_REDUCTION) && \
    defined(FMMS_WARPGROUP_SMEM_STAGE)
using CandidateEVT = StagedWarpGroupCandidateEVT;
#elif defined(FMMS_USE_WARPGROUP_REDUCTION)
using CandidateEVT = WarpGroupCandidateEVT;
#else
using CandidateEVT = PackedCandidateEVT;
#endif

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
#if defined(FMMS_SM100_2SM) && defined(FMMS_PER_CTA_CANDIDATES)
  int m_tiles = (test_case.m + kTileM / 2 - 1) / (kTileM / 2);
#else
  int m_tiles = (test_case.m + kTileM - 1) / kTileM;
#endif
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
#if defined(FMMS_FINAL_REDUCTION) || defined(FMMS_PER_CTA_CANDIDATES)
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
#if defined(FMMS_SM100_2SM) && defined(FMMS_PER_CTA_CANDIDATES)
  constexpr int kCandidateTileM = kTileM / 2;
#else
  constexpr int kCandidateTileM = kTileM;
#endif
  for (int tile = 0; tile < m_tiles; ++tile) {
    int begin = tile * kCandidateTileM;
    int end = std::min(begin + kCandidateTileM, test_case.m);
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
