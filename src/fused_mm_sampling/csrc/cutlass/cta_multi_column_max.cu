// Validates independent CTA-local max-with-index reduction for 128 N columns.

#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "max_with_index.cuh"

#if !defined(FMMS_ARCH_SM90) && !defined(FMMS_ARCH_SM100)
#error "Compile with FMMS_ARCH_SM90 or FMMS_ARCH_SM100"
#endif

namespace fmms_cta_multi_column_max {

using fmms_cutlass::MaxWithIndex;
using fmms_cutlass::choose_max;
using fmms_cutlass::float_bits;
using fmms_cutlass::reduce_warp_xor;

constexpr int kM = 128;
constexpr int kN = 128;
constexpr int kWarps = 4;
constexpr int kWarpSize = 32;
constexpr int kThreads = kWarps * kWarpSize;
#if defined(FMMS_ARCH_SM90)
constexpr int kEpilogueNWidth = 32;
constexpr unsigned kCtaParticipantMask = 0x11111111u;
constexpr int kCtaLaneStride = 4;
constexpr char const* kArchitecture = "sm90";
#else
constexpr int kEpilogueNWidth = 16;
constexpr unsigned kCtaParticipantMask = 0xffffffffu;
constexpr int kCtaLaneStride = 1;
constexpr char const* kArchitecture = "sm100";
#endif

__device__ MaxWithIndex thread_local_candidate(
    float const* values,
    int test_case,
    int column,
    int warp,
    int lane) {
#if defined(FMMS_ARCH_SM90)
  int column_lane_group = (column % 8) / 2;
  if (lane % 4 != column_lane_group) {
    return {-INFINITY, INT_MAX};
  }
  int base_m = warp * 16 + lane / 4;
  int positions[4] = {base_m, base_m + 8, base_m + 64, base_m + 72};
  MaxWithIndex local{
      values[(test_case * kM + positions[0]) * kN + column],
      positions[0]};
  for (int position = 1; position < 4; ++position) {
    int m = positions[position];
    local = choose_max(
        local,
        {values[(test_case * kM + m) * kN + column], m});
  }
  return local;
#else
  int m = warp * kWarpSize + lane;
  return {values[(test_case * kM + m) * kN + column], m};
#endif
}

__global__ void reduce_cta_columns(
    float const* values,
    MaxWithIndex* results,
    int case_count) {
  __shared__ MaxWithIndex warp_results[kWarps][kN];
  int lane = int(threadIdx.x) % kWarpSize;
  int warp = int(threadIdx.x) / kWarpSize;

  for (int test_case = 0; test_case < case_count; ++test_case) {
    for (int column = 0; column < kN; ++column) {
      MaxWithIndex local =
          thread_local_candidate(values, test_case, column, warp, lane);
#if defined(FMMS_ARCH_SM90)
      int column_lane_group = (column % 8) / 2;
      unsigned column_mask = 0x11111111u << column_lane_group;
      if (lane % 4 == column_lane_group) {
        local = reduce_warp_xor(local, column_mask, 4);
        if (lane == column_lane_group) {
          warp_results[warp][column] = local;
        }
      }
#else
      local = reduce_warp_xor(local, 0xffffffffu, 1);
      if (lane == 0) {
        warp_results[warp][column] = local;
      }
#endif
    }
    __syncthreads();

    if (warp == 0) {
      int participant = lane / kCtaLaneStride;
      bool participates =
          (kCtaParticipantMask & (1u << lane)) != 0;
      for (int column = 0; column < kN; ++column) {
        MaxWithIndex local =
            participates && participant < kWarps
            ? warp_results[participant][column]
            : MaxWithIndex{-INFINITY, INT_MAX};
        if (participates) {
          local = reduce_warp_xor(
              local, kCtaParticipantMask, kCtaLaneStride);
          if (lane == 0) {
            results[test_case * kN + column] = local;
          }
        }
      }
    }
    __syncthreads();
  }
}

void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

int winner_m(int test_case, int column) {
  return test_case == 0
      ? (37 * column + 11) % kM
      : (53 * column + 7) % kM;
}

float winner_value(int test_case, int column) {
  return test_case == 0
      ? 1000.0f + float(column)
      : -0.25f - float(column) / 256.0f;
}

std::vector<float> make_values(int case_count) {
  std::vector<float> values(case_count * kM * kN);
  for (int test_case = 0; test_case < case_count; ++test_case) {
    for (int m = 0; m < kM; ++m) {
      for (int column = 0; column < kN; ++column) {
        values[(test_case * kM + m) * kN + column] =
            -10000.0f - float(m);
      }
    }
    for (int column = 0; column < kN; ++column) {
      int m = winner_m(test_case, column);
      values[(test_case * kM + m) * kN + column] =
          winner_value(test_case, column);
    }
  }
  return values;
}

MaxWithIndex host_reference(
    std::vector<float> const& values,
    int test_case,
    int column) {
  MaxWithIndex result{
      values[(test_case * kM) * kN + column],
      0};
  for (int m = 1; m < kM; ++m) {
    result = choose_max(
        result,
        {values[(test_case * kM + m) * kN + column], m});
  }
  return result;
}

char const* boundary_name(int column) {
  int within_iteration = column % kEpilogueNWidth;
  if (within_iteration == 0) {
    return "start";
  }
  if (within_iteration == kEpilogueNWidth - 1) {
    return "end";
  }
  return "interior";
}

void run_tests() {
  char const* case_names[] = {"independent_unique", "all_negative"};
  constexpr int kCaseCount = sizeof(case_names) / sizeof(case_names[0]);
  std::vector<float> host_values = make_values(kCaseCount);
  float* device_values = nullptr;
  MaxWithIndex* device_results = nullptr;
  check_cuda(
      cudaMalloc(&device_values, host_values.size() * sizeof(float)),
      "cudaMalloc(values)");
  check_cuda(
      cudaMalloc(&device_results, kCaseCount * kN * sizeof(MaxWithIndex)),
      "cudaMalloc(results)");
  check_cuda(
      cudaMemcpy(
          device_values,
          host_values.data(),
          host_values.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      "copy values");

  reduce_cta_columns<<<1, kThreads>>>(
      device_values, device_results, kCaseCount);
  check_cuda(cudaGetLastError(), "launch");
  check_cuda(cudaDeviceSynchronize(), "synchronize");
  std::vector<MaxWithIndex> results(kCaseCount * kN);
  check_cuda(
      cudaMemcpy(
          results.data(),
          device_results,
          results.size() * sizeof(MaxWithIndex),
          cudaMemcpyDeviceToHost),
      "copy results");

  std::cout
      << "architecture,case,column,epi_n,boundary,expected_value_bits,"
         "actual_value_bits,expected_index,actual_index,pass\n";
  for (int test_case = 0; test_case < kCaseCount; ++test_case) {
    for (int column = 0; column < kN; ++column) {
      MaxWithIndex expected =
          host_reference(host_values, test_case, column);
      MaxWithIndex actual = results[test_case * kN + column];
      bool passed =
          float_bits(expected.value) == float_bits(actual.value) &&
          expected.index == actual.index;
      std::cout << kArchitecture << ',' << case_names[test_case] << ','
                << column << ',' << column / kEpilogueNWidth << ','
                << boundary_name(column) << ','
                << float_bits(expected.value) << ','
                << float_bits(actual.value) << ',' << expected.index << ','
                << actual.index << ',' << (passed ? 1 : 0) << '\n';
    }
  }

  check_cuda(cudaFree(device_results), "cudaFree(results)");
  check_cuda(cudaFree(device_values), "cudaFree(values)");
}

}  // namespace fmms_cta_multi_column_max

int main() {
  fmms_cta_multi_column_max::run_tests();
  return 0;
}
