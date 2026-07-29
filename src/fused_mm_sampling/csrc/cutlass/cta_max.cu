// Models shared-memory max-with-index reduction across four CUTLASS consumer roles.

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "max_with_index.cuh"

#if !defined(FMMS_ARCH_SM90) && !defined(FMMS_ARCH_SM100)
#error "Compile with FMMS_ARCH_SM90 or FMMS_ARCH_SM100"
#endif

namespace fmms_cta_max {

using fmms_cutlass::MaxWithIndex;
using fmms_cutlass::choose_max;
using fmms_cutlass::float_bits;

constexpr int kWarps = 4;
constexpr int kWarpSize = 32;
#if defined(FMMS_ARCH_SM90)
constexpr int kParticipantCount = 8;
constexpr unsigned kParticipantMask = 0x11111111u;
constexpr int kLaneStride = 4;
constexpr char const* kArchitecture = "sm90";
#else
constexpr int kParticipantCount = 32;
constexpr unsigned kParticipantMask = 0xffffffffu;
constexpr int kLaneStride = 1;
constexpr char const* kArchitecture = "sm100";
#endif

struct TestCase {
  char name[48];
  std::vector<float> values;
  std::vector<int> indices;
};

__device__ MaxWithIndex reduce_warp_m_lanes(MaxWithIndex local) {
  for (int offset = kLaneStride; offset < kWarpSize; offset *= 2) {
    MaxWithIndex peer{
        __shfl_xor_sync(kParticipantMask, local.value, offset),
        __shfl_xor_sync(kParticipantMask, local.index, offset)};
    local = choose_max(local, peer);
  }
  return local;
}

__global__ void reduce_cta(
    float const* values,
    int const* indices,
    MaxWithIndex* results,
    int case_count) {
  __shared__ MaxWithIndex warp_results[kWarps];
  int lane = int(threadIdx.x) % kWarpSize;
  int warp = int(threadIdx.x) / kWarpSize;
  bool participates = (kParticipantMask & (1u << lane)) != 0;
  int participant = lane / kLaneStride;

  for (int test_case = 0; test_case < case_count; ++test_case) {
    MaxWithIndex local{-INFINITY, INT_MAX};
    if (participates) {
      int input_offset =
          (test_case * kWarps + warp) * kParticipantCount + participant;
      local = {values[input_offset], indices[input_offset]};
      local = reduce_warp_m_lanes(local);
      if (lane == 0) {
        warp_results[warp] = local;
      }
    }
    __syncthreads();

    if (warp == 0 && participates) {
      local = participant < kWarps
          ? warp_results[participant]
          : MaxWithIndex{-INFINITY, INT_MAX};
      local = reduce_warp_m_lanes(local);
      if (lane == 0) {
        results[test_case] = local;
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

TestCase base_case(char const* name) {
  TestCase test_case;
  std::snprintf(test_case.name, sizeof(test_case.name), "%s", name);
  int count = kWarps * kParticipantCount;
  test_case.values.resize(count);
  test_case.indices.resize(count);
  for (int position = 0; position < count; ++position) {
    test_case.values[position] = -float(position + 2);
    test_case.indices[position] = 1000 + position;
  }
  return test_case;
}

std::vector<TestCase> make_cases() {
  std::vector<TestCase> cases;
  for (int winner_warp = 0; winner_warp < kWarps; ++winner_warp) {
    TestCase test_case = base_case("unused");
    std::snprintf(
        test_case.name,
        sizeof(test_case.name),
        "maximum_in_warp_%d",
        winner_warp + 4);
    int position = winner_warp * kParticipantCount + (winner_warp + 1);
    test_case.values[position] = 32.0f;
    cases.push_back(test_case);
  }

  TestCase all_negative = base_case("all_negative");
  all_negative.values.back() = -0.5f;
  cases.push_back(all_negative);

  TestCase tie_earlier = base_case("tie_low_index_in_earlier_warp");
  int earlier = 1;
  int later = (kWarps - 1) * kParticipantCount + 2;
  tie_earlier.values[earlier] = 7.25f;
  tie_earlier.values[later] = 7.25f;
  tie_earlier.indices[earlier] = 17;
  tie_earlier.indices[later] = 91;
  cases.push_back(tie_earlier);

  TestCase tie_later = tie_earlier;
  std::snprintf(
      tie_later.name,
      sizeof(tie_later.name),
      "tie_low_index_in_later_warp");
  tie_later.indices[earlier] = 91;
  tie_later.indices[later] = 17;
  cases.push_back(tie_later);
  return cases;
}

MaxWithIndex host_reference(TestCase const& test_case) {
  MaxWithIndex result{test_case.values[0], test_case.indices[0]};
  for (size_t position = 1; position < test_case.values.size(); ++position) {
    result = choose_max(
        result,
        {test_case.values[position], test_case.indices[position]});
  }
  return result;
}

void run_tests() {
  std::vector<TestCase> cases = make_cases();
  size_t input_count = cases.size() * kWarps * kParticipantCount;
  std::vector<float> host_values;
  std::vector<int> host_indices;
  host_values.reserve(input_count);
  host_indices.reserve(input_count);
  for (TestCase const& test_case : cases) {
    host_values.insert(
        host_values.end(), test_case.values.begin(), test_case.values.end());
    host_indices.insert(
        host_indices.end(), test_case.indices.begin(), test_case.indices.end());
  }

  float* device_values = nullptr;
  int* device_indices = nullptr;
  MaxWithIndex* device_results = nullptr;
  check_cuda(
      cudaMalloc(&device_values, input_count * sizeof(float)),
      "cudaMalloc(values)");
  check_cuda(
      cudaMalloc(&device_indices, input_count * sizeof(int)),
      "cudaMalloc(indices)");
  check_cuda(
      cudaMalloc(&device_results, cases.size() * sizeof(MaxWithIndex)),
      "cudaMalloc(results)");
  check_cuda(
      cudaMemcpy(
          device_values,
          host_values.data(),
          input_count * sizeof(float),
          cudaMemcpyHostToDevice),
      "copy values");
  check_cuda(
      cudaMemcpy(
          device_indices,
          host_indices.data(),
          input_count * sizeof(int),
          cudaMemcpyHostToDevice),
      "copy indices");

  reduce_cta<<<1, kWarps * kWarpSize>>>(
      device_values,
      device_indices,
      device_results,
      int(cases.size()));
  check_cuda(cudaGetLastError(), "launch");
  check_cuda(cudaDeviceSynchronize(), "synchronize");
  std::vector<MaxWithIndex> results(cases.size());
  check_cuda(
      cudaMemcpy(
          results.data(),
          device_results,
          results.size() * sizeof(MaxWithIndex),
          cudaMemcpyDeviceToHost),
      "copy results");

  std::cout
      << "architecture,case,expected_value_bits,actual_value_bits,"
         "expected_index,actual_index,pass\n";
  for (size_t test_case = 0; test_case < cases.size(); ++test_case) {
    MaxWithIndex expected = host_reference(cases[test_case]);
    MaxWithIndex actual = results[test_case];
    bool passed =
        float_bits(expected.value) == float_bits(actual.value) &&
        expected.index == actual.index;
    std::cout << kArchitecture << ',' << cases[test_case].name << ','
              << float_bits(expected.value) << ','
              << float_bits(actual.value) << ',' << expected.index << ','
              << actual.index << ',' << (passed ? 1 : 0) << '\n';
  }

  check_cuda(cudaFree(device_results), "cudaFree(results)");
  check_cuda(cudaFree(device_indices), "cudaFree(indices)");
  check_cuda(cudaFree(device_values), "cudaFree(values)");
}

}  // namespace fmms_cta_max

int main() {
  fmms_cta_max::run_tests();
  return 0;
}
