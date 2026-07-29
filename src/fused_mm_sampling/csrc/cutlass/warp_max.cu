// Validates shuffle-based max-with-index over architecture-specific M lanes.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "max_with_index.cuh"

#if !defined(FMMS_ARCH_SM90) && !defined(FMMS_ARCH_SM100)
#error "Compile with FMMS_ARCH_SM90 or FMMS_ARCH_SM100"
#endif

namespace fmms_warp_max {

using fmms_cutlass::MaxWithIndex;
using fmms_cutlass::choose_max;
using fmms_cutlass::float_bits;
using fmms_cutlass::reduce_warp_xor;

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

__global__ void reduce_warps(
    float const* values,
    int const* indices,
    MaxWithIndex* results,
    int case_count) {
  int lane = int(threadIdx.x) % kWarpSize;
  int warp = int(threadIdx.x) / kWarpSize;
  if ((kParticipantMask & (1u << lane)) == 0) {
    return;
  }
  int participant = lane / kLaneStride;
  for (int test_case = 0; test_case < case_count; ++test_case) {
    int input_offset =
        (test_case * kWarps + warp) * kParticipantCount + participant;
    MaxWithIndex winner = reduce_warp_xor(
        {values[input_offset], indices[input_offset]},
        kParticipantMask,
        kLaneStride);
    int output_offset =
        (test_case * kWarps + warp) * kParticipantCount + participant;
    results[output_offset] = winner;
  }
}

void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

std::vector<TestCase> make_cases() {
  std::vector<TestCase> cases;
  for (int winner = 0; winner < kParticipantCount; ++winner) {
    TestCase test_case;
    std::snprintf(
        test_case.name,
        sizeof(test_case.name),
        "maximum_in_lane_%02d",
        winner * kLaneStride);
    test_case.values.resize(kParticipantCount);
    test_case.indices.resize(kParticipantCount);
    for (int participant = 0; participant < kParticipantCount; ++participant) {
      test_case.values[participant] = float(participant - kParticipantCount);
      test_case.indices[participant] = 1000 + participant;
    }
    test_case.values[winner] = 32.0f;
    cases.push_back(test_case);
  }

  TestCase all_negative;
  std::snprintf(
      all_negative.name,
      sizeof(all_negative.name),
      "all_negative");
  all_negative.values.resize(kParticipantCount);
  all_negative.indices.resize(kParticipantCount);
  for (int participant = 0; participant < kParticipantCount; ++participant) {
    all_negative.values[participant] = -float(participant + 2);
    all_negative.indices[participant] = 2000 + participant;
  }
  all_negative.values[kParticipantCount - 1] = -0.5f;
  cases.push_back(all_negative);

  TestCase tie_earlier;
  std::snprintf(
      tie_earlier.name,
      sizeof(tie_earlier.name),
      "tie_low_index_in_earlier_lane");
  tie_earlier.values.assign(kParticipantCount, -4.0f);
  tie_earlier.indices.resize(kParticipantCount);
  for (int participant = 0; participant < kParticipantCount; ++participant) {
    tie_earlier.indices[participant] = 3000 + participant;
  }
  tie_earlier.values[0] = 7.25f;
  tie_earlier.values[kParticipantCount - 1] = 7.25f;
  tie_earlier.indices[0] = 17;
  tie_earlier.indices[kParticipantCount - 1] = 91;
  cases.push_back(tie_earlier);

  TestCase tie_later = tie_earlier;
  std::snprintf(
      tie_later.name,
      sizeof(tie_later.name),
      "tie_low_index_in_later_lane");
  tie_later.indices[0] = 91;
  tie_later.indices[kParticipantCount - 1] = 17;
  cases.push_back(tie_later);
  return cases;
}

MaxWithIndex host_reference(TestCase const& test_case) {
  MaxWithIndex result{test_case.values[0], test_case.indices[0]};
  for (int participant = 1; participant < kParticipantCount; ++participant) {
    result = choose_max(
        result,
        {test_case.values[participant], test_case.indices[participant]});
  }
  return result;
}

void run_tests() {
  std::vector<TestCase> cases = make_cases();
  size_t input_count = cases.size() * kWarps * kParticipantCount;
  std::vector<float> host_values(input_count);
  std::vector<int> host_indices(input_count);
  for (size_t test_case = 0; test_case < cases.size(); ++test_case) {
    for (int warp = 0; warp < kWarps; ++warp) {
      size_t offset =
          (test_case * kWarps + warp) * kParticipantCount;
      for (int participant = 0; participant < kParticipantCount; ++participant) {
        host_values[offset + participant] = cases[test_case].values[participant];
        host_indices[offset + participant] =
            cases[test_case].indices[participant] + warp * 100;
      }
    }
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
      cudaMalloc(&device_results, input_count * sizeof(MaxWithIndex)),
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

  reduce_warps<<<1, kWarps * kWarpSize>>>(
      device_values,
      device_indices,
      device_results,
      int(cases.size()));
  check_cuda(cudaGetLastError(), "launch");
  check_cuda(cudaDeviceSynchronize(), "synchronize");
  std::vector<MaxWithIndex> results(input_count);
  check_cuda(
      cudaMemcpy(
          results.data(),
          device_results,
          input_count * sizeof(MaxWithIndex),
          cudaMemcpyDeviceToHost),
      "copy results");

  std::cout
      << "architecture,case,warp,output_lane,expected_value_bits,"
         "actual_value_bits,expected_index,actual_index,pass\n";
  for (size_t test_case = 0; test_case < cases.size(); ++test_case) {
    MaxWithIndex base_expected = host_reference(cases[test_case]);
    for (int warp = 0; warp < kWarps; ++warp) {
      MaxWithIndex expected{
          base_expected.value,
          base_expected.index + warp * 100};
      size_t offset =
          (test_case * kWarps + warp) * kParticipantCount;
      for (int participant = 0; participant < kParticipantCount; ++participant) {
        MaxWithIndex actual = results[offset + participant];
        bool passed =
            float_bits(expected.value) == float_bits(actual.value) &&
            expected.index == actual.index;
        std::cout << kArchitecture << ',' << cases[test_case].name << ','
                  << warp + 4 << ',' << participant * kLaneStride << ','
                  << float_bits(expected.value) << ','
                  << float_bits(actual.value) << ',' << expected.index << ','
                  << actual.index << ',' << (passed ? 1 : 0) << '\n';
      }
    }
  }

  check_cuda(cudaFree(device_results), "cudaFree(results)");
  check_cuda(cudaFree(device_indices), "cudaFree(indices)");
  check_cuda(cudaFree(device_values), "cudaFree(values)");
}

}  // namespace fmms_warp_max

int main() {
  fmms_warp_max::run_tests();
  return 0;
}
