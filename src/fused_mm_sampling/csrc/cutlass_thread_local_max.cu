// Validates deterministic max-with-index over one thread-owned FP32 fragment.

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass_max_with_index.cuh"

#if !defined(FMMS_ARCH_SM90) && !defined(FMMS_ARCH_SM100)
#error "Compile with FMMS_ARCH_SM90 or FMMS_ARCH_SM100"
#endif

namespace fmms_thread_local_max {

using fmms_cutlass::MaxWithIndex;
using fmms_cutlass::choose_max;
using fmms_cutlass::float_bits;

constexpr int kConsumerThreads = 128;
constexpr int kFragmentSize = 16;

enum class VisitOrder : int {
  kAscending,
  kDescending,
};

__device__ MaxWithIndex reduce_thread_fragment(
    float const* values,
    int const* indices,
    VisitOrder order) {
  int first = order == VisitOrder::kAscending ? 0 : kFragmentSize - 1;
  MaxWithIndex result{values[first], indices[first]};
  for (int step = 1; step < kFragmentSize; ++step) {
    int slot =
        order == VisitOrder::kAscending ? step : kFragmentSize - 1 - step;
    result = choose_max(result, {values[slot], indices[slot]});
  }
  return result;
}

__global__ void reduce_fragments(
    float const* values,
    int const* indices,
    MaxWithIndex* results,
    int case_count,
    VisitOrder order) {
  int consumer_thread = int(threadIdx.x);
  if (consumer_thread >= kConsumerThreads) {
    return;
  }
  for (int test_case = 0; test_case < case_count; ++test_case) {
    int offset =
        (test_case * kConsumerThreads + consumer_thread) * kFragmentSize;
    results[test_case * kConsumerThreads + consumer_thread] =
        reduce_thread_fragment(values + offset, indices + offset, order);
  }
}

void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

struct TestCase {
  char const* name;
  std::vector<float> values;
  std::vector<int> indices;
};

std::vector<TestCase> make_cases() {
  std::vector<TestCase> cases;
  for (int winning_slot = 0; winning_slot < kFragmentSize; ++winning_slot) {
    std::vector<float> values(kFragmentSize);
    std::vector<int> indices(kFragmentSize);
    for (int slot = 0; slot < kFragmentSize; ++slot) {
      values[slot] = float(slot - kFragmentSize);
      indices[slot] = 1000 + slot;
    }
    values[winning_slot] = 32.0f;
    static char names[kFragmentSize][32];
    std::snprintf(
        names[winning_slot],
        sizeof(names[winning_slot]),
        "maximum_in_slot_%02d",
        winning_slot);
    cases.push_back({names[winning_slot], values, indices});
  }

  std::vector<float> negative_values(kFragmentSize);
  std::vector<int> ordered_indices(kFragmentSize);
  for (int slot = 0; slot < kFragmentSize; ++slot) {
    negative_values[slot] = -float(slot + 2);
    ordered_indices[slot] = 2000 + slot;
  }
  negative_values[11] = -0.5f;
  cases.push_back({"all_negative", negative_values, ordered_indices});

  std::vector<float> tie_values(kFragmentSize, -4.0f);
  tie_values[3] = 7.25f;
  tie_values[12] = 7.25f;
  std::vector<int> tie_low_index_first = ordered_indices;
  tie_low_index_first[3] = 17;
  tie_low_index_first[12] = 91;
  cases.push_back(
      {"tie_low_index_in_earlier_slot", tie_values, tie_low_index_first});

  std::vector<int> tie_low_index_last = ordered_indices;
  tie_low_index_last[3] = 91;
  tie_low_index_last[12] = 17;
  cases.push_back(
      {"tie_low_index_in_later_slot", tie_values, tie_low_index_last});
  return cases;
}

MaxWithIndex host_reference(TestCase const& test_case) {
  MaxWithIndex result{test_case.values[0], test_case.indices[0]};
  for (int slot = 1; slot < kFragmentSize; ++slot) {
    result = choose_max(
        result,
        {test_case.values[slot], test_case.indices[slot]});
  }
  return result;
}

void run_tests() {
  std::vector<TestCase> cases = make_cases();
  size_t value_count = cases.size() * kConsumerThreads * kFragmentSize;
  std::vector<float> host_values(value_count);
  std::vector<int> host_indices(value_count);
  for (size_t test_case = 0; test_case < cases.size(); ++test_case) {
    for (int thread = 0; thread < kConsumerThreads; ++thread) {
      size_t offset =
          (test_case * kConsumerThreads + thread) * kFragmentSize;
      for (int slot = 0; slot < kFragmentSize; ++slot) {
        host_values[offset + slot] = cases[test_case].values[slot];
        host_indices[offset + slot] = cases[test_case].indices[slot];
      }
    }
  }

  float* device_values = nullptr;
  int* device_indices = nullptr;
  MaxWithIndex* device_results = nullptr;
  size_t result_count = cases.size() * kConsumerThreads;
  check_cuda(
      cudaMalloc(&device_values, value_count * sizeof(float)),
      "cudaMalloc(values)");
  check_cuda(
      cudaMalloc(&device_indices, value_count * sizeof(int)),
      "cudaMalloc(indices)");
  check_cuda(
      cudaMalloc(&device_results, result_count * sizeof(MaxWithIndex)),
      "cudaMalloc(results)");
  check_cuda(
      cudaMemcpy(
          device_values,
          host_values.data(),
          value_count * sizeof(float),
          cudaMemcpyHostToDevice),
      "copy values");
  check_cuda(
      cudaMemcpy(
          device_indices,
          host_indices.data(),
          value_count * sizeof(int),
          cudaMemcpyHostToDevice),
      "copy indices");

  std::cout
      << "architecture,case,visit_order,thread,expected_value_bits,"
         "actual_value_bits,expected_index,actual_index,pass\n";
  for (VisitOrder order :
       {VisitOrder::kAscending, VisitOrder::kDescending}) {
    reduce_fragments<<<1, kConsumerThreads>>>(
        device_values,
        device_indices,
        device_results,
        int(cases.size()),
        order);
    check_cuda(cudaGetLastError(), "launch");
    check_cuda(cudaDeviceSynchronize(), "synchronize");

    std::vector<MaxWithIndex> results(result_count);
    check_cuda(
        cudaMemcpy(
            results.data(),
            device_results,
            result_count * sizeof(MaxWithIndex),
            cudaMemcpyDeviceToHost),
        "copy results");
    char const* order_name =
        order == VisitOrder::kAscending ? "ascending" : "descending";
#if defined(FMMS_ARCH_SM90)
    char const* architecture = "sm90";
#else
    char const* architecture = "sm100";
#endif
    for (size_t test_case = 0; test_case < cases.size(); ++test_case) {
      MaxWithIndex expected = host_reference(cases[test_case]);
      for (int thread = 0; thread < kConsumerThreads; ++thread) {
        MaxWithIndex actual =
            results[test_case * kConsumerThreads + thread];
        bool passed =
            float_bits(actual.value) == float_bits(expected.value) &&
            actual.index == expected.index;
        std::cout << architecture << ',' << cases[test_case].name << ','
                  << order_name << ',' << thread + 128 << ','
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

}  // namespace fmms_thread_local_max

int main() {
  fmms_thread_local_max::run_tests();
  return 0;
}
