// Validates predicated CTA-local max-with-index reduction on boundary tiles.

#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "max_harness.h"
#include "max_with_index.cuh"

#if !defined(FMMS_ARCH_SM90) && !defined(FMMS_ARCH_SM100)
#error "Compile with FMMS_ARCH_SM90 or FMMS_ARCH_SM100"
#endif

namespace fmms_cta_boundary_max {

using fmms_cutlass::MaxWithIndex;
using fmms_cutlass::choose_max;
using fmms_cutlass::float_bits;
using fmms_cutlass::reduce_warp_xor;
using fmms_harness::check_cuda;

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kWarps = 4;
constexpr int kWarpSize = 32;
constexpr int kThreads = kWarps * kWarpSize;
constexpr float kOutputCanary = 123456.0f;
#if defined(FMMS_ARCH_SM90)
constexpr unsigned kCtaParticipantMask = 0x11111111u;
constexpr int kCtaLaneStride = 4;
constexpr char const* kArchitecture = "sm90";
#else
constexpr unsigned kCtaParticipantMask = 0xffffffffu;
constexpr int kCtaLaneStride = 1;
constexpr char const* kArchitecture = "sm100";
#endif

__device__ MaxWithIndex thread_local_candidate(
    float const* values,
    int valid_m,
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
  MaxWithIndex local{-INFINITY, INT_MAX};
  for (int position : positions) {
    if (position < valid_m) {
      local = choose_max(
          local,
          {values[position * kTileN + column], position});
    }
  }
  return local;
#else
  int m = warp * kWarpSize + lane;
  return m < valid_m
      ? MaxWithIndex{values[m * kTileN + column], m}
      : MaxWithIndex{-INFINITY, INT_MAX};
#endif
}

__global__ void reduce_boundary_tile(
    float const* values,
    MaxWithIndex* results,
    int valid_m,
    int valid_n) {
  __shared__ MaxWithIndex warp_results[kWarps][kTileN];
  int lane = int(threadIdx.x) % kWarpSize;
  int warp = int(threadIdx.x) / kWarpSize;

  for (int column = 0; column < valid_n; ++column) {
    MaxWithIndex local =
        thread_local_candidate(values, valid_m, column, warp, lane);
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
    bool participates = (kCtaParticipantMask & (1u << lane)) != 0;
    for (int column = 0; column < valid_n; ++column) {
      MaxWithIndex local =
          participates && participant < kWarps
          ? warp_results[participant][column]
          : MaxWithIndex{-INFINITY, INT_MAX};
      if (participates) {
        local =
            reduce_warp_xor(local, kCtaParticipantMask, kCtaLaneStride);
        if (lane == 0) {
          results[column] = local;
        }
      }
    }
  }
}

int boundary_extent(int extent, int tile_extent) {
  int remainder = extent % tile_extent;
  return remainder == 0 ? tile_extent : remainder;
}

std::vector<float> make_values(int valid_m, int valid_n) {
  std::vector<float> values(kTileM * kTileN, -10000.0f);
  for (int m = 0; m < kTileM; ++m) {
    for (int n = 0; n < kTileN; ++n) {
      if (m >= valid_m) {
        values[m * kTileN + n] = 100000.0f + float(m);
      } else if (n < valid_n) {
        values[m * kTileN + n] = -1000.0f - float(m);
      }
    }
  }
  for (int n = 0; n < valid_n; ++n) {
    values[(valid_m - 1) * kTileN + n] = 1000.0f + float(n);
  }
  return values;
}

void run_shape(int m_extent, int n_extent) {
  int valid_m = boundary_extent(m_extent, kTileM);
  int valid_n = boundary_extent(n_extent, kTileN);
  int m_tile = (m_extent - 1) / kTileM;
  int n_tile = (n_extent - 1) / kTileN;
  std::vector<float> host_values = make_values(valid_m, valid_n);
  std::vector<MaxWithIndex> host_results(
      kTileN, {kOutputCanary, INT_MAX});
  float* device_values = nullptr;
  MaxWithIndex* device_results = nullptr;
  check_cuda(
      cudaMalloc(&device_values, host_values.size() * sizeof(float)),
      "cudaMalloc(values)");
  check_cuda(
      cudaMalloc(&device_results, host_results.size() * sizeof(MaxWithIndex)),
      "cudaMalloc(results)");
  check_cuda(
      cudaMemcpy(
          device_values,
          host_values.data(),
          host_values.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      "copy values");
  check_cuda(
      cudaMemcpy(
          device_results,
          host_results.data(),
          host_results.size() * sizeof(MaxWithIndex),
          cudaMemcpyHostToDevice),
      "initialize results");

  reduce_boundary_tile<<<1, kThreads>>>(
      device_values, device_results, valid_m, valid_n);
  check_cuda(cudaGetLastError(), "launch");
  check_cuda(cudaDeviceSynchronize(), "synchronize");
  check_cuda(
      cudaMemcpy(
          host_results.data(),
          device_results,
          host_results.size() * sizeof(MaxWithIndex),
          cudaMemcpyDeviceToHost),
      "copy results");

  for (int column = 0; column < kTileN; ++column) {
    bool valid_column = column < valid_n;
    MaxWithIndex expected =
        valid_column
        ? MaxWithIndex{1000.0f + float(column), valid_m - 1}
        : MaxWithIndex{kOutputCanary, INT_MAX};
    MaxWithIndex actual = host_results[column];
    bool passed =
        float_bits(expected.value) == float_bits(actual.value) &&
        expected.index == actual.index;
    std::cout << kArchitecture << ',' << m_extent << ',' << n_extent << ','
              << m_tile << ',' << n_tile << ',' << valid_m << ',' << valid_n
              << ',' << column << ',' << (valid_column ? 1 : 0) << ','
              << (valid_m - 1) << ',' << (kTileM - valid_m) << ','
              << (valid_m < kTileM
                      ? float_bits(100000.0f + float(valid_m))
                      : 0)
              << ',' << float_bits(expected.value) << ','
              << float_bits(actual.value) << ',' << expected.index << ','
              << actual.index << ',' << (passed ? 1 : 0) << '\n';
  }

  check_cuda(cudaFree(device_results), "cudaFree(results)");
  check_cuda(cudaFree(device_values), "cudaFree(values)");
}

void run_tests() {
  int const m_extents[] = {100, 127, 128, 129, 255, 256, 257};
  int const n_extents[] = {1, 2, 63, 64, 65, 127, 128, 129};
  std::cout
      << "architecture,m_extent,n_extent,m_tile,n_tile,valid_m,valid_n,column,"
         "valid_column,final_valid_m,padded_m_count,padded_m_sentinel_bits,"
         "expected_value_bits,actual_value_bits,expected_index,actual_index,"
         "pass\n";
  for (int m_extent : m_extents) {
    for (int n_extent : n_extents) {
      run_shape(m_extent, n_extent);
    }
  }
}

}  // namespace fmms_cta_boundary_max

int main() {
  fmms_cta_boundary_max::run_tests();
  return 0;
}
