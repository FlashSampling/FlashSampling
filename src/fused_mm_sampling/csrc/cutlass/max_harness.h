// Host-side helpers shared by the max-with-index gate harnesses (Gates 1b-1f).
//
// The reduction kernels and per-gate case builders stay in each `.cu`; only
// the verbatim-repeated CUDA boilerplate lives here.

#pragma once

#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

namespace fmms_harness {

inline void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace fmms_harness
