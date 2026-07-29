#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace fmms_cutlass {

struct MaxWithIndex {
  float value;
  int index;
};

__host__ __device__ inline MaxWithIndex choose_max(
    MaxWithIndex current,
    MaxWithIndex candidate) {
  if (candidate.value > current.value ||
      (candidate.value == current.value && candidate.index < current.index)) {
    return candidate;
  }
  return current;
}

inline uint32_t float_bits(float value) {
  union {
    float value;
    uint32_t bits;
  } representation{value};
  return representation.bits;
}

}  // namespace fmms_cutlass
