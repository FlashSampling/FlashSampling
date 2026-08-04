#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace fmms {

struct Philox4x32 {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

__host__ __device__ inline uint32_t philox_mul_hi(uint32_t a, uint32_t b) {
#ifdef __CUDA_ARCH__
  return __umulhi(a, b);
#else
  return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
#endif
}

__host__ __device__ inline Philox4x32 philox4x32_10(
    uint64_t seed, uint32_t sample_idx, uint32_t hidden_idx,
    uint64_t vocab_idx) {
  constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
  constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
  constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
  constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

  Philox4x32 value{
      static_cast<uint32_t>(vocab_idx),
      static_cast<uint32_t>(vocab_idx >> 32),
      hidden_idx,
      sample_idx,
  };
  uint32_t key0 = static_cast<uint32_t>(seed);
  uint32_t key1 = static_cast<uint32_t>(seed >> 32);
  #pragma unroll
  for (int round = 0; round < 10; ++round) {
    uint32_t lo0 = kPhiloxM0 * value.x;
    uint32_t hi0 = philox_mul_hi(kPhiloxM0, value.x);
    uint32_t lo1 = kPhiloxM1 * value.z;
    uint32_t hi1 = philox_mul_hi(kPhiloxM1, value.z);
    value = {hi1 ^ value.y ^ key0, lo1, hi0 ^ value.w ^ key1, lo0};
    key0 += kPhiloxW0;
    key1 += kPhiloxW1;
  }
  return value;
}

__host__ __device__ inline float uniform_open_closed(uint32_t value) {
  // Matches cuRAND's (0, 1] convention without introducing zero.
  return (static_cast<float>(value) + 1.0f) * 0x1p-32f;
}

__host__ __device__ inline float uniform_open_open(uint32_t value) {
  // Use 23 random bits so both half-step endpoints are exactly representable.
  return (static_cast<float>(value >> 9) + 0.5f) * 0x1p-23f;
}

}  // namespace fmms
