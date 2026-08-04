#include "stateless_philox.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr) do { \
  cudaError_t status = (expr); \
  if (status != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(status)); \
    std::exit(1); \
  } \
} while (0)

__global__ void generate(
    fmms::Philox4x32* output, uint64_t seed, uint32_t samples,
    uint32_t hidden_states, uint64_t vocab_size, uint32_t tile_vocab) {
  uint32_t sample_idx = blockIdx.z;
  uint32_t hidden_idx = blockIdx.y;
  uint64_t tile_start = static_cast<uint64_t>(blockIdx.x) * tile_vocab;
  for (uint32_t local_vocab = threadIdx.x; local_vocab < tile_vocab;
       local_vocab += blockDim.x) {
    uint64_t vocab_idx = tile_start + local_vocab;
    if (vocab_idx >= vocab_size) continue;
    uint64_t linear =
        (static_cast<uint64_t>(sample_idx) * hidden_states + hidden_idx)
        * vocab_size + vocab_idx;
    output[linear] = fmms::philox4x32_10(
        seed, sample_idx, hidden_idx, vocab_idx);
  }
}

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::fprintf(stderr, "usage: %s THREADS TILE_V SEED OUTPUT [--profile]\n", argv[0]);
    return 2;
  }
  int threads = std::atoi(argv[1]);
  uint32_t tile_vocab = std::strtoul(argv[2], nullptr, 10);
  uint64_t seed = std::strtoull(argv[3], nullptr, 10);
  constexpr uint32_t kSamples = 4;
  constexpr uint32_t kHiddenStates = 4;
  constexpr uint64_t kVocabSize = 65536;
  constexpr uint64_t kCount =
      static_cast<uint64_t>(kSamples) * kHiddenStates * kVocabSize;

  fmms::Philox4x32* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_output, kCount * sizeof(*device_output)));
  dim3 blocks(
      (kVocabSize + tile_vocab - 1) / tile_vocab, kHiddenStates, kSamples);
  generate<<<blocks, threads>>>(device_output, seed, kSamples, kHiddenStates,
                                kVocabSize, tile_vocab);
  CUDA_CHECK(cudaGetLastError());
  if (argc == 6) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(device_output));
    return 0;
  }
  std::vector<fmms::Philox4x32> output(kCount);
  CUDA_CHECK(cudaMemcpy(output.data(), device_output,
                        kCount * sizeof(*device_output), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_output));

  FILE* file = std::fopen(argv[4], "wb");
  if (file == nullptr ||
      std::fwrite(output.data(), sizeof(*output.data()), kCount, file) != kCount) {
    std::fprintf(stderr, "failed to write %s\n", argv[4]);
    return 1;
  }
  std::fclose(file);
  return 0;
}
