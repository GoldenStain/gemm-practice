#include "bench/cpp_timing.h"
#include "gemm/cuda_gemms.cu"
#include "gemm/gemms.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <random>
#include <ratio>
#include <string>
#include <unordered_map>
#include <vector_types.h>

std::unordered_map<std::string, Fn> &KernelBench::table() {
  static std::unordered_map<std::string, Fn> t;
  return t;
}

void setup_register() {
  KernelBench::add("stupid_cuda_gemm", stupid_cuda_gemm);
}

void KernelBench::add(const std::string &name, Fn fn) {
  KernelBench::table().emplace(name, fn);
}

void KernelBench::benchmark(const std::string &name, int iters, int warmup) {
  std::mt19937_64 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  // First, we need to make sure the value is correct.
  matrix<float> C(ROWS, COLS);
  matrix<float> A(ROWS, COLS);
  matrix<float> B(ROWS, COLS);
  constexpr size_t copy_size = ROWS * COLS * sizeof(float);
  float *dA = nullptr;
  float *dB = nullptr;
  float *dC = nullptr;
  float *hC = (float *)malloc(copy_size);
  cudaMalloc(&dA, ROWS * COLS * sizeof(float));
  cudaMalloc(&dB, ROWS * COLS * sizeof(float));
  cudaMalloc(&dC, ROWS * COLS * sizeof(float));

  std::generate(A.data.begin(), A.data.end(), [&] { return dist(gen); });
  std::generate(B.data.begin(), B.data.end(), [&] { return dist(gen); });

  cudaMemcpy(dA, A.data.data(), copy_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data.data(), copy_size, cudaMemcpyHostToDevice);

  auto fn = KernelBench::table().at(name);
  // Calculate reference.
  gemm(A.data.data(), B.data.data(), C.data.data(), ROWS, COLS, ROWS);
  // Calculate with CUDA
  dim3 grid{CEIL_DIV(ROWS, 32), CEIL_DIV(COLS, 32), 1};
  dim3 block{32, 32, 1};
  fn<<<grid, block>>>(dA, dB, dC, ROWS, COLS, ROWS);
  cudaMemcpy(hC, dC, copy_size, cudaMemcpyDeviceToHost);

  assert(check_match<float>(C.data.data(), hC, copy_size / sizeof(float)));

  // GPU part
  // warmup
  for (int i = 0; i < warmup; i++) {
    fn<<<grid, block>>>(A.data.data(), B.data.data(), dC, ROWS, COLS, ROWS);
  }
  // Real timing
  // GPU part
  cudaDeviceSynchronize();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    fn<<<grid, block>>>(A.data.data(), B.data.data(), dC, ROWS, COLS, ROWS);
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();
  double gpu_time =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  // CPU part
  // warm up
  for (int i = 0; i < warmup; i++) {
    gemm(A.data.data(), B.data.data(), C.data.data(), ROWS, COLS, ROWS);
  }
  // real timing
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    gemm(A.data.data(), B.data.data(), C.data.data(), ROWS, COLS, ROWS);
  }
  t1 = std::chrono::steady_clock::now();
  double cpu_time =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
  printf("GPU time: %lfmicro seconds,\nCPU time: %lfmicro seconds", gpu_time,
         cpu_time);
}
