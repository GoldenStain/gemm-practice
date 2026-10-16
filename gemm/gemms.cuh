#pragma once
#include "gemm/gemms.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <sys/types.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t last_error = (call);                                           \
    if (last_error != cudaSuccess) {                                           \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                       \
              cudaGetErrorString(last_error));                                 \
      exit(1);                                                                 \
    }                                                                          \
  } while (0);

__global__ void stupid_cuda_gemm(float *A, float *B, float *C, int m, int n,
                                 int k);

template <int BLOCKSIZE = 32>
__global__ void naive_cuda_gemm(float *A, float *B, float *C, int m, int n,
                                int k) {
  uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  if (x < m && y < n) {
    for (int ii = 0; ii < k; ii++) {
      INDEX(C, x, y, n) += INDEX(A, x, ii, k) * INDEX(B, ii, y, n);
    }
  }
}
