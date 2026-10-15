#include "gemm/gemms.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/types.h>

__global__ void stupid_cuda_gemm(float *A, float *B, float *C, int m, int n,
                                 int k) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  for (int ii = 0; ii < k; ii++) {
    INDEX(C, x, y, n) += INDEX(A, x, ii, k) * INDEX(B, ii, y, n);
  }
}
