#pragma once
#include <cuda_runtime.h>

__global__ void stupid_cuda_gemm(float *A, float *B, float *C, int m, int n,
                                 int k);
