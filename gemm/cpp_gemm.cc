#include "gemm/gemms.h"
void gemm(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int ii = 0; ii < k; ii++) {
        INDEX(C, i, j, n) += INDEX(A, i, ii, n) * INDEX(B, ii, j, n);
      }
    }
  }
}
