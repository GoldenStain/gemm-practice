#pragma once
#include <cstddef>
#include <vector>
#define INDEX(X, I, J, CNT) (X[I * CNT + J])
#define CEIL_DIV(X, Y) ((X + Y - 1) / Y)

inline constexpr size_t ROWS = 1024;
inline constexpr size_t COLS = 1024;

template <typename T> struct matrix {
  std::vector<T> data;
  int cols, rows;
  matrix(int c, int r) : rows(r), cols(c), data(r * c) {}
  matrix() = delete;
  T operator()(int i, int j) { return data[i * cols + j]; }
  T operator()(int i, int j) const { return data[i * cols + j]; }
  T *operator[](int row) { return data.data() + row * cols; }
  T *operator[](int row) const { return data.data() + row * cols; }
};

void gemm(float *A, float *B, float *C, int m, int n, int k);
