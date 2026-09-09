#include "cpp_gemm.h"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

const int ROWS = 1024, COLS = 1024;

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

int main(int argc, char *argv[]) {
  int m = ROWS;
  int n = COLS;
  int k = ROWS;
  std::string input_file(argv[1]);
  std::ifstream fin(input_file);
  if (!fin) {
    std::cerr << ("Cannot open file") << input_file << "\n";
    return 1;
  }
  std::cout << "Strat computing...\n";
  matrix<float> A{ROWS, COLS};
  matrix<float> B{ROWS, COLS};
  matrix<float> C{ROWS, COLS};
  float C1[ROWS * COLS] = {0};
  for (int i = 0; i < ROWS; i++)
    for (int j = 0; j < COLS; j++)
      fin >> A[i][j];
  for (int i = 0; i < ROWS; i++)
    for (int j = 0; j < COLS; j++)
      fin >> B[i][j];
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      for (int ii = 0; ii < k; ii++)
        C[i][j] += A[i][ii] * B[ii][j];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++)
      std::cout << C[i][j] << " ";
    std::cout << "\n";
  }
  gemm(A.data.data(), B.data.data(), C1, m, n, k);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      assert(C[i][j] == C1[i * n + j]);
  std::cout << "Two ways of computation equals";
}
