#pragma once
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

using Fn = void (*)(float *, float *, float *, int, int, int);

class KernelBench {
  static std::unordered_map<std::string, Fn> &table();

public:
  static void add(const std::string &name, Fn fn);
  static void benchmark(const std::string &name, int iters, int warmup = 3);
};

void setup_register();

template <class T> bool check_match(T *left, T *right, size_t size) {
  bool result = false;
  for (int i = 0; i < size; i++) {
    if (std::abs(left[i] - right[i]) > 1e-4) {
      printf("position: %d\n", i);
      result = true;
    }
    if (result) {
      break;
    }
  }
  return !result;
}
