#include "bench/cpp_timing.h"
#include <string>
int main(int argc, char *argv[]) {
  setup_register();
  std::string func_name(argv[1]);
  // Register
  // Run
  KernelBench::benchmark(func_name, 3, 3);
}
