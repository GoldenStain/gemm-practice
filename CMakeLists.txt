cmake_minimum_required(VERSION 3.24)
project(gemm_practice LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(gemm STATIC
  gemm/cpp_gemm.cc
  gemm/cuda_gemms.cu)
target_include_directories(gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(cpp_reference
  reference/cpp_reference.cc)
target_link_libraries(cpp_reference PRIVATE gemm)

add_executable(bench
  bench/bench_main.cc
  bench/cpp_timing.cu)
target_link_libraries(bench PRIVATE gemm)
