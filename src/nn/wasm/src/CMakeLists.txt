cmake_minimum_required(VERSION 2.8.9)
project(nn_ops)
set(CMAKE_BUILD_TYPE Release)

include_directories(
  ./
  external/
  external/tensorflow/
  external/tensorflow/tensorflow/contrib/makefile/downloads/absl/
  external/tensorflow/tensorflow/contrib/makefile/downloads/eigen/
  external/tensorflow/tensorflow/contrib/makefile/downloads/gemmlowp/
  external/tensorflow/tensorflow/lite/experimental/
  external/tensorflow/tensorflow/lite/kernels/
  external/tensorflow/third_party/
  )

set(SOURCES
    bind/src/binding.cpp
    external/tensorflow/tensorflow/lite/kernels/cpu_backend_context.cc
    external/tensorflow/tensorflow/lite/kernels/cpu_backend_gemm_eigen.cc
    external/tensorflow/tensorflow/lite/experimental/ruy/allocator.cc
    external/tensorflow/tensorflow/lite/experimental/ruy/thread_pool.cc
   )

add_executable(nn_ops ${SOURCES})
set_property(TARGET nn_ops PROPERTY CXX_STANDARD 11)
set_target_properties(nn_ops PROPERTIES LINK_FLAGS "-s WASM=1 -s NO_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s SINGLE_FILE=1 -s MODULARIZE=1 --memory-init-file 0 --bind")
