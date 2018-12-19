cmake_minimum_required(VERSION 2.8.9)
project(nn_ops)
set(CMAKE_BUILD_TYPE Release)

include_directories(
  ./
  external/
  external/tensorflow/
  external/tensorflow/tensorflow/contrib/makefile/downloads/eigen/
  external/tensorflow/tensorflow/contrib/makefile/downloads/gemmlowp/
  external/tensorflow/tensorflow/lite/kernels/
  external/tensorflow/third_party/
  )

set(SOURCES
    bind/src/binding.cpp
   )

add_executable(nn_ops ${SOURCES})
set_property(TARGET nn_ops PROPERTY CXX_STANDARD 11)
set_target_properties(nn_ops PROPERTIES LINK_FLAGS "-s WASM=1 -s NO_FILESYSTEM=1 -s ALLOW_MEMORY_GROWTH=1 -s SINGLE_FILE=1 -s MODULARIZE=1 --memory-init-file 0 --bind")

set(test_files "")
set(test_dir ${CMAKE_CURRENT_SOURCE_DIR}/bind/test)
file(GLOB_RECURSE files RELATIVE "${test_dir}" "${test_dir}/*")
foreach(f ${files})
  message(STATUS "copy ${test_dir}/${f} ${CMAKE_CURRENT_BINARY_DIR}/${f}")
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${f}"
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different "${test_dir}/${f}" "${CMAKE_CURRENT_BINARY_DIR}/${f}"
                     DEPENDS "${test_dir}/${f}"
                     COMMENT "Copying ${f}"
                    )
  list(APPEND test_files "${test_dir}/${f}" "${CMAKE_CURRENT_BINARY_DIR}/${f}")
endforeach()

message(STATUS "${test_files}")

add_custom_target(nn_ops_test ALL
                  DEPENDS nn_ops ${test_files})

