cmake_minimum_required (VERSION 3.13)

set(SRCS dummy_tensor_ops.cpp
         binary.cpp)
target_sources(kernels PRIVATE ${SRCS})