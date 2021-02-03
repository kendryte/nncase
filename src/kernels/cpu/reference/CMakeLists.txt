cmake_minimum_required (VERSION 3.13)

set(SRCS convolution.cpp
         tensor_compute.cpp)
target_sources(kernels PRIVATE ${SRCS})
