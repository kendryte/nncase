cmake_minimum_required (VERSION 3.13)

set(SRCS convolution.cpp concat.cpp slice.cpp)
target_sources(kernels PRIVATE ${SRCS})