cmake_minimum_required (VERSION 3.13)

set(SRCS convolution.cpp)
target_sources(kernels PRIVATE ${SRCS})
