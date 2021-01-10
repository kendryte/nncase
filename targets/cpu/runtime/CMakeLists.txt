cmake_minimum_required (VERSION 3.8)

set(SRCS cpu_runtime.cpp)

add_library(nncase_simulator_cpu OBJECT ${SRCS})
target_link_libraries(nncase_simulator_cpu PRIVATE nncase)

