cmake_minimum_required (VERSION 3.8)

set(SRCS cpu_target.cpp)

add_library(nncase_targets_cpu SHARED ${SRCS})
target_link_libraries(nncase_targets_cpu PRIVATE nncase)
set_target_properties(nncase_targets_cpu PROPERTIES
                                         OUTPUT_NAME "nncase.targets.cpu")

nncase_add_runtime(cpu)
