cmake_minimum_required (VERSION 3.8)

set(SRCS cpu_target.cpp
         codegen/cpu_ops.cpp)

add_library(nncase_targets_cpu SHARED ${SRCS})
target_link_libraries(nncase_targets_cpu PRIVATE nncase nncase_simulator_cpu)
set_target_properties(nncase_targets_cpu PROPERTIES
                                         OUTPUT_NAME "nncase.targets.cpu")

add_subdirectory(runtime)
nncase_add_runtime(cpu)
