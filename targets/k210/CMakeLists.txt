cmake_minimum_required (VERSION 3.8)

set(SRCS k210_target.cpp)

add_library(nncase_targets_k210 SHARED ${SRCS})
target_link_libraries(nncase_targets_k210 PRIVATE nncase nncase_modules_k210)
set_target_properties(nncase_targets_k210 PROPERTIES
                                          OUTPUT_NAME "nncase.targets.k210")
