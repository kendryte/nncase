cmake_minimum_required (VERSION 3.8)

add_subdirectory(codegen)
add_subdirectory(ir)

set(SRCS dummy.cpp)

add_library(nncase_modules_k210 SHARED ${SRCS})
target_link_libraries(nncase_modules_k210 PRIVATE codegen_k210 ir_k210)
set_target_properties(nncase_modules_k210 PROPERTIES
                                          OUTPUT_NAME "nncase.modules.k210")
