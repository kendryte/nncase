cmake_minimum_required (VERSION 3.8)

set(SRCS ops/k210_evaluators.cpp)

add_library(evaluator_k210 OBJECT ${SRCS})
target_link_libraries(evaluator_k210 PUBLIC nncase)
target_compile_definitions(evaluator_k210 PUBLIC -DNNCASE_MODULES_K210_DLL)
set_target_properties(evaluator_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)