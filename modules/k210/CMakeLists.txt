cmake_minimum_required (VERSION 3.8)

include_directories(include)

add_subdirectory(src/codegen)
add_subdirectory(src/ir)
add_subdirectory(src/schedule)
add_subdirectory(src/transforms/k210)
add_subdirectory(src/evaluator)
add_subdirectory(src/runtime)

set(SRCS src/dummy.cpp)

add_library(nncase_modules_k210 SHARED ${SRCS})
target_include_directories(nncase_modules_k210 PUBLIC include)
target_link_libraries(nncase_modules_k210 PRIVATE codegen_k210 ir_k210 evaluator_k210 schedule_k210 transforms_k210 simulator_k210 nncase)
set_target_properties(nncase_modules_k210 PROPERTIES
                                          OUTPUT_NAME "nncase.modules.k210")
install(TARGETS nncase_modules_k210)
