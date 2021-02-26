cmake_minimum_required (VERSION 3.8)

set(SRCS kpu_buffer_allocator.cpp)

add_library(schedule_k210 OBJECT ${SRCS})
target_link_libraries(schedule_k210 PUBLIC nncase)
target_compile_definitions(schedule_k210 PUBLIC -DNNCASE_MODULES_K210_DLL)
set_target_properties(schedule_k210 PROPERTIES POSITION_INDEPENDENT_CODE ON)