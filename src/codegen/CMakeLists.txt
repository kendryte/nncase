cmake_minimum_required (VERSION 3.8)

set(SRCS module_builder.cpp
         model_builder.cpp)

add_library(codegen OBJECT ${SRCS})
target_link_libraries(codegen PUBLIC ir schedule)
target_link_libraries(codegen PRIVATE evaluator)
target_compile_definitions(codegen PUBLIC -DNNCASE_DLL)
set_target_properties(codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(stackvm)