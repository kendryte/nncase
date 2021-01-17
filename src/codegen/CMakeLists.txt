cmake_minimum_required (VERSION 3.8)

set(SRCS module_builder.cpp
         ops/neutral/neutral_ops.cpp)

add_library(codegen OBJECT ${SRCS})
target_link_libraries(codegen PUBLIC ir schedule)
target_link_libraries(codegen PRIVATE evaluator)
set_target_properties(codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)