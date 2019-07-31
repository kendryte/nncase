cmake_minimum_required (VERSION 3.8)

set(SRCS codegen.cpp
         ops/neutral/neutral_ops.cpp
         ops/k210/k210_ops.cpp)

add_library(codegen STATIC ${SRCS})
target_include_directories(codegen PUBLIC include)
target_link_libraries(codegen PUBLIC ir scheduler)
target_link_libraries(codegen PRIVATE evaluator)