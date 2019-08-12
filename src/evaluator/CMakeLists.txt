cmake_minimum_required (VERSION 3.8)

set(SRCS evaluator.cpp
         quantizer.cpp
         ops/neutral/neutral_ops.cpp
         ops/k210/k210_ops.cpp)

add_library(evaluator STATIC ${SRCS})
target_include_directories(evaluator PUBLIC include)
target_link_libraries(evaluator PUBLIC ir scheduler)