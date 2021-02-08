cmake_minimum_required (VERSION 3.8)

set(SRCS evaluator.cpp
         quantizer.cpp
         ops/neutral/neutral_ops.cpp)

add_library(evaluator OBJECT ${SRCS})
target_link_libraries(evaluator PUBLIC ir schedule simulator CONAN_PKG::mpark-variant)
set_property(TARGET evaluator PROPERTY POSITION_INDEPENDENT_CODE ON)