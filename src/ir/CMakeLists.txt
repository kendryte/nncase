cmake_minimum_required (VERSION 3.8)
include(ops/${NNCASE_TARGET}/src.cmake)

set(SRCS connectors.cpp
         node.cpp
         graph.cpp
         visitor.cpp
         ops/conv2d.cpp
         ops/matmul.cpp
         ops/transpose.cpp
         ops/reduce.cpp
         ops/reduce_window2d.cpp
         ops/binary.cpp
         ops/concat.cpp
         ops/fake_quantize.cpp
         ops/fake_dequantize.cpp
         ops/quantize.cpp
         ops/dequantize.cpp
         ops/unary.cpp
         ops/pad.cpp
         ops/reshape.cpp
         ops/resize_image.cpp
         ops/strided_slice.cpp)

add_library(ir STATIC ${SRCS} ${TARGET_SRCS})
target_include_directories(ir PUBLIC include)
target_include_directories(ir PUBLIC ${THIRD_PARTY}/xtensor/include)
target_link_libraries(ir PUBLIC common)