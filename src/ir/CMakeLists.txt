cmake_minimum_required (VERSION 3.8)

set(SRCS connectors.cpp
         node.cpp
         graph.cpp
         visitor.cpp
         ops/conv2d.cpp
         ops/conv2d_transpose.cpp
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
         ops/strided_slice.cpp
         ops/k210/fake_kpu_conv2d.cpp
         ops/k210/fake_piecewise_linear.cpp
         ops/k210/kpu_conv2d.cpp)

add_library(ir STATIC ${SRCS})
target_include_directories(ir PUBLIC include)
target_include_directories(ir PUBLIC ${THIRD_PARTY}/xtensor/include)
target_link_libraries(ir PUBLIC common)