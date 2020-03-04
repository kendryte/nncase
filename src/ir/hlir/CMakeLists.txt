cmake_minimum_required (VERSION 3.8)

set(SRCS connectors.cpp
         node.cpp
         graph.cpp
         visitor.cpp
         placeholders.cpp
         ops/conv2d.cpp
         ops/conv2d_transpose.cpp
         ops/fused_unary.cpp
         ops/matmul.cpp
         ops/transpose.cpp
         ops/reduce.cpp
         ops/reduce_window2d.cpp
         ops/binary.cpp
         ops/concat.cpp
         ops/clamp.cpp
         ops/constant.cpp
         ops/quantize.cpp
         ops/dequantize.cpp
         ops/unary.cpp
         ops/pad.cpp
         ops/reshape.cpp
         ops/resize_image.cpp
         ops/strided_slice.cpp
         ops/table_lookup.cpp
         ops/k210/fake_kpu_conv2d.cpp
         ops/k210/kpu_conv2d.cpp
         ops/k210/kpu_data_exchange.cpp)

add_library(hlir STATIC ${SRCS})
target_include_directories(hlir PUBLIC ../include)
target_include_directories(hlir PUBLIC ${THIRD_PARTY}/xtensor/include)
target_link_libraries(hlir PUBLIC common llir)
