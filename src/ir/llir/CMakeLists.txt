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
         ops/quantize.cpp
         ops/dequantize.cpp
         ops/unary.cpp
         ops/pad.cpp
         ops/memory_copy.cpp
         ops/nnil_method.cpp
         ops/resize_image.cpp
         ops/strided_slice.cpp
         ops/table_lookup.cpp
         ops/k210/fake_kpu_conv2d.cpp
         ops/k210/kpu_conv2d.cpp)

add_library(llir STATIC ${SRCS})
target_include_directories(llir PUBLIC ../include)
target_include_directories(llir PUBLIC ${THIRD_PARTY}/xtensor/include)
target_link_libraries(llir PUBLIC common)
