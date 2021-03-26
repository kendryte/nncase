cmake_minimum_required (VERSION 3.8)

set(SRCS tflite_importer.cpp
         ops/conv2d.cpp
         ops/transpose_conv.cpp
         ops/pool2d.cpp
         ops/binary.cpp
         ops/cast.cpp
         ops/reduce.cpp
         ops/unary.cpp
         ops/concat.cpp
         ops/matmul.cpp
         ops/pad.cpp
         ops/reshape.cpp
         ops/shape.cpp
         ops/softmax.cpp
         ops/slice.cpp
         ops/resize_image.cpp
         ops/transpose.cpp
         ops/space_to_batch.cpp
         ops/l2_normalization.cpp
         ops/logistic.cpp
         ops/quantize.cpp
         ops/activations.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
FLATBUFFERS_GENERATE_C_HEADERS(TFLITE_FB ${THIRD_PARTY}/tflite/schema.fbs)

add_library(tflite_importer OBJECT ${SRCS} ${TFLITE_FB_OUTPUTS})
target_include_directories(tflite_importer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(tflite_importer PUBLIC ir CONAN_PKG::flatbuffers)
set_target_properties(tflite_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
