cmake_minimum_required (VERSION 3.8)

build_flatbuffers("${THIRD_PARTY}/tflite/schema.fbs" "" TFLITE_FB "" "${CMAKE_CURRENT_BINARY_DIR}" "" "")

set(SRCS tflite_importer.cpp
         ops/conv2d.cpp
         ops/transpose_conv.cpp
         ops/pool2d.cpp
         ops/binary.cpp
         ops/cast.cpp
         ops/compare.cpp
         ops/custom.cpp
         ops/reduce.cpp
         ops/reduce_arg.cpp
         ops/reduce_prod.cpp
         ops/unary.cpp
         ops/concat.cpp
         ops/matmul.cpp
         ops/pad.cpp
         ops/reshape.cpp
         ops/shape.cpp
         ops/softmax.cpp
         ops/slice.cpp
         ops/resize_image.cpp
         ops/tile.cpp
         ops/transpose.cpp
         ops/space_to_batch.cpp
         ops/l2_normalization.cpp
         ops/logistic.cpp
         ops/quantize.cpp
         ops/activations.cpp
         ops/squared_difference.cpp
         ops/gather.cpp
         ops/gather_nd.cpp
         ops/onehot.cpp
         ops/log_softmax.cpp
         ops/split.cpp)

add_library(tflite_importer OBJECT ${SRCS})
target_link_libraries(tflite_importer PUBLIC ir flatbuffers::flatbuffers)
target_include_directories(tflite_importer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(tflite_importer TFLITE_FB)
set_target_properties(tflite_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
