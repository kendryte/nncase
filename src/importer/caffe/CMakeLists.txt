cmake_minimum_required (VERSION 3.8)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS caffe.proto)

set(SRCS caffe.pb.cc 
        caffe_importer.cpp
        ops/activations.cpp
        ops/binary.cpp
        ops/concat.cpp
        ops/convolution.cpp
        ops/permute.cpp
        ops/placeholders.cpp
        ops/reshape.cpp
        ops/slice.cpp
        ops/softmax.cpp
        ops/split.cpp
        ops/pooling.cpp
        ops/batchnorm.cpp
        ops/scale.cpp
        ops/reverse.cpp
        ops/lstm.cpp
        ops/inner_product.cpp
        ops/batchnorm.cpp)

add_library(caffe_importer ${SRCS})
target_include_directories(caffe_importer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(caffe_importer PUBLIC ${PARENT_SOURCE_DIR}/include)
target_link_libraries(caffe_importer PUBLIC ir protobuf::libprotobuf)
target_link_libraries(caffe_importer PRIVATE)
set_target_properties(caffe_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)