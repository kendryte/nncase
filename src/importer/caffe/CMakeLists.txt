cmake_minimum_required (VERSION 3.8)

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
        ops/indicator.cpp
        ops/pooling.cpp
        ops/batch_norm.cpp
        ops/scale.cpp
        #ops/reverse.cpp
        )

add_library(caffe_importer ${SRCS})
target_include_directories(caffe_importer PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(caffe_importer PUBLIC ${PARENT_SOURCE_DIR}/include)
target_link_libraries(caffe_importer PUBLIC ir)
target_link_libraries(caffe_importer PRIVATE CONAN_PKG::protobuf)
set_target_properties(caffe_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)