cmake_minimum_required (VERSION 3.8)

set(SRCS tflite/tflite_importer.cpp
         tflite/ops/conv2d.cpp
         tflite/ops/transpose_conv.cpp
         tflite/ops/pool2d.cpp
         tflite/ops/binary.cpp
         tflite/ops/reduce.cpp
         tflite/ops/unary.cpp
         tflite/ops/concat.cpp
         tflite/ops/matmul.cpp
         tflite/ops/pad.cpp
         tflite/ops/reshape.cpp
         tflite/ops/softmax.cpp
         tflite/ops/resize_image.cpp
         tflite/ops/transpose.cpp
         tflite/ops/leaky_relu.cpp
         tflite/ops/space_to_batch.cpp
         tflite/ops/l2_normalization.cpp
         tflite/ops/logistic.cpp
         tflite/ops/activations.cpp
         paddle/paddle_importer.cpp
         paddle/ops/placeholders.cpp
         paddle/ops/conv2d.cpp
         caffe/caffe_importer.cpp
         caffe/ops/placeholders.cpp
         caffe/ops/convolution.cpp
         caffe/ops/activations.cpp
         caffe/ops/split.cpp
         caffe/ops/concat.cpp
         caffe/ops/binary.cpp
         caffe/ops/permute.cpp
         caffe/ops/reshape.cpp
         caffe/ops/softmax.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS paddle/framework.proto caffe/caffe.proto)
FLATBUFFERS_GENERATE_C_HEADERS(TFLITE_FB tflite/schema.fbs)

add_library(importer STATIC ${SRCS} ${PROTO_SRCS} ${TFLITE_FB_OUTPUTS})
target_include_directories(importer PUBLIC include)
target_link_libraries(importer PUBLIC ir CONAN_PKG::boost)
target_link_libraries(importer PRIVATE CONAN_PKG::flatbuffers CONAN_PKG::protobuf)