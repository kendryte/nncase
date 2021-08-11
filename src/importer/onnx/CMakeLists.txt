include(TestBigEndian)
test_big_endian(BIG_ENDIAN)

set(ONNX_IMPORTER_SOURCES
    onnx_importer.cpp
    )

set(ONNX_IMPORTER_OPS_SOURCES
    ops/activations.cpp
    ops/binary.cpp
    ops/clip.cpp
    ops/batchnorm.cpp
    ops/concat.cpp
    ops/constant.cpp
    ops/conv.cpp
    ops/cast.cpp
    ops/dropout.cpp
    ops/gather.cpp
    ops/gather_nd.cpp
    ops/gemm.cpp
    ops/flatten.cpp
    ops/identity.cpp
    ops/instancenorm.cpp
    ops/lpnorm.cpp
    ops/lrn.cpp
    ops/matmul.cpp
    ops/onehot.cpp
    ops/pad.cpp
    ops/pool.cpp
    ops/reduce.cpp
    ops/reshape.cpp
    ops/resize.cpp
    ops/shape.cpp
    ops/slice.cpp
    ops/softmax.cpp
    ops/squeeze.cpp
    ops/sum.cpp
    ops/transpose.cpp
    ops/unary.cpp
    ops/upsample.cpp
    ops/unsqueeze.cpp
    )

add_library(onnx_importer ${ONNX_IMPORTER_SOURCES} ${ONNX_IMPORTER_OPS_SOURCES})

target_compile_definitions(onnx_importer PRIVATE NATIVE_IS_BIG_ENDIAN=${BIG_ENDIAN})
target_include_directories(onnx_importer PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
get_filename_component(PARENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
target_include_directories(onnx_importer PUBLIC ${PARENT_SOURCE_DIR}/include)

target_link_libraries(onnx_importer PUBLIC ir)
target_link_libraries(onnx_importer PRIVATE onnx_proto protobuf::libprotoc protobuf::libprotobuf)
set_target_properties(onnx_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
