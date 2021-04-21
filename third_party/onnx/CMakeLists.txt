add_library(onnx_proto STATIC onnx.proto)
protobuf_generate(TARGET onnx_proto)
target_include_directories(onnx_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(onnx_proto PUBLIC CONAN_PKG::protobuf)
set_property(TARGET onnx_proto PROPERTY POSITION_INDEPENDENT_CODE ON)

if (WIN32)
    target_compile_options(onnx_proto PUBLIC /wd4996 /wd4244)
endif()
