cmake_minimum_required (VERSION 3.8)

add_subdirectory(tflite)
add_subdirectory(onnx)
add_subdirectory(caffe)
add_subdirectory(pnnx)

add_library(importer OBJECT importer.cpp)
target_include_directories(importer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(importer PUBLIC tflite_importer onnx_proto onnx_importer caffe_importer pnnx_importer)
set_target_properties(importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
