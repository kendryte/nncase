cmake_minimum_required (VERSION 3.8)

add_subdirectory(tflite)
add_subdirectory(onnx)

add_library(importer OBJECT importer.cpp)
target_include_directories(importer PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../third_party/onnx/)
target_include_directories(importer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(importer PUBLIC tflite_importer onnx_importer)
set_target_properties(importer PROPERTIES POSITION_INDEPENDENT_CODE ON)