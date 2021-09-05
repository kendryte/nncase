cmake_minimum_required (VERSION 3.8)

add_subdirectory(tflite)
#add_subdirectory(onnx)

add_library(importer OBJECT importer.cpp)
#target_link_libraries(importer PUBLIC tflite_importer onnx_proto onnx_importer)
target_link_libraries(importer PUBLIC tflite_importer)
target_compile_definitions(importer PUBLIC -DNNCASE_DLL)
set_target_properties(importer PROPERTIES POSITION_INDEPENDENT_CODE ON)