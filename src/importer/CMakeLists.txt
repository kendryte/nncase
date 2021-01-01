cmake_minimum_required (VERSION 3.8)

add_subdirectory(tflite)

add_library(importer OBJECT importer.cpp)
target_link_libraries(importer PUBLIC tflite_importer)
set_target_properties(importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
