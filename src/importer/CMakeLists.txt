cmake_minimum_required (VERSION 3.8)

set(SRCS tflite/tflite_importer.cpp)

add_library(importer STATIC ${SRCS})
target_include_directories(importer PUBLIC include)
target_link_libraries(importer PUBLIC ir)
target_link_libraries(importer PRIVATE CONAN_PKG::flatbuffers)