cmake_minimum_required (VERSION 3.8)

set(SRCS compiler.cpp
         plugin_loader.cpp)

add_library(nncase SHARED ${SRCS})
target_link_libraries(nncase PRIVATE ir tflite_importer evaluator importer schedule codegen transforms targets simulator simulator_stackvm)
target_link_libraries(nncase PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::xtensor CONAN_PKG::mpark-variant)
target_compile_definitions(nncase PRIVATE -DNNCASE_DLL)
target_link_libraries(nncase PRIVATE CONAN_PKG::magic_enum CONAN_PKG::fmt)
