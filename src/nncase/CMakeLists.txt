cmake_minimum_required (VERSION 3.8)

set(SRCS compiler.cpp
         simulator.cpp)

add_library(nncase SHARED ${SRCS})
target_link_libraries(nncase PRIVATE data ir tflite_importer evaluator importer schedule codegen codegen_stackvm transforms targets simulator simulator_stackvm plugin)
target_link_libraries(nncase PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::xtensor CONAN_PKG::mpark-variant)
target_compile_definitions(nncase PRIVATE -DNNCASE_DLL)
target_compile_definitions(nncase PUBLIC -DNNCASE_SHARED_LIBS)
target_link_libraries(nncase PRIVATE CONAN_PKG::magic_enum CONAN_PKG::fmt)
