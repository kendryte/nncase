cmake_minimum_required (VERSION 3.8)

set(SRCS dataset.cpp)

add_library(data OBJECT ${SRCS})
target_link_libraries(data PUBLIC CONAN_PKG::xtensor CONAN_PKG::gsl-lite)
target_compile_definitions(data PUBLIC -DNNCASE_DLL)
target_link_libraries(data PRIVATE CONAN_PKG::opencv)
set_property(TARGET data PROPERTY POSITION_INDEPENDENT_CODE ON)