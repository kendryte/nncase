cmake_minimum_required (VERSION 3.8)

set(SRCS plugin_loader.cpp)

add_library(plugin OBJECT ${SRCS})
target_link_libraries(plugin PRIVATE targets CONAN_PKG::fmt)
target_link_libraries(plugin PUBLIC CONAN_PKG::gsl-lite CONAN_PKG::xtensor CONAN_PKG::mpark-variant)
target_compile_definitions(plugin PRIVATE -DNNCASE_DLL)
set_target_properties(plugin PROPERTIES POSITION_INDEPENDENT_CODE ON)
